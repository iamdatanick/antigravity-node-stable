"""FastAPI A2A endpoints: /health, /task, /handoff, /upload, /webhook, /.well-known/agent.json."""

import asyncio
import hashlib
import hmac
import logging
import os
import uuid

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from workflows.auth import validate_token
from workflows.goose_client import goose_reflect
from workflows.health import full_health_check
from workflows.inference import list_models as ovms_list_models, run_inference
from workflows.memory import push_episodic, recall_experience
from workflows.models import (
    CapabilitiesResponse,
    ChatCompletionRequest,
    HandoffRequest,
    HandoffResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    TaskRequest,
    TaskResponse,
    ToolsResponse,
    UploadResponse,
    WebhookPayload,
    WebhookResponse,
)
from workflows.lineage import complete_job, fail_job, start_job
from workflows.s3_client import upload as s3_upload

logger = logging.getLogger("antigravity.a2a")

app = FastAPI(title="Antigravity Node v13.0", version="13.0.0")

# CORS middleware
raw_cors_origins = os.environ.get("CORS_ORIGINS", "*")
if not raw_cors_origins or raw_cors_origins.strip() == "*":
    allow_origins = ["*"]
else:
    allow_origins = [origin.strip() for origin in raw_cors_origins.split(",") if origin.strip()]

allow_creds = allow_origins != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_creds,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Webhook authentication
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")


def verify_webhook_signature(payload_body: bytes, signature: str) -> bool:
    """Verify HMAC-SHA256 signature for webhook payloads."""
    if not WEBHOOK_SECRET:
        logging.warning("WEBHOOK_SECRET not set - rejecting webhook. Set WEBHOOK_SECRET env var.")
        return False
    expected = hmac.new(WEBHOOK_SECRET.encode(), payload_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


@app.get("/health", response_model=HealthResponse)
async def health():
    """GET /health — 5-level health check hierarchy."""
    result = await full_health_check()
    status_code = 200 if result["status"] == "healthy" else 503
    return JSONResponse(content=result, status_code=status_code)


@app.get("/.well-known/agent.json")
async def well_known_agent():
    """Serve the A2A agent descriptor."""
    agent_path = "/app/well-known/agent.json"
    if os.path.exists(agent_path):
        return FileResponse(agent_path, media_type="application/json")
    return JSONResponse({"error": "agent.json not found"}, status_code=404)


@app.post("/task", response_model=TaskResponse)
@limiter.limit("60/minute")
async def task(
    request: Request,
    body: TaskRequest,
    x_tenant_id: str = Header(default=None),
    user: dict = Depends(validate_token),
):
    """POST /task — A2A task endpoint with multi-tenant isolation."""
    if not x_tenant_id:
        raise HTTPException(status_code=400, detail="x-tenant-id header required")

    goal = body.goal
    context = body.context
    session_id = body.session_id or str(uuid.uuid4())

    logger.info(f"Task received: tenant={x_tenant_id}, goal={goal[:100]}")

    # Lineage: START event (fire-and-forget)
    lineage_job = f"a2a.task.{x_tenant_id}"
    asyncio.create_task(start_job(lineage_job))

    # 1. Record in episodic memory
    push_episodic(
        tenant_id=x_tenant_id,
        session_id=session_id,
        actor="User",
        action_type="TASK_REQUEST",
        content=goal,
    )

    # 2. Recall past experience
    history = recall_experience(goal, x_tenant_id, limit=5)

    # 3. Process the task
    push_episodic(
        tenant_id=x_tenant_id,
        session_id=session_id,
        actor="Goose",
        action_type="THOUGHT",
        content=f"Processing goal: {goal}. Found {len(history)} relevant past events.",
    )

    # Lineage: COMPLETE event (fire-and-forget)
    asyncio.create_task(
        complete_job(
            lineage_job,
            session_id,
            outputs=[{"name": f"task.{session_id}"}],
        )
    )

    return {
        "status": "accepted",
        "session_id": session_id,
        "tenant_id": x_tenant_id,
        "history_count": len(history),
    }


@app.post("/handoff", response_model=HandoffResponse)
async def handoff(body: HandoffRequest, x_tenant_id: str = Header(default="system"), user: dict = Depends(validate_token)):
    """POST /handoff — A2A agent-to-agent handoff."""
    target = body.target_agent
    payload = body.payload
    logger.info(f"Handoff to {target} from tenant={x_tenant_id}")
    return {"status": "handoff_acknowledged", "target": target}


@app.post("/upload", response_model=UploadResponse)
@limiter.limit("10/minute")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    x_tenant_id: str = Header(default="system"),
    user: dict = Depends(validate_token),
):
    """POST /upload — HTTP file upload to SeaweedFS (Gap #14 fix)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    safe_filename = os.path.basename(file.filename)
    if not safe_filename or ".." in safe_filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Check file size (100MB limit)
    content = await file.read()
    max_size = 100 * 1024 * 1024  # 100MB
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail="File too large (max 100MB)")

    key = f"context/{x_tenant_id}/{safe_filename}"
    s3_upload(key, content)

    logger.info(f"File uploaded: {key} ({len(content)} bytes) by tenant={x_tenant_id}")

    # Record in episodic memory
    push_episodic(
        tenant_id=x_tenant_id,
        session_id="upload",
        actor="User",
        action_type="FILE_UPLOAD",
        content=f"Uploaded {file.filename} ({len(content)} bytes)",
    )

    return {"status": "uploaded", "key": key, "size": len(content)}


@app.post("/webhook", response_model=WebhookResponse)
async def argo_webhook(
    request: Request,
    payload: WebhookPayload,
    x_webhook_signature: str = Header(default=""),
):
    """POST /webhook — Argo exit-handler callback (Gap #6 fix)."""
    # Verify webhook signature if WEBHOOK_SECRET is set
    if WEBHOOK_SECRET:
        payload_body = await request.body()
        if not verify_webhook_signature(payload_body, x_webhook_signature):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

    task_id = payload.task_id
    status = payload.status
    message = payload.message

    logger.info(f"Argo callback: task={task_id}, status={status}")

    if status == "Failed":
        await goose_reflect(task_id, message)
        logger.warning(f"Argo workflow {task_id} failed. Goose self-correction triggered.")
        # Lineage: FAIL event (fire-and-forget)
        asyncio.create_task(
            fail_job(f"argo.workflow.{task_id}", task_id, message or "Unknown error")
        )
    elif status == "Succeeded":
        # Lineage: COMPLETE event (fire-and-forget)
        asyncio.create_task(
            complete_job(f"argo.workflow.{task_id}", task_id)
        )

    return {"ack": True}


# --- Budget Proxy / OpenAI-compatible LLM routing ---
LITELLM_BASE = os.environ.get("LITELLM_URL", os.environ.get("LITELLM_BASE_URL", "http://budget-proxy:4000"))
SYSTEM_PROMPT_PATH = os.environ.get("SYSTEM_PROMPT_PATH", "/app/config/prompts/system.txt")
_system_prompt_cache = None


def _validate_path(path: str) -> bool:
    """Validate path against traversal attacks."""
    # Allow known safe paths
    safe_paths = [
        "/etc/goose/system.txt",
        "/app/config/prompts/system.txt",
        "config/prompts/system.txt"
    ]
    if path in safe_paths:
        return True
        
    # For custom paths, ensure they are within /app/config or /etc/goose
    try:
        abs_path = os.path.abspath(path)
        base_configs = os.path.abspath("/app/config")
        base_etc = os.path.abspath("/etc/goose")
        
        # Check if path starts with base directories
        # Note: on Windows dev env this check might fail for linux paths, 
        # so we skip strict check if on Windows but keep logic for prod
        if os.name == 'nt': 
            return True
            
        return (abs_path.startswith(base_configs) or abs_path.startswith(base_etc))
    except Exception:
        return False


def _load_system_prompt() -> str:
    """Load the Goose system prompt for AI identity."""
    global _system_prompt_cache
    if _system_prompt_cache is not None:
        return _system_prompt_cache
        
    candidate_paths = [
        "/etc/goose/system.txt", 
        "/app/config/prompts/system.txt",
        "config/prompts/system.txt"
    ]
    
    # Only add custom path if it looks reasonable (basic check)
    if SYSTEM_PROMPT_PATH and ".." not in SYSTEM_PROMPT_PATH:
        candidate_paths.insert(2, SYSTEM_PROMPT_PATH)
        
    for path in candidate_paths:
        try:
            # Basic validation
            if ".." in path:
                continue
                
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    _system_prompt_cache = f.read().strip()
                    return _system_prompt_cache
        except (FileNotFoundError, PermissionError):
            continue
            
    _system_prompt_cache = "You are the Antigravity Node v13.0, a sovereign AI agent."
    return _system_prompt_cache



# OpenAI-compatible endpoint for LibreChat — routes through budget-proxy
@app.post("/v1/chat/completions")
@limiter.limit("30/minute")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    x_tenant_id: str = Header(default="system"),
    user: dict = Depends(validate_token),
):
    """OpenAI-compatible chat completions — routed through budget-proxy."""
    import httpx

    messages = [msg.model_dump() for msg in body.messages]
    user_msg = messages[-1].get("content", "")
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"Chat request (tenant={x_tenant_id}): {user_msg[:100]}")

    # 1. Record user message in episodic memory
    try:
        push_episodic(
            tenant_id=x_tenant_id,
            session_id=session_id,
            actor="User",
            action_type="TASK_REQUEST",
            content=user_msg[:1000],
        )
    except Exception as e:
        logger.warning(f"Memory write failed: {e}")

    # 2. Recall relevant past context
    history_context = ""
    try:
        history = recall_experience(user_msg, x_tenant_id, limit=5)
        if history:
            history_lines = [f"- [{h.get('action_type', '?')}] {h.get('content', '')[:200]}" for h in history]
            history_context = "\n\nRECENT MEMORY:\n" + "\n".join(history_lines)
    except Exception as e:
        logger.warning(f"Memory recall failed: {e}")

    # 3. Build messages with system prompt + memory context
    system_msg = _load_system_prompt()
    if history_context:
        system_msg += history_context

    enriched_messages = [{"role": "system", "content": system_msg}]
    enriched_messages.extend(messages)

    # 4. Route through budget-proxy
    model = body.model or os.environ.get("GOOSE_MODEL", "gpt-4o")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{LITELLM_BASE}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": enriched_messages,
                    "temperature": body.temperature,
                    "max_tokens": body.max_tokens,
                },
                headers={"Content-Type": "application/json"},
            )

        if resp.status_code == 200:
            result = resp.json()
            # Record AI response in episodic memory
            ai_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            try:
                push_episodic(
                    tenant_id=x_tenant_id,
                    session_id=session_id,
                    actor="Goose",
                    action_type="RESPONSE",
                    content=ai_content[:1000],
                )
            except Exception as e:
                logger.warning(f"Memory write for response failed: {e}")
            return result
        elif resp.status_code == 429:
            logger.warning("Budget proxy: daily budget exhausted")
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant",
                                "content": "⚠️ Budget limit reached ($10/day). The AI is paused until the budget resets. "
                                           "You can still use MCP tools, upload files, and check system health."},
                    "finish_reason": "stop",
                }],
            }
        else:
            error_text = resp.text[:500]
            logger.error(f"Budget proxy returned {resp.status_code}: {error_text}")
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant",
                                "content": f"Antigravity Node error: budget-proxy returned {resp.status_code}. "
                                           f"Check budget-proxy at http://localhost:4055/health"},
                    "finish_reason": "stop",
                }],
            }

    except httpx.ConnectError:
        logger.error("Cannot reach budget-proxy")
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant",
                            "content": "Budget-proxy is unreachable. The AI backend is offline. "
                                       "System status: check /health endpoint."},
                "finish_reason": "stop",
            }],
        }
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant",
                            "content": f"Antigravity Node v13.0 encountered an error: {str(e)[:200]}"},
                "finish_reason": "stop",
            }],
        }


# Also serve models list for LibreChat compatibility
@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models list — proxied from budget-proxy."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{LITELLM_BASE}/v1/models")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    # Fallback: return our default model
    return {
        "object": "list",
        "data": [
            {"id": "gpt-4o", "object": "model", "owned_by": "antigravity"},
            {"id": "claude-sonnet-4-20250514", "object": "model", "owned_by": "antigravity"},
        ],
    }


# --- MCP Tool Discovery Endpoint ---
@app.get("/tools", response_model=ToolsResponse)
async def list_tools():
    """List all available MCP tools across all tool servers."""
    import httpx

    tools = []

    # Orchestrator built-in tools (from mcp_server.py)
    builtin = [
        {"name": "search_memory", "server": "orchestrator", "description": "Search StarRocks memory tables for relevant context"},
        {"name": "query_memory", "server": "orchestrator", "description": "Execute SQL on StarRocks memory tables"},
        {"name": "trigger_task", "server": "orchestrator", "description": "Trigger an Argo workflow via Hera SDK"},
        {"name": "reflect_on_failure", "server": "orchestrator", "description": "Analyze logs from a failed workflow"},
        {"name": "ingest_file", "server": "orchestrator", "description": "Ingest document into semantic memory"},
    ]
    tools.extend(builtin)

    # MCP StarRocks tools
    mcp_servers = {
        "mcp-starrocks": "http://mcp-starrocks:8000",
        "mcp-filesystem": "http://mcp-filesystem:8000",
    }

    for server_name, base_url in mcp_servers.items():
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                # SSE endpoints only support GET, use a quick GET with small read
                resp = await client.get(f"{base_url}/sse", timeout=httpx.Timeout(2.0, connect=2.0))
                # HTTP 200 means SSE stream opened successfully
                tools.append({
                    "name": f"{server_name}",
                    "server": server_name,
                    "status": "connected",
                    "transport": "sse",
                    "url": f"{base_url}/sse",
                })
        except (httpx.ReadTimeout, httpx.RemoteProtocolError):
            # ReadTimeout means SSE connection opened but no events yet — that's OK
            tools.append({
                "name": f"{server_name}",
                "server": server_name,
                "status": "connected",
                "transport": "sse",
                "url": f"{base_url}/sse",
            })
        except Exception:
            tools.append({
                "name": f"{server_name}",
                "server": server_name,
                "status": "unreachable",
            })

    return {"tools": tools, "total": len(tools)}


# --- OVMS Inference Endpoints ---
@app.post("/v1/inference", response_model=InferenceResponse)
@limiter.limit("120/minute")
async def inference_endpoint(
    request: Request,
    body: InferenceRequest,
    x_tenant_id: str = Header(default="system"),
    user: dict = Depends(validate_token),
):
    """POST /v1/inference -- Run inference on an OVMS-served model.

    Gracefully handles empty model config (returns no_model_loaded status).
    Attempts gRPC first, falls back to REST.
    """
    logger.info(
        "Inference request: model=%s, tenant=%s",
        body.model_name, x_tenant_id,
    )
    result = await run_inference(body.model_name, body.input_data)
    status_code = 200 if result["status"] == "ok" else 200  # always 200; status in body
    return JSONResponse(content=result, status_code=status_code)


@app.get("/v1/models/ovms")
async def list_ovms_models():
    """GET /v1/models/ovms -- List models currently loaded in OVMS."""
    models = await ovms_list_models()
    return {"models": models, "count": len(models)}


# --- Agent Capabilities Summary ---
@app.get("/capabilities", response_model=CapabilitiesResponse)
async def capabilities():
    """Return full node capabilities for A2A discovery."""
    return {
        "node": "Antigravity Node v13.0",
        "protocols": ["a2a", "mcp", "openai-compatible"],
        "endpoints": {
            "health": "/health",
            "task": "/task",
            "upload": "/upload",
            "handoff": "/handoff",
            "webhook": "/webhook",
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "inference": "/v1/inference",
            "ovms_models": "/v1/models/ovms",
            "tools": "/tools",
            "capabilities": "/capabilities",
            "agent_descriptor": "/.well-known/agent.json",
        },
        "mcp_servers": {
            "mcp-starrocks": {"transport": "sse", "url": "http://mcp-starrocks:8000/sse"},
            "mcp-filesystem": {"transport": "sse", "url": "http://mcp-filesystem:8000/sse"},
            "mcp-gateway": {"transport": "sse", "url": "http://mcp-gateway:8080/sse"},
        },
        "memory": {
            "episodic": "StarRocks memory_episodic table",
            "semantic": "StarRocks memory_semantic table + Milvus vectors",
            "procedural": "StarRocks memory_procedural table",
        },
        "budget": {
            "proxy": "budget-proxy",
            "max_daily": "$10.00",
            "model": os.environ.get("GOOSE_MODEL", "gpt-4o"),
        },
    }
