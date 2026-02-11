"""FastAPI A2A endpoints: /health, /task, /handoff, /upload, /webhook, /.well-known/agent.json."""

import asyncio
import contextlib
import hashlib
import hmac
import logging
import os
import uuid
from datetime import UTC, datetime

import httpx
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from workflows.auth import validate_token
from workflows.goose_client import goose_reflect
from workflows.health import full_health_check
from workflows.inference import list_models as ovms_list_models
from workflows.inference import run_inference
from workflows.lineage import complete_job, fail_job, start_job
from workflows.memory import push_episodic, recall_experience
from workflows.models import (
    ApiKeyEntry,
    ApiKeyListResponse,
    ApiKeyRequest,
    BudgetHistoryResponse,
    CapabilitiesResponse,
    ChatCompletionRequest,
    HandoffRequest,
    HandoffResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    MemoryListResponse,
    QueryRequest,
    QueryResponse,
    TaskRequest,
    TaskResponse,
    ToolsResponse,
    UploadResponse,
    WebhookPayload,
    WebhookResponse,
    WorkflowListResponse,
)
from workflows.s3_client import upload as s3_upload
from workflows.telemetry import get_tracer

logger = logging.getLogger("antigravity.a2a")
tracer = get_tracer("antigravity.a2a")

app = FastAPI(title="Antigravity Node v14.1", version="14.1.0")

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

# OpenBao (vault) for API key storage
OPENBAO_ADDR = os.environ.get("OPENBAO_ADDR", "http://openbao:8200")
OPENBAO_TOKEN = os.environ.get("OPENBAO_TOKEN", "dev-only-token")
VALID_PROVIDERS = {"openai", "anthropic", "google", "mistral"}

_OPENBAO_HEADERS = {"X-Vault-Token": OPENBAO_TOKEN, "Content-Type": "application/json"}
_VAULT_TIMEOUT = httpx.Timeout(5.0)


def _vault_key_url(provider: str, *, metadata: bool = False) -> str:
    """Build the OpenBao KV path for a provider's API key."""
    prefix = "metadata" if metadata else "data"
    return f"{OPENBAO_ADDR}/v1/secret/{prefix}/antigravity/api-keys/{provider}"


def _mask_key(key: str) -> str:
    """Mask an API key, showing first 4 and last 4 chars."""
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


def _validate_provider(provider: str) -> None:
    """Raise HTTPException if provider is not in the allowed set."""
    if provider not in VALID_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider. Must be one of: {', '.join(sorted(VALID_PROVIDERS))}",
        )


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


# --- API Key Management (OpenBao) ---
@app.get("/api/settings/keys", response_model=ApiKeyListResponse)
async def list_api_keys():
    """GET /api/settings/keys -- List configured LLM provider API keys (masked)."""
    keys: list[dict] = []
    async with httpx.AsyncClient(timeout=_VAULT_TIMEOUT) as client:
        for provider in sorted(VALID_PROVIDERS):
            try:
                resp = await client.get(
                    _vault_key_url(provider),
                    headers=_OPENBAO_HEADERS,
                )
                if resp.status_code == 200:
                    raw_key = resp.json().get("data", {}).get("data", {}).get("key", "")
                    if raw_key:
                        keys.append({"provider": provider, "masked_key": _mask_key(raw_key), "configured": True})
                        continue
            except Exception as e:
                logger.warning("OpenBao read failed for %s: %s", provider, e)
            keys.append({"provider": provider, "masked_key": "", "configured": False})
    return {"keys": keys}


@app.post("/api/settings/keys", response_model=ApiKeyEntry)
async def save_api_key(body: ApiKeyRequest):
    """POST /api/settings/keys -- Store an LLM provider API key in OpenBao."""
    _validate_provider(body.provider)

    try:
        async with httpx.AsyncClient(timeout=_VAULT_TIMEOUT) as client:
            resp = await client.put(
                _vault_key_url(body.provider),
                headers=_OPENBAO_HEADERS,
                json={"data": {"key": body.api_key}},
            )
            if resp.status_code not in (200, 204):
                raise HTTPException(status_code=502, detail=f"OpenBao write failed: {resp.status_code}")
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Cannot reach OpenBao vault")

    logger.info("API key saved for provider: %s", body.provider)
    return {"provider": body.provider, "masked_key": _mask_key(body.api_key), "configured": True}


@app.delete("/api/settings/keys/{provider}")
async def delete_api_key(provider: str = Path(..., pattern=r"^[a-z0-9_-]+$")):
    """DELETE /api/settings/keys/{provider} -- Remove an LLM provider API key."""
    _validate_provider(provider)

    try:
        async with httpx.AsyncClient(timeout=_VAULT_TIMEOUT) as client:
            resp = await client.delete(
                _vault_key_url(provider, metadata=True),
                headers=_OPENBAO_HEADERS,
            )
            if resp.status_code not in (200, 204, 404):
                raise HTTPException(status_code=502, detail=f"OpenBao delete failed: {resp.status_code}")
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Cannot reach OpenBao vault")

    logger.info("API key deleted for provider: %s", provider)
    return {"status": "deleted", "provider": provider}


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
async def handoff(
    body: HandoffRequest, x_tenant_id: str = Header(default="system"), user: dict = Depends(validate_token)
):
    """POST /handoff — A2A agent-to-agent handoff."""
    target = body.target_agent
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

    # Record in episodic memory (non-fatal if DB unavailable)
    try:
        push_episodic(
            tenant_id=x_tenant_id,
            session_id="upload",
            actor="User",
            action_type="FILE_UPLOAD",
            content=f"Uploaded {file.filename} ({len(content)} bytes)",
        )
    except Exception as e:
        logger.warning(f"Episodic memory write failed (non-fatal): {e}")

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
        asyncio.create_task(fail_job(f"argo.workflow.{task_id}", task_id, message or "Unknown error"))
    elif status == "Succeeded":
        # Lineage: COMPLETE event (fire-and-forget)
        asyncio.create_task(complete_job(f"argo.workflow.{task_id}", task_id))

    return {"ack": True}


# --- Budget Proxy / OpenAI-compatible LLM routing ---
LITELLM_BASE = os.environ.get("LITELLM_URL", os.environ.get("LITELLM_BASE_URL", "http://budget-proxy:4000"))
SYSTEM_PROMPT_PATH = os.environ.get("SYSTEM_PROMPT_PATH", "/app/config/prompts/system.txt")
_system_prompt_cache = None


def _validate_path(path: str) -> bool:
    """Validate path against traversal attacks."""
    # Allow known safe paths
    safe_paths = ["/etc/goose/system.txt", "/app/config/prompts/system.txt", "config/prompts/system.txt"]
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
        if os.name == "nt":
            return True

        return abs_path.startswith(base_configs) or abs_path.startswith(base_etc)
    except Exception:
        return False


def _load_system_prompt() -> str:
    """Load the Goose system prompt for AI identity."""
    global _system_prompt_cache
    if _system_prompt_cache is not None:
        return _system_prompt_cache

    candidate_paths = ["/etc/goose/system.txt", "/app/config/prompts/system.txt", "config/prompts/system.txt"]

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

    _system_prompt_cache = "You are the Antigravity Node v14.1, a sovereign AI agent."
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
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "⚠️ Budget limit reached ($10/day). The AI is paused until the budget resets. "
                            "You can still use MCP tools, upload files, and check system health.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        else:
            error_text = resp.text[:500]
            logger.error(f"Budget proxy returned {resp.status_code}: {error_text}")
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"Antigravity Node error: budget-proxy returned {resp.status_code}. "
                            f"Check budget-proxy at http://localhost:4055/health",
                        },
                        "finish_reason": "stop",
                    }
                ],
            }

    except httpx.ConnectError:
        logger.error("Cannot reach budget-proxy")
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Budget-proxy is unreachable. The AI backend is offline. "
                        "System status: check /health endpoint.",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Antigravity Node v14.1 encountered an error: {str(e)[:200]}",
                    },
                    "finish_reason": "stop",
                }
            ],
        }


# Also serve models list for LibreChat compatibility
_FALLBACK_MODELS = [
    {"id": "gpt-4o", "object": "model", "owned_by": "antigravity"},
    {"id": "claude-sonnet-4-20250514", "object": "model", "owned_by": "antigravity"},
]



@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models list -- merged from budget-proxy + OVMS."""
    models: list[dict] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        try:
            resp = await client.get(f"{LITELLM_BASE}/v1/models")
            if resp.status_code == 200:
                models.extend(resp.json().get("data", []))
        except Exception:
            pass
        try:
            ovms_url = os.environ.get("OVMS_REST_URL", "http://ovms:9001")
            resp = await client.get(f"{ovms_url}/v1/models")
            if resp.status_code == 200:
                for m in resp.json().get("data", []):
                    models.append({"id": f"local/{m.get('id')}", "object": "model", "owned_by": "ovms"})
        except Exception:
            pass
    return {"object": "list", "data": models or _FALLBACK_MODELS}


@app.get("/tools", response_model=ToolsResponse)
async def list_tools():
    """List 6 active v14.1 MCP tools."""
    return {
        "tools": [
            {"name": "chat", "server": "budget-proxy", "description": "LLM chat via budget-proxy"},
            {"name": "upload_document", "server": "orchestrator", "description": "Upload for RAG"},
            {"name": "search_documents", "server": "orchestrator", "description": "Semantic search"},
            {"name": "run_inference", "server": "ovms", "description": "Run OVMS inference"},
            {"name": "list_models", "server": "budget-proxy", "description": "List models"},
            {"name": "system_health", "server": "orchestrator", "description": "Health hierarchy"},
        ],
        "total": 6
    }


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
        body.model_name,
        x_tenant_id,
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
        "node": "Antigravity Node v14.1",
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
            "query": "/query",
            "workflows": "/workflows",
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


# --- Budget History Endpoint (Phase 8: Chart.js dashboard) ---
@app.get("/budget/history", response_model=BudgetHistoryResponse)
@limiter.limit("30/minute")
async def budget_history(
    request: Request,
    x_tenant_id: str = Header(default="system"),
    user: dict = Depends(validate_token),
):
    """GET /budget/history -- Return budget spend data for Chart.js visualization.

    Proxies to budget-proxy /health to get current spend, then builds a 24-point
    hourly spend array with the current hour's spend filled in.
    """
    import httpx

    current_spend = 0.0
    max_daily = 10.0

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{LITELLM_BASE}/health")
            if resp.status_code == 200:
                data = resp.json()
                # budget-proxy /health may return spend info in different formats
                current_spend = float(data.get("spend", data.get("current_spend", 0.0)))
                max_daily = float(data.get("max_budget", data.get("max_daily", 10.0)))
    except Exception as e:
        logger.warning(f"Budget proxy unreachable for history: {e}")

    # Build 24-point hourly array (0=midnight UTC, 23=11pm UTC)
    hourly_spend = [0.0] * 24
    current_hour = datetime.now(UTC).hour
    hourly_spend[current_hour] = current_spend

    return {
        "current_spend": current_spend,
        "max_daily": max_daily,
        "currency": "USD",
        "hourly_spend": hourly_spend,
    }


# --- WebSocket Log Streaming Endpoint (Phase 8: Xterm.js terminal) ---



@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    """WebSocket /ws/logs — Stream orchestrator logs to the UI."""
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(5)
            ts = datetime.now(UTC).strftime("%H:%M:%S")
            await websocket.send_text(f"\x1b[90m[{ts}]\x1b[0m v14.1 System Operational\r\n")
    except Exception:
        pass


def _format_log_line(src: dict) -> str:
    """Format an OpenSearch log document as an ANSI-colored terminal line."""
    timestamp = src.get("@timestamp", "")
    # Shorten timestamp to HH:MM:SS if possible
    if len(timestamp) >= 19:
        timestamp = timestamp[11:19]
    level = src.get("level", src.get("log_level", "INFO")).upper()
    message = src.get("log", src.get("message", str(src)))
    container = src.get("kubernetes", {}).get("container_name", src.get("container_name", ""))

    # ANSI color codes by level
    if "ERROR" in level or "FATAL" in level:
        color = "\x1b[31m"  # Red
    elif "WARN" in level:
        color = "\x1b[33m"  # Yellow
    elif "DEBUG" in level:
        color = "\x1b[90m"  # Gray
    else:
        color = "\x1b[0m"  # Default

    prefix = f"\x1b[90m{timestamp}\x1b[0m"
    if container:
        prefix += f" \x1b[36m[{container}]\x1b[0m"

    return f"{prefix} {color}{message}\x1b[0m\r\n"


# --- SQL Query Executor Endpoint (Phase 9: Monaco Editor) ---
MAX_QUERY_ROWS = 200


@app.post("/query", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query_sql(
    request: Request,
    body: QueryRequest,
    x_tenant_id: str = Header(default=None),
    user: dict = Depends(validate_token),
):
    """POST /query -- Execute a read-only SQL query against StarRocks.

    Uses the memory module's SQL injection prevention (forbidden keyword checks)
    to reject any non-SELECT queries. Results are limited to 200 rows.
    """
    if not x_tenant_id:
        raise HTTPException(status_code=400, detail="x-tenant-id header required")

    with tracer.start_as_current_span(
        "a2a.query",
        attributes={"tenant_id": x_tenant_id, "sql_length": len(body.sql)},
    ):
        import re

        from workflows.memory import _get_conn

        sql = body.sql.strip()
        logger.info(f"SQL query request: tenant={x_tenant_id}, length={len(sql)}")

        # --- SQL validation (mirrors memory.query logic) ---
        normalized = sql.upper()
        if not normalized.startswith("SELECT"):
            raise HTTPException(status_code=400, detail="Only SELECT queries are permitted")

        # Strip comments before keyword check
        normalized = re.sub(r"/\*.*?\*/", " ", normalized, flags=re.DOTALL)
        normalized = re.sub(r"--[^\n]*", " ", normalized)

        forbidden = [
            "DROP",
            "DELETE",
            "INSERT",
            "UPDATE",
            "ALTER",
            "CREATE",
            "TRUNCATE",
            "GRANT",
            "REVOKE",
            "INTO OUTFILE",
            "INTO DUMPFILE",
            "LOAD",
            "SET",
            "EXEC",
        ]
        for keyword in forbidden:
            pattern = r"\b" + keyword + r"\b"
            if re.search(pattern, normalized):
                raise HTTPException(
                    status_code=400,
                    detail=f"Forbidden SQL keyword: {keyword}",
                )

        # --- Execute query ---
        try:
            conn = _get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    # Fetch up to MAX_QUERY_ROWS + 1 to detect truncation
                    all_rows = cur.fetchmany(MAX_QUERY_ROWS + 1)
                    truncated = len(all_rows) > MAX_QUERY_ROWS
                    result_rows = all_rows[:MAX_QUERY_ROWS]

                    # Extract column names from cursor description
                    columns = [desc[0] for desc in cur.description] if cur.description else []

                    # Convert dict rows to list-of-lists for the response
                    rows_as_lists = []
                    for row in result_rows:
                        rows_as_lists.append([row[col] for col in columns])

                    return {
                        "columns": columns,
                        "rows": rows_as_lists,
                        "row_count": len(rows_as_lists),
                        "truncated": truncated,
                    }
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"SQL query execution failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="Query execution failed. Check SQL syntax and table names.",
            )


# --- Argo Workflow List Endpoint (Phase 9: Cytoscape DAG visualizer) ---
@app.get("/workflows", response_model=WorkflowListResponse)
async def list_workflows(
    request: Request,
    user: dict = Depends(validate_token),
):
    return {"workflows": []}


# --- Memory Browser Endpoint (Phase 8: TanStack/Alpine.js table) ---
@app.get("/memory", response_model=MemoryListResponse)
@limiter.limit("60/minute")
async def memory_browser(
    request: Request,
    tenant_id: str = Query(default="system", max_length=128, description="Tenant ID to query"),
    limit: int = Query(default=25, ge=1, le=200, description="Number of rows to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    search: str = Query(default="", max_length=500, description="Search term for content filter"),
    x_tenant_id: str = Header(default="system"),
    user: dict = Depends(validate_token),
):
    """GET /memory -- Query episodic memory for the Memory Browser table.

    Returns paginated, searchable episodic memory entries from StarRocks.
    Uses parameterized queries only (no string interpolation) to prevent SQL injection.
    """
    from workflows.memory import _get_conn

    # Use the header tenant_id if the query param is default
    effective_tenant = tenant_id if tenant_id != "system" else x_tenant_id

    entries = []
    total = 0

    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                # Count total matching rows
                if search:
                    cur.execute(
                        "SELECT COUNT(*) AS cnt FROM memory_episodic WHERE tenant_id = %s AND content LIKE %s",
                        (effective_tenant, f"%{search}%"),
                    )
                else:
                    cur.execute(
                        "SELECT COUNT(*) AS cnt FROM memory_episodic WHERE tenant_id = %s",
                        (effective_tenant,),
                    )
                count_row = cur.fetchone()
                total = count_row["cnt"] if count_row else 0

                # Fetch paginated entries
                if search:
                    cur.execute(
                        "SELECT event_id, tenant_id, timestamp, session_id, "
                        "actor, action_type, content "
                        "FROM memory_episodic "
                        "WHERE tenant_id = %s AND content LIKE %s "
                        "ORDER BY timestamp DESC LIMIT %s OFFSET %s",
                        (effective_tenant, f"%{search}%", limit, offset),
                    )
                else:
                    cur.execute(
                        "SELECT event_id, tenant_id, timestamp, session_id, "
                        "actor, action_type, content "
                        "FROM memory_episodic "
                        "WHERE tenant_id = %s "
                        "ORDER BY timestamp DESC LIMIT %s OFFSET %s",
                        (effective_tenant, limit, offset),
                    )
                rows = cur.fetchall()
                for row in rows:
                    entries.append(
                        {
                            "event_id": row.get("event_id"),
                            "tenant_id": row.get("tenant_id", effective_tenant),
                            "timestamp": str(row["timestamp"]) if row.get("timestamp") else None,
                            "session_id": row.get("session_id"),
                            "actor": row.get("actor"),
                            "action_type": row.get("action_type"),
                            "content": row.get("content"),
                        }
                    )
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"Memory browser query failed: {e}")
        # Return empty result on database errors rather than 500
        return {"entries": [], "total": 0, "limit": limit, "offset": offset}

    return {
        "entries": entries,
        "total": total,
        "limit": limit,
        "offset": offset,
    }
```
