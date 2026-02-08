"""FastAPI A2A endpoints: /health, /task, /handoff, /upload, /webhook, /.well-known/agent.json."""

import os
import logging
import uuid
from fastapi import FastAPI, Header, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse

from workflows.health import full_health_check
from workflows.memory import push_episodic, recall_experience
from workflows.s3_client import upload as s3_upload
from workflows.goose_client import execute_tool_with_correction, goose_reflect

logger = logging.getLogger("antigravity.a2a")

app = FastAPI(title="Antigravity Node v13.0", version="13.0.0")


@app.get("/health")
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


@app.post("/task")
async def task(
    body: dict,
    x_tenant_id: str = Header(default=None),
):
    """POST /task — A2A task endpoint with multi-tenant isolation."""
    if not x_tenant_id:
        raise HTTPException(status_code=400, detail="x-tenant-id header required")

    goal = body.get("goal", "")
    context = body.get("context", "")
    session_id = body.get("session_id", str(uuid.uuid4()))

    logger.info(f"Task received: tenant={x_tenant_id}, goal={goal[:100]}")

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

    return {
        "status": "accepted",
        "session_id": session_id,
        "tenant_id": x_tenant_id,
        "history_count": len(history),
    }


@app.post("/handoff")
async def handoff(body: dict, x_tenant_id: str = Header(default="system")):
    """POST /handoff — A2A agent-to-agent handoff."""
    target = body.get("target_agent", "")
    payload = body.get("payload", {})
    logger.info(f"Handoff to {target} from tenant={x_tenant_id}")
    return {"status": "handoff_acknowledged", "target": target}


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    x_tenant_id: str = Header(default="system"),
):
    """POST /upload — HTTP file upload to SeaweedFS (Gap #14 fix)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Check file size (100MB limit)
    content = await file.read()
    max_size = 100 * 1024 * 1024  # 100MB
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail="File too large (max 100MB)")

    key = f"context/{x_tenant_id}/{file.filename}"
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


@app.post("/webhook")
async def argo_webhook(payload: dict):
    """POST /webhook — Argo exit-handler callback (Gap #6 fix)."""
    task_id = payload.get("task_id", "unknown")
    status = payload.get("status", "unknown")
    message = payload.get("message", "")

    logger.info(f"Argo callback: task={task_id}, status={status}")

    if status == "Failed":
        await goose_reflect(task_id, message)
        logger.warning(f"Argo workflow {task_id} failed. Goose self-correction triggered.")

    return {"ack": True}


# --- LiteLLM/OpenAI proxy client ---
LITELLM_BASE = os.environ.get("LITELLM_BASE_URL", "http://litellm:4000")
SYSTEM_PROMPT_PATH = os.environ.get("SYSTEM_PROMPT_PATH", "/app/well-known/../config/prompts/system.txt")
_system_prompt_cache = None


def _load_system_prompt() -> str:
    """Load the Goose system prompt for AI identity."""
    global _system_prompt_cache
    if _system_prompt_cache is not None:
        return _system_prompt_cache
    for path in ["/etc/goose/system.txt", "/app/config/prompts/system.txt",
                 SYSTEM_PROMPT_PATH, "config/prompts/system.txt"]:
        try:
            with open(path, "r") as f:
                _system_prompt_cache = f.read().strip()
                return _system_prompt_cache
        except FileNotFoundError:
            continue
    _system_prompt_cache = "You are the Antigravity Node v13.0, a sovereign AI agent."
    return _system_prompt_cache


# OpenAI-compatible endpoint for Open WebUI — routes through LiteLLM
@app.post("/v1/chat/completions")
async def chat_completions(body: dict, x_tenant_id: str = Header(default="system")):
    """OpenAI-compatible chat completions — routed through LiteLLM proxy."""
    import httpx

    messages = body.get("messages", [])
    if not messages:
        return {"choices": [{"message": {"role": "assistant", "content": "No input provided."}}]}

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

    # 4. Route through LiteLLM proxy
    model = body.get("model", os.environ.get("GOOSE_MODEL", "gpt-4o"))
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{LITELLM_BASE}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": enriched_messages,
                    "temperature": body.get("temperature", 0.7),
                    "max_tokens": body.get("max_tokens", 2048),
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
            logger.warning("LiteLLM budget exhausted — returning budget error")
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
            logger.error(f"LiteLLM returned {resp.status_code}: {error_text}")
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant",
                                "content": f"Antigravity Node error: LiteLLM proxy returned {resp.status_code}. "
                                           f"Check LiteLLM config at http://localhost:4055/health"},
                    "finish_reason": "stop",
                }],
            }

    except httpx.ConnectError:
        logger.error("Cannot reach LiteLLM proxy")
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant",
                            "content": "🔌 LiteLLM proxy is unreachable. The AI backend is offline. "
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


# Also serve models list for Open WebUI compatibility
@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models list — proxied from LiteLLM."""
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
@app.get("/tools")
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


# --- Agent Capabilities Summary ---
@app.get("/capabilities")
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
            "proxy": "LiteLLM",
            "max_daily": "$10.00",
            "model": os.environ.get("GOOSE_MODEL", "gpt-4o"),
        },
    }
