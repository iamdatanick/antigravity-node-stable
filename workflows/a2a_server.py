"""Antigravity Node v14.1 — A2A Server (FastAPI endpoints)."""

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import os
import uuid

import httpx
from fastapi import Depends, FastAPI, Header, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse

from workflows.auth import validate_token
from workflows.goose_client import goose_reflect, list_tools
from workflows.health import full_health_check
from workflows.inference import list_models as ovms_list_models
from workflows.inference import run_inference
from workflows.memory import push_episodic, push_semantic, recall_experience
from workflows.models import (
    ChatCompletionRequest,
    HandoffRequest,
    InferenceRequest,
    TaskRequest,
    WebhookPayload,
)
from workflows.resilience import get_circuit_states, is_killed, trigger_kill
from workflows.s3_client import upload as s3_upload

logger = logging.getLogger("antigravity.a2a")

app = FastAPI(title="Antigravity Node", version="14.1")

LITELLM_URL = os.environ.get("LITELLM_URL", "http://budget-proxy:4000")
WELL_KNOWN_DIR = os.environ.get("WELL_KNOWN_DIR", "/app/well-known")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
SYSTEM_PROMPT_PATH = os.environ.get("SYSTEM_PROMPT_PATH", "/app/config/system_prompt.txt")


def _load_system_prompt() -> str:
    """Load system prompt from file if it exists."""
    try:
        with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are a helpful AI assistant."


# ---------------------------------------------------------------------------
# Background RAG processing
# ---------------------------------------------------------------------------

async def _process_upload_for_rag(data: bytes, filename: str, tenant_id: str, s3_key: str):
    """Background task: extract text, chunk, store in vector + semantic memory."""
    try:
        from workflows.document_processor import chunk_text, process_bytes

        text = process_bytes(data, filename)
        if not text or not text.strip():
            logger.info(f"RAG skip: no text extracted from {filename}")
            return

        chunks = chunk_text(text)
        doc_id = hashlib.sha256(f"{tenant_id}/{filename}".encode()).hexdigest()[:16]

        # Store chunks in semantic memory (StarRocks)
        for i, chunk in enumerate(chunks):
            push_semantic(
                doc_id=doc_id,
                tenant_id=tenant_id,
                chunk_id=i,
                content=chunk,
                source_uri=f"s3://{s3_key}",
            )

        # Store in vector DB (ChromaDB) if available
        try:
            from workflows.vector_store import vector_store

            ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"tenant_id": tenant_id, "filename": filename, "chunk": i} for i in range(len(chunks))]
            await vector_store.add_documents(chunks, metadatas, ids)
        except Exception as e:
            logger.warning(f"Vector store unavailable, skipping: {e}")

        logger.info(f"RAG complete: {filename} → {len(chunks)} chunks indexed for tenant {tenant_id}")
    except Exception as e:
        logger.error(f"RAG processing failed for {filename}: {e}")


# --- Health ---


@app.get("/health")
async def health():
    if is_killed():
        return JSONResponse({"status": "killed", "message": "Kill switch activated"}, status_code=503)
    result = await full_health_check()
    result["circuits"] = get_circuit_states()
    status_code = 200 if result.get("status") == "healthy" else 503
    return JSONResponse(result, status_code=status_code)


# --- Capabilities ---


@app.get("/capabilities")
async def capabilities():
    from workflows.mcp_registry import registry

    mcp_servers = {
        "filesystem": {"url": "http://mcp-filesystem:8000/sse", "status": "active"},
        "memory": {"url": "http://mcp-starrocks:8000/sse", "status": "active"},
    }
    for name, server in registry.servers.items():
        mcp_servers[name] = {"url": server["url"], "status": "registered"}

    return {
        "node": "Antigravity Node v14.1 Phoenix",
        "protocols": ["a2a", "mcp", "grpc", "openlineage"],
        "endpoints": {
            "health": "GET /health",
            "task": "POST /task",
            "chat": "POST /v1/chat/completions",
            "upload": "POST /upload",
            "tools": "GET /tools",
            "webhook": "POST /webhook",
            "handoff": "POST /handoff",
            "agent_card": "GET /.well-known/agent.json",
            "mcp_servers": "GET /mcp/servers",
        },
        "mcp_servers": mcp_servers,
        "memory": {
            "episodic": "StarRocks memory_episodic",
            "semantic": "StarRocks memory_semantic",
            "vector": "ChromaDB antigravity_docs",
        },
        "budget": {
            "daily_limit_usd": float(os.environ.get("DAILY_BUDGET_USD", "50")),
            "provider": "budget-proxy",
        },
    }


# --- Agent Card ---


@app.get("/.well-known/agent.json")
async def agent_card():
    path = os.path.join(WELL_KNOWN_DIR, "agent.json")
    if os.path.isfile(path):
        return FileResponse(path, media_type="application/json")
    return JSONResponse({"error": "agent.json not found"}, status_code=404)


# --- Tools ---


@app.get("/tools")
async def tools():
    tool_list = list_tools()
    return {"tools": tool_list, "total": len(tool_list)}


# --- Task Submission (auth-protected) ---


@app.post("/task")
async def task(
    body: TaskRequest,
    x_tenant_id: str | None = Header(default=None),
    token_payload: dict = Depends(validate_token),
):
    if not x_tenant_id:
        return JSONResponse({"detail": "x-tenant-id header required"}, status_code=400)

    session_id = body.session_id or str(uuid.uuid4())

    history = recall_experience(body.goal, tenant_id=x_tenant_id, limit=10)

    push_episodic(
        tenant_id=x_tenant_id,
        session_id=session_id,
        actor=token_payload.get("sub", "anonymous"),
        action_type="task_submit",
        content=body.goal,
    )

    return {
        "status": "accepted",
        "session_id": session_id,
        "tenant_id": x_tenant_id,
        "history_count": len(history),
    }


# --- Handoff (auth-protected) ---


@app.post("/handoff")
async def handoff(
    body: HandoffRequest,
    token_payload: dict = Depends(validate_token),
):
    return {"status": "handoff_acknowledged", "target": body.target_agent}


# --- Webhook ---


@app.post("/webhook")
async def webhook(request: Request):
    raw_body = await request.body()

    if WEBHOOK_SECRET:
        sig_header = request.headers.get("x-webhook-signature", "")
        expected = hmac.new(WEBHOOK_SECRET.encode(), raw_body, hashlib.sha256).hexdigest()
        provided = sig_header.removeprefix("sha256=")
        if not hmac.compare_digest(expected, provided):
            return JSONResponse({"detail": "Invalid webhook signature"}, status_code=401)

    body = WebhookPayload.model_validate_json(raw_body)
    if body.status == "Failed":
        await goose_reflect(body.task_id, body.message or "Unknown error")
    return {"ack": True}


# --- File Upload (auth-protected, triggers RAG pipeline) ---


@app.post("/upload")
async def upload(
    file: UploadFile,
    x_tenant_id: str | None = Header(default=None),
    token_payload: dict = Depends(validate_token),
):
    if not x_tenant_id:
        return JSONResponse({"error": "x-tenant-id header required"}, status_code=400)

    data = await file.read()
    key = f"{x_tenant_id}/{file.filename}"
    s3_upload(key, data)

    push_episodic(
        tenant_id=x_tenant_id,
        session_id="upload",
        actor=token_payload.get("sub", "anonymous"),
        action_type="file_upload",
        content=f"Uploaded {file.filename} ({len(data)} bytes)",
    )

    # Trigger RAG processing in background (non-blocking)
    asyncio.create_task(_process_upload_for_rag(data, file.filename, x_tenant_id, key))

    return {"status": "uploaded", "key": key, "size": len(data), "rag": "processing"}


# --- Chat Completions (auth-protected, OpenAI-compatible proxy) ---


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    x_tenant_id: str | None = Header(default=None),
    token_payload: dict = Depends(validate_token),
):
    tenant = x_tenant_id or "anonymous"

    recall_experience(
        body.messages[-1].content if body.messages else "",
        tenant_id=tenant,
        limit=5,
    )

    model = body.model or os.environ.get("GOOSE_MODEL", "gpt-4o")
    payload = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in body.messages],
        "temperature": body.temperature,
        "max_tokens": body.max_tokens,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{LITELLM_URL}/v1/chat/completions",
            json=payload,
            timeout=60.0,
        )
        result = resp.json()

    push_episodic(
        tenant_id=tenant,
        session_id="chat",
        actor=token_payload.get("sub", "anonymous"),
        action_type="chat_completion",
        content=json.dumps({"model": model, "messages_count": len(body.messages)}),
    )

    return result


# --- Inference ---


@app.post("/v1/inference")
async def inference(body: InferenceRequest):
    return await run_inference(body.model_name, body.input_data)


@app.get("/v1/models/ovms")
async def get_ovms_models():
    models = await ovms_list_models()
    return {"models": models, "count": len(models)}


# --- Models ---


@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": "gpt-4o"}, {"id": "tinyllama"}]}


# --- MCP Server Registry ---


@app.get("/mcp/servers")
async def mcp_servers_list():
    """List all registered MCP servers (built-in + third-party)."""
    from workflows.mcp_registry import registry

    return registry.list_servers()


@app.post("/mcp/servers")
async def mcp_servers_register(
    request: Request,
    token_payload: dict = Depends(validate_token),
):
    """Register a third-party MCP server with optional OAuth."""
    from workflows.mcp_registry import registry

    body = await request.json()
    name = body.get("name")
    url = body.get("url")
    if not name or not url:
        return JSONResponse({"detail": "name and url required"}, status_code=400)

    registry.register(
        name=name,
        url=url,
        transport=body.get("transport", "sse"),
        oauth_config=body.get("oauth"),
        description=body.get("description", ""),
    )
    return {"status": "registered", "name": name, "url": url}


@app.delete("/mcp/servers/{name}")
async def mcp_servers_remove(
    name: str,
    token_payload: dict = Depends(validate_token),
):
    """Remove a registered third-party MCP server."""
    from workflows.mcp_registry import registry

    if registry.remove(name):
        return {"status": "removed", "name": name}
    return JSONResponse({"detail": f"Server '{name}' not found"}, status_code=404)


@app.post("/mcp/servers/{name}/tools")
async def mcp_server_call_tool(
    name: str,
    request: Request,
    token_payload: dict = Depends(validate_token),
):
    """Call a tool on a registered MCP server (handles OAuth automatically)."""
    from workflows.mcp_registry import registry

    body = await request.json()
    tool_name = body.get("tool")
    params = body.get("params", {})

    if not tool_name:
        return JSONResponse({"detail": "tool name required"}, status_code=400)

    result = await registry.call_tool(name, tool_name, params)
    return result


# --- Admin (auth-protected) ---


@app.get("/admin/circuits")
async def admin_circuits(token_payload: dict = Depends(validate_token)):
    """Circuit breaker status for all external service calls."""
    return get_circuit_states()


@app.post("/admin/kill-switch")
async def admin_kill_switch(token_payload: dict = Depends(validate_token)):
    """Emergency stop — halts all orchestrator operations."""
    return trigger_kill()


# --- WebSocket Logs ---


@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    await websocket.accept()
    try:
        process = await asyncio.create_subprocess_exec(
            "tail", "-f", "/proc/1/fd/1",
            stdout=asyncio.subprocess.PIPE,
        )
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            await websocket.send_text(line.decode())
    except Exception:
        pass
    finally:
        with contextlib.suppress(Exception):
            process.terminate()
        await websocket.close()
