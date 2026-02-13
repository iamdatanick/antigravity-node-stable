"""Antigravity Node v14.1 — A2A Server (FastAPI endpoints)."""

import asyncio
import contextlib
import hashlib
import hmac
import json
import os
import uuid

import httpx
from fastapi import FastAPI, Header, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse

from workflows.auth import validate_token  # noqa: F401 — patched by tests
from workflows.goose_client import goose_reflect, list_tools
from workflows.health import full_health_check
from workflows.inference import list_models as ovms_list_models
from workflows.inference import run_inference
from workflows.memory import push_episodic, recall_experience
from workflows.models import (
    ChatCompletionRequest,
    HandoffRequest,
    InferenceRequest,
    TaskRequest,
    WebhookPayload,
)
from workflows.resilience import get_circuit_states, is_killed, trigger_kill
from workflows.s3_client import upload as s3_upload

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
        },
        "mcp_servers": {
            "filesystem": {"url": "http://mcp-filesystem:8000/sse", "status": "active"},
            "memory": {"url": "http://mcp-starrocks:8000/sse", "status": "active"},
        },
        "memory": {
            "episodic": "StarRocks memory_episodic",
            "semantic": "StarRocks memory_semantic",
            "vector": "Milvus v2.4",
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


# --- Task Submission ---


@app.post("/task")
async def task(body: TaskRequest, x_tenant_id: str | None = Header(default=None)):
    if not x_tenant_id:
        return JSONResponse({"detail": "x-tenant-id header required"}, status_code=400)

    session_id = body.session_id or str(uuid.uuid4())

    history = recall_experience(body.goal, tenant_id=x_tenant_id, limit=10)

    push_episodic(
        tenant_id=x_tenant_id,
        session_id=session_id,
        actor="user",
        action_type="task_submit",
        content=body.goal,
    )

    return {
        "status": "accepted",
        "session_id": session_id,
        "tenant_id": x_tenant_id,
        "history_count": len(history),
    }


# --- Handoff ---


@app.post("/handoff")
async def handoff(body: HandoffRequest):
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


# --- File Upload ---


@app.post("/upload")
async def upload(file: UploadFile, x_tenant_id: str | None = Header(default=None)):
    if not x_tenant_id:
        return JSONResponse({"error": "x-tenant-id header required"}, status_code=400)

    data = await file.read()
    key = f"{x_tenant_id}/{file.filename}"
    s3_upload(key, data)

    push_episodic(
        tenant_id=x_tenant_id,
        session_id="upload",
        actor="user",
        action_type="file_upload",
        content=f"Uploaded {file.filename} ({len(data)} bytes)",
    )

    return {"status": "uploaded", "key": key, "size": len(data)}


# --- Chat Completions (OpenAI-compatible proxy) ---


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    x_tenant_id: str | None = Header(default=None),
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
        actor="assistant",
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


# --- Admin ---


@app.get("/admin/circuits")
async def admin_circuits():
    """Circuit breaker status for all external service calls."""
    return get_circuit_states()


@app.post("/admin/kill-switch")
async def admin_kill_switch():
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
