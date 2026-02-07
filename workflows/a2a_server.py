"""FastAPI A2A endpoints: /health, /task, /handoff, /upload, /webhook, /.well-known/agent.json."""

import os
import json
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


# OpenAI-compatible endpoint for Open WebUI
@app.post("/v1/chat/completions")
async def chat_completions(body: dict):
    """OpenAI-compatible chat completions — proxied by Open WebUI."""
    messages = body.get("messages", [])
    if not messages:
        return {"choices": [{"message": {"role": "assistant", "content": "No input provided."}}]}

    user_msg = messages[-1].get("content", "")
    logger.info(f"Chat completion request: {user_msg[:100]}")

    # Simple echo response — in production, would route to Goose
    response_text = f"Antigravity Node v13.0 received: {user_msg}"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
    }
