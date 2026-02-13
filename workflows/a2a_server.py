import asyncio
import hashlib
import hmac
import os
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from workflows.goose_client import goose_reflect, list_tools
from workflows.health import full_health_check
from workflows.inference import list_models as ovms_list_models
from workflows.inference import run_inference
from workflows.memory import push_episodic, recall_experience
from workflows.models import (
    CapabilitiesResponse,
    ChatCompletionRequest,
    HandoffRequest,
    HandoffResponse,
    InferenceRequest,
    InferenceResponse,
    TaskRequest,
    TaskResponse,
    ToolInfo,
    ToolsResponse,
    UploadResponse,
    WebhookPayload,
    WebhookResponse,
)
from workflows.resilience import get_circuit_states, is_killed, trigger_kill
from workflows.s3_client import upload as s3_upload

app = FastAPI(title="Antigravity Node", version="13.1")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded"},
    )


# Constants
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
LITELLM_URL = os.environ.get("LITELLM_URL", "http://litellm:4000")
GOOSE_MODEL = os.environ.get("GOOSE_MODEL", "gpt-4o")
SYSTEM_PROMPT_PATH = str(Path("/app/data/system_prompt.txt"))


def validate_token(token: str) -> bool:
    """Validate authentication token (placeholder)."""
    # TODO: Implement actual token validation
    return True


def _load_system_prompt() -> str:
    """Load system prompt from file or return default."""
    prompt_path = Path(SYSTEM_PROMPT_PATH)
    if prompt_path.exists():
        return prompt_path.read_text()
    return "You are a helpful AI assistant."


@app.get("/health")
async def health():
    """Health check endpoint with multi-level diagnostics."""
    if is_killed():
        return JSONResponse({"status": "killed", "message": "Kill switch activated"}, status_code=503)
    result = await full_health_check()
    result["circuits"] = get_circuit_states()
    status_code = 200 if result.get("status") == "healthy" else 503
    return JSONResponse(result, status_code=status_code)


@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    """WebSocket endpoint for streaming logs."""
    await websocket.accept()
    process = await asyncio.create_subprocess_exec("tail", "-f", "/proc/1/fd/1", stdout=asyncio.subprocess.PIPE)
    try:
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            await websocket.send_text(line.decode())
    except Exception:
        process.terminate()
        await websocket.close()


@app.get("/capabilities", response_model=CapabilitiesResponse)
async def capabilities():
    """Return node capabilities and configuration."""
    return CapabilitiesResponse(
        node="Antigravity Node v14.1 Phoenix",
        protocols=["http", "grpc", "mcp", "a2a"],
        endpoints={
            "/health": "Health check with multi-level diagnostics",
            "/task": "Task submission endpoint",
            "/webhook": "Webhook callback endpoint",
            "/upload": "File upload endpoint",
            "/v1/chat/completions": "Chat completions endpoint",
            "/tools": "List available MCP tools",
            "/capabilities": "Node capabilities",
        },
        mcp_servers={
            "starrocks": {"status": "active", "url": "http://mcp-starrocks:8080"},
            "filesystem": {"status": "active", "url": "http://mcp-filesystem:8080"},
        },
        memory={
            "episodic": "starrocks",
            "semantic": "milvus",
            "object_store": "seaweedfs",
        },
        budget={
            "daily_limit_usd": float(os.environ.get("DAILY_BUDGET_USD", "50")),
            "proxy_url": LITELLM_URL,
        },
    )


@app.get("/tools", response_model=ToolsResponse)
async def tools():
    """List available MCP tools."""
    tools_list = list_tools()
    tool_infos = [
        ToolInfo(
            name=t["name"],
            server=t.get("server", "goose"),
            description=t.get("description", ""),
            status="active",
        )
        for t in tools_list
    ]
    return ToolsResponse(tools=tool_infos, total=len(tool_infos))


@app.get("/.well-known/agent.json")
async def agent_card():
    """Return agent card if it exists."""
    agent_json_path = Path("/app/well-known/agent.json")
    if agent_json_path.exists():
        return FileResponse(agent_json_path, media_type="application/json")
    raise HTTPException(status_code=404, detail="Agent card not found")


@app.post("/task", response_model=TaskResponse)
async def task_endpoint(
    request: TaskRequest,
    x_tenant_id: str | None = Header(None, alias="x-tenant-id"),
):
    """Submit a task for execution."""
    if not x_tenant_id:
        raise HTTPException(status_code=400, detail="x-tenant-id header is required")

    session_id = request.session_id or str(uuid.uuid4())

    # Recall past experiences for context
    history = recall_experience(goal=request.goal, tenant_id=x_tenant_id, limit=5)

    # Log task submission
    push_episodic(
        tenant_id=x_tenant_id,
        session_id=session_id,
        actor="system",
        action_type="task_submitted",
        content=f"Goal: {request.goal[:200]}",
    )

    return TaskResponse(
        status="accepted",
        session_id=session_id,
        tenant_id=x_tenant_id,
        history_count=len(history),
    )


@app.post("/handoff", response_model=HandoffResponse)
async def handoff_endpoint(request: HandoffRequest):
    """Hand off to another agent."""
    return HandoffResponse(
        status="handoff_acknowledged",
        target=request.target_agent,
    )


@app.post("/webhook", response_model=WebhookResponse)
async def webhook_endpoint(
    payload: WebhookPayload,
    x_signature: str | None = Header(None, alias="x-signature"),
):
    """Webhook callback from Argo Workflows."""
    # Validate signature if secret is configured
    if WEBHOOK_SECRET:
        if not x_signature:
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

        # Create signature from payload
        payload_str = (
            f"{payload.task_id}{payload.status}{payload.message or ''}"
        )
        expected_sig = hmac.new(
            WEBHOOK_SECRET.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(x_signature, expected_sig):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

    # Trigger reflection on failure
    if payload.status == "Failed":
        await goose_reflect(payload.task_id, payload.message or "Task failed")

    return WebhookResponse(ack=True)


@app.post("/upload", response_model=UploadResponse)
async def upload_endpoint(
    file: UploadFile = File(...),
    x_tenant_id: str | None = Header(None, alias="x-tenant-id"),
):
    """Upload a file to object storage."""
    if not x_tenant_id:
        raise HTTPException(status_code=400, detail="x-tenant-id header is required")

    # Read file content
    content = await file.read()
    file_key = f"{x_tenant_id}/{uuid.uuid4()}-{file.filename}"

    # Upload to S3
    s3_upload(file_key, content)

    # Log upload
    push_episodic(
        tenant_id=x_tenant_id,
        session_id=str(uuid.uuid4()),
        actor="user",
        action_type="file_uploaded",
        content=f"Uploaded {file.filename} ({len(content)} bytes) to {file_key}",
    )

    return UploadResponse(
        status="uploaded",
        key=file_key,
        size=len(content),
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    x_tenant_id: str | None = Header(None, alias="x-tenant-id"),
):
    """Chat completions endpoint with memory integration."""
    if not x_tenant_id:
        raise HTTPException(status_code=400, detail="x-tenant-id header is required")

    # Load system prompt
    system_prompt = _load_system_prompt()

    # Get user message for context
    user_msg = ""
    for msg in request.messages:
        if msg.role == "user":
            user_msg = msg.content
            break

    # Recall relevant memories
    if user_msg:
        recall_experience(goal=user_msg[:500], tenant_id=x_tenant_id, limit=3)

    # Forward to LiteLLM/Budget Proxy
    model = request.model or GOOSE_MODEL
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{LITELLM_URL}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            },
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        # Get JSON - handle both sync and async (for mocked tests)
        result_json = response.json()
        if hasattr(result_json, "__await__"):
            result = await result_json
        else:
            result = result_json

        # Log interaction
        assistant_msg = ""
        if "choices" in result and len(result["choices"]) > 0:
            assistant_msg = result["choices"][0].get("message", {}).get("content", "")

        push_episodic(
            tenant_id=x_tenant_id,
            session_id=str(uuid.uuid4()),
            actor="assistant",
            action_type="chat_completion",
            content=f"Q: {user_msg[:200]} A: {assistant_msg[:200]}",
        )

        return result


@app.post("/v1/inference", response_model=InferenceResponse)
async def inference_endpoint(request: InferenceRequest):
    """OVMS inference endpoint."""
    try:
        result = await run_inference(request.model_name, request.input_data)
        return InferenceResponse(
            status="success",
            model=request.model_name,
            outputs=result.get("outputs", {}),
            latency_ms=result.get("latency_ms", 0),
        )
    except Exception as e:
        return InferenceResponse(
            status="error",
            model=request.model_name,
            message=str(e),
        )


@app.get("/v1/models/ovms")
async def list_ovms_models_endpoint():
    """List models available in OVMS."""
    models = await ovms_list_models()
    return {"models": models}


@app.get("/v1/models")
async def list_models():
    """List available models (for compatibility)."""
    return {"data": [{"id": "gpt-4o"}, {"id": "tinyllama"}]}


@app.get("/admin/circuits")
async def admin_circuits():
    """Circuit breaker status for all external service calls."""
    return get_circuit_states()


@app.post("/admin/kill-switch")
async def admin_kill_switch():
    """Emergency stop — halts all orchestrator operations."""
    return trigger_kill()

