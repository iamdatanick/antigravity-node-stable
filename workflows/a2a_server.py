import asyncio, os, subprocess
from fastapi import FastAPI, WebSocket
from workflows.health import full_health_check
from workflows.resilience import get_circuit_states, is_killed, trigger_kill

app = FastAPI(title="Antigravity Node", version="13.1")


@app.get("/health")
async def health():
    if is_killed():
        return {"status": "killed", "message": "Kill switch activated"}
    result = await full_health_check()
    result["circuits"] = get_circuit_states()
    return result


@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    await websocket.accept()
    process = await asyncio.create_subprocess_exec("tail", "-f", "/proc/1/fd/1", stdout=subprocess.PIPE)
    try:
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            await websocket.send_text(line.decode())
    except Exception:
        process.terminate()
        await websocket.close()


@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": "gpt-4o"}, {"id": "tinyllama"}]}


@app.get("/admin/circuits")
async def admin_circuits():
    """Circuit breaker status for all external service calls."""
    return get_circuit_states()


@app.post("/admin/kill-switch")
async def admin_kill_switch():
    """Emergency stop — halts all orchestrator operations."""
    return trigger_kill()
