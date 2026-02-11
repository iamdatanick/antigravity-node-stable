
import asyncio, os, subprocess
from fastapi import FastAPI, WebSocket
from workflows.health import full_health_check

app = FastAPI()

@app.get("/health")
async def health():
    return await full_health_check()

@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    await websocket.accept()
    process = await asyncio.create_subprocess_exec("tail", "-f", "/proc/1/fd/1", stdout=subprocess.PIPE)
    try:
        while True:
            line = await process.stdout.readline()
            if not line: break
            await websocket.send_text(line.decode())
    except:
        process.terminate()
        await websocket.close()

@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": "gpt-4o"}, {"id": "tinyllama"}]}
