import asyncio
import logging
import os
import subprocess
from datetime import datetime, UTC
from typing import List, Optional
import httpx
from fastapi import FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from workflows.health import full_health_check
from workflows.models import ChatRequest, ChatResponse, ToolsResponse, WorkflowListResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("antigravity")

app = FastAPI(title="Antigravity Node", version="14.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return await full_health_check()

@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    await websocket.accept()
    # Professional fix: tail the actual container logs
    process = await asyncio.create_subprocess_exec(
        "tail", "-f", "/proc/1/fd/1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    try:
        while True:
            line = await process.stdout.readline()
            if not line: break
            await websocket.send_text(line.decode())
    except Exception:
        process.terminate()
        await websocket.close()

@app.get("/tools", response_model=ToolsResponse)
async def list_tools():
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

@app.get("/workflows", response_model=WorkflowListResponse)
async def list_workflows(request: Request):
    return {"workflows": []}

@app.get("/v1/models")
async def list_models():
    ovms_url = os.environ.get("OVMS_REST_URL", "http://ovms:9001")
    models = [{"id": "gpt-4o", "object": "model", "owned_by": "openai"}]
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{ovms_url}/v1/models")
            if resp.status_code == 200:
                for m in resp.json().get("data", []):
                    models.append({"id": f"local/{m.get('id')}", "object": "model", "owned_by": "ovms"})
    except:
        pass
    return {"data": models}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
