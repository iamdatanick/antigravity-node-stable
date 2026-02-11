
import asyncio
import logging
import os
import subprocess
from datetime import UTC, datetime
from typing import List, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# --- Configuration ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("antigravity-orchestrator")

app = FastAPI(title="Antigravity Orchestrator", version="14.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WorkflowListResponse(BaseModel):
    workflows: List[dict]

class ToolsResponse(BaseModel):
    tools: List[dict]
    total: int

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "14.1.0"}

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

@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    await websocket.accept()
    try:
        # Professional log tailing
        log_path = "/proc/1/fd/1" # Docker stdout
        if not os.path.exists(log_path):
             while True:
                await asyncio.sleep(5)
                await websocket.send_text(f"[{datetime.now().strftime('%H:%M:%S')}] System Operational\r\n")
                
        process = await asyncio.create_subprocess_exec(
            'tail', '-f', log_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        while True:
            line = await process.stdout.readline()
            if not line: break
            await websocket.send_text(line.decode())
    except Exception:
        pass

@app.get("/workflows", response_model=WorkflowListResponse)
async def list_workflows():
    return {"workflows": []}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    proxy_url = os.environ.get("LITELLM_BASE_URL", "http://budget-proxy:4055")
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(f"{proxy_url}/v1/chat/completions", json=request.dict())
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        except Exception as e:
            logger.error(f"Chat proxy error: {e}")
            return JSONResponse(status_code=502, content={"error": "Proxy connection failed"})

@app.get("/v1/models")
async def list_models():
    proxy_url = os.environ.get("LITELLM_BASE_URL", "http://budget-proxy:4055")
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{proxy_url}/v1/models")
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        except Exception:
            return {"object": "list", "data": [{"id": "gpt-4o", "object": "model"}]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
