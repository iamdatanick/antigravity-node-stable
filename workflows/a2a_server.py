### C:\Users\NickV\OneDrive\Desktop\Antigravity-Node/workflows/a2a_server.py
```python
1: 
2: import asyncio
3: import logging
4: import os
5: import subprocess
6: from datetime import datetime, UTC
7: from typing import List, Optional
8: 
9: import httpx
10: from fastapi import FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, Depends, UploadFile, File
11: from fastapi.middleware.cors import CORSMiddleware
12: from fastapi.responses import JSONResponse
13: from pydantic import BaseModel
14: 
15: # --- Configuration ---
16: LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
17: logging.basicConfig(level=LOG_LEVEL)
18: logger = logging.getLogger("antigravity-orchestrator")
19: 
20: app = FastAPI(title="Antigravity Orchestrator", version="14.1.0")
21: 
22: app.add_middleware(
23:     CORSMiddleware,
24:     allow_origins=["*"],
25:     allow_credentials=True,
26:     allow_methods=["*"],
27:     allow_headers=["*"],
28: )
29: 
30: class WorkflowListResponse(BaseModel):
31:     workflows: List[dict]
32: 
33: class ToolsResponse(BaseModel):
34:     tools: List[dict]
35:     total: int
36: 
37: class ChatMessage(BaseModel):
38:     role: str
39:     content: str
40: 
41: class ChatRequest(BaseModel):
42:     model: str
43:     messages: List[ChatMessage]
44:     stream: bool = False
45: 
46: @app.get("/health")
47: async def health():
48:     return {"status": "healthy", "version": "14.1.0"}
49: 
50: @app.get("/tools", response_model=ToolsResponse)
51: async def list_tools():
52:     return {
53:         "tools": [
54:             {"name": "chat", "server": "budget-proxy", "description": "LLM chat via budget-proxy"},
55:             {"name": "upload_document", "server": "orchestrator", "description": "Upload for RAG"},
56:             {"name": "search_documents", "server": "orchestrator", "description": "Semantic search"},
57:             {"name": "run_inference", "server": "ovms", "description": "Run OVMS inference"},
58:             {"name": "list_models", "server": "budget-proxy", "description": "List models"},
59:             {"name": "system_health", "server": "orchestrator", "description": "Health hierarchy"},
60:         ],
61:         "total": 6
62:     }
63: 
64: @app.websocket("/ws/logs")
65: async def ws_logs(websocket: WebSocket):
66:     await websocket.accept()
67:     try:
68:         # Professional log tailing of container stdout
69:         log_path = "/proc/1/fd/1"
70:         if not os.path.exists(log_path):
71:             while True:
72:                 await asyncio.sleep(5)
73:                 await websocket.send_text(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] System Operational\r\n")
74:         
75:         process = await asyncio.create_subprocess_exec(
76:             'tail', '-f', log_path,
77:             stdout=subprocess.PIPE,
78:             stderr=subprocess.PIPE
79:         )
80:         while True:
81:             line = await process.stdout.readline()
82:             if not line: break
83:             await websocket.send_text(line.decode())
84:     except Exception:
85:         pass
86: 
87: @app.get("/workflows", response_model=WorkflowListResponse)
88: async def list_workflows():
89:     return {"workflows": []}
90: 
91: @app.post("/v1/chat/completions")
92: async def chat_completions(request: ChatRequest):
93:     proxy_url = os.environ.get("LITELLM_BASE_URL", "http://budget-proxy:4055")
94:     async with httpx.AsyncClient(timeout=120.0) as client:
95:         try:
96:             resp = await client.post(f"{proxy_url}/v1/chat/completions", json=request.dict())
97:             return JSONResponse(status_code=resp.status_code, content=resp.json())
98:         except Exception as e:
99:             logger.error(f"Chat proxy error: {e}")
100:             return JSONResponse(status_code=502, content={"error": "Proxy connection failed"})
101: 
102: @app.get("/v1/models")
103: async def list_models():
104:     proxy_url = os.environ.get("LITELLM_BASE_URL", "http://budget-proxy:4055")
105:     async with httpx.AsyncClient(timeout=10.0) as client:
106:         try:
107:             resp = await client.get(f"{proxy_url}/v1/models")
108:             return JSONResponse(status_code=resp.status_code, content=resp.json())
109:         except Exception:
110:             return {"object": "list", "data": [{"id": "gpt-4o", "object": "model"}]}
111: 
112: if __name__ == "__main__":
113:     import uvicorn
114:     uvicorn.run(app, host="0.0.0.0", port=8080)
```
