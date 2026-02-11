### C:\Users\NickV\OneDrive\Desktop\Antigravity-Node/src/budget-proxy/proxy.py
```python
1: import asyncio
2: import logging
3: import os
4: import time
5: import httpx
6: from fastapi import FastAPI, HTTPException, Request
7: from fastapi.responses import JSONResponse
8: 
9: logging.basicConfig(level=logging.INFO)
10: logger = logging.getLogger("budget-proxy")
11: 
12: app = FastAPI(title="Budget Proxy", version="14.1.0")
13: 
14: # Config
15: OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
16: GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
17: LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "http://ovms:9001/v1")
18: LOCAL_DEFAULT_MODEL = os.environ.get("LOCAL_DEFAULT_MODEL", "tinyllama")
19: MODEL_MAP = {"gpt-4o": "tinyllama", "gpt-3.5-turbo": "tinyllama"}
DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", "999999")) # Set high as per user instruction
20: 
21: _daily_spend = 0.0
22: _spend_lock = asyncio.Lock()
23: 
24: async def _route_model(model: str) -> tuple[str, dict, str]:
25:     # Vertex AI / Google Routing
26:     if model.startswith("vertex/") or model.startswith("gemini-"):
27:         effective_model = model.replace("vertex/", "")
28:         url = f"https://generativelanguage.googleapis.com/v1beta/models/{effective_model}:streamGenerateContent?key={GOOGLE_API_KEY}"
29:         # Note: Google uses a different format, but for a simple proxy we might need a wrapper 
30:         # For now, we route it. If it's standard OpenAI-compatible Google endpoint:
31:         return ("https://generativelanguage.googleapis.com/v1beta/openai", {"Authorization": f"Bearer {GOOGLE_API_KEY}"}, effective_model)
32: 
33:     if model.startswith("local/"):
34:         return (LOCAL_LLM_URL, {}, model.removeprefix("local/"))
35:     
36:     if OPENAI_API_KEY:
37:         return ("https://api.openai.com/v1", {"Authorization": f"Bearer {OPENAI_API_KEY}"}, model)
38:     
39:     # Map production names to local fallback for test stability
40:     mapping = {
41:         "gpt-4o": LOCAL_DEFAULT_MODEL,
42:         "gpt-4o-mini": LOCAL_DEFAULT_MODEL,
43:         "claude-3-5-sonnet-20240620": LOCAL_DEFAULT_MODEL
44:     }
45:     target = mapping.get(model, LOCAL_DEFAULT_MODEL)
46:     logger.info(f"No API key. Mapping {model} -> {target}")
47:     return (LOCAL_LLM_URL, {}, target)
48: 
49: @app.get("/health")
50: async def health():
51:     return {"status": "ok", "daily_spend": round(_daily_spend, 4)}
52: 
53: @app.post("/v1/chat/completions")
54: async def chat_completions(request: Request):
55:     try:
56:         body = await request.json()
57:     except Exception:
58:         raise HTTPException(status_code=400, detail="Invalid JSON body")
59:         
60:     model = body.get("model", "gpt-4o")
61:     base_url, headers, effective_model = await _route_model(model)
62:     body["model"] = effective_model
63:     
64:     async with httpx.AsyncClient(timeout=120.0) as client:
65:         try:
66:             resp = await client.post(
67:                 f"{base_url}/chat/completions",
68:                 json=body,
69:                 headers={**headers, "Content-Type": "application/json"},
70:             )
71:             if resp.status_code != 200:
72:                 logger.error(f"Upstream {base_url} returned {resp.status_code}: {resp.text}")
73:                 # Try to return the upstream error
74:                 try:
75:                     return JSONResponse(status_code=resp.status_code, content=resp.json())
76:                 except:
77:                     return JSONResponse(status_code=resp.status_code, content={"error": resp.text})
78:             return JSONResponse(content=resp.json())
79:         except Exception as e:
80:             logger.error(f"Proxy error: {e}")
81:             raise HTTPException(status_code=500, detail=str(e))
82: 
83: @app.post("/v1/embeddings")
84: async def create_embeddings(request: Request):
85:     if not OPENAI_API_KEY:
86:         raise HTTPException(status_code=503, detail="No embedding API key available")
87:     try:
88:         body = await request.json()
89:     except Exception:
90:         raise HTTPException(status_code=400, detail="Invalid JSON body")
91: 
92:     async with httpx.AsyncClient(timeout=30.0) as client:
93:         try:
94:             resp = await client.post(
95:                 "https://api.openai.com/v1/embeddings",
96:                 json=body,
97:                 headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
98:             )
99:             return JSONResponse(status_code=resp.status_code, content=resp.json())
100:         except Exception as e:
101:             logger.error(f"Embedding error: {e}")
102:             raise HTTPException(status_code=500, detail=str(e))
103: 
104: @app.get("/v1/models")
105: async def list_models():
106:     # Return a rich list for the UI
107:     return {"object": "list", "data": [
108:         {"id": "gpt-4o", "object": "model", "owned_by": "openai"},
109:         {"id": "gpt-4o-mini", "object": "model", "owned_by": "openai"},
110:         {"id": "tinyllama", "object": "model", "owned_by": "local"},
111:         {"id": "vertex/gemini-1.5-pro", "object": "model", "owned_by": "vertex"},
112:         {"id": "vertex/gemini-1.5-flash", "object": "model", "owned_by": "vertex"},
113:         {"id": "vertex/medlm-large", "object": "model", "owned_by": "vertex"},
114:     ]}
115: 
116: if __name__ == "__main__":
117:     import uvicorn
118:     uvicorn.run(app, host="0.0.0.0", port=4055)
```
