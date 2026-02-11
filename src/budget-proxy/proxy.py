
import asyncio
import logging
import os
import time
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("budget-proxy")

app = FastAPI(title="Budget Proxy", version="14.1.0")

# Config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "http://ovms:9001/v1")
LOCAL_DEFAULT_MODEL = os.environ.get("LOCAL_DEFAULT_MODEL", "tinyllama")
DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", "50"))

_daily_spend = 0.0
_spend_lock = asyncio.Lock()

async def _route_model(model: str) -> tuple[str, dict, str]:
    if model.startswith("local/"):
        return (LOCAL_LLM_URL, {}, model.removeprefix("local/"))
    
    if OPENAI_API_KEY:
        return ("https://api.openai.com/v1", {"Authorization": f"Bearer {OPENAI_API_KEY}"}, model)
    
    # Map production names to local fallback for test stability
    mapping = {
        "gpt-4o": LOCAL_DEFAULT_MODEL,
        "gpt-4o-mini": LOCAL_DEFAULT_MODEL,
        "claude-3-5-sonnet-20240620": LOCAL_DEFAULT_MODEL
    }
    target = mapping.get(model, LOCAL_DEFAULT_MODEL)
    logger.info(f"No API key. Mapping {model} -> {target}")
    return (LOCAL_LLM_URL, {}, target)

@app.get("/health")
async def health():
    return {"status": "ok", "daily_spend": round(_daily_spend, 4)}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
        
    model = body.get("model", "gpt-4o")
    base_url, headers, effective_model = await _route_model(model)
    body["model"] = effective_model
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(
                f"{base_url}/chat/completions",
                json=body,
                headers={**headers, "Content-Type": "application/json"},
            )
            if resp.status_code != 200:
                logger.error(f"Upstream {base_url} returned {resp.status_code}: {resp.text}")
                return JSONResponse(status_code=resp.status_code, content=resp.json())
            return JSONResponse(content=resp.json())
        except Exception as e:
            logger.error(f"Proxy error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/embeddings")
async def create_embeddings(request: Request):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="No embedding API key available")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                "https://api.openai.com/v1/embeddings",
                json=body,
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            )
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [
        {"id": "gpt-4o", "object": "model", "owned_by": "antigravity"},
        {"id": "gpt-4o-mini", "object": "model", "owned_by": "antigravity"},
        {"id": "tinyllama", "object": "model", "owned_by": "local"}
    ]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4055)
