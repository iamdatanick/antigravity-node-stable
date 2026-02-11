
import asyncio
import logging
import os
import time
from datetime import UTC, datetime
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("budget-proxy")

app = FastAPI(title="Budget Proxy", version="14.1.0")

# Config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "http://ovms:9001/v1")
DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", "50"))
LOCAL_DEFAULT_MODEL = os.environ.get("LOCAL_DEFAULT_MODEL", "tinyllama")

_daily_spend = 0.0
_spend_lock = asyncio.Lock()

COST_TABLE = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "tinyllama": {"input": 0.0, "output": 0.0},
}

async def _route_model(model: str) -> tuple[str, dict, str]:
    if model.startswith("local/"):
        return (LOCAL_LLM_URL, {}, model.removeprefix("local/"))
    
    if OPENAI_API_KEY:
        return ("https://api.openai.com/v1", {"Authorization": f"Bearer {OPENAI_API_KEY}"}, model)
    
    # Fallback to local
    logger.info(f"No API key. Mapping {model} to local default: {LOCAL_DEFAULT_MODEL}")
    return (LOCAL_LLM_URL, {}, LOCAL_DEFAULT_MODEL)

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
                logger.error(f"Upstream returned {resp.status_code}: {resp.text}")
                return JSONResponse(status_code=resp.status_code, content=resp.json())
            return JSONResponse(content=resp.json())
        except Exception as e:
            logger.error(f"Proxy error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/embeddings")
async def create_embeddings(request: Request):
    """Route embedding requests to OpenAI - budget-tracked."""
    global _daily_spend
    
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
        
    key = OPENAI_API_KEY
    if not key:
        raise HTTPException(status_code=503, detail="No embedding API key available")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                "https://api.openai.com/v1/embeddings",
                json=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
            if resp.status_code != 200:
                return JSONResponse(status_code=resp.status_code, content=resp.json())
            
            result = resp.json()
            usage = result.get("usage", {})
            cost = usage.get("total_tokens", 0) / 1_000_000 * 0.02
            
            async with _spend_lock:
                _daily_spend += cost
                
            return JSONResponse(content=result)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [
        {"id": "gpt-4o", "object": "model", "owned_by": "antigravity"},
        {"id": "gpt-4o-mini", "object": "model", "owned_by": "antigravity"},
        {"id": "text-embedding-3-small", "object": "model", "owned_by": "antigravity"}
    ]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4055)
