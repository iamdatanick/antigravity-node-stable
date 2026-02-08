"""Budget Proxy - Lightweight LLM API router with cost controls.

Routes requests to OpenAI, Anthropic, or local OVMS endpoints
with daily budget enforcement and request logging.
Apache-2.0 licensed, sovereign replacement for LiteLLM.
"""

import os
import json
import time
import logging
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("budget-proxy")

app = FastAPI(title="Budget Proxy", version="1.0.0")

# Config from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "http://ovms:8000/v1")
DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", "50"))

# In-memory cost tracking (resets daily)
_daily_spend = 0.0
_spend_date = datetime.now(timezone.utc).date()

# Approximate cost per 1K tokens (input/output)
COST_TABLE = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5-20251001": {"input": 0.001, "output": 0.005},
    "local": {"input": 0.0, "output": 0.0},
}


def _reset_if_new_day():
    global _daily_spend, _spend_date
    today = datetime.now(timezone.utc).date()
    if today != _spend_date:
        logger.info(f"New day: resetting spend from ${_daily_spend:.4f}")
        _daily_spend = 0.0
        _spend_date = today


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    costs = COST_TABLE.get(model, COST_TABLE.get("gpt-4o-mini"))
    return (prompt_tokens / 1000 * costs["input"]) + (completion_tokens / 1000 * costs["output"])


def _route_model(model: str) -> tuple[str, dict, str]:
    """Returns (base_url, headers, effective_model)."""
    if model.startswith("claude"):
        return (
            "https://api.anthropic.com/v1",
            {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"},
            model,
        )
    if model.startswith("local/") or not OPENAI_API_KEY:
        return (LOCAL_LLM_URL, {}, model.removeprefix("local/"))
    return (
        "https://api.openai.com/v1",
        {"Authorization": f"Bearer {OPENAI_API_KEY}"},
        model,
    )


@app.get("/health")
async def health():
    _reset_if_new_day()
    return {
        "status": "ok",
        "daily_spend_usd": round(_daily_spend, 4),
        "daily_budget_usd": DAILY_BUDGET_USD,
        "remaining_usd": round(DAILY_BUDGET_USD - _daily_spend, 4),
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global _daily_spend
    _reset_if_new_day()

    if _daily_spend >= DAILY_BUDGET_USD:
        raise HTTPException(
            status_code=429,
            detail=f"Daily budget exhausted: ${_daily_spend:.2f} / ${DAILY_BUDGET_USD:.2f}",
        )

    body = await request.json()
    model = body.get("model", "gpt-4o-mini")
    base_url, headers, effective_model = _route_model(model)
    body["model"] = effective_model

    start = time.monotonic()
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            json=body,
            headers={**headers, "Content-Type": "application/json"},
        )

    elapsed = time.monotonic() - start

    if resp.status_code != 200:
        return JSONResponse(status_code=resp.status_code, content=resp.json())

    result = resp.json()
    usage = result.get("usage", {})
    cost = _estimate_cost(model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
    _daily_spend += cost

    logger.info(
        f"model={model} tokens={usage.get('total_tokens', 0)} "
        f"cost=${cost:.4f} total=${_daily_spend:.4f} elapsed={elapsed:.2f}s"
    )

    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000)
