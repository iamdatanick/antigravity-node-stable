"""Budget Proxy - Lightweight LLM API router with cost controls.

Routes requests to OpenAI, Anthropic, or local OVMS endpoints
with daily budget enforcement and request logging.
Apache-2.0 licensed, sovereign replacement for LiteLLM.
"""

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

app = FastAPI(title="Budget Proxy", version="1.0.0")

# Config from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "http://ovms:8000/v1")
DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", "50"))
OPENBAO_ADDR = os.environ.get("OPENBAO_ADDR", "http://openbao:8200")
OPENBAO_TOKEN = os.environ.get("OPENBAO_TOKEN", "dev-only-token")

# In-memory cost tracking (resets daily)
_daily_spend = 0.0
_spend_date = datetime.now(UTC).date()
_spend_lock = asyncio.Lock()

# Hourly spend tracking (24-hour window, resets with day)
_hourly_spend: list[float] = [0.0] * 24
_current_hour: int = datetime.now(UTC).hour

# Cached API keys from OpenBao (TTL-based)
_key_cache: dict[str, str] = {}
_key_cache_time: float = 0.0
_KEY_CACHE_TTL = 60.0  # Re-fetch from vault every 60s

# Approximate cost per 1K tokens (input/output)
COST_TABLE = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5-20251001": {"input": 0.001, "output": 0.005},
    "local": {"input": 0.0, "output": 0.0},
}


async def _fetch_vault_key(provider: str) -> str:
    """Fetch an API key from OpenBao KV v2."""
    global _key_cache, _key_cache_time
    now = time.monotonic()
    if now - _key_cache_time < _KEY_CACHE_TTL and provider in _key_cache:
        return _key_cache[provider]

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{OPENBAO_ADDR}/v1/secret/data/antigravity/api-keys/{provider}",
                headers={"X-Vault-Token": OPENBAO_TOKEN},
            )
            if resp.status_code == 200:
                data = resp.json()
                key = data.get("data", {}).get("data", {}).get("key", "")
                if key:
                    _key_cache[provider] = key
                    _key_cache_time = now
                    logger.info(f"Loaded {provider} API key from OpenBao")
                    return key
    except Exception as e:
        logger.debug(f"OpenBao key fetch for {provider} failed: {e}")
    return ""


def _reset_if_new_day():
    global _daily_spend, _spend_date, _hourly_spend, _current_hour
    today = datetime.now(UTC).date()
    if today != _spend_date:
        logger.info(f"New day: resetting spend from ${_daily_spend:.4f}")
        _daily_spend = 0.0
        _hourly_spend = [0.0] * 24
        _current_hour = datetime.now(UTC).hour
        _spend_date = today


def _update_hourly(cost: float):
    """Track spend per hour for the budget history chart."""
    global _current_hour
    now_hour = datetime.now(UTC).hour
    if now_hour != _current_hour:
        # Zero out skipped hours
        h = (_current_hour + 1) % 24
        while h != (now_hour + 1) % 24:
            _hourly_spend[h] = 0.0
            h = (h + 1) % 24
        _current_hour = now_hour
    _hourly_spend[now_hour] += cost


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    costs = COST_TABLE.get(model, COST_TABLE.get("gpt-4o-mini"))
    return (prompt_tokens / 1000 * costs["input"]) + (completion_tokens / 1000 * costs["output"])


async def _route_model(model: str) -> tuple[str, dict, str]:
    """Returns (base_url, headers, effective_model)."""
    if model.startswith("claude"):
        key = ANTHROPIC_API_KEY or await _fetch_vault_key("anthropic")
        if key:
            return (
                "https://api.anthropic.com/v1",
                {"x-api-key": key, "anthropic-version": "2023-06-01"},
                model,
            )
        return (LOCAL_LLM_URL, {}, model)

    if model.startswith("local/"):
        return (LOCAL_LLM_URL, {}, model.removeprefix("local/"))

    # OpenAI and other models
    key = OPENAI_API_KEY or await _fetch_vault_key("openai")
    if key:
        return (
            "https://api.openai.com/v1",
            {"Authorization": f"Bearer {key}"},
            model,
        )
    return (LOCAL_LLM_URL, {}, model)


@app.get("/health")
async def health():
    _reset_if_new_day()
    return {
        "status": "ok",
        "daily_spend_usd": round(_daily_spend, 4),
        "daily_budget_usd": DAILY_BUDGET_USD,
        "remaining_usd": round(DAILY_BUDGET_USD - _daily_spend, 4),
    }


@app.get("/budget/history")
async def budget_history():
    """Return hourly spend breakdown for the UI budget chart."""
    _reset_if_new_day()
    return {
        "current_spend": round(_daily_spend, 4),
        "max_daily": DAILY_BUDGET_USD,
        "currency": "USD",
        "hourly_spend": [round(h, 6) for h in _hourly_spend],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global _daily_spend

    async with _spend_lock:
        _reset_if_new_day()
        if _daily_spend >= DAILY_BUDGET_USD:
            raise HTTPException(
                status_code=429,
                detail=f"Daily budget exhausted: ${_daily_spend:.2f} / ${DAILY_BUDGET_USD:.2f}",
            )

    body = await request.json()
    model = body.get("model", "gpt-4o-mini")
    base_url, headers, effective_model = await _route_model(model)
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

    async with _spend_lock:
        _daily_spend += cost
        _update_hourly(cost)
        current_spend = _daily_spend

    logger.info(
        f"model={model} tokens={usage.get('total_tokens', 0)} "
        f"cost=${cost:.4f} total=${current_spend:.4f} elapsed={elapsed:.2f}s"
    )

    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "4055")))
