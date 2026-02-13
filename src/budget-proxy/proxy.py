"""Budget Proxy — LLM API router with cost controls and multi-provider routing."""

import os
from datetime import UTC, datetime

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# --- Configuration ---
DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", 50.0))
LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "http://ollama:11434/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# --- In-memory budget state ---
_daily_spend = 0.0
_spend_date = datetime.now(UTC).date()
_hourly_spend = [0.0] * 24

# --- Cost table (USD per 1K tokens) ---
COST_TABLE = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "local": {"input": 0.0, "output": 0.0},
}


def _reset_if_new_day():
    """Reset spend counters at midnight UTC."""
    global _daily_spend, _spend_date, _hourly_spend
    today = datetime.now(UTC).date()
    if _spend_date != today:
        _daily_spend = 0.0
        _spend_date = today
        _hourly_spend = [0.0] * 24


def _estimate_cost(model: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> float:
    """Estimate cost in USD based on model and token counts."""
    prices = COST_TABLE.get(model, COST_TABLE.get("gpt-4o-mini"))
    return (prompt_tokens / 1000.0 * prices["input"]) + (completion_tokens / 1000.0 * prices["output"])


async def _route_model(model: str) -> tuple[str, dict, str]:
    """Route model to the correct provider. Returns (base_url, headers, model_name)."""
    if model.startswith("local/"):
        return LOCAL_LLM_URL, {}, model.removeprefix("local/")

    if "claude" in model and ANTHROPIC_API_KEY:
        return (
            "https://api.anthropic.com/v1",
            {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"},
            model,
        )

    if OPENAI_API_KEY:
        return (
            "https://api.openai.com/v1",
            {"Authorization": f"Bearer {OPENAI_API_KEY}"},
            model,
        )

    # Fallback to local
    return LOCAL_LLM_URL, {}, model


# --- Endpoints ---


@app.get("/health")
async def health():
    _reset_if_new_day()
    return {
        "status": "ok",
        "daily_spend_usd": _daily_spend,
        "daily_budget_usd": DAILY_BUDGET_USD,
        "remaining_usd": DAILY_BUDGET_USD - _daily_spend,
    }


@app.get("/history")
async def budget_history():
    _reset_if_new_day()
    return {
        "current_spend": _daily_spend,
        "max_daily": DAILY_BUDGET_USD,
        "currency": "USD",
        "hourly_spend": _hourly_spend,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global _daily_spend, _hourly_spend

    _reset_if_new_day()

    if _daily_spend >= DAILY_BUDGET_USD:
        return JSONResponse(
            {"error": {"message": "Daily budget exceeded", "type": "budget_exceeded"}},
            status_code=429,
        )

    body = await request.json()
    model = body.get("model", "gpt-4o")

    base_url, headers, routed_model = await _route_model(model)

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            json={**body, "model": routed_model},
            headers=headers,
            timeout=60.0,
        )
        result = resp.json()

    if resp.status_code == 200:
        usage = result.get("usage", {})
        cost = _estimate_cost(
            model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
        _daily_spend += cost
        hour = datetime.now(UTC).hour
        _hourly_spend[hour] += cost

    return JSONResponse(status_code=resp.status_code, content=result)


@app.get("/v1/models")
async def list_models():
    models = []
    for name in COST_TABLE:
        if name == "local":
            continue
        models.append({"id": name, "object": "model", "owned_by": "provider"})
    models.append({"id": "tinyllama", "object": "model", "owned_by": "local"})
    return {"object": "list", "data": models}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000)
