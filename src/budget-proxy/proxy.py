import os
from datetime import UTC, datetime

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# Budget tracking state
_daily_spend = 0.0
_spend_date = datetime.now(UTC).date()
_hourly_spend = [0.0] * 24  # Track hourly spend

# Configuration
DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", 99999.0))
LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "http://ovms:9001/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Cost table: {model: (input_per_1k, output_per_1k)}
COST_TABLE = {
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "claude-sonnet-4-20250514": (0.003, 0.015),
    "local": (0.0, 0.0),
}


def _reset_if_new_day():
    """Reset spend counter if it's a new day."""
    global _daily_spend, _spend_date, _hourly_spend
    today = datetime.now(UTC).date()
    if today != _spend_date:
        _daily_spend = 0.0
        _spend_date = today
        _hourly_spend = [0.0] * 24


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost in USD for given token usage."""
    if model.startswith("local"):
        return 0.0

    # Get pricing, fall back to gpt-4o-mini if unknown
    input_cost, output_cost = COST_TABLE.get(model, COST_TABLE["gpt-4o-mini"])

    total_cost = (prompt_tokens / 1000.0) * input_cost + (completion_tokens / 1000.0) * output_cost
    return total_cost


async def _route_model(model: str):
    """Route model request to appropriate provider.

    Returns: (base_url, headers, model_name)
    """
    # Local models
    if model.startswith("local/"):
        return (LOCAL_LLM_URL, {}, model.replace("local/", ""))

    # Anthropic Claude models
    if "claude" in model.lower():
        if not ANTHROPIC_API_KEY:
            return (LOCAL_LLM_URL, {}, model)
        return (
            "https://api.anthropic.com",
            {
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            model,
        )

    # OpenAI models
    if OPENAI_API_KEY:
        return (
            "https://api.openai.com/v1",
            {"Authorization": f"Bearer {OPENAI_API_KEY}"},
            model,
        )

    # Fall back to local if no API key
    return (LOCAL_LLM_URL, {}, model)


@app.get("/health")
async def health():
    """Health endpoint with budget info."""
    _reset_if_new_day()
    return {
        "status": "ok",
        "daily_spend_usd": _daily_spend,
        "daily_budget_usd": DAILY_BUDGET_USD,
        "remaining_usd": DAILY_BUDGET_USD - _daily_spend,
    }


@app.get("/history")
async def budget_history():
    """Budget history with hourly breakdown."""
    _reset_if_new_day()
    return {
        "current_spend": _daily_spend,
        "max_daily": DAILY_BUDGET_USD,
        "currency": "USD",
        "hourly_spend": _hourly_spend,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Chat completions proxy with budget enforcement."""
    global _daily_spend

    _reset_if_new_day()

    # Check budget
    if _daily_spend >= DAILY_BUDGET_USD:
        raise HTTPException(
            status_code=429,
            detail=f"Daily budget of ${DAILY_BUDGET_USD} exceeded. Current spend: ${_daily_spend:.2f}",
        )

    body = await request.json()
    model = body.get("model", "gpt-4o")

    # Route to appropriate provider
    base_url, headers, routed_model = await _route_model(model)

    # Make request
    async with httpx.AsyncClient() as client:
        try:
            url = f"{base_url.rstrip('/')}/v1/chat/completions"
            resp = await client.post(
                url,
                json={**body, "model": routed_model},
                headers=headers,
                timeout=60.0,
            )

            # Track spend if successful
            if resp.status_code == 200:
                resp_data = resp.json()
                usage = resp_data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                cost = _estimate_cost(model, prompt_tokens, completion_tokens)
                _daily_spend += cost

                # Track hourly spend
                hour = datetime.now(UTC).hour
                _hourly_spend[hour] += cost

            return JSONResponse(status_code=resp.status_code, content=resp.json())
        except httpx.HTTPStatusError as e:
            return JSONResponse(status_code=e.response.status_code, content=e.response.json())
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")


@app.get("/v1/models")
async def list_models():
    """List available models in OpenAI format."""
    models = []
    for model_name in COST_TABLE:
        if model_name != "local":
            models.append(
                {
                    "id": model_name,
                    "object": "model",
                    "owned_by": "budget-proxy",
                }
            )

    return {
        "object": "list",
        "data": models,
    }
