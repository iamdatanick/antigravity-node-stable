"""Tests for src/budget-proxy/proxy.py — LLM API router with cost controls.

Covers health endpoint, budget enforcement, routing logic,
cost estimation, and daily reset.
"""

import os

# ---------------------------------------------------------------------------
# Import helpers — add budget-proxy to path
# ---------------------------------------------------------------------------
import sys
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add the budget-proxy source directory so we can import proxy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "budget-proxy"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_proxy_state():
    """Reset in-memory budget state before every test."""
    import proxy

    proxy._daily_spend = 0.0
    proxy._spend_date = datetime.now(UTC).date()
    yield


@pytest.fixture
def client():
    """HTTPX AsyncClient backed by the FastAPI app (no network)."""
    import httpx
    from proxy import app

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )


# ---------------------------------------------------------------------------
# Tests for /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /health."""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client):
        """Health endpoint returns status ok with budget info."""
        resp = await client.get("/health")
        assert resp.status_code == 200

        body = resp.json()
        assert body["status"] == "ok"
        assert "daily_spend_usd" in body
        assert "daily_budget_usd" in body
        assert "remaining_usd" in body

    @pytest.mark.asyncio
    async def test_health_shows_remaining_budget(self, client):
        """Remaining budget equals total minus spend."""
        import proxy

        proxy._daily_spend = 12.5
        proxy.DAILY_BUDGET_USD = 50.0

        resp = await client.get("/health")
        body = resp.json()

        assert body["daily_spend_usd"] == 12.5
        assert body["remaining_usd"] == 37.5


# ---------------------------------------------------------------------------
# Tests for /budget/history
# ---------------------------------------------------------------------------


class TestBudgetHistory:
    """Tests for GET /budget/history hourly spend data."""

    @pytest.mark.asyncio
    async def test_budget_history_returns_200(self, client):
        """Budget history endpoint returns 200."""
        resp = await client.get("/history")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_budget_history_has_required_fields(self, client):
        """Budget history returns current_spend, max_daily, currency, hourly_spend."""
        resp = await client.get("/history")
        data = resp.json()
        assert "current_spend" in data
        assert "max_daily" in data
        assert "currency" in data
        assert "hourly_spend" in data
        assert isinstance(data["hourly_spend"], list)
        assert len(data["hourly_spend"]) == 24

    @pytest.mark.asyncio
    async def test_budget_history_spend_matches_health(self, client):
        """Budget history current_spend matches /health daily_spend_usd."""
        import proxy

        proxy._daily_spend = 7.5
        history = (await client.get("/history")).json()
        health = (await client.get("/health")).json()
        assert history["current_spend"] == health["daily_spend_usd"]
        assert history["max_daily"] == health["daily_budget_usd"]


# ---------------------------------------------------------------------------
# Tests for budget enforcement
# ---------------------------------------------------------------------------


class TestBudgetEnforcement:
    """Tests for daily budget limits on /v1/chat/completions."""

    @pytest.mark.asyncio
    async def test_budget_exceeded_returns_429(self, client):
        """When daily spend >= budget, requests are rejected with 429."""
        import proxy

        proxy.DAILY_BUDGET_USD = 10.0
        proxy._daily_spend = 10.0  # exactly at limit

        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]},
        )

        assert resp.status_code == 429

    @pytest.mark.asyncio
    async def test_within_budget_succeeds(self, client):
        """When spend is below budget, request is forwarded to upstream."""
        import proxy

        proxy.DAILY_BUDGET_USD = 50.0
        proxy._daily_spend = 5.0

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        with patch("proxy.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client

            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "choices" in body

    @pytest.mark.asyncio
    async def test_spend_accumulates(self, client):
        """After a successful request, _daily_spend increases."""
        import proxy

        proxy.DAILY_BUDGET_USD = 100.0
        proxy._daily_spend = 0.0

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Answer"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
        }

        with patch("proxy.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client

            await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            )

        # Spend should have increased
        assert proxy._daily_spend > 0.0


# ---------------------------------------------------------------------------
# Tests for daily reset
# ---------------------------------------------------------------------------


class TestDailyReset:
    """Tests for _reset_if_new_day budget reset logic."""

    def test_reset_on_new_day(self):
        """Spend resets to 0 when the date changes."""
        from datetime import timedelta

        import proxy

        proxy._daily_spend = 42.0
        proxy._spend_date = datetime.now(UTC).date() - timedelta(days=1)

        proxy._reset_if_new_day()

        assert proxy._daily_spend == 0.0
        assert proxy._spend_date == datetime.now(UTC).date()

    def test_no_reset_same_day(self):
        """Spend does NOT reset on the same day."""
        import proxy

        proxy._daily_spend = 15.0
        proxy._spend_date = datetime.now(UTC).date()

        proxy._reset_if_new_day()

        assert proxy._daily_spend == 15.0


# ---------------------------------------------------------------------------
# Tests for model routing
# ---------------------------------------------------------------------------


class TestRouteModel:
    """Tests for _route_model provider selection."""

    @pytest.mark.asyncio
    async def test_route_openai(self):
        """OpenAI models route to api.openai.com."""
        import proxy

        proxy.OPENAI_API_KEY = "sk-test"

        base, headers, model = await proxy._route_model("gpt-4o")

        assert "openai.com" in base
        assert "Authorization" in headers
        assert model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_route_anthropic(self):
        """Claude models route to api.anthropic.com."""
        import proxy

        proxy.ANTHROPIC_API_KEY = "sk-ant-test"

        base, headers, model = await proxy._route_model("claude-sonnet-4-20250514")

        assert "anthropic.com" in base
        assert "x-api-key" in headers
        assert "anthropic-version" in headers
        assert model == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_route_local(self):
        """local/ prefix routes to LOCAL_LLM_URL (OVMS)."""
        import proxy

        proxy.LOCAL_LLM_URL = "http://ovms:8000/v1"

        base, headers, model = await proxy._route_model("local/llama3")

        assert base == "http://ovms:8000/v1"
        assert model == "llama3"  # prefix stripped

    @pytest.mark.asyncio
    async def test_route_local_when_no_openai_key(self):
        """When OPENAI_API_KEY is empty, non-Claude models route local."""
        import proxy

        proxy.OPENAI_API_KEY = ""
        proxy.LOCAL_LLM_URL = "http://ovms:8000/v1"

        base, headers, model = await proxy._route_model("gpt-4o")

        assert base == "http://ovms:8000/v1"


# ---------------------------------------------------------------------------
# Tests for cost estimation
# ---------------------------------------------------------------------------


class TestEstimateCost:
    """Tests for _estimate_cost token pricing."""

    def test_known_model_cost(self):
        """Known model uses its price table entry."""
        from proxy import _estimate_cost

        cost = _estimate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=1000)
        # gpt-4o: input $0.005/1K, output $0.015/1K => $0.005 + $0.015 = $0.02
        assert abs(cost - 0.02) < 1e-6

    def test_unknown_model_falls_back(self):
        """Unknown model falls back to gpt-4o-mini pricing."""
        from proxy import _estimate_cost

        cost = _estimate_cost("unknown-model-v9", prompt_tokens=1000, completion_tokens=1000)
        # gpt-4o-mini: input $0.00015/1K, output $0.0006/1K => $0.00075
        assert abs(cost - 0.00075) < 1e-6

    def test_local_model_zero_cost(self):
        """Local models have zero cost."""
        from proxy import _estimate_cost

        cost = _estimate_cost("local", prompt_tokens=5000, completion_tokens=5000)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# Tests for upstream error forwarding
# ---------------------------------------------------------------------------


class TestUpstreamErrors:
    """Tests for forwarding upstream non-200 responses."""

    @pytest.mark.asyncio
    async def test_upstream_500_forwarded(self, client):
        """Upstream 500 is returned to caller."""
        import proxy

        proxy.DAILY_BUDGET_USD = 100.0
        proxy._daily_spend = 0.0

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": {"message": "Internal server error"}}

        with patch("proxy.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client

            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_upstream_401_forwarded(self, client):
        """Upstream 401 (invalid API key) is forwarded."""
        import proxy

        proxy.DAILY_BUDGET_USD = 100.0
        proxy._daily_spend = 0.0

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"error": {"message": "Invalid API key"}}

        with patch("proxy.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client

            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 401
