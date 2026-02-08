"""Tests for workflows/health.py — 5-level health check hierarchy.

Covers L0-L4 individual levels, _check_url/_check_tcp helpers,
and full_health_check aggregate logic.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

# ---------------------------------------------------------------------------
# Helper: mock aiohttp response
# ---------------------------------------------------------------------------


def _make_response(status: int = 200):
    """Create a mock aiohttp response with the given status code."""
    resp = AsyncMock()
    resp.status = status
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _make_session(responses: dict[str, int] | None = None, fail_urls: set[str] | None = None):
    """Create a mock aiohttp.ClientSession.

    Parameters
    ----------
    responses : dict mapping URL substrings to HTTP status codes.
    fail_urls : set of URL substrings that should raise ConnectionError.
    """
    responses = responses or {}
    fail_urls = fail_urls or set()

    session = AsyncMock(spec=aiohttp.ClientSession)

    def _get_side_effect(url, **kwargs):
        for substr in fail_urls:
            if substr in str(url):
                raise aiohttp.ClientConnectorError(connection_key=MagicMock(), os_error=OSError("Connection refused"))
        for substr, status in responses.items():
            if substr in str(url):
                return _make_response(status)
        return _make_response(200)

    session.get = MagicMock(side_effect=_get_side_effect)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


# ---------------------------------------------------------------------------
# Tests for _check_url
# ---------------------------------------------------------------------------


class TestCheckUrl:
    """Tests for the _check_url helper."""

    @pytest.mark.asyncio
    async def test_healthy_200(self):
        """200 response marks service healthy."""
        from workflows.health import _check_url

        session = _make_session({"example": 200})
        result = await _check_url(session, "svc", "http://example:8080/health")

        assert result["name"] == "svc"
        assert result["healthy"] is True
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_unhealthy_500(self):
        """Non-200 response marks service unhealthy (accept_any=False)."""
        from workflows.health import _check_url

        session = _make_session({"example": 500})
        result = await _check_url(session, "svc", "http://example:8080/health")

        assert result["healthy"] is False
        assert result["error"] is None  # no exception, just bad status

    @pytest.mark.asyncio
    async def test_accept_any_500_still_healthy(self):
        """With accept_any=True, even a 500 counts as healthy (reachable)."""
        from workflows.health import _check_url

        session = _make_session({"example": 500})
        result = await _check_url(session, "svc", "http://example:8080/health", accept_any=True)

        assert result["healthy"] is True

    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Connection error marks service unhealthy with error string."""
        from workflows.health import _check_url

        session = _make_session(fail_urls={"example"})
        result = await _check_url(session, "svc", "http://example:8080/health")

        assert result["healthy"] is False
        assert result["error"] is not None
        assert len(result["error"]) > 0

    @pytest.mark.asyncio
    async def test_timeout_marks_unhealthy(self):
        """Timeout exception marks service unhealthy."""
        from workflows.health import _check_url

        session = AsyncMock(spec=aiohttp.ClientSession)

        def _raise_timeout(url, **kwargs):
            raise TimeoutError("Timed out")

        session.get = MagicMock(side_effect=_raise_timeout)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)

        result = await _check_url(session, "slow", "http://slow:8080/health")

        assert result["healthy"] is False
        assert "Timed out" in result["error"]


# ---------------------------------------------------------------------------
# Tests for _check_tcp
# ---------------------------------------------------------------------------


class TestCheckTcp:
    """Tests for the _check_tcp helper."""

    @pytest.mark.asyncio
    async def test_tcp_healthy(self):
        """Successful TCP connection marks service healthy."""
        from workflows.health import _check_tcp

        mock_writer = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch("workflows.health.asyncio.open_connection", new_callable=AsyncMock) as mock_open:
            mock_open.return_value = (AsyncMock(), mock_writer)
            result = await _check_tcp("nats", "localhost", 4222)

        assert result["name"] == "nats"
        assert result["healthy"] is True
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_tcp_connection_refused(self):
        """Connection refused marks service unhealthy."""
        from workflows.health import _check_tcp

        with patch("workflows.health.asyncio.open_connection", new_callable=AsyncMock) as mock_open:
            mock_open.side_effect = ConnectionRefusedError("Connection refused")
            result = await _check_tcp("nats", "localhost", 4222)

        assert result["healthy"] is False
        assert "refused" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_tcp_timeout(self):
        """Timeout marks service unhealthy."""

        from workflows.health import _check_tcp

        with patch("workflows.health.asyncio.wait_for", new_callable=AsyncMock) as mock_wait:
            mock_wait.side_effect = TimeoutError()
            result = await _check_tcp("nats", "localhost", 4222)

        assert result["healthy"] is False
        assert result["error"] is not None


# ---------------------------------------------------------------------------
# Tests for individual levels (L0-L4)
# ---------------------------------------------------------------------------


class TestCheckLevels:
    """Tests for check_level_0 through check_level_4."""

    @pytest.mark.asyncio
    async def test_level_0_structure(self):
        """L0 returns proper structure with level, name, and checks."""
        from workflows.health import check_level_0

        mock_session = _make_session()
        mock_writer = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with (
            patch("workflows.health.aiohttp.ClientSession") as mock_cls,
            patch("workflows.health.asyncio.open_connection", new_callable=AsyncMock) as mock_tcp,
        ):
            mock_cls.return_value = mock_session
            mock_tcp.return_value = (AsyncMock(), mock_writer)

            result = await check_level_0()

        assert result["level"] == "L0"
        assert result["name"] == "infrastructure"
        assert isinstance(result["checks"], (list, tuple))
        assert len(result["checks"]) >= 2  # seaweedfs + nats

    @pytest.mark.asyncio
    async def test_level_1_structure(self):
        """L1 returns orchestration checks."""
        from workflows.health import check_level_1

        mock_session = _make_session()
        with patch("workflows.health.aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value = mock_session
            result = await check_level_1()

        assert result["level"] == "L1"
        assert result["name"] == "orchestration"
        assert isinstance(result["checks"], (list, tuple))

    @pytest.mark.asyncio
    async def test_level_2_structure(self):
        """L2 returns services checks (starrocks, milvus, etc.)."""
        from workflows.health import check_level_2

        mock_session = _make_session()
        with patch("workflows.health.aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value = mock_session
            result = await check_level_2()

        assert result["level"] == "L2"
        assert result["name"] == "services"
        assert len(result["checks"]) >= 4  # starrocks, milvus, openbao, keycloak, ovms

    @pytest.mark.asyncio
    async def test_level_3_structure(self):
        """L3 returns agent checks."""
        from workflows.health import check_level_3

        mock_session = _make_session()
        with patch("workflows.health.aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value = mock_session
            result = await check_level_3()

        assert result["level"] == "L3"
        assert result["name"] == "agent"

    @pytest.mark.asyncio
    async def test_level_4_structure(self):
        """L4 returns observability checks."""
        from workflows.health import check_level_4

        mock_session = _make_session()
        with patch("workflows.health.aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value = mock_session
            result = await check_level_4()

        assert result["level"] == "L4"
        assert result["name"] == "observability"
        assert len(result["checks"]) >= 4  # budget_proxy, perses, opensearch, marquez

    @pytest.mark.asyncio
    async def test_level_check_names_present(self):
        """Each check dict has name, healthy, and error keys."""
        from workflows.health import check_level_1

        mock_session = _make_session()
        with patch("workflows.health.aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value = mock_session
            result = await check_level_1()

        for check in result["checks"]:
            assert "name" in check
            assert "healthy" in check
            assert "error" in check


# ---------------------------------------------------------------------------
# Tests for full_health_check
# ---------------------------------------------------------------------------


class TestFullHealthCheck:
    """Tests for the full_health_check aggregate function."""

    @pytest.mark.asyncio
    async def test_all_healthy(self):
        """When every service is healthy, aggregate status is 'healthy'."""
        from workflows.health import full_health_check

        mock_session = _make_session()
        mock_writer = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with (
            patch("workflows.health.aiohttp.ClientSession") as mock_cls,
            patch("workflows.health.asyncio.open_connection", new_callable=AsyncMock) as mock_tcp,
        ):
            mock_cls.return_value = mock_session
            mock_tcp.return_value = (AsyncMock(), mock_writer)

            result = await full_health_check()

        assert result["status"] == "healthy"
        assert "levels" in result
        assert len(result["levels"]) == 5

    @pytest.mark.asyncio
    async def test_degraded_when_service_fails(self):
        """When any service is unhealthy, aggregate status is 'degraded'."""
        from workflows.health import full_health_check

        # Make one URL fail
        mock_session = _make_session(fail_urls={"budget-proxy"})
        mock_writer = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with (
            patch("workflows.health.aiohttp.ClientSession") as mock_cls,
            patch("workflows.health.asyncio.open_connection", new_callable=AsyncMock) as mock_tcp,
        ):
            mock_cls.return_value = mock_session
            mock_tcp.return_value = (AsyncMock(), mock_writer)

            result = await full_health_check()

        assert result["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_levels_order(self):
        """Levels L0-L4 appear in order."""
        from workflows.health import full_health_check

        mock_session = _make_session()
        mock_writer = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with (
            patch("workflows.health.aiohttp.ClientSession") as mock_cls,
            patch("workflows.health.asyncio.open_connection", new_callable=AsyncMock) as mock_tcp,
        ):
            mock_cls.return_value = mock_session
            mock_tcp.return_value = (AsyncMock(), mock_writer)

            result = await full_health_check()

        level_names = [lv["level"] for lv in result["levels"]]
        assert level_names == ["L0", "L1", "L2", "L3", "L4"]

    @pytest.mark.asyncio
    async def test_degraded_when_non_200(self):
        """A 503 response (without accept_any) causes degraded status.

        The starrocks URL resolves to http://localhost:8030/api/health
        (from env vars), so we match on '8030' to target it specifically.
        """
        from workflows.health import full_health_check

        mock_session = _make_session(responses={"8030": 503})
        mock_writer = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with (
            patch("workflows.health.aiohttp.ClientSession") as mock_cls,
            patch("workflows.health.asyncio.open_connection", new_callable=AsyncMock) as mock_tcp,
        ):
            mock_cls.return_value = mock_session
            mock_tcp.return_value = (AsyncMock(), mock_writer)

            result = await full_health_check()

        assert result["status"] == "degraded"
