"""Tests for OVMS inference pipeline (workflows/inference.py)."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# Ensure workflows package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def ovms_env(monkeypatch):
    """Set OVMS env vars for all tests."""
    monkeypatch.setenv("OVMS_GRPC", "localhost:9000")
    monkeypatch.setenv("OVMS_REST", "http://localhost:8000")


def _make_response(status_code=200, json_data=None, text=""):
    """Create a MagicMock that behaves like an httpx.Response.

    httpx Response.json() and .text are synchronous, so we use MagicMock
    (not AsyncMock) to avoid returning coroutines.
    """
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp


def _make_async_client(get_response=None, post_response=None, get_side_effect=None, post_side_effect=None):
    """Create an AsyncMock that acts as httpx.AsyncClient context manager."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    if get_side_effect:
        mock_client.get = AsyncMock(side_effect=get_side_effect)
    elif get_response is not None:
        mock_client.get = AsyncMock(return_value=get_response)
    if post_side_effect:
        mock_client.post = AsyncMock(side_effect=post_side_effect)
    elif post_response is not None:
        mock_client.post = AsyncMock(return_value=post_response)
    return mock_client


# ---------------------------------------------------------------------------
# Unit tests: inference module functions
# ---------------------------------------------------------------------------


class TestOvmsHealthCheck:
    """Tests for ovms_health_check()."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """OVMS reachable with empty model list returns healthy."""
        resp = _make_response(200, {"model_config_list": []})

        with patch("workflows.inference.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = _make_async_client(get_response=resp)

            from workflows.inference import ovms_health_check

            result = await ovms_health_check()

        assert result["name"] == "ovms"
        assert result["healthy"] is True
        assert result["error"] is None
        assert result["model_count"] == 0

    @pytest.mark.asyncio
    async def test_health_check_with_models(self):
        """OVMS reachable with models returns healthy + count."""
        resp = _make_response(
            200,
            {
                "model_config_list": [
                    {"config": {"name": "resnet", "base_path": "/models/resnet"}},
                    {"config": {"name": "bert", "base_path": "/models/bert"}},
                ]
            },
        )

        with patch("workflows.inference.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = _make_async_client(get_response=resp)

            from workflows.inference import ovms_health_check

            result = await ovms_health_check()

        assert result["healthy"] is True
        assert result["model_count"] == 2

    @pytest.mark.asyncio
    async def test_health_check_unreachable(self):
        """OVMS unreachable returns unhealthy."""
        with patch("workflows.inference.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = _make_async_client(
                get_side_effect=httpx.ConnectError("Connection refused"),
            )

            from workflows.inference import ovms_health_check

            result = await ovms_health_check()

        assert result["name"] == "ovms"
        assert result["healthy"] is False
        assert result["error"] is not None


class TestListModels:
    """Tests for list_models()."""

    @pytest.mark.asyncio
    async def test_list_models_empty(self):
        """Empty config returns empty list."""
        resp = _make_response(200, {"model_config_list": []})

        with patch("workflows.inference.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = _make_async_client(get_response=resp)

            from workflows.inference import list_models

            result = await list_models()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_models_populated(self):
        """Config with models returns their names."""
        resp = _make_response(
            200,
            {
                "model_config_list": [
                    {"config": {"name": "resnet", "base_path": "/models/resnet"}},
                ]
            },
        )

        with patch("workflows.inference.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = _make_async_client(get_response=resp)

            from workflows.inference import list_models

            result = await list_models()

        assert result == ["resnet"]

    @pytest.mark.asyncio
    async def test_list_models_unreachable(self):
        """OVMS unreachable returns empty list (no crash)."""
        with patch("workflows.inference.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = _make_async_client(
                get_side_effect=httpx.ConnectError("refused"),
            )

            from workflows.inference import list_models

            result = await list_models()

        assert result == []


class TestRunInference:
    """Tests for run_inference()."""

    @pytest.mark.asyncio
    async def test_no_models_loaded(self):
        """When OVMS has no models, return no_model_loaded status."""
        with patch("workflows.inference.list_models", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []

            from workflows.inference import run_inference

            result = await run_inference("resnet", {"input": [[1.0, 2.0]]})

        assert result["status"] == "no_model_loaded"
        assert "message" in result

    @pytest.mark.asyncio
    async def test_model_not_found(self):
        """When requested model is not in available list."""
        with patch("workflows.inference.list_models", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = ["bert"]

            from workflows.inference import run_inference

            result = await run_inference("resnet", {"input": [[1.0, 2.0]]})

        assert result["status"] == "model_not_found"
        assert "resnet" in result["message"]
        assert "bert" in result["message"]

    @pytest.mark.asyncio
    async def test_rest_inference_success(self):
        """Successful REST inference (gRPC stubs unavailable)."""
        resp = _make_response(200, {"outputs": {"output": [0.9, 0.1]}})

        with (
            patch("workflows.inference.list_models", new_callable=AsyncMock) as mock_list,
            patch("workflows.inference._get_predict_service_stub", return_value=(None, None, None)),
            patch("workflows.inference.httpx.AsyncClient") as mock_cls,
        ):
            mock_list.return_value = ["resnet"]
            mock_cls.return_value = _make_async_client(post_response=resp)

            from workflows.inference import run_inference

            result = await run_inference("resnet", {"input": [[1.0, 2.0, 3.0]]})

        assert result["status"] == "ok"
        assert result["model"] == "resnet"
        assert result["outputs"] == {"output": [0.9, 0.1]}
        assert "latency_ms" in result
        assert result["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_rest_inference_404(self):
        """REST returns 404 for model."""
        resp = _make_response(404, text="Model not found")

        with (
            patch("workflows.inference.list_models", new_callable=AsyncMock) as mock_list,
            patch("workflows.inference._get_predict_service_stub", return_value=(None, None, None)),
            patch("workflows.inference.httpx.AsyncClient") as mock_cls,
        ):
            mock_list.return_value = ["resnet"]
            mock_cls.return_value = _make_async_client(post_response=resp)

            from workflows.inference import run_inference

            result = await run_inference("resnet", {"input": [[1.0]]})

        assert result["status"] == "model_not_found"

    @pytest.mark.asyncio
    async def test_rest_inference_server_error(self):
        """REST returns 500 yields error status."""
        resp = _make_response(500, text="Internal server error")

        with (
            patch("workflows.inference.list_models", new_callable=AsyncMock) as mock_list,
            patch("workflows.inference._get_predict_service_stub", return_value=(None, None, None)),
            patch("workflows.inference.httpx.AsyncClient") as mock_cls,
        ):
            mock_list.return_value = ["resnet"]
            mock_cls.return_value = _make_async_client(post_response=resp)

            from workflows.inference import run_inference

            result = await run_inference("resnet", {"input": [[1.0]]})

        assert result["status"] == "error"
        assert "500" in result["message"]

    @pytest.mark.asyncio
    async def test_rest_inference_unreachable(self):
        """REST endpoint unreachable returns error."""
        with (
            patch("workflows.inference.list_models", new_callable=AsyncMock) as mock_list,
            patch("workflows.inference._get_predict_service_stub", return_value=(None, None, None)),
            patch("workflows.inference.httpx.AsyncClient") as mock_cls,
        ):
            mock_list.return_value = ["resnet"]
            mock_cls.return_value = _make_async_client(
                post_side_effect=httpx.ConnectError("refused"),
            )

            from workflows.inference import run_inference

            result = await run_inference("resnet", {"input": [[1.0]]})

        assert result["status"] == "error"
        assert "unreachable" in result["message"].lower()


# ---------------------------------------------------------------------------
# Integration tests: FastAPI endpoint (requires slowapi)
# ---------------------------------------------------------------------------

try:
    import slowapi  # noqa: F401

    _HAS_SLOWAPI = True
except ImportError:
    _HAS_SLOWAPI = False


@pytest.fixture
def client():
    """FastAPI TestClient with inference routes available."""
    from fastapi.testclient import TestClient

    from workflows.a2a_server import app

    return TestClient(app)


@pytest.mark.skipif(not _HAS_SLOWAPI, reason="slowapi not installed")
class TestInferenceEndpoint:
    """Tests for POST /v1/inference."""

    @patch("workflows.a2a_server.validate_token", return_value={"sub": "test"})
    @patch("workflows.a2a_server.run_inference", new_callable=AsyncMock)
    def test_inference_no_model_loaded(self, mock_infer, mock_auth, client):
        """POST /v1/inference with empty OVMS returns no_model_loaded."""
        mock_infer.return_value = {
            "status": "no_model_loaded",
            "message": "OVMS has no models loaded.",
        }
        resp = client.post(
            "/v1/inference",
            json={"model_name": "resnet", "input_data": {"input": [[1.0]]}},
            headers={"x-tenant-id": "test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "no_model_loaded"

    @patch("workflows.a2a_server.validate_token", return_value={"sub": "test"})
    @patch("workflows.a2a_server.run_inference", new_callable=AsyncMock)
    def test_inference_success(self, mock_infer, mock_auth, client):
        """POST /v1/inference with loaded model returns outputs."""
        mock_infer.return_value = {
            "status": "ok",
            "model": "resnet",
            "outputs": {"output": [0.9, 0.1]},
            "latency_ms": 12.5,
        }
        resp = client.post(
            "/v1/inference",
            json={"model_name": "resnet", "input_data": {"input": [[1.0, 2.0]]}},
            headers={"x-tenant-id": "test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model"] == "resnet"
        assert data["outputs"] == {"output": [0.9, 0.1]}
        assert data["latency_ms"] == 12.5

    @patch("workflows.a2a_server.validate_token", return_value={"sub": "test"})
    @patch("workflows.a2a_server.run_inference", new_callable=AsyncMock)
    def test_inference_model_not_found(self, mock_infer, mock_auth, client):
        """POST /v1/inference for missing model returns model_not_found."""
        mock_infer.return_value = {
            "status": "model_not_found",
            "message": "Model 'foo' not found. Available: ['resnet']",
        }
        resp = client.post(
            "/v1/inference",
            json={"model_name": "foo", "input_data": {"input": [[1.0]]}},
            headers={"x-tenant-id": "test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "model_not_found"

    def test_inference_missing_model_name(self, client):
        """POST /v1/inference without model_name returns 422."""
        resp = client.post(
            "/v1/inference",
            json={"input_data": {"input": [[1.0]]}},
        )
        assert resp.status_code == 422


@pytest.mark.skipif(not _HAS_SLOWAPI, reason="slowapi not installed")
class TestOvmsModelsEndpoint:
    """Tests for GET /v1/models/ovms."""

    @patch("workflows.a2a_server.ovms_list_models", new_callable=AsyncMock)
    def test_list_ovms_models_empty(self, mock_list, client):
        """GET /v1/models/ovms with no models."""
        mock_list.return_value = []
        resp = client.get("/v1/models/ovms")
        assert resp.status_code == 200
        data = resp.json()
        assert data["models"] == []
        assert data["count"] == 0

    @patch("workflows.a2a_server.ovms_list_models", new_callable=AsyncMock)
    def test_list_ovms_models_populated(self, mock_list, client):
        """GET /v1/models/ovms with loaded models."""
        mock_list.return_value = ["resnet", "bert"]
        resp = client.get("/v1/models/ovms")
        assert resp.status_code == 200
        data = resp.json()
        assert data["models"] == ["resnet", "bert"]
        assert data["count"] == 2
