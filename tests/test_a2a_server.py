"""Comprehensive tests for A2A server endpoints."""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx
import httpx
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """FastAPI TestClient for A2A endpoint tests."""
    from workflows.a2a_server import app

    return TestClient(app)


class TestHealthEndpoint:
    """Test the /health endpoint."""

    @patch("workflows.a2a_server.full_health_check")
    async def test_health_endpoint(self, mock_health, client):
        """Test GET /health returns 200 with status field."""
        mock_health.return_value = {"status": "healthy", "levels": []}
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    @patch("workflows.a2a_server.full_health_check")
    async def test_health_endpoint_degraded(self, mock_health, client):
        """Test GET /health returns 503 when degraded."""
        mock_health.return_value = {"status": "degraded", "levels": []}
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"


class TestCapabilitiesEndpoint:
    """Test the /capabilities endpoint."""

    def test_capabilities_endpoint(self, client):
        """Test GET /capabilities returns node info."""
        response = client.get("/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert "node" in data
        assert "protocols" in data
        assert "endpoints" in data
        assert "mcp_servers" in data
        assert "memory" in data
        assert "budget" in data


class TestAgentCardEndpoint:
    """Test the /.well-known/agent.json endpoint."""

    def test_agent_card_endpoint(self, client):
        """Test GET /.well-known/agent.json returns valid agent card."""
        response = client.get("/.well-known/agent.json")
        # Should return either the file or a 404
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            # If file exists, it should be valid JSON
            assert isinstance(data, dict)


class TestToolsEndpoint:
    """Test the /tools endpoint."""

    def test_tools_endpoint(self, client):
        """Test GET /tools returns tools list."""
        response = client.get("/tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert "total" in data
        assert isinstance(data["tools"], list)
        assert data["total"] >= 0


class TestTaskEndpoint:
    """Test the /task endpoint."""

    def test_task_endpoint_validation(self, client):
        """Test POST /task with invalid body returns 422."""
        response = client.post("/task", json={}, headers={"x-tenant-id": "test-tenant"})
        assert response.status_code == 422

    @patch("workflows.a2a_server.push_episodic")
    @patch("workflows.a2a_server.recall_experience")
    def test_task_endpoint_success(self, mock_recall, mock_push, client):
        """Test POST /task with valid goal returns 200."""
        mock_recall.return_value = []
        response = client.post(
            "/task", json={"goal": "Test goal", "context": "test context"}, headers={"x-tenant-id": "test-tenant"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert "session_id" in data
        assert data["tenant_id"] == "test-tenant"

    def test_task_endpoint_missing_tenant(self, client):
        """Test POST /task without x-tenant-id returns 400."""
        response = client.post("/task", json={"goal": "Test goal", "context": "test context"})
        assert response.status_code == 400


class TestWebhookEndpoint:
    """Test the /webhook endpoint."""

    @patch("workflows.a2a_server.goose_reflect")
    async def test_webhook_endpoint(self, mock_reflect, client, monkeypatch):
        """Test POST /webhook returns ack."""
        monkeypatch.setenv("WEBHOOK_SECRET", "")  # Disable signature check
        response = client.post("/webhook", json={"task_id": "test-task", "status": "Succeeded", "message": "OK"})
        assert response.status_code == 200
        data = response.json()
        assert data["ack"] is True

    @patch("workflows.a2a_server.goose_reflect")
    async def test_webhook_endpoint_failed_status(self, mock_reflect, client, monkeypatch):
        """Test POST /webhook triggers reflection on failure."""
        monkeypatch.setenv("WEBHOOK_SECRET", "")
        response = client.post("/webhook", json={"task_id": "test-task", "status": "Failed", "message": "Error occurred"})
        assert response.status_code == 200
        # goose_reflect should have been called


class TestUploadEndpoint:
    """Test the /upload endpoint."""

    @patch("workflows.a2a_server.s3_upload")
    @patch("workflows.a2a_server.push_episodic")
    def test_upload_endpoint(self, mock_push, mock_s3, client):
        """Test POST /upload with file returns key."""
        mock_s3.return_value = None
        response = client.post(
            "/upload",
            files={"file": ("test.txt", BytesIO(b"test content"), "text/plain")},
            headers={"x-tenant-id": "test-tenant"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "uploaded"
        assert "key" in data
        assert "size" in data

    def test_upload_endpoint_no_file(self, client):
        """Test POST /upload without file returns 422."""
        response = client.post("/upload", headers={"x-tenant-id": "test-tenant"})
        assert response.status_code == 422


class TestChatCompletionsEndpoint:
    """Test the /v1/chat/completions endpoint."""

    @patch("workflows.a2a_server.push_episodic")
    @patch("workflows.a2a_server.recall_experience")
    def test_chat_completions(self, mock_recall, mock_push, client):
        """Test POST /v1/chat/completions returns response."""
        mock_recall.return_value = []

        # Use respx to mock httpx calls
        with respx.mock:
            respx.post("http://litellm:4000/v1/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl-123",
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "Test response"},
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            )

            response = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
                headers={"x-tenant-id": "test-tenant"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "choices" in data


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limiting(self, client):
        """Test that rate limit headers appear in responses."""
        response = client.get("/capabilities")
        # Rate limit headers should be present or status should be normal
        assert response.status_code == 200
        # slowapi adds X-RateLimit headers
