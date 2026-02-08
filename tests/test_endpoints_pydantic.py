"""Integration tests for FastAPI endpoints with Pydantic validation."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """FastAPI TestClient for A2A endpoint tests."""
    from workflows.a2a_server import app
    return TestClient(app)


class TestTaskEndpoint:
    """Test /task endpoint with Pydantic validation."""

    @patch("workflows.a2a_server.push_episodic")
    @patch("workflows.a2a_server.recall_experience")
    def test_task_valid_request(self, mock_recall, mock_push, client):
        """Test /task with valid request."""
        mock_recall.return_value = []

        response = client.post(
            "/task",
            json={"goal": "Analyze sales data", "context": "Q4 report"},
            headers={"x-tenant-id": "tenant-1"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert data["tenant_id"] == "tenant-1"
        assert "session_id" in data
        assert data["history_count"] == 0

    @patch("workflows.a2a_server.push_episodic")
    @patch("workflows.a2a_server.recall_experience")
    def test_task_with_session_id(self, mock_recall, mock_push, client):
        """Test /task with explicit session_id."""
        mock_recall.return_value = []

        response = client.post(
            "/task",
            json={"goal": "Test task", "session_id": "my-session-123"},
            headers={"x-tenant-id": "tenant-1"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "my-session-123"

    def test_task_missing_goal(self, client):
        """Test /task returns 422 when goal is missing."""
        response = client.post(
            "/task",
            json={"context": "Some context"},
            headers={"x-tenant-id": "tenant-1"}
        )

        assert response.status_code == 422
        assert "goal" in response.json()["detail"][0]["loc"]

    def test_task_empty_goal(self, client):
        """Test /task returns 422 when goal is empty."""
        response = client.post(
            "/task",
            json={"goal": ""},
            headers={"x-tenant-id": "tenant-1"}
        )

        assert response.status_code == 422

    def test_task_missing_tenant_header(self, client):
        """Test /task returns 400 when x-tenant-id header is missing."""
        response = client.post(
            "/task",
            json={"goal": "Test goal"}
        )

        assert response.status_code == 400
        assert "x-tenant-id" in response.json()["detail"]


class TestHandoffEndpoint:
    """Test /handoff endpoint with Pydantic validation."""

    def test_handoff_valid_request(self, client):
        """Test /handoff with valid request."""
        response = client.post(
            "/handoff",
            json={"target_agent": "agent-2", "payload": {"key": "value"}},
            headers={"x-tenant-id": "tenant-1"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "handoff_acknowledged"
        assert data["target"] == "agent-2"

    def test_handoff_missing_target(self, client):
        """Test /handoff returns 422 when target_agent is missing."""
        response = client.post(
            "/handoff",
            json={"payload": {"key": "value"}},
            headers={"x-tenant-id": "tenant-1"}
        )

        assert response.status_code == 422
        assert "target_agent" in response.json()["detail"][0]["loc"]


class TestWebhookEndpoint:
    """Test /webhook endpoint with Pydantic validation."""

    @patch("workflows.a2a_server.goose_reflect", new_callable=AsyncMock)
    def test_webhook_success(self, mock_reflect, client):
        """Test /webhook with successful status."""
        response = client.post(
            "/webhook",
            json={"task_id": "task-123", "status": "Succeeded"}
        )

        assert response.status_code == 200
        assert response.json()["ack"] is True
        mock_reflect.assert_not_called()

    @patch("workflows.a2a_server.goose_reflect", new_callable=AsyncMock)
    def test_webhook_failure(self, mock_reflect, client):
        """Test /webhook with failed status triggers reflection."""
        response = client.post(
            "/webhook",
            json={"task_id": "task-123", "status": "Failed", "message": "OOMKilled"}
        )

        assert response.status_code == 200
        assert response.json()["ack"] is True
        mock_reflect.assert_called_once_with("task-123", "OOMKilled")

    def test_webhook_defaults(self, client):
        """Test /webhook with default values."""
        response = client.post(
            "/webhook",
            json={}
        )

        assert response.status_code == 200
        assert response.json()["ack"] is True


class TestChatCompletionsEndpoint:
    """Test /v1/chat/completions endpoint with Pydantic validation."""

    @patch("workflows.a2a_server.push_episodic")
    @patch("workflows.a2a_server.recall_experience")
    @patch("httpx.AsyncClient")
    def test_chat_completions_valid(self, mock_client_cls, mock_recall, mock_push, client):
        """Test /v1/chat/completions with valid request."""
        mock_recall.return_value = []

        # Mock LiteLLM response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Test response"}}]
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "gpt-4",
                "temperature": 0.8,
                "max_tokens": 1024
            },
            headers={"x-tenant-id": "tenant-1"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    def test_chat_completions_missing_messages(self, client):
        """Test /v1/chat/completions returns 422 when messages are missing."""
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4"},
            headers={"x-tenant-id": "tenant-1"}
        )

        assert response.status_code == 422
        assert "messages" in response.json()["detail"][0]["loc"]

    def test_chat_completions_empty_messages(self, client):
        """Test /v1/chat/completions returns 422 when messages list is empty."""
        response = client.post(
            "/v1/chat/completions",
            json={"messages": []},
            headers={"x-tenant-id": "tenant-1"}
        )

        assert response.status_code == 422

    def test_chat_completions_invalid_temperature(self, client):
        """Test /v1/chat/completions validates temperature range."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 3.0
            },
            headers={"x-tenant-id": "tenant-1"}
        )

        assert response.status_code == 422


class TestUploadEndpoint:
    """Test /upload endpoint with Pydantic validation."""

    @patch("workflows.a2a_server.s3_upload")
    @patch("workflows.a2a_server.push_episodic")
    def test_upload_valid_file(self, mock_push, mock_s3, client):
        """Test /upload with valid file."""
        mock_s3.return_value = None

        response = client.post(
            "/upload",
            files={"file": ("test.txt", b"test content", "text/plain")},
            headers={"x-tenant-id": "tenant-1"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "uploaded"
        assert "test.txt" in data["key"]
        assert data["size"] == 12


class TestToolsEndpoint:
    """Test /tools endpoint with Pydantic validation."""

    @patch("httpx.AsyncClient")
    def test_tools_list(self, mock_client_cls, client):
        """Test /tools returns list of tools."""
        # Mock MCP server responses
        mock_response = AsyncMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=Exception("unreachable"))
        mock_client_cls.return_value = mock_client

        response = client.get("/tools")

        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert "total" in data
        assert data["total"] > 0


class TestCapabilitiesEndpoint:
    """Test /capabilities endpoint with Pydantic validation."""

    def test_capabilities(self, client):
        """Test /capabilities returns node capabilities."""
        response = client.get("/capabilities")

        assert response.status_code == 200
        data = response.json()
        assert data["node"] == "Antigravity Node v13.0"
        assert "a2a" in data["protocols"]
        assert "endpoints" in data
        assert "mcp_servers" in data
        assert "memory" in data
        assert "budget" in data


class TestHealthEndpoint:
    """Test /health endpoint with Pydantic validation."""

    @patch("workflows.a2a_server.full_health_check")
    def test_health_healthy(self, mock_health, client):
        """Test /health returns healthy status."""
        mock_health.return_value = {
            "status": "healthy",
            "levels": []
        }

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @patch("workflows.a2a_server.full_health_check")
    def test_health_unhealthy(self, mock_health, client):
        """Test /health returns 503 when unhealthy."""
        mock_health.return_value = {
            "status": "unhealthy",
            "levels": []
        }

        response = client.get("/health")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
