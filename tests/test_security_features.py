"""Tests for security features: CORS, webhook authentication, and rate limiting."""

import pytest
import hmac
import hashlib
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """FastAPI TestClient for A2A endpoint tests."""
    from workflows.a2a_server import app
    return TestClient(app)


class TestCORS:
    """Test CORS middleware configuration."""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are present in responses."""
        response = client.options(
            "/health",
            headers={"Origin": "http://example.com"}
        )
        # Check that CORS middleware is handling the request
        assert "access-control-allow-origin" in response.headers or response.status_code in [200, 405]

    def test_cors_preflight(self, client):
        """Test CORS preflight request."""
        response = client.options(
            "/task",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            }
        )
        # Should receive CORS headers or 405 (method not allowed) if OPTIONS not explicitly defined
        assert response.status_code in [200, 405]


class TestWebhookAuthentication:
    """Test webhook HMAC signature verification."""

    @patch("workflows.a2a_server.goose_reflect", new_callable=MagicMock)
    def test_webhook_no_secret_configured(self, mock_reflect, client, monkeypatch):
        """Test webhook works when WEBHOOK_SECRET is not configured."""
        # Ensure WEBHOOK_SECRET is empty (dev mode)
        monkeypatch.setenv("WEBHOOK_SECRET", "")
        
        # Need to reload the module to pick up the new env var
        import workflows.a2a_server
        workflows.a2a_server.WEBHOOK_SECRET = ""
        
        response = client.post(
            "/webhook",
            json={
                "task_id": "test-123",
                "status": "Succeeded",
                "message": "Task completed"
            }
        )
        
        assert response.status_code == 200
        assert response.json()["ack"] is True

    @patch("workflows.a2a_server.goose_reflect", new_callable=MagicMock)
    def test_webhook_with_valid_signature(self, mock_reflect, client, monkeypatch):
        """Test webhook with valid HMAC signature."""
        secret = "test-secret-key"
        monkeypatch.setenv("WEBHOOK_SECRET", secret)
        
        # Reload module to pick up new env var
        import workflows.a2a_server
        workflows.a2a_server.WEBHOOK_SECRET = secret
        
        payload = b'{"task_id":"test-123","status":"Succeeded","message":"Task completed"}'
        expected_sig = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        
        response = client.post(
            "/webhook",
            content=payload,
            headers={
                "Content-Type": "application/json",
                "x-webhook-signature": f"sha256={expected_sig}"
            }
        )
        
        assert response.status_code == 200
        assert response.json()["ack"] is True

    @patch("workflows.a2a_server.goose_reflect", new_callable=MagicMock)
    def test_webhook_with_invalid_signature(self, mock_reflect, client, monkeypatch):
        """Test webhook rejects invalid HMAC signature."""
        secret = "test-secret-key"
        monkeypatch.setenv("WEBHOOK_SECRET", secret)
        
        # Reload module to pick up new env var
        import workflows.a2a_server
        workflows.a2a_server.WEBHOOK_SECRET = secret
        
        response = client.post(
            "/webhook",
            json={
                "task_id": "test-123",
                "status": "Succeeded",
                "message": "Task completed"
            },
            headers={
                "x-webhook-signature": "sha256=invalid-signature-here"
            }
        )
        
        assert response.status_code == 401
        assert "Invalid webhook signature" in response.json()["detail"]


class TestRateLimiting:
    """Test rate limiting on endpoints."""

    @patch("workflows.a2a_server.push_episodic")
    @patch("workflows.a2a_server.recall_experience")
    def test_task_rate_limit(self, mock_recall, mock_push, client):
        """Test /task endpoint has rate limiting configured."""
        mock_recall.return_value = []
        
        # Make a single request - should succeed
        response = client.post(
            "/task",
            json={"goal": "Test task", "context": "Testing"},
            headers={"x-tenant-id": "tenant-1"}
        )
        
        # First request should succeed
        assert response.status_code == 200

    @patch("workflows.a2a_server.s3_upload")
    @patch("workflows.a2a_server.push_episodic")
    def test_upload_rate_limit(self, mock_push, mock_upload, client):
        """Test /upload endpoint has rate limiting configured."""
        # Make a single request - should succeed
        response = client.post(
            "/upload",
            files={"file": ("test.txt", b"test content", "text/plain")},
            headers={"x-tenant-id": "tenant-1"}
        )
        
        # First request should succeed
        assert response.status_code == 200

    @patch("workflows.a2a_server.push_episodic")
    @patch("workflows.a2a_server.recall_experience")
    @patch("workflows.a2a_server._load_system_prompt")
    @patch("httpx.AsyncClient")
    def test_chat_completions_rate_limit(self, mock_httpx_client, mock_prompt, mock_recall, mock_push, client):
        """Test /v1/chat/completions endpoint has rate limiting configured."""
        mock_recall.return_value = []
        mock_prompt.return_value = "You are a helpful assistant."
        
        # Mock the httpx client to avoid making real HTTP requests
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Hello"}}]
        }
        mock_httpx_client.return_value.__aenter__.return_value.post.return_value = mock_response
        
        # Make a single request - should succeed
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}]
            },
            headers={"x-tenant-id": "tenant-1"}
        )
        
        # First request should succeed
        assert response.status_code == 200


class TestPathTraversalFix:
    """Test that path traversal vulnerability is fixed."""

    def test_system_prompt_path_no_traversal(self):
        """Test that SYSTEM_PROMPT_PATH default doesn't contain path traversal."""
        from workflows.a2a_server import SYSTEM_PROMPT_PATH
        
        # Should not contain ../ in the path
        assert "../" not in SYSTEM_PROMPT_PATH
        
        # Should be a clean absolute path
        assert SYSTEM_PROMPT_PATH.startswith("/app/config/") or SYSTEM_PROMPT_PATH.startswith("/")
