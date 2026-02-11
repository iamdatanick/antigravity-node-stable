"""Antigravity Node v14.1 — Shared Test Fixtures.

Provides mock services, test database connections, and FastAPI test clients
for all test modules. No real service connections required for unit tests.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add workflows to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "workflows"))


# ---------------------------------------------------------------------------
# FastAPI Test Client
# ---------------------------------------------------------------------------


@pytest.fixture
def test_client():
    """FastAPI TestClient for A2A endpoint tests."""
    from a2a_server import app
    from fastapi.testclient import TestClient

    return TestClient(app)


# ---------------------------------------------------------------------------
# Mock Budget Proxy (formerly LiteLLM)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_litellm():
    """Mock budget-proxy for budget enforcement tests."""
    normal_response = AsyncMock()
    normal_response.status_code = 200
    normal_response.json.return_value = {
        "choices": [{"message": {"content": "Test response"}}],
        "usage": {"total_tokens": 150, "prompt_tokens": 50, "completion_tokens": 100},
    }

    budget_exceeded_response = AsyncMock()
    budget_exceeded_response.status_code = 429
    budget_exceeded_response.json.return_value = {
        "error": {"message": "Budget exceeded. Max budget: $10.00", "type": "budget_exceeded"}
    }

    yield {
        "normal": normal_response,
        "budget_exceeded": budget_exceeded_response,
    }


# ---------------------------------------------------------------------------
# Mock Goose
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_goose():
    """Mock Goose subprocess for agent tests."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = '{"response": "I analyzed the data", "tools_used": ["query_starrocks"]}'
    mock_proc.stderr = ""

    with patch("goose_client.subprocess") as mock_subprocess:
        mock_subprocess.run.return_value = mock_proc
        yield {
            "process": mock_proc,
            "subprocess": mock_subprocess,
        }


# ---------------------------------------------------------------------------
# Test Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_task_request():
    """Sample A2A task request payload."""
    return {
        "goal": "Analyze the uploaded sales data and generate a summary report",
        "context": "User uploaded sales_q4.csv via mobile app",
    }


@pytest.fixture
def sample_upload_file(tmp_path):
    """Create a temporary CSV file for upload tests."""
    csv_file = tmp_path / "test_upload.csv"
    csv_file.write_text("name,value\nAlice,100\nBob,200\nCharlie,300\n")
    return csv_file


@pytest.fixture
def sample_webhook_payload():
    """Sample Argo webhook callback payload."""
    return {
        "task_id": "ingest-pipeline-abc123",
        "status": "Failed",
        "message": "Pod failed: OOMKilled",
    }


@pytest.fixture
def sample_webhook_success():
    """Sample successful Argo webhook callback payload."""
    return {
        "task_id": "ingest-pipeline-def456",
        "status": "Succeeded",
        "message": "",
    }


# ---------------------------------------------------------------------------
# Environment Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Set environment variables for all tests (v14.1 service set)."""
    # L0 Infrastructure
    monkeypatch.setenv("ETCD_HOST", "localhost")
    monkeypatch.setenv("CEPH_HOST", "localhost")
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://localhost:8000")
    monkeypatch.setenv("S3_ENDPOINT", "http://localhost:8000")
    # L0 Secrets
    monkeypatch.setenv("OPENBAO_ADDR", "http://localhost:8200")
    monkeypatch.setenv("OPENBAO_TOKEN", "test-token")
    # L2 Inference
    monkeypatch.setenv("OVMS_GRPC", "localhost:9000")
    monkeypatch.setenv("OVMS_REST_URL", "http://localhost:9001")
    monkeypatch.setenv("OVMS_REST", "http://localhost:9001")
    # L3 Observability
    monkeypatch.setenv("OTEL_COLLECTOR_HOST", "localhost")
    # Application
    monkeypatch.setenv("GOD_MODE_ITERATIONS", "3")
    monkeypatch.setenv("GOOSE_PROVIDER", "openai")
    monkeypatch.setenv("GOOSE_MODEL", "gpt-4o")
    monkeypatch.setenv("VALKEY_URL", "redis://localhost:6379")
    # Lineage (still used in v14.1 for OpenLineage/Marquez)
    monkeypatch.setenv("OPENLINEAGE_URL", "http://localhost:5000")
    monkeypatch.setenv("OPENLINEAGE_NAMESPACE", "test")
