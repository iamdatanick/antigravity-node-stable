"""Antigravity Node v13.0 — Shared Test Fixtures.

Provides mock services, test database connections, and FastAPI test clients
for all test modules. No real service connections required for unit tests.
"""

import os
import sys
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Add workflows to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "workflows"))


# ---------------------------------------------------------------------------
# FastAPI Test Client
# ---------------------------------------------------------------------------

@pytest.fixture
def test_client():
    """FastAPI TestClient for A2A endpoint tests."""
    from fastapi.testclient import TestClient
    from a2a_server import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# Mock StarRocks
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_starrocks():
    """Mock StarRocks connection for memory tests."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        {
            "event_id": 1,
            "tenant_id": "test-tenant",
            "timestamp": "2025-01-15 10:30:00",
            "session_id": "sess-001",
            "actor": "Goose",
            "action_type": "THOUGHT",
            "content": "Analyzing user request",
        }
    ]
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.close = MagicMock()

    with patch("memory.pymysql") as mock_pymysql:
        mock_pymysql.connect.return_value = mock_conn
        mock_pymysql.cursors.DictCursor = MagicMock()
        yield {
            "connection": mock_conn,
            "cursor": mock_cursor,
            "pymysql": mock_pymysql,
        }


# ---------------------------------------------------------------------------
# Mock SeaweedFS (boto3 S3)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_seaweedfs():
    """Mock SeaweedFS S3 client for storage tests."""
    mock_client = MagicMock()
    mock_client.put_object.return_value = {"ResponseMetadata": {"HTTPStatusCode": 200}}
    mock_client.get_object.return_value = {
        "Body": MagicMock(read=MagicMock(return_value=b"test data")),
        "ContentLength": 9,
    }
    mock_client.list_objects_v2.return_value = {
        "Contents": [
            {"Key": "context/test-tenant/test.csv", "Size": 1024},
            {"Key": "context/test-tenant/report.pdf", "Size": 2048},
        ]
    }
    mock_client.head_bucket.return_value = {}
    mock_client.create_bucket.return_value = {}

    with patch("s3_client.boto3") as mock_boto3:
        mock_boto3.client.return_value = mock_client
        yield {
            "client": mock_client,
            "boto3": mock_boto3,
        }


# ---------------------------------------------------------------------------
# Mock Milvus
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_milvus():
    """Mock Milvus connection for vector search tests."""
    mock_collection = MagicMock()
    mock_collection.search.return_value = [
        [
            MagicMock(
                id=1,
                distance=0.95,
                entity=MagicMock(get=lambda k: {"content": "test result", "doc_id": "doc-1"}.get(k)),
            )
        ]
    ]
    mock_collection.insert.return_value = MagicMock(primary_keys=[1])

    with patch("tools.milvus_tool.connections") as mock_conn, \
         patch("tools.milvus_tool.Collection") as mock_coll_cls:
        mock_coll_cls.return_value = mock_collection
        yield {
            "collection": mock_collection,
            "connections": mock_conn,
        }


# ---------------------------------------------------------------------------
# Mock Argo Client
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_argo():
    """Mock Argo client for workflow submission tests."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "metadata": {"name": "test-workflow-abc123"},
        "status": {"phase": "Running"},
    }
    mock_response.raise_for_status = MagicMock()

    with patch("workflow_defs.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client
        yield {
            "client": mock_client,
            "response": mock_response,
        }


# ---------------------------------------------------------------------------
# Mock LiteLLM
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_litellm():
    """Mock LiteLLM proxy for budget enforcement tests."""
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
    """Set environment variables for all tests."""
    monkeypatch.setenv("STARROCKS_HOST", "localhost")
    monkeypatch.setenv("STARROCKS_HTTP_PORT", "8030")
    monkeypatch.setenv("STARROCKS_PORT", "9030")
    monkeypatch.setenv("S3_ENDPOINT", "http://localhost:8333")
    monkeypatch.setenv("ARGO_SERVER", "localhost:2746")
    monkeypatch.setenv("MILVUS_HOST", "localhost")
    monkeypatch.setenv("MILVUS_PORT", "19530")
    monkeypatch.setenv("NATS_URL", "nats://localhost:4222")
    monkeypatch.setenv("VALKEY_URL", "redis://localhost:6379")
    monkeypatch.setenv("OPENBAO_ADDR", "http://localhost:8200")
    monkeypatch.setenv("OPENBAO_TOKEN", "test-token")
    monkeypatch.setenv("OPENLINEAGE_URL", "http://localhost:5000")
    monkeypatch.setenv("OPENLINEAGE_NAMESPACE", "test")
    monkeypatch.setenv("GOD_MODE_ITERATIONS", "3")
    monkeypatch.setenv("GOOSE_PROVIDER", "openai")
    monkeypatch.setenv("GOOSE_MODEL", "gpt-4o")
