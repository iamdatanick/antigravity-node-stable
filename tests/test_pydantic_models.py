"""Tests for Pydantic models and FastAPI endpoint validation."""

import pytest
from pydantic import ValidationError
from workflows.models import (
    TaskRequest, TaskResponse, HandoffRequest, HandoffResponse,
    WebhookPayload, WebhookResponse, ChatMessage, ChatCompletionRequest,
    UploadResponse, HealthCheck, HealthLevel, HealthResponse,
    ToolInfo, ToolsResponse, CapabilitiesResponse
)


class TestTaskModels:
    """Test /task endpoint models."""

    def test_task_request_valid(self):
        """Test valid TaskRequest creation."""
        req = TaskRequest(goal="Analyze sales data", context="Q4 report", session_id="sess-123")
        assert req.goal == "Analyze sales data"
        assert req.context == "Q4 report"
        assert req.session_id == "sess-123"

    def test_task_request_defaults(self):
        """Test TaskRequest with default values."""
        req = TaskRequest(goal="Test goal")
        assert req.goal == "Test goal"
        assert req.context == ""
        assert req.session_id is None

    def test_task_request_goal_required(self):
        """Test TaskRequest requires goal field."""
        with pytest.raises(ValidationError) as exc_info:
            TaskRequest()
        assert "goal" in str(exc_info.value)

    def test_task_request_goal_min_length(self):
        """Test TaskRequest goal minimum length validation."""
        with pytest.raises(ValidationError) as exc_info:
            TaskRequest(goal="")
        assert "at least 1 character" in str(exc_info.value)

    def test_task_request_goal_max_length(self):
        """Test TaskRequest goal maximum length validation."""
        with pytest.raises(ValidationError) as exc_info:
            TaskRequest(goal="x" * 10001)
        assert "at most 10000 characters" in str(exc_info.value)

    def test_task_response_valid(self):
        """Test valid TaskResponse creation."""
        resp = TaskResponse(
            status="accepted",
            session_id="sess-123",
            tenant_id="tenant-1",
            history_count=5
        )
        assert resp.status == "accepted"
        assert resp.history_count == 5


class TestHandoffModels:
    """Test /handoff endpoint models."""

    def test_handoff_request_valid(self):
        """Test valid HandoffRequest creation."""
        req = HandoffRequest(target_agent="agent-2", payload={"key": "value"})
        assert req.target_agent == "agent-2"
        assert req.payload == {"key": "value"}

    def test_handoff_request_defaults(self):
        """Test HandoffRequest with default values."""
        req = HandoffRequest(target_agent="agent-2")
        assert req.payload == {}

    def test_handoff_request_target_required(self):
        """Test HandoffRequest requires target_agent field."""
        with pytest.raises(ValidationError) as exc_info:
            HandoffRequest()
        assert "target_agent" in str(exc_info.value)

    def test_handoff_response_valid(self):
        """Test valid HandoffResponse creation."""
        resp = HandoffResponse(status="handoff_acknowledged", target="agent-2")
        assert resp.status == "handoff_acknowledged"
        assert resp.target == "agent-2"


class TestWebhookModels:
    """Test /webhook endpoint models."""

    def test_webhook_payload_valid(self):
        """Test valid WebhookPayload creation."""
        payload = WebhookPayload(task_id="task-123", status="Succeeded", message="Done")
        assert payload.task_id == "task-123"
        assert payload.status == "Succeeded"
        assert payload.message == "Done"

    def test_webhook_payload_defaults(self):
        """Test WebhookPayload with default values."""
        payload = WebhookPayload()
        assert payload.task_id == "unknown"
        assert payload.status == "unknown"
        assert payload.message == ""

    def test_webhook_response_valid(self):
        """Test valid WebhookResponse creation."""
        resp = WebhookResponse(ack=True)
        assert resp.ack is True


class TestChatModels:
    """Test /v1/chat/completions endpoint models."""

    def test_chat_message_valid(self):
        """Test valid ChatMessage creation."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_message_required_fields(self):
        """Test ChatMessage requires role and content."""
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(role="user")
        assert "content" in str(exc_info.value)

    def test_chat_completion_request_valid(self):
        """Test valid ChatCompletionRequest creation."""
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            model="gpt-4",
            temperature=0.8,
            max_tokens=1024
        )
        assert len(req.messages) == 1
        assert req.model == "gpt-4"
        assert req.temperature == 0.8
        assert req.max_tokens == 1024

    def test_chat_completion_request_defaults(self):
        """Test ChatCompletionRequest with default values."""
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")]
        )
        assert req.model is None
        assert req.temperature == 0.7
        assert req.max_tokens == 2048

    def test_chat_completion_request_messages_required(self):
        """Test ChatCompletionRequest requires messages."""
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(messages=[])
        assert "at least 1 item" in str(exc_info.value)

    def test_chat_completion_request_temperature_range(self):
        """Test ChatCompletionRequest temperature validation."""
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Hello")],
                temperature=3.0
            )
        assert "less than or equal to 2" in str(exc_info.value)


class TestUploadModels:
    """Test /upload endpoint models."""

    def test_upload_response_valid(self):
        """Test valid UploadResponse creation."""
        resp = UploadResponse(status="uploaded", key="context/tenant/file.txt", size=1024)
        assert resp.status == "uploaded"
        assert resp.key == "context/tenant/file.txt"
        assert resp.size == 1024


class TestHealthModels:
    """Test /health endpoint models."""

    def test_health_check_valid(self):
        """Test valid HealthCheck creation."""
        check = HealthCheck(name="starrocks", healthy=True)
        assert check.name == "starrocks"
        assert check.healthy is True
        assert check.error is None

    def test_health_level_valid(self):
        """Test valid HealthLevel creation."""
        level = HealthLevel(
            level="L1",
            name="Core Services",
            checks=[HealthCheck(name="test", healthy=True)]
        )
        assert level.level == "L1"
        assert len(level.checks) == 1

    def test_health_response_valid(self):
        """Test valid HealthResponse creation."""
        resp = HealthResponse(
            status="healthy",
            levels=[
                HealthLevel(
                    level="L1",
                    name="Core",
                    checks=[HealthCheck(name="test", healthy=True)]
                )
            ]
        )
        assert resp.status == "healthy"
        assert len(resp.levels) == 1


class TestToolsModels:
    """Test /tools endpoint models."""

    def test_tool_info_valid(self):
        """Test valid ToolInfo creation."""
        tool = ToolInfo(
            name="search_memory",
            server="orchestrator",
            description="Search memory",
            status="connected"
        )
        assert tool.name == "search_memory"
        assert tool.server == "orchestrator"

    def test_tools_response_valid(self):
        """Test valid ToolsResponse creation."""
        resp = ToolsResponse(
            tools=[ToolInfo(name="test", server="test-server")],
            total=1
        )
        assert len(resp.tools) == 1
        assert resp.total == 1


class TestCapabilitiesModels:
    """Test /capabilities endpoint models."""

    def test_capabilities_response_valid(self):
        """Test valid CapabilitiesResponse creation."""
        resp = CapabilitiesResponse(
            node="Antigravity Node v13.0",
            protocols=["a2a", "mcp"],
            endpoints={},
            mcp_servers={},
            memory={},
            budget={}
        )
        assert resp.node == "Antigravity Node v13.0"
        assert "a2a" in resp.protocols
