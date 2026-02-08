"""Tests for workflows/goose_client.py — Goose binary wrapper and MCP tools.

Covers tool registry, execute_tool dispatch, read_context path traversal
prevention, goose_reflect, and error handling.

NOTE: ``dbutils`` is not installed in the test environment, so we mock it
before importing ``workflows.memory`` (used transitively by query_starrocks).
"""

import json
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Mock dbutils before any transitive import of workflows.memory
# ---------------------------------------------------------------------------

_mock_dbutils = MagicMock()
_mock_dbutils.pooled_db.PooledDB = MagicMock()
sys.modules.setdefault("dbutils", _mock_dbutils)
sys.modules.setdefault("dbutils.pooled_db", _mock_dbutils.pooled_db)

from workflows.goose_client import (  # noqa: E402
    TOOLS,
    list_tools,
    execute_tool,
    execute_tool_with_correction,
    goose_reflect,
)


# ---------------------------------------------------------------------------
# Tests for tool registry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    """Tests for the TOOLS list and list_tools()."""

    def test_tools_list_not_empty(self):
        """TOOLS registry contains at least one tool."""
        assert len(TOOLS) > 0

    def test_each_tool_has_required_fields(self):
        """Every tool definition has name, description, and params."""
        for tool in TOOLS:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool missing 'description': {tool}"
            assert "params" in tool, f"Tool missing 'params': {tool}"

    def test_list_tools_returns_same_as_TOOLS(self):
        """list_tools() returns the TOOLS constant."""
        assert list_tools() is TOOLS

    def test_expected_tools_present(self):
        """Key tools are registered: query_starrocks, search_vectors, read_context."""
        names = {t["name"] for t in TOOLS}
        assert "query_starrocks" in names
        assert "search_vectors" in names
        assert "read_context" in names
        assert "store_artifact" in names
        assert "get_secret" in names
        assert "publish_event" in names
        assert "submit_workflow" in names


# ---------------------------------------------------------------------------
# Tests for execute_tool — query_starrocks
# ---------------------------------------------------------------------------

class TestExecuteToolQueryStarrocks:
    """Tests for execute_tool('query_starrocks', ...)."""

    @pytest.mark.asyncio
    async def test_query_starrocks_returns_json(self):
        """query_starrocks tool returns JSON-serialized query results."""
        with patch("workflows.memory._get_conn") as mock_get_conn:
            conn = MagicMock()
            cursor = MagicMock()
            cursor.__enter__ = MagicMock(return_value=cursor)
            cursor.__exit__ = MagicMock(return_value=False)
            cursor.fetchall.return_value = [{"count": 5}]
            conn.cursor.return_value = cursor
            mock_get_conn.return_value = conn

            result = await execute_tool("query_starrocks", {"sql": "SELECT count(*) FROM memory_episodic"})

        parsed = json.loads(result)
        assert parsed == [{"count": 5}]


# ---------------------------------------------------------------------------
# Tests for execute_tool — read_context (path traversal prevention)
# ---------------------------------------------------------------------------

class TestReadContextPathTraversal:
    """Tests for path traversal prevention in read_context tool."""

    @pytest.mark.asyncio
    async def test_dotdot_blocked(self):
        """Patterns containing '..' are blocked."""
        result = await execute_tool("read_context", {"pattern": "../../etc/passwd"})
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Invalid pattern" in parsed["error"]

    @pytest.mark.asyncio
    async def test_absolute_path_blocked(self):
        """Patterns starting with '/' are blocked."""
        result = await execute_tool("read_context", {"pattern": "/etc/shadow"})
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Invalid pattern" in parsed["error"]

    @pytest.mark.asyncio
    async def test_valid_pattern_succeeds(self):
        """A simple glob pattern within /app/context/ succeeds."""
        with patch("glob.glob", return_value=["/app/context/data.csv"]), \
             patch("os.path.realpath", side_effect=lambda p: p if "/app/context" in p else p):
            result = await execute_tool("read_context", {"pattern": "*.csv"})

        parsed = json.loads(result)
        assert "files" in parsed

    @pytest.mark.asyncio
    async def test_symlink_escape_blocked(self):
        """Symlink that resolves outside /app/context/ is blocked.

        The read_context code calls os.path.realpath twice:
        1. On the full_pattern (file being read)
        2. On the base directory (/app/context/)
        We mock realpath so the file resolves to /etc/shadow (escape)
        while the base directory resolves to itself (anchored).
        """
        base_real = "/app/context"

        def _fake_realpath(p):
            # Base directory resolves to itself
            if p == "/app/context/":
                return base_real
            # The target file resolves outside the base
            return "/etc/shadow"

        with patch("os.path.realpath", side_effect=_fake_realpath), \
             patch("glob.glob", return_value=[]):
            result = await execute_tool("read_context", {"pattern": "evil_link"})

        parsed = json.loads(result)
        assert "error" in parsed
        assert "Path traversal blocked" in parsed["error"]

    @pytest.mark.asyncio
    async def test_default_pattern(self):
        """When no pattern provided, defaults to '*'."""
        with patch("glob.glob", return_value=[]), \
             patch("os.path.realpath", side_effect=lambda p: p):
            result = await execute_tool("read_context", {})

        parsed = json.loads(result)
        # Should not error — uses default '*'
        assert "files" in parsed


# ---------------------------------------------------------------------------
# Tests for execute_tool — unknown tool
# ---------------------------------------------------------------------------

class TestExecuteToolUnknown:
    """Tests for unknown tool name handling."""

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        """Unknown tool name returns a JSON error."""
        result = await execute_tool("nonexistent_tool", {"x": 1})
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Unknown tool" in parsed["error"]


# ---------------------------------------------------------------------------
# Tests for goose_reflect
# ---------------------------------------------------------------------------

class TestGooseReflect:
    """Tests for the goose_reflect self-correction function."""

    @pytest.mark.asyncio
    async def test_reflect_returns_dict(self):
        """goose_reflect returns a dict with reflected=True and task_id."""
        result = await goose_reflect("task-123", "Something failed")

        assert isinstance(result, dict)
        assert result["reflected"] is True
        assert result["task_id"] == "task-123"

    @pytest.mark.asyncio
    async def test_reflect_handles_empty_message(self):
        """goose_reflect works with empty failure message."""
        result = await goose_reflect("task-456", "")

        assert result["reflected"] is True
        assert result["task_id"] == "task-456"


# ---------------------------------------------------------------------------
# Tests for execute_tool_with_correction (retry logic)
# ---------------------------------------------------------------------------

class TestExecuteToolWithCorrection:
    """Tests for the retry wrapper."""

    @pytest.mark.asyncio
    async def test_successful_execution_no_retry(self):
        """Successful tool execution returns on first try."""
        with patch("workflows.goose_client.execute_tool", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = json.dumps({"status": "ok"})

            result = await execute_tool_with_correction("query_starrocks", {"sql": "SELECT 1"})

        assert json.loads(result)["status"] == "ok"
        assert mock_exec.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure_then_succeed(self):
        """Tool execution retries on failure and succeeds on second attempt."""
        call_count = 0

        async def _flaky(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Transient error")
            return json.dumps({"status": "recovered"})

        with patch("workflows.goose_client.execute_tool", side_effect=_flaky):
            result = await execute_tool_with_correction("query_starrocks", {"sql": "SELECT 1"})

        assert json.loads(result)["status"] == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self):
        """After 3 failed attempts, tenacity raises RetryError."""
        from tenacity import RetryError

        async def _always_fail(*args, **kwargs):
            raise RuntimeError("Permanent failure")

        with patch("workflows.goose_client.execute_tool", side_effect=_always_fail):
            with pytest.raises(RetryError):
                await execute_tool_with_correction("query_starrocks", {"sql": "SELECT 1"})


# ---------------------------------------------------------------------------
# Tests for execute_tool — store_artifact
# ---------------------------------------------------------------------------

class TestExecuteToolStoreArtifact:
    """Tests for execute_tool('store_artifact', ...)."""

    @pytest.mark.asyncio
    async def test_store_artifact_calls_upload(self):
        """store_artifact tool delegates to s3_client.upload."""
        with patch("workflows.s3_client.upload") as mock_upload:
            result = await execute_tool("store_artifact", {"key": "test/file.txt", "data": "hello"})

        parsed = json.loads(result)
        assert parsed["status"] == "uploaded"
        assert parsed["key"] == "test/file.txt"
        assert mock_upload.called
