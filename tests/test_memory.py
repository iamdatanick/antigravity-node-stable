"""Tests for workflows/memory.py — StarRocks memory push and query.

Covers push_episodic, push_semantic, recall_experience, query validation,
SQL injection prevention, and thread-safe event ID generation.

NOTE: ``dbutils`` is not installed in the test environment, so we mock it
before importing ``workflows.memory``.
"""

import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock dbutils before importing workflows.memory
# ---------------------------------------------------------------------------

_mock_dbutils = MagicMock()
_mock_pooled_db = MagicMock()
_mock_dbutils.pooled_db.PooledDB = _mock_pooled_db

sys.modules.setdefault("dbutils", _mock_dbutils)
sys.modules.setdefault("dbutils.pooled_db", _mock_dbutils.pooled_db)

# Now safe to import
from workflows.memory import (  # noqa: E402
    push_episodic,
    push_semantic,
    query,
    recall_experience,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_conn():
    """Create a mock StarRocks pooled connection with cursor context manager."""
    conn = MagicMock()
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    cursor.fetchall.return_value = []
    conn.cursor.return_value = cursor
    return conn, cursor


# ---------------------------------------------------------------------------
# Tests for push_episodic
# ---------------------------------------------------------------------------


class TestPushEpisodic:
    """Test push_episodic function."""

    @patch("workflows.memory._get_conn")
    def test_push_episodic_writes_correct_sql(self, mock_get_conn):
        """push_episodic inserts into memory_episodic with all required columns."""
        conn, cursor = _mock_conn()
        mock_get_conn.return_value = conn

        push_episodic(
            tenant_id="test-tenant",
            session_id="test-session",
            actor="User",
            action_type="TEST_ACTION",
            content="Test content",
        )

        assert cursor.execute.called
        sql = cursor.execute.call_args[0][0]
        params = cursor.execute.call_args[0][1]

        assert "INSERT INTO memory_episodic" in sql
        assert "tenant_id" in sql
        assert params[1] == "test-tenant"
        assert params[2] == "test-session"
        assert params[3] == "User"
        assert params[4] == "TEST_ACTION"
        assert params[5] == "Test content"

    @patch("workflows.memory._get_conn")
    def test_push_episodic_commits(self, mock_get_conn):
        """push_episodic commits the transaction."""
        conn, _ = _mock_conn()
        mock_get_conn.return_value = conn

        push_episodic("t", "s", "a", "type", "c")

        assert conn.commit.called

    @patch("workflows.memory._get_conn")
    def test_push_episodic_closes_connection(self, mock_get_conn):
        """push_episodic always closes the connection (finally block)."""
        conn, _ = _mock_conn()
        mock_get_conn.return_value = conn

        push_episodic("t", "s", "a", "type", "c")

        assert conn.close.called

    @patch("workflows.memory._get_conn")
    def test_push_episodic_generates_unique_event_ids(self, mock_get_conn):
        """Successive calls produce distinct event_id values."""
        conn, cursor = _mock_conn()
        mock_get_conn.return_value = conn

        event_ids = []

        def _capture():
            push_episodic("t", "s", "a", "type", "c")
            params = cursor.execute.call_args[0][1]
            event_ids.append(params[0])

        _capture()
        _capture()
        _capture()

        assert len(set(event_ids)) == 3


# ---------------------------------------------------------------------------
# Tests for push_semantic
# ---------------------------------------------------------------------------


class TestPushSemantic:
    """Test push_semantic function."""

    @patch("workflows.memory._get_conn")
    def test_push_semantic_inserts_correct_table(self, mock_get_conn):
        """push_semantic writes to memory_semantic table."""
        conn, cursor = _mock_conn()
        mock_get_conn.return_value = conn

        push_semantic(
            doc_id="doc-001",
            tenant_id="tenant-a",
            chunk_id=0,
            content="Knowledge chunk text",
            source_uri="s3://bucket/doc.pdf",
        )

        sql = cursor.execute.call_args[0][0]
        params = cursor.execute.call_args[0][1]

        assert "INSERT INTO memory_semantic" in sql
        assert params[0] == "doc-001"
        assert params[1] == "tenant-a"
        assert params[2] == 0
        assert params[3] == "Knowledge chunk text"
        assert params[4] == "s3://bucket/doc.pdf"

    @patch("workflows.memory._get_conn")
    def test_push_semantic_commits_and_closes(self, mock_get_conn):
        """push_semantic commits and closes connection."""
        conn, _ = _mock_conn()
        mock_get_conn.return_value = conn

        push_semantic("d", "t", 1, "c", "uri")

        assert conn.commit.called
        assert conn.close.called


# ---------------------------------------------------------------------------
# Tests for recall_experience
# ---------------------------------------------------------------------------


class TestRecallExperience:
    """Test recall_experience function."""

    @patch("workflows.memory._get_conn")
    def test_recall_returns_list(self, mock_get_conn):
        """recall_experience returns a list of rows."""
        conn, cursor = _mock_conn()
        cursor.fetchall.return_value = [
            {"action_type": "THOUGHT", "content": "Analyzing data", "timestamp": "2025-01-01"},
        ]
        mock_get_conn.return_value = conn

        result = recall_experience("goal", "test-tenant", limit=5)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["action_type"] == "THOUGHT"

    @patch("workflows.memory._get_conn")
    def test_recall_uses_select(self, mock_get_conn):
        """recall_experience issues a SELECT query with tenant filter and LIMIT."""
        conn, cursor = _mock_conn()
        mock_get_conn.return_value = conn

        recall_experience("goal", "tenant-x", limit=10)

        sql = cursor.execute.call_args[0][0]
        params = cursor.execute.call_args[0][1]

        assert "SELECT" in sql
        assert "memory_episodic" in sql
        assert "LIMIT" in sql
        assert params[0] == "tenant-x"
        assert params[1] == 10


# ---------------------------------------------------------------------------
# Tests for query() — SQL validation & injection prevention
# ---------------------------------------------------------------------------


class TestQueryValidation:
    """Test the query() function's SQL allow-list and injection defenses."""

    @patch("workflows.memory._get_conn")
    def test_select_query_allowed(self, mock_get_conn):
        """Simple SELECT query executes without error."""
        conn, cursor = _mock_conn()
        cursor.fetchall.return_value = [{"count": 42}]
        mock_get_conn.return_value = conn

        result = query("SELECT count(*) FROM memory_episodic")

        assert cursor.execute.called
        assert result == [{"count": 42}]

    def test_drop_blocked(self):
        """DROP TABLE is rejected."""
        with pytest.raises(ValueError, match="Forbidden SQL keyword"):
            query("SELECT 1; DROP TABLE memory_episodic")

    def test_delete_blocked(self):
        """DELETE is rejected."""
        with pytest.raises(ValueError, match="Forbidden SQL keyword"):
            query("SELECT 1; DELETE FROM memory_episodic")

    def test_insert_blocked(self):
        """INSERT is rejected."""
        with pytest.raises(ValueError, match="Forbidden SQL keyword"):
            query("SELECT 1; INSERT INTO memory_episodic VALUES (1)")

    def test_update_blocked(self):
        """UPDATE is rejected."""
        with pytest.raises(ValueError, match="Forbidden SQL keyword"):
            query("SELECT 1; UPDATE memory_episodic SET content='hacked'")

    def test_truncate_blocked(self):
        """TRUNCATE is rejected."""
        with pytest.raises(ValueError, match="Forbidden SQL keyword"):
            query("SELECT 1; TRUNCATE TABLE memory_episodic")

    def test_alter_blocked(self):
        """ALTER is rejected."""
        with pytest.raises(ValueError, match="Forbidden SQL keyword"):
            query("SELECT 1; ALTER TABLE memory_episodic ADD COLUMN x INT")

    def test_grant_blocked(self):
        """GRANT is rejected."""
        with pytest.raises(ValueError, match="Forbidden SQL keyword"):
            query("SELECT 1; GRANT ALL ON *.* TO 'root'")

    def test_non_select_start_blocked(self):
        """Queries not starting with SELECT are rejected."""
        with pytest.raises(ValueError, match="Only SELECT"):
            query("DELETE FROM memory_episodic WHERE 1=1")

    def test_into_outfile_blocked(self):
        """INTO OUTFILE exfiltration is blocked."""
        with pytest.raises(ValueError, match="Forbidden SQL keyword"):
            query("SELECT * FROM memory_episodic INTO OUTFILE '/tmp/data.csv'")

    def test_comment_based_injection_blocked(self):
        """SQL comments hiding forbidden keywords are still caught."""
        with pytest.raises(ValueError, match="Forbidden SQL keyword"):
            query("SELECT /* safe */ 1; DROP TABLE x")

    @patch("workflows.memory._get_conn")
    def test_query_closes_connection(self, mock_get_conn):
        """query() always closes the connection."""
        conn, cursor = _mock_conn()
        cursor.fetchall.return_value = []
        mock_get_conn.return_value = conn

        query("SELECT 1")

        assert conn.close.called

    def test_exec_blocked(self):
        """EXEC is rejected."""
        with pytest.raises(ValueError, match="Forbidden SQL keyword"):
            query("SELECT 1; EXEC sp_executesql 'DROP TABLE x'")

    def test_create_blocked(self):
        """CREATE is rejected."""
        with pytest.raises(ValueError, match="Forbidden SQL keyword"):
            query("SELECT 1; CREATE TABLE evil (id INT)")


# ---------------------------------------------------------------------------
# Tests for thread-safe counter
# ---------------------------------------------------------------------------


class TestThreadSafeCounter:
    """Test thread-safe event ID generation."""

    def test_counter_produces_unique_sequential_ids(self):
        """itertools.count produces unique sequential values."""
        import itertools

        counter = itertools.count(1)
        ids = [next(counter) for _ in range(100)]

        assert len(ids) == len(set(ids))
        assert ids == list(range(1, 101))

    @patch("workflows.memory._get_conn")
    def test_concurrent_push_unique_ids(self, mock_get_conn):
        """Concurrent push_episodic calls produce unique event IDs."""
        conn, cursor = _mock_conn()
        mock_get_conn.return_value = conn

        event_ids = []
        lock = threading.Lock()

        def _push_and_capture():
            push_episodic("tenant", "session", "actor", "type", "content")
            params = cursor.execute.call_args[0][1]
            with lock:
                event_ids.append(params[0])

        threads = [threading.Thread(target=_push_and_capture) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(event_ids) == len(set(event_ids))


# ---------------------------------------------------------------------------
# Tests for forbidden keywords list completeness
# ---------------------------------------------------------------------------


class TestForbiddenKeywordsList:
    """Verify the forbidden keywords list covers known SQL injection vectors."""

    def test_all_expected_keywords_present(self):
        """The forbidden list includes all critical write/admin operations."""
        expected_blocked = [
            "DROP",
            "DELETE",
            "INSERT",
            "UPDATE",
            "ALTER",
            "CREATE",
            "TRUNCATE",
            "GRANT",
            "REVOKE",
            "LOAD",
            "SET",
            "EXEC",
        ]

        for kw in expected_blocked:
            test_sql = f"SELECT 1; {kw} something"
            with pytest.raises(ValueError, match="Forbidden SQL keyword"):
                query(test_sql)
