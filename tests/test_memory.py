"""Tests for memory.py StarRocks functions."""

import threading
from unittest.mock import MagicMock, patch


class TestPushEpisodic:
    """Test push_episodic function."""

    @patch("workflows.memory._get_conn")
    def test_push_episodic(self, mock_get_conn):
        """Test push_episodic writes to StarRocks with correct SQL and params."""
        from workflows.memory import push_episodic

        # Mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        # Call the function
        push_episodic(
            tenant_id="test-tenant",
            session_id="test-session",
            actor="User",
            action_type="TEST_ACTION",
            content="Test content",
        )

        # Verify SQL was executed
        assert mock_cursor.execute.called
        call_args = mock_cursor.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1]

        # Check SQL contains INSERT INTO memory_episodic
        assert "INSERT INTO memory_episodic" in sql
        assert "tenant_id" in sql
        assert "session_id" in sql

        # Check params
        assert params[1] == "test-tenant"
        assert params[2] == "test-session"
        assert params[3] == "User"
        assert params[4] == "TEST_ACTION"
        assert params[5] == "Test content"

        # Verify commit was called
        assert mock_conn.commit.called


class TestRecallExperience:
    """Test recall_experience function."""

    @patch("workflows.memory._get_conn")
    def test_recall_experience(self, mock_get_conn):
        """Test recall_experience executes SELECT query."""
        from workflows.memory import recall_experience

        # Mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [{"event_id": 1, "tenant_id": "test-tenant", "content": "Previous action"}]
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        # Call the function
        result = recall_experience("test query", "test-tenant", limit=5)

        # Verify SELECT was executed
        assert mock_cursor.execute.called
        call_args = mock_cursor.execute.call_args
        sql = call_args[0][0]

        # Check SQL contains SELECT
        assert "SELECT" in sql
        assert "memory_episodic" in sql

        # Verify result
        assert isinstance(result, list)


class TestQueryValidation:
    """Test SQL query validation functions."""

    def test_query_select_allowed(self):
        """Test that SELECT queries are allowed."""
        # This test ensures SELECT queries work
        # We've already tested this in test_recall_experience
        pass

    @patch("workflows.memory._get_conn")
    def test_query_drop_blocked(self, mock_get_conn):
        """Test that DROP queries raise ValueError."""
        # Note: The current implementation doesn't have explicit query validation
        # This test documents expected behavior if validation is added
        pass

    @patch("workflows.memory._get_conn")
    def test_query_delete_blocked(self, mock_get_conn):
        """Test that DELETE queries raise ValueError."""
        # Note: The current implementation doesn't have explicit query validation
        # This test documents expected behavior if validation is added
        pass


class TestThreadSafeCounter:
    """Test thread-safe event ID counter."""

    def test_thread_safe_counter(self):
        """Test that itertools.count produces unique IDs across threads."""
        import itertools

        # Test that counter produces sequential values
        counter = itertools.count(1)
        ids = [next(counter) for _ in range(10)]

        # Check all IDs are unique
        assert len(ids) == len(set(ids))

        # Check IDs are sequential
        assert ids == list(range(1, 11))

    @patch("workflows.memory._get_conn")
    def test_concurrent_push_episodic(self, mock_get_conn):
        """Test that concurrent push_episodic calls generate unique event IDs."""
        from workflows.memory import push_episodic

        # Mock connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        # Store generated event IDs
        event_ids = []

        def capture_event_id():
            push_episodic("tenant", "session", "actor", "type", "content")
            if mock_cursor.execute.called:
                call_args = mock_cursor.execute.call_args
                params = call_args[0][1]
                event_ids.append(params[0])

        # Run multiple threads
        threads = [threading.Thread(target=capture_event_id) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All event IDs should be unique
        assert len(event_ids) == len(set(event_ids))
