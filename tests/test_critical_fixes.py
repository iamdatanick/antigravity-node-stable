"""Tests for critical bug fixes: workflow templates, thread safety, and SQL injection prevention."""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch, AsyncMock
import threading

# Add workflows to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "workflows"))


# ---------------------------------------------------------------------------
# Test 1: Workflow Templates Fix
# ---------------------------------------------------------------------------

def test_workflow_manifest_has_single_templates_key():
    """Test that workflow manifest has only one 'templates' key with both templates."""
    # Read the workflow_defs.py file to verify the fix
    workflow_defs_path = os.path.join(
        os.path.dirname(__file__), "..", "workflows", "workflow_defs.py"
    )
    
    with open(workflow_defs_path, "r") as f:
        lines = f.readlines()
    
    # Find the spec section and count "templates" keys
    in_spec = False
    spec_depth = 0
    templates_keys = []
    
    for i, line in enumerate(lines, 1):
        if '"spec":' in line or "'spec':" in line:
            in_spec = True
            spec_depth = 0
        
        if in_spec:
            # Track brace depth
            spec_depth += line.count('{') - line.count('}')
            
            # Check for "templates" key (look for "templates": pattern)
            if '"templates":' in line or "'templates':" in line:
                # Verify it's actually a key by checking for the colon
                if ':' in line and 'templates' in line:
                    templates_keys.append(i)
            
            # Exit spec section when we close all braces
            if spec_depth < 0:
                break
    
    # Should have exactly ONE "templates" key in the spec
    assert len(templates_keys) == 1, f"Expected 1 'templates' key in spec, found {len(templates_keys)} at lines {templates_keys}"
    
    # Verify both templates are present in the file
    content = ''.join(lines)
    assert '"name": "notify-agent"' in content or "'name': 'notify-agent'" in content, "Missing notify-agent template"
    
    # Verify the correct args are used (with params, not without)
    assert "executed with params:" in content, "Missing correct args with params"
    
    # Verify onExit is set
    assert '"onExit": "notify-agent"' in content or "'onExit': 'notify-agent'" in content, "Missing onExit configuration"


# ---------------------------------------------------------------------------
# Test 2: Thread-Safe Event Counter
# ---------------------------------------------------------------------------

def test_event_counter_thread_safety():
    """Test that _event_counter is thread-safe using itertools.count."""
    # Mock pymysql before importing
    with patch.dict("sys.modules", {"pymysql": MagicMock(), "pymysql.cursors": MagicMock()}):
        import importlib
        import memory
        importlib.reload(memory)
        
        # Verify that _event_counter is an itertools.count object
        import itertools
        assert isinstance(memory._event_counter, itertools.count)
        
        # Test thread safety by calling push_episodic from multiple threads
        results = []
        errors = []
        
        def push_event(thread_id):
            try:
                with patch("memory._get_conn") as mock_conn:
                    mock_cursor = MagicMock()
                    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
                    mock_cursor.__exit__ = MagicMock(return_value=False)
                    mock_conn.return_value.cursor.return_value = mock_cursor
                    mock_conn.return_value.commit = MagicMock()
                    mock_conn.return_value.close = MagicMock()
                    
                    # Get the event_id that would be generated
                    # We can't easily capture it without modifying the function,
                    # but we can verify no exceptions occur
                    memory.push_episodic(
                        tenant_id=f"tenant-{thread_id}",
                        session_id=f"session-{thread_id}",
                        actor="Test",
                        action_type="TEST",
                        content=f"Test from thread {thread_id}",
                    )
                    results.append(thread_id)
            except Exception as e:
                errors.append((thread_id, e))
        
        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=push_event, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10


# ---------------------------------------------------------------------------
# Test 3: SQL Injection Prevention in memory.query()
# ---------------------------------------------------------------------------

def test_query_prevents_sql_injection():
    """Test that memory.query() prevents SQL injection attacks."""
    # Mock pymysql before importing
    with patch.dict("sys.modules", {"pymysql": MagicMock(), "pymysql.cursors": MagicMock()}):
        import importlib
        import memory
        importlib.reload(memory)
        
        # Test 1: Only SELECT statements should be allowed
        with pytest.raises(ValueError, match="Only SELECT queries are permitted"):
            memory.query("DROP TABLE memory_episodic")
        
        with pytest.raises(ValueError, match="Only SELECT queries are permitted"):
            memory.query("INSERT INTO memory_episodic VALUES (1, 'test')")
        
        with pytest.raises(ValueError, match="Only SELECT queries are permitted"):
            memory.query("UPDATE memory_episodic SET content='hacked'")
        
        with pytest.raises(ValueError, match="Only SELECT queries are permitted"):
            memory.query("DELETE FROM memory_episodic")
        
        # Test 2: Forbidden keywords in SELECT statements should be blocked
        with pytest.raises(ValueError, match="Forbidden SQL keyword"):
            memory.query("SELECT * FROM memory_episodic; DROP TABLE users")
        
        with pytest.raises(ValueError, match="Forbidden SQL keyword"):
            memory.query("SELECT * FROM memory_episodic WHERE content = 'x' OR 1=1; DELETE FROM memory_episodic")
        
        # Test 3: SQL comment bypass attempts should be blocked
        with pytest.raises(ValueError, match="Forbidden SQL keyword"):
            memory.query("SELECT * FROM memory_episodic/**/DROP/**/TABLE users")
        
        with pytest.raises(ValueError, match="Forbidden SQL keyword"):
            memory.query("SELECT * FROM memory_episodic -- comment\nDROP TABLE users")
        
        with pytest.raises(ValueError, match="Forbidden SQL keyword"):
            memory.query("SELECT * FROM memory_episodic\nDROP\nTABLE users")
        
        # Test 4: Valid SELECT statements should work (with mocked connection)
        with patch("memory._get_conn") as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [{"id": 1, "content": "test"}]
            mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
            mock_cursor.__exit__ = MagicMock(return_value=False)
            mock_conn.return_value.cursor.return_value = mock_cursor
            mock_conn.return_value.close = MagicMock()
            
            result = memory.query("SELECT * FROM memory_episodic")
            assert result == [{"id": 1, "content": "test"}]
            
            result = memory.query("SELECT actor, content FROM memory_episodic WHERE tenant_id = 'test'")
            assert result == [{"id": 1, "content": "test"}]


def test_query_allows_column_names_with_forbidden_keywords():
    """Test that column names containing forbidden keywords are allowed."""
    # Mock pymysql before importing
    with patch.dict("sys.modules", {"pymysql": MagicMock(), "pymysql.cursors": MagicMock()}):
        import importlib
        import memory
        importlib.reload(memory)
        
        # These should NOT raise errors because the keywords are part of column names
        with patch("memory._get_conn") as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
            mock_cursor.__exit__ = MagicMock(return_value=False)
            mock_conn.return_value.cursor.return_value = mock_cursor
            mock_conn.return_value.close = MagicMock()
            
            # Column names with keywords should work
            result = memory.query("SELECT create_date, update_time FROM memory_episodic")
            assert result == []
            
            result = memory.query("SELECT inserted_at FROM memory_episodic")
            assert result == []


# ---------------------------------------------------------------------------
# Test 4: SQL Injection Prevention in trace_viewer.py
# ---------------------------------------------------------------------------

def test_trace_viewer_uses_parameterized_queries():
    """Test that trace_viewer.py uses parameterized queries instead of string interpolation."""
    # Read the trace_viewer.py file to verify it uses parameterized queries
    trace_viewer_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "trace-viewer", "trace_viewer.py"
    )
    
    with open(trace_viewer_path, "r") as f:
        content = f.read()
    
    # Verify it uses %s placeholders for parameterized queries
    assert "actor = %s" in content, "Missing parameterized query for actor filter"
    assert "action_type = %s" in content, "Missing parameterized query for action_type filter"
    assert "LIMIT %s" in content, "Missing parameterized query for limit"
    assert "params.append" in content, "Missing params.append() calls for parameterized queries"
    
    # Verify it does NOT use f-string interpolation for SQL (security risk)
    assert "actor = '{actor_filter}'" not in content, "Found SQL injection vulnerability: f-string in SQL query"
    assert "action_type = '{action_filter}'" not in content, "Found SQL injection vulnerability: f-string in SQL query"


# ---------------------------------------------------------------------------
# Test 5: Unused Imports Removed
# ---------------------------------------------------------------------------

def test_main_py_no_unused_imports():
    """Test that workflows/main.py does not import unused modules."""
    main_path = os.path.join(os.path.dirname(__file__), "..", "workflows", "main.py")
    
    with open(main_path, "r") as f:
        lines = f.readlines()
    
    # Check import section (first 20 lines typically contain imports)
    import_lines = [line.strip() for line in lines[:20] if line.strip().startswith("import ")]
    
    # Should NOT import time (unused)
    time_imports = [line for line in import_lines if line == "import time" or line.startswith("import time,")]
    assert len(time_imports) == 0, f"Found unused 'import time' in main.py: {time_imports}"
    
    # Check for unused variables in the full file
    content = ''.join(lines)
    assert "MAX_RETRIES = " not in content, "Found unused variable MAX_RETRIES"
    assert "BASE_DELAY = " not in content, "Found unused variable BASE_DELAY"
    assert "MAX_DELAY = " not in content, "Found unused variable MAX_DELAY"


def test_a2a_server_no_unused_json_import():
    """Test that workflows/a2a_server.py does not import unused json."""
    a2a_path = os.path.join(os.path.dirname(__file__), "..", "workflows", "a2a_server.py")
    
    with open(a2a_path, "r") as f:
        lines = f.readlines()
    
    # Check import section (first 20 lines typically contain imports)
    import_lines = [line.strip() for line in lines[:20] if line.strip().startswith("import ")]
    
    # Should NOT have standalone "import json" in the import section
    json_imports = [line for line in import_lines if line == "import json" or line.startswith("import json,")]
    assert len(json_imports) == 0, f"Found unused 'import json' in a2a_server.py: {json_imports}"


def test_goose_client_no_unused_subprocess_import():
    """Test that workflows/goose_client.py does not import unused subprocess."""
    goose_path = os.path.join(os.path.dirname(__file__), "..", "workflows", "goose_client.py")
    
    with open(goose_path, "r") as f:
        lines = f.readlines()
    
    # Check import section (first 20 lines typically contain imports)
    import_lines = [line.strip() for line in lines[:20] if line.strip().startswith("import ")]
    
    # Should NOT import subprocess (unused in the current code)
    subprocess_imports = [line for line in import_lines if line == "import subprocess" or line.startswith("import subprocess,")]
    assert len(subprocess_imports) == 0, f"Found unused 'import subprocess' in goose_client.py: {subprocess_imports}"

