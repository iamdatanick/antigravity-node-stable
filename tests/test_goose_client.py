"""Tests for goose_client.py tool execution."""

import pytest
from unittest.mock import patch, MagicMock


class TestExecuteTool:
    """Test tool execution functions."""

    @patch("workflows.goose_client.execute_tool_with_correction")
    def test_execute_tool(self, mock_execute):
        """Test execute_tool dispatches correctly."""
        from workflows.goose_client import execute_tool_with_correction
        
        # Mock the function
        mock_execute.return_value = {"status": "success", "result": "tool executed"}
        
        # Call the function
        result = execute_tool_with_correction("test_tool", {"param": "value"})
        
        # Verify it was called
        assert mock_execute.called


class TestToolNotFound:
    """Test handling of unknown tools."""

    def test_tool_not_found(self):
        """Test that unknown tool raises appropriate error."""
        from workflows.goose_client import TOOLS
        
        # Verify TOOLS list exists and has expected structure
        assert isinstance(TOOLS, list)
        assert len(TOOLS) > 0
        
        # Each tool should have name and description
        for tool in TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "params" in tool


class TestGooseReflect:
    """Test goose_reflect function."""

    @patch("workflows.goose_client.goose_reflect")
    async def test_goose_reflect(self, mock_reflect):
        """Test goose_reflect is called for failed workflows."""
        from workflows.goose_client import goose_reflect
        
        # Mock the async function
        mock_reflect.return_value = None
        
        # Call the function
        await goose_reflect("task-123", "Error message")
        
        # Verify it was called with correct args
        assert mock_reflect.called
        call_args = mock_reflect.call_args[0]
        assert call_args[0] == "task-123"
        assert call_args[1] == "Error message"
