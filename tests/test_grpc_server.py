"""Tests for gRPC server functionality."""

import os
import sys
from unittest.mock import MagicMock, patch

import grpc

# Add workflows to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "workflows"))


def test_grpc_server_imports():
    """Test that grpc_server module can be imported."""
    import grpc_server
    assert grpc_server.GRPC_PORT == 8081
    assert hasattr(grpc_server, "SuperBuilderServicer")
    assert hasattr(grpc_server, "serve_grpc")


def test_servicer_class_exists():
    """Test that SuperBuilderServicer class has the correct method."""
    import grpc_server
    servicer = grpc_server.SuperBuilderServicer()
    assert hasattr(servicer, "ExecuteWorkflow")

    # Verify it's not async (should be a regular method)
    import inspect
    assert not inspect.iscoroutinefunction(servicer.ExecuteWorkflow)


def test_execute_workflow_sync_call():
    """Test that ExecuteWorkflow properly calls async submit_workflow in sync context."""
    import grpc_server

    servicer = grpc_server.SuperBuilderServicer()

    # Mock request and context
    mock_request = MagicMock()
    mock_request.workflow_name = "test-workflow"

    mock_context = MagicMock()
    mock_context.invocation_metadata.return_value = []
    mock_context.set_code = MagicMock()
    mock_context.set_details = MagicMock()

    # Mock the submit_workflow function
    with patch("workflows.workflow_defs.submit_workflow") as mock_submit:
        # Make it an async function that returns a run_id
        async def mock_async_submit(name, params):
            return "test-workflow-abc123"
        mock_submit.side_effect = mock_async_submit

        # Call the method
        result = servicer.ExecuteWorkflow(mock_request, mock_context)

    # Verify context was set correctly
    mock_context.set_code.assert_called_once_with(grpc.StatusCode.OK)
    assert mock_context.set_details.called

    # Verify result is None (placeholder)
    assert result is None


def test_health_check_imports():
    """Test that health check dependencies are available."""
    from grpc_health.v1 import health

    # Verify the health servicer can be instantiated
    health_servicer = health.HealthServicer()
    assert health_servicer is not None
