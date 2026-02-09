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
    """Test that SuperBuilderServicer class has the correct methods."""
    import grpc_server

    servicer = grpc_server.SuperBuilderServicer()
    assert hasattr(servicer, "ExecuteWorkflow")
    assert hasattr(servicer, "GetWorkflowStatus")

    # Verify they're not async (should be regular methods)
    import inspect

    assert not inspect.iscoroutinefunction(servicer.ExecuteWorkflow)
    assert not inspect.iscoroutinefunction(servicer.GetWorkflowStatus)


def test_execute_workflow_sync_call():
    """Test that ExecuteWorkflow properly calls async submit_workflow and returns response."""
    import grpc_server
    from workflows import superbuilder_pb2

    servicer = grpc_server.SuperBuilderServicer()

    # Mock request and context
    mock_request = superbuilder_pb2.WorkflowRequest(workflow_name="test-workflow", tenant_id="test-tenant")

    mock_context = MagicMock()
    mock_context.invocation_metadata.return_value = []

    # Mock the submit_workflow function
    with patch("workflows.workflow_defs.submit_workflow") as mock_submit:
        # Make it an async function that returns a run_id
        async def mock_async_submit(name, params):
            return "test-workflow-abc123"

        mock_submit.side_effect = mock_async_submit

        # Call the method
        response = servicer.ExecuteWorkflow(mock_request, mock_context)

    # Verify the response
    assert isinstance(response, superbuilder_pb2.WorkflowResponse)
    assert response.status == "submitted"
    assert response.run_id == "test-workflow-abc123"
    assert response.agent_id == "test-workflow-abc123"


def test_get_workflow_status():
    """Test that GetWorkflowStatus properly calls async get_workflow_status and returns response."""
    import grpc_server
    from workflows import superbuilder_pb2

    servicer = grpc_server.SuperBuilderServicer()

    # Mock request and context
    mock_request = superbuilder_pb2.StatusRequest(run_id="test-workflow-abc123")
    mock_context = MagicMock()

    # Mock the get_workflow_status function
    with patch("workflows.workflow_defs.get_workflow_status") as mock_status:
        # Make it an async function that returns status
        async def mock_async_status(run_id):
            return {"name": run_id, "phase": "Succeeded"}

        mock_status.side_effect = mock_async_status

        # Call the method
        response = servicer.GetWorkflowStatus(mock_request, mock_context)

    # Verify the response
    assert isinstance(response, superbuilder_pb2.StatusResponse)
    assert response.run_id == "test-workflow-abc123"
    assert response.phase == "Succeeded"
    assert "Succeeded" in response.message


def test_health_check_imports():
    """Test that health check dependencies are available."""
    from grpc_health.v1 import health

    # Verify the health servicer can be instantiated
    health_servicer = health.HealthServicer()
    assert health_servicer is not None
