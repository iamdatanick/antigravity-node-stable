"""Tests for workflow_defs.py Argo workflow submission."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestWorkflowManifest:
    """Test workflow manifest structure."""

    async def test_workflow_manifest_structure(self):
        """Test that workflow manifest has correct structure with templates."""
        from workflows.workflow_defs import submit_workflow
        
        # The submit_workflow function creates a manifest internally
        # We'll verify it calls the API with correct structure
        with patch("workflows.workflow_defs.httpx.AsyncClient") as mock_httpx:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"metadata": {"name": "test-workflow-abc123"}}
            
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_response
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_httpx.return_value = mock_client_instance
            
            # Call submit_workflow
            result = await submit_workflow("test-workflow", {"param1": "value1"})
            
            # Verify httpx.AsyncClient was used
            assert mock_httpx.called
            
            # Verify post was called
            assert mock_client_instance.post.called
            call_args = mock_client_instance.post.call_args
            
            # Check the manifest structure
            manifest = call_args[1]["json"]
            assert "apiVersion" in manifest
            assert manifest["apiVersion"] == "argoproj.io/v1alpha1"
            assert "kind" in manifest
            assert manifest["kind"] == "Workflow"
            assert "spec" in manifest
            assert "templates" in manifest["spec"]
            
            # Should have both main template and exit handler
            templates = manifest["spec"]["templates"]
            assert isinstance(templates, list)
            assert len(templates) >= 2  # At least main template and notify-agent


class TestSubmitWorkflow:
    """Test workflow submission."""

    async def test_submit_workflow(self):
        """Test submit_workflow posts manifest to Argo API."""
        from workflows.workflow_defs import submit_workflow
        
        with patch("workflows.workflow_defs.httpx.AsyncClient") as mock_httpx:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"metadata": {"name": "test-workflow-abc123"}}
            
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_response
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_httpx.return_value = mock_client_instance
            
            # Call submit_workflow
            result = await submit_workflow("test-workflow", {"param1": "value1"})
            
            # Verify post was called
            assert mock_client_instance.post.called
            
            # Verify result contains workflow name
            assert isinstance(result, str)

    async def test_submit_workflow_with_params(self):
        """Test submit_workflow includes parameters in manifest."""
        from workflows.workflow_defs import submit_workflow
        
        with patch("workflows.workflow_defs.httpx.AsyncClient") as mock_httpx:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"metadata": {"name": "test-workflow-abc123"}}
            
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_response
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_httpx.return_value = mock_client_instance
            
            # Call with parameters
            params = {"param1": "value1", "param2": "value2"}
            result = await submit_workflow("test-workflow", params)
            
            # Verify parameters were included
            call_args = mock_client_instance.post.call_args
            manifest = call_args[1]["json"]
            
            assert "arguments" in manifest["spec"]
            assert "parameters" in manifest["spec"]["arguments"]
            parameters = manifest["spec"]["arguments"]["parameters"]
            
            # Check parameters are included
            param_names = [p["name"] for p in parameters]
            assert "param1" in param_names
            assert "param2" in param_names
