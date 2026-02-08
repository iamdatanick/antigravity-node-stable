"""Tests for workflow_defs.py Argo workflow submission."""

import pytest
from unittest.mock import AsyncMock, patch
import respx
import httpx


class TestWorkflowManifest:
    """Test workflow manifest structure."""

    async def test_workflow_manifest_structure(self):
        """Test that workflow manifest has correct structure with templates."""
        from workflows.workflow_defs import submit_workflow

        # Mock the Argo API using respx - use pattern to match both hostnames
        with respx.mock:
            route = respx.post(url__regex=r"http://.+:\d+/api/v1/workflows/argo").mock(
                return_value=httpx.Response(200, json={"metadata": {"name": "test-workflow-abc123"}})
            )

            # Call submit_workflow
            await submit_workflow("test-workflow", {"param1": "value1"})

            # Verify the request was made
            assert route.called

            # Check the manifest structure
            request = route.calls[0].request
            manifest = request.content

            # The manifest should be valid JSON
            import json

            manifest_json = json.loads(manifest)
            assert "workflow" in manifest_json
            workflow = manifest_json["workflow"]
            assert "apiVersion" in workflow
            assert workflow["apiVersion"] == "argoproj.io/v1alpha1"
            assert "kind" in workflow
            assert workflow["kind"] == "Workflow"
            assert "spec" in workflow
            assert "templates" in workflow["spec"]

            # Should have both main template and exit handler
            templates = workflow["spec"]["templates"]
            assert isinstance(templates, list)
            assert len(templates) >= 2  # At least main template and notify-agent


class TestSubmitWorkflow:
    """Test workflow submission."""

    async def test_submit_workflow(self):
        """Test submit_workflow posts manifest to Argo API."""
        from workflows.workflow_defs import submit_workflow

        with respx.mock:
            route = respx.post(url__regex=r"http://.+:\d+/api/v1/workflows/argo").mock(
                return_value=httpx.Response(200, json={"metadata": {"name": "test-workflow-abc123"}})
            )

            # Call submit_workflow
            result = await submit_workflow("test-workflow", {"param1": "value1"})

            # Verify post was called
            assert route.called
            assert "test-workflow" in result

    async def test_submit_workflow_with_params(self):
        """Test submit_workflow includes parameters in manifest."""
        from workflows.workflow_defs import submit_workflow

        with respx.mock:
            route = respx.post(url__regex=r"http://.+:\d+/api/v1/workflows/argo").mock(
                return_value=httpx.Response(200, json={"metadata": {"name": "test-workflow-abc123"}})
            )

            # Call with parameters
            params = {"param1": "value1", "param2": "value2"}
            await submit_workflow("test-workflow", params)

            # Verify parameters were included
            assert route.called
            request = route.calls[0].request
            import json

            manifest_json = json.loads(request.content)
            workflow = manifest_json["workflow"]

            assert "arguments" in workflow["spec"]
            assert "parameters" in workflow["spec"]["arguments"]
            parameters = workflow["spec"]["arguments"]["parameters"]

            # Check parameters are included
            param_names = [p["name"] for p in parameters]
            assert "param1" in param_names
            assert "param2" in param_names
