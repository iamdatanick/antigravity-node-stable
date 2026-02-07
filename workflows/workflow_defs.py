"""Hera SDK workflow definitions — replaces flytekit @task/@workflow."""

import os
import logging
import uuid

logger = logging.getLogger("antigravity.workflows")

ARGO_SERVER = os.environ.get("ARGO_SERVER", "k3d:2746")
ARGO_NAMESPACE = os.environ.get("ARGO_NAMESPACE", "argo")


async def submit_workflow(name: str, params: dict) -> str:
    """Submit an Argo workflow via the REST API."""
    import httpx

    run_id = f"{name}-{uuid.uuid4().hex[:8]}"
    workflow_manifest = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {
            "generateName": f"{name}-",
            "namespace": ARGO_NAMESPACE,
        },
        "spec": {
            "entrypoint": name,
            "arguments": {
                "parameters": [
                    {"name": k, "value": str(v)} for k, v in params.items()
                ]
            },
            # Exit handler for closed-loop feedback (Gap #6)
            "onExit": "notify-agent",
            "templates": [
                {
                    "name": name,
                    "container": {
                        "image": "python:3.11-slim",
                        "command": ["python", "-c"],
                        "args": [f"print('Workflow {name} executed with params: {params}')"],
                    },
                },
                {
                    "name": "notify-agent",
                    "http": {
                        "url": "http://antigravity_brain:8080/webhook",
                        "method": "POST",
                        "body": '{"task_id":"{{workflow.name}}","status":"{{workflow.status}}"}',
                    },
                },
            ],
        },
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"http://{ARGO_SERVER}/api/v1/workflows/{ARGO_NAMESPACE}",
                json={"workflow": workflow_manifest},
            )
            if resp.status_code in (200, 201):
                result = resp.json()
                run_id = result.get("metadata", {}).get("name", run_id)
                logger.info(f"Submitted workflow: {run_id}")
            else:
                logger.warning(f"Argo submit returned {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.warning(f"Argo submission failed: {e}. Workflow {run_id} not submitted.")

    return run_id


async def get_workflow_status(run_id: str) -> dict:
    """Get the status of an Argo workflow."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"http://{ARGO_SERVER}/api/v1/workflows/{ARGO_NAMESPACE}/{run_id}"
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "name": data.get("metadata", {}).get("name"),
                    "phase": data.get("status", {}).get("phase", "Unknown"),
                }
    except Exception as e:
        logger.warning(f"Failed to get workflow status: {e}")
    return {"name": run_id, "phase": "Unknown"}
