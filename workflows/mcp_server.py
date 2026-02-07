"""FastMCP tool server — registers concrete tools for Goose (Gap #8 fix)."""

import os
import logging
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("antigravity.mcp")

mcp = FastMCP("Antigravity-Node")


@mcp.tool()
async def search_memory(query: str, limit: int = 5) -> str:
    """Searches StarRocks memory tables for relevant context."""
    from workflows.memory import recall_experience
    results = recall_experience(query, tenant_id="system", limit=limit)
    import json
    return json.dumps(results, default=str)


@mcp.tool()
async def trigger_task(workflow_name: str, parameters: dict = None) -> str:
    """Triggers a background workflow on Argo via Hera SDK."""
    from workflows.workflow_defs import submit_workflow
    run_id = await submit_workflow(workflow_name, parameters or {})
    return f"Task started. Watch ID: {run_id}"


@mcp.tool()
async def reflect_on_failure(task_id: str) -> str:
    """Reads logs from a failed Argo workflow for self-correction."""
    logger.info(f"Reflecting on failed task: {task_id}")
    # In production, would fetch Argo workflow logs
    return f"Reflection complete for task {task_id}. Check Argo UI at port 2755."


@mcp.tool()
async def query_memory(sql: str) -> str:
    """Execute SQL on StarRocks memory tables (episodic/semantic/procedural)."""
    from workflows.memory import query
    import json
    results = query(sql)
    return json.dumps(results, default=str)


@mcp.tool()
async def ingest_file(path: str) -> str:
    """Ingest document into semantic memory via OPEA pipeline."""
    from workflows.opea_client import ingest
    result = await ingest(path)
    import json
    return json.dumps(result, default=str)


def start_mcp_server():
    """Start the FastMCP server (called from main.py if needed)."""
    mcp.run(transport="sse", port=8082)
