"""
Metrics Collector MCP Server.

Exposes metrics collection and cost tracking as MCP tools.

Usage:
    python -m agentic_workflows.observability.mcp_metrics

Tools provided:
- record_usage: Record token usage for an agent call
- get_summary: Get usage summary with cost breakdown
- get_agent_usage: Get usage for a specific agent
- get_remaining_budget: Check remaining budget
- check_budget: Check if budget is exceeded
- reset_metrics: Reset all metrics
- export_metrics: Export all usage records
"""

import asyncio
import json
import sys
from typing import Any

# Try to import MCP SDK
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        TextContent,
        Tool,
    )

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)

from agentic_workflows.observability.metrics import (
    MetricsCollector,
    Model,
)

# Global metrics collector instance
_metrics: MetricsCollector | None = None


def get_metrics(
    budget_usd: float | None = None, budget_tokens: int | None = None
) -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector(
            budget_usd=budget_usd,
            budget_tokens=budget_tokens,
        )
    return _metrics


def create_server() -> "Server":
    """Create and configure the metrics MCP server."""
    if not MCP_AVAILABLE:
        raise ImportError("MCP SDK required. Install with: pip install mcp")

    server = Server("metrics-collector")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List all available tools."""
        return [
            Tool(
                name="record_usage",
                description="Record token usage for an agent API call.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string", "description": "Agent identifier"},
                        "model": {
                            "type": "string",
                            "enum": ["opus", "sonnet", "haiku"],
                            "description": "Model used (opus, sonnet, haiku)",
                        },
                        "input_tokens": {
                            "type": "integer",
                            "description": "Number of input tokens",
                        },
                        "output_tokens": {
                            "type": "integer",
                            "description": "Number of output tokens",
                        },
                        "metadata": {"type": "object", "description": "Optional metadata"},
                    },
                    "required": ["agent_id", "model", "input_tokens", "output_tokens"],
                },
            ),
            Tool(
                name="get_summary",
                description="Get usage summary with cost breakdown by model and agent.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "start_time": {
                            "type": "number",
                            "description": "Filter start timestamp (Unix epoch)",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Filter end timestamp (Unix epoch)",
                        },
                    },
                },
            ),
            Tool(
                name="get_agent_usage",
                description="Get usage statistics for a specific agent.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string", "description": "Agent identifier"}
                    },
                    "required": ["agent_id"],
                },
            ),
            Tool(
                name="get_remaining_budget",
                description="Get remaining budget information (cost and tokens).",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="check_budget",
                description="Check if budget limits have been exceeded.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="set_budget",
                description="Set budget limits for cost and/or tokens.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "budget_usd": {"type": "number", "description": "Maximum budget in USD"},
                        "budget_tokens": {"type": "integer", "description": "Maximum token budget"},
                    },
                },
            ),
            Tool(
                name="reset_metrics",
                description="Reset all collected metrics. Use with caution.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "confirm": {
                            "type": "boolean",
                            "description": "Must be true to confirm reset",
                        }
                    },
                    "required": ["confirm"],
                },
            ),
            Tool(
                name="export_metrics",
                description="Export all usage records as JSON.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": ["json", "summary"],
                            "description": "Export format (json for raw data, summary for formatted)",
                        }
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle tool calls."""
        try:
            result = await _handle_tool(name, arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": str(e),
                            "tool": name,
                        },
                        indent=2,
                    ),
                )
            ]

    return server


async def _handle_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle individual tool calls."""
    metrics = get_metrics()

    if name == "record_usage":
        agent_id = arguments.get("agent_id", "")
        model_str = arguments.get("model", "sonnet").lower()
        input_tokens = arguments.get("input_tokens", 0)
        output_tokens = arguments.get("output_tokens", 0)
        metadata = arguments.get("metadata", {})

        # Map string to Model enum
        model_map = {
            "opus": Model.OPUS,
            "sonnet": Model.SONNET,
            "haiku": Model.HAIKU,
        }
        model = model_map.get(model_str, Model.SONNET)

        usage = metrics.record(
            agent_id=agent_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata=metadata,
        )

        return {
            "recorded": True,
            "agent_id": agent_id,
            "model": model.value,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": usage.cost_usd,
            "timestamp": usage.timestamp,
        }

    elif name == "get_summary":
        start_time = arguments.get("start_time")
        end_time = arguments.get("end_time")

        summary = metrics.get_summary(start_time=start_time, end_time=end_time)

        return {
            "total_cost_usd": summary.total_cost_usd,
            "total_input_tokens": summary.total_input_tokens,
            "total_output_tokens": summary.total_output_tokens,
            "total_tokens": summary.total_input_tokens + summary.total_output_tokens,
            "calls_count": summary.calls_count,
            "by_model": summary.by_model,
            "by_agent": summary.by_agent,
            "time_range_seconds": summary.time_range_seconds,
        }

    elif name == "get_agent_usage":
        agent_id = arguments.get("agent_id", "")
        return metrics.get_agent_usage(agent_id)

    elif name == "get_remaining_budget":
        return metrics.get_remaining_budget()

    elif name == "check_budget":
        exceeded, reason = metrics.is_budget_exceeded()
        return {
            "exceeded": exceeded,
            "reason": reason if reason else "Budget not exceeded",
            "budget": metrics.get_remaining_budget(),
        }

    elif name == "set_budget":
        budget_usd = arguments.get("budget_usd")
        budget_tokens = arguments.get("budget_tokens")

        if budget_usd is not None:
            metrics.budget_usd = budget_usd
        if budget_tokens is not None:
            metrics.budget_tokens = budget_tokens

        return {
            "budget_set": True,
            "budget_usd": metrics.budget_usd,
            "budget_tokens": metrics.budget_tokens,
        }

    elif name == "reset_metrics":
        confirm = arguments.get("confirm", False)
        if not confirm:
            return {
                "reset": False,
                "error": "Must confirm reset by setting confirm=true",
            }

        metrics.reset()
        return {
            "reset": True,
            "message": "All metrics have been reset",
        }

    elif name == "export_metrics":
        export_format = arguments.get("format", "json")

        if export_format == "summary":
            return {
                "format": "summary",
                "content": metrics.format_summary(),
            }
        else:
            return {
                "format": "json",
                "records": metrics.export(),
            }

    else:
        return {
            "error": f"Unknown tool: {name}",
            "available_tools": [
                "record_usage",
                "get_summary",
                "get_agent_usage",
                "get_remaining_budget",
                "check_budget",
                "set_budget",
                "reset_metrics",
                "export_metrics",
            ],
        }


async def main():
    """Run the metrics MCP server."""
    if not MCP_AVAILABLE:
        print("Error: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)

    server = create_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
