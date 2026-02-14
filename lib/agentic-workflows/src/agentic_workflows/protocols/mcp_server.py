"""MCP Server for Agentic Workflows.

Exposes skills registry as MCP tools for Claude Code integration.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.lowlevel.server import NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
)

from agentic_workflows.skills import (
    SkillRegistry,
    discover_default_skills,
    get_registry,
)
from agentic_workflows.tools import ToolSearchTool, create_tool_search_config

logger = logging.getLogger(__name__)


class AgenticWorkflowsMCPServer:
    """MCP Server that exposes agentic workflows skills as tools."""

    def __init__(self, name: str = "agentic-workflows"):
        self.name = name
        self.server = Server(name)
        self._registry: SkillRegistry | None = None
        self._tool_search: ToolSearchTool | None = None
        self._initialized = False

        # Register handlers
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup MCP protocol handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List all available tools from skills registry."""
            await self._ensure_initialized()
            return self._get_tools()

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Execute a tool by name."""
            await self._ensure_initialized()
            result = await self._execute_tool(name, arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _ensure_initialized(self):
        """Lazy initialization of skills registry."""
        if self._initialized:
            return

        # Discover skills from default locations
        count = discover_default_skills()
        logger.info(f"Discovered {count} skills")

        self._registry = get_registry()
        self._tool_search = ToolSearchTool(create_tool_search_config())
        self._initialized = True

    def _get_tools(self) -> list[Tool]:
        """Get all tools from skills registry."""
        tools = []

        # Add tool search tool
        tools.append(
            Tool(
                name="skill_search",
                description="Search for skills by keyword or category. Use this to find relevant skills before invoking them.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (keyword or regex pattern)",
                        },
                        "category": {
                            "type": "string",
                            "description": "Filter by category (code, data, ops, research, writing, business, creative, communication, specialized, meta, phuc)",
                            "enum": [
                                "code",
                                "data",
                                "ops",
                                "research",
                                "writing",
                                "business",
                                "creative",
                                "communication",
                                "specialized",
                                "meta",
                                "phuc",
                            ],
                        },
                    },
                    "required": ["query"],
                },
            )
        )

        # Add skill invocation tool
        tools.append(
            Tool(
                name="skill_invoke",
                description="Invoke a skill by name with provided context. The skill will return guidance for the requested task.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Name of the skill to invoke",
                        },
                        "context": {
                            "type": "string",
                            "description": "Context or query to pass to the skill",
                        },
                    },
                    "required": ["skill_name"],
                },
            )
        )

        # Add skill list tool
        tools.append(
            Tool(
                name="skill_list",
                description="List all available skills grouped by domain.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "description": "Filter by domain (phuc, lifesci, devops, agentic-workflows)",
                        }
                    },
                },
            )
        )

        # Add registered skills as individual tools (deferred loading)
        if self._registry:
            for name, skill in self._registry._skills.items():
                if skill.defer_loading:
                    # Deferred skills get added with minimal schema
                    tools.append(
                        Tool(
                            name=f"skill_{name}",
                            description=skill.description[:200]
                            if skill.description
                            else f"Invoke {name} skill",
                            inputSchema={
                                "type": "object",
                                "properties": {
                                    "context": {"type": "string", "description": "Task context"}
                                },
                            },
                        )
                    )

        return tools

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return results."""

        if name == "skill_search":
            query = arguments.get("query", "")
            category = arguments.get("category")

            # Search directly in registry since ToolSearchTool needs the registry
            results = []
            import re

            pattern = re.compile(query, re.IGNORECASE)

            for skill_name, skill in self._registry._skills.items():
                if category and skill.domain != category:
                    continue

                if pattern.search(skill_name) or (
                    skill.description and pattern.search(skill.description)
                ):
                    results.append(
                        {
                            "name": skill_name,
                            "level": skill.level,
                            "domain": skill.domain,
                            "description": skill.description[:100] if skill.description else "",
                        }
                    )

            return {
                "status": "success",
                "count": len(results),
                "results": results[:10],  # Limit to 10
            }

        elif name == "skill_invoke":
            skill_name = arguments.get("skill_name", "")
            context = arguments.get("context", "")

            if not self._registry:
                return {"status": "error", "message": "Registry not initialized"}

            skill = self._registry.get_skill(skill_name)
            if not skill:
                return {"status": "error", "message": f"Skill '{skill_name}' not found"}

            # Load full skill context
            skill_context = self._registry.load_skill_context(skill_name)
            return {
                "status": "success",
                "skill": skill_name,
                "level": skill.level,
                "domain": skill.domain,
                "allowed_tools": skill.allowed_tools,
                "context": skill_context[:5000] if skill_context else None,  # Truncate for safety
            }

        elif name == "skill_list":
            domain_filter = arguments.get("domain")

            if not self._registry:
                return {"status": "error", "message": "Registry not initialized"}

            skills_by_domain: dict[str, list[dict]] = {}
            for skill_name, skill in self._registry._skills.items():
                if domain_filter and skill.domain != domain_filter:
                    continue

                if skill.domain not in skills_by_domain:
                    skills_by_domain[skill.domain] = []

                skills_by_domain[skill.domain].append(
                    {
                        "name": skill_name,
                        "level": skill.level,
                        "description": skill.description[:100] if skill.description else "",
                    }
                )

            return {
                "status": "success",
                "domains": list(skills_by_domain.keys()),
                "total_count": sum(len(v) for v in skills_by_domain.values()),
                "skills": skills_by_domain,
            }

        elif name.startswith("skill_"):
            # Direct skill invocation
            skill_name = name[6:]  # Remove "skill_" prefix
            context = arguments.get("context", "")

            if not self._registry:
                return {"status": "error", "message": "Registry not initialized"}

            skill = self._registry.get_skill(skill_name)
            if not skill:
                return {"status": "error", "message": f"Skill '{skill_name}' not found"}

            skill_context = self._registry.load_skill_context(skill_name)
            return {
                "status": "success",
                "skill": skill_name,
                "context": skill_context[:5000] if skill_context else None,
            }

        return {"status": "error", "message": f"Unknown tool: {name}"}

    async def run(self):
        """Run the MCP server using stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name=self.name,
                    server_version="5.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(
                            prompts_changed=False, resources_changed=False, tools_changed=True
                        ),
                        experimental_capabilities={},
                    ),
                ),
            )


async def serve():
    """Main entry point for MCP server."""
    logging.basicConfig(level=logging.INFO)
    server = AgenticWorkflowsMCPServer()
    await server.run()


def main():
    """Synchronous entry point."""
    asyncio.run(serve())


if __name__ == "__main__":
    main()
