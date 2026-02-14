"""Advanced Tool Use features for Agentic Workflows v5.0.

Implements:
- Tool Search Tool wrapper (85% token reduction)
- Programmatic Tool Calling executor
- Defer loading configuration
- Beta header management
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Beta headers for advanced features
BETA_HEADERS = [
    "advanced-tool-use-2025-11-20",
    "interleaved-thinking-2025-05-14",
    "mcp-client-2025-11-20",
    "skills-2025-10-02",
    "structured-outputs-2025-11-13",
]


@dataclass
class ToolSearchConfig:
    """Configuration for Tool Search Tool."""

    enabled: bool = True
    defer_loading: bool = True
    index_path: Path = None
    always_loaded_tools: list[str] = field(default_factory=list)
    search_regex_enabled: bool = True
    max_results: int = 10


class ToolSearchTool:
    """
    Tool Search Tool implementation for reduced token usage.

    Allows Claude to discover tools on-demand instead of loading
    all tool definitions upfront. Reduces token usage by ~85%.
    """

    def __init__(self, config: ToolSearchConfig, skill_registry=None):
        self.config = config
        self.skill_registry = skill_registry
        self._tool_index: dict[str, dict] = {}
        self._load_index()

    def _load_index(self):
        """Load tool index from file or registry."""
        if self.config.index_path and self.config.index_path.exists():
            with open(self.config.index_path) as f:
                data = json.load(f)
                self._tool_index = self._flatten_index(data)
        elif self.skill_registry:
            for name, skill in self.skill_registry.skills.items():
                self._tool_index[name] = {
                    "name": name,
                    "description": skill.description,
                    "defer_loading": name not in self.config.always_loaded_tools,
                }

    def _flatten_index(self, data: dict) -> dict[str, dict]:
        """Flatten nested index structure."""
        flat = {}
        if "agents" in data:
            for category, agents in data["agents"].items():
                for agent in agents:
                    flat[agent["name"]] = {
                        "name": agent["name"],
                        "description": agent.get("desc", ""),
                        "category": category,
                        "model": agent.get("model", "sonnet"),
                        "defer_loading": True,
                    }
        if "skills" in data:
            for category, skills in data["skills"].items():
                for skill in skills:
                    flat[skill["name"]] = {
                        "name": skill["name"],
                        "description": skill.get("desc", ""),
                        "category": category,
                        "defer_loading": True,
                    }
        return flat

    def search(self, query: str, category: str = None) -> list[dict]:
        """
        Search for tools matching the query.

        Args:
            query: Search term or regex pattern
            category: Optional category filter

        Returns:
            List of matching tool definitions
        """
        results = []
        pattern = re.compile(query, re.IGNORECASE) if self.config.search_regex_enabled else None

        for name, tool in self._tool_index.items():
            if category and tool.get("category") != category:
                continue

            if pattern:
                if pattern.search(name) or pattern.search(tool.get("description", "")):
                    results.append(tool)
            else:
                if (
                    query.lower() in name.lower()
                    or query.lower() in tool.get("description", "").lower()
                ):
                    results.append(tool)

        return results[: self.config.max_results]

    def get_tool_definition(self) -> dict:
        """Get the Tool Search Tool definition for Claude API."""
        return {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}

    def get_all_tool_definitions(self) -> list[dict]:
        """Get all tool definitions with defer_loading flags."""
        tools = [self.get_tool_definition()]

        for name, tool in self._tool_index.items():
            defer = tool.get("defer_loading", True)
            if name in self.config.always_loaded_tools:
                defer = False

            tools.append(
                {
                    "name": f"skill_{name}",
                    "description": tool.get("description", ""),
                    "defer_loading": defer,
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "description": "Action to perform"},
                            "params": {"type": "object", "description": "Action parameters"},
                        },
                        "required": ["action"],
                    },
                }
            )

        return tools


class ProgrammaticToolCalling:
    """
    Programmatic Tool Calling executor.

    Allows Claude to orchestrate tools through code instead of
    API round-trips. Reduces inference passes for multi-tool workflows.
    """

    def __init__(self):
        self._tool_handlers: dict[str, Callable] = {}

    def register_handler(self, name: str, handler: Callable):
        """Register a tool handler function."""
        self._tool_handlers[name] = handler

    async def execute(self, code: str, context: dict = None) -> Any:
        """
        Execute programmatic tool calling code.

        Args:
            code: Python code that orchestrates tool calls
            context: Optional execution context

        Returns:
            Result of code execution
        """
        # Create sandboxed execution environment
        local_vars = {"tools": self._tool_handlers, "context": context or {}, "results": {}}

        # Add async support
        import asyncio

        local_vars["asyncio"] = asyncio

        # Execute code
        exec(code, {"__builtins__": {}}, local_vars)

        return local_vars.get("results", {})


def get_beta_headers() -> dict[str, str]:
    """Get beta headers for API calls."""
    return {"anthropic-beta": ",".join(BETA_HEADERS)}


def create_tool_search_config(
    index_path: str = None, always_loaded: list[str] = None
) -> ToolSearchConfig:
    """Factory function to create ToolSearchConfig."""
    return ToolSearchConfig(
        enabled=True,
        defer_loading=True,
        index_path=Path(index_path) if index_path else None,
        always_loaded_tools=always_loaded
        or ["cloudflare-d1", "pharma-npi-ndc", "analytics-attribution"],
        search_regex_enabled=True,
        max_results=10,
    )
