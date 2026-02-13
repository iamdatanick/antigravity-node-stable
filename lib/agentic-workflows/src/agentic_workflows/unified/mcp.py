"""
Unified MCP Server for Agentic Workflows.

This module provides a unified MCP server that combines:
- Standard MCP (tools, resources, prompts)
- MCP-UI (interactive widgets)
- A2A compatibility layer

Features:
- Single MCP server with full feature support
- Tool registration from all skill formats
- UI resource creation and serving
- A2A protocol bridging
- Session management

Example:
    >>> from agentic_workflows.unified import UnifiedMCPServer
    >>>
    >>> server = UnifiedMCPServer("my-server")
    >>>
    >>> @server.tool("search")
    >>> async def search(query: str) -> str:
    ...     return f"Results for: {query}"
    >>>
    >>> @server.ui_tool("products")
    >>> async def products() -> UIResource:
    ...     return create_list_widget(items)
    >>>
    >>> await server.run()
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Awaitable, TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server import Server
    from mcp.types import Tool, TextContent

logger = logging.getLogger(__name__)


class MCPCapability(Enum):
    """MCP server capabilities."""
    TOOLS = "tools"
    RESOURCES = "resources"
    PROMPTS = "prompts"
    UI = "ui"                # MCP-UI extension
    A2A = "a2a"              # A2A compatibility
    STREAMING = "streaming"
    SUBSCRIPTIONS = "subscriptions"


@dataclass
class MCPToolDefinition:
    """Tool definition for MCP server."""
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]
    is_ui: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPResourceDefinition:
    """Resource definition for MCP server."""
    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"
    handler: Callable[..., Awaitable[Any]] | None = None
    content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPPromptDefinition:
    """Prompt template definition for MCP server."""
    name: str
    description: str
    arguments: list[dict[str, Any]] = field(default_factory=list)
    template: str = ""


@dataclass
class UnifiedMCPConfig:
    """Configuration for unified MCP server."""
    name: str = "unified-mcp-server"
    version: str = "1.0.0"

    # Capabilities
    enable_tools: bool = True
    enable_resources: bool = True
    enable_prompts: bool = True
    enable_ui: bool = True
    enable_a2a: bool = True
    enable_streaming: bool = True

    # Skills integration
    auto_load_skills: bool = True
    skill_domains: list[str] | None = None

    # UI settings
    default_frame_width: int = 600
    default_frame_height: int = 400

    # A2A settings
    a2a_base_url: str = ""
    a2a_organization: str = "Agentic Workflows"


class UnifiedMCPServer:
    """Unified MCP server with full feature support.

    Combines standard MCP, MCP-UI, and A2A compatibility into
    a single server implementation.

    Features:
        - Standard MCP tools, resources, and prompts
        - MCP-UI widget resources
        - A2A protocol compatibility
        - Skill-based tool registration
        - Session management

    Example:
        >>> server = UnifiedMCPServer("my-server")
        >>>
        >>> # Register a simple tool
        >>> @server.tool("greet")
        >>> async def greet(name: str) -> str:
        ...     return f"Hello, {name}!"
        >>>
        >>> # Register a UI tool
        >>> @server.ui_tool("dashboard")
        >>> async def dashboard() -> dict:
        ...     return {"type": "rawHtml", "html": "<h1>Dashboard</h1>"}
        >>>
        >>> # Run the server
        >>> await server.run()
    """

    def __init__(self, config: UnifiedMCPConfig | str | None = None):
        """Initialize unified MCP server.

        Args:
            config: Server configuration or name string.
        """
        if isinstance(config, str):
            self.config = UnifiedMCPConfig(name=config)
        else:
            self.config = config or UnifiedMCPConfig()

        # Storage
        self._tools: dict[str, MCPToolDefinition] = {}
        self._resources: dict[str, MCPResourceDefinition] = {}
        self._prompts: dict[str, MCPPromptDefinition] = {}
        self._ui_resources: dict[str, dict[str, Any]] = {}

        # MCP server instance (lazy)
        self._server: Server | None = None

        # Skill registry (lazy)
        self._skill_registry: Any = None

        # Sessions
        self._sessions: dict[str, dict[str, Any]] = {}

        # Initialize
        self._initialized = False

    def _get_mcp_server(self) -> "Server":
        """Get or create underlying MCP server."""
        if self._server is None:
            try:
                from mcp.server import Server
                self._server = Server(self.config.name)
                self._setup_handlers()
            except ImportError:
                raise ImportError(
                    "MCP SDK not installed. Install with: pip install mcp"
                )
        return self._server

    def _setup_handlers(self) -> None:
        """Set up MCP protocol handlers."""
        server = self._server
        if server is None:
            return

        # List tools handler
        @server.list_tools()
        async def list_tools():
            return self._get_tool_list()

        # Call tool handler
        @server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]):
            return await self._execute_tool(name, arguments)

        # List resources handler (if enabled)
        if self.config.enable_resources:
            @server.list_resources()
            async def list_resources():
                return self._get_resource_list()

            @server.read_resource()
            async def read_resource(uri: str):
                return await self._read_resource(uri)

        # List prompts handler (if enabled)
        if self.config.enable_prompts:
            @server.list_prompts()
            async def list_prompts():
                return self._get_prompt_list()

            @server.get_prompt()
            async def get_prompt(name: str, arguments: dict[str, Any] | None):
                return await self._get_prompt(name, arguments or {})

    async def _ensure_initialized(self) -> None:
        """Ensure server is initialized."""
        if self._initialized:
            return

        # Load skills if configured
        if self.config.auto_load_skills:
            await self._load_skills()

        self._initialized = True

    async def _load_skills(self) -> None:
        """Load skills from unified registry."""
        try:
            from agentic_workflows.unified.skills import get_unified_registry

            registry = get_unified_registry()

            # Get skills as tools
            domains = self.config.skill_domains
            mcp_tools = registry.get_mcp_tools(domains)

            for tool_def in mcp_tools:
                # Create skill invocation handler
                skill_name = tool_def["name"].replace("skill_", "").replace("_", "-")

                async def skill_handler(
                    action: str,
                    context: dict = None,
                    _name: str = skill_name,
                ) -> str:
                    skill_context = registry.load_skill_context(_name)
                    return json.dumps({
                        "skill": _name,
                        "action": action,
                        "context": skill_context[:5000] if skill_context else None,
                    })

                self._tools[tool_def["name"]] = MCPToolDefinition(
                    name=tool_def["name"],
                    description=tool_def.get("description", ""),
                    input_schema=tool_def.get("inputSchema", {}),
                    handler=skill_handler,
                    metadata={"source": "skill", "skill_name": skill_name},
                )

            logger.info(f"Loaded {len(mcp_tools)} skills as MCP tools")

        except Exception as e:
            logger.warning(f"Failed to load skills: {e}")

    def _get_tool_list(self) -> list[dict[str, Any]]:
        """Get list of tools in MCP format."""
        try:
            from mcp.types import Tool

            tools = []
            for tool_def in self._tools.values():
                tools.append(Tool(
                    name=tool_def.name,
                    description=tool_def.description,
                    inputSchema=tool_def.input_schema,
                ))
            return tools

        except ImportError:
            # Return raw dicts if MCP types not available
            return [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                }
                for t in self._tools.values()
            ]

    async def _execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Execute a tool and return results."""
        await self._ensure_initialized()

        tool_def = self._tools.get(name)
        if tool_def is None:
            return [{"type": "text", "text": f"Unknown tool: {name}"}]

        try:
            result = await tool_def.handler(**arguments)

            # Handle UI tools specially
            if tool_def.is_ui:
                return self._format_ui_result(result)

            # Format regular result
            if isinstance(result, str):
                return [{"type": "text", "text": result}]
            elif isinstance(result, dict):
                return [{"type": "text", "text": json.dumps(result, indent=2)}]
            else:
                return [{"type": "text", "text": str(result)}]

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return [{"type": "text", "text": f"Error: {str(e)}"}]

    def _format_ui_result(self, result: Any) -> list[dict[str, Any]]:
        """Format UI tool result."""
        if isinstance(result, dict):
            # Store UI resource
            uri = f"ui://{uuid.uuid4()}"
            self._ui_resources[uri] = result

            return [{
                "type": "resource",
                "resource": {
                    "uri": uri,
                    "mimeType": "application/x-mcp-ui",
                    "text": json.dumps(result),
                },
            }]
        else:
            return [{"type": "text", "text": str(result)}]

    def _get_resource_list(self) -> list[dict[str, Any]]:
        """Get list of resources."""
        try:
            from mcp.types import Resource

            resources = []
            for res_def in self._resources.values():
                resources.append(Resource(
                    uri=res_def.uri,
                    name=res_def.name,
                    description=res_def.description,
                    mimeType=res_def.mime_type,
                ))

            # Add UI resources
            for uri, ui_res in self._ui_resources.items():
                resources.append(Resource(
                    uri=uri,
                    name=ui_res.get("name", "UI Resource"),
                    description="MCP-UI widget resource",
                    mimeType="application/x-mcp-ui",
                ))

            return resources

        except ImportError:
            return [
                {
                    "uri": r.uri,
                    "name": r.name,
                    "description": r.description,
                    "mimeType": r.mime_type,
                }
                for r in self._resources.values()
            ]

    async def _read_resource(self, uri: str) -> dict[str, Any]:
        """Read a resource by URI."""
        # Check UI resources
        if uri in self._ui_resources:
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/x-mcp-ui",
                    "text": json.dumps(self._ui_resources[uri]),
                }]
            }

        # Check standard resources
        res_def = self._resources.get(uri)
        if res_def is None:
            return {"contents": []}

        # Get content
        if res_def.handler:
            content = await res_def.handler()
        else:
            content = res_def.content or ""

        if isinstance(content, dict):
            content = json.dumps(content)

        return {
            "contents": [{
                "uri": uri,
                "mimeType": res_def.mime_type,
                "text": str(content),
            }]
        }

    def _get_prompt_list(self) -> list[dict[str, Any]]:
        """Get list of prompts."""
        try:
            from mcp.types import Prompt, PromptArgument

            prompts = []
            for prompt_def in self._prompts.values():
                prompts.append(Prompt(
                    name=prompt_def.name,
                    description=prompt_def.description,
                    arguments=[
                        PromptArgument(**arg)
                        for arg in prompt_def.arguments
                    ],
                ))
            return prompts

        except ImportError:
            return [
                {
                    "name": p.name,
                    "description": p.description,
                    "arguments": p.arguments,
                }
                for p in self._prompts.values()
            ]

    async def _get_prompt(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Get a prompt with arguments filled in."""
        prompt_def = self._prompts.get(name)
        if prompt_def is None:
            return {"messages": []}

        # Fill in template
        content = prompt_def.template
        for key, value in arguments.items():
            content = content.replace(f"{{{key}}}", str(value))

        return {
            "messages": [{
                "role": "user",
                "content": {"type": "text", "text": content},
            }]
        }

    def tool(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable:
        """Decorator to register a tool.

        Args:
            name: Tool name (defaults to function name).
            description: Tool description (defaults to docstring).

        Returns:
            Decorator function.

        Example:
            @server.tool("greet")
            async def greet(name: str) -> str:
                '''Greet a person.'''
                return f"Hello, {name}!"
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or ""

            # Build input schema from type hints
            import inspect
            sig = inspect.signature(func)
            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue

                param_type = "string"
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == dict:
                        param_type = "object"
                    elif param.annotation == list:
                        param_type = "array"

                properties[param_name] = {"type": param_type}

                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

            input_schema = {
                "type": "object",
                "properties": properties,
            }
            if required:
                input_schema["required"] = required

            self._tools[tool_name] = MCPToolDefinition(
                name=tool_name,
                description=tool_desc.strip(),
                input_schema=input_schema,
                handler=func,
            )

            return func

        return decorator

    def ui_tool(
        self,
        name: str | None = None,
        description: str | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> Callable:
        """Decorator to register a UI tool.

        UI tools return resources that render as interactive widgets.

        Args:
            name: Tool name.
            description: Tool description.
            frame_width: Preferred frame width.
            frame_height: Preferred frame height.

        Returns:
            Decorator function.

        Example:
            @server.ui_tool("dashboard")
            async def dashboard() -> dict:
                return {
                    "type": "rawHtml",
                    "htmlString": "<h1>Dashboard</h1>",
                }
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or "UI widget tool"

            # Wrap handler to include UI metadata
            async def ui_handler(**kwargs) -> dict:
                result = await func(**kwargs)

                # Add UI metadata if not present
                if isinstance(result, dict) and "metadata" not in result:
                    result["metadata"] = {
                        "preferredFrameSize": {
                            "width": frame_width or self.config.default_frame_width,
                            "height": frame_height or self.config.default_frame_height,
                        }
                    }

                return result

            # Build input schema
            import inspect
            sig = inspect.signature(func)
            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                properties[param_name] = {"type": "string"}
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

            input_schema = {
                "type": "object",
                "properties": properties,
            }
            if required:
                input_schema["required"] = required

            self._tools[tool_name] = MCPToolDefinition(
                name=tool_name,
                description=tool_desc.strip(),
                input_schema=input_schema,
                handler=ui_handler,
                is_ui=True,
            )

            return func

        return decorator

    def resource(
        self,
        uri: str,
        name: str | None = None,
        description: str = "",
        mime_type: str = "text/plain",
    ) -> Callable:
        """Decorator to register a resource.

        Args:
            uri: Resource URI.
            name: Resource name.
            description: Resource description.
            mime_type: MIME type.

        Returns:
            Decorator function.
        """
        def decorator(func: Callable) -> Callable:
            res_name = name or func.__name__

            self._resources[uri] = MCPResourceDefinition(
                uri=uri,
                name=res_name,
                description=description or func.__doc__ or "",
                mime_type=mime_type,
                handler=func,
            )

            return func

        return decorator

    def add_resource(
        self,
        uri: str,
        content: str,
        name: str = "",
        description: str = "",
        mime_type: str = "text/plain",
    ) -> None:
        """Add a static resource.

        Args:
            uri: Resource URI.
            content: Resource content.
            name: Resource name.
            description: Resource description.
            mime_type: MIME type.
        """
        self._resources[uri] = MCPResourceDefinition(
            uri=uri,
            name=name or uri,
            description=description,
            mime_type=mime_type,
            content=content,
        )

    def prompt(
        self,
        name: str | None = None,
        description: str = "",
        arguments: list[dict[str, Any]] | None = None,
    ) -> Callable:
        """Decorator to register a prompt template.

        Args:
            name: Prompt name.
            description: Prompt description.
            arguments: Prompt arguments.

        Returns:
            Decorator function.
        """
        def decorator(func: Callable) -> Callable:
            prompt_name = name or func.__name__

            # Get template from function return or docstring
            template = func.__doc__ or ""

            self._prompts[prompt_name] = MCPPromptDefinition(
                name=prompt_name,
                description=description,
                arguments=arguments or [],
                template=template,
            )

            return func

        return decorator

    def add_prompt(
        self,
        name: str,
        template: str,
        description: str = "",
        arguments: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add a prompt template.

        Args:
            name: Prompt name.
            template: Template string with {arg} placeholders.
            description: Prompt description.
            arguments: Argument definitions.
        """
        self._prompts[name] = MCPPromptDefinition(
            name=name,
            description=description,
            arguments=arguments or [],
            template=template,
        )

    def register_skill_tools(
        self,
        domains: list[str] | None = None,
    ) -> int:
        """Register tools from skill registry.

        Args:
            domains: Optional domain filter.

        Returns:
            Number of tools registered.
        """
        try:
            from agentic_workflows.unified.skills import get_unified_registry

            registry = get_unified_registry()
            mcp_tools = registry.get_mcp_tools(domains)

            for tool_def in mcp_tools:
                skill_name = tool_def["name"].replace("skill_", "").replace("_", "-")

                async def handler(
                    action: str = "",
                    context: dict = None,
                    _name: str = skill_name,
                ) -> str:
                    skill_context = registry.load_skill_context(_name)
                    return json.dumps({
                        "skill": _name,
                        "action": action,
                        "context": skill_context[:5000] if skill_context else None,
                    })

                self._tools[tool_def["name"]] = MCPToolDefinition(
                    name=tool_def["name"],
                    description=tool_def.get("description", ""),
                    input_schema=tool_def.get("inputSchema", {}),
                    handler=handler,
                )

            return len(mcp_tools)

        except Exception as e:
            logger.warning(f"Failed to register skill tools: {e}")
            return 0

    def get_a2a_card(self) -> dict[str, Any]:
        """Get A2A agent card for this server.

        Returns:
            A2A AgentCard dictionary.
        """
        skills = []
        for tool_def in self._tools.values():
            skills.append({
                "id": tool_def.name,
                "name": tool_def.name.replace("_", " ").title(),
                "description": tool_def.description,
                "inputModes": ["text"],
                "outputModes": ["text"] if not tool_def.is_ui else ["text", "ui"],
            })

        return {
            "name": self.config.name,
            "description": f"Unified MCP Server: {self.config.name}",
            "url": self.config.a2a_base_url,
            "version": self.config.version,
            "protocolVersion": "0.3",
            "provider": {
                "organization": self.config.a2a_organization,
            },
            "capabilities": {
                "streaming": self.config.enable_streaming,
                "tasks": True,
                "ui": self.config.enable_ui,
            },
            "skills": skills,
            "defaultInputModes": ["text"],
            "defaultOutputModes": ["text"],
        }

    async def run(self) -> None:
        """Run the MCP server using stdio transport."""
        try:
            from mcp.server.stdio import stdio_server
            from mcp.server.models import InitializationOptions
            from mcp.server.lowlevel.server import NotificationOptions

            server = self._get_mcp_server()
            await self._ensure_initialized()

            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name=self.config.name,
                        server_version=self.config.version,
                        capabilities=server.get_capabilities(
                            notification_options=NotificationOptions(
                                prompts_changed=self.config.enable_prompts,
                                resources_changed=self.config.enable_resources,
                                tools_changed=True,
                            ),
                            experimental_capabilities={},
                        ),
                    ),
                )
        except ImportError:
            raise ImportError(
                "MCP SDK not installed. Install with: pip install mcp"
            )

    def get_stats(self) -> dict[str, Any]:
        """Get server statistics."""
        return {
            "name": self.config.name,
            "version": self.config.version,
            "tools_count": len(self._tools),
            "resources_count": len(self._resources),
            "prompts_count": len(self._prompts),
            "ui_resources_count": len(self._ui_resources),
            "sessions_count": len(self._sessions),
            "capabilities": {
                "tools": self.config.enable_tools,
                "resources": self.config.enable_resources,
                "prompts": self.config.enable_prompts,
                "ui": self.config.enable_ui,
                "a2a": self.config.enable_a2a,
                "streaming": self.config.enable_streaming,
            },
        }


def create_unified_mcp_server(
    name: str = "unified-mcp-server",
    version: str = "1.0.0",
    auto_load_skills: bool = True,
    enable_ui: bool = True,
    enable_a2a: bool = True,
    **kwargs,
) -> UnifiedMCPServer:
    """Factory function to create a unified MCP server.

    Args:
        name: Server name.
        version: Server version.
        auto_load_skills: Auto-load skills from registry.
        enable_ui: Enable MCP-UI support.
        enable_a2a: Enable A2A compatibility.
        **kwargs: Additional configuration options.

    Returns:
        Configured UnifiedMCPServer.

    Example:
        >>> server = create_unified_mcp_server("my-server")
        >>>
        >>> @server.tool("hello")
        >>> async def hello(name: str = "World") -> str:
        ...     return f"Hello, {name}!"
        >>>
        >>> await server.run()
    """
    config = UnifiedMCPConfig(
        name=name,
        version=version,
        auto_load_skills=auto_load_skills,
        enable_ui=enable_ui,
        enable_a2a=enable_a2a,
        **kwargs,
    )
    return UnifiedMCPServer(config)
