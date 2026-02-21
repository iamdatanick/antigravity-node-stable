"""Model Context Protocol (MCP) client implementation.

Implements MCP specification 2025-11-25.
Reference: https://modelcontextprotocol.io/specification/2025-11-25
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

# Try to import the official MCP SDK
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.types import Tool as MCPToolType

    MCP_SDK_AVAILABLE = True
except ImportError:
    MCP_SDK_AVAILABLE = False


class MCPTransport(Enum):
    """MCP transport types."""

    SSE = "sse"
    STDIO = "stdio"
    HTTP = "http"  # Legacy/fallback


@dataclass
class MCPServerConfig:
    """MCP server configuration."""

    name: str = "default"
    url: str | None = None  # For SSE transport
    command: str | None = None  # For stdio transport
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    transport: MCPTransport = MCPTransport.SSE
    timeout: float = 30.0

# Backwards compatibility
MCPConfig = MCPServerConfig


@dataclass
class MCPTool:
    """An MCP tool definition."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_claude_tool(self) -> dict[str, Any]:
        """Convert to Claude tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class MCPResource:
    """An MCP resource."""

    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPPrompt:
    """An MCP prompt template."""

    name: str
    description: str = ""
    arguments: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPTask:
    """An MCP task (2025-11-25 spec feature)."""

    id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float = 0.0
    result: Any = None
    error: str | None = None


class MCPClient:
    """Client for Model Context Protocol servers.

    Supports:
    - SSE transport (primary, for remote servers)
    - stdio transport (for local processes)
    - HTTP transport (legacy fallback)
    - Tool discovery and execution
    - Resource access
    - Prompt templates
    - Task tracking (2025-11-25 spec)
    """

    PROTOCOL_VERSION = "2025-11-25"

    def __init__(
        self,
        config: MCPServerConfig,
        on_message: Callable[[dict[str, Any]], None] | None = None,
    ):
        """Initialize MCP client.

        Args:
            config: Server configuration.
            on_message: Callback for received messages.
        """
        self.config = config
        self.on_message = on_message

        self._tools: dict[str, MCPTool] = {}
        self._resources: dict[str, MCPResource] = {}
        self._prompts: dict[str, MCPPrompt] = {}
        self._tasks: dict[str, MCPTask] = {}
        self._initialized = False
        self._server_info: dict[str, Any] = {}

        # Session management
        self._session: ClientSession | None = None
        self._context_manager = None

        # HTTP client for fallback
        self._http_client: httpx.AsyncClient | None = None

    async def connect(self) -> bool:
        """Connect to MCP server.

        Returns:
            True if connected successfully.
        """
        if self.config.transport == MCPTransport.SSE:
            return await self._connect_sse()
        elif self.config.transport == MCPTransport.STDIO:
            return await self._connect_stdio()
        else:
            return await self._connect_http()

    async def _connect_sse(self) -> bool:
        """Connect via SSE transport."""
        if not MCP_SDK_AVAILABLE:
            raise ImportError("MCP SDK required for SSE transport. Install with: pip install mcp")

        if not self.config.url:
            raise ValueError("URL required for SSE transport")

        try:
            # Use the MCP SDK's SSE client
            self._context_manager = sse_client(self.config.url)
            read, write = await self._context_manager.__aenter__()

            self._session = ClientSession(read, write)
            await self._session.__aenter__()

            # Initialize the session
            result = await self._session.initialize()
            self._server_info = {
                "name": result.serverInfo.name if result.serverInfo else "unknown",
                "version": result.serverInfo.version if result.serverInfo else "unknown",
                "capabilities": result.capabilities.__dict__ if result.capabilities else {},
            }
            self._initialized = True

            # Discover available tools
            await self.refresh_tools()

            return True

        except Exception as e:
            self._initialized = False
            raise ConnectionError(f"Failed to connect to MCP server: {e}") from e

    async def _connect_stdio(self) -> bool:
        """Connect via stdio transport."""
        if not MCP_SDK_AVAILABLE:
            raise ImportError("MCP SDK required for stdio transport. Install with: pip install mcp")

        if not self.config.command:
            raise ValueError("Command required for stdio transport")

        try:
            self._context_manager = stdio_client(
                self.config.command,
                self.config.args,
                env=self.config.env or None,
            )
            read, write = await self._context_manager.__aenter__()

            self._session = ClientSession(read, write)
            await self._session.__aenter__()

            result = await self._session.initialize()
            self._server_info = {
                "name": result.serverInfo.name if result.serverInfo else "unknown",
                "version": result.serverInfo.version if result.serverInfo else "unknown",
            }
            self._initialized = True

            await self.refresh_tools()

            return True

        except Exception as e:
            self._initialized = False
            raise ConnectionError(f"Failed to connect to MCP server: {e}") from e

    async def _connect_http(self) -> bool:
        """Connect via HTTP transport (legacy fallback)."""
        if not self.config.url:
            raise ValueError("URL required for HTTP transport")

        self._http_client = httpx.AsyncClient(timeout=self.config.timeout)

        try:
            # Send initialize request
            response = await self._http_client.post(
                self.config.url,
                json={
                    "jsonrpc": "2.0",
                    "id": str(uuid.uuid4()),
                    "method": "initialize",
                    "params": {
                        "protocolVersion": self.PROTOCOL_VERSION,
                        "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                        "clientInfo": {"name": "agentic-workflows", "version": "2.0.0"},
                    },
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            result = response.json()
            if "error" in result:
                raise ConnectionError(f"Initialize failed: {result['error']}")

            self._server_info = result.get("result", {})
            self._initialized = True

            await self.refresh_tools()

            return True

        except Exception as e:
            self._initialized = False
            raise ConnectionError(f"Failed to connect to MCP server: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from server."""
        if self._session:
            await self._session.__aexit__(None, None, None)
            self._session = None

        if self._context_manager:
            await self._context_manager.__aexit__(None, None, None)
            self._context_manager = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._initialized = False
        self._tools.clear()
        self._resources.clear()
        self._prompts.clear()

    async def refresh_tools(self) -> list[MCPTool]:
        """Refresh available tools from server.

        Returns:
            List of available tools.
        """
        if self._session and MCP_SDK_AVAILABLE:
            result = await self._session.list_tools()
            tools = []
            for tool in result.tools:
                mcp_tool = MCPTool(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema if hasattr(tool, "inputSchema") else {},
                    metadata={},
                )
                tools.append(mcp_tool)
                self._tools[mcp_tool.name] = mcp_tool
            return tools

        elif self._http_client:
            response = await self._http_client.post(
                self.config.url,
                json={
                    "jsonrpc": "2.0",
                    "id": str(uuid.uuid4()),
                    "method": "tools/list",
                    "params": {},
                },
            )
            result = response.json()

            if "error" in result:
                return []

            tools = []
            for tool_data in result.get("result", {}).get("tools", []):
                tool = MCPTool(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {}),
                )
                tools.append(tool)
                self._tools[tool.name] = tool

            return tools

        return []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Call a tool.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            Tool result.

        Raises:
            ValueError: If tool not found.
            RuntimeError: If tool call fails.
        """
        if name not in self._tools:
            # Try refreshing tools first
            await self.refresh_tools()
            if name not in self._tools:
                raise ValueError(f"Tool not found: {name}")

        if self._session and MCP_SDK_AVAILABLE:
            result = await self._session.call_tool(name, arguments or {})
            # Extract content from result
            if hasattr(result, "content") and result.content:
                contents = []
                for item in result.content:
                    if hasattr(item, "text"):
                        contents.append(item.text)
                    elif hasattr(item, "data"):
                        contents.append(item.data)
                return "\n".join(str(c) for c in contents) if contents else result

            return result

        elif self._http_client:
            response = await self._http_client.post(
                self.config.url,
                json={
                    "jsonrpc": "2.0",
                    "id": str(uuid.uuid4()),
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments or {}},
                },
            )
            result = response.json()

            if "error" in result:
                raise RuntimeError(
                    f"Tool call failed: {result['error'].get('message', 'Unknown error')}"
                )

            return result.get("result")

        raise RuntimeError("Not connected to MCP server")

    async def list_resources(self) -> list[MCPResource]:
        """List available resources.

        Returns:
            List of resources.
        """
        if self._session and MCP_SDK_AVAILABLE:
            result = await self._session.list_resources()
            resources = []
            for res in result.resources:
                resource = MCPResource(
                    uri=res.uri,
                    name=res.name or res.uri,
                    description=res.description or "",
                    mime_type=res.mimeType or "text/plain",
                )
                resources.append(resource)
                self._resources[resource.uri] = resource
            return resources

        elif self._http_client:
            response = await self._http_client.post(
                self.config.url,
                json={
                    "jsonrpc": "2.0",
                    "id": str(uuid.uuid4()),
                    "method": "resources/list",
                    "params": {},
                },
            )
            result = response.json()

            if "error" in result:
                return []

            resources = []
            for res_data in result.get("result", {}).get("resources", []):
                resource = MCPResource(
                    uri=res_data["uri"],
                    name=res_data.get("name", res_data["uri"]),
                    description=res_data.get("description", ""),
                    mime_type=res_data.get("mimeType", "text/plain"),
                )
                resources.append(resource)
                self._resources[resource.uri] = resource

            return resources

        return []

    async def read_resource(self, uri: str) -> Any:
        """Read a resource.

        Args:
            uri: Resource URI.

        Returns:
            Resource content.
        """
        if self._session and MCP_SDK_AVAILABLE:
            result = await self._session.read_resource(uri)
            if hasattr(result, "contents") and result.contents:
                contents = []
                for item in result.contents:
                    if hasattr(item, "text"):
                        contents.append(item.text)
                    elif hasattr(item, "blob"):
                        contents.append(item.blob)
                return "\n".join(str(c) for c in contents) if contents else result
            return result

        elif self._http_client:
            response = await self._http_client.post(
                self.config.url,
                json={
                    "jsonrpc": "2.0",
                    "id": str(uuid.uuid4()),
                    "method": "resources/read",
                    "params": {"uri": uri},
                },
            )
            result = response.json()

            if "error" in result:
                raise RuntimeError(f"Resource read failed: {result['error'].get('message')}")

            return result.get("result")

        raise RuntimeError("Not connected to MCP server")

    async def list_prompts(self) -> list[MCPPrompt]:
        """List available prompts.

        Returns:
            List of prompts.
        """
        if self._session and MCP_SDK_AVAILABLE:
            result = await self._session.list_prompts()
            prompts = []
            for prompt in result.prompts:
                mcp_prompt = MCPPrompt(
                    name=prompt.name,
                    description=prompt.description or "",
                    arguments=prompt.arguments or [],
                )
                prompts.append(mcp_prompt)
                self._prompts[mcp_prompt.name] = mcp_prompt
            return prompts

        elif self._http_client:
            response = await self._http_client.post(
                self.config.url,
                json={
                    "jsonrpc": "2.0",
                    "id": str(uuid.uuid4()),
                    "method": "prompts/list",
                    "params": {},
                },
            )
            result = response.json()

            if "error" in result:
                return []

            prompts = []
            for prompt_data in result.get("result", {}).get("prompts", []):
                prompt = MCPPrompt(
                    name=prompt_data["name"],
                    description=prompt_data.get("description", ""),
                    arguments=prompt_data.get("arguments", []),
                )
                prompts.append(prompt)
                self._prompts[prompt.name] = prompt

            return prompts

        return []

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Get a prompt with arguments filled in.

        Args:
            name: Prompt name.
            arguments: Prompt arguments.

        Returns:
            Rendered prompt.
        """
        if self._session and MCP_SDK_AVAILABLE:
            result = await self._session.get_prompt(name, arguments or {})
            return result

        elif self._http_client:
            response = await self._http_client.post(
                self.config.url,
                json={
                    "jsonrpc": "2.0",
                    "id": str(uuid.uuid4()),
                    "method": "prompts/get",
                    "params": {"name": name, "arguments": arguments or {}},
                },
            )
            result = response.json()

            if "error" in result:
                raise RuntimeError(f"Get prompt failed: {result['error'].get('message')}")

            return result.get("result")

        raise RuntimeError("Not connected to MCP server")

    def get_tool(self, name: str) -> MCPTool | None:
        """Get tool by name."""
        return self._tools.get(name)

    def get_all_tools(self) -> list[MCPTool]:
        """Get all available tools."""
        return list(self._tools.values())

    def get_tools_for_claude(self) -> list[dict[str, Any]]:
        """Get tools in Claude API format."""
        return [tool.to_claude_tool() for tool in self._tools.values()]

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._initialized

    @property
    def server_info(self) -> dict[str, Any]:
        """Get server information."""
        return self._server_info

    async def __aenter__(self) -> MCPClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()


class MCPRegistry:
    """Registry for managing multiple MCP server connections."""

    def __init__(self):
        self._clients: dict[str, MCPClient] = {}

    async def register(self, config: MCPServerConfig) -> MCPClient:
        """Register and connect to an MCP server.

        Args:
            config: Server configuration.

        Returns:
            Connected client.
        """
        client = MCPClient(config)
        await client.connect()
        self._clients[config.name] = client
        return client

    async def unregister(self, name: str) -> bool:
        """Unregister and disconnect from an MCP server.

        Args:
            name: Server name.

        Returns:
            True if unregistered.
        """
        if name in self._clients:
            await self._clients[name].disconnect()
            del self._clients[name]
            return True
        return False

    def get_client(self, name: str) -> MCPClient | None:
        """Get client by server name."""
        return self._clients.get(name)

    def get_all_tools(self) -> dict[str, list[MCPTool]]:
        """Get all tools from all servers."""
        return {name: client.get_all_tools() for name, client in self._clients.items()}

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on a specific server.

        Args:
            server_name: Server name.
            tool_name: Tool name.
            arguments: Tool arguments.

        Returns:
            Tool result.
        """
        client = self._clients.get(server_name)
        if not client:
            raise ValueError(f"Unknown server: {server_name}")
        return await client.call_tool(tool_name, arguments)

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for client in self._clients.values():
            await client.disconnect()
        self._clients.clear()

    @property
    def servers(self) -> list[str]:
        """Get list of registered server names."""
        return list(self._clients.keys())
