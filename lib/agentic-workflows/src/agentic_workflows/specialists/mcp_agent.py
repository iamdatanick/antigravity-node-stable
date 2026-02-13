"""MCP specialist agent for Model Context Protocol operations.

Handles MCP server management, tool discovery, and protocol operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import SpecialistAgent, SpecialistConfig, SpecialistCapability
from ..protocols.mcp_client import MCPClient, MCPConfig, MCPTransport, MCPRegistry


@dataclass
class MCPSpecialistConfig(SpecialistConfig):
    """MCP specialist-specific configuration."""

    default_servers: list[dict[str, Any]] = field(default_factory=list)
    auto_discover_tools: bool = True


class MCPSpecialistAgent(SpecialistAgent):
    """Specialist agent for MCP protocol operations.

    Capabilities:
    - MCP server management
    - Tool discovery and execution
    - Resource access
    - Protocol operations
    """

    def __init__(self, config: MCPSpecialistConfig | None = None, **kwargs):
        self.mcp_config = config or MCPSpecialistConfig()
        super().__init__(config=self.mcp_config, **kwargs)

        self._registry = MCPRegistry()
        self._clients: dict[str, MCPClient] = {}

        self.register_handler("connect_server", self._connect_server)
        self.register_handler("disconnect_server", self._disconnect_server)
        self.register_handler("list_servers", self._list_servers)
        self.register_handler("list_tools", self._list_tools)
        self.register_handler("call_tool", self._call_tool)
        self.register_handler("list_resources", self._list_resources)
        self.register_handler("read_resource", self._read_resource)

    @property
    def capabilities(self) -> list[SpecialistCapability]:
        return [
            SpecialistCapability.MCP_PROTOCOL,
            SpecialistCapability.TOOL_INTEGRATION,
        ]

    @property
    def service_name(self) -> str:
        return "Model Context Protocol"

    async def _connect(self) -> None:
        """Initialize MCP connections."""
        for server_config in self.mcp_config.default_servers:
            try:
                await self._connect_server(
                    name=server_config.get("name", "default"),
                    endpoint=server_config.get("endpoint", ""),
                    transport=server_config.get("transport", "sse"),
                )
            except Exception as e:
                self.logger.warning(f"Failed to connect to MCP server: {e}")

    async def _disconnect(self) -> None:
        """Disconnect all MCP servers."""
        for name in list(self._clients.keys()):
            await self._disconnect_server(name)

    async def _health_check(self) -> bool:
        """Check MCP health."""
        return len(self._clients) > 0

    async def _connect_server(
        self,
        name: str,
        endpoint: str,
        transport: str = "sse",
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Connect to an MCP server.

        Args:
            name: Server name.
            endpoint: Server endpoint.
            transport: Transport type (sse, stdio, http).
            headers: Optional HTTP headers.

        Returns:
            Connection result.
        """
        if name in self._clients:
            return {"name": name, "connected": True, "exists": True}

        transport_type = MCPTransport(transport)
        config = MCPConfig(
            endpoint=endpoint,
            transport=transport_type,
            headers=headers or {},
        )

        client = MCPClient(config)
        connected = await client.connect()

        if connected:
            self._clients[name] = client
            self._registry.register(name, client)
            return {
                "name": name,
                "connected": True,
                "tools": len(client.available_tools),
                "resources": len(client.available_resources),
            }
        else:
            return {"name": name, "connected": False, "error": "Connection failed"}

    async def _disconnect_server(self, name: str) -> dict[str, Any]:
        """Disconnect from an MCP server.

        Args:
            name: Server name.

        Returns:
            Disconnection result.
        """
        if name not in self._clients:
            return {"name": name, "disconnected": False, "error": "Not found"}

        client = self._clients.pop(name)
        self._registry.unregister(name)
        await client.disconnect()

        return {"name": name, "disconnected": True}

    async def _list_servers(self) -> list[dict[str, Any]]:
        """List connected MCP servers.

        Returns:
            List of server info.
        """
        servers = []
        for name, client in self._clients.items():
            servers.append({
                "name": name,
                "connected": client._connected,
                "tools": len(client.available_tools),
                "resources": len(client.available_resources),
                "endpoint": client.config.endpoint,
            })
        return servers

    async def _list_tools(self, server: str | None = None) -> list[dict[str, Any]]:
        """List available tools.

        Args:
            server: Optional server filter.

        Returns:
            List of tools.
        """
        if server:
            if server not in self._clients:
                return []
            return self._clients[server].available_tools

        # Aggregate from all servers
        all_tools = []
        for name, client in self._clients.items():
            for tool in client.available_tools:
                tool_info = dict(tool)
                tool_info["server"] = name
                all_tools.append(tool_info)
        return all_tools

    async def _call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        server: str | None = None,
    ) -> dict[str, Any]:
        """Call an MCP tool.

        Args:
            name: Tool name.
            arguments: Tool arguments.
            server: Specific server to use.

        Returns:
            Tool result.
        """
        if server:
            if server not in self._clients:
                return {"error": f"Server not found: {server}"}
            client = self._clients[server]
        else:
            # Find server with this tool
            client = None
            for c in self._clients.values():
                if any(t.get("name") == name for t in c.available_tools):
                    client = c
                    break
            if not client:
                return {"error": f"Tool not found: {name}"}

        try:
            result = await client.call_tool(name, arguments or {})
            return {"result": result, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _list_resources(self, server: str | None = None) -> list[dict[str, Any]]:
        """List available resources.

        Args:
            server: Optional server filter.

        Returns:
            List of resources.
        """
        if server:
            if server not in self._clients:
                return []
            return self._clients[server].available_resources

        # Aggregate from all servers
        all_resources = []
        for name, client in self._clients.items():
            for resource in client.available_resources:
                resource_info = dict(resource)
                resource_info["server"] = name
                all_resources.append(resource_info)
        return all_resources

    async def _read_resource(
        self,
        uri: str,
        server: str | None = None,
    ) -> dict[str, Any]:
        """Read a resource.

        Args:
            uri: Resource URI.
            server: Specific server to use.

        Returns:
            Resource content.
        """
        if server:
            if server not in self._clients:
                return {"error": f"Server not found: {server}"}
            client = self._clients[server]
        else:
            # Find server with this resource
            client = None
            for c in self._clients.values():
                if any(r.get("uri") == uri for r in c.available_resources):
                    client = c
                    break
            if not client:
                return {"error": f"Resource not found: {uri}"}

        try:
            result = await client.read_resource(uri)
            return {"content": result, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    def get_tools_for_claude(self) -> list[dict[str, Any]]:
        """Get all tools in Claude format.

        Returns:
            Tools formatted for Claude API.
        """
        return self._registry.get_all_tools_for_claude()
