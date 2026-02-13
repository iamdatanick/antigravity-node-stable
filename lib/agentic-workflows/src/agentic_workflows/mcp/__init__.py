"""MCP Server Registry for Agentic Workflows.

Provides auto-discovery and management of MCP servers with OAuth 2.1 support.

Usage:
    from agentic_workflows.mcp import MCPServerRegistry, MCPServerConfig

    registry = MCPServerRegistry()
    await registry.load_from_config("~/.claude/.mcp.json")
    server = registry.get("filesystem")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class MCPTransport(Enum):
    """MCP server transport types."""

    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"


@dataclass
class MCPStdioServerConfig:
    """Configuration for stdio-based MCP server."""

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None


@dataclass
class MCPSSEServerConfig:
    """Configuration for SSE-based MCP server."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class MCPHttpServerConfig:
    """Configuration for HTTP-based MCP server."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    method: str = "POST"


@dataclass
class OAuthTokens:
    """OAuth 2.1 token storage."""

    access_token: str
    refresh_token: str | None = None
    expires_at: float | None = None
    token_type: str = "Bearer"

    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        if self.expires_at is None:
            return False
        return time.time() >= self.expires_at

    def to_header(self) -> str:
        """Get Authorization header value."""
        return f"{self.token_type} {self.access_token}"


@dataclass
class MCPServerConfig:
    """Unified MCP server configuration."""

    name: str
    transport: MCPTransport = MCPTransport.STDIO

    # Stdio config
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None

    # HTTP/SSE config
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    # OAuth 2.1
    oauth: dict[str, Any] | None = None
    tokens: OAuthTokens | None = None

    # Identity verification
    identity: dict[str, Any] | None = None

    # Metadata
    description: str = ""
    enabled: bool = True

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> MCPServerConfig:
        """Create config from dictionary."""
        # Determine transport
        transport = MCPTransport.STDIO
        if "url" in data:
            if data.get("transport") == "sse":
                transport = MCPTransport.SSE
            else:
                transport = MCPTransport.HTTP

        return cls(
            name=name,
            transport=transport,
            command=data.get("command"),
            args=data.get("args", []),
            env=data.get("env", {}),
            cwd=data.get("cwd"),
            url=data.get("url"),
            headers=data.get("headers", {}),
            oauth=data.get("oauth"),
            identity=data.get("identity"),
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
        )


# Built-in MCP servers with common configurations
BUILTIN_SERVERS: dict[str, dict[str, Any]] = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem"],
        "description": "Local filesystem access",
    },
    "github": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"},
        "description": "GitHub API integration",
    },
    "slack": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-slack"],
        "env": {"SLACK_TOKEN": "${SLACK_TOKEN}"},
        "description": "Slack workspace integration",
    },
    "google-drive": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-gdrive"],
        "description": "Google Drive access",
    },
    "postgres": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-postgres"],
        "env": {"DATABASE_URL": "${DATABASE_URL}"},
        "description": "PostgreSQL database access",
    },
    "puppeteer": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
        "description": "Browser automation with Puppeteer",
    },
    "brave-search": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env": {"BRAVE_API_KEY": "${BRAVE_API_KEY}"},
        "description": "Brave Search API",
    },
    "memory": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
        "description": "Persistent memory storage",
    },
}


@dataclass
class MCPServerInstance:
    """Running MCP server instance."""

    config: MCPServerConfig
    process: subprocess.Popen | None = None
    connected: bool = False
    error: str | None = None
    tools: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)


class MCPServerRegistry:
    """Registry for discovering and managing MCP servers.

    Example:
        registry = MCPServerRegistry()

        # Load from config file
        await registry.load_from_config("~/.claude/.mcp.json")

        # Get built-in server
        fs = registry.get("filesystem")

        # Start a server
        await registry.start("filesystem")

        # List available tools
        tools = registry.list_tools()
    """

    def __init__(self):
        """Initialize registry."""
        self._servers: dict[str, MCPServerInstance] = {}
        self._config_path: Path | None = None

    async def load_from_config(self, path: str) -> int:
        """Load servers from mcp.json config file.

        Args:
            path: Path to mcp.json or directory containing it.

        Returns:
            Number of servers loaded.
        """
        config_path = Path(path).expanduser()

        # If directory, look for .mcp.json
        if config_path.is_dir():
            config_path = config_path / ".mcp.json"

        if not config_path.exists():
            logger.warning(f"MCP config not found: {config_path}")
            return 0

        self._config_path = config_path

        try:
            with open(config_path) as f:
                data = json.load(f)

            servers = data.get("mcpServers", {})
            for name, config in servers.items():
                self.register(MCPServerConfig.from_dict(name, config))

            logger.info(f"Loaded {len(servers)} MCP servers from {config_path}")
            return len(servers)

        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            return 0

    def register(self, config: MCPServerConfig) -> None:
        """Register an MCP server configuration.

        Args:
            config: Server configuration.
        """
        self._servers[config.name] = MCPServerInstance(config=config)
        logger.debug(f"Registered MCP server: {config.name}")

    def register_builtin(self, name: str) -> bool:
        """Register a built-in server.

        Args:
            name: Name of built-in server.

        Returns:
            True if registered successfully.
        """
        if name not in BUILTIN_SERVERS:
            logger.error(f"Unknown built-in server: {name}")
            return False

        config = MCPServerConfig.from_dict(name, BUILTIN_SERVERS[name])
        self.register(config)
        return True

    def get(self, name: str) -> MCPServerInstance | None:
        """Get a server by name.

        Args:
            name: Server name.

        Returns:
            Server instance or None.
        """
        return self._servers.get(name)

    def list_servers(self) -> list[str]:
        """Get list of registered server names.

        Returns:
            List of server names.
        """
        return list(self._servers.keys())

    def list_builtin(self) -> list[str]:
        """Get list of available built-in servers.

        Returns:
            List of built-in server names.
        """
        return list(BUILTIN_SERVERS.keys())

    async def start(self, name: str) -> bool:
        """Start an MCP server.

        Args:
            name: Server name to start.

        Returns:
            True if started successfully.
        """
        instance = self._servers.get(name)
        if not instance:
            logger.error(f"Server not found: {name}")
            return False

        if instance.connected:
            return True

        config = instance.config
        if config.transport != MCPTransport.STDIO:
            # HTTP/SSE servers don't need to be started
            instance.connected = True
            return True

        if not config.command:
            logger.error(f"No command for server: {name}")
            return False

        try:
            # Expand environment variables
            env = os.environ.copy()
            for key, value in config.env.items():
                if value.startswith("${") and value.endswith("}"):
                    var_name = value[2:-1]
                    env[key] = os.environ.get(var_name, "")
                else:
                    env[key] = value

            # Start process
            cmd = [config.command] + config.args
            instance.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=config.cwd,
            )
            instance.connected = True
            logger.info(f"Started MCP server: {name}")
            return True

        except Exception as e:
            instance.error = str(e)
            logger.error(f"Failed to start server {name}: {e}")
            return False

    async def stop(self, name: str) -> bool:
        """Stop an MCP server.

        Args:
            name: Server name to stop.

        Returns:
            True if stopped successfully.
        """
        instance = self._servers.get(name)
        if not instance or not instance.connected:
            return False

        if instance.process:
            instance.process.terminate()
            try:
                instance.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                instance.process.kill()
            instance.process = None

        instance.connected = False
        logger.info(f"Stopped MCP server: {name}")
        return True

    async def refresh_oauth_token(self, name: str) -> bool:
        """Refresh OAuth 2.1 token for a server.

        Args:
            name: Server name.

        Returns:
            True if refreshed successfully.
        """
        instance = self._servers.get(name)
        if not instance:
            return False

        config = instance.config
        if not config.oauth or not config.tokens:
            return False

        if not config.tokens.refresh_token:
            logger.warning(f"No refresh token for server: {name}")
            return False

        try:
            import httpx

            token_url = config.oauth.get("token_url")
            client_id = config.oauth.get("client_id")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    token_url,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": config.tokens.refresh_token,
                        "client_id": client_id,
                    },
                )
                response.raise_for_status()
                data = response.json()

                config.tokens = OAuthTokens(
                    access_token=data["access_token"],
                    refresh_token=data.get("refresh_token", config.tokens.refresh_token),
                    expires_at=time.time() + data.get("expires_in", 3600),
                    token_type=data.get("token_type", "Bearer"),
                )
                return True

        except Exception as e:
            logger.error(f"Failed to refresh token for {name}: {e}")
            return False

    def list_tools(self) -> dict[str, list[str]]:
        """Get tools from all connected servers.

        Returns:
            Dict mapping server name to list of tool names.
        """
        return {
            name: instance.tools for name, instance in self._servers.items() if instance.connected
        }

    async def shutdown(self) -> None:
        """Stop all running servers."""
        for name in list(self._servers.keys()):
            await self.stop(name)


# Import from submodules
from agentic_workflows.mcp.auth import (
    APIKeyAuth,
    AuthConfig,
    AuthType,
    BasicAuth,
    OAuth21Client,
    OAuth21Config,
    PKCEChallenge,
    TokenSet,
)
from agentic_workflows.mcp.prompts import (
    BUILTIN_PROMPTS,
    Prompt,
    PromptArgument,
    PromptHandler,
    PromptManager,
    PromptMessage,
)
from agentic_workflows.mcp.resources import (
    Resource,
    ResourceContent,
    ResourceHandler,
    ResourceManager,
    ResourceTemplate,
)
from agentic_workflows.mcp.sampling import (
    ModelPreferences,
    SamplingHandler,
    SamplingManager,
    SamplingMessage,
    SamplingRequest,
    SamplingResult,
    StopReason,
)
from agentic_workflows.mcp.transports import (
    BaseTransport,
    HTTPTransport,
    MCPMessage,
    SSETransport,
    StdioTransport,
)

__all__ = [
    # Registry
    "MCPTransport",
    "MCPStdioServerConfig",
    "MCPSSEServerConfig",
    "MCPHttpServerConfig",
    "MCPServerConfig",
    "MCPServerInstance",
    "MCPServerRegistry",
    "OAuthTokens",
    "BUILTIN_SERVERS",
    # Transports
    "MCPMessage",
    "BaseTransport",
    "StdioTransport",
    "SSETransport",
    "HTTPTransport",
    # Auth
    "AuthType",
    "PKCEChallenge",
    "TokenSet",
    "OAuth21Config",
    "OAuth21Client",
    "APIKeyAuth",
    "BasicAuth",
    "AuthConfig",
    # Resources
    "Resource",
    "ResourceContent",
    "ResourceTemplate",
    "ResourceHandler",
    "ResourceManager",
    # Prompts
    "Prompt",
    "PromptArgument",
    "PromptMessage",
    "PromptHandler",
    "PromptManager",
    "BUILTIN_PROMPTS",
    # Sampling
    "StopReason",
    "SamplingMessage",
    "ModelPreferences",
    "SamplingRequest",
    "SamplingResult",
    "SamplingHandler",
    "SamplingManager",
]
