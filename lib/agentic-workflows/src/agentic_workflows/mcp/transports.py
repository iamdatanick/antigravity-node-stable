"""MCP Transport Implementations.

Provides stdio, SSE, and HTTP transport implementations for MCP protocol.

Usage:
    from agentic_workflows.mcp.transports import StdioTransport, SSETransport, HTTPTransport

    transport = StdioTransport(command="npx", args=["-y", "@mcp/server"])
    await transport.connect()
    response = await transport.send({"method": "tools/list"})
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MCPMessage:
    """MCP JSON-RPC message."""

    jsonrpc: str = "2.0"
    id: str | int | None = None
    method: str | None = None
    params: dict[str, Any] | None = None
    result: Any | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        msg = {"jsonrpc": self.jsonrpc}
        if self.id is not None:
            msg["id"] = self.id
        if self.method:
            msg["method"] = self.method
        if self.params:
            msg["params"] = self.params
        if self.result is not None:
            msg["result"] = self.result
        if self.error:
            msg["error"] = self.error
        return msg

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPMessage:
        """Create from dictionary."""
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            method=data.get("method"),
            params=data.get("params"),
            result=data.get("result"),
            error=data.get("error"),
        )

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, data: str) -> MCPMessage:
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(data))


class BaseTransport(ABC):
    """Abstract base class for MCP transports."""

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    async def send(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Send message and wait for response."""
        pass

    @abstractmethod
    async def receive(self) -> AsyncIterator[dict[str, Any]]:
        """Receive messages (for notifications)."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected."""
        pass


class StdioTransport(BaseTransport):
    """Stdio-based MCP transport.

    Communicates with MCP servers via stdin/stdout using newline-delimited JSON.

    Example:
        transport = StdioTransport(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem"],
        )
        await transport.connect()
        response = await transport.send({"method": "tools/list", "id": 1})
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ):
        """Initialize stdio transport.

        Args:
            command: Command to execute.
            args: Command arguments.
            env: Environment variables.
            cwd: Working directory.
        """
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.cwd = cwd
        self._process: asyncio.subprocess.Process | None = None
        self._message_id = 0
        self._pending: dict[str | int, asyncio.Future] = {}
        self._read_task: asyncio.Task | None = None

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._process is not None and self._process.returncode is None

    async def connect(self) -> bool:
        """Start the MCP server process."""
        try:
            import os

            # Prepare environment
            full_env = os.environ.copy()
            for key, value in self.env.items():
                if value.startswith("${") and value.endswith("}"):
                    var_name = value[2:-1]
                    full_env[key] = os.environ.get(var_name, "")
                else:
                    full_env[key] = value

            # Start process
            cmd = [self.command] + self.args
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=full_env,
                cwd=self.cwd,
            )

            # Start reader task
            self._read_task = asyncio.create_task(self._read_loop())
            logger.info(f"Connected to MCP server: {self.command}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    async def disconnect(self) -> None:
        """Stop the MCP server process."""
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
            self._read_task = None

        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
            self._process = None

        # Cancel pending futures
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()

    async def send(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Send message and wait for response.

        Args:
            message: Message to send.

        Returns:
            Response message or None on error.
        """
        if not self.is_connected or not self._process or not self._process.stdin:
            return None

        # Assign message ID if needed
        if "id" not in message and message.get("method"):
            self._message_id += 1
            message["id"] = self._message_id

        # Create future for response
        msg_id = message.get("id")
        if msg_id is not None:
            future: asyncio.Future = asyncio.get_event_loop().create_future()
            self._pending[msg_id] = future

        try:
            # Send message
            data = json.dumps(message) + "\n"
            self._process.stdin.write(data.encode())
            await self._process.stdin.drain()

            # Wait for response if we have an ID
            if msg_id is not None:
                response = await asyncio.wait_for(future, timeout=30.0)
                return response

            return None

        except asyncio.TimeoutError:
            if msg_id in self._pending:
                del self._pending[msg_id]
            logger.error(f"Timeout waiting for response to message {msg_id}")
            return None
        except Exception as e:
            if msg_id in self._pending:
                del self._pending[msg_id]
            logger.error(f"Error sending message: {e}")
            return None

    async def _read_loop(self) -> None:
        """Read messages from stdout."""
        if not self._process or not self._process.stdout:
            return

        try:
            while self.is_connected:
                line = await self._process.stdout.readline()
                if not line:
                    break

                try:
                    message = json.loads(line.decode())
                    msg_id = message.get("id")

                    if msg_id is not None and msg_id in self._pending:
                        self._pending[msg_id].set_result(message)
                        del self._pending[msg_id]
                    else:
                        # Handle notification
                        logger.debug(f"Received notification: {message}")

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from server: {e}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in read loop: {e}")

    async def receive(self) -> AsyncIterator[dict[str, Any]]:
        """Receive messages (notifications)."""
        # For stdio, notifications come through the read loop
        # This is a placeholder for external consumers
        while self.is_connected:
            await asyncio.sleep(0.1)
            yield {}


class SSETransport(BaseTransport):
    """Server-Sent Events (SSE) transport for MCP.

    Example:
        transport = SSETransport(url="http://localhost:3000/mcp/sse")
        await transport.connect()
        async for event in transport.receive():
            print(event)
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        auth_token: str | None = None,
    ):
        """Initialize SSE transport.

        Args:
            url: SSE endpoint URL.
            headers: Additional headers.
            auth_token: Bearer token for authentication.
        """
        self.url = url
        self.headers = headers or {}
        self.auth_token = auth_token
        self._connected = False
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._listen_task: asyncio.Task | None = None

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    async def connect(self) -> bool:
        """Connect to SSE endpoint."""
        try:
            self._connected = True
            self._listen_task = asyncio.create_task(self._listen_loop())
            logger.info(f"Connected to SSE endpoint: {self.url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to SSE: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from SSE endpoint."""
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None
        self._connected = False

    async def _listen_loop(self) -> None:
        """Listen for SSE events."""
        try:
            import httpx

            headers = self.headers.copy()
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            async with httpx.AsyncClient() as client:
                async with client.stream("GET", self.url, headers=headers) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            data = line[5:].strip()
                            try:
                                event = json.loads(data)
                                await self._event_queue.put(event)
                            except json.JSONDecodeError:
                                pass

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"SSE error: {e}")
            self._connected = False

    async def send(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Send message via POST (SSE is receive-only, POST for requests)."""
        try:
            import httpx

            headers = {"Content-Type": "application/json", **self.headers}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            # Construct POST URL from SSE URL
            post_url = self.url.replace("/sse", "/rpc")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    post_url,
                    json=message,
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return None

    async def receive(self) -> AsyncIterator[dict[str, Any]]:
        """Receive SSE events."""
        while self._connected:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0,
                )
                yield event
            except asyncio.TimeoutError:
                continue


class HTTPTransport(BaseTransport):
    """HTTP transport for MCP (request/response model).

    Example:
        transport = HTTPTransport(url="http://localhost:3000/mcp")
        await transport.connect()
        response = await transport.send({"method": "tools/list", "id": 1})
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        auth_token: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize HTTP transport.

        Args:
            url: HTTP endpoint URL.
            headers: Additional headers.
            auth_token: Bearer token for authentication.
            timeout: Request timeout in seconds.
        """
        self.url = url
        self.headers = headers or {}
        self.auth_token = auth_token
        self.timeout = timeout
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    async def connect(self) -> bool:
        """Mark as connected (HTTP is stateless)."""
        self._connected = True
        return True

    async def disconnect(self) -> None:
        """Mark as disconnected."""
        self._connected = False

    async def send(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Send HTTP request and get response.

        Args:
            message: JSON-RPC message.

        Returns:
            Response message or None on error.
        """
        try:
            import httpx

            headers = {"Content-Type": "application/json", **self.headers}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.url,
                    json=message,
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"HTTP request failed: {e}")
            return None

    async def receive(self) -> AsyncIterator[dict[str, Any]]:
        """HTTP is request/response only - no streaming receive."""
        # HTTP doesn't support server-push
        return
        yield  # Make this a generator


__all__ = [
    "MCPMessage",
    "BaseTransport",
    "StdioTransport",
    "SSETransport",
    "HTTPTransport",
]
