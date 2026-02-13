"""A2A Protocol Client Implementation.

Provides a client for communicating with A2A-compliant agents.
Supports streaming, polling, and multiple transport protocols.

Reference: https://github.com/a2aproject/a2a-python
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, TypeVar, Generic

import httpx

from agentic_workflows.a2a.a2a_types import (
    AgentCard,
    AgentSkill,
    Task,
    TaskState,
    TaskStatus,
    Message,
    MessageRole,
    TextPart,
    Artifact,
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCError,
    A2AErrorCode,
    TransportProtocol,
    StreamEvent,
    TaskStatusEvent,
    MessageEvent,
    ArtifactEvent,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ClientConfig:
    """Configuration for A2A client.

    Attributes:
        base_url: Base URL for the A2A agent.
        timeout: Request timeout in seconds.
        streaming: Enable streaming responses.
        polling_interval: Interval for status polling in seconds.
        max_retries: Maximum retry attempts for failed requests.
        retry_delay: Delay between retries in seconds.
        transports: Preferred transport protocols in order.
        headers: Additional headers to include in requests.
    """

    base_url: str = ""
    """Base URL for the A2A agent."""

    timeout: float = 60.0
    """Request timeout in seconds."""

    streaming: bool = True
    """Enable streaming responses."""

    polling_interval: float = 1.0
    """Interval for status polling in seconds."""

    max_retries: int = 3
    """Maximum retry attempts for failed requests."""

    retry_delay: float = 1.0
    """Delay between retries in seconds."""

    transports: list[TransportProtocol] = field(
        default_factory=lambda: [TransportProtocol.JSONRPC]
    )
    """Preferred transport protocols in order."""

    headers: dict[str, str] = field(default_factory=dict)
    """Additional headers to include in requests."""

    verify_ssl: bool = True
    """Verify SSL certificates."""


# =============================================================================
# Middleware / Interceptors
# =============================================================================


class ClientCallInterceptor(ABC):
    """Abstract base for client call interceptors.

    Interceptors can modify requests before sending and responses
    after receiving. Use for logging, authentication, metrics, etc.
    """

    @abstractmethod
    async def intercept_request(
        self,
        request: JSONRPCRequest,
        context: dict[str, Any],
    ) -> JSONRPCRequest:
        """Intercept and optionally modify outgoing request.

        Args:
            request: The outgoing request.
            context: Request context (url, headers, etc.).

        Returns:
            Modified or original request.
        """
        pass

    @abstractmethod
    async def intercept_response(
        self,
        response: JSONRPCResponse,
        context: dict[str, Any],
    ) -> JSONRPCResponse:
        """Intercept and optionally modify incoming response.

        Args:
            response: The incoming response.
            context: Response context.

        Returns:
            Modified or original response.
        """
        pass


class LoggingInterceptor(ClientCallInterceptor):
    """Interceptor that logs all requests and responses."""

    def __init__(self, log_level: int = logging.DEBUG):
        self.log_level = log_level

    async def intercept_request(
        self,
        request: JSONRPCRequest,
        context: dict[str, Any],
    ) -> JSONRPCRequest:
        """Log outgoing request."""
        logger.log(
            self.log_level,
            f"A2A Request: {request.method} -> {context.get('url', 'unknown')}",
        )
        return request

    async def intercept_response(
        self,
        response: JSONRPCResponse,
        context: dict[str, Any],
    ) -> JSONRPCResponse:
        """Log incoming response."""
        status = "error" if response.is_error else "success"
        logger.log(
            self.log_level,
            f"A2A Response: {status} (id={response.id})",
        )
        return response


class AuthInterceptor(ClientCallInterceptor):
    """Interceptor that adds authentication headers."""

    def __init__(
        self,
        token: str | None = None,
        api_key: str | None = None,
        api_key_header: str = "X-API-Key",
    ):
        self.token = token
        self.api_key = api_key
        self.api_key_header = api_key_header

    async def intercept_request(
        self,
        request: JSONRPCRequest,
        context: dict[str, Any],
    ) -> JSONRPCRequest:
        """Add auth headers to request context."""
        headers = context.setdefault("headers", {})

        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        if self.api_key:
            headers[self.api_key_header] = self.api_key

        return request

    async def intercept_response(
        self,
        response: JSONRPCResponse,
        context: dict[str, Any],
    ) -> JSONRPCResponse:
        """Pass through response unchanged."""
        return response


class RetryInterceptor(ClientCallInterceptor):
    """Interceptor that handles retries on failure."""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_codes: set[int] | None = None,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_codes = retry_codes or {
            A2AErrorCode.INTERNAL_ERROR.value,
        }

    async def intercept_request(
        self,
        request: JSONRPCRequest,
        context: dict[str, Any],
    ) -> JSONRPCRequest:
        """Track retry count in context."""
        context.setdefault("retry_count", 0)
        return request

    async def intercept_response(
        self,
        response: JSONRPCResponse,
        context: dict[str, Any],
    ) -> JSONRPCResponse:
        """Check if retry is needed."""
        if response.error and response.error.code in self.retry_codes:
            retry_count = context.get("retry_count", 0)
            if retry_count < self.max_retries:
                context["should_retry"] = True
                context["retry_delay"] = self.retry_delay * (retry_count + 1)

        return response


# =============================================================================
# Event Consumers
# =============================================================================


class EventConsumer(ABC):
    """Abstract base for event consumers.

    Event consumers process streaming events from A2A servers.
    """

    @abstractmethod
    async def on_event(self, event: StreamEvent) -> None:
        """Handle a streaming event.

        Args:
            event: The streaming event to process.
        """
        pass


class CallbackEventConsumer(EventConsumer):
    """Event consumer that delegates to callbacks."""

    def __init__(
        self,
        on_message: Callable[[Message], None] | None = None,
        on_status: Callable[[TaskStatus], None] | None = None,
        on_artifact: Callable[[Artifact], None] | None = None,
    ):
        self.on_message = on_message
        self.on_status = on_status
        self.on_artifact = on_artifact

    async def on_event(self, event: StreamEvent) -> None:
        """Dispatch event to appropriate callback."""
        if isinstance(event, MessageEvent) and self.on_message:
            self.on_message(event.message)
        elif isinstance(event, TaskStatusEvent) and self.on_status:
            self.on_status(event.status)
        elif isinstance(event, ArtifactEvent) and self.on_artifact:
            self.on_artifact(event.artifact)


class QueueEventConsumer(EventConsumer):
    """Event consumer that puts events in an async queue."""

    def __init__(self, queue: asyncio.Queue[StreamEvent] | None = None):
        self.queue = queue or asyncio.Queue()

    async def on_event(self, event: StreamEvent) -> None:
        """Put event in queue."""
        await self.queue.put(event)


# =============================================================================
# A2A Client
# =============================================================================


class A2AClient:
    """Client for A2A (Agent-to-Agent) protocol.

    Provides methods for:
    - Agent discovery via Agent Cards
    - Task submission and tracking
    - Streaming and polling for responses
    - Session management for conversations

    Example:
        ```python
        async with A2AClient(ClientConfig(base_url="https://agent.example.com")) as client:
            card = await client.get_card()
            print(f"Connected to: {card.name}")

            task = await client.send_message(Message.user("Hello!"))
            print(f"Response: {task.agent_response}")
        ```
    """

    def __init__(
        self,
        config: ClientConfig | None = None,
        interceptors: list[ClientCallInterceptor] | None = None,
        consumers: list[EventConsumer] | None = None,
    ):
        """Initialize A2A client.

        Args:
            config: Client configuration.
            interceptors: Request/response interceptors.
            consumers: Event consumers for streaming.
        """
        self.config = config or ClientConfig()
        self.interceptors = interceptors or []
        self.consumers = consumers or []

        self._http_client: httpx.AsyncClient | None = None
        self._agent_card: AgentCard | None = None
        self._tasks: dict[str, Task] = {}
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if connected to an agent."""
        return self._connected and self._http_client is not None

    @property
    def agent_card(self) -> AgentCard | None:
        """Get the connected agent's card."""
        return self._agent_card

    async def connect(self, url: str | None = None) -> AgentCard:
        """Connect to an A2A agent.

        Fetches the agent card from the well-known endpoint.

        Args:
            url: Agent URL (overrides config).

        Returns:
            The agent's card.

        Raises:
            ConnectionError: If connection fails.
        """
        target_url = url or self.config.base_url
        if not target_url:
            raise ValueError("No URL provided")

        # Create HTTP client
        self._http_client = httpx.AsyncClient(
            timeout=self.config.timeout,
            verify=self.config.verify_ssl,
            headers=self.config.headers,
        )

        # Fetch agent card
        try:
            card_url = f"{target_url.rstrip('/')}/.well-known/agent.json"
            response = await self._http_client.get(card_url)
            response.raise_for_status()

            card_data = response.json()
            self._agent_card = AgentCard.from_dict(card_data)
            self.config.base_url = target_url
            self._connected = True

            logger.info(f"Connected to A2A agent: {self._agent_card.name}")
            return self._agent_card

        except httpx.HTTPError as e:
            await self.disconnect()
            raise ConnectionError(f"Failed to connect to A2A agent: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from the agent."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._agent_card = None
        self._connected = False
        self._tasks.clear()

    async def get_card(self) -> AgentCard:
        """Get the agent's card.

        Returns:
            The agent card.

        Raises:
            RuntimeError: If not connected.
        """
        if not self._connected or not self._agent_card:
            if self.config.base_url:
                return await self.connect()
            raise RuntimeError("Not connected to an agent")

        return self._agent_card

    async def send_message(
        self,
        request: Message | str,
        task_id: str | None = None,
        context_id: str | None = None,
        skill_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Send a message and stream events.

        Args:
            request: Message to send (or string for simple text).
            task_id: Existing task ID to continue.
            context_id: Context/session identifier.
            skill_id: Specific skill to invoke.
            metadata: Additional request metadata.

        Yields:
            Stream events (messages, status updates, artifacts).

        Raises:
            RuntimeError: If not connected.
        """
        if not self._http_client:
            raise RuntimeError("Not connected to an agent")

        # Normalize message
        if isinstance(request, str):
            message = Message.user(request)
        else:
            message = request

        # Build RPC params
        params: dict[str, Any] = {
            "message": message.to_dict(),
        }

        if task_id:
            params["id"] = task_id
        else:
            params["id"] = str(uuid.uuid4())

        if context_id:
            params["contextId"] = context_id

        if skill_id:
            params["skillId"] = skill_id

        if metadata:
            params["metadata"] = metadata

        # Use streaming or polling based on config
        if self.config.streaming:
            async for event in self._send_streaming(params):
                yield event
        else:
            async for event in self._send_polling(params):
                yield event

    async def _send_streaming(
        self,
        params: dict[str, Any],
    ) -> AsyncIterator[StreamEvent]:
        """Send message with streaming response."""
        rpc_request = JSONRPCRequest(
            method="message/stream",
            params=params,
        )

        # Apply interceptors
        context: dict[str, Any] = {
            "url": f"{self.config.base_url}/a2a/stream",
            "headers": dict(self.config.headers),
        }

        for interceptor in self.interceptors:
            rpc_request = await interceptor.intercept_request(rpc_request, context)

        # Send streaming request
        try:
            async with self._http_client.stream(
                "POST",
                context["url"],
                json=rpc_request.to_dict(),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    **context.get("headers", {}),
                },
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    try:
                        data = json.loads(line[6:])
                        event = self._parse_stream_event(data)

                        if event:
                            # Notify consumers
                            for consumer in self.consumers:
                                await consumer.on_event(event)

                            yield event

                    except json.JSONDecodeError:
                        continue

        except httpx.HTTPError as e:
            logger.error(f"Streaming request failed: {e}")
            raise

    async def _send_polling(
        self,
        params: dict[str, Any],
    ) -> AsyncIterator[StreamEvent]:
        """Send message with polling for response."""
        # Send initial message
        rpc_request = JSONRPCRequest(
            method="message/send",
            params=params,
        )

        response = await self._rpc_call(rpc_request)

        if response.is_error:
            raise RuntimeError(f"A2A error: {response.error.message}")

        # Parse initial task
        task = Task.from_dict(response.result)
        self._tasks[task.id] = task

        # Yield initial status
        yield TaskStatusEvent(
            task_id=task.id,
            status=task.status or TaskStatus(state=task.state),
        )

        # Yield any messages
        for msg in task.messages:
            yield MessageEvent(message=msg, task_id=task.id)

        # Poll until complete
        while not task.is_complete:
            await asyncio.sleep(self.config.polling_interval)

            task = await self.get_task(task.id)
            if not task:
                break

            yield TaskStatusEvent(
                task_id=task.id,
                status=task.status or TaskStatus(state=task.state),
            )

        # Yield artifacts
        for artifact in task.artifacts:
            yield ArtifactEvent(artifact=artifact, task_id=task.id)

    def _parse_stream_event(self, data: dict[str, Any]) -> StreamEvent | None:
        """Parse a streaming event from JSON data."""
        kind = data.get("kind", "")

        if kind == "task_status" or "status" in data:
            return TaskStatusEvent(
                task_id=data.get("taskId", data.get("id", "")),
                status=TaskStatus.from_dict(data.get("status", data)),
            )

        elif kind == "message" or "message" in data:
            msg_data = data.get("message", data)
            return MessageEvent(
                message=Message.from_dict(msg_data),
                task_id=data.get("taskId"),
            )

        elif kind == "artifact" or "artifact" in data:
            return ArtifactEvent(
                artifact=Artifact.from_dict(data.get("artifact", data)),
                task_id=data.get("taskId", ""),
            )

        return None

    async def get_task(self, task_id: str) -> Task | None:
        """Get task status.

        Args:
            task_id: Task identifier.

        Returns:
            Task or None if not found.
        """
        rpc_request = JSONRPCRequest(
            method="tasks/get",
            params={"id": task_id},
        )

        response = await self._rpc_call(rpc_request)

        if response.is_error:
            if response.error.code == A2AErrorCode.TASK_NOT_FOUND.value:
                return None
            raise RuntimeError(f"Failed to get task: {response.error.message}")

        task = Task.from_dict(response.result)
        self._tasks[task.id] = task
        return task

    async def cancel_task(self, task_id: str) -> Task:
        """Cancel a task.

        Args:
            task_id: Task to cancel.

        Returns:
            Updated task.

        Raises:
            RuntimeError: If cancellation fails.
        """
        rpc_request = JSONRPCRequest(
            method="tasks/cancel",
            params={"id": task_id},
        )

        response = await self._rpc_call(rpc_request)

        if response.is_error:
            raise RuntimeError(f"Failed to cancel task: {response.error.message}")

        task = Task.from_dict(response.result)
        self._tasks[task.id] = task
        return task

    async def send_input(
        self,
        task_id: str,
        message: Message | str,
    ) -> AsyncIterator[StreamEvent]:
        """Send additional input to a task waiting for input.

        Args:
            task_id: Task identifier.
            message: Input message.

        Yields:
            Stream events.
        """
        async for event in self.send_message(
            request=message,
            task_id=task_id,
        ):
            yield event

    async def wait_for_completion(
        self,
        task_id: str,
        timeout: float | None = None,
    ) -> Task:
        """Wait for a task to complete.

        Args:
            task_id: Task to wait for.
            timeout: Maximum wait time in seconds.

        Returns:
            Completed task.

        Raises:
            TimeoutError: If timeout exceeded.
            ValueError: If task not found.
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            task = await self.get_task(task_id)

            if task is None:
                raise ValueError(f"Task not found: {task_id}")

            if task.is_complete:
                return task

            if timeout:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(
                        f"Task {task_id} did not complete within {timeout}s"
                    )

            await asyncio.sleep(self.config.polling_interval)

    async def _rpc_call(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """Execute a JSON-RPC call.

        Args:
            request: The RPC request.

        Returns:
            The RPC response.
        """
        if not self._http_client:
            raise RuntimeError("Not connected to an agent")

        # Build context
        context: dict[str, Any] = {
            "url": f"{self.config.base_url}/a2a",
            "headers": dict(self.config.headers),
        }

        # Apply request interceptors
        for interceptor in self.interceptors:
            request = await interceptor.intercept_request(request, context)

        # Make request with retries
        while True:
            try:
                http_response = await self._http_client.post(
                    context["url"],
                    json=request.to_dict(),
                    headers={
                        "Content-Type": "application/json",
                        **context.get("headers", {}),
                    },
                )
                http_response.raise_for_status()

                result = http_response.json()

                # Build response
                if "error" in result:
                    error_data = result["error"]
                    response = JSONRPCResponse(
                        id=result.get("id", request.id),
                        error=JSONRPCError(
                            code=error_data.get("code", -32603),
                            message=error_data.get("message", "Unknown error"),
                            data=error_data.get("data"),
                        ),
                    )
                else:
                    response = JSONRPCResponse(
                        id=result.get("id", request.id),
                        result=result.get("result"),
                    )

                # Apply response interceptors
                for interceptor in self.interceptors:
                    response = await interceptor.intercept_response(response, context)

                # Check for retry
                if context.get("should_retry"):
                    delay = context.get("retry_delay", 1.0)
                    context["retry_count"] = context.get("retry_count", 0) + 1
                    context.pop("should_retry", None)
                    await asyncio.sleep(delay)
                    continue

                return response

            except httpx.HTTPError as e:
                logger.error(f"HTTP error in RPC call: {e}")
                return JSONRPCResponse(
                    id=request.id,
                    error=JSONRPCError(
                        code=A2AErrorCode.INTERNAL_ERROR.value,
                        message=str(e),
                    ),
                )

    async def __aenter__(self) -> A2AClient:
        """Async context manager entry."""
        if self.config.base_url:
            await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()


# =============================================================================
# Convenience Functions
# =============================================================================


async def discover_agent(url: str) -> AgentCard:
    """Discover an agent by fetching its card.

    Args:
        url: Agent URL.

    Returns:
        Agent card.
    """
    async with A2AClient(ClientConfig(base_url=url)) as client:
        return await client.get_card()


async def send_task(
    url: str,
    message: str,
    skill_id: str | None = None,
    timeout: float = 60.0,
) -> Task:
    """Send a task to an agent and wait for completion.

    Args:
        url: Agent URL.
        message: Task message.
        skill_id: Optional skill to invoke.
        timeout: Maximum wait time.

    Returns:
        Completed task.
    """
    config = ClientConfig(base_url=url, streaming=False)

    async with A2AClient(config) as client:
        task_id: str | None = None

        async for event in client.send_message(message, skill_id=skill_id):
            if isinstance(event, TaskStatusEvent):
                task_id = event.task_id

        if task_id:
            return await client.wait_for_completion(task_id, timeout=timeout)

        raise RuntimeError("No task created")
