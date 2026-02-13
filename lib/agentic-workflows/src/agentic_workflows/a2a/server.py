"""A2A Protocol Server Implementation.

Provides a server for exposing agents via the A2A protocol.
Integrates with FastAPI for HTTP handling.

Reference: https://github.com/a2aproject/a2a-python
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Callable, TypeVar, Generic

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
    StreamEvent,
    TaskStatusEvent,
    MessageEvent,
    ArtifactEvent,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Request Context
# =============================================================================


@dataclass
class RequestContext:
    """Context for an incoming A2A request.

    Provides access to request details and utilities for the executor.

    Attributes:
        message: The incoming message.
        task_id: Associated task ID.
        context_id: Optional context/session ID.
        skill_id: Optional skill being invoked.
        metadata: Request metadata.
        headers: HTTP headers.
    """

    message: Message
    """The incoming message."""

    task_id: str
    """Associated task ID."""

    context_id: str | None = None
    """Optional context/session ID."""

    skill_id: str | None = None
    """Optional skill being invoked."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Request metadata."""

    headers: dict[str, str] = field(default_factory=dict)
    """HTTP headers."""

    user_id: str | None = None
    """Authenticated user ID (if any)."""

    created_at: datetime = field(default_factory=datetime.utcnow)
    """When the request was received."""

    def get_text(self) -> str:
        """Get text content from the message."""
        return self.message.get_text()


# =============================================================================
# Event Queue
# =============================================================================


class EventQueue:
    """Queue for publishing A2A events.

    Used by executors to publish status updates, messages,
    and artifacts during task execution.
    """

    def __init__(self, task_id: str):
        """Initialize event queue.

        Args:
            task_id: Associated task ID.
        """
        self.task_id = task_id
        self._queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        self._closed = False

    async def publish_status(
        self,
        state: TaskState,
        message: str | None = None,
    ) -> None:
        """Publish a status update event.

        Args:
            state: New task state.
            message: Optional status message.
        """
        if self._closed:
            return

        event = TaskStatusEvent(
            task_id=self.task_id,
            status=TaskStatus(state=state, message=message),
        )
        await self._queue.put(event)

    async def publish_message(self, message: Message) -> None:
        """Publish a message event.

        Args:
            message: The message to publish.
        """
        if self._closed:
            return

        event = MessageEvent(
            message=message,
            task_id=self.task_id,
        )
        await self._queue.put(event)

    async def publish_text(self, text: str, role: MessageRole = MessageRole.AGENT) -> None:
        """Publish a text message.

        Args:
            text: Text content.
            role: Message role (default: agent).
        """
        message = Message(
            role=role,
            content=text,
            parts=[TextPart(text=text)],
            task_id=self.task_id,
        )
        await self.publish_message(message)

    async def publish_artifact(self, artifact: Artifact) -> None:
        """Publish an artifact event.

        Args:
            artifact: The artifact to publish.
        """
        if self._closed:
            return

        event = ArtifactEvent(
            artifact=artifact,
            task_id=self.task_id,
        )
        await self._queue.put(event)

    async def get_events(self) -> AsyncIterator[StreamEvent]:
        """Iterate over events in the queue.

        Yields:
            Events as they are published.
        """
        while not self._closed:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )
                yield event

                # Check for terminal status
                if isinstance(event, TaskStatusEvent):
                    if event.status.state in (
                        TaskState.COMPLETED,
                        TaskState.FAILED,
                        TaskState.CANCELED,
                    ):
                        break

            except asyncio.TimeoutError:
                continue

    def close(self) -> None:
        """Close the event queue."""
        self._closed = True


# =============================================================================
# Agent Executor
# =============================================================================


class AgentExecutor(ABC):
    """Abstract base class for agent execution logic.

    Implement this class to define how your agent processes
    incoming messages and produces responses.

    Example:
        ```python
        class MyExecutor(AgentExecutor):
            async def execute(
                self,
                context: RequestContext,
                events: EventQueue,
            ) -> None:
                await events.publish_status(TaskState.WORKING)
                response = await process_message(context.get_text())
                await events.publish_text(response)
                await events.publish_status(TaskState.COMPLETED)

            async def cancel(self, task_id: str) -> bool:
                return True  # Accept cancellation
        ```
    """

    @abstractmethod
    async def execute(
        self,
        context: RequestContext,
        events: EventQueue,
    ) -> None:
        """Execute the agent logic for a request.

        Args:
            context: Request context with message and metadata.
            events: Queue for publishing response events.

        Raises:
            Exception: Any exception will result in task failure.
        """
        pass

    @abstractmethod
    async def cancel(self, task_id: str) -> bool:
        """Handle task cancellation request.

        Args:
            task_id: Task to cancel.

        Returns:
            True if cancellation was successful.
        """
        pass

    async def on_initialize(self) -> None:
        """Called when the server starts.

        Override to add initialization logic.
        """
        pass

    async def on_shutdown(self) -> None:
        """Called when the server stops.

        Override to add cleanup logic.
        """
        pass


class SimpleExecutor(AgentExecutor):
    """Simple executor that wraps a callable.

    Example:
        ```python
        async def handle(text: str) -> str:
            return f"You said: {text}"

        executor = SimpleExecutor(handle)
        ```
    """

    def __init__(
        self,
        handler: Callable[[str], str] | Callable[[str], AsyncIterator[str]],
        is_streaming: bool = False,
    ):
        """Initialize simple executor.

        Args:
            handler: Function to process messages.
            is_streaming: Whether handler yields chunks.
        """
        self.handler = handler
        self.is_streaming = is_streaming
        self._active_tasks: set[str] = set()

    async def execute(
        self,
        context: RequestContext,
        events: EventQueue,
    ) -> None:
        """Execute the handler."""
        self._active_tasks.add(context.task_id)

        try:
            await events.publish_status(TaskState.WORKING)

            text = context.get_text()

            if self.is_streaming:
                # Streaming handler
                full_response = []
                async for chunk in self.handler(text):
                    if context.task_id not in self._active_tasks:
                        # Task was canceled
                        await events.publish_status(TaskState.CANCELED)
                        return

                    full_response.append(chunk)
                    await events.publish_text(chunk)

                # Final response
                await events.publish_status(TaskState.COMPLETED)
            else:
                # Simple handler
                if asyncio.iscoroutinefunction(self.handler):
                    response = await self.handler(text)
                else:
                    response = self.handler(text)

                await events.publish_text(response)
                await events.publish_status(TaskState.COMPLETED)

        except Exception as e:
            logger.error(f"Executor error: {e}")
            await events.publish_status(
                TaskState.FAILED,
                message=str(e),
            )
            raise

        finally:
            self._active_tasks.discard(context.task_id)

    async def cancel(self, task_id: str) -> bool:
        """Cancel by removing from active tasks."""
        if task_id in self._active_tasks:
            self._active_tasks.discard(task_id)
            return True
        return False


# =============================================================================
# Task Storage
# =============================================================================


class TaskStorage(ABC):
    """Abstract base for task storage backends."""

    @abstractmethod
    async def save(self, task: Task) -> None:
        """Save a task."""
        pass

    @abstractmethod
    async def get(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        pass

    @abstractmethod
    async def delete(self, task_id: str) -> bool:
        """Delete a task."""
        pass

    @abstractmethod
    async def list_by_context(self, context_id: str) -> list[Task]:
        """List tasks by context ID."""
        pass


class InMemoryTaskStorage(TaskStorage):
    """In-memory task storage implementation."""

    def __init__(self, max_tasks: int = 1000):
        self.max_tasks = max_tasks
        self._tasks: dict[str, Task] = {}
        self._context_index: dict[str, set[str]] = {}

    async def save(self, task: Task) -> None:
        """Save task to memory."""
        # Evict old tasks if needed
        if len(self._tasks) >= self.max_tasks:
            oldest_id = next(iter(self._tasks))
            await self.delete(oldest_id)

        self._tasks[task.id] = task

        if task.context_id:
            if task.context_id not in self._context_index:
                self._context_index[task.context_id] = set()
            self._context_index[task.context_id].add(task.id)

    async def get(self, task_id: str) -> Task | None:
        """Get task from memory."""
        return self._tasks.get(task_id)

    async def delete(self, task_id: str) -> bool:
        """Delete task from memory."""
        if task_id in self._tasks:
            task = self._tasks.pop(task_id)
            if task.context_id and task.context_id in self._context_index:
                self._context_index[task.context_id].discard(task_id)
            return True
        return False

    async def list_by_context(self, context_id: str) -> list[Task]:
        """List tasks by context."""
        task_ids = self._context_index.get(context_id, set())
        return [self._tasks[tid] for tid in task_ids if tid in self._tasks]


# =============================================================================
# A2A Server
# =============================================================================


class A2AServer:
    """A2A Protocol Server.

    Provides HTTP endpoints for A2A protocol communication.
    Integrates with FastAPI for request handling.

    Example:
        ```python
        from fastapi import FastAPI

        app = FastAPI()
        executor = MyAgentExecutor()
        agent_card = AgentCard(name="My Agent", description="...")

        server = A2AServer(
            executor=executor,
            card=agent_card,
        )
        server.mount(app)

        # Now the app serves:
        # GET /.well-known/agent.json - Agent card
        # POST /a2a - JSON-RPC endpoint
        # POST /a2a/stream - Streaming endpoint
        ```
    """

    def __init__(
        self,
        executor: AgentExecutor,
        card: AgentCard,
        storage: TaskStorage | None = None,
        prefix: str = "",
    ):
        """Initialize A2A server.

        Args:
            executor: Agent executor for processing requests.
            card: Agent card describing capabilities.
            storage: Task storage backend (default: in-memory).
            prefix: URL prefix for endpoints.
        """
        self.executor = executor
        self.card = card
        self.storage = storage or InMemoryTaskStorage()
        self.prefix = prefix.rstrip("/")

        self._running = False

    def mount(self, app: Any) -> None:
        """Mount A2A endpoints on a FastAPI application.

        Args:
            app: FastAPI application instance.
        """
        # Import FastAPI types
        try:
            from fastapi import Request, Response
            from fastapi.responses import JSONResponse, StreamingResponse
        except ImportError:
            raise ImportError(
                "FastAPI is required for A2A server. "
                "Install with: pip install fastapi"
            )

        # Well-known agent card endpoint
        @app.get(f"{self.prefix}/.well-known/agent.json")
        async def get_agent_card() -> JSONResponse:
            """Return the agent card."""
            return JSONResponse(content=self.card.to_dict())

        # JSON-RPC endpoint
        @app.post(f"{self.prefix}/a2a")
        async def handle_rpc(request: Request) -> JSONResponse:
            """Handle JSON-RPC requests."""
            try:
                body = await request.json()
                response = await self._handle_rpc_request(body, dict(request.headers))
                return JSONResponse(content=response.to_dict())

            except json.JSONDecodeError:
                error = JSONRPCError.from_code(A2AErrorCode.PARSE_ERROR)
                response = JSONRPCResponse(id="", error=error)
                return JSONResponse(content=response.to_dict(), status_code=400)

        # Streaming endpoint
        @app.post(f"{self.prefix}/a2a/stream")
        async def handle_stream(request: Request) -> StreamingResponse:
            """Handle streaming requests."""
            try:
                body = await request.json()
                headers = dict(request.headers)

                return StreamingResponse(
                    self._handle_stream_request(body, headers),
                    media_type="text/event-stream",
                )

            except json.JSONDecodeError:
                error = JSONRPCError.from_code(A2AErrorCode.PARSE_ERROR)
                return JSONResponse(
                    content=JSONRPCResponse(id="", error=error).to_dict(),
                    status_code=400,
                )

        # Startup/shutdown hooks
        @app.on_event("startup")
        async def on_startup():
            await self.start()

        @app.on_event("shutdown")
        async def on_shutdown():
            await self.stop()

    async def start(self) -> None:
        """Start the server."""
        if self._running:
            return

        await self.executor.on_initialize()
        self._running = True
        logger.info(f"A2A Server started: {self.card.name}")

    async def stop(self) -> None:
        """Stop the server."""
        if not self._running:
            return

        await self.executor.on_shutdown()
        self._running = False
        logger.info(f"A2A Server stopped: {self.card.name}")

    async def _handle_rpc_request(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> JSONRPCResponse:
        """Handle a JSON-RPC request."""
        # Validate request structure
        if body.get("jsonrpc") != "2.0":
            return JSONRPCResponse(
                id=body.get("id", ""),
                error=JSONRPCError.from_code(A2AErrorCode.INVALID_REQUEST),
            )

        method = body.get("method", "")
        params = body.get("params", {})
        request_id = body.get("id", str(uuid.uuid4()))

        # Route to handler
        try:
            if method == "message/send":
                result = await self._handle_message_send(params, headers)
            elif method == "tasks/get":
                result = await self._handle_task_get(params)
            elif method == "tasks/cancel":
                result = await self._handle_task_cancel(params)
            else:
                return JSONRPCResponse(
                    id=request_id,
                    error=JSONRPCError.from_code(A2AErrorCode.METHOD_NOT_FOUND),
                )

            return JSONRPCResponse(id=request_id, result=result)

        except Exception as e:
            logger.error(f"RPC handler error: {e}")
            return JSONRPCResponse(
                id=request_id,
                error=JSONRPCError(
                    code=A2AErrorCode.INTERNAL_ERROR.value,
                    message=str(e),
                ),
            )

    async def _handle_message_send(
        self,
        params: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """Handle message/send method."""
        # Parse message
        message_data = params.get("message", {})
        message = Message.from_dict(message_data)

        # Get or create task
        task_id = params.get("id", str(uuid.uuid4()))
        context_id = params.get("contextId")
        skill_id = params.get("skillId")
        metadata = params.get("metadata", {})

        # Check for existing task
        existing_task = await self.storage.get(task_id)

        if existing_task:
            # Add message to existing task
            existing_task.add_message(message)
            await self.storage.save(existing_task)
            task = existing_task
        else:
            # Create new task
            task = Task(
                id=task_id,
                context_id=context_id,
                messages=[message],
            )
            await self.storage.save(task)

        # Build request context
        context = RequestContext(
            message=message,
            task_id=task_id,
            context_id=context_id,
            skill_id=skill_id,
            metadata=metadata,
            headers=headers,
        )

        # Execute in background
        event_queue = EventQueue(task_id)

        async def execute_and_update():
            try:
                await self.executor.execute(context, event_queue)
            except Exception as e:
                logger.error(f"Execution error: {e}")
                await event_queue.publish_status(TaskState.FAILED, str(e))
            finally:
                event_queue.close()

                # Collect final state
                final_task = await self.storage.get(task_id)
                if final_task:
                    # Update from events
                    async for event in event_queue.get_events():
                        if isinstance(event, MessageEvent):
                            final_task.add_message(event.message)
                        elif isinstance(event, ArtifactEvent):
                            final_task.add_artifact(event.artifact)
                        elif isinstance(event, TaskStatusEvent):
                            final_task.update_state(
                                event.status.state,
                                event.status.message,
                            )
                    await self.storage.save(final_task)

        asyncio.create_task(execute_and_update())

        # Return current task state
        task.update_state(TaskState.PENDING)
        return task.to_dict()

    async def _handle_task_get(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle tasks/get method."""
        task_id = params.get("id", "")

        task = await self.storage.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        return task.to_dict()

    async def _handle_task_cancel(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle tasks/cancel method."""
        task_id = params.get("id", "")

        task = await self.storage.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        if task.is_complete:
            raise ValueError(f"Task already complete: {task_id}")

        # Request cancellation from executor
        canceled = await self.executor.cancel(task_id)

        if canceled:
            task.update_state(TaskState.CANCELED)
            await self.storage.save(task)

        return task.to_dict()

    async def _handle_stream_request(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> AsyncIterator[str]:
        """Handle streaming request."""
        # Parse request
        params = body.get("params", {})
        message_data = params.get("message", {})
        message = Message.from_dict(message_data)

        task_id = params.get("id", str(uuid.uuid4()))
        context_id = params.get("contextId")
        skill_id = params.get("skillId")
        metadata = params.get("metadata", {})

        # Create task
        task = Task(
            id=task_id,
            context_id=context_id,
            messages=[message],
        )
        await self.storage.save(task)

        # Build request context
        context = RequestContext(
            message=message,
            task_id=task_id,
            context_id=context_id,
            skill_id=skill_id,
            metadata=metadata,
            headers=headers,
        )

        # Create event queue
        event_queue = EventQueue(task_id)

        # Start execution
        async def execute():
            try:
                await self.executor.execute(context, event_queue)
            except Exception as e:
                logger.error(f"Execution error: {e}")
                await event_queue.publish_status(TaskState.FAILED, str(e))
            finally:
                event_queue.close()

        execution_task = asyncio.create_task(execute())

        # Stream events
        try:
            async for event in event_queue.get_events():
                # Update stored task
                stored_task = await self.storage.get(task_id)
                if stored_task:
                    if isinstance(event, MessageEvent):
                        stored_task.add_message(event.message)
                    elif isinstance(event, ArtifactEvent):
                        stored_task.add_artifact(event.artifact)
                    elif isinstance(event, TaskStatusEvent):
                        stored_task.update_state(
                            event.status.state,
                            event.status.message,
                        )
                    await self.storage.save(stored_task)

                # Yield SSE data
                yield f"data: {json.dumps(event.to_dict())}\n\n"

        finally:
            event_queue.close()
            if not execution_task.done():
                execution_task.cancel()


# =============================================================================
# Factory Functions
# =============================================================================


def create_server(
    executor: AgentExecutor,
    name: str,
    description: str,
    skills: list[AgentSkill] | None = None,
    url: str = "",
    **card_kwargs,
) -> A2AServer:
    """Create an A2A server with a new agent card.

    Args:
        executor: Agent executor.
        name: Agent name.
        description: Agent description.
        skills: Agent skills.
        url: Agent URL.
        **card_kwargs: Additional agent card options.

    Returns:
        Configured A2A server.
    """
    card = AgentCard(
        name=name,
        description=description,
        skills=skills or [],
        url=url,
        **card_kwargs,
    )

    return A2AServer(executor=executor, card=card)


def create_simple_server(
    handler: Callable[[str], str],
    name: str,
    description: str,
    **kwargs,
) -> A2AServer:
    """Create a simple A2A server from a handler function.

    Args:
        handler: Function that takes text and returns response.
        name: Agent name.
        description: Agent description.
        **kwargs: Additional server options.

    Returns:
        Configured A2A server.
    """
    executor = SimpleExecutor(handler)
    return create_server(
        executor=executor,
        name=name,
        description=description,
        **kwargs,
    )
