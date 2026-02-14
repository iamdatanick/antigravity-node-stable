"""Agent-to-Agent (A2A) protocol client implementation.

Implements A2A Protocol v0.3 specification.
Reference: https://github.com/a2aproject/A2A
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

# Try to import the official A2A SDK
try:
    import a2a_python

    A2A_SDK_AVAILABLE = True
except ImportError:
    A2A_SDK_AVAILABLE = False


class A2ATaskState(Enum):
    """A2A task states per v0.3 spec."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class A2AMessageRole(Enum):
    """A2A message roles."""

    USER = "user"
    AGENT = "agent"


class A2APartType(Enum):
    """A2A message part types."""

    TEXT = "text"
    FILE = "file"
    DATA = "data"


@dataclass
class A2APart:
    """A part of an A2A message."""

    type: A2APartType
    text: str | None = None
    file_uri: str | None = None
    file_name: str | None = None
    mime_type: str | None = None
    data: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {"type": self.type.value}
        if self.type == A2APartType.TEXT:
            result["text"] = self.text
        elif self.type == A2APartType.FILE:
            result["file"] = {
                "uri": self.file_uri,
                "name": self.file_name,
                "mimeType": self.mime_type,
            }
        elif self.type == A2APartType.DATA:
            result["data"] = self.data
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> A2APart:
        """Create from dictionary."""
        part_type = A2APartType(data["type"])
        if part_type == A2APartType.TEXT:
            return cls(type=part_type, text=data.get("text", ""))
        elif part_type == A2APartType.FILE:
            file_data = data.get("file", {})
            return cls(
                type=part_type,
                file_uri=file_data.get("uri"),
                file_name=file_data.get("name"),
                mime_type=file_data.get("mimeType"),
            )
        else:
            return cls(type=part_type, data=data.get("data"))

    @classmethod
    def text(cls, content: str) -> A2APart:
        """Create a text part."""
        return cls(type=A2APartType.TEXT, text=content)


@dataclass
class A2AMessage:
    """An A2A message."""

    role: A2AMessageRole
    parts: list[A2APart]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def user_text(cls, content: str) -> A2AMessage:
        """Create a user text message."""
        return cls(role=A2AMessageRole.USER, parts=[A2APart.text(content)])

    @classmethod
    def agent_text(cls, content: str) -> A2AMessage:
        """Create an agent text message."""
        return cls(role=A2AMessageRole.AGENT, parts=[A2APart.text(content)])

    def get_text(self) -> str:
        """Extract text content from message."""
        texts = []
        for part in self.parts:
            if part.type == A2APartType.TEXT and part.text:
                texts.append(part.text)
        return "\n".join(texts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role.value,
            "parts": [p.to_dict() for p in self.parts],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> A2AMessage:
        """Create from dictionary."""
        return cls(
            role=A2AMessageRole(data["role"]),
            parts=[A2APart.from_dict(p) for p in data.get("parts", [])],
            metadata=data.get("metadata", {}),
        )


@dataclass
class A2AArtifact:
    """An A2A artifact (task output)."""

    id: str
    name: str
    mime_type: str = "text/plain"
    content: Any = None
    uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "name": self.name,
            "mimeType": self.mime_type,
            "metadata": self.metadata,
        }
        if self.content is not None:
            result["content"] = self.content
        if self.uri:
            result["uri"] = self.uri
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> A2AArtifact:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            mime_type=data.get("mimeType", "text/plain"),
            content=data.get("content"),
            uri=data.get("uri"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class A2ATask:
    """An A2A task."""

    id: str
    session_id: str | None = None
    state: A2ATaskState = A2ATaskState.SUBMITTED
    messages: list[A2AMessage] = field(default_factory=list)
    artifacts: list[A2AArtifact] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def is_complete(self) -> bool:
        """Check if task is complete."""
        return self.state in (
            A2ATaskState.COMPLETED,
            A2ATaskState.FAILED,
            A2ATaskState.CANCELED,
        )

    @property
    def is_successful(self) -> bool:
        """Check if task completed successfully."""
        return self.state == A2ATaskState.COMPLETED

    @property
    def last_message(self) -> A2AMessage | None:
        """Get the last message."""
        return self.messages[-1] if self.messages else None

    def get_text_response(self) -> str:
        """Get text content from agent's last message."""
        for msg in reversed(self.messages):
            if msg.role == A2AMessageRole.AGENT:
                return msg.get_text()
        return ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "sessionId": self.session_id,
            "status": {"state": self.state.value},
            "history": [m.to_dict() for m in self.messages],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> A2ATask:
        """Create from dictionary."""
        messages = [A2AMessage.from_dict(m) for m in data.get("history", [])]
        artifacts = [A2AArtifact.from_dict(a) for a in data.get("artifacts", [])]

        status = data.get("status", {})
        state_str = status.get("state", "submitted")

        return cls(
            id=data["id"],
            session_id=data.get("sessionId"),
            state=A2ATaskState(state_str),
            messages=messages,
            artifacts=artifacts,
            metadata=data.get("metadata", {}),
            error=status.get("error"),
        )


@dataclass
class A2ASkill:
    """An A2A agent skill."""

    id: str
    name: str
    description: str = ""
    input_modes: list[str] = field(default_factory=lambda: ["text"])
    output_modes: list[str] = field(default_factory=lambda: ["text"])


@dataclass
class A2AAgentCard:
    """A2A agent capability card (/.well-known/agent.json)."""

    name: str
    description: str
    url: str
    version: str = "0.3"
    protocol_version: str = "0.3"
    capabilities: dict[str, Any] = field(default_factory=dict)
    authentication: dict[str, Any] = field(default_factory=dict)
    skills: list[A2ASkill] = field(default_factory=list)
    default_input_modes: list[str] = field(default_factory=lambda: ["text"])
    default_output_modes: list[str] = field(default_factory=lambda: ["text"])
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serving."""
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "protocolVersion": self.protocol_version,
            "capabilities": self.capabilities,
            "authentication": self.authentication,
            "skills": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "inputModes": s.input_modes,
                    "outputModes": s.output_modes,
                }
                for s in self.skills
            ],
            "defaultInputModes": self.default_input_modes,
            "defaultOutputModes": self.default_output_modes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> A2AAgentCard:
        """Create from dictionary."""
        skills = [
            A2ASkill(
                id=s.get("id", ""),
                name=s.get("name", ""),
                description=s.get("description", ""),
                input_modes=s.get("inputModes", ["text"]),
                output_modes=s.get("outputModes", ["text"]),
            )
            for s in data.get("skills", [])
        ]

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            url=data.get("url", ""),
            version=data.get("version", "0.3"),
            protocol_version=data.get("protocolVersion", "0.3"),
            capabilities=data.get("capabilities", {}),
            authentication=data.get("authentication", {}),
            skills=skills,
            default_input_modes=data.get("defaultInputModes", ["text"]),
            default_output_modes=data.get("defaultOutputModes", ["text"]),
            metadata=data.get("metadata", {}),
        )


class A2AClient:
    """Client for Agent-to-Agent protocol v0.3.

    Features:
    - Agent discovery via Agent Cards
    - Task submission and tracking
    - Streaming support via SSE
    - Session management for multi-turn conversations
    - Artifact retrieval
    """

    PROTOCOL_VERSION = "0.3"

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 60.0,
        on_message: Callable[[A2AMessage], None] | None = None,
    ):
        """Initialize A2A client.

        Args:
            base_url: Base URL for A2A server.
            timeout: Request timeout.
            on_message: Callback for received messages.
        """
        self.base_url = base_url
        self.timeout = timeout
        self.on_message = on_message

        self._http_client: httpx.AsyncClient | None = None
        self._agent_card: A2AAgentCard | None = None
        self._sessions: dict[str, str] = {}  # session_id -> last_task_id

    async def connect(self, url: str | None = None) -> A2AAgentCard | None:
        """Connect to an A2A agent.

        Args:
            url: Agent URL (uses base_url if not provided).

        Returns:
            Agent card if successful.
        """
        target_url = url or self.base_url
        if not target_url:
            raise ValueError("No URL provided")

        self._http_client = httpx.AsyncClient(timeout=self.timeout)

        # Fetch agent card from well-known location
        try:
            response = await self._http_client.get(f"{target_url}/.well-known/agent.json")
            response.raise_for_status()

            card_data = response.json()
            self._agent_card = A2AAgentCard.from_dict(card_data)
            self.base_url = target_url

            return self._agent_card

        except Exception as e:
            raise ConnectionError(f"Failed to discover agent: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from agent."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._agent_card = None
        self._sessions.clear()

    async def send_task(
        self,
        message: str | A2AMessage,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        skill_id: str | None = None,
    ) -> A2ATask:
        """Send a task to the agent.

        Args:
            message: Task message (string or A2AMessage).
            session_id: Optional session for continuity.
            metadata: Additional metadata.
            skill_id: Optional skill to invoke.

        Returns:
            Created task.
        """
        if not self._http_client or not self.base_url:
            raise RuntimeError("Not connected to an agent")

        # Create message
        if isinstance(message, str):
            msg = A2AMessage.user_text(message)
        else:
            msg = message

        # Build JSON-RPC request
        task_id = str(uuid.uuid4())
        params: dict[str, Any] = {
            "id": task_id,
            "message": msg.to_dict(),
        }

        if session_id:
            params["sessionId"] = session_id

        if metadata:
            params["metadata"] = metadata

        if skill_id:
            params["skillId"] = skill_id

        request_data = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": params,
        }

        # Send request
        response = await self._http_client.post(
            f"{self.base_url}/a2a",
            json=request_data,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        result = response.json()

        if "error" in result:
            raise RuntimeError(
                f"Task creation failed: {result['error'].get('message', 'Unknown error')}"
            )

        task = A2ATask.from_dict(result.get("result", {}))

        # Track session
        if task.session_id:
            self._sessions[task.session_id] = task.id

        return task

    async def get_task(self, task_id: str) -> A2ATask | None:
        """Get task status.

        Args:
            task_id: Task ID.

        Returns:
            Task or None if not found.
        """
        if not self._http_client or not self.base_url:
            raise RuntimeError("Not connected to an agent")

        request_data = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/get",
            "params": {"id": task_id},
        }

        try:
            response = await self._http_client.post(
                f"{self.base_url}/a2a",
                json=request_data,
            )
            response.raise_for_status()

            result = response.json()

            if "error" in result:
                return None

            return A2ATask.from_dict(result.get("result", {}))

        except Exception:
            return None

    async def wait_for_completion(
        self,
        task_id: str,
        poll_interval: float = 1.0,
        timeout: float | None = None,
    ) -> A2ATask:
        """Wait for task completion.

        Args:
            task_id: Task to wait for.
            poll_interval: Polling interval in seconds.
            timeout: Maximum wait time.

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

            # Check timeout
            if timeout:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")

            await asyncio.sleep(poll_interval)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.

        Args:
            task_id: Task to cancel.

        Returns:
            True if cancelled.
        """
        if not self._http_client or not self.base_url:
            raise RuntimeError("Not connected to an agent")

        request_data = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/cancel",
            "params": {"id": task_id},
        }

        try:
            response = await self._http_client.post(
                f"{self.base_url}/a2a",
                json=request_data,
            )
            response.raise_for_status()

            result = response.json()
            return "error" not in result

        except Exception:
            return False

    async def send_input(
        self,
        task_id: str,
        message: str | A2AMessage,
    ) -> A2ATask:
        """Send input to a task waiting for input.

        Args:
            task_id: Task ID.
            message: Input message.

        Returns:
            Updated task.
        """
        if not self._http_client or not self.base_url:
            raise RuntimeError("Not connected to an agent")

        if isinstance(message, str):
            msg = A2AMessage.user_text(message)
        else:
            msg = message

        request_data = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "id": task_id,
                "message": msg.to_dict(),
            },
        }

        response = await self._http_client.post(
            f"{self.base_url}/a2a",
            json=request_data,
        )
        response.raise_for_status()

        result = response.json()

        if "error" in result:
            raise RuntimeError(f"Send input failed: {result['error'].get('message')}")

        return A2ATask.from_dict(result.get("result", {}))

    async def stream_task(
        self,
        message: str | A2AMessage,
        session_id: str | None = None,
    ) -> AsyncIterator[A2AMessage]:
        """Stream task responses via SSE.

        Args:
            message: Task message.
            session_id: Optional session ID.

        Yields:
            A2AMessage objects as they arrive.
        """
        if not self._http_client or not self.base_url:
            raise RuntimeError("Not connected to an agent")

        if isinstance(message, str):
            msg = A2AMessage.user_text(message)
        else:
            msg = message

        request_data = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/stream",
            "params": {
                "id": str(uuid.uuid4()),
                "message": msg.to_dict(),
            },
        }

        if session_id:
            request_data["params"]["sessionId"] = session_id

        async with self._http_client.stream(
            "POST",
            f"{self.base_url}/a2a/stream",
            json=request_data,
            headers={"Accept": "text/event-stream"},
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])

                        if "message" in data:
                            message_obj = A2AMessage.from_dict(data["message"])
                            if self.on_message:
                                self.on_message(message_obj)
                            yield message_obj

                    except json.JSONDecodeError:
                        continue

    async def get_artifact(self, task_id: str, artifact_id: str) -> A2AArtifact | None:
        """Get a specific artifact from a task.

        Args:
            task_id: Task ID.
            artifact_id: Artifact ID.

        Returns:
            Artifact or None if not found.
        """
        task = await self.get_task(task_id)
        if task:
            for artifact in task.artifacts:
                if artifact.id == artifact_id:
                    return artifact
        return None

    @property
    def agent_card(self) -> A2AAgentCard | None:
        """Get connected agent's card."""
        return self._agent_card

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._http_client is not None and self._agent_card is not None

    @property
    def agent_skills(self) -> list[A2ASkill]:
        """Get agent's skills."""
        return self._agent_card.skills if self._agent_card else []

    async def __aenter__(self) -> A2AClient:
        """Async context manager entry."""
        if self.base_url:
            await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()


class A2AAgentServer:
    """Helper for serving an A2A agent.

    Provides utilities for creating the agent card and handling requests.
    """

    def __init__(
        self,
        name: str,
        description: str,
        url: str,
        skills: list[A2ASkill] | None = None,
    ):
        """Initialize A2A agent server.

        Args:
            name: Agent name.
            description: Agent description.
            url: Agent URL.
            skills: List of skills.
        """
        self.name = name
        self.description = description
        self.url = url
        self.skills = skills or []

    def get_agent_card(self) -> A2AAgentCard:
        """Get the agent card for this server."""
        return A2AAgentCard(
            name=self.name,
            description=self.description,
            url=self.url,
            skills=self.skills,
            capabilities={
                "streaming": True,
                "tasks": True,
                "artifacts": True,
            },
        )

    def get_agent_card_json(self) -> str:
        """Get agent card as JSON string for serving."""
        return json.dumps(self.get_agent_card().to_dict(), indent=2)
