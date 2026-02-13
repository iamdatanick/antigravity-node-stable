"""A2A Protocol Type Definitions.

Implements the Agent-to-Agent (A2A) Protocol v0.3 type system.
Reference: https://github.com/a2aproject/a2a-python

This module provides comprehensive type definitions for A2A protocol
communication including:
- Agent Cards for capability discovery
- Agent Skills for task capabilities
- Tasks for stateful operations
- Messages for conversation turns
- Artifacts for task outputs
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar, Generic


# =============================================================================
# Enums
# =============================================================================


class TaskState(Enum):
    """A2A task lifecycle states.

    Defines the complete lifecycle of a task in the A2A protocol.
    Tasks transition through these states during execution.
    """

    PENDING = "pending"
    """Task submitted but not yet started."""

    WORKING = "working"
    """Task is actively being processed."""

    INPUT_REQUIRED = "input-required"
    """Task is blocked waiting for user input."""

    COMPLETED = "completed"
    """Task finished successfully."""

    FAILED = "failed"
    """Task failed with an error."""

    CANCELED = "canceled"
    """Task was canceled by user or system."""

    REJECTED = "rejected"
    """Task was rejected by the agent (e.g., policy violation)."""

    AUTH_REQUIRED = "auth-required"
    """Task requires authentication to proceed."""

    UNKNOWN = "unknown"
    """Task state cannot be determined."""


class MessageRole(Enum):
    """Role of the message sender."""

    USER = "user"
    """Message from a user or client agent."""

    AGENT = "agent"
    """Message from the receiving agent."""


class PartType(Enum):
    """Types of message content parts."""

    TEXT = "text"
    """Plain text content."""

    FILE = "file"
    """File content (inline or URI reference)."""

    DATA = "data"
    """Structured data (JSON-compatible)."""


class TransportProtocol(Enum):
    """A2A transport protocols."""

    JSONRPC = "JSONRPC"
    """JSON-RPC 2.0 over HTTP(S)."""

    GRPC = "grpc"
    """gRPC transport."""

    HTTP_JSON = "HTTP+JSON"
    """Plain HTTP with JSON payloads."""


class SecuritySchemeType(Enum):
    """Types of security schemes."""

    API_KEY = "apiKey"
    """API key authentication."""

    HTTP = "http"
    """HTTP authentication (Bearer, Basic, etc.)."""

    OAUTH2 = "oauth2"
    """OAuth 2.0 authentication."""

    OPENID_CONNECT = "openIdConnect"
    """OpenID Connect authentication."""

    MUTUAL_TLS = "mutualTLS"
    """Mutual TLS authentication."""


class ApiKeyLocation(Enum):
    """Location of API key in request."""

    HEADER = "header"
    QUERY = "query"
    COOKIE = "cookie"


# =============================================================================
# Error Codes
# =============================================================================


class A2AErrorCode(Enum):
    """JSON-RPC error codes for A2A protocol."""

    # Standard JSON-RPC errors
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # A2A-specific errors
    TASK_NOT_FOUND = -32001
    TASK_NOT_CANCELABLE = -32002
    PUSH_NOTIFICATIONS_UNSUPPORTED = -32003
    UNSUPPORTED_OPERATION = -32004
    CONTENT_TYPE_MISMATCH = -32005
    INVALID_AGENT_RESPONSE = -32006
    AUTH_EXTENDED_CARD_NOT_CONFIGURED = -32007


# =============================================================================
# Message Parts
# =============================================================================


@dataclass
class TextPart:
    """Text content part of a message."""

    text: str
    """The text content."""

    kind: str = field(default="text", init=False)
    """Part type discriminator."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Optional metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {"kind": self.kind, "text": self.text}
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextPart:
        """Create from dictionary."""
        return cls(
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class FileContent:
    """File content with either inline bytes or URI reference."""

    name: str | None = None
    """File name."""

    mime_type: str = "application/octet-stream"
    """MIME type of the file."""

    uri: str | None = None
    """URI reference to the file."""

    bytes_data: bytes | None = None
    """Inline file data (base64 encoded in JSON)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {"mimeType": self.mime_type}
        if self.name:
            result["name"] = self.name
        if self.uri:
            result["uri"] = self.uri
        if self.bytes_data:
            import base64
            result["bytes"] = base64.b64encode(self.bytes_data).decode("utf-8")
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileContent:
        """Create from dictionary."""
        bytes_data = None
        if "bytes" in data:
            import base64
            bytes_data = base64.b64decode(data["bytes"])
        return cls(
            name=data.get("name"),
            mime_type=data.get("mimeType", "application/octet-stream"),
            uri=data.get("uri"),
            bytes_data=bytes_data,
        )


@dataclass
class FilePart:
    """File content part of a message."""

    file: FileContent
    """The file content."""

    kind: str = field(default="file", init=False)
    """Part type discriminator."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Optional metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {"kind": self.kind, "file": self.file.to_dict()}
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FilePart:
        """Create from dictionary."""
        return cls(
            file=FileContent.from_dict(data.get("file", {})),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DataPart:
    """Structured data part of a message."""

    data: dict[str, Any]
    """The structured data."""

    kind: str = field(default="data", init=False)
    """Part type discriminator."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Optional metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {"kind": self.kind, "data": self.data}
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataPart:
        """Create from dictionary."""
        return cls(
            data=data.get("data", {}),
            metadata=data.get("metadata", {}),
        )


# Union type for message parts
MessagePart = TextPart | FilePart | DataPart


def parse_message_part(data: dict[str, Any]) -> MessagePart:
    """Parse a message part from dictionary based on kind discriminator."""
    kind = data.get("kind", "text")
    if kind == "text":
        return TextPart.from_dict(data)
    elif kind == "file":
        return FilePart.from_dict(data)
    elif kind == "data":
        return DataPart.from_dict(data)
    else:
        # Default to text for unknown kinds
        return TextPart(text=str(data))


# =============================================================================
# Security Schemes
# =============================================================================


@dataclass
class APIKeySecurityScheme:
    """API key security scheme."""

    name: str
    """Name of the header, query param, or cookie."""

    location: ApiKeyLocation = ApiKeyLocation.HEADER
    """Where the API key is passed."""

    type: str = field(default="apiKey", init=False)
    """Scheme type discriminator."""

    description: str = ""
    """Human-readable description."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "name": self.name,
            "in": self.location.value,
            "description": self.description,
        }


@dataclass
class HTTPSecurityScheme:
    """HTTP authentication security scheme."""

    scheme: str = "bearer"
    """HTTP auth scheme (bearer, basic, etc.)."""

    bearer_format: str | None = None
    """Format hint for bearer tokens (e.g., 'JWT')."""

    type: str = field(default="http", init=False)
    """Scheme type discriminator."""

    description: str = ""
    """Human-readable description."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "type": self.type,
            "scheme": self.scheme,
            "description": self.description,
        }
        if self.bearer_format:
            result["bearerFormat"] = self.bearer_format
        return result


@dataclass
class OAuth2SecurityScheme:
    """OAuth 2.0 security scheme."""

    flows: dict[str, Any]
    """OAuth 2.0 flow configurations."""

    type: str = field(default="oauth2", init=False)
    """Scheme type discriminator."""

    description: str = ""
    """Human-readable description."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "flows": self.flows,
            "description": self.description,
        }


SecurityScheme = APIKeySecurityScheme | HTTPSecurityScheme | OAuth2SecurityScheme


# =============================================================================
# Agent Skill
# =============================================================================


@dataclass
class AgentSkill:
    """A discrete capability offered by an agent.

    Skills represent specific tasks or functions an agent can perform.
    Each skill has its own input/output modes and optional examples.

    Attributes:
        id: Unique identifier for the skill.
        name: Human-readable name.
        description: Detailed description of what the skill does.
        tags: Keywords for categorization and discovery.
        examples: Example prompts that invoke this skill.
        input_modes: Supported input MIME types (overrides agent defaults).
        output_modes: Supported output MIME types (overrides agent defaults).
    """

    id: str
    """Unique identifier for the skill."""

    name: str
    """Human-readable skill name."""

    description: str = ""
    """Detailed description of the skill's capabilities."""

    tags: list[str] = field(default_factory=list)
    """Keywords for categorization."""

    examples: list[str] = field(default_factory=list)
    """Example prompts that invoke this skill."""

    input_modes: list[str] = field(default_factory=list)
    """Supported input MIME types."""

    output_modes: list[str] = field(default_factory=list)
    """Supported output MIME types."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional skill-specific metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
        }
        if self.description:
            result["description"] = self.description
        if self.tags:
            result["tags"] = self.tags
        if self.examples:
            result["examples"] = self.examples
        if self.input_modes:
            result["inputModes"] = self.input_modes
        if self.output_modes:
            result["outputModes"] = self.output_modes
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentSkill:
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            examples=data.get("examples", []),
            input_modes=data.get("inputModes", []),
            output_modes=data.get("outputModes", []),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Agent Provider
# =============================================================================


@dataclass
class AgentProvider:
    """Information about the organization providing an agent.

    Attributes:
        organization: Name of the providing organization.
        url: URL to the organization's website.
    """

    organization: str
    """Name of the providing organization."""

    url: str = ""
    """URL to the organization's website."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {"organization": self.organization}
        if self.url:
            result["url"] = self.url
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentProvider:
        """Create from dictionary."""
        return cls(
            organization=data.get("organization", ""),
            url=data.get("url", ""),
        )


# =============================================================================
# Agent Interface
# =============================================================================


@dataclass
class AgentInterface:
    """An endpoint interface for an agent.

    Defines how to communicate with an agent, including the URL
    and transport protocol.

    Attributes:
        url: Endpoint URL.
        transport: Transport protocol (JSONRPC, GRPC, HTTP+JSON).
    """

    url: str
    """Endpoint URL."""

    transport: TransportProtocol = TransportProtocol.JSONRPC
    """Transport protocol."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "transport": self.transport.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentInterface:
        """Create from dictionary."""
        transport_str = data.get("transport", "JSONRPC")
        try:
            transport = TransportProtocol(transport_str)
        except ValueError:
            transport = TransportProtocol.JSONRPC
        return cls(
            url=data.get("url", ""),
            transport=transport,
        )


# =============================================================================
# Agent Capabilities
# =============================================================================


@dataclass
class AgentCapabilities:
    """Capability flags for an agent.

    Indicates what optional features the agent supports.
    """

    streaming: bool = False
    """Supports streaming responses via SSE."""

    push_notifications: bool = False
    """Supports push notifications."""

    state_history: bool = False
    """Maintains and exposes task state history."""

    extensions: list[str] = field(default_factory=list)
    """List of supported extension URIs."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "streaming": self.streaming,
            "pushNotifications": self.push_notifications,
            "stateHistory": self.state_history,
            "extensions": self.extensions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentCapabilities:
        """Create from dictionary."""
        return cls(
            streaming=data.get("streaming", False),
            push_notifications=data.get("pushNotifications", False),
            state_history=data.get("stateHistory", False),
            extensions=data.get("extensions", []),
        )


# =============================================================================
# Agent Card
# =============================================================================


@dataclass
class AgentCard:
    """A2A Agent Card for capability discovery.

    The Agent Card is a JSON document served at /.well-known/agent.json
    that describes an agent's capabilities, skills, and connection details.

    This implements the A2A Protocol v0.3 specification.

    Attributes:
        name: Human-readable agent name.
        description: Description of agent capabilities.
        skills: List of discrete skills the agent offers.
        provider: Information about the providing organization.
        interfaces: List of endpoint interfaces.
        url: Primary endpoint URL.
        version: Agent version string.
        protocol_version: A2A protocol version (default: "0.3.0").
        capabilities: Optional capability flags.
        default_input_modes: Default supported input MIME types.
        default_output_modes: Default supported output MIME types.
        icon_url: URL to agent icon.
        documentation_url: URL to documentation.
        security_schemes: Available authentication schemes.
        security: Required security for accessing the agent.
    """

    name: str
    """Human-readable agent name."""

    description: str
    """Description of agent capabilities."""

    skills: list[AgentSkill] = field(default_factory=list)
    """List of discrete skills the agent offers."""

    provider: AgentProvider | None = None
    """Information about the providing organization."""

    interfaces: list[AgentInterface] = field(default_factory=list)
    """List of additional endpoint interfaces."""

    url: str = ""
    """Primary endpoint URL."""

    version: str = "1.0.0"
    """Agent version string."""

    protocol_version: str = "0.3.0"
    """A2A protocol version."""

    capabilities: AgentCapabilities | None = None
    """Optional capability flags."""

    default_input_modes: list[str] = field(default_factory=lambda: ["text/plain"])
    """Default supported input MIME types."""

    default_output_modes: list[str] = field(default_factory=lambda: ["text/plain"])
    """Default supported output MIME types."""

    icon_url: str = ""
    """URL to agent icon."""

    documentation_url: str = ""
    """URL to documentation."""

    security_schemes: dict[str, SecurityScheme] = field(default_factory=dict)
    """Available authentication schemes."""

    security: list[dict[str, list[str]]] = field(default_factory=list)
    """Required security for accessing the agent."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional agent metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "protocolVersion": self.protocol_version,
            "version": self.version,
            "skills": [s.to_dict() for s in self.skills],
            "defaultInputModes": self.default_input_modes,
            "defaultOutputModes": self.default_output_modes,
        }

        if self.url:
            result["url"] = self.url
        if self.provider:
            result["provider"] = self.provider.to_dict()
        if self.interfaces:
            result["additionalInterfaces"] = [i.to_dict() for i in self.interfaces]
        if self.capabilities:
            result["capabilities"] = self.capabilities.to_dict()
        if self.icon_url:
            result["iconUrl"] = self.icon_url
        if self.documentation_url:
            result["documentationUrl"] = self.documentation_url
        if self.security_schemes:
            result["securitySchemes"] = {
                k: v.to_dict() for k, v in self.security_schemes.items()
            }
        if self.security:
            result["security"] = self.security
        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentCard:
        """Create from dictionary."""
        skills = [AgentSkill.from_dict(s) for s in data.get("skills", [])]
        interfaces = [
            AgentInterface.from_dict(i)
            for i in data.get("additionalInterfaces", [])
        ]

        provider = None
        if "provider" in data:
            provider = AgentProvider.from_dict(data["provider"])

        capabilities = None
        if "capabilities" in data:
            capabilities = AgentCapabilities.from_dict(data["capabilities"])

        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            skills=skills,
            provider=provider,
            interfaces=interfaces,
            url=data.get("url", ""),
            version=data.get("version", "1.0.0"),
            protocol_version=data.get("protocolVersion", "0.3.0"),
            capabilities=capabilities,
            default_input_modes=data.get("defaultInputModes", ["text/plain"]),
            default_output_modes=data.get("defaultOutputModes", ["text/plain"]),
            icon_url=data.get("iconUrl", ""),
            documentation_url=data.get("documentationUrl", ""),
            security=data.get("security", []),
            metadata=data.get("metadata", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent)

    def get_skill(self, skill_id: str) -> AgentSkill | None:
        """Get a skill by ID."""
        for skill in self.skills:
            if skill.id == skill_id:
                return skill
        return None

    def has_skill(self, skill_id: str) -> bool:
        """Check if agent has a skill."""
        return self.get_skill(skill_id) is not None


# =============================================================================
# Message
# =============================================================================


@dataclass
class Message:
    """A single conversation turn in the A2A protocol.

    Messages represent communication between users and agents.
    Each message has a role (user or agent) and contains one or
    more content parts.

    Attributes:
        role: The sender's role (user or agent).
        content: Text content for simple messages.
        parts: Structured content parts for complex messages.
        context_id: Optional context/conversation identifier.
        message_id: Unique message identifier.
        task_id: Associated task ID (if part of a task).
        metadata: Additional message metadata.
    """

    role: MessageRole
    """The sender's role."""

    content: str = ""
    """Text content for simple messages."""

    parts: list[MessagePart] = field(default_factory=list)
    """Structured content parts."""

    context_id: str | None = None
    """Optional context/conversation identifier."""

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique message identifier."""

    task_id: str | None = None
    """Associated task ID."""

    reference_task_ids: list[str] = field(default_factory=list)
    """Related task IDs for context."""

    extensions: list[str] = field(default_factory=list)
    """Extension URIs used in this message."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional message metadata."""

    kind: str = field(default="message", init=False)
    """Type discriminator."""

    def __post_init__(self) -> None:
        """Ensure parts list is populated from content if needed."""
        if self.content and not self.parts:
            self.parts = [TextPart(text=self.content)]

    @classmethod
    def user(cls, content: str, **kwargs) -> Message:
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content, **kwargs)

    @classmethod
    def agent(cls, content: str, **kwargs) -> Message:
        """Create an agent message."""
        return cls(role=MessageRole.AGENT, content=content, **kwargs)

    def get_text(self) -> str:
        """Extract all text content from the message."""
        texts = []
        for part in self.parts:
            if isinstance(part, TextPart):
                texts.append(part.text)
        return "\n".join(texts) if texts else self.content

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "kind": self.kind,
            "messageId": self.message_id,
            "role": self.role.value,
            "parts": [p.to_dict() for p in self.parts],
        }

        if self.context_id:
            result["contextId"] = self.context_id
        if self.task_id:
            result["taskId"] = self.task_id
        if self.reference_task_ids:
            result["referenceTaskIds"] = self.reference_task_ids
        if self.extensions:
            result["extensions"] = self.extensions
        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Create from dictionary."""
        parts = [parse_message_part(p) for p in data.get("parts", [])]

        return cls(
            role=MessageRole(data.get("role", "user")),
            content="",
            parts=parts,
            context_id=data.get("contextId"),
            message_id=data.get("messageId", str(uuid.uuid4())),
            task_id=data.get("taskId"),
            reference_task_ids=data.get("referenceTaskIds", []),
            extensions=data.get("extensions", []),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Artifact
# =============================================================================


@dataclass
class Artifact:
    """A task output artifact.

    Artifacts represent outputs produced by task execution,
    such as generated files, code, or structured data.

    Attributes:
        id: Unique artifact identifier.
        name: Human-readable artifact name.
        mime_type: Content MIME type.
        content: Inline content (for small artifacts).
        uri: URI reference (for large artifacts).
        description: Description of the artifact.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique artifact identifier."""

    name: str = ""
    """Human-readable artifact name."""

    mime_type: str = "text/plain"
    """Content MIME type."""

    content: Any = None
    """Inline content."""

    uri: str | None = None
    """URI reference for external content."""

    description: str = ""
    """Description of the artifact."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional artifact metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "mimeType": self.mime_type,
        }

        if self.content is not None:
            result["content"] = self.content
        if self.uri:
            result["uri"] = self.uri
        if self.description:
            result["description"] = self.description
        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Artifact:
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            mime_type=data.get("mimeType", "text/plain"),
            content=data.get("content"),
            uri=data.get("uri"),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Task Status
# =============================================================================


@dataclass
class TaskStatus:
    """Current status of a task.

    Provides a snapshot of task state with optional message.

    Attributes:
        state: Current task state.
        message: Optional human-readable status message.
        timestamp: When this status was recorded.
    """

    state: TaskState
    """Current task state."""

    message: str | None = None
    """Optional human-readable status message."""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    """When this status was recorded."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "state": self.state.value,
            "timestamp": self.timestamp.isoformat() + "Z",
        }
        if self.message:
            result["message"] = self.message
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskStatus:
        """Create from dictionary."""
        timestamp = datetime.utcnow()
        if "timestamp" in data:
            try:
                ts_str = data["timestamp"].replace("Z", "+00:00")
                timestamp = datetime.fromisoformat(ts_str)
            except (ValueError, AttributeError):
                pass

        return cls(
            state=TaskState(data.get("state", "pending")),
            message=data.get("message"),
            timestamp=timestamp,
        )


# =============================================================================
# Task
# =============================================================================


@dataclass
class Task:
    """A stateful operation between client and agent.

    Tasks represent the primary unit of work in the A2A protocol.
    They maintain conversation history, artifacts, and state.

    Attributes:
        id: Server-generated unique identifier.
        state: Current task state.
        messages: Conversation history.
        artifacts: Task outputs.
        context_id: Optional context identifier.
        status: Detailed status information.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Server-generated unique identifier."""

    state: TaskState = TaskState.PENDING
    """Current task state."""

    messages: list[Message] = field(default_factory=list)
    """Conversation history."""

    artifacts: list[Artifact] = field(default_factory=list)
    """Task outputs."""

    context_id: str | None = None
    """Optional context identifier."""

    status: TaskStatus | None = None
    """Detailed status information."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional task metadata."""

    kind: str = field(default="task", init=False)
    """Type discriminator."""

    @property
    def is_complete(self) -> bool:
        """Check if task is in a terminal state."""
        return self.state in (
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELED,
            TaskState.REJECTED,
        )

    @property
    def is_successful(self) -> bool:
        """Check if task completed successfully."""
        return self.state == TaskState.COMPLETED

    @property
    def requires_input(self) -> bool:
        """Check if task is waiting for input."""
        return self.state == TaskState.INPUT_REQUIRED

    @property
    def last_message(self) -> Message | None:
        """Get the most recent message."""
        return self.messages[-1] if self.messages else None

    @property
    def agent_response(self) -> str:
        """Get the last agent response text."""
        for msg in reversed(self.messages):
            if msg.role == MessageRole.AGENT:
                return msg.get_text()
        return ""

    def add_message(self, message: Message) -> None:
        """Add a message to the task history."""
        if not message.task_id:
            message.task_id = self.id
        self.messages.append(message)

    def add_artifact(self, artifact: Artifact) -> None:
        """Add an artifact to the task."""
        self.artifacts.append(artifact)

    def update_state(self, state: TaskState, message: str | None = None) -> None:
        """Update task state and status."""
        self.state = state
        self.status = TaskStatus(state=state, message=message)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "kind": self.kind,
            "id": self.id,
            "status": self.status.to_dict() if self.status else {"state": self.state.value},
            "history": [m.to_dict() for m in self.messages],
            "artifacts": [a.to_dict() for a in self.artifacts],
        }

        if self.context_id:
            result["contextId"] = self.context_id
        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """Create from dictionary."""
        messages = [Message.from_dict(m) for m in data.get("history", [])]
        artifacts = [Artifact.from_dict(a) for a in data.get("artifacts", [])]

        status_data = data.get("status", {})
        state = TaskState(status_data.get("state", "pending"))
        status = TaskStatus.from_dict(status_data) if status_data else None

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            state=state,
            messages=messages,
            artifacts=artifacts,
            context_id=data.get("contextId"),
            status=status,
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# JSON-RPC Types
# =============================================================================


@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 request."""

    method: str
    """RPC method name."""

    params: dict[str, Any] = field(default_factory=dict)
    """Method parameters."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Request identifier."""

    jsonrpc: str = field(default="2.0", init=False)
    """Protocol version."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "method": self.method,
            "params": self.params,
        }


@dataclass
class JSONRPCError:
    """JSON-RPC 2.0 error."""

    code: int
    """Error code."""

    message: str
    """Error message."""

    data: Any = None
    """Additional error data."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.data is not None:
            result["data"] = self.data
        return result

    @classmethod
    def from_code(cls, code: A2AErrorCode, message: str | None = None, data: Any = None) -> JSONRPCError:
        """Create from error code enum."""
        default_messages = {
            A2AErrorCode.PARSE_ERROR: "Parse error",
            A2AErrorCode.INVALID_REQUEST: "Invalid request",
            A2AErrorCode.METHOD_NOT_FOUND: "Method not found",
            A2AErrorCode.INVALID_PARAMS: "Invalid params",
            A2AErrorCode.INTERNAL_ERROR: "Internal error",
            A2AErrorCode.TASK_NOT_FOUND: "Task not found",
            A2AErrorCode.TASK_NOT_CANCELABLE: "Task not cancelable",
        }
        return cls(
            code=code.value,
            message=message or default_messages.get(code, "Unknown error"),
            data=data,
        )


@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0 response."""

    id: str
    """Request identifier."""

    result: Any = None
    """Result on success."""

    error: JSONRPCError | None = None
    """Error on failure."""

    jsonrpc: str = field(default="2.0", init=False)
    """Protocol version."""

    @property
    def is_error(self) -> bool:
        """Check if this is an error response."""
        return self.error is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
        }
        if self.error:
            result["error"] = self.error.to_dict()
        else:
            result["result"] = self.result
        return result

    @classmethod
    def success(cls, id: str, result: Any) -> JSONRPCResponse:
        """Create a success response."""
        return cls(id=id, result=result)

    @classmethod
    def failure(cls, id: str, error: JSONRPCError) -> JSONRPCResponse:
        """Create an error response."""
        return cls(id=id, error=error)


# =============================================================================
# Streaming Events
# =============================================================================


@dataclass
class TaskStatusEvent:
    """Event indicating task status change."""

    task_id: str
    """Task identifier."""

    status: TaskStatus
    """New status."""

    kind: str = field(default="task_status", init=False)
    """Event type discriminator."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "kind": self.kind,
            "taskId": self.task_id,
            "status": self.status.to_dict(),
        }


@dataclass
class MessageEvent:
    """Event containing a new message."""

    message: Message
    """The message content."""

    task_id: str | None = None
    """Associated task ID."""

    kind: str = field(default="message", init=False)
    """Event type discriminator."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "kind": self.kind,
            "message": self.message.to_dict(),
        }
        if self.task_id:
            result["taskId"] = self.task_id
        return result


@dataclass
class ArtifactEvent:
    """Event containing a new artifact."""

    artifact: Artifact
    """The artifact content."""

    task_id: str
    """Associated task ID."""

    kind: str = field(default="artifact", init=False)
    """Event type discriminator."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "kind": self.kind,
            "taskId": self.task_id,
            "artifact": self.artifact.to_dict(),
        }


# Union type for streaming events
StreamEvent = TaskStatusEvent | MessageEvent | ArtifactEvent
