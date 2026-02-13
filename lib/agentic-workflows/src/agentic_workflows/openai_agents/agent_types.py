"""Agent type definitions compatible with OpenAI Agents SDK patterns.

This module provides dataclasses and types that mirror the OpenAI Agents SDK
architecture while integrating with agentic_workflows internal patterns.

Key types:
- AgentConfig: Core agent configuration
- HandoffConfig: Agent-to-agent transfer configuration
- GuardrailConfig: Input/output validation configuration
- SessionConfig: Conversation history management
- RunConfig: Execution parameters

Reference: https://github.com/openai/openai-agents-python
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar, Generic, TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_workflows.openai_agents.agent import OpenAIAgent
    from agentic_workflows.openai_agents.guardrails import InputGuardrail, OutputGuardrail


# Type variable for structured output
T = TypeVar("T")


class ModelProvider(Enum):
    """Supported model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    CUSTOM = "custom"


class HandoffStrategy(Enum):
    """Strategies for handling agent handoffs."""

    IMMEDIATE = "immediate"  # Transfer immediately on handoff call
    DEFERRED = "deferred"  # Queue handoff for later processing
    CONDITIONAL = "conditional"  # Transfer only if conditions met
    ROUND_ROBIN = "round_robin"  # Distribute among multiple targets


class OutputType(Enum):
    """Types of output format."""

    TEXT = "text"
    JSON = "json"
    STRUCTURED = "structured"
    STREAMING = "streaming"


@dataclass
class ToolConfig:
    """Configuration for a tool available to an agent.

    Attributes:
        name: Unique tool identifier.
        description: Human-readable description for the LLM.
        parameters: JSON schema for tool parameters.
        handler: Callable that executes the tool.
        requires_confirmation: Whether to require user confirmation before execution.
        timeout_seconds: Maximum execution time.
        retry_count: Number of retries on failure.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    handler: Callable[..., Any] | None = None
    requires_confirmation: bool = False
    timeout_seconds: float = 30.0
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function schema format.

        Returns:
            OpenAI-compatible function schema.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Convert to Anthropic tool schema format.

        Returns:
            Anthropic-compatible tool schema.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters or {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }


@dataclass
class MCPServerConfig:
    """Configuration for an MCP (Model Context Protocol) server.

    Attributes:
        name: Server identifier.
        url: Server endpoint URL.
        transport: Transport protocol (stdio, http, ws).
        tools: List of tool names to import from server.
        auth: Authentication configuration.
    """

    name: str
    url: str = ""
    transport: str = "stdio"  # "stdio", "http", "websocket"
    tools: list[str] = field(default_factory=list)
    command: str = ""  # For stdio transport
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    auth: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 60.0


@dataclass
class HandoffConfig:
    """Configuration for agent-to-agent handoffs.

    Mirrors OpenAI Agents SDK handoff patterns while integrating
    with agentic_workflows HandoffManager.

    Attributes:
        target_agent: The agent to hand off to.
        description: When to use this handoff (shown to LLM).
        strategy: Handoff execution strategy.
        conditions: Conditions that must be met for handoff.
        context_filter: Fields to include in handoff context.
        on_handoff: Callback when handoff occurs.
    """

    target_agent: "OpenAIAgent | str"
    description: str = ""
    strategy: HandoffStrategy = HandoffStrategy.IMMEDIATE
    conditions: list[Callable[[dict[str, Any]], bool]] = field(default_factory=list)
    context_filter: list[str] | None = None  # None = include all
    preserve_history: bool = True
    on_handoff: Callable[[dict[str, Any]], None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def should_handoff(self, context: dict[str, Any]) -> bool:
        """Check if handoff conditions are met.

        Args:
            context: Current execution context.

        Returns:
            True if all conditions pass.
        """
        if not self.conditions:
            return True
        return all(condition(context) for condition in self.conditions)

    def filter_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """Filter context for handoff.

        Args:
            context: Full execution context.

        Returns:
            Filtered context based on context_filter.
        """
        if self.context_filter is None:
            return context
        return {k: v for k, v in context.items() if k in self.context_filter}


@dataclass
class GuardrailConfig:
    """Configuration for input/output guardrails.

    Attributes:
        name: Guardrail identifier.
        type: "input" or "output".
        validator: Validation function.
        on_fail: Action on validation failure.
        message: Custom failure message.
        enabled: Whether guardrail is active.
    """

    name: str
    type: str = "input"  # "input" or "output"
    validator: Callable[[str], tuple[bool, str]] | None = None
    on_fail: str = "block"  # "block", "warn", "sanitize"
    message: str = ""
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, content: str) -> tuple[bool, str]:
        """Run validation on content.

        Args:
            content: Content to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not self.enabled:
            return True, ""

        if self.validator is None:
            return True, ""

        return self.validator(content)


@dataclass
class SessionConfig:
    """Configuration for conversation session management.

    Supports multiple storage backends for conversation history.

    Attributes:
        session_id: Unique session identifier.
        storage_type: Backend type (memory, sqlite, redis, postgres).
        max_history: Maximum messages to retain.
        ttl_seconds: Session expiration time.
        compression: Whether to compress stored messages.
    """

    session_id: str = ""
    storage_type: str = "memory"  # "memory", "sqlite", "redis", "postgres"
    storage_config: dict[str, Any] = field(default_factory=dict)
    max_history: int = 100
    ttl_seconds: float | None = None  # None = no expiration
    compression: bool = False
    auto_persist: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    """Configuration for agent run execution.

    Controls the agent loop behavior and output handling.

    Attributes:
        max_turns: Maximum LLM call iterations.
        output_type: Expected output format.
        output_schema: JSON schema for structured output.
        stream: Enable streaming output.
        parallel_tool_calls: Allow parallel tool execution.
        timeout_seconds: Total run timeout.
    """

    max_turns: int = 10
    output_type: OutputType = OutputType.TEXT
    output_schema: type[Any] | dict[str, Any] | None = None
    stream: bool = False
    parallel_tool_calls: bool = True
    timeout_seconds: float = 300.0
    trace_enabled: bool = True
    trace_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # Budget limits
    max_tokens: int | None = None
    max_cost_usd: float | None = None

    # Retry configuration
    retry_on_failure: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class AgentConfig:
    """Main agent configuration compatible with OpenAI Agents SDK.

    This is the primary configuration class for creating OpenAI-style agents
    that integrate with agentic_workflows.

    Attributes:
        name: Agent identifier.
        instructions: System prompt / instructions for the agent.
        model: LLM model identifier.
        tools: List of available tools.
        handoffs: List of handoff configurations.
        guardrails: Input/output validation rules.
        mcp_servers: MCP servers to connect to.
        session_config: Conversation history configuration.
    """

    # Core identity
    name: str
    instructions: str = ""
    description: str = ""

    # Model configuration
    model: str = "gpt-4o"
    provider: ModelProvider = ModelProvider.OPENAI
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Tools and capabilities
    tools: list[ToolConfig] = field(default_factory=list)
    handoffs: list[HandoffConfig] = field(default_factory=list)
    guardrails: list[GuardrailConfig] = field(default_factory=list)

    # MCP integration
    mcp_servers: list[MCPServerConfig] = field(default_factory=list)

    # Session management
    session_config: SessionConfig | None = None

    # Default run configuration
    default_run_config: RunConfig = field(default_factory=RunConfig)

    # Metadata
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Validate the configuration.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        if not self.name:
            errors.append("Agent name is required")
        elif not str(self.name).replace("_", "").replace("-", "").isalnum():
            errors.append("Agent name must be alphanumeric (with _ or -)")

        if self.temperature < 0 or self.temperature > 2:
            errors.append("Temperature must be between 0 and 2")

        if self.max_tokens < 1:
            errors.append("Max tokens must be positive")

        # Validate tools
        tool_names = set()
        for tool in self.tools:
            if not tool.name:
                errors.append("Tool name is required")
            elif tool.name in tool_names:
                errors.append(f"Duplicate tool name: {tool.name}")
            else:
                tool_names.add(tool.name)

        # Validate handoffs
        for handoff in self.handoffs:
            if not handoff.target_agent:
                errors.append("Handoff target agent is required")

        return errors

    def get_input_guardrails(self) -> list[GuardrailConfig]:
        """Get all input guardrails."""
        return [g for g in self.guardrails if g.type == "input"]

    def get_output_guardrails(self) -> list[GuardrailConfig]:
        """Get all output guardrails."""
        return [g for g in self.guardrails if g.type == "output"]

    def get_tool_by_name(self, name: str) -> ToolConfig | None:
        """Get tool configuration by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def add_tool(self, tool: ToolConfig) -> None:
        """Add a tool to the agent."""
        self.tools.append(tool)

    def add_handoff(self, handoff: HandoffConfig) -> None:
        """Add a handoff target to the agent."""
        self.handoffs.append(handoff)

    def add_guardrail(self, guardrail: GuardrailConfig) -> None:
        """Add a guardrail to the agent."""
        self.guardrails.append(guardrail)


@dataclass
class Message:
    """A message in the conversation.

    Compatible with both OpenAI and Anthropic message formats.
    """

    role: str  # "user", "assistant", "system", "tool"
    content: str | list[dict[str, Any]]
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI message format."""
        msg = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic message format."""
        # Anthropic uses different role names
        role_map = {"user": "user", "assistant": "assistant", "tool": "user"}
        return {
            "role": role_map.get(self.role, self.role),
            "content": self.content,
        }


@dataclass
class ToolCall:
    """A tool call request from the LLM.

    Attributes:
        id: Unique call identifier.
        name: Tool name.
        arguments: Tool arguments.
        result: Tool execution result (populated after execution).
    """

    id: str
    name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None
    duration_ms: float | None = None

    @property
    def succeeded(self) -> bool:
        """Check if tool call succeeded."""
        return self.error is None and self.result is not None


@dataclass
class HandoffRequest:
    """A request to hand off to another agent.

    Attributes:
        target_agent: Agent to hand off to.
        reason: Why handoff is occurring.
        context: Context to pass to target.
        messages: Conversation history to transfer.
    """

    target_agent: str
    reason: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# Type alias for output types
OutputT = TypeVar("OutputT")


@dataclass
class RunResult(Generic[OutputT]):
    """Result from an agent run.

    Attributes:
        output: Final output (text or structured).
        messages: Full conversation history.
        tool_calls: All tool calls made.
        handoffs: Any handoffs that occurred.
        metadata: Execution metadata.
    """

    output: OutputT
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    handoffs: list[HandoffRequest] = field(default_factory=list)

    # Execution metrics
    total_turns: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    duration_seconds: float = 0.0

    # Status
    success: bool = True
    error: str | None = None
    final_agent: str = ""

    # Tracing
    trace_id: str = ""
    span_id: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def last_message(self) -> Message | None:
        """Get the last message in the conversation."""
        return self.messages[-1] if self.messages else None

    @property
    def assistant_messages(self) -> list[Message]:
        """Get all assistant messages."""
        return [m for m in self.messages if m.role == "assistant"]


@dataclass
class RunResultStreaming(Generic[OutputT]):
    """Streaming result from an agent run.

    Provides async iteration over response chunks.
    """

    def __init__(self):
        self._chunks: list[str] = []
        self._complete: bool = False
        self._final_result: RunResult[OutputT] | None = None

    def add_chunk(self, chunk: str) -> None:
        """Add a chunk to the stream."""
        self._chunks.append(chunk)

    def complete(self, result: RunResult[OutputT]) -> None:
        """Mark stream as complete with final result."""
        self._complete = True
        self._final_result = result

    @property
    def is_complete(self) -> bool:
        """Check if stream is complete."""
        return self._complete

    @property
    def text_so_far(self) -> str:
        """Get all text received so far."""
        return "".join(self._chunks)

    @property
    def final_result(self) -> RunResult[OutputT] | None:
        """Get final result if complete."""
        return self._final_result

    async def __aiter__(self):
        """Async iterate over chunks."""
        for chunk in self._chunks:
            yield chunk
