"""MCP Sampling Module.

Provides sampling/completion capabilities for MCP protocol.

Usage:
    from agentic_workflows.mcp.sampling import SamplingManager, SamplingRequest

    manager = SamplingManager()
    result = await manager.create_message(SamplingRequest(
        messages=[{"role": "user", "content": "Hello"}],
        maxTokens=100,
    ))
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class StopReason(Enum):
    """Reasons for stopping generation."""

    END_TURN = "endTurn"
    STOP_SEQUENCE = "stopSequence"
    MAX_TOKENS = "maxTokens"


@dataclass
class SamplingMessage:
    """Message for sampling request.

    Attributes:
        role: Message role (user or assistant).
        content: Message content.
    """

    role: str
    content: str | dict[str, Any] | list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to MCP format."""
        content = self.content
        if isinstance(content, str):
            content = {"type": "text", "text": content}
        return {
            "role": self.role,
            "content": content,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SamplingMessage:
        """Create from MCP format."""
        return cls(
            role=data["role"],
            content=data["content"],
        )


@dataclass
class ModelPreferences:
    """Model preferences for sampling.

    Attributes:
        hints: Model name hints.
        costPriority: Priority for cost optimization (0-1).
        speedPriority: Priority for speed (0-1).
        intelligencePriority: Priority for intelligence (0-1).
    """

    hints: list[dict[str, str]] | None = None
    costPriority: float | None = None
    speedPriority: float | None = None
    intelligencePriority: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to MCP format."""
        result = {}
        if self.hints:
            result["hints"] = self.hints
        if self.costPriority is not None:
            result["costPriority"] = self.costPriority
        if self.speedPriority is not None:
            result["speedPriority"] = self.speedPriority
        if self.intelligencePriority is not None:
            result["intelligencePriority"] = self.intelligencePriority
        return result


@dataclass
class SamplingRequest:
    """Request for sampling/createMessage.

    Attributes:
        messages: Conversation messages.
        modelPreferences: Model selection preferences.
        systemPrompt: System prompt.
        includeContext: Context inclusion mode.
        temperature: Sampling temperature.
        maxTokens: Maximum tokens to generate.
        stopSequences: Stop sequences.
        metadata: Additional metadata.
    """

    messages: list[SamplingMessage] = field(default_factory=list)
    modelPreferences: ModelPreferences | None = None
    systemPrompt: str | None = None
    includeContext: str = "none"  # "none", "thisServer", "allServers"
    temperature: float | None = None
    maxTokens: int = 1024
    stopSequences: list[str] | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to MCP format."""
        result = {
            "messages": [m.to_dict() for m in self.messages],
            "maxTokens": self.maxTokens,
        }
        if self.modelPreferences:
            result["modelPreferences"] = self.modelPreferences.to_dict()
        if self.systemPrompt:
            result["systemPrompt"] = self.systemPrompt
        if self.includeContext != "none":
            result["includeContext"] = self.includeContext
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.stopSequences:
            result["stopSequences"] = self.stopSequences
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SamplingRequest:
        """Create from MCP format."""
        messages = [SamplingMessage.from_dict(m) for m in data.get("messages", [])]
        model_prefs = None
        if "modelPreferences" in data:
            mp = data["modelPreferences"]
            model_prefs = ModelPreferences(
                hints=mp.get("hints"),
                costPriority=mp.get("costPriority"),
                speedPriority=mp.get("speedPriority"),
                intelligencePriority=mp.get("intelligencePriority"),
            )
        return cls(
            messages=messages,
            modelPreferences=model_prefs,
            systemPrompt=data.get("systemPrompt"),
            includeContext=data.get("includeContext", "none"),
            temperature=data.get("temperature"),
            maxTokens=data.get("maxTokens", 1024),
            stopSequences=data.get("stopSequences"),
            metadata=data.get("metadata"),
        )


@dataclass
class SamplingResult:
    """Result from sampling/createMessage.

    Attributes:
        role: Response role.
        content: Response content.
        model: Model used.
        stopReason: Reason for stopping.
    """

    role: str
    content: str | dict[str, Any]
    model: str
    stopReason: StopReason | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to MCP format."""
        content = self.content
        if isinstance(content, str):
            content = {"type": "text", "text": content}
        result = {
            "role": self.role,
            "content": content,
            "model": self.model,
        }
        if self.stopReason:
            result["stopReason"] = self.stopReason.value
        return result

    @property
    def text(self) -> str:
        """Extract text content."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, dict):
            return self.content.get("text", "")
        return ""


# Type for sampling handlers
SamplingHandler = Callable[[SamplingRequest], SamplingResult]


class SamplingManager:
    """Manages MCP sampling requests.

    This allows MCP servers to request LLM completions from the client.

    Example:
        manager = SamplingManager()

        # Set handler for processing requests
        manager.set_handler(my_llm_handler)

        # Process sampling request
        result = await manager.create_message(SamplingRequest(
            messages=[SamplingMessage(role="user", content="Hello")],
            maxTokens=100,
        ))
    """

    def __init__(self):
        """Initialize sampling manager."""
        self._handler: SamplingHandler | None = None
        self._model_map: dict[str, str] = {
            "claude-3-opus": "claude-opus-4-0-20250514",
            "claude-3-sonnet": "claude-sonnet-4-20250514",
            "claude-3-haiku": "claude-3-5-haiku-20241022",
            "opus": "claude-opus-4-0-20250514",
            "sonnet": "claude-sonnet-4-20250514",
            "haiku": "claude-3-5-haiku-20241022",
        }
        self._default_model = "claude-sonnet-4-20250514"

    def set_handler(self, handler: SamplingHandler) -> None:
        """Set handler for sampling requests.

        Args:
            handler: Function to handle sampling requests.
        """
        self._handler = handler

    def set_model_map(self, model_map: dict[str, str]) -> None:
        """Set model name mapping.

        Args:
            model_map: Map from hints to actual model names.
        """
        self._model_map = model_map

    def set_default_model(self, model: str) -> None:
        """Set default model.

        Args:
            model: Default model name.
        """
        self._default_model = model

    def _select_model(self, preferences: ModelPreferences | None) -> str:
        """Select model based on preferences.

        Args:
            preferences: Model preferences.

        Returns:
            Selected model name.
        """
        if not preferences or not preferences.hints:
            return self._default_model

        # Check hints against model map
        for hint in preferences.hints:
            hint_name = hint.get("name", "")
            if hint_name in self._model_map:
                return self._model_map[hint_name]

        # Use priority-based selection
        if preferences.intelligencePriority and preferences.intelligencePriority > 0.7:
            return self._model_map.get("opus", self._default_model)
        if preferences.speedPriority and preferences.speedPriority > 0.7:
            return self._model_map.get("haiku", self._default_model)

        return self._default_model

    async def create_message(self, request: SamplingRequest) -> SamplingResult:
        """Create a message (sampling request).

        Args:
            request: Sampling request.

        Returns:
            Sampling result.

        Raises:
            RuntimeError: If no handler configured.
        """
        if not self._handler:
            # Use built-in Claude handler
            return await self._default_handler(request)

        return self._handler(request)

    async def _default_handler(self, request: SamplingRequest) -> SamplingResult:
        """Default handler using Anthropic API.

        Args:
            request: Sampling request.

        Returns:
            Sampling result.
        """
        try:
            import anthropic

            client = anthropic.Anthropic()
            model = self._select_model(request.modelPreferences)

            # Convert messages
            messages = []
            for msg in request.messages:
                content = msg.content
                if isinstance(content, dict) and content.get("type") == "text":
                    content = content["text"]
                messages.append(
                    {
                        "role": msg.role,
                        "content": content,
                    }
                )

            # Create message
            kwargs = {
                "model": model,
                "max_tokens": request.maxTokens,
                "messages": messages,
            }
            if request.systemPrompt:
                kwargs["system"] = request.systemPrompt
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.stopSequences:
                kwargs["stop_sequences"] = request.stopSequences

            response = client.messages.create(**kwargs)

            # Map stop reason
            stop_reason = None
            if response.stop_reason == "end_turn":
                stop_reason = StopReason.END_TURN
            elif response.stop_reason == "stop_sequence":
                stop_reason = StopReason.STOP_SEQUENCE
            elif response.stop_reason == "max_tokens":
                stop_reason = StopReason.MAX_TOKENS

            # Extract text
            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text

            return SamplingResult(
                role="assistant",
                content=text,
                model=model,
                stopReason=stop_reason,
            )

        except ImportError:
            raise RuntimeError("anthropic package not installed")
        except Exception as e:
            logger.error(f"Sampling failed: {e}")
            raise RuntimeError(f"Sampling failed: {e}")

    async def handle_request(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle MCP sampling/createMessage request.

        Args:
            params: MCP request params.

        Returns:
            MCP response.
        """
        request = SamplingRequest.from_dict(params)
        result = await self.create_message(request)
        return result.to_dict()


__all__ = [
    "StopReason",
    "SamplingMessage",
    "ModelPreferences",
    "SamplingRequest",
    "SamplingResult",
    "SamplingHandler",
    "SamplingManager",
]
