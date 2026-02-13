"""Claude agent wrapper with native Anthropic SDK integration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

import anthropic
from anthropic.types import Message, MessageParam, ToolUseBlock, TextBlock

from ..observability.metrics import MetricsCollector, Model
from ..security.injection_defense import PromptInjectionDefense, ScanResult


class SecurityError(Exception):
    """Raised when security check fails."""

    def __init__(self, message: str, scan_result: ScanResult | None = None):
        super().__init__(message)
        self.scan_result = scan_result


@dataclass
class ClaudeAgentConfig:
    """Configuration for Claude agent."""

    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 4096
    temperature: float = 1.0
    system_prompt: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    stop_sequences: list[str] | None = None


@dataclass
class ToolResult:
    """Result from a tool execution."""

    tool_use_id: str
    content: str
    is_error: bool = False


class ClaudeAgent:
    """Agent wrapper for Anthropic Claude API.

    Provides:
    - Message thread management
    - Tool use handling with automatic execution
    - Token/cost tracking via MetricsCollector
    - Streaming support
    - Injection defense integration
    """

    MODEL_MAPPING = {
        "opus": Model.OPUS,
        "sonnet": Model.SONNET,
        "haiku": Model.HAIKU,
    }

    def __init__(
        self,
        agent_id: str,
        config: ClaudeAgentConfig,
        metrics: MetricsCollector | None = None,
        defense: PromptInjectionDefense | None = None,
        tool_handlers: dict[str, Callable] | None = None,
        api_key: str | None = None,
    ):
        """Initialize Claude agent.

        Args:
            agent_id: Unique identifier for this agent.
            config: Agent configuration.
            metrics: Optional metrics collector for token/cost tracking.
            defense: Optional injection defense for input scanning.
            tool_handlers: Dict mapping tool names to handler functions.
            api_key: Optional API key (uses ANTHROPIC_API_KEY env var if not provided).
        """
        self.agent_id = agent_id
        self.config = config
        self.metrics = metrics
        self.defense = defense
        self.tool_handlers = tool_handlers or {}

        # Initialize clients
        client_kwargs = {"api_key": api_key} if api_key else {}
        self._client = anthropic.Anthropic(**client_kwargs)
        self._async_client = anthropic.AsyncAnthropic(**client_kwargs)
        self._message_history: list[MessageParam] = []

    async def send_message(
        self,
        content: str,
        *,
        scan_input: bool = True,
        auto_execute_tools: bool = True,
        max_tool_iterations: int = 10,
    ) -> Message:
        """Send message and get response.

        Args:
            content: User message content.
            scan_input: Whether to scan for injection attacks.
            auto_execute_tools: Whether to automatically execute tool calls.
            max_tool_iterations: Max tool call iterations to prevent infinite loops.

        Returns:
            Claude's response message.

        Raises:
            SecurityError: If injection is detected and scan_input is True.
        """
        # Scan for injection if defense is configured
        if scan_input and self.defense:
            result = self.defense.scan(content)
            if not result.is_safe:
                raise SecurityError(
                    f"Potential injection detected: {result.matches}",
                    scan_result=result,
                )

        # Add to history
        self._message_history.append({"role": "user", "content": content})

        # Make API call
        response = await self._async_client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=self.config.system_prompt or anthropic.NOT_GIVEN,
            tools=self.config.tools or anthropic.NOT_GIVEN,
            stop_sequences=self.config.stop_sequences or anthropic.NOT_GIVEN,
            messages=self._message_history,
        )

        # Track metrics
        self._record_metrics(response)

        # Handle tool use
        iterations = 0
        while (
            response.stop_reason == "tool_use"
            and auto_execute_tools
            and iterations < max_tool_iterations
        ):
            response = await self._execute_tools(response)
            self._record_metrics(response)
            iterations += 1

        # Add assistant response to history
        self._message_history.append({"role": "assistant", "content": response.content})

        return response

    def send_message_sync(
        self,
        content: str,
        *,
        scan_input: bool = True,
        auto_execute_tools: bool = True,
    ) -> Message:
        """Synchronous version of send_message."""
        # Scan for injection
        if scan_input and self.defense:
            result = self.defense.scan(content)
            if not result.is_safe:
                raise SecurityError(
                    f"Potential injection detected: {result.matches}",
                    scan_result=result,
                )

        # Add to history
        self._message_history.append({"role": "user", "content": content})

        # Make API call
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=self.config.system_prompt or anthropic.NOT_GIVEN,
            tools=self.config.tools or anthropic.NOT_GIVEN,
            messages=self._message_history,
        )

        # Track metrics
        self._record_metrics(response)

        # Add to history
        self._message_history.append({"role": "assistant", "content": response.content})

        return response

    async def _execute_tools(self, response: Message) -> Message:
        """Execute tool calls and continue conversation.

        Args:
            response: Response containing tool use blocks.

        Returns:
            Next response after tool execution.
        """
        tool_results: list[dict[str, Any]] = []

        for block in response.content:
            if isinstance(block, ToolUseBlock):
                handler = self.tool_handlers.get(block.name)
                if handler:
                    try:
                        result = await self._call_handler(handler, block.input)

                        # Scan tool output for indirect injection
                        if self.defense:
                            output_str = str(result)
                            scan_result = self.defense.scan_tool_output(output_str)
                            if not scan_result.is_safe:
                                result = f"[FILTERED: Tool output contained suspicious content]"

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        })
                    except Exception as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error: {e}",
                            "is_error": True,
                        })
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error: No handler for tool '{block.name}'",
                        "is_error": True,
                    })

        if tool_results:
            # Add assistant response and tool results to history
            self._message_history.append({"role": "assistant", "content": response.content})
            self._message_history.append({"role": "user", "content": tool_results})

            return await self._async_client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.config.system_prompt or anthropic.NOT_GIVEN,
                tools=self.config.tools or anthropic.NOT_GIVEN,
                messages=self._message_history,
            )

        return response

    async def stream_message(self, content: str) -> AsyncIterator[str]:
        """Stream response text chunks.

        Args:
            content: User message content.

        Yields:
            Text chunks as they arrive.

        Raises:
            SecurityError: If injection is detected.
        """
        if self.defense:
            result = self.defense.scan(content)
            if not result.is_safe:
                raise SecurityError(
                    f"Potential injection detected: {result.matches}",
                    scan_result=result,
                )

        self._message_history.append({"role": "user", "content": content})

        async with self._async_client.messages.stream(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=self.config.system_prompt or anthropic.NOT_GIVEN,
            messages=self._message_history,
        ) as stream:
            async for text in stream.text_stream:
                yield text

            message = await stream.get_final_message()
            self._message_history.append({"role": "assistant", "content": message.content})

            self._record_metrics(message)

    def reset_history(self) -> None:
        """Clear message history."""
        self._message_history = []

    def get_history(self) -> list[MessageParam]:
        """Get current message history."""
        return list(self._message_history)

    def set_history(self, history: list[MessageParam]) -> None:
        """Set message history (for context restoration)."""
        self._message_history = list(history)

    def add_tool(self, name: str, handler: Callable) -> None:
        """Add a tool handler.

        Args:
            name: Tool name (must match tool definition).
            handler: Function to handle tool calls.
        """
        self.tool_handlers[name] = handler

    def remove_tool(self, name: str) -> bool:
        """Remove a tool handler.

        Args:
            name: Tool name to remove.

        Returns:
            True if removed, False if not found.
        """
        if name in self.tool_handlers:
            del self.tool_handlers[name]
            return True
        return False

    def _record_metrics(self, response: Message) -> None:
        """Record token usage metrics."""
        if self.metrics:
            model_key = self._get_model_key()
            self.metrics.record(
                agent_id=self.agent_id,
                model=model_key,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

    def _get_model_key(self) -> Model:
        """Map model string to Model enum."""
        model_lower = self.config.model.lower()
        for key, value in self.MODEL_MAPPING.items():
            if key in model_lower:
                return value
        return Model.SONNET  # Default

    @staticmethod
    async def _call_handler(handler: Callable, input_data: dict) -> Any:
        """Call handler, supporting both sync and async.

        Args:
            handler: Handler function.
            input_data: Input arguments as dict.

        Returns:
            Handler result.
        """
        if asyncio.iscoroutinefunction(handler):
            return await handler(**input_data)
        return handler(**input_data)

    def get_text_response(self, response: Message) -> str:
        """Extract text content from response.

        Args:
            response: Claude response message.

        Returns:
            Concatenated text from all TextBlock content.
        """
        texts = []
        for block in response.content:
            if isinstance(block, TextBlock):
                texts.append(block.text)
        return "\n".join(texts)

    @property
    def model(self) -> str:
        """Get current model."""
        return self.config.model

    @property
    def history_length(self) -> int:
        """Get number of messages in history."""
        return len(self._message_history)

    def __repr__(self) -> str:
        return f"ClaudeAgent(id={self.agent_id}, model={self.config.model})"
