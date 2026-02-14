"""Anthropic Claude SDK Integration.

Provides native integration with the Anthropic Claude API.

Usage:
    from agentic_workflows.integrations.claude_sdk import ClaudeSDK, ClaudeAgent

    sdk = ClaudeSDK()
    agent = ClaudeAgent(
        name="assistant",
        instructions="You are a helpful assistant.",
    )
    result = await sdk.run(agent, messages=[...])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ClaudeAgentConfig:
    """Configuration for Claude agent."""

    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False


@dataclass
class ClaudeAgent:
    """Claude agent definition.

    Example:
        agent = ClaudeAgent(
            name="code-reviewer",
            instructions="Review code for bugs and style issues.",
            tools=[...],
        )
    """

    name: str
    instructions: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)
    config: ClaudeAgentConfig = field(default_factory=ClaudeAgentConfig)

    def get_system_prompt(self) -> str:
        """Get system prompt."""
        return self.instructions


@dataclass
class ClaudeResult:
    """Result from Claude API call."""

    content: str = ""
    role: str = "assistant"
    stop_reason: str | None = None
    tool_use: dict[str, Any] | None = None
    usage: dict[str, int] | None = None
    model: str = ""


class ClaudeSDK:
    """Anthropic Claude SDK wrapper.

    Example:
        sdk = ClaudeSDK()
        agent = ClaudeAgent(name="assistant", instructions="...")
        result = await sdk.run(agent, messages=[
            {"role": "user", "content": "Hello"}
        ])
    """

    def __init__(self, api_key: str | None = None):
        """Initialize SDK.

        Args:
            api_key: Optional API key (uses ANTHROPIC_API_KEY env var if not provided).
        """
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                if self.api_key:
                    self._client = anthropic.Anthropic(api_key=self.api_key)
                else:
                    self._client = anthropic.Anthropic()
            except ImportError:
                raise RuntimeError("anthropic package not installed")
        return self._client

    async def run(
        self,
        agent: ClaudeAgent,
        messages: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> ClaudeResult:
        """Run agent with messages.

        Args:
            agent: Agent definition.
            messages: Conversation messages.
            context: Optional context to include.

        Returns:
            ClaudeResult with response.
        """
        client = self._get_client()

        # Build system prompt
        system = agent.get_system_prompt()
        if context:
            import json

            system = f"{system}\n\nContext:\n{json.dumps(context, indent=2)}"

        # Build request
        kwargs = {
            "model": agent.config.model,
            "max_tokens": agent.config.max_tokens,
            "system": system,
            "messages": messages,
        }

        if agent.tools:
            kwargs["tools"] = agent.tools
        if agent.config.temperature is not None:
            kwargs["temperature"] = agent.config.temperature
        if agent.config.top_p is not None:
            kwargs["top_p"] = agent.config.top_p
        if agent.config.stop_sequences:
            kwargs["stop_sequences"] = agent.config.stop_sequences

        try:
            response = client.messages.create(**kwargs)

            # Parse response
            result = ClaudeResult(
                stop_reason=response.stop_reason,
                model=response.model,
            )

            if response.usage:
                result.usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }

            for block in response.content:
                if hasattr(block, "text"):
                    result.content = block.text
                elif hasattr(block, "type") and block.type == "tool_use":
                    result.tool_use = {
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }

            return result

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

    async def stream(
        self,
        agent: ClaudeAgent,
        messages: list[dict[str, Any]],
    ):
        """Stream response from agent.

        Args:
            agent: Agent definition.
            messages: Conversation messages.

        Yields:
            Response chunks.
        """
        client = self._get_client()

        kwargs = {
            "model": agent.config.model,
            "max_tokens": agent.config.max_tokens,
            "system": agent.get_system_prompt(),
            "messages": messages,
        }

        if agent.tools:
            kwargs["tools"] = agent.tools

        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """Count tokens in text.

        Args:
            text: Text to count.
            model: Optional model for tokenizer.

        Returns:
            Token count.
        """
        client = self._get_client()
        return client.count_tokens(text)


__all__ = [
    "ClaudeAgentConfig",
    "ClaudeAgent",
    "ClaudeResult",
    "ClaudeSDK",
]
