"""OpenAI SDK Integration.

Provides integration with OpenAI API for hybrid deployments.

Usage:
    from agentic_workflows.integrations.openai_sdk import OpenAISDK, OpenAIAgent

    sdk = OpenAISDK()
    agent = OpenAIAgent(
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
class OpenAIAgentConfig:
    """Configuration for OpenAI agent."""

    model: str = "gpt-4o"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: list[str] | None = None


@dataclass
class OpenAIAgent:
    """OpenAI agent definition.

    Example:
        agent = OpenAIAgent(
            name="assistant",
            instructions="You are a helpful assistant.",
            tools=[...],
        )
    """

    name: str
    instructions: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)
    config: OpenAIAgentConfig = field(default_factory=OpenAIAgentConfig)


@dataclass
class OpenAIResult:
    """Result from OpenAI API call."""

    content: str = ""
    role: str = "assistant"
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    usage: dict[str, int] | None = None
    model: str = ""


class OpenAISDK:
    """OpenAI SDK wrapper.

    Example:
        sdk = OpenAISDK()
        agent = OpenAIAgent(name="assistant", instructions="...")
        result = await sdk.run(agent, messages=[
            {"role": "user", "content": "Hello"}
        ])
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """Initialize SDK.

        Args:
            api_key: Optional API key.
            base_url: Optional base URL for API.
        """
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                kwargs = {}
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                if self.base_url:
                    kwargs["base_url"] = self.base_url

                self._client = OpenAI(**kwargs)
            except ImportError:
                raise RuntimeError("openai package not installed")
        return self._client

    async def run(
        self,
        agent: OpenAIAgent,
        messages: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> OpenAIResult:
        """Run agent with messages.

        Args:
            agent: Agent definition.
            messages: Conversation messages.
            context: Optional context.

        Returns:
            OpenAIResult with response.
        """
        client = self._get_client()

        # Build messages with system prompt
        full_messages = []
        system_content = agent.instructions
        if context:
            import json

            system_content = f"{system_content}\n\nContext:\n{json.dumps(context, indent=2)}"

        full_messages.append({"role": "system", "content": system_content})
        full_messages.extend(messages)

        # Build request
        kwargs = {
            "model": agent.config.model,
            "max_tokens": agent.config.max_tokens,
            "temperature": agent.config.temperature,
            "messages": full_messages,
        }

        if agent.tools:
            # Convert to OpenAI function format
            kwargs["tools"] = self._convert_tools(agent.tools)

        if agent.config.top_p is not None:
            kwargs["top_p"] = agent.config.top_p
        if agent.config.stop:
            kwargs["stop"] = agent.config.stop
        if agent.config.frequency_penalty:
            kwargs["frequency_penalty"] = agent.config.frequency_penalty
        if agent.config.presence_penalty:
            kwargs["presence_penalty"] = agent.config.presence_penalty

        try:
            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]

            result = OpenAIResult(
                finish_reason=choice.finish_reason,
                model=response.model,
            )

            if choice.message.content:
                result.content = choice.message.content

            if choice.message.tool_calls:
                result.tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ]

            if response.usage:
                result.usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return result

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Claude tool format to OpenAI format.

        Args:
            tools: Tools in Claude format.

        Returns:
            Tools in OpenAI format.
        """
        converted = []
        for tool in tools:
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                }
            )
        return converted

    async def stream(
        self,
        agent: OpenAIAgent,
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

        full_messages = [
            {"role": "system", "content": agent.instructions},
            *messages,
        ]

        kwargs = {
            "model": agent.config.model,
            "max_tokens": agent.config.max_tokens,
            "messages": full_messages,
            "stream": True,
        }

        response = client.chat.completions.create(**kwargs)
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


__all__ = [
    "OpenAIAgentConfig",
    "OpenAIAgent",
    "OpenAIResult",
    "OpenAISDK",
]
