"""Runner for executing OpenAI-style agents.

This module provides the Runner class that orchestrates agent execution,
implementing the agent loop pattern from OpenAI Agents SDK.

The Agent Loop:
1. LLM call with current context
2. Process response (text or tool calls)
3. Execute tool calls if any
4. Handle handoffs if requested
5. Check for final output
6. Loop or return result

Reference: https://github.com/openai/openai-agents-python
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from agentic_workflows.openai_agents.agent import OpenAIAgent
from agentic_workflows.openai_agents.agent_types import (
    HandoffRequest,
    Message,
    RunConfig,
    RunResult,
    ToolCall,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RunContext:
    """Context maintained during agent run.

    Tracks state across the agent loop iterations.
    """

    agent: OpenAIAgent
    config: RunConfig
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    handoffs: list[HandoffRequest] = field(default_factory=list)

    # Metrics
    turn_count: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    start_time: float = field(default_factory=time.time)

    # Tracing
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    span_id: str = ""

    # State
    current_agent: OpenAIAgent | None = None
    is_complete: bool = False
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        """Get elapsed time."""
        return time.time() - self.start_time


class Runner:
    """Runs agents through the agentic loop.

    The Runner orchestrates agent execution following the OpenAI Agents SDK
    pattern:
    1. Build messages with system prompt and conversation history
    2. Call LLM with tools
    3. Process response (text, tool calls, or handoffs)
    4. Execute any tool calls
    5. Handle handoffs by switching to target agent
    6. Repeat until max turns or final output

    Example:
        runner = Runner()

        # Async execution
        result = await runner.run(agent, "What's the weather?")

        # Sync execution
        result = runner.run_sync(agent, "Calculate 2+2")

        # Streaming
        async for chunk in runner.run_streamed(agent, "Tell a story"):
            print(chunk, end="")
    """

    def __init__(
        self,
        llm_client: Any = None,
        default_config: RunConfig | None = None,
        on_tool_call: Callable[[ToolCall], None] | None = None,
        on_handoff: Callable[[HandoffRequest], None] | None = None,
        on_turn_complete: Callable[[int, Message], None] | None = None,
    ):
        """Initialize runner.

        Args:
            llm_client: LLM client to use (auto-detected if None).
            default_config: Default run configuration.
            on_tool_call: Callback when tool is called.
            on_handoff: Callback when handoff occurs.
            on_turn_complete: Callback after each turn.
        """
        self.llm_client = llm_client
        self.default_config = default_config or RunConfig()
        self.on_tool_call = on_tool_call
        self.on_handoff = on_handoff
        self.on_turn_complete = on_turn_complete

        # Agent registry for handoffs
        self._agents: dict[str, OpenAIAgent] = {}

    def register_agent(self, agent: OpenAIAgent) -> None:
        """Register an agent for handoff support.

        Args:
            agent: Agent to register.
        """
        self._agents[agent.name] = agent

    def get_agent(self, name: str) -> OpenAIAgent | None:
        """Get registered agent by name.

        Args:
            name: Agent name.

        Returns:
            Agent or None.
        """
        return self._agents.get(name)

    async def run(
        self,
        agent: OpenAIAgent,
        input: str | list[Message],
        config: RunConfig | None = None,
        context: dict[str, Any] | None = None,
    ) -> RunResult[str]:
        """Run agent asynchronously.

        Args:
            agent: Agent to run.
            input: User input (string or messages).
            config: Run configuration.
            context: Additional context.

        Returns:
            RunResult with output and metadata.
        """
        run_config = config or agent.config.default_run_config or self.default_config

        # Initialize MCP servers if needed
        await agent.initialize_mcp_servers()

        # Build initial messages
        messages = self._build_initial_messages(agent, input)

        # Create run context
        ctx = RunContext(
            agent=agent,
            config=run_config,
            messages=messages,
            current_agent=agent,
            trace_id=run_config.trace_name or str(uuid.uuid4())[:16],
        )

        # Run agent loop
        try:
            result = await self._run_loop(ctx)
            return result
        except Exception as e:
            logger.error(f"Agent run failed: {e}")
            return RunResult(
                output="",
                messages=ctx.messages,
                tool_calls=ctx.tool_calls,
                handoffs=ctx.handoffs,
                total_turns=ctx.turn_count,
                total_tokens=ctx.total_tokens,
                duration_seconds=ctx.duration_seconds,
                success=False,
                error=str(e),
                trace_id=ctx.trace_id,
            )

    def run_sync(
        self,
        agent: OpenAIAgent,
        input: str | list[Message],
        config: RunConfig | None = None,
        context: dict[str, Any] | None = None,
    ) -> RunResult[str]:
        """Run agent synchronously.

        Wrapper around async run() for sync contexts.

        Args:
            agent: Agent to run.
            input: User input.
            config: Run configuration.
            context: Additional context.

        Returns:
            RunResult with output.
        """
        return asyncio.run(self.run(agent, input, config, context))

    async def run_streamed(
        self,
        agent: OpenAIAgent,
        input: str | list[Message],
        config: RunConfig | None = None,
    ) -> AsyncIterator[str]:
        """Run agent with streaming output.

        Yields output chunks as they're generated.

        Args:
            agent: Agent to run.
            input: User input.
            config: Run configuration.

        Yields:
            Output text chunks.
        """
        run_config = config or agent.config.default_run_config or self.default_config
        stream_config = RunConfig(**{k: v for k, v in vars(run_config).items() if k != "stream"})
        stream_config.stream = True

        # Initialize
        await agent.initialize_mcp_servers()
        messages = self._build_initial_messages(agent, input)

        ctx = RunContext(
            agent=agent,
            config=stream_config,
            messages=messages,
            current_agent=agent,
        )

        # Streaming loop
        async for chunk in self._run_loop_streamed(ctx):
            yield chunk

    def _build_initial_messages(
        self,
        agent: OpenAIAgent,
        input: str | list[Message],
    ) -> list[Message]:
        """Build initial message list.

        Args:
            agent: Agent to run.
            input: User input.

        Returns:
            Initial message list.
        """
        messages = [agent.get_system_message()]

        if isinstance(input, str):
            # Validate input
            is_valid, error = agent.validate_input(input)
            if not is_valid:
                raise ValueError(f"Input validation failed: {error}")

            messages.append(Message(role="user", content=input))
        else:
            messages.extend(input)

        return messages

    async def _run_loop(self, ctx: RunContext) -> RunResult[str]:
        """Execute the agent loop.

        Args:
            ctx: Run context.

        Returns:
            Final result.
        """
        while not ctx.is_complete and ctx.turn_count < ctx.config.max_turns:
            ctx.turn_count += 1
            logger.debug(f"Turn {ctx.turn_count} for agent {ctx.current_agent.name}")

            # Check timeout
            if ctx.duration_seconds > ctx.config.timeout_seconds:
                ctx.error = "Run timeout exceeded"
                break

            # Call LLM
            response = await self._call_llm(ctx)

            if response is None:
                ctx.error = ctx.error or "LLM call failed"
                break

            # Process response
            await self._process_response(ctx, response)

            # Callback
            if self.on_turn_complete and ctx.messages:
                self.on_turn_complete(ctx.turn_count, ctx.messages[-1])

        # Build final result
        output = self._extract_output(ctx)

        # Validate output
        is_valid, error = ctx.current_agent.validate_output(output)
        if not is_valid:
            ctx.error = f"Output validation failed: {error}"
            ctx.is_complete = False

        return RunResult(
            output=output,
            messages=ctx.messages,
            tool_calls=ctx.tool_calls,
            handoffs=ctx.handoffs,
            total_turns=ctx.turn_count,
            total_tokens=ctx.total_tokens,
            total_cost_usd=ctx.total_cost_usd,
            duration_seconds=ctx.duration_seconds,
            success=ctx.is_complete and ctx.error is None,
            error=ctx.error,
            final_agent=ctx.current_agent.name,
            trace_id=ctx.trace_id,
        )

    async def _run_loop_streamed(self, ctx: RunContext) -> AsyncIterator[str]:
        """Execute agent loop with streaming.

        Args:
            ctx: Run context.

        Yields:
            Output chunks.
        """
        while not ctx.is_complete and ctx.turn_count < ctx.config.max_turns:
            ctx.turn_count += 1

            if ctx.duration_seconds > ctx.config.timeout_seconds:
                break

            # Stream LLM response
            async for chunk in self._call_llm_streamed(ctx):
                yield chunk

    async def _call_llm(self, ctx: RunContext) -> dict[str, Any] | None:
        """Call the LLM.

        Args:
            ctx: Run context.

        Returns:
            LLM response or None on error.
        """
        agent = ctx.current_agent

        # Build request
        messages = [m.to_openai_format() for m in ctx.messages]
        tools = agent.get_tools_schema()

        try:
            client = self.llm_client or self._resolve_llm_client()
            response = await self._call_client(
                client,
                messages,
                tools,
                agent.model,
                agent.config,
            )
            return response

        except Exception as e:  # pragma: no cover - captured by run()
            ctx.error = str(e)
            logger.error(f"LLM call failed: {e}")
            return None

    def _resolve_llm_client(self) -> Any:
        """Resolve the LLM client from configuration or environment."""
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        if openai_key:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "openai package not installed. Install dependency to run agents against OpenAI."
                ) from exc
            return AsyncOpenAI(api_key=openai_key)

        if anthropic_key:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:
                raise RuntimeError(
                    "anthropic package not installed. Install dependency to run agents against Anthropic."
                ) from exc
            return AsyncAnthropic(api_key=anthropic_key)

        raise RuntimeError(
            "No LLM client configured. Provide llm_client or set OPENAI_API_KEY/ANTHROPIC_API_KEY."
        )

    async def _call_client(
        self,
        client: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        config: Any,
    ) -> dict[str, Any]:
        """Call LLM client.

        Args:
            client: LLM client instance.
            messages: Messages to send.
            tools: Tool schemas.
            model: Model name.
            config: Agent config.

        Returns:
            LLM response.
        """
        # Detect client type and call appropriately
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            # OpenAI-style client
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools if tools else None,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            return self._parse_openai_response(response)

        elif hasattr(client, "messages") and hasattr(client.messages, "create"):
            # Anthropic-style client
            response = await client.messages.create(
                model=model,
                messages=messages[1:],  # Remove system for Anthropic
                system=messages[0]["content"] if messages else "",
                tools=[self._to_anthropic_tool(t) for t in tools] if tools else None,
                max_tokens=config.max_tokens,
            )
            return self._parse_anthropic_response(response)

        else:
            raise ValueError(f"Unknown client type: {type(client)}")

    def _parse_openai_response(self, response: Any) -> dict[str, Any]:
        """Parse OpenAI API response.

        Args:
            response: OpenAI response object.

        Returns:
            Normalized response dict.
        """
        choice = response.choices[0]
        message = choice.message

        result = {
            "content": message.content or "",
            "role": "assistant",
            "tool_calls": [],
            "finish_reason": choice.finish_reason,
        }

        if message.tool_calls:
            for tc in message.tool_calls:
                result["tool_calls"].append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    }
                )

        return result

    def _parse_anthropic_response(self, response: Any) -> dict[str, Any]:
        """Parse Anthropic API response.

        Args:
            response: Anthropic response object.

        Returns:
            Normalized response dict.
        """
        result = {
            "content": "",
            "role": "assistant",
            "tool_calls": [],
            "finish_reason": response.stop_reason,
        }

        for block in response.content:
            if block.type == "text":
                result["content"] += block.text
            elif block.type == "tool_use":
                result["tool_calls"].append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    }
                )

        return result

    def _to_anthropic_tool(self, openai_tool: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI tool schema to Anthropic format.

        Args:
            openai_tool: OpenAI tool schema.

        Returns:
            Anthropic tool schema.
        """
        func = openai_tool.get("function", {})
        return {
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {}),
        }

    async def _call_llm_streamed(self, ctx: RunContext) -> AsyncIterator[str]:
        """Call LLM with streaming.

        Args:
            ctx: Run context.

        Yields:
            Response chunks.
        """
        # Mock streaming for now
        response = await self._call_llm(ctx)
        if response:
            content = response.get("content", "")
            # Simulate streaming by yielding words
            for word in content.split():
                yield word + " "
                await asyncio.sleep(0.01)

            # Process complete response
            await self._process_response(ctx, response)

    async def _process_response(self, ctx: RunContext, response: dict[str, Any]) -> None:
        """Process LLM response.

        Args:
            ctx: Run context.
            response: LLM response.
        """
        content = response.get("content", "")
        tool_calls_data = response.get("tool_calls", [])
        finish_reason = response.get("finish_reason", "")

        # Add assistant message
        assistant_msg = Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls_data if tool_calls_data else None,
        )
        ctx.messages.append(assistant_msg)

        # Check for tool calls
        if tool_calls_data:
            await self._handle_tool_calls(ctx, tool_calls_data)
        else:
            # No tool calls - check if we're done
            if finish_reason == "stop" and content:
                ctx.is_complete = True

    async def _handle_tool_calls(
        self,
        ctx: RunContext,
        tool_calls_data: list[dict[str, Any]],
    ) -> None:
        """Handle tool calls from LLM.

        Args:
            ctx: Run context.
            tool_calls_data: Tool call data from response.
        """
        for tc_data in tool_calls_data:
            tool_call = ToolCall(
                id=tc_data.get("id", str(uuid.uuid4())[:8]),
                name=tc_data.get("name", ""),
                arguments=tc_data.get("arguments", {}),
            )

            # Check for handoff
            if tool_call.name.startswith("handoff_to_"):
                target = tool_call.name.replace("handoff_to_", "")
                await self._handle_handoff(ctx, target, tool_call.arguments)
                continue

            # Execute tool
            ctx.tool_calls.append(tool_call)

            if self.on_tool_call:
                self.on_tool_call(tool_call)

            try:
                if ctx.config.parallel_tool_calls:
                    # Will be batched later
                    result = await ctx.current_agent.execute_tool(tool_call)
                else:
                    result = await ctx.current_agent.execute_tool(tool_call)

                # Add tool result message
                tool_msg = Message(
                    role="tool",
                    content=json.dumps(result) if not isinstance(result, str) else result,
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                )
                ctx.messages.append(tool_msg)

            except Exception as e:
                logger.error(f"Tool {tool_call.name} failed: {e}")
                tool_msg = Message(
                    role="tool",
                    content=f"Error: {str(e)}",
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                )
                ctx.messages.append(tool_msg)

    async def _handle_handoff(
        self,
        ctx: RunContext,
        target_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Handle agent handoff.

        Args:
            ctx: Run context.
            target_name: Target agent name.
            arguments: Handoff arguments.
        """
        # Find target agent
        target_agent = self.get_agent(target_name)

        if target_agent is None:
            # Check if agent has it as a direct handoff
            handoff_config = ctx.current_agent.get_handoff_config(target_name)
            if handoff_config and isinstance(handoff_config.target_agent, OpenAIAgent):
                target_agent = handoff_config.target_agent

        if target_agent is None:
            logger.error(f"Handoff target not found: {target_name}")
            ctx.messages.append(
                Message(
                    role="tool",
                    content=f"Error: Agent {target_name} not found",
                    name=f"handoff_to_{target_name}",
                )
            )
            return

        # Create handoff request
        handoff_request = HandoffRequest(
            target_agent=target_name,
            reason=arguments.get("reason", ""),
            context={"messages": ctx.messages},
            messages=ctx.messages.copy(),
        )
        ctx.handoffs.append(handoff_request)

        if self.on_handoff:
            self.on_handoff(handoff_request)

        # Switch to target agent
        logger.info(f"Handing off from {ctx.current_agent.name} to {target_name}")

        # Update context with new agent's system message
        ctx.current_agent = target_agent
        ctx.messages = [target_agent.get_system_message()] + ctx.messages[1:]

    def _extract_output(self, ctx: RunContext) -> str:
        """Extract final output from context.

        Args:
            ctx: Run context.

        Returns:
            Final output text.
        """
        # Get last assistant message
        for msg in reversed(ctx.messages):
            if msg.role == "assistant" and msg.content:
                return msg.content

        return ""


# Convenience functions


async def run(
    agent: OpenAIAgent,
    input: str | list[Message],
    config: RunConfig | None = None,
) -> RunResult[str]:
    """Run an agent asynchronously.

    Convenience function that creates a Runner and executes.

    Args:
        agent: Agent to run.
        input: User input.
        config: Run configuration.

    Returns:
        Run result.

    Example:
        result = await run(agent, "Hello!")
        print(result.output)
    """
    runner = Runner()
    return await runner.run(agent, input, config)


def run_sync(
    agent: OpenAIAgent,
    input: str | list[Message],
    config: RunConfig | None = None,
) -> RunResult[str]:
    """Run an agent synchronously.

    Convenience function for sync contexts.

    Args:
        agent: Agent to run.
        input: User input.
        config: Run configuration.

    Returns:
        Run result.

    Example:
        result = run_sync(agent, "Hello!")
        print(result.output)
    """
    runner = Runner()
    return runner.run_sync(agent, input, config)


async def run_streamed(
    agent: OpenAIAgent,
    input: str | list[Message],
    config: RunConfig | None = None,
) -> AsyncIterator[str]:
    """Run agent with streaming output.

    Args:
        agent: Agent to run.
        input: User input.
        config: Run configuration.

    Yields:
        Output chunks.

    Example:
        async for chunk in run_streamed(agent, "Tell a story"):
            print(chunk, end="")
    """
    runner = Runner()
    async for chunk in runner.run_streamed(agent, input, config):
        yield chunk
