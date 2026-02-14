"""OpenAI Agents SDK Integration Module for agentic_workflows.

This module provides an integration layer that allows agentic_workflows agents
to be used with patterns from the OpenAI Agents SDK. It adapts internal agent
implementations to the OpenAI Agents API design while maintaining full
compatibility with agentic_workflows features.

Key Features:
- Agent creation compatible with OpenAI Agents SDK patterns
- Tool registration via @function_tool decorator
- MCP (Model Context Protocol) server integration
- Agent-to-agent handoffs
- Input/output guardrails
- Built-in tracing and observability

Quick Start:

    from agentic_workflows.openai_agents import (
        create_agent,
        function_tool,
        run,
        run_sync,
    )

    # Define a tool
    @function_tool(name="get_weather")
    def get_weather(location: str) -> str:
        '''Get weather for a location.'''
        return f"Weather in {location}: Sunny, 72F"

    # Create an agent
    agent = create_agent(
        name="assistant",
        instructions="You are a helpful assistant.",
        tools=[get_weather],
        model="gpt-4o",
    )

    # Run the agent (async)
    result = await run(agent, "What's the weather in NYC?")
    print(result.output)

    # Or run synchronously
    result = run_sync(agent, "Hello!")

Multi-Agent Handoffs:

    # Create specialist agents
    billing = create_agent(name="billing", instructions="Handle billing questions")
    support = create_agent(name="support", instructions="Handle support questions")

    # Create triage agent with handoffs
    triage = create_agent(
        name="triage",
        instructions="Route users to the right specialist.",
        handoffs=[billing, support],
    )

    # Run with automatic routing
    result = await run(triage, "I have a billing question")

Guardrails:

    from agentic_workflows.openai_agents import (
        InputGuardrail,
        OutputGuardrail,
        InjectionDefenseGuardrail,
        PIIDetectorGuardrail,
    )

    # Create agent with guardrails
    agent = create_agent(
        name="secure_agent",
        instructions="You are a secure assistant.",
        guardrails=[
            InjectionDefenseGuardrail(),
            PIIDetectorGuardrail(redact=True),
        ],
    )

Reference: https://github.com/openai/openai-agents-python
"""

from __future__ import annotations

# Agent implementation
from agentic_workflows.openai_agents.agent import (
    OpenAIAgent,
    create_agent,
    function_tool,
)

# Agent Types
from agentic_workflows.openai_agents.agent_types import (
    # Core configuration
    AgentConfig,
    # Guardrail configuration
    GuardrailConfig,
    # Handoff configuration
    HandoffConfig,
    HandoffRequest,
    HandoffStrategy,
    MCPServerConfig,
    # Message types
    Message,
    # Enums
    ModelProvider,
    OutputType,
    RunConfig,
    # Result types
    RunResult,
    RunResultStreaming,
    SessionConfig,
    ToolCall,
    # Tool configuration
    ToolConfig,
)

# Guardrails
from agentic_workflows.openai_agents.guardrails import (
    # Base classes
    BaseGuardrail,
    # Built-in guardrails
    ContentFilterGuardrail,
    GuardrailAction,
    GuardrailChain,
    GuardrailResult,
    InjectionDefenseGuardrail,
    InputGuardrail,
    LengthGuardrail,
    OutputGuardrail,
    PIIDetectorGuardrail,
    ToxicityGuardrail,
    # Decorator
    guardrail,
)

# Runner
from agentic_workflows.openai_agents.runner import (
    RunContext,
    Runner,
    run,
    run_streamed,
    run_sync,
)

# Tracing
from agentic_workflows.openai_agents.tracing import (
    AgentTrace,
    # Core tracing
    AgentTracer,
    TraceEvent,
    TraceEventType,
    TraceSpan,
    export_trace_to_chrome,
    # Export functions
    export_trace_to_json,
    export_trace_to_otlp,
    # Functions
    get_tracer,
    set_tracer,
    traced,
)

__all__ = [
    # ============================================================
    # Agent Types
    # ============================================================
    "AgentConfig",
    "RunConfig",
    "SessionConfig",
    "ToolConfig",
    "MCPServerConfig",
    "HandoffConfig",
    "HandoffStrategy",
    "HandoffRequest",
    "GuardrailConfig",
    "Message",
    "ToolCall",
    "RunResult",
    "RunResultStreaming",
    "ModelProvider",
    "OutputType",
    # ============================================================
    # Agent Implementation
    # ============================================================
    "OpenAIAgent",
    "function_tool",
    "create_agent",
    # ============================================================
    # Runner
    # ============================================================
    "Runner",
    "RunContext",
    "run",
    "run_sync",
    "run_streamed",
    # ============================================================
    # Guardrails
    # ============================================================
    "BaseGuardrail",
    "InputGuardrail",
    "OutputGuardrail",
    "GuardrailResult",
    "GuardrailAction",
    "GuardrailChain",
    "ContentFilterGuardrail",
    "PIIDetectorGuardrail",
    "InjectionDefenseGuardrail",
    "LengthGuardrail",
    "ToxicityGuardrail",
    "guardrail",
    # ============================================================
    # Tracing
    # ============================================================
    "AgentTracer",
    "AgentTrace",
    "TraceSpan",
    "TraceEvent",
    "TraceEventType",
    "get_tracer",
    "set_tracer",
    "traced",
    "export_trace_to_json",
    "export_trace_to_otlp",
    "export_trace_to_chrome",
]


# ============================================================================
# Factory Functions
# ============================================================================


def agent(
    name: str,
    instructions: str = "",
    model: str = "gpt-4o",
    tools: list = None,
    handoffs: list = None,
    guardrails: list = None,
    **kwargs,
) -> OpenAIAgent:
    """Create an OpenAI-style agent.

    Alias for create_agent() for OpenAI SDK compatibility.

    Args:
        name: Agent name.
        instructions: System prompt.
        model: LLM model.
        tools: Available tools.
        handoffs: Handoff targets.
        guardrails: Input/output guardrails.
        **kwargs: Additional configuration.

    Returns:
        Configured OpenAIAgent.

    Example:
        my_agent = agent(
            name="helper",
            instructions="You help users.",
            tools=[search, calculate],
        )
    """
    return create_agent(
        name=name,
        instructions=instructions,
        model=model,
        tools=tools,
        handoffs=handoffs,
        guardrails=guardrails,
        **kwargs,
    )


def tool(
    name: str | None = None,
    description: str | None = None,
    **kwargs,
):
    """Decorator to create a tool from a function.

    Alias for function_tool() for brevity.

    Args:
        name: Tool name.
        description: Tool description.
        **kwargs: Additional options.

    Returns:
        Decorated function.

    Example:
        @tool(name="search")
        def search_web(query: str) -> str:
            '''Search the web.'''
            return f"Results for: {query}"
    """
    return function_tool(name=name, description=description, **kwargs)


def create_runner(
    llm_client=None,
    default_config: RunConfig | None = None,
    **kwargs,
) -> Runner:
    """Create a runner for agent execution.

    Args:
        llm_client: LLM client instance.
        default_config: Default run configuration.
        **kwargs: Additional options.

    Returns:
        Configured Runner.

    Example:
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        runner = create_runner(llm_client=client)
        result = await runner.run(agent, "Hello!")
    """
    return Runner(
        llm_client=llm_client,
        default_config=default_config,
        **kwargs,
    )


def create_guardrail_chain(
    guardrails: list[BaseGuardrail] | None = None,
    include_defaults: bool = True,
) -> GuardrailChain:
    """Create a guardrail chain with optional defaults.

    Args:
        guardrails: Custom guardrails.
        include_defaults: Include default guardrails.

    Returns:
        GuardrailChain instance.

    Example:
        chain = create_guardrail_chain(
            guardrails=[CustomGuardrail()],
            include_defaults=True,
        )
    """
    all_guardrails = []

    if include_defaults:
        all_guardrails.extend(
            [
                LengthGuardrail(max_length=100000),
                InjectionDefenseGuardrail(),
            ]
        )

    if guardrails:
        all_guardrails.extend(guardrails)

    return GuardrailChain(all_guardrails)


# ============================================================================
# Convenience Functions
# ============================================================================


async def run_with_tracing(
    agent: OpenAIAgent,
    input: str,
    config: RunConfig | None = None,
    trace_name: str | None = None,
) -> tuple[RunResult, AgentTrace]:
    """Run an agent with full tracing enabled.

    Args:
        agent: Agent to run.
        input: User input.
        config: Run configuration.
        trace_name: Name for the trace.

    Returns:
        Tuple of (RunResult, AgentTrace).

    Example:
        result, trace = await run_with_tracing(agent, "Hello!")
        print(result.output)
        print(trace.to_json())
    """
    tracer = get_tracer()
    trace_id = trace_name or f"{agent.name}_{agent._run_count}"

    with tracer.trace(agent.name, trace_id=trace_id) as trace:
        result = await run(agent, input, config)

        # Record metrics
        tracer.record_llm_call(
            model=agent.model,
            input_tokens=result.total_tokens // 2,  # Estimate
            output_tokens=result.total_tokens // 2,
            cost_usd=result.total_cost_usd,
        )

        for tool_call in result.tool_calls:
            tracer.record_tool_call(
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                result=tool_call.result,
                error=tool_call.error,
                duration_ms=tool_call.duration_ms or 0,
            )

        for handoff in result.handoffs:
            tracer.record_handoff(
                from_agent=agent.name,
                to_agent=handoff.target_agent,
                reason=handoff.reason,
            )

    return result, trace


def create_secure_agent(
    name: str,
    instructions: str = "",
    model: str = "gpt-4o",
    tools: list = None,
    **kwargs,
) -> OpenAIAgent:
    """Create an agent with default security guardrails.

    Includes injection defense, PII detection, and content filtering.

    Args:
        name: Agent name.
        instructions: System prompt.
        model: LLM model.
        tools: Available tools.
        **kwargs: Additional options.

    Returns:
        Secured OpenAIAgent.

    Example:
        agent = create_secure_agent(
            name="secure_assistant",
            instructions="You are a secure assistant.",
        )
    """
    default_guardrails = [
        InjectionDefenseGuardrail(
            name="injection_defense",
            sensitivity=0.8,
        ),
        PIIDetectorGuardrail(
            name="pii_detection",
            redact=True,
            action=GuardrailAction.SANITIZE,
        ),
        ContentFilterGuardrail(
            name="content_filter",
        ),
        LengthGuardrail(
            name="length_check",
            max_length=50000,
        ),
    ]

    existing_guardrails = kwargs.pop("guardrails", []) or []
    all_guardrails = default_guardrails + existing_guardrails

    return create_agent(
        name=name,
        instructions=instructions,
        model=model,
        tools=tools,
        guardrails=all_guardrails,
        **kwargs,
    )


# ============================================================================
# Module Version
# ============================================================================

__version__ = "1.0.0"
__author__ = "agentic_workflows"
__description__ = "OpenAI Agents SDK Integration for agentic_workflows"
