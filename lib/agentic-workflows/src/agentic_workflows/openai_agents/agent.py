"""OpenAI Agent wrapper for agentic_workflows.

This module provides the OpenAIAgent class that wraps internal agentic_workflows
agents to be compatible with OpenAI Agents SDK patterns.

Key features:
- Compatible with OpenAI Agents SDK API
- Adapts agentic_workflows agents to openai-agents patterns
- Support for tools via @function_tool decorator pattern
- MCP server integration
- Handoff support between agents

Reference: https://github.com/openai/openai-agents-python
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, ParamSpec, TypeVar, overload

from agentic_workflows.openai_agents.agent_types import (
    AgentConfig,
    GuardrailConfig,
    HandoffConfig,
    HandoffRequest,
    MCPServerConfig,
    Message,
    RunConfig,
    RunResult,
    ToolCall,
    ToolConfig,
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


def function_tool(
    name: str | None = None,
    description: str | None = None,
    requires_confirmation: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to convert a function into an agent tool.

    Mirrors the OpenAI Agents SDK @function_tool pattern. Extracts
    function signature and docstring to create tool schema.

    Args:
        name: Tool name (defaults to function name).
        description: Tool description (defaults to docstring).
        requires_confirmation: Whether to require user confirmation.

    Returns:
        Decorated function with tool metadata.

    Example:
        @function_tool(name="get_weather")
        def get_weather(location: str, unit: str = "celsius") -> str:
            '''Get weather for a location.'''
            return f"Weather in {location}: 22{unit[0].upper()}"
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Extract function metadata
        func_name = name or func.__name__
        func_description = description or (func.__doc__ or "").strip()

        # Build JSON schema from type hints
        sig = inspect.signature(func)
        hints = func.__annotations__

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = hints.get(param_name, Any)
            json_type = _python_type_to_json(param_type)

            properties[param_name] = json_type

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        # Create tool config
        tool_config = ToolConfig(
            name=func_name,
            description=func_description,
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
            handler=func,
            requires_confirmation=requires_confirmation,
        )

        # Attach metadata to function
        func._tool_config = tool_config  # type: ignore
        func._is_tool = True  # type: ignore

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(*args, **kwargs)

        wrapper._tool_config = tool_config  # type: ignore
        wrapper._is_tool = True  # type: ignore

        return wrapper  # type: ignore

    return decorator


def _python_type_to_json(python_type: type) -> dict[str, Any]:
    """Convert Python type hint to JSON schema type.

    Args:
        python_type: Python type annotation.

    Returns:
        JSON schema type definition.
    """
    # Handle basic types
    type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
        type(None): {"type": "null"},
    }

    if python_type in type_mapping:
        return type_mapping[python_type]

    # Handle Optional, List, Dict from typing
    origin = getattr(python_type, "__origin__", None)

    if origin is list:
        args = getattr(python_type, "__args__", (Any,))
        return {
            "type": "array",
            "items": _python_type_to_json(args[0]) if args else {},
        }

    if origin is dict:
        return {"type": "object"}

    if origin is type(None) or python_type is type(None):
        return {"type": "null"}

    # Handle Union (including Optional)
    if hasattr(python_type, "__origin__") and python_type.__origin__ is type(
        int | str
    ):
        args = python_type.__args__
        # Check for Optional (Union with None)
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return _python_type_to_json(non_none_args[0])
        return {"anyOf": [_python_type_to_json(a) for a in args]}

    # Default to string for unknown types
    return {"type": "string"}


class OpenAIAgent:
    """Agent compatible with OpenAI Agents SDK patterns.

    This class wraps agentic_workflows internal agents and exposes them
    through an API similar to the OpenAI Agents SDK.

    Features:
    - Tool registration via @function_tool decorator
    - MCP server integration for dynamic tools
    - Agent-to-agent handoffs
    - Input/output guardrails
    - Conversation history management

    Example:
        # Create agent
        agent = OpenAIAgent(
            name="assistant",
            instructions="You are a helpful assistant.",
            tools=[search_tool, calculator_tool],
        )

        # Run agent
        result = await agent.run("What's the weather?")
        print(result.output)

        # With handoff
        triage = OpenAIAgent(name="triage", handoffs=[specialist])
    """

    def __init__(
        self,
        name: str,
        instructions: str = "",
        model: str = "gpt-4o",
        tools: list[ToolConfig | Callable[..., Any]] | None = None,
        handoffs: list["OpenAIAgent | HandoffConfig"] | None = None,
        guardrails: list[GuardrailConfig] | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
    ):
        """Initialize OpenAI-style agent.

        Args:
            name: Agent identifier.
            instructions: System prompt.
            model: LLM model to use.
            tools: List of tools (ToolConfig or decorated functions).
            handoffs: Agents this agent can hand off to.
            guardrails: Input/output validation.
            mcp_servers: MCP servers for additional tools.
            config: Full AgentConfig (overrides other params).
        """
        # Use provided config or create new
        if config:
            self.config = config
        else:
            self.config = AgentConfig(
                name=name,
                instructions=instructions,
                model=model,
            )

        # Process tools
        self._tools: dict[str, ToolConfig] = {}
        if tools:
            for tool in tools:
                self._register_tool(tool)

        # Process handoffs
        self._handoffs: dict[str, HandoffConfig] = {}
        if handoffs:
            for handoff in handoffs:
                self._register_handoff(handoff)

        # Add guardrails
        if guardrails:
            self.config.guardrails.extend(guardrails)

        # Add MCP servers
        if mcp_servers:
            self.config.mcp_servers.extend(mcp_servers)

        # Runtime state
        self._mcp_tools: dict[str, ToolConfig] = {}
        self._mcp_initialized: bool = False

        # Identity
        self.agent_id = str(uuid.uuid4())[:12]
        self._run_count = 0

        # Validate
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid agent config: {errors}")

    @property
    def name(self) -> str:
        """Get agent name."""
        return self.config.name

    @property
    def instructions(self) -> str:
        """Get agent instructions."""
        return self.config.instructions

    @property
    def model(self) -> str:
        """Get model name."""
        return self.config.model

    @property
    def tools(self) -> list[ToolConfig]:
        """Get all registered tools."""
        return list(self._tools.values()) + list(self._mcp_tools.values())

    @property
    def handoff_targets(self) -> list[str]:
        """Get names of agents this agent can hand off to."""
        return list(self._handoffs.keys())

    def _register_tool(self, tool: ToolConfig | Callable[..., Any]) -> None:
        """Register a tool with the agent.

        Args:
            tool: ToolConfig or decorated function.
        """
        if isinstance(tool, ToolConfig):
            self._tools[tool.name] = tool
            self.config.tools.append(tool)
        elif hasattr(tool, "_tool_config"):
            # Function decorated with @function_tool
            tool_config: ToolConfig = tool._tool_config  # type: ignore
            self._tools[tool_config.name] = tool_config
            self.config.tools.append(tool_config)
        elif callable(tool):
            # Plain function - auto-convert
            tool_config = self._function_to_tool_config(tool)
            self._tools[tool_config.name] = tool_config
            self.config.tools.append(tool_config)
        else:
            raise ValueError(f"Invalid tool type: {type(tool)}")

    def _function_to_tool_config(self, func: Callable[..., Any]) -> ToolConfig:
        """Convert a plain function to ToolConfig.

        Args:
            func: Function to convert.

        Returns:
            ToolConfig for the function.
        """
        sig = inspect.signature(func)
        hints = func.__annotations__

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = hints.get(param_name, Any)
            properties[param_name] = _python_type_to_json(param_type)

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return ToolConfig(
            name=func.__name__,
            description=(func.__doc__ or "").strip() or f"Call {func.__name__}",
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
            handler=func,
        )

    def _register_handoff(self, handoff: "OpenAIAgent | HandoffConfig") -> None:
        """Register a handoff target.

        Args:
            handoff: Agent or HandoffConfig.
        """
        if isinstance(handoff, HandoffConfig):
            target_name = (
                handoff.target_agent.name
                if isinstance(handoff.target_agent, OpenAIAgent)
                else str(handoff.target_agent)
            )
            self._handoffs[target_name] = handoff
            self.config.handoffs.append(handoff)
        elif isinstance(handoff, OpenAIAgent):
            config = HandoffConfig(
                target_agent=handoff,
                description=f"Hand off to {handoff.name}",
            )
            self._handoffs[handoff.name] = config
            self.config.handoffs.append(config)
        else:
            raise ValueError(f"Invalid handoff type: {type(handoff)}")

    async def initialize_mcp_servers(self) -> None:
        """Initialize MCP server connections and load tools.

        Connects to configured MCP servers and imports their tools.
        """
        if self._mcp_initialized:
            return

        for server_config in self.config.mcp_servers:
            try:
                tools = await self._connect_mcp_server(server_config)
                for tool in tools:
                    self._mcp_tools[tool.name] = tool
                logger.info(
                    f"Loaded {len(tools)} tools from MCP server: {server_config.name}"
                )
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {server_config.name}: {e}")

        self._mcp_initialized = True

    async def _connect_mcp_server(
        self, config: MCPServerConfig
    ) -> list[ToolConfig]:
        """Connect to an MCP server and get available tools.

        Args:
            config: MCP server configuration.

        Returns:
            List of tools from the server.
        """
        # Import MCP client
        try:
            from agentic_workflows.protocols.mcp_client import MCPClient
        except ImportError:
            logger.warning("MCP client not available")
            return []

        # Create client based on transport
        if config.transport == "stdio":
            # For stdio transport, use command
            client = MCPClient(
                server_name=config.name,
                command=config.command,
                args=config.args,
                env=config.env,
            )
        else:
            # For HTTP/WebSocket
            client = MCPClient(
                server_name=config.name,
                url=config.url,
                transport=config.transport,
            )

        # Get tools
        await client.connect()
        mcp_tools = await client.list_tools()

        # Convert to ToolConfig
        tools = []
        for mcp_tool in mcp_tools:
            # Filter tools if specified
            if config.tools and mcp_tool.name not in config.tools:
                continue

            tool_config = ToolConfig(
                name=f"{config.name}_{mcp_tool.name}",
                description=mcp_tool.description or "",
                parameters=mcp_tool.input_schema or {},
                handler=lambda args, t=mcp_tool.name, c=client: c.call_tool(t, args),
                metadata={"mcp_server": config.name, "original_name": mcp_tool.name},
            )
            tools.append(tool_config)

        return tools

    def get_tool(self, name: str) -> ToolConfig | None:
        """Get a tool by name.

        Args:
            name: Tool name.

        Returns:
            ToolConfig or None if not found.
        """
        return self._tools.get(name) or self._mcp_tools.get(name)

    async def execute_tool(
        self,
        tool_call: ToolCall,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a tool call.

        Args:
            tool_call: Tool call to execute.
            context: Optional execution context.

        Returns:
            Tool execution result.
        """
        tool = self.get_tool(tool_call.name)
        if tool is None:
            raise ValueError(f"Unknown tool: {tool_call.name}")

        if tool.handler is None:
            raise ValueError(f"Tool {tool_call.name} has no handler")

        start_time = time.time()

        try:
            # Execute handler
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(**tool_call.arguments)
            else:
                result = tool.handler(**tool_call.arguments)

            tool_call.result = result
            tool_call.duration_ms = (time.time() - start_time) * 1000

            return result

        except Exception as e:
            tool_call.error = str(e)
            tool_call.duration_ms = (time.time() - start_time) * 1000
            raise

    def get_handoff_config(self, target_name: str) -> HandoffConfig | None:
        """Get handoff configuration for a target agent.

        Args:
            target_name: Target agent name.

        Returns:
            HandoffConfig or None.
        """
        return self._handoffs.get(target_name)

    def create_handoff_request(
        self,
        target_name: str,
        reason: str = "",
        context: dict[str, Any] | None = None,
        messages: list[Message] | None = None,
    ) -> HandoffRequest:
        """Create a handoff request.

        Args:
            target_name: Target agent name.
            reason: Reason for handoff.
            context: Context to pass.
            messages: Conversation history.

        Returns:
            HandoffRequest instance.
        """
        handoff_config = self.get_handoff_config(target_name)

        if handoff_config is None:
            raise ValueError(f"No handoff configured for: {target_name}")

        # Filter context if configured
        filtered_context = (
            handoff_config.filter_context(context or {})
            if context
            else {}
        )

        return HandoffRequest(
            target_agent=target_name,
            reason=reason,
            context=filtered_context,
            messages=messages or [],
        )

    def get_system_message(self) -> Message:
        """Build the system message for the agent.

        Returns:
            System message with instructions and tool descriptions.
        """
        # Build instructions
        content = self.instructions

        # Add handoff instructions if available
        if self._handoffs:
            content += "\n\nYou can hand off to the following agents:\n"
            for name, config in self._handoffs.items():
                desc = config.description or f"Hand off to {name}"
                content += f"- {name}: {desc}\n"

        return Message(role="system", content=content)

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """Get tool schemas for LLM.

        Returns:
            List of tool schemas in OpenAI format.
        """
        schemas = []

        for tool in self.tools:
            schemas.append(tool.to_openai_schema())

        # Add handoff as tools
        for name, config in self._handoffs.items():
            schemas.append({
                "type": "function",
                "function": {
                    "name": f"handoff_to_{name}",
                    "description": config.description or f"Hand off to {name}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Reason for the handoff",
                            },
                        },
                        "required": ["reason"],
                    },
                },
            })

        return schemas

    def validate_input(self, content: str) -> tuple[bool, str]:
        """Run input guardrails.

        Args:
            content: Input content.

        Returns:
            Tuple of (is_valid, error_message).
        """
        for guardrail in self.config.get_input_guardrails():
            is_valid, error = guardrail.validate(content)
            if not is_valid:
                if guardrail.on_fail == "block":
                    return False, guardrail.message or error
                elif guardrail.on_fail == "warn":
                    logger.warning(f"Input guardrail warning: {error}")
        return True, ""

    def validate_output(self, content: str) -> tuple[bool, str]:
        """Run output guardrails.

        Args:
            content: Output content.

        Returns:
            Tuple of (is_valid, error_message).
        """
        for guardrail in self.config.get_output_guardrails():
            is_valid, error = guardrail.validate(content)
            if not is_valid:
                if guardrail.on_fail == "block":
                    return False, guardrail.message or error
                elif guardrail.on_fail == "warn":
                    logger.warning(f"Output guardrail warning: {error}")
        return True, ""

    def clone(
        self,
        name: str | None = None,
        instructions: str | None = None,
        **overrides: Any,
    ) -> "OpenAIAgent":
        """Create a clone of this agent with modifications.

        Args:
            name: New agent name.
            instructions: New instructions.
            **overrides: Additional config overrides.

        Returns:
            New OpenAIAgent instance.
        """
        from dataclasses import replace

        new_config = replace(
            self.config,
            name=name or f"{self.config.name}_clone",
            instructions=instructions or self.config.instructions,
            **overrides,
        )

        return OpenAIAgent(config=new_config)

    def __repr__(self) -> str:
        return (
            f"OpenAIAgent(name={self.name!r}, model={self.model!r}, "
            f"tools={len(self._tools)}, handoffs={len(self._handoffs)})"
        )


def create_agent(
    name: str,
    instructions: str = "",
    model: str = "gpt-4o",
    tools: list[ToolConfig | Callable[..., Any]] | None = None,
    handoffs: list[OpenAIAgent | HandoffConfig] | None = None,
    **kwargs: Any,
) -> OpenAIAgent:
    """Factory function to create an OpenAI-style agent.

    This is the primary way to create agents, matching the OpenAI SDK pattern.

    Args:
        name: Agent name.
        instructions: System prompt.
        model: LLM model.
        tools: Available tools.
        handoffs: Handoff targets.
        **kwargs: Additional config options.

    Returns:
        Configured OpenAIAgent instance.

    Example:
        agent = create_agent(
            name="assistant",
            instructions="You are helpful.",
            tools=[search, calculator],
            model="gpt-4o",
        )
    """
    return OpenAIAgent(
        name=name,
        instructions=instructions,
        model=model,
        tools=tools,
        handoffs=handoffs,
        **kwargs,
    )
