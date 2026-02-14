"""
Unified Agent for Agentic Workflows.

This module provides a UnifiedAgent class that can work seamlessly with:
- A2A protocol (agent-to-agent communication)
- OpenAI Agents SDK (handoffs, guardrails)
- MCP servers and tools
- All skill formats (Anthropic SKILL.md, OpenAI Codex)

Features:
- Multi-protocol support in single agent
- Automatic protocol detection and adaptation
- Unified tool/skill interface
- Session management across protocols
- Handoff support to other agents

Example:
    >>> from agentic_workflows.unified import UnifiedAgent
    >>>
    >>> agent = UnifiedAgent(
    ...     name="assistant",
    ...     instructions="You are a helpful assistant.",
    ...     model="claude-sonnet-4",
    ... )
    >>>
    >>> # Run with tools
    >>> result = await agent.run("What's the weather?")
    >>>
    >>> # Run with handoffs
    >>> result = await agent.run_with_handoffs("Billing question")
    >>>
    >>> # Expose as A2A agent
    >>> a2a_server = agent.expose_as_a2a()
    >>>
    >>> # Expose as MCP server
    >>> mcp_server = agent.expose_as_mcp()
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from agentic_workflows.unified.bridge import (
    UniversalMessage,
    UniversalTool,
    get_protocol_bridge,
)
from agentic_workflows.unified.skills import (
    ToolFormat,
    UnifiedSkillRegistry,
    get_unified_registry,
)

if TYPE_CHECKING:
    from agentic_workflows.unified.mcp import UnifiedMCPServer

logger = logging.getLogger(__name__)


class AgentCapability(Enum):
    """Agent capabilities for protocol negotiation."""

    TOOLS = "tools"
    STREAMING = "streaming"
    HANDOFFS = "handoffs"
    GUARDRAILS = "guardrails"
    MCP = "mcp"
    A2A = "a2a"
    OPENAI_COMPATIBLE = "openai"
    CLAUDE_COMPATIBLE = "claude"
    SKILLS = "skills"
    UI = "ui"


class AgentState(Enum):
    """Agent execution states."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING_INPUT = "waiting_input"
    HANDOFF_PENDING = "handoff_pending"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentTool:
    """Unified tool representation for agents."""

    name: str
    description: str
    handler: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=dict)
    is_async: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_universal(self) -> UniversalTool:
        """Convert to universal tool format."""
        return UniversalTool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            handler=self.handler,
            metadata=self.metadata,
        )


@dataclass
class AgentHandoff:
    """Configuration for agent handoff."""

    target_agent: UnifiedAgent | str
    condition: Callable[[str], bool] | None = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentGuardrail:
    """Guardrail for input/output filtering."""

    name: str
    type: str  # "input" or "output"
    handler: Callable[[str], tuple[bool, str]]  # (allowed, modified_content)
    enabled: bool = True


@dataclass
class UnifiedAgentConfig:
    """Configuration for unified agent."""

    # Identity
    name: str = "unified-agent"
    description: str = ""
    version: str = "1.0.0"

    # Model settings
    model: str = "claude-sonnet-4"
    temperature: float = 0.7
    max_tokens: int = 4096

    # Instructions
    instructions: str = ""
    system_prompt: str = ""

    # Capabilities
    enable_tools: bool = True
    enable_streaming: bool = True
    enable_handoffs: bool = True
    enable_guardrails: bool = True
    enable_skills: bool = True

    # Protocol support
    enable_a2a: bool = True
    enable_mcp: bool = True
    enable_openai_compat: bool = True

    # Skill configuration
    skill_domains: list[str] | None = None
    auto_load_skills: bool = True

    # A2A configuration
    a2a_base_url: str = ""
    a2a_organization: str = "Agentic Workflows"

    # Budget limits
    max_cost_usd: float | None = None
    max_tokens_total: int | None = None
    max_runtime_seconds: float | None = None


@dataclass
class AgentResult:
    """Result from agent execution."""

    success: bool
    output: str = ""
    error: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    handoffs: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_a2a_task(self) -> dict[str, Any]:
        """Convert to A2A task result."""
        state = "completed" if self.success else "failed"

        history = []
        for msg in self.messages:
            history.append(
                UniversalMessage(
                    role=msg.get("role", "assistant"),
                    content=msg.get("content", ""),
                ).to_a2a_message()
            )

        return {
            "id": str(uuid.uuid4()),
            "status": {"state": state, "error": self.error},
            "history": history,
            "artifacts": self.artifacts,
            "metadata": self.metrics,
        }


class UnifiedAgent:
    """Unified agent that works across all protocols.

    The UnifiedAgent provides a single interface that can:
    - Execute with Claude, OpenAI, or other LLMs
    - Use tools from MCP servers, skills, or custom handlers
    - Communicate via A2A protocol with other agents
    - Be exposed as an A2A or MCP server
    - Support handoffs and guardrails

    Example:
        >>> agent = UnifiedAgent(
        ...     name="helper",
        ...     instructions="You help users with tasks.",
        ...     model="claude-sonnet-4",
        ... )
        >>>
        >>> # Add a tool
        >>> @agent.tool("search")
        >>> async def search(query: str) -> str:
        ...     return f"Results for: {query}"
        >>>
        >>> # Run the agent
        >>> result = await agent.run("Search for Python tutorials")
        >>> print(result.output)
    """

    def __init__(
        self,
        config: UnifiedAgentConfig | None = None,
        name: str | None = None,
        instructions: str = "",
        model: str | None = None,
        tools: list[AgentTool | Callable] | None = None,
        handoffs: list[AgentHandoff | UnifiedAgent] | None = None,
        guardrails: list[AgentGuardrail] | None = None,
        **kwargs,
    ):
        """Initialize unified agent.

        Args:
            config: Full configuration object.
            name: Agent name (shorthand).
            instructions: Agent instructions (shorthand).
            model: LLM model (shorthand).
            tools: List of tools.
            handoffs: List of handoff targets.
            guardrails: List of guardrails.
            **kwargs: Additional config options.
        """
        # Build config
        if config is None:
            config = UnifiedAgentConfig(
                name=name or "unified-agent",
                instructions=instructions,
                model=model or "claude-sonnet-4",
                **kwargs,
            )
        self.config = config

        # Agent ID
        self.agent_id = str(uuid.uuid4())[:12]

        # State
        self._state = AgentState.IDLE
        self._session_id: str | None = None

        # Tools
        self._tools: dict[str, AgentTool] = {}
        if tools:
            for tool in tools:
                self.add_tool(tool)

        # Handoffs
        self._handoffs: list[AgentHandoff] = []
        if handoffs:
            for handoff in handoffs:
                self.add_handoff(handoff)

        # Guardrails
        self._guardrails: list[AgentGuardrail] = []
        if guardrails:
            for guardrail in guardrails:
                self._guardrails.append(guardrail)

        # Skill registry (lazy)
        self._skill_registry: UnifiedSkillRegistry | None = None

        # Protocol bridge
        self._bridge = get_protocol_bridge()

        # Conversation history
        self._messages: list[dict[str, Any]] = []

        # Metrics
        self._total_tokens = 0
        self._total_cost = 0.0
        self._run_count = 0

    @property
    def name(self) -> str:
        """Get agent name."""
        return self.config.name

    @property
    def state(self) -> AgentState:
        """Get current state."""
        return self._state

    @property
    def capabilities(self) -> list[AgentCapability]:
        """Get agent capabilities."""
        caps = [AgentCapability.TOOLS]

        if self.config.enable_streaming:
            caps.append(AgentCapability.STREAMING)
        if self.config.enable_handoffs and self._handoffs:
            caps.append(AgentCapability.HANDOFFS)
        if self.config.enable_guardrails and self._guardrails:
            caps.append(AgentCapability.GUARDRAILS)
        if self.config.enable_a2a:
            caps.append(AgentCapability.A2A)
        if self.config.enable_mcp:
            caps.append(AgentCapability.MCP)
        if self.config.enable_openai_compat:
            caps.append(AgentCapability.OPENAI_COMPATIBLE)
        if self.config.enable_skills:
            caps.append(AgentCapability.SKILLS)

        return caps

    def _get_skill_registry(self) -> UnifiedSkillRegistry:
        """Get or create skill registry."""
        if self._skill_registry is None:
            self._skill_registry = get_unified_registry()
        return self._skill_registry

    def add_tool(self, tool: AgentTool | Callable) -> None:
        """Add a tool to the agent.

        Args:
            tool: Tool instance or callable.
        """
        if callable(tool) and not isinstance(tool, AgentTool):
            # Convert function to tool
            import inspect

            sig = inspect.signature(tool)

            properties = {}
            required = []
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                properties[param_name] = {"type": "string"}
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

            tool = AgentTool(
                name=tool.__name__,
                description=tool.__doc__ or "",
                handler=tool,
                parameters={
                    "type": "object",
                    "properties": properties,
                    "required": required if required else [],
                },
                is_async=asyncio.iscoroutinefunction(tool),
            )

        self._tools[tool.name] = tool

    def tool(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable:
        """Decorator to register a tool.

        Args:
            name: Tool name.
            description: Tool description.

        Returns:
            Decorator function.

        Example:
            @agent.tool("greet")
            async def greet(name: str) -> str:
                '''Greet someone.'''
                return f"Hello, {name}!"
        """

        def decorator(func: Callable) -> Callable:
            import inspect

            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or ""

            sig = inspect.signature(func)
            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue

                param_type = "string"
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == dict:
                        param_type = "object"
                    elif param.annotation == list:
                        param_type = "array"

                properties[param_name] = {"type": param_type}
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

            tool = AgentTool(
                name=tool_name,
                description=tool_desc.strip(),
                handler=func,
                parameters={
                    "type": "object",
                    "properties": properties,
                    "required": required if required else [],
                },
                is_async=asyncio.iscoroutinefunction(func),
            )

            self._tools[tool_name] = tool
            return func

        return decorator

    def add_handoff(self, handoff: AgentHandoff | UnifiedAgent) -> None:
        """Add a handoff target.

        Args:
            handoff: Handoff configuration or target agent.
        """
        if isinstance(handoff, UnifiedAgent):
            handoff = AgentHandoff(
                target_agent=handoff,
                description=f"Handoff to {handoff.name}",
            )
        self._handoffs.append(handoff)

    def add_guardrail(self, guardrail: AgentGuardrail) -> None:
        """Add a guardrail.

        Args:
            guardrail: Guardrail configuration.
        """
        self._guardrails.append(guardrail)

    def load_skills(
        self,
        domains: list[str] | None = None,
    ) -> int:
        """Load skills as tools.

        Args:
            domains: Optional domain filter.

        Returns:
            Number of skills loaded.
        """
        registry = self._get_skill_registry()
        domains = domains or self.config.skill_domains

        skills = list(registry)
        if domains:
            skills = [s for s in skills if s.domain in domains]

        for skill in skills:
            # Create tool handler for skill
            async def skill_handler(
                action: str = "",
                context: dict = None,
                _skill_name: str = skill.name,
            ) -> str:
                skill_context = registry.load_skill_context(_skill_name)
                return json.dumps(
                    {
                        "skill": _skill_name,
                        "action": action,
                        "context": skill_context[:5000] if skill_context else None,
                    }
                )

            tool = AgentTool(
                name=f"skill_{skill.name.replace('-', '_')}",
                description=skill.description,
                handler=skill_handler,
                parameters={
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "context": {"type": "object"},
                    },
                    "required": ["action"],
                },
                metadata={"skill_name": skill.name, "skill_domain": skill.domain},
            )
            self._tools[tool.name] = tool

        return len(skills)

    def get_tools(
        self,
        format: ToolFormat = ToolFormat.CLAUDE_API,
    ) -> list[dict[str, Any]]:
        """Get tools in specified format.

        Args:
            format: Target tool format.

        Returns:
            List of tool definitions.
        """
        tools = []
        for tool in self._tools.values():
            universal = tool.to_universal()

            if format == ToolFormat.CLAUDE_API:
                tools.append(universal.to_claude_tool())
            elif format == ToolFormat.OPENAI_API:
                tools.append(universal.to_openai_function())
            elif format == ToolFormat.MCP_TOOL:
                tools.append(universal.to_mcp_tool())
            elif format == ToolFormat.A2A_SKILL:
                tools.append(universal.to_a2a_skill())

        return tools

    async def _apply_input_guardrails(self, input_text: str) -> tuple[bool, str]:
        """Apply input guardrails.

        Returns:
            Tuple of (allowed, modified_text).
        """
        text = input_text
        for guardrail in self._guardrails:
            if guardrail.type == "input" and guardrail.enabled:
                allowed, text = guardrail.handler(text)
                if not allowed:
                    return False, text
        return True, text

    async def _apply_output_guardrails(self, output_text: str) -> tuple[bool, str]:
        """Apply output guardrails.

        Returns:
            Tuple of (allowed, modified_text).
        """
        text = output_text
        for guardrail in self._guardrails:
            if guardrail.type == "output" and guardrail.enabled:
                allowed, text = guardrail.handler(text)
                if not allowed:
                    return False, text
        return True, text

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute a tool by name."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            if tool.is_async:
                result = await tool.handler(**arguments)
            else:
                result = tool.handler(**arguments)
            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"error": str(e)}

    async def _check_handoffs(self, input_text: str) -> AgentHandoff | None:
        """Check if any handoff should be triggered."""
        for handoff in self._handoffs:
            if handoff.condition:
                if handoff.condition(input_text):
                    return handoff
        return None

    async def run(
        self,
        input_text: str,
        session_id: str | None = None,
        **kwargs,
    ) -> AgentResult:
        """Run the agent with input.

        Args:
            input_text: User input text.
            session_id: Optional session ID for continuity.
            **kwargs: Additional options.

        Returns:
            Agent execution result.
        """
        self._state = AgentState.RUNNING
        self._session_id = session_id or str(uuid.uuid4())
        self._run_count += 1

        try:
            # Apply input guardrails
            if self.config.enable_guardrails:
                allowed, input_text = await self._apply_input_guardrails(input_text)
                if not allowed:
                    self._state = AgentState.COMPLETED
                    return AgentResult(
                        success=False,
                        error="Input blocked by guardrail",
                        output=input_text,
                    )

            # Add user message to history
            self._messages.append(
                {
                    "role": "user",
                    "content": input_text,
                }
            )

            # Build system prompt
            system_prompt = self.config.system_prompt or self.config.instructions

            # Get tools
            tools = self.get_tools(ToolFormat.CLAUDE_API)

            # For now, simulate response (real implementation would call LLM)
            output = await self._generate_response(
                system_prompt,
                self._messages,
                tools,
            )

            # Apply output guardrails
            if self.config.enable_guardrails:
                allowed, output = await self._apply_output_guardrails(output)
                if not allowed:
                    self._state = AgentState.COMPLETED
                    return AgentResult(
                        success=False,
                        error="Output blocked by guardrail",
                        output="I cannot provide that response.",
                    )

            # Add assistant message to history
            self._messages.append(
                {
                    "role": "assistant",
                    "content": output,
                }
            )

            self._state = AgentState.COMPLETED
            return AgentResult(
                success=True,
                output=output,
                messages=self._messages.copy(),
                metrics={
                    "run_count": self._run_count,
                    "total_tokens": self._total_tokens,
                },
            )

        except Exception as e:
            self._state = AgentState.FAILED
            logger.error(f"Agent execution failed: {e}")
            return AgentResult(
                success=False,
                error=str(e),
            )

    async def _generate_response(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> str:
        """Generate response using LLM.

        This is a placeholder - real implementation would call
        the appropriate LLM based on config.model.
        """
        # Try to use Claude
        try:
            import anthropic

            client = anthropic.AsyncAnthropic()

            # Format messages
            formatted_messages = []
            for msg in messages:
                formatted_messages.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                    }
                )

            response = await client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=system_prompt,
                messages=formatted_messages,
                tools=tools if tools else None,
            )

            # Extract text response
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text

            return ""

        except ImportError:
            # Fallback response
            return (
                f"[Agent {self.name}] Received: {messages[-1]['content'] if messages else 'empty'}"
            )

    async def run_with_handoffs(
        self,
        input_text: str,
        session_id: str | None = None,
        max_handoffs: int = 5,
        **kwargs,
    ) -> AgentResult:
        """Run agent with automatic handoff support.

        Args:
            input_text: User input.
            session_id: Session ID.
            max_handoffs: Maximum handoff chain length.
            **kwargs: Additional options.

        Returns:
            Final agent result.
        """
        current_agent = self
        current_input = input_text
        handoff_chain = []
        handoff_count = 0

        while handoff_count < max_handoffs:
            # Check for handoffs
            handoff = await current_agent._check_handoffs(current_input)

            if handoff is None:
                # No handoff, run normally
                result = await current_agent.run(current_input, session_id, **kwargs)
                result.handoffs = handoff_chain
                return result

            # Record handoff
            handoff_chain.append(
                {
                    "from": current_agent.name,
                    "to": handoff.target_agent.name
                    if isinstance(handoff.target_agent, UnifiedAgent)
                    else handoff.target_agent,
                    "reason": handoff.description,
                }
            )

            # Switch to target agent
            if isinstance(handoff.target_agent, UnifiedAgent):
                current_agent = handoff.target_agent
            else:
                # String reference - would need agent registry lookup
                break

            handoff_count += 1

        # Max handoffs reached
        return AgentResult(
            success=False,
            error=f"Maximum handoff chain length ({max_handoffs}) exceeded",
            handoffs=handoff_chain,
        )

    def expose_as_a2a(
        self,
        base_url: str = "",
    ) -> Any:
        """Create A2A server exposing this agent.

        Args:
            base_url: Base URL for the A2A server.

        Returns:
            A2AServer instance.
        """
        try:
            from agentic_workflows.a2a import (
                A2AServer,
                AgentCapabilities,
                AgentCard,
                AgentProvider,
                AgentSkill,
            )

            # Build agent card
            skills = []
            for tool in self._tools.values():
                skills.append(
                    AgentSkill(
                        id=tool.name,
                        name=tool.name.replace("_", " ").title(),
                        description=tool.description,
                    )
                )

            for handoff in self._handoffs:
                target_name = (
                    handoff.target_agent.name
                    if isinstance(handoff.target_agent, UnifiedAgent)
                    else str(handoff.target_agent)
                )
                skills.append(
                    AgentSkill(
                        id=f"handoff_{target_name}",
                        name=f"Handoff to {target_name}",
                        description=handoff.description,
                    )
                )

            card = AgentCard(
                name=self.name,
                description=self.config.description or self.config.instructions[:200],
                url=base_url or self.config.a2a_base_url,
                skills=skills,
                capabilities=AgentCapabilities(
                    streaming=self.config.enable_streaming,
                ),
                provider=AgentProvider(
                    organization=self.config.a2a_organization,
                ),
            )

            # Create executor that uses this agent
            from agentic_workflows.a2a import AgentExecutor, EventQueue, RequestContext

            class UnifiedAgentExecutor(AgentExecutor):
                def __init__(self, agent: UnifiedAgent):
                    self._agent = agent

                async def execute(
                    self,
                    context: RequestContext,
                    events: EventQueue,
                ) -> None:
                    # Extract input
                    text_parts = []
                    for part in context.message.parts:
                        if hasattr(part, "text"):
                            text_parts.append(part.text)
                    user_input = "\n".join(text_parts)

                    # Run agent
                    result = await self._agent.run(user_input, context.task_id)

                    # Publish result
                    await events.publish_text(result.output)
                    if result.success:
                        await events.publish_status("completed")
                    else:
                        await events.publish_status("failed", result.error)

                async def cancel(self, task_id: str) -> bool:
                    return True

            return A2AServer(
                executor=UnifiedAgentExecutor(self),
                card=card,
            )

        except ImportError as e:
            logger.warning(f"A2A module not available: {e}")
            return None

    def expose_as_mcp(self) -> UnifiedMCPServer:
        """Create MCP server exposing this agent.

        Returns:
            UnifiedMCPServer instance.
        """
        from agentic_workflows.unified.mcp import (
            UnifiedMCPConfig,
            UnifiedMCPServer,
        )

        config = UnifiedMCPConfig(
            name=f"agent-{self.name}",
            version=self.config.version,
            a2a_base_url=self.config.a2a_base_url,
            a2a_organization=self.config.a2a_organization,
        )

        server = UnifiedMCPServer(config)

        # Register all tools
        for tool in self._tools.values():
            server._tools[tool.name] = tool

        # Add agent invocation tool
        async def invoke_agent(
            message: str,
            session_id: str = "",
        ) -> str:
            result = await self.run(message, session_id or None)
            return json.dumps(
                {
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                }
            )

        from agentic_workflows.unified.mcp import MCPToolDefinition

        server._tools[f"invoke_{self.name}"] = MCPToolDefinition(
            name=f"invoke_{self.name}",
            description=f"Invoke the {self.name} agent with a message",
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message to send"},
                    "session_id": {"type": "string", "description": "Session ID"},
                },
                "required": ["message"],
            },
            handler=invoke_agent,
        )

        return server

    def get_a2a_card(self) -> dict[str, Any]:
        """Get A2A agent card for this agent.

        Returns:
            Agent card dictionary.
        """
        skills = []
        for tool in self._tools.values():
            skills.append(tool.to_universal().to_a2a_skill())

        return {
            "name": self.name,
            "description": self.config.description or self.config.instructions[:200],
            "url": self.config.a2a_base_url,
            "version": self.config.version,
            "protocolVersion": "0.3",
            "provider": {
                "organization": self.config.a2a_organization,
            },
            "capabilities": {
                "streaming": self.config.enable_streaming,
                "handoffs": self.config.enable_handoffs and len(self._handoffs) > 0,
            },
            "skills": skills,
            "defaultInputModes": ["text"],
            "defaultOutputModes": ["text"],
        }

    def get_status(self) -> dict[str, Any]:
        """Get agent status."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "state": self._state.value,
            "session_id": self._session_id,
            "capabilities": [c.value for c in self.capabilities],
            "tools_count": len(self._tools),
            "handoffs_count": len(self._handoffs),
            "guardrails_count": len(self._guardrails),
            "metrics": {
                "run_count": self._run_count,
                "total_tokens": self._total_tokens,
                "total_cost": self._total_cost,
            },
        }

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._messages.clear()

    def __repr__(self) -> str:
        return f"UnifiedAgent(name={self.name}, state={self._state.value})"


def create_unified_agent(
    name: str = "agent",
    instructions: str = "",
    model: str = "claude-sonnet-4",
    tools: list[Callable] | None = None,
    handoffs: list[UnifiedAgent] | None = None,
    enable_skills: bool = True,
    **kwargs,
) -> UnifiedAgent:
    """Factory function to create a unified agent.

    Args:
        name: Agent name.
        instructions: Agent instructions.
        model: LLM model.
        tools: List of tool functions.
        handoffs: List of handoff targets.
        enable_skills: Auto-load skills.
        **kwargs: Additional configuration.

    Returns:
        Configured UnifiedAgent.

    Example:
        >>> @some_tool
        >>> async def search(query: str) -> str:
        ...     return f"Results for: {query}"
        >>>
        >>> agent = create_unified_agent(
        ...     name="assistant",
        ...     instructions="You help users.",
        ...     tools=[search],
        ... )
    """
    agent = UnifiedAgent(
        name=name,
        instructions=instructions,
        model=model,
        tools=tools,
        handoffs=handoffs,
        enable_skills=enable_skills,
        **kwargs,
    )

    # Auto-load skills if enabled
    if enable_skills:
        agent.load_skills()

    return agent
