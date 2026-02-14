"""Base agent class and configuration."""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from agentic_workflows.security.scope_validator import Scope

if TYPE_CHECKING:
    from agentic_workflows.artifacts import ArtifactManager


class AgentState(Enum):
    """Agent lifecycle states."""

    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class AgentConfig:
    """Agent configuration."""

    # Identity
    name: str
    description: str = ""
    version: str = "1.0.0"

    # Model settings
    model: str = "claude-sonnet-4"
    temperature: float = 0.7
    max_tokens: int = 4096

    # Security
    scope: Scope = Scope.STANDARD
    allowed_tools: list[str] = field(default_factory=list)
    blocked_tools: list[str] = field(default_factory=list)

    # Budget
    max_cost_usd: float | None = None
    max_tokens_total: int | None = None
    max_runtime_seconds: float | None = None

    # Behavior
    system_prompt: str = ""
    personality: str = ""
    constraints: list[str] = field(default_factory=list)

    # Metadata
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Validate configuration.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        if not self.name:
            errors.append("Agent name is required")

        if self.temperature < 0 or self.temperature > 1:
            errors.append("Temperature must be between 0 and 1")

        if self.max_tokens < 1:
            errors.append("Max tokens must be positive")

        if self.max_cost_usd is not None and self.max_cost_usd < 0:
            errors.append("Max cost must be non-negative")

        return errors


@dataclass
class AgentContext:
    """Runtime context for an agent."""

    agent_id: str
    session_id: str | None = None
    conversation_id: str | None = None
    parent_agent_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Runtime state
    messages: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result from agent execution."""

    success: bool
    output: Any = None
    error: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0


class BaseAgent(ABC):
    """Base class for all agents.

    Provides lifecycle management, configuration, and common functionality.
    Subclass this to create custom agents.
    """

    def __init__(
        self,
        config: AgentConfig,
        on_state_change: Callable[[AgentState, AgentState], None] | None = None,
        artifact_manager: ArtifactManager | None = None,
    ):
        """Initialize agent.

        Args:
            config: Agent configuration.
            on_state_change: Callback for state changes.
            artifact_manager: Optional artifact manager for producing artifacts.
        """
        self.config = config
        self.on_state_change = on_state_change
        self.artifact_manager = artifact_manager

        # Identity
        self.agent_id = str(uuid.uuid4())[:12]

        # State
        self._state = AgentState.CREATED
        self._context: AgentContext | None = None
        self._start_time: float | None = None
        self._end_time: float | None = None

        # Metrics
        self._total_tokens = 0
        self._total_cost = 0.0
        self._call_count = 0

    @property
    def state(self) -> AgentState:
        """Get current state."""
        return self._state

    @property
    def context(self) -> AgentContext | None:
        """Get current context."""
        return self._context

    @property
    def is_running(self) -> bool:
        """Check if agent is running."""
        return self._state == AgentState.RUNNING

    @property
    def runtime_seconds(self) -> float:
        """Get total runtime."""
        if self._start_time is None:
            return 0.0
        end = self._end_time or time.time()
        return end - self._start_time

    def _set_state(self, new_state: AgentState) -> None:
        """Set state and notify."""
        old_state = self._state
        self._state = new_state

        if self.on_state_change:
            self.on_state_change(old_state, new_state)

    async def initialize(self) -> bool:
        """Initialize the agent.

        Override to add custom initialization logic.

        Returns:
            True if successful.
        """
        self._set_state(AgentState.INITIALIZING)

        try:
            # Validate config
            errors = self.config.validate()
            if errors:
                self._set_state(AgentState.FAILED)
                return False

            # Custom initialization
            await self._on_initialize()

            self._set_state(AgentState.READY)
            return True

        except Exception:
            self._set_state(AgentState.FAILED)
            raise

    async def _on_initialize(self) -> None:
        """Override for custom initialization."""
        pass

    async def run(
        self,
        input_data: Any,
        context: AgentContext | None = None,
    ) -> AgentResult:
        """Run the agent.

        Args:
            input_data: Input for the agent.
            context: Optional execution context.

        Returns:
            Agent result.
        """
        if self._state not in (AgentState.READY, AgentState.PAUSED):
            return AgentResult(
                success=False,
                error=f"Cannot run agent in state: {self._state.value}",
            )

        # Setup context
        self._context = context or AgentContext(agent_id=self.agent_id)
        self._start_time = time.time()
        self._set_state(AgentState.RUNNING)

        try:
            # Check budget
            if not self._check_budget():
                self._set_state(AgentState.FAILED)
                return AgentResult(
                    success=False,
                    error="Budget exceeded",
                )

            # Execute
            output = await self._execute(input_data)

            self._end_time = time.time()
            self._set_state(AgentState.COMPLETED)

            return AgentResult(
                success=True,
                output=output,
                artifacts=self._context.artifacts if self._context else {},
                metrics={
                    "tokens": self._total_tokens,
                    "cost_usd": self._total_cost,
                    "calls": self._call_count,
                },
                duration_seconds=self.runtime_seconds,
            )

        except Exception as e:
            self._end_time = time.time()
            self._set_state(AgentState.FAILED)

            return AgentResult(
                success=False,
                error=str(e),
                duration_seconds=self.runtime_seconds,
            )

    @abstractmethod
    async def _execute(self, input_data: Any) -> Any:
        """Execute the agent's main logic.

        Override this in subclasses.

        Args:
            input_data: Input data.

        Returns:
            Output data.
        """
        pass

    def _check_budget(self) -> bool:
        """Check if within budget limits."""
        if self.config.max_cost_usd and self._total_cost >= self.config.max_cost_usd:
            return False

        if self.config.max_tokens_total and self._total_tokens >= self.config.max_tokens_total:
            return False

        if (
            self.config.max_runtime_seconds
            and self.runtime_seconds >= self.config.max_runtime_seconds
        ):
            return False

        return True

    def record_usage(self, tokens: int, cost: float) -> None:
        """Record token/cost usage.

        Args:
            tokens: Tokens used.
            cost: Cost in USD.
        """
        self._total_tokens += tokens
        self._total_cost += cost
        self._call_count += 1

    def create_artifact(
        self,
        name: str,
        artifact_type: str,
        content: Any,
        **metadata,
    ) -> Any:
        """Create an artifact if manager is available.

        Args:
            name: Artifact name.
            artifact_type: Type of artifact (code, text, report, etc.).
            content: Artifact content.
            **metadata: Additional metadata.

        Returns:
            Created artifact or None if no manager.
        """
        if not self.artifact_manager:
            return None

        from agentic_workflows.artifacts import ArtifactType

        type_map = {
            "code": ArtifactType.CODE,
            "text": ArtifactType.TEXT,
            "json": ArtifactType.JSON,
            "markdown": ArtifactType.MARKDOWN,
            "report": ArtifactType.REPORT,
            "diff": ArtifactType.DIFF,
            "log": ArtifactType.LOG,
        }

        art_type = type_map.get(artifact_type.lower(), ArtifactType.TEXT)
        return self.artifact_manager.create(name, art_type, content, **metadata)

    def pause(self) -> bool:
        """Pause the agent.

        Returns:
            True if paused.
        """
        if self._state == AgentState.RUNNING:
            self._set_state(AgentState.PAUSED)
            return True
        return False

    def resume(self) -> bool:
        """Resume the agent.

        Returns:
            True if resumed.
        """
        if self._state == AgentState.PAUSED:
            self._set_state(AgentState.RUNNING)
            return True
        return False

    def terminate(self) -> None:
        """Terminate the agent."""
        self._end_time = time.time()
        self._set_state(AgentState.TERMINATED)

    def get_status(self) -> dict[str, Any]:
        """Get agent status."""
        return {
            "agent_id": self.agent_id,
            "name": self.config.name,
            "state": self._state.value,
            "runtime_seconds": self.runtime_seconds,
            "metrics": {
                "tokens": self._total_tokens,
                "cost_usd": self._total_cost,
                "calls": self._call_count,
            },
            "budget": {
                "max_cost_usd": self.config.max_cost_usd,
                "max_tokens": self.config.max_tokens_total,
                "max_runtime_seconds": self.config.max_runtime_seconds,
            },
        }

    def __repr__(self) -> str:
        return f"Agent({self.config.name}, id={self.agent_id}, state={self._state.value})"


class SimpleAgent(BaseAgent):
    """Simple agent that wraps a callable.

    Example:
        agent = SimpleAgent(
            config=AgentConfig(name="greeter"),
            handler=lambda x: f"Hello, {x}!",
        )
        result = await agent.run("World")
    """

    def __init__(
        self,
        config: AgentConfig,
        handler: Callable[[Any], Any],
        **kwargs,
    ):
        """Initialize simple agent.

        Args:
            config: Agent configuration.
            handler: Function to execute (sync or async).
            **kwargs: Additional base agent options.
        """
        super().__init__(config, **kwargs)
        self.handler = handler

    async def _execute(self, input_data: Any) -> Any:
        """Execute the handler."""
        if asyncio.iscoroutinefunction(self.handler):
            return await self.handler(input_data)
        else:
            return self.handler(input_data)


class CompositeAgent(BaseAgent):
    """Agent composed of multiple sub-agents.

    Executes sub-agents in sequence or parallel.
    """

    def __init__(
        self,
        config: AgentConfig,
        agents: list[BaseAgent],
        parallel: bool = False,
        **kwargs,
    ):
        """Initialize composite agent.

        Args:
            config: Agent configuration.
            agents: Sub-agents to compose.
            parallel: Run agents in parallel.
            **kwargs: Additional base agent options.
        """
        super().__init__(config, **kwargs)
        self.agents = agents
        self.parallel = parallel

    async def _on_initialize(self) -> None:
        """Initialize all sub-agents."""
        for agent in self.agents:
            await agent.initialize()

    async def _execute(self, input_data: Any) -> list[Any]:
        """Execute all sub-agents."""
        if self.parallel:
            results = await asyncio.gather(
                *[agent.run(input_data, self._context) for agent in self.agents],
                return_exceptions=True,
            )
            return [r.output if isinstance(r, AgentResult) else r for r in results]
        else:
            results = []
            current_input = input_data
            for agent in self.agents:
                result = await agent.run(current_input, self._context)
                results.append(result.output)
                current_input = result.output
            return results
