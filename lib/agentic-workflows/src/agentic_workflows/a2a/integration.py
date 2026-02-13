"""A2A Integration with Agentic Workflows.

Provides adapters for converting internal agents to A2A-compatible
servers and vice versa.

This module bridges the agentic_workflows agent system with the
A2A protocol for agent-to-agent communication.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from agentic_workflows.a2a.a2a_types import (
    AgentCard,
    AgentSkill,
    AgentProvider,
    AgentCapabilities,
    Task,
    TaskState,
    TaskStatus,
    Message,
    MessageRole,
    TextPart,
    Artifact,
)
from agentic_workflows.a2a.server import (
    A2AServer,
    AgentExecutor,
    RequestContext,
    EventQueue,
    InMemoryTaskStorage,
)

if TYPE_CHECKING:
    from agentic_workflows.agents.base import BaseAgent, AgentConfig, AgentResult

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class A2AIntegrationConfig:
    """Configuration for A2A integration.

    Attributes:
        base_url: Base URL where the agent will be served.
        organization: Provider organization name.
        organization_url: Provider organization URL.
        version: Agent version string.
        streaming: Enable streaming responses.
        icon_url: URL to agent icon.
        documentation_url: URL to documentation.
    """

    base_url: str = ""
    """Base URL where the agent will be served."""

    organization: str = "Agentic Workflows"
    """Provider organization name."""

    organization_url: str = ""
    """Provider organization URL."""

    version: str = "1.0.0"
    """Agent version string."""

    streaming: bool = True
    """Enable streaming responses."""

    icon_url: str = ""
    """URL to agent icon."""

    documentation_url: str = ""
    """URL to documentation."""

    default_input_modes: list[str] = field(
        default_factory=lambda: ["text/plain"]
    )
    """Default input MIME types."""

    default_output_modes: list[str] = field(
        default_factory=lambda: ["text/plain", "application/json"]
    )
    """Default output MIME types."""


# =============================================================================
# Skill Conversion
# =============================================================================


def skill_from_tool(
    name: str,
    description: str = "",
    input_schema: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    examples: list[str] | None = None,
) -> AgentSkill:
    """Convert a tool definition to an A2A skill.

    Args:
        name: Tool name.
        description: Tool description.
        input_schema: Tool input schema.
        tags: Categorization tags.
        examples: Example invocations.

    Returns:
        A2A skill definition.
    """
    return AgentSkill(
        id=name.lower().replace(" ", "_").replace("-", "_"),
        name=name,
        description=description,
        tags=tags or [],
        examples=examples or [],
    )


def skill_from_capability(
    name: str,
    capability_type: str,
    description: str = "",
) -> AgentSkill:
    """Convert a capability to an A2A skill.

    Args:
        name: Capability name.
        capability_type: Type of capability.
        description: Capability description.

    Returns:
        A2A skill definition.
    """
    return AgentSkill(
        id=name.lower().replace(" ", "_"),
        name=name,
        description=description,
        tags=[capability_type],
    )


# =============================================================================
# Agent Card Builder
# =============================================================================


class AgentCardBuilder:
    """Builder for creating A2A Agent Cards from internal agents.

    Example:
        ```python
        from agentic_workflows.agents.base import AgentConfig

        config = AgentConfig(name="MyAgent", description="Does things")

        card = (
            AgentCardBuilder()
            .from_config(config)
            .with_url("https://example.com/agents/myagent")
            .with_skill("analyze", "Analyze data")
            .build()
        )
        ```
    """

    def __init__(self):
        """Initialize the builder."""
        self._name: str = ""
        self._description: str = ""
        self._url: str = ""
        self._version: str = "1.0.0"
        self._skills: list[AgentSkill] = []
        self._provider: AgentProvider | None = None
        self._capabilities: AgentCapabilities | None = None
        self._input_modes: list[str] = ["text/plain"]
        self._output_modes: list[str] = ["text/plain"]
        self._metadata: dict[str, Any] = {}

    def from_config(self, config: "AgentConfig") -> AgentCardBuilder:
        """Build from an AgentConfig.

        Args:
            config: Agent configuration.

        Returns:
            Self for chaining.
        """
        self._name = config.name
        self._description = config.description or f"Agent: {config.name}"
        self._version = config.version

        # Convert allowed tools to skills
        for tool in config.allowed_tools:
            self._skills.append(
                AgentSkill(
                    id=tool,
                    name=tool.replace("_", " ").title(),
                    description=f"Tool: {tool}",
                    tags=["tool"],
                )
            )

        # Add tags as metadata
        if config.tags:
            self._metadata["tags"] = config.tags

        return self

    def from_agent(self, agent: "BaseAgent") -> AgentCardBuilder:
        """Build from a BaseAgent instance.

        Args:
            agent: Agent instance.

        Returns:
            Self for chaining.
        """
        return self.from_config(agent.config)

    def with_name(self, name: str) -> AgentCardBuilder:
        """Set agent name.

        Args:
            name: Agent name.

        Returns:
            Self for chaining.
        """
        self._name = name
        return self

    def with_description(self, description: str) -> AgentCardBuilder:
        """Set agent description.

        Args:
            description: Agent description.

        Returns:
            Self for chaining.
        """
        self._description = description
        return self

    def with_url(self, url: str) -> AgentCardBuilder:
        """Set agent URL.

        Args:
            url: Agent URL.

        Returns:
            Self for chaining.
        """
        self._url = url
        return self

    def with_version(self, version: str) -> AgentCardBuilder:
        """Set agent version.

        Args:
            version: Version string.

        Returns:
            Self for chaining.
        """
        self._version = version
        return self

    def with_skill(
        self,
        skill_id: str,
        name: str,
        description: str = "",
        tags: list[str] | None = None,
        examples: list[str] | None = None,
    ) -> AgentCardBuilder:
        """Add a skill.

        Args:
            skill_id: Skill identifier.
            name: Skill name.
            description: Skill description.
            tags: Categorization tags.
            examples: Example invocations.

        Returns:
            Self for chaining.
        """
        self._skills.append(
            AgentSkill(
                id=skill_id,
                name=name,
                description=description,
                tags=tags or [],
                examples=examples or [],
            )
        )
        return self

    def with_provider(
        self,
        organization: str,
        url: str = "",
    ) -> AgentCardBuilder:
        """Set provider information.

        Args:
            organization: Organization name.
            url: Organization URL.

        Returns:
            Self for chaining.
        """
        self._provider = AgentProvider(organization=organization, url=url)
        return self

    def with_capabilities(
        self,
        streaming: bool = False,
        push_notifications: bool = False,
        state_history: bool = False,
    ) -> AgentCardBuilder:
        """Set capability flags.

        Args:
            streaming: Supports streaming.
            push_notifications: Supports push.
            state_history: Maintains history.

        Returns:
            Self for chaining.
        """
        self._capabilities = AgentCapabilities(
            streaming=streaming,
            push_notifications=push_notifications,
            state_history=state_history,
        )
        return self

    def with_input_modes(self, *modes: str) -> AgentCardBuilder:
        """Set input MIME types.

        Args:
            *modes: MIME types.

        Returns:
            Self for chaining.
        """
        self._input_modes = list(modes)
        return self

    def with_output_modes(self, *modes: str) -> AgentCardBuilder:
        """Set output MIME types.

        Args:
            *modes: MIME types.

        Returns:
            Self for chaining.
        """
        self._output_modes = list(modes)
        return self

    def with_metadata(self, **kwargs) -> AgentCardBuilder:
        """Add metadata.

        Args:
            **kwargs: Key-value metadata.

        Returns:
            Self for chaining.
        """
        self._metadata.update(kwargs)
        return self

    def build(self) -> AgentCard:
        """Build the agent card.

        Returns:
            Configured AgentCard.
        """
        return AgentCard(
            name=self._name,
            description=self._description,
            url=self._url,
            version=self._version,
            skills=self._skills,
            provider=self._provider,
            capabilities=self._capabilities,
            default_input_modes=self._input_modes,
            default_output_modes=self._output_modes,
            metadata=self._metadata,
        )


# =============================================================================
# Agent Adapter Executor
# =============================================================================


class AgentAdapterExecutor(AgentExecutor):
    """Executor that wraps an internal agent for A2A.

    Adapts BaseAgent instances to work with the A2A protocol.
    """

    def __init__(
        self,
        agent: "BaseAgent",
        streaming: bool = True,
    ):
        """Initialize adapter.

        Args:
            agent: Internal agent to wrap.
            streaming: Enable streaming responses.
        """
        self.agent = agent
        self.streaming = streaming
        self._active_tasks: dict[str, asyncio.Task] = {}

    async def execute(
        self,
        context: RequestContext,
        events: EventQueue,
    ) -> None:
        """Execute the agent and publish events.

        Args:
            context: Request context.
            events: Event queue for publishing.
        """
        await events.publish_status(TaskState.WORKING)

        try:
            # Ensure agent is initialized
            if not self.agent.is_running:
                await self.agent.initialize()

            # Get input text
            input_text = context.get_text()

            # Create execution task
            async def run_agent():
                return await self.agent.run(input_text)

            task = asyncio.create_task(run_agent())
            self._active_tasks[context.task_id] = task

            try:
                result = await task
            finally:
                self._active_tasks.pop(context.task_id, None)

            # Process result
            if result.success:
                # Publish response
                response_text = str(result.output) if result.output else ""
                await events.publish_text(response_text)

                # Publish artifacts
                if result.artifacts:
                    for name, content in result.artifacts.items():
                        artifact = Artifact(
                            name=name,
                            content=content,
                        )
                        await events.publish_artifact(artifact)

                await events.publish_status(TaskState.COMPLETED)
            else:
                await events.publish_status(
                    TaskState.FAILED,
                    message=result.error or "Execution failed",
                )

        except asyncio.CancelledError:
            await events.publish_status(TaskState.CANCELED)
            raise

        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            await events.publish_status(TaskState.FAILED, message=str(e))
            raise

    async def cancel(self, task_id: str) -> bool:
        """Cancel a running task.

        Args:
            task_id: Task to cancel.

        Returns:
            True if canceled.
        """
        if task_id in self._active_tasks:
            self._active_tasks[task_id].cancel()
            return True
        return False

    async def on_initialize(self) -> None:
        """Initialize the wrapped agent."""
        await self.agent.initialize()

    async def on_shutdown(self) -> None:
        """Terminate the wrapped agent."""
        self.agent.terminate()


# =============================================================================
# A2A Agent Adapter
# =============================================================================


class A2AAgentAdapter:
    """Adapter for exposing internal agents via A2A protocol.

    Wraps an internal agent and provides A2A server functionality.

    Example:
        ```python
        from agentic_workflows.agents.base import BaseAgent, AgentConfig

        class MyAgent(BaseAgent):
            async def _execute(self, input_data):
                return f"Processed: {input_data}"

        agent = MyAgent(AgentConfig(name="MyAgent"))

        adapter = A2AAgentAdapter(agent)
        server = adapter.create_server()

        # Mount on FastAPI
        from fastapi import FastAPI
        app = FastAPI()
        server.mount(app)
        ```
    """

    def __init__(
        self,
        agent: "BaseAgent",
        config: A2AIntegrationConfig | None = None,
    ):
        """Initialize adapter.

        Args:
            agent: Internal agent to wrap.
            config: Integration configuration.
        """
        self.agent = agent
        self.config = config or A2AIntegrationConfig()
        self._server: A2AServer | None = None

    def to_agent_card(self) -> AgentCard:
        """Convert internal agent to A2A AgentCard.

        Returns:
            A2A agent card.
        """
        builder = AgentCardBuilder().from_agent(self.agent)

        # Apply configuration
        if self.config.base_url:
            builder.with_url(self.config.base_url)

        builder.with_version(self.config.version)

        if self.config.organization:
            builder.with_provider(
                self.config.organization,
                self.config.organization_url,
            )

        builder.with_capabilities(
            streaming=self.config.streaming,
        )

        builder.with_input_modes(*self.config.default_input_modes)
        builder.with_output_modes(*self.config.default_output_modes)

        return builder.build()

    def create_executor(self) -> AgentExecutor:
        """Create an executor for the agent.

        Returns:
            A2A executor wrapping the agent.
        """
        return AgentAdapterExecutor(
            agent=self.agent,
            streaming=self.config.streaming,
        )

    def create_server(
        self,
        card: AgentCard | None = None,
    ) -> A2AServer:
        """Create an A2A server for the agent.

        Args:
            card: Optional custom agent card.

        Returns:
            Configured A2A server.
        """
        if self._server:
            return self._server

        self._server = A2AServer(
            executor=self.create_executor(),
            card=card or self.to_agent_card(),
        )

        return self._server

    @property
    def server(self) -> A2AServer | None:
        """Get the server instance if created."""
        return self._server


# =============================================================================
# Message Conversion
# =============================================================================


def to_a2a_message(
    content: str,
    role: str = "user",
    **metadata,
) -> Message:
    """Convert content to A2A message.

    Args:
        content: Message content.
        role: Message role (user/agent).
        **metadata: Additional metadata.

    Returns:
        A2A message.
    """
    msg_role = MessageRole.USER if role == "user" else MessageRole.AGENT

    return Message(
        role=msg_role,
        content=content,
        parts=[TextPart(text=content)],
        metadata=metadata,
    )


def from_a2a_message(message: Message) -> dict[str, Any]:
    """Convert A2A message to internal format.

    Args:
        message: A2A message.

    Returns:
        Internal message dictionary.
    """
    return {
        "role": message.role.value,
        "content": message.get_text(),
        "metadata": message.metadata,
    }


def to_a2a_task(
    task_id: str,
    messages: list[dict[str, Any]],
    state: str = "completed",
    artifacts: list[dict[str, Any]] | None = None,
) -> Task:
    """Convert internal task data to A2A task.

    Args:
        task_id: Task identifier.
        messages: Message history.
        state: Task state.
        artifacts: Task artifacts.

    Returns:
        A2A task.
    """
    a2a_messages = [
        to_a2a_message(
            content=m.get("content", ""),
            role=m.get("role", "user"),
        )
        for m in messages
    ]

    a2a_artifacts = [
        Artifact(
            id=a.get("id", ""),
            name=a.get("name", ""),
            content=a.get("content"),
        )
        for a in (artifacts or [])
    ]

    return Task(
        id=task_id,
        state=TaskState(state),
        messages=a2a_messages,
        artifacts=a2a_artifacts,
    )


def from_a2a_task(task: Task) -> dict[str, Any]:
    """Convert A2A task to internal format.

    Args:
        task: A2A task.

    Returns:
        Internal task dictionary.
    """
    return {
        "id": task.id,
        "state": task.state.value,
        "messages": [from_a2a_message(m) for m in task.messages],
        "artifacts": [
            {
                "id": a.id,
                "name": a.name,
                "content": a.content,
            }
            for a in task.artifacts
        ],
        "metadata": task.metadata,
    }


# =============================================================================
# Factory Functions
# =============================================================================


def create_a2a_adapter(
    agent: "BaseAgent",
    base_url: str = "",
    organization: str = "Agentic Workflows",
    **config_kwargs,
) -> A2AAgentAdapter:
    """Create an A2A adapter for an agent.

    Args:
        agent: Internal agent to wrap.
        base_url: Base URL for the agent.
        organization: Provider organization.
        **config_kwargs: Additional config options.

    Returns:
        Configured adapter.
    """
    config = A2AIntegrationConfig(
        base_url=base_url,
        organization=organization,
        **config_kwargs,
    )

    return A2AAgentAdapter(agent=agent, config=config)


def create_a2a_server_for_agent(
    agent: "BaseAgent",
    base_url: str = "",
    organization: str = "Agentic Workflows",
    **config_kwargs,
) -> A2AServer:
    """Create an A2A server for an agent.

    Convenience function that creates adapter and server in one call.

    Args:
        agent: Internal agent to wrap.
        base_url: Base URL for the agent.
        organization: Provider organization.
        **config_kwargs: Additional config options.

    Returns:
        Configured A2A server.
    """
    adapter = create_a2a_adapter(
        agent=agent,
        base_url=base_url,
        organization=organization,
        **config_kwargs,
    )

    return adapter.create_server()
