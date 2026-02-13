"""A2A (Agent-to-Agent) Protocol Integration Module.

This module provides comprehensive support for the A2A protocol,
enabling agents to communicate with other A2A-compliant agents.

Reference: https://github.com/a2aproject/a2a-python

Components:
    - Types: Core A2A protocol types and dataclasses
    - Client: Client for communicating with A2A agents
    - Server: Server for exposing agents via A2A protocol
    - Integration: Adapters for agentic_workflows agents

Example - Creating an A2A Client:
    ```python
    from agentic_workflows.a2a import A2AClient, ClientConfig, Message

    async with A2AClient(ClientConfig(base_url="https://agent.example.com")) as client:
        # Get agent capabilities
        card = await client.get_card()
        print(f"Agent: {card.name}")
        print(f"Skills: {[s.name for s in card.skills]}")

        # Send a message and get response
        async for event in client.send_message("Hello, agent!"):
            print(event)
    ```

Example - Creating an A2A Server:
    ```python
    from fastapi import FastAPI
    from agentic_workflows.a2a import (
        create_a2a_server,
        AgentCard,
        AgentSkill,
        SimpleExecutor,
    )

    app = FastAPI()

    async def handle_message(text: str) -> str:
        return f"You said: {text}"

    server = create_a2a_server(
        executor=SimpleExecutor(handle_message),
        name="Echo Agent",
        description="An agent that echoes messages",
        skills=[
            AgentSkill(id="echo", name="Echo", description="Echoes input")
        ],
    )

    server.mount(app)
    ```

Example - Adapting an Internal Agent:
    ```python
    from agentic_workflows.agents.base import BaseAgent, AgentConfig
    from agentic_workflows.a2a import A2AAgentAdapter, A2AIntegrationConfig

    class MyAgent(BaseAgent):
        async def _execute(self, input_data):
            return f"Processed: {input_data}"

    agent = MyAgent(AgentConfig(name="MyAgent", description="My agent"))

    adapter = A2AAgentAdapter(
        agent=agent,
        config=A2AIntegrationConfig(
            base_url="https://example.com/agents/myagent",
            organization="My Org",
        ),
    )

    server = adapter.create_server()
    ```
"""

from __future__ import annotations

# =============================================================================
# Types
# =============================================================================
from agentic_workflows.a2a.a2a_types import (
    A2AErrorCode,
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    # Core Types
    AgentSkill,
    ApiKeyLocation,
    # Security
    APIKeySecurityScheme,
    Artifact,
    ArtifactEvent,
    DataPart,
    FileContent,
    FilePart,
    HTTPSecurityScheme,
    JSONRPCError,
    # JSON-RPC
    JSONRPCRequest,
    JSONRPCResponse,
    Message,
    MessageEvent,
    MessagePart,
    MessageRole,
    OAuth2SecurityScheme,
    PartType,
    SecurityScheme,
    SecuritySchemeType,
    StreamEvent,
    Task,
    # Enums
    TaskState,
    TaskStatus,
    # Events
    TaskStatusEvent,
    # Message Parts
    TextPart,
    TransportProtocol,
    parse_message_part,
)

# =============================================================================
# Client
# =============================================================================
from agentic_workflows.a2a.client import (
    # Client
    A2AClient,
    AuthInterceptor,
    CallbackEventConsumer,
    # Interceptors
    ClientCallInterceptor,
    # Configuration
    ClientConfig,
    # Consumers
    EventConsumer,
    LoggingInterceptor,
    QueueEventConsumer,
    RetryInterceptor,
    # Utilities
    discover_agent,
    send_task,
)

# =============================================================================
# Integration
# =============================================================================
from agentic_workflows.a2a.integration import (
    # Adapter
    A2AAgentAdapter,
    # Configuration
    A2AIntegrationConfig,
    # Executor
    AgentAdapterExecutor,
    # Card Builder
    AgentCardBuilder,
    # Factory
    create_a2a_adapter,
    create_a2a_server_for_agent,
    from_a2a_message,
    from_a2a_task,
    skill_from_capability,
    # Skill Conversion
    skill_from_tool,
    # Message Conversion
    to_a2a_message,
    to_a2a_task,
)

# =============================================================================
# Server
# =============================================================================
from agentic_workflows.a2a.server import (
    # Server
    A2AServer,
    # Executor
    AgentExecutor,
    EventQueue,
    InMemoryTaskStorage,
    # Context
    RequestContext,
    SimpleExecutor,
    # Storage
    TaskStorage,
    # Factory
    create_server,
    create_simple_server,
)

# =============================================================================
# Factory Function
# =============================================================================


def create_a2a_server(
    name: str,
    description: str,
    handler: callable = None,
    executor: AgentExecutor = None,
    agent: BaseAgent = None,
    skills: list[AgentSkill] = None,
    base_url: str = "",
    organization: str = "Agentic Workflows",
    streaming: bool = True,
    **kwargs,
) -> A2AServer:
    """Create an A2A server with flexible configuration.

    This is the primary factory function for creating A2A servers.
    It supports three modes:
    1. Handler function: Simple text-in/text-out processing
    2. Custom executor: Full control over execution
    3. Internal agent: Wrap an existing agentic_workflows agent

    Args:
        name: Agent name.
        description: Agent description.
        handler: Optional handler function (text -> text).
        executor: Optional custom executor.
        agent: Optional internal agent to wrap.
        skills: Agent skills list.
        base_url: Base URL for the agent.
        organization: Provider organization name.
        streaming: Enable streaming responses.
        **kwargs: Additional options passed to AgentCard.

    Returns:
        Configured A2AServer ready to mount on FastAPI.

    Raises:
        ValueError: If neither handler, executor, nor agent is provided.

    Examples:
        Simple handler:
        ```python
        server = create_a2a_server(
            name="Echo",
            description="Echoes messages",
            handler=lambda text: f"Echo: {text}",
        )
        ```

        With executor:
        ```python
        class MyExecutor(AgentExecutor):
            async def execute(self, context, events):
                await events.publish_text("Hello!")
                await events.publish_status(TaskState.COMPLETED)

            async def cancel(self, task_id):
                return True

        server = create_a2a_server(
            name="MyAgent",
            description="Custom agent",
            executor=MyExecutor(),
        )
        ```

        Wrapping internal agent:
        ```python
        from agentic_workflows.agents.base import SimpleAgent, AgentConfig

        agent = SimpleAgent(
            config=AgentConfig(name="Simple"),
            handler=lambda x: f"Result: {x}",
        )

        server = create_a2a_server(
            name="Simple Agent",
            description="Simple processing",
            agent=agent,
        )
        ```
    """
    # Determine executor
    if executor is not None:
        final_executor = executor
    elif handler is not None:
        final_executor = SimpleExecutor(handler)
    elif agent is not None:
        from agentic_workflows.a2a.integration import AgentAdapterExecutor

        final_executor = AgentAdapterExecutor(agent, streaming=streaming)
    else:
        raise ValueError("Must provide one of: handler, executor, or agent")

    # Build agent card
    card = AgentCard(
        name=name,
        description=description,
        url=base_url,
        skills=skills or [],
        provider=AgentProvider(organization=organization) if organization else None,
        capabilities=AgentCapabilities(streaming=streaming),
        **kwargs,
    )

    # Create server
    return A2AServer(
        executor=final_executor,
        card=card,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "TaskState",
    "MessageRole",
    "PartType",
    "TransportProtocol",
    "SecuritySchemeType",
    "ApiKeyLocation",
    "A2AErrorCode",
    # Message Parts
    "TextPart",
    "FilePart",
    "DataPart",
    "FileContent",
    "MessagePart",
    "parse_message_part",
    # Security
    "APIKeySecurityScheme",
    "HTTPSecurityScheme",
    "OAuth2SecurityScheme",
    "SecurityScheme",
    # Core Types
    "AgentSkill",
    "AgentProvider",
    "AgentInterface",
    "AgentCapabilities",
    "AgentCard",
    "Message",
    "Artifact",
    "TaskStatus",
    "Task",
    # JSON-RPC
    "JSONRPCRequest",
    "JSONRPCResponse",
    "JSONRPCError",
    # Events
    "TaskStatusEvent",
    "MessageEvent",
    "ArtifactEvent",
    "StreamEvent",
    # Client Config
    "ClientConfig",
    # Interceptors
    "ClientCallInterceptor",
    "LoggingInterceptor",
    "AuthInterceptor",
    "RetryInterceptor",
    # Consumers
    "EventConsumer",
    "CallbackEventConsumer",
    "QueueEventConsumer",
    # Client
    "A2AClient",
    # Client Utilities
    "discover_agent",
    "send_task",
    # Server Context
    "RequestContext",
    "EventQueue",
    # Executors
    "AgentExecutor",
    "SimpleExecutor",
    "AgentAdapterExecutor",
    # Storage
    "TaskStorage",
    "InMemoryTaskStorage",
    # Server
    "A2AServer",
    # Integration Config
    "A2AIntegrationConfig",
    # Skill Conversion
    "skill_from_tool",
    "skill_from_capability",
    # Card Builder
    "AgentCardBuilder",
    # Adapter
    "A2AAgentAdapter",
    # Message Conversion
    "to_a2a_message",
    "from_a2a_message",
    "to_a2a_task",
    "from_a2a_task",
    # Factory Functions
    "create_server",
    "create_simple_server",
    "create_a2a_adapter",
    "create_a2a_server_for_agent",
    "create_a2a_server",
]
