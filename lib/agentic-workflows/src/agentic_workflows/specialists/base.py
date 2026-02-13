"""Base specialist agent for PHUC stack components.

Provides common functionality for all specialist agents.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..agents.base import AgentConfig, AgentState, BaseAgent


class SpecialistCapability(Enum):
    """Capabilities that specialist agents can provide."""

    # Storage operations
    OBJECT_STORAGE = "object_storage"
    BLOB_MANAGEMENT = "blob_management"

    # Messaging
    EVENT_STREAMING = "event_streaming"
    MESSAGE_QUEUE = "message_queue"
    PUBSUB = "pubsub"

    # Data/Analytics
    OLAP_QUERY = "olap_query"
    REAL_TIME_ANALYTICS = "real_time_analytics"
    SQL_QUERY = "sql_query"

    # Customer Data
    PROFILE_MANAGEMENT = "profile_management"
    SEGMENTATION = "segmentation"
    EVENT_TRACKING = "event_tracking"

    # Vector/AI
    VECTOR_SEARCH = "vector_search"
    EMBEDDING_STORAGE = "embedding_storage"
    SIMILARITY_SEARCH = "similarity_search"

    # ML
    MODEL_SERVING = "model_serving"
    INFERENCE = "inference"
    MODEL_MANAGEMENT = "model_management"

    # Orchestration
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    DAG_MANAGEMENT = "dag_management"
    TASK_SCHEDULING = "task_scheduling"

    # Governance
    DATA_CATALOG = "data_catalog"
    LINEAGE_TRACKING = "lineage_tracking"
    METADATA_MANAGEMENT = "metadata_management"

    # AI/LLM
    LLM_ORCHESTRATION = "llm_orchestration"
    CHAIN_EXECUTION = "chain_execution"
    TOOL_INTEGRATION = "tool_integration"
    MCP_PROTOCOL = "mcp_protocol"

    # Security
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    SECRET_MANAGEMENT = "secret_management"
    ENCRYPTION = "encryption"
    CDN = "cdn"
    WAF = "waf"
    DDOS_PROTECTION = "ddos_protection"


@dataclass
class SpecialistConfig(AgentConfig):
    """Configuration for specialist agents."""

    # Connection settings
    endpoint: str = ""
    api_key: str | None = None
    api_secret: str | None = None
    timeout: float = 30.0
    max_retries: int = 3

    # Feature flags
    enable_caching: bool = True
    enable_metrics: bool = True
    enable_health_check: bool = True

    # Health check settings
    health_check_interval: float = 60.0

    # Custom settings
    custom_settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpecialistResult:
    """Result from a specialist operation."""

    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


class SpecialistAgent(BaseAgent, ABC):
    """Base class for PHUC stack specialist agents.

    Specialist agents provide domain-specific functionality for
    components of the PHUC marketing intelligence platform.

    Subclasses must implement:
    - capabilities: List of capabilities this agent provides
    - _connect: Establish connection to the service
    - _disconnect: Close connection
    - _health_check: Check service health
    """

    def __init__(
        self,
        config: SpecialistConfig | None = None,
        **kwargs,
    ):
        """Initialize specialist agent.

        Args:
            config: Specialist configuration.
            **kwargs: Additional arguments for BaseAgent.
        """
        self.specialist_config = config or SpecialistConfig()
        super().__init__(config=self.specialist_config, **kwargs)

        self._connected = False
        self._health_check_task: asyncio.Task | None = None
        self._last_health_check: float = 0
        self._is_healthy = False

        # Operation handlers
        self._handlers: dict[str, Callable[..., Any]] = {}

    @property
    @abstractmethod
    def capabilities(self) -> list[SpecialistCapability]:
        """List of capabilities this specialist provides."""
        pass

    @property
    @abstractmethod
    def service_name(self) -> str:
        """Name of the service this specialist manages."""
        pass

    @abstractmethod
    async def _connect(self) -> None:
        """Establish connection to the service."""
        pass

    @abstractmethod
    async def _disconnect(self) -> None:
        """Close connection to the service."""
        pass

    @abstractmethod
    async def _health_check(self) -> bool:
        """Check if the service is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        pass

    async def initialize(self) -> None:
        """Initialize the specialist agent."""
        await super().initialize()
        await self._connect()
        self._connected = True

        if self.specialist_config.enable_health_check:
            self._start_health_check()

    async def shutdown(self) -> None:
        """Shutdown the specialist agent."""
        self._stop_health_check()

        if self._connected:
            await self._disconnect()
            self._connected = False

        await super().shutdown()

    def _start_health_check(self) -> None:
        """Start periodic health checks."""
        if self._health_check_task is not None:
            return

        async def check_loop():
            while self.state == AgentState.RUNNING:
                try:
                    self._is_healthy = await self._health_check()
                    self._last_health_check = asyncio.get_event_loop().time()
                except Exception as e:
                    self._is_healthy = False
                    self.logger.warning(f"Health check failed: {e}")

                await asyncio.sleep(self.specialist_config.health_check_interval)

        self._health_check_task = asyncio.create_task(check_loop())

    def _stop_health_check(self) -> None:
        """Stop periodic health checks."""
        if self._health_check_task is not None:
            self._health_check_task.cancel()
            self._health_check_task = None

    def register_handler(
        self,
        operation: str,
        handler: Callable[..., Any],
    ) -> None:
        """Register a handler for an operation.

        Args:
            operation: Operation name.
            handler: Handler function.
        """
        self._handlers[operation] = handler

    async def execute(
        self,
        operation: str,
        **kwargs,
    ) -> SpecialistResult:
        """Execute an operation.

        Args:
            operation: Operation to execute.
            **kwargs: Operation arguments.

        Returns:
            Operation result.
        """
        import time

        start = time.time()

        if operation not in self._handlers:
            return SpecialistResult(
                success=False,
                error=f"Unknown operation: {operation}",
            )

        try:
            handler = self._handlers[operation]
            result = (
                await handler(**kwargs)
                if asyncio.iscoroutinefunction(handler)
                else handler(**kwargs)
            )

            return SpecialistResult(
                success=True,
                data=result,
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return SpecialistResult(
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def has_capability(self, capability: SpecialistCapability) -> bool:
        """Check if agent has a capability.

        Args:
            capability: Capability to check.

        Returns:
            True if agent has capability.
        """
        return capability in self.capabilities

    @property
    def is_connected(self) -> bool:
        """Check if connected to service."""
        return self._connected

    @property
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self._is_healthy

    def get_status(self) -> dict[str, Any]:
        """Get agent status.

        Returns:
            Status dictionary.
        """
        return {
            "agent_id": self.agent_id,
            "service": self.service_name,
            "state": self.state.value,
            "connected": self._connected,
            "healthy": self._is_healthy,
            "capabilities": [c.value for c in self.capabilities],
            "last_health_check": self._last_health_check,
        }

    async def process_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Process an incoming message.

        Args:
            message: Message to process.

        Returns:
            Response message.
        """
        operation = message.get("operation")
        params = message.get("params", {})

        if not operation:
            return {"error": "No operation specified"}

        result = await self.execute(operation, **params)

        return {
            "success": result.success,
            "data": result.data,
            "error": result.error,
            "metadata": result.metadata,
            "duration_ms": result.duration_ms,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"service={self.service_name}, "
            f"connected={self._connected}, "
            f"healthy={self._is_healthy})"
        )
