"""Stack Supervisor for coordinating PHUC specialist agents.

Provides centralized management and orchestration of all specialists.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .base import SpecialistAgent, SpecialistCapability, SpecialistConfig, SpecialistResult


class RoutingStrategy(Enum):
    """Strategies for routing requests to specialists."""

    CAPABILITY = "capability"  # Route by required capability
    ROUND_ROBIN = "round_robin"  # Distribute evenly
    LOAD_BASED = "load_based"  # Route to least loaded
    PRIORITY = "priority"  # Route by priority order


@dataclass
class SupervisorConfig:
    """Configuration for the stack supervisor."""

    # Routing settings
    routing_strategy: RoutingStrategy = RoutingStrategy.CAPABILITY
    enable_fallback: bool = True
    max_retries: int = 3

    # Health check settings
    health_check_interval: float = 30.0
    auto_reconnect: bool = True

    # Performance settings
    concurrent_limit: int = 10
    timeout: float = 60.0


@dataclass
class SpecialistRegistration:
    """Registration info for a specialist."""

    specialist: SpecialistAgent
    priority: int = 0
    enabled: bool = True
    request_count: int = 0
    error_count: int = 0


class StackSupervisor:
    """Supervisor for coordinating PHUC stack specialist agents.

    Features:
    - Specialist registration and discovery
    - Capability-based routing
    - Health monitoring
    - Load balancing
    - Error recovery
    """

    def __init__(self, config: SupervisorConfig | None = None):
        """Initialize the stack supervisor.

        Args:
            config: Supervisor configuration.
        """
        self.config = config or SupervisorConfig()

        # Registered specialists
        self._specialists: dict[str, SpecialistRegistration] = {}

        # Capability index: capability -> list of specialist names
        self._capability_index: dict[SpecialistCapability, list[str]] = {}

        # Round-robin counters for load balancing
        self._rr_counters: dict[SpecialistCapability, int] = {}

        # Supervisor state
        self._running = False
        self._health_task: asyncio.Task | None = None
        self._semaphore: asyncio.Semaphore | None = None

        # Event hooks
        self._hooks: dict[str, list[Callable]] = {
            "on_request": [],
            "on_response": [],
            "on_error": [],
            "on_health_change": [],
        }

    async def start(self) -> None:
        """Start the supervisor and all specialists."""
        self._running = True
        self._semaphore = asyncio.Semaphore(self.config.concurrent_limit)

        # Initialize all specialists
        await asyncio.gather(
            *[self._initialize_specialist(name, reg) for name, reg in self._specialists.items()]
        )

        # Start health monitoring
        if self.config.health_check_interval > 0:
            self._health_task = asyncio.create_task(self._health_check_loop())

    async def stop(self) -> None:
        """Stop the supervisor and all specialists."""
        self._running = False

        # Stop health monitoring
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Shutdown all specialists
        await asyncio.gather(
            *[reg.specialist.shutdown() for reg in self._specialists.values()],
            return_exceptions=True,
        )

    async def _initialize_specialist(
        self,
        name: str,
        registration: SpecialistRegistration,
    ) -> None:
        """Initialize a specialist agent."""
        try:
            await registration.specialist.initialize()
        except Exception as e:
            registration.enabled = False
            await self._emit("on_error", name, e)

    def register(
        self,
        name: str,
        specialist: SpecialistAgent,
        priority: int = 0,
    ) -> None:
        """Register a specialist agent.

        Args:
            name: Unique name for the specialist.
            specialist: The specialist agent instance.
            priority: Priority for routing (higher = preferred).
        """
        registration = SpecialistRegistration(
            specialist=specialist,
            priority=priority,
        )
        self._specialists[name] = registration

        # Update capability index
        for capability in specialist.capabilities:
            if capability not in self._capability_index:
                self._capability_index[capability] = []
            self._capability_index[capability].append(name)
            # Sort by priority
            self._capability_index[capability].sort(
                key=lambda n: self._specialists[n].priority,
                reverse=True,
            )

    def unregister(self, name: str) -> None:
        """Unregister a specialist agent.

        Args:
            name: Specialist name.
        """
        if name not in self._specialists:
            return

        specialist = self._specialists.pop(name).specialist

        # Update capability index
        for capability in specialist.capabilities:
            if capability in self._capability_index:
                self._capability_index[capability] = [
                    n for n in self._capability_index[capability] if n != name
                ]

    def get_specialist(self, name: str) -> SpecialistAgent | None:
        """Get a specialist by name.

        Args:
            name: Specialist name.

        Returns:
            The specialist or None.
        """
        reg = self._specialists.get(name)
        return reg.specialist if reg else None

    def find_specialists(
        self,
        capability: SpecialistCapability,
        only_healthy: bool = True,
    ) -> list[SpecialistAgent]:
        """Find specialists with a capability.

        Args:
            capability: Required capability.
            only_healthy: Only return healthy specialists.

        Returns:
            List of matching specialists.
        """
        names = self._capability_index.get(capability, [])
        specialists = []

        for name in names:
            reg = self._specialists[name]
            if not reg.enabled:
                continue
            if only_healthy and not reg.specialist.is_healthy:
                continue
            specialists.append(reg.specialist)

        return specialists

    async def execute(
        self,
        operation: str,
        capability: SpecialistCapability | None = None,
        specialist_name: str | None = None,
        **kwargs,
    ) -> SpecialistResult:
        """Execute an operation on a specialist.

        Args:
            operation: Operation to execute.
            capability: Required capability (for routing).
            specialist_name: Specific specialist to use.
            **kwargs: Operation arguments.

        Returns:
            Operation result.
        """
        await self._emit("on_request", operation, kwargs)

        # Find target specialist
        if specialist_name:
            reg = self._specialists.get(specialist_name)
            if not reg:
                return SpecialistResult(
                    success=False,
                    error=f"Specialist not found: {specialist_name}",
                )
            specialists = [(specialist_name, reg)]
        elif capability:
            specialists = self._route_by_capability(capability)
            if not specialists:
                return SpecialistResult(
                    success=False,
                    error=f"No specialist available for: {capability.value}",
                )
        else:
            return SpecialistResult(
                success=False,
                error="Must specify capability or specialist_name",
            )

        # Try specialists with retry/fallback
        last_error = None
        for attempt in range(self.config.max_retries):
            for name, reg in specialists:
                if not reg.enabled:
                    continue

                try:
                    async with self._semaphore:
                        result = await asyncio.wait_for(
                            reg.specialist.execute(operation, **kwargs),
                            timeout=self.config.timeout,
                        )

                    reg.request_count += 1

                    if result.success:
                        await self._emit("on_response", name, result)
                        return result
                    else:
                        last_error = result.error

                except asyncio.TimeoutError:
                    last_error = "Operation timed out"
                    reg.error_count += 1
                except Exception as e:
                    last_error = str(e)
                    reg.error_count += 1
                    await self._emit("on_error", name, e)

                if not self.config.enable_fallback:
                    break

        return SpecialistResult(
            success=False,
            error=last_error or "All specialists failed",
        )

    def _route_by_capability(
        self,
        capability: SpecialistCapability,
    ) -> list[tuple[str, SpecialistRegistration]]:
        """Route request to specialists by capability.

        Args:
            capability: Required capability.

        Returns:
            Ordered list of (name, registration) tuples.
        """
        names = self._capability_index.get(capability, [])
        if not names:
            return []

        if self.config.routing_strategy == RoutingStrategy.CAPABILITY:
            # Priority order
            return [(n, self._specialists[n]) for n in names]

        elif self.config.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            # Rotate through specialists
            if capability not in self._rr_counters:
                self._rr_counters[capability] = 0

            idx = self._rr_counters[capability] % len(names)
            self._rr_counters[capability] += 1

            rotated = names[idx:] + names[:idx]
            return [(n, self._specialists[n]) for n in rotated]

        elif self.config.routing_strategy == RoutingStrategy.LOAD_BASED:
            # Sort by request count (least loaded first)
            sorted_names = sorted(
                names,
                key=lambda n: self._specialists[n].request_count,
            )
            return [(n, self._specialists[n]) for n in sorted_names]

        else:  # PRIORITY
            return [(n, self._specialists[n]) for n in names]

    async def _health_check_loop(self) -> None:
        """Periodic health check for all specialists."""
        while self._running:
            await asyncio.sleep(self.config.health_check_interval)

            for name, reg in self._specialists.items():
                if not reg.enabled:
                    continue

                was_healthy = reg.specialist.is_healthy

                try:
                    is_healthy = await reg.specialist._health_check()

                    if was_healthy and not is_healthy:
                        await self._emit("on_health_change", name, False)
                    elif not was_healthy and is_healthy:
                        await self._emit("on_health_change", name, True)

                except Exception:
                    if was_healthy:
                        await self._emit("on_health_change", name, False)

    def on(self, event: str, handler: Callable) -> None:
        """Register an event handler.

        Args:
            event: Event name.
            handler: Handler function.
        """
        if event in self._hooks:
            self._hooks[event].append(handler)

    async def _emit(self, event: str, *args) -> None:
        """Emit an event to handlers."""
        for handler in self._hooks.get(event, []):
            try:
                result = handler(*args)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

    def get_status(self) -> dict[str, Any]:
        """Get supervisor status.

        Returns:
            Status information.
        """
        specialists_status = {}
        for name, reg in self._specialists.items():
            specialists_status[name] = {
                "service": reg.specialist.service_name,
                "enabled": reg.enabled,
                "healthy": reg.specialist.is_healthy,
                "connected": reg.specialist.is_connected,
                "priority": reg.priority,
                "request_count": reg.request_count,
                "error_count": reg.error_count,
                "capabilities": [c.value for c in reg.specialist.capabilities],
            }

        return {
            "running": self._running,
            "specialist_count": len(self._specialists),
            "healthy_count": sum(1 for r in self._specialists.values() if r.specialist.is_healthy),
            "capabilities": list(self._capability_index.keys()),
            "specialists": specialists_status,
        }

    async def broadcast(
        self,
        operation: str,
        capability: SpecialistCapability | None = None,
        **kwargs,
    ) -> dict[str, SpecialistResult]:
        """Broadcast an operation to multiple specialists.

        Args:
            operation: Operation to execute.
            capability: Filter by capability.
            **kwargs: Operation arguments.

        Returns:
            Dict of specialist name to result.
        """
        if capability:
            targets = [
                (name, reg)
                for name, reg in self._specialists.items()
                if capability in reg.specialist.capabilities and reg.enabled
            ]
        else:
            targets = [(name, reg) for name, reg in self._specialists.items() if reg.enabled]

        results = {}
        tasks = []

        for name, reg in targets:

            async def execute_one(n: str, r: SpecialistRegistration):
                try:
                    return n, await r.specialist.execute(operation, **kwargs)
                except Exception as e:
                    return n, SpecialistResult(success=False, error=str(e))

            tasks.append(execute_one(name, reg))

        completed = await asyncio.gather(*tasks)
        results = {name: result for name, result in completed}

        return results

    async def __aenter__(self) -> StackSupervisor:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()


def create_phuc_stack_supervisor(
    configs: dict[str, SpecialistConfig] | None = None,
) -> StackSupervisor:
    """Create a supervisor with all PHUC stack specialists.

    Args:
        configs: Optional specialist configurations.

    Returns:
        Configured supervisor.
    """
    from .airflow_agent import AirflowAgent, AirflowConfig
    from .cloudflare_agent import CloudflareAgent, CloudflareConfig
    from .kafka_agent import KafkaAgent, KafkaConfig
    from .keycloak_agent import KeycloakAgent, KeycloakConfig
    from .langchain_agent import LangChainAgent, LangChainConfig
    from .mcp_agent import MCPSpecialistAgent, MCPSpecialistConfig
    from .milvus_agent import MilvusAgent, MilvusConfig
    from .minio_agent import MinIOAgent, MinIOConfig
    from .openmetadata_agent import OpenMetadataAgent, OpenMetadataConfig
    from .ovms_agent import OVMSAgent, OVMSConfig
    from .starrocks_agent import StarRocksAgent, StarRocksConfig
    from .unomi_agent import UnomiAgent, UnomiConfig
    from .vault_agent import VaultAgent, VaultConfig

    configs = configs or {}

    supervisor = StackSupervisor()

    # Register all specialists
    specialists = [
        ("minio", MinIOAgent, MinIOConfig, 1),
        ("kafka", KafkaAgent, KafkaConfig, 1),
        ("starrocks", StarRocksAgent, StarRocksConfig, 1),
        ("unomi", UnomiAgent, UnomiConfig, 1),
        ("milvus", MilvusAgent, MilvusConfig, 1),
        ("ovms", OVMSAgent, OVMSConfig, 1),
        ("airflow", AirflowAgent, AirflowConfig, 1),
        ("openmetadata", OpenMetadataAgent, OpenMetadataConfig, 1),
        ("langchain", LangChainAgent, LangChainConfig, 2),  # Higher priority
        ("mcp", MCPSpecialistAgent, MCPSpecialistConfig, 2),
        ("keycloak", KeycloakAgent, KeycloakConfig, 3),  # Security - highest
        ("vault", VaultAgent, VaultConfig, 3),
        ("cloudflare", CloudflareAgent, CloudflareConfig, 2),
    ]

    for name, agent_class, config_class, priority in specialists:
        config = configs.get(name, config_class())
        agent = agent_class(config=config)
        supervisor.register(name, agent, priority=priority)

    return supervisor
