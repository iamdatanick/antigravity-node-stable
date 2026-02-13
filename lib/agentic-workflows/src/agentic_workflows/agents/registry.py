"""Agent registry for managing agent definitions and instances."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Type

from agentic_workflows.agents.base import AgentConfig, BaseAgent


@dataclass
class AgentDefinition:
    """Definition of an agent type in the registry."""

    name: str
    description: str
    config: AgentConfig
    agent_class: Type[BaseAgent] | None = None
    factory: Callable[[AgentConfig], BaseAgent] | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def create_instance(self) -> BaseAgent:
        """Create an instance of this agent.

        Returns:
            New agent instance.

        Raises:
            ValueError: If no class or factory is set.
        """
        if self.factory:
            return self.factory(self.config)
        elif self.agent_class:
            return self.agent_class(self.config)
        else:
            raise ValueError(f"No agent class or factory for: {self.name}")


class AgentRegistry:
    """Registry for agent definitions and instances.

    Features:
    - Register agent types
    - Create agent instances
    - Track active agents
    - Search by tags
    """

    def __init__(self):
        """Initialize registry."""
        self._definitions: dict[str, AgentDefinition] = {}
        self._instances: dict[str, BaseAgent] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        description: str,
        config: AgentConfig,
        agent_class: Type[BaseAgent] | None = None,
        factory: Callable[[AgentConfig], BaseAgent] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentDefinition:
        """Register an agent definition.

        Args:
            name: Unique agent name.
            description: Agent description.
            config: Default configuration.
            agent_class: Agent class to instantiate.
            factory: Factory function for creating agents.
            tags: Tags for searching.
            metadata: Additional metadata.

        Returns:
            Created definition.

        Raises:
            ValueError: If name already exists.
        """
        with self._lock:
            if name in self._definitions:
                raise ValueError(f"Agent '{name}' already registered")

            definition = AgentDefinition(
                name=name,
                description=description,
                config=config,
                agent_class=agent_class,
                factory=factory,
                tags=tags or [],
                metadata=metadata or {},
            )

            self._definitions[name] = definition
            return definition

    def unregister(self, name: str) -> bool:
        """Unregister an agent definition.

        Args:
            name: Agent name.

        Returns:
            True if unregistered.
        """
        with self._lock:
            if name in self._definitions:
                del self._definitions[name]
                return True
            return False

    def get_definition(self, name: str) -> AgentDefinition | None:
        """Get agent definition by name."""
        return self._definitions.get(name)

    def list_definitions(self) -> list[AgentDefinition]:
        """List all registered definitions."""
        return list(self._definitions.values())

    def search_by_tags(self, tags: list[str]) -> list[AgentDefinition]:
        """Search definitions by tags.

        Args:
            tags: Tags to search for (any match).

        Returns:
            Matching definitions.
        """
        return [
            d for d in self._definitions.values()
            if any(t in d.tags for t in tags)
        ]

    def create(
        self,
        name: str,
        config_overrides: dict[str, Any] | None = None,
    ) -> BaseAgent:
        """Create an agent instance from definition.

        Args:
            name: Agent definition name.
            config_overrides: Configuration overrides.

        Returns:
            New agent instance.

        Raises:
            ValueError: If definition not found.
        """
        definition = self.get_definition(name)
        if definition is None:
            raise ValueError(f"Agent definition not found: {name}")

        # Apply config overrides
        config = definition.config
        if config_overrides:
            config_dict = {
                "name": config.name,
                "description": config.description,
                "version": config.version,
                "model": config.model,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "scope": config.scope,
                "allowed_tools": list(config.allowed_tools),
                "blocked_tools": list(config.blocked_tools),
                "max_cost_usd": config.max_cost_usd,
                "max_tokens_total": config.max_tokens_total,
                "max_runtime_seconds": config.max_runtime_seconds,
                "system_prompt": config.system_prompt,
                "personality": config.personality,
                "constraints": list(config.constraints),
                "tags": list(config.tags),
                "metadata": dict(config.metadata),
            }
            config_dict.update(config_overrides)
            config = AgentConfig(**config_dict)

        # Create instance
        instance = definition.create_instance()

        # Track instance
        with self._lock:
            self._instances[instance.agent_id] = instance

        return instance

    def get_instance(self, agent_id: str) -> BaseAgent | None:
        """Get tracked agent instance by ID."""
        return self._instances.get(agent_id)

    def list_instances(self) -> list[BaseAgent]:
        """List all tracked agent instances."""
        return list(self._instances.values())

    def remove_instance(self, agent_id: str) -> bool:
        """Remove tracked instance.

        Args:
            agent_id: Agent instance ID.

        Returns:
            True if removed.
        """
        with self._lock:
            if agent_id in self._instances:
                del self._instances[agent_id]
                return True
            return False

    def clear_instances(self) -> int:
        """Clear all tracked instances.

        Returns:
            Number of instances cleared.
        """
        with self._lock:
            count = len(self._instances)
            self._instances.clear()
            return count

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            return {
                "definitions": len(self._definitions),
                "instances": len(self._instances),
                "definitions_by_tag": self._count_by_tag(),
            }

    def _count_by_tag(self) -> dict[str, int]:
        """Count definitions by tag."""
        counts: dict[str, int] = {}
        for d in self._definitions.values():
            for tag in d.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return counts


# Global registry instance
_global_registry: AgentRegistry | None = None


def get_registry() -> AgentRegistry:
    """Get or create global registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry


def set_registry(registry: AgentRegistry) -> None:
    """Set global registry."""
    global _global_registry
    _global_registry = registry


def register_agent(
    name: str,
    description: str = "",
    **kwargs,
) -> Callable:
    """Decorator to register an agent class.

    Usage:
        @register_agent("my-agent", "My agent description")
        class MyAgent(BaseAgent):
            ...
    """
    def decorator(cls: Type[BaseAgent]) -> Type[BaseAgent]:
        config = AgentConfig(name=name, description=description)
        get_registry().register(
            name=name,
            description=description,
            config=config,
            agent_class=cls,
            **kwargs,
        )
        return cls
    return decorator
