"""Agent Card for capability discovery and description."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class AuthenticationType(Enum):
    """Authentication types."""

    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    OAUTH2 = "oauth2"
    BASIC = "basic"


class CapabilityType(Enum):
    """Types of agent capabilities."""

    CONVERSATION = "conversation"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    DATA_ANALYSIS = "data_analysis"
    DOCUMENT_PROCESSING = "document_processing"
    RESEARCH = "research"
    TASK_EXECUTION = "task_execution"
    FILE_OPERATIONS = "file_operations"
    WEB_BROWSING = "web_browsing"
    API_INTEGRATION = "api_integration"


@dataclass
class Capability:
    """An agent capability."""

    name: str
    capability_type: CapabilityType
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.capability_type.value,
            "description": self.description,
            "inputSchema": self.input_schema,
            "outputSchema": self.output_schema,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Capability:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            capability_type=CapabilityType(data.get("type", "conversation")),
            description=data.get("description", ""),
            input_schema=data.get("inputSchema", {}),
            output_schema=data.get("outputSchema", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Constraint:
    """An agent constraint or limitation."""

    name: str
    description: str
    constraint_type: str = "limitation"  # "limitation", "requirement", "boundary"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.constraint_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Constraint:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            constraint_type=data.get("type", "limitation"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AgentCard:
    """Agent capability card for discovery.

    The Agent Card describes an agent's capabilities, constraints,
    and contact information for discovery by other agents or systems.
    """

    # Identity
    name: str
    description: str
    version: str = "1.0.0"

    # Contact
    url: str = ""
    documentation_url: str = ""
    support_email: str = ""

    # Capabilities
    capabilities: list[Capability] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)

    # Authentication
    authentication_type: AuthenticationType = AuthenticationType.NONE
    authentication_config: dict[str, Any] = field(default_factory=dict)

    # Resources
    tools: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    protocols: list[str] = field(default_factory=list)

    # Limits
    rate_limit: int | None = None  # Requests per minute
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None

    # Metadata
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_capability(
        self,
        name: str,
        capability_type: CapabilityType,
        description: str = "",
        **kwargs,
    ) -> AgentCard:
        """Add a capability.

        Args:
            name: Capability name.
            capability_type: Type of capability.
            description: Description.
            **kwargs: Additional capability options.

        Returns:
            Self for chaining.
        """
        capability = Capability(
            name=name,
            capability_type=capability_type,
            description=description,
            **kwargs,
        )
        self.capabilities.append(capability)
        return self

    def add_constraint(
        self,
        name: str,
        description: str,
        constraint_type: str = "limitation",
    ) -> AgentCard:
        """Add a constraint.

        Args:
            name: Constraint name.
            description: Description.
            constraint_type: Type of constraint.

        Returns:
            Self for chaining.
        """
        constraint = Constraint(
            name=name,
            description=description,
            constraint_type=constraint_type,
        )
        self.constraints.append(constraint)
        return self

    def has_capability(self, capability_type: CapabilityType) -> bool:
        """Check if agent has a capability type."""
        return any(c.capability_type == capability_type for c in self.capabilities)

    def get_capabilities_by_type(self, capability_type: CapabilityType) -> list[Capability]:
        """Get capabilities of a specific type."""
        return [c for c in self.capabilities if c.capability_type == capability_type]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "url": self.url,
            "capabilities": [c.to_dict() for c in self.capabilities],
            "constraints": [c.to_dict() for c in self.constraints],
            "authentication": {
                "type": self.authentication_type.value,
                "config": self.authentication_config,
            },
            "tools": self.tools,
            "models": self.models,
            "protocols": self.protocols,
            "tags": self.tags,
            "metadata": self.metadata,
        }

        if self.documentation_url:
            data["documentationUrl"] = self.documentation_url
        if self.support_email:
            data["supportEmail"] = self.support_email
        if self.rate_limit:
            data["rateLimit"] = self.rate_limit
        if self.max_input_tokens:
            data["maxInputTokens"] = self.max_input_tokens
        if self.max_output_tokens:
            data["maxOutputTokens"] = self.max_output_tokens

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentCard:
        """Create from dictionary."""
        capabilities = [
            Capability.from_dict(c)
            for c in data.get("capabilities", [])
        ]
        constraints = [
            Constraint.from_dict(c)
            for c in data.get("constraints", [])
        ]

        auth_data = data.get("authentication", {})

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            url=data.get("url", ""),
            documentation_url=data.get("documentationUrl", ""),
            support_email=data.get("supportEmail", ""),
            capabilities=capabilities,
            constraints=constraints,
            authentication_type=AuthenticationType(auth_data.get("type", "none")),
            authentication_config=auth_data.get("config", {}),
            tools=data.get("tools", []),
            models=data.get("models", []),
            protocols=data.get("protocols", []),
            rate_limit=data.get("rateLimit"),
            max_input_tokens=data.get("maxInputTokens"),
            max_output_tokens=data.get("maxOutputTokens"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> AgentCard:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> AgentCard:
        """Create from YAML string."""
        return cls.from_dict(yaml.safe_load(yaml_str))

    def save(self, path: str | Path, format: str = "json") -> None:
        """Save agent card to file.

        Args:
            path: File path.
            format: "json" or "yaml".
        """
        path = Path(path)

        if format == "json":
            content = self.to_json()
        elif format == "yaml":
            content = self.to_yaml()
        else:
            raise ValueError(f"Unknown format: {format}")

        path.write_text(content)

    @classmethod
    def load(cls, path: str | Path) -> AgentCard:
        """Load agent card from file.

        Args:
            path: File path.

        Returns:
            Loaded agent card.
        """
        path = Path(path)
        content = path.read_text()

        if path.suffix in (".yml", ".yaml"):
            return cls.from_yaml(content)
        else:
            return cls.from_json(content)

    def generate_openapi_schema(self) -> dict[str, Any]:
        """Generate OpenAPI schema from agent card.

        Returns:
            OpenAPI specification dictionary.
        """
        paths = {}

        for capability in self.capabilities:
            path = f"/capabilities/{capability.name}"
            paths[path] = {
                "post": {
                    "summary": capability.description or capability.name,
                    "operationId": capability.name,
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": capability.input_schema or {"type": "object"},
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": capability.output_schema or {"type": "object"},
                                }
                            }
                        }
                    }
                }
            }

        return {
            "openapi": "3.0.0",
            "info": {
                "title": self.name,
                "description": self.description,
                "version": self.version,
            },
            "servers": [{"url": self.url}] if self.url else [],
            "paths": paths,
        }

    def __str__(self) -> str:
        """String representation."""
        return f"AgentCard({self.name} v{self.version})"


# Builder pattern for easier creation
class AgentCardBuilder:
    """Builder for creating agent cards."""

    def __init__(self, name: str, description: str):
        """Initialize builder."""
        self._card = AgentCard(name=name, description=description)

    def version(self, version: str) -> AgentCardBuilder:
        """Set version."""
        self._card.version = version
        return self

    def url(self, url: str) -> AgentCardBuilder:
        """Set URL."""
        self._card.url = url
        return self

    def documentation(self, url: str) -> AgentCardBuilder:
        """Set documentation URL."""
        self._card.documentation_url = url
        return self

    def capability(
        self,
        name: str,
        capability_type: CapabilityType,
        description: str = "",
        **kwargs,
    ) -> AgentCardBuilder:
        """Add capability."""
        self._card.add_capability(name, capability_type, description, **kwargs)
        return self

    def constraint(
        self,
        name: str,
        description: str,
        constraint_type: str = "limitation",
    ) -> AgentCardBuilder:
        """Add constraint."""
        self._card.add_constraint(name, description, constraint_type)
        return self

    def authentication(
        self,
        auth_type: AuthenticationType,
        config: dict[str, Any] | None = None,
    ) -> AgentCardBuilder:
        """Set authentication."""
        self._card.authentication_type = auth_type
        if config:
            self._card.authentication_config = config
        return self

    def tools(self, *tools: str) -> AgentCardBuilder:
        """Add tools."""
        self._card.tools.extend(tools)
        return self

    def models(self, *models: str) -> AgentCardBuilder:
        """Add models."""
        self._card.models.extend(models)
        return self

    def protocols(self, *protocols: str) -> AgentCardBuilder:
        """Add protocols."""
        self._card.protocols.extend(protocols)
        return self

    def tags(self, *tags: str) -> AgentCardBuilder:
        """Add tags."""
        self._card.tags.extend(tags)
        return self

    def rate_limit(self, limit: int) -> AgentCardBuilder:
        """Set rate limit."""
        self._card.rate_limit = limit
        return self

    def token_limits(
        self,
        max_input: int | None = None,
        max_output: int | None = None,
    ) -> AgentCardBuilder:
        """Set token limits."""
        if max_input:
            self._card.max_input_tokens = max_input
        if max_output:
            self._card.max_output_tokens = max_output
        return self

    def metadata(self, **kwargs) -> AgentCardBuilder:
        """Add metadata."""
        self._card.metadata.update(kwargs)
        return self

    def build(self) -> AgentCard:
        """Build the agent card."""
        return self._card
