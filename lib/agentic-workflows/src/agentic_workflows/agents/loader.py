"""Agent loader for YAML-based agent definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agentic_workflows.agents.base import AgentConfig, BaseAgent, SimpleAgent
from agentic_workflows.agents.registry import AgentRegistry, get_registry
from agentic_workflows.security.scope_validator import Scope


@dataclass
class AgentTemplate:
    """Parsed agent template."""

    # Frontmatter
    name: str
    description: str = ""
    model: str = "claude-sonnet-4"
    scope: str = "standard"
    version: str = "1.0.0"
    tools: list[str] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Body sections
    identity: str = ""
    capabilities: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    instructions: str = ""

    def to_config(self) -> AgentConfig:
        """Convert template to AgentConfig."""
        scope_map = {
            "minimal": Scope.MINIMAL,
            "standard": Scope.STANDARD,
            "elevated": Scope.ELEVATED,
            "admin": Scope.ADMIN,
        }

        return AgentConfig(
            name=self.name,
            description=self.description,
            version=self.version,
            model=self.model,
            scope=scope_map.get(self.scope.lower(), Scope.STANDARD),
            allowed_tools=self.tools,
            max_cost_usd=self.budget.get("max_cost_usd"),
            max_tokens_total=self.budget.get("max_tokens"),
            max_runtime_seconds=self.budget.get("max_runtime_seconds"),
            system_prompt=self._build_system_prompt(),
            constraints=self.constraints,
            metadata=self.metadata,
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt from template sections."""
        sections = []

        if self.identity:
            sections.append(f"## Identity\n{self.identity}")

        if self.capabilities:
            caps = "\n".join(f"- {c}" for c in self.capabilities)
            sections.append(f"## Capabilities\n{caps}")

        if self.constraints:
            cons = "\n".join(f"- {c}" for c in self.constraints)
            sections.append(f"## Constraints\n{cons}")

        if self.instructions:
            sections.append(f"## Instructions\n{self.instructions}")

        if self.examples:
            exs = "\n\n".join(self.examples)
            sections.append(f"## Examples\n{exs}")

        return "\n\n".join(sections)


def parse_agent_template(content: str) -> AgentTemplate:
    """Parse agent template from markdown with YAML frontmatter.

    Args:
        content: Template content.

    Returns:
        Parsed template.
    """
    # Split frontmatter and body
    frontmatter = {}
    body = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                pass
            body = parts[2].strip()

    # Parse frontmatter
    template = AgentTemplate(
        name=frontmatter.get("name", "unnamed"),
        description=frontmatter.get("description", ""),
        model=frontmatter.get("model", "claude-sonnet-4"),
        scope=frontmatter.get("scope", "standard"),
        version=frontmatter.get("version", "1.0.0"),
        tools=frontmatter.get("tools", []),
        budget=frontmatter.get("budget", {}),
        metadata=frontmatter.get("metadata", {}),
    )

    # Parse body sections
    sections = _parse_markdown_sections(body)

    template.identity = sections.get("identity", "")
    template.capabilities = _parse_list(sections.get("capabilities", ""))
    template.constraints = _parse_list(sections.get("constraints", ""))
    template.examples = _parse_examples(sections.get("examples", ""))
    template.instructions = sections.get("instructions", "")

    return template


def _parse_markdown_sections(content: str) -> dict[str, str]:
    """Parse markdown into sections by headers."""
    sections: dict[str, str] = {}
    current_section = ""
    current_content: list[str] = []

    for line in content.split("\n"):
        if line.startswith("## "):
            # Save previous section
            if current_section:
                sections[current_section.lower()] = "\n".join(current_content).strip()

            # Start new section
            current_section = line[3:].strip()
            current_content = []
        else:
            current_content.append(line)

    # Save last section
    if current_section:
        sections[current_section.lower()] = "\n".join(current_content).strip()

    return sections


def _parse_list(content: str) -> list[str]:
    """Parse markdown list items."""
    items = []
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("- "):
            items.append(line[2:].strip())
        elif line.startswith("* "):
            items.append(line[2:].strip())
    return items


def _parse_examples(content: str) -> list[str]:
    """Parse example blocks."""
    examples = []
    current = []

    for line in content.split("\n"):
        if line.startswith("### ") or line.startswith("**Example"):
            if current:
                examples.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        examples.append("\n".join(current).strip())

    return [e for e in examples if e]


def load_agent_from_yaml(path: str | Path) -> AgentConfig:
    """Load agent configuration from YAML file.

    Args:
        path: Path to YAML file.

    Returns:
        Agent configuration.
    """
    path = Path(path)
    content = path.read_text()

    # Check if it's markdown with frontmatter or pure YAML
    if content.startswith("---") and "## " in content:
        # Markdown template
        template = parse_agent_template(content)
        return template.to_config()
    else:
        # Pure YAML
        data = yaml.safe_load(content)
        return _yaml_to_config(data)


def _yaml_to_config(data: dict[str, Any]) -> AgentConfig:
    """Convert YAML data to AgentConfig."""
    scope_map = {
        "minimal": Scope.MINIMAL,
        "standard": Scope.STANDARD,
        "elevated": Scope.ELEVATED,
        "admin": Scope.ADMIN,
    }

    budget = data.get("budget", {})

    return AgentConfig(
        name=data.get("name", "unnamed"),
        description=data.get("description", ""),
        version=data.get("version", "1.0.0"),
        model=data.get("model", "claude-sonnet-4"),
        scope=scope_map.get(data.get("scope", "standard").lower(), Scope.STANDARD),
        allowed_tools=data.get("tools", []),
        max_cost_usd=budget.get("max_cost_usd"),
        max_tokens_total=budget.get("max_tokens"),
        max_runtime_seconds=budget.get("max_runtime_seconds"),
        system_prompt=data.get("system_prompt", ""),
        constraints=data.get("constraints", []),
        tags=data.get("tags", []),
        metadata=data.get("metadata", {}),
    )


class AgentLoader:
    """Loader for agent definitions from files.

    Features:
    - Load from YAML/Markdown files
    - Load from directories
    - Register with global registry
    """

    def __init__(self, registry: AgentRegistry | None = None):
        """Initialize loader.

        Args:
            registry: Registry to use (default: global).
        """
        self.registry = registry or get_registry()

    def load_file(
        self,
        path: str | Path,
        register: bool = True,
    ) -> AgentConfig:
        """Load agent from file.

        Args:
            path: Path to agent file.
            register: Whether to register in registry.

        Returns:
            Loaded configuration.
        """
        path = Path(path)
        config = load_agent_from_yaml(path)

        if register:
            self.registry.register(
                name=config.name,
                description=config.description,
                config=config,
                agent_class=SimpleAgent,
                tags=config.tags,
                metadata={"source_file": str(path)},
            )

        return config

    def load_directory(
        self,
        directory: str | Path,
        pattern: str = "*.yaml",
        recursive: bool = False,
        register: bool = True,
    ) -> list[AgentConfig]:
        """Load all agents from directory.

        Args:
            directory: Directory path.
            pattern: File pattern to match.
            recursive: Search recursively.
            register: Whether to register.

        Returns:
            List of loaded configurations.
        """
        directory = Path(directory)
        configs = []

        if recursive:
            files = directory.rglob(pattern)
        else:
            files = directory.glob(pattern)

        for file in files:
            try:
                config = self.load_file(file, register=register)
                configs.append(config)
            except Exception as e:
                # Log error but continue
                print(f"Warning: Failed to load {file}: {e}")

        # Also try markdown files
        md_pattern = pattern.replace(".yaml", ".md").replace(".yml", ".md")
        if recursive:
            md_files = directory.rglob(md_pattern)
        else:
            md_files = directory.glob(md_pattern)

        for file in md_files:
            try:
                config = self.load_file(file, register=register)
                configs.append(config)
            except Exception:
                pass

        return configs

    def create_from_file(
        self,
        path: str | Path,
        config_overrides: dict[str, Any] | None = None,
    ) -> BaseAgent:
        """Create agent instance from file.

        Args:
            path: Path to agent file.
            config_overrides: Configuration overrides.

        Returns:
            Agent instance.
        """
        config = load_agent_from_yaml(path)

        if config_overrides:
            config_dict = {
                "name": config.name,
                "description": config.description,
                "version": config.version,
                "model": config.model,
                "scope": config.scope,
                "allowed_tools": list(config.allowed_tools),
                "max_cost_usd": config.max_cost_usd,
                "max_tokens_total": config.max_tokens_total,
                "max_runtime_seconds": config.max_runtime_seconds,
                "system_prompt": config.system_prompt,
                "constraints": list(config.constraints),
                "tags": list(config.tags),
                "metadata": dict(config.metadata),
            }
            config_dict.update(config_overrides)
            config = AgentConfig(**config_dict)

        return SimpleAgent(config, handler=lambda x: x)

    def get_template_content(
        self,
        name: str,
        description: str = "",
        model: str = "claude-sonnet-4",
        scope: str = "standard",
        tools: list[str] | None = None,
    ) -> str:
        """Generate template content for a new agent.

        Args:
            name: Agent name.
            description: Agent description.
            model: Model to use.
            scope: Security scope.
            tools: Allowed tools.

        Returns:
            Template markdown content.
        """
        tools_str = "\n".join(f"  - {t}" for t in (tools or ["Read", "Glob", "Grep"]))

        return f"""---
name: {name}
description: {description}
model: {model}
scope: {scope}
tools:
{tools_str}
budget:
  max_tokens: 50000
  max_cost_usd: 1.00
---

## Identity
You are {name}, an AI assistant.

## Capabilities
- Capability 1
- Capability 2

## Constraints
- Constraint 1
- Constraint 2

## Instructions
Your instructions go here.
"""
