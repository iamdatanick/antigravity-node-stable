"""Agent-Skill Loader for Agentic Workflows v5.0.

Integrates agents with the skill ecosystem by:
- Parsing skills field from agent definitions
- Injecting skill context at agent startup
- Handling skill dependencies
- Managing allowed-tools restrictions
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AgentDefinition:
    """Parsed agent definition from markdown file."""

    name: str
    description: str
    model: str
    tools: list[str]
    skills: list[str]
    security_scope: str
    timeout: int
    max_iterations: int
    recovery: str
    guardrails: list[str]
    triggers: list[str]
    instructions: str
    source_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentSkillLoader:
    """
    Loads and integrates agents with skills.

    Responsibilities:
    1. Parse agent markdown files
    2. Load required skills into context
    3. Validate skill dependencies
    4. Enforce tool restrictions
    """

    def __init__(self, skill_registry, agent_paths: list[Path] = None):
        self.skill_registry = skill_registry
        self.agent_paths = agent_paths or [Path.home() / ".claude" / "agents"]
        self._agents: dict[str, AgentDefinition] = {}

    def parse_agent_md(self, path: Path) -> AgentDefinition:
        """
        Parse agent definition from markdown file.

        Args:
            path: Path to agent markdown file

        Returns:
            Parsed AgentDefinition
        """
        content = path.read_text(encoding="utf-8")

        # Extract YAML frontmatter
        match = re.match(r"^---\n(.*?)\n---\n(.*)$", content, re.DOTALL)
        if not match:
            raise ValueError(f"Invalid agent file format: {path}")

        frontmatter = yaml.safe_load(match.group(1))
        instructions = match.group(2).strip()

        # Parse tools - handle both string and list formats
        tools_raw = frontmatter.get("tools", frontmatter.get("allowed-tools", []))
        if isinstance(tools_raw, str):
            tools = [t.strip() for t in tools_raw.split(",")]
        else:
            tools = tools_raw

        return AgentDefinition(
            name=frontmatter.get("name", path.stem),
            description=frontmatter.get("description", ""),
            model=frontmatter.get("model", "sonnet"),
            tools=tools,
            skills=frontmatter.get("skills", []),
            security_scope=frontmatter.get("security_scope", "standard"),
            timeout=frontmatter.get("timeout", 30000),
            max_iterations=frontmatter.get("max_iterations", 10),
            recovery=frontmatter.get("recovery", "retry"),
            guardrails=frontmatter.get("guardrails", []),
            triggers=frontmatter.get("triggers", []),
            instructions=instructions,
            source_path=path,
            metadata=frontmatter,
        )

    def discover_agents(self) -> int:
        """
        Discover all agents from configured paths.

        Returns:
            Number of agents discovered
        """
        count = 0
        for base_path in self.agent_paths:
            if not base_path.exists():
                continue
            for md_file in base_path.rglob("*.md"):
                try:
                    agent = self.parse_agent_md(md_file)
                    self._agents[agent.name] = agent
                    count += 1
                except Exception as e:
                    print(f"Warning: Failed to parse {md_file}: {e}")
        return count

    def get_agent(self, name: str) -> AgentDefinition | None:
        """Get agent by name."""
        return self._agents.get(name)

    def load_agent_context(self, name: str) -> str:
        """
        Load full context for an agent including skills.

        Args:
            name: Agent name

        Returns:
            Combined context string with agent instructions and skill content
        """
        agent = self._agents.get(name)
        if not agent:
            raise ValueError(f"Agent not found: {name}")

        context_parts = [f"# Agent: {agent.name}\n\n{agent.instructions}"]

        # Load skill contexts
        if agent.skills and self.skill_registry:
            # Resolve dependencies
            all_skills = self.skill_registry.resolve_dependencies(agent.skills)

            context_parts.append("\n\n# Available Skills\n")
            for skill_name in all_skills:
                try:
                    skill_context = self.skill_registry.load_skill_context(skill_name)
                    context_parts.append(f"\n## Skill: {skill_name}\n{skill_context}")
                except Exception as e:
                    context_parts.append(f"\n## Skill: {skill_name}\n(Failed to load: {e})")

        return "\n".join(context_parts)

    def validate_agent_tools(self, agent_name: str) -> list[str]:
        """
        Validate that agent's tools are allowed by its skills.

        Returns:
            List of validation warnings
        """
        agent = self._agents.get(agent_name)
        if not agent:
            return [f"Agent not found: {agent_name}"]

        warnings = []

        # Collect allowed tools from all skills
        allowed_tools = set(agent.tools)

        for skill_name in agent.skills:
            if skill_name in self.skill_registry._skills:
                skill = self.skill_registry._skills[skill_name]
                skill_tools = set(skill.allowed_tools)
                # Check for conflicts
                if skill_tools and not skill_tools.intersection(allowed_tools):
                    warnings.append(
                        f"Skill '{skill_name}' allows tools {skill_tools} "
                        f"but agent has {allowed_tools}"
                    )

        return warnings

    def get_agents_by_trigger(self, trigger: str) -> list[AgentDefinition]:
        """Find agents that match a trigger phrase."""
        matching = []
        trigger_lower = trigger.lower()

        for agent in self._agents.values():
            for t in agent.triggers:
                if t.lower() in trigger_lower or trigger_lower in t.lower():
                    matching.append(agent)
                    break

        return matching

    @property
    def agents(self) -> dict[str, AgentDefinition]:
        """Get all loaded agents."""
        return self._agents.copy()


def create_agent_loader(skill_registry=None) -> AgentSkillLoader:
    """Factory function to create AgentSkillLoader with default paths."""
    return AgentSkillLoader(
        skill_registry=skill_registry,
        agent_paths=[
            Path.home() / ".claude" / "agents",
            Path.home() / ".claude" / "agents" / "phuc-specialists",
            Path.home() / ".claude" / "agents" / "code",
            Path.home() / ".claude" / "agents" / "data",
            Path.home() / ".claude" / "agents" / "meta",
        ],
    )
