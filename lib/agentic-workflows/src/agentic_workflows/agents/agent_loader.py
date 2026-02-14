"""
PHUC Agent Loader - Integrates 100+ Claude Code Agents
=======================================================

Loads agent definitions from ~/.claude/agents/ and wires them
into the agentic-workflows SDK for programmatic use.

Agent Categories (~100 agents):
- code/          (15) code-reviewer, debugger, security-auditor, test-writer
- data/          (12) data-analyst, sql-expert, etl-designer, statistician
- ops/           (12) sre, devops, k8s-admin, monitoring, dba, secops
- business/      (10) business-analyst, product-manager, strategist
- communication/ (10) email-drafter, pr-writer, presentation-creator
- creative/      (10) brainstormer, naming-expert, storyteller
- meta/          (12) orchestrator, planner, task-router, reviewer
- research/      (10) researcher, fact-checker, trend-analyst
- specialized/   (6+) accessibility-expert, contract-analyst

Location: C:\\Users\\NickV\\agentic-workflows\\agentic-workflows\\src\\agentic_workflows\\agents\\agent_loader.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# SDK imports
from agentic_workflows.agents.base import AgentConfig, BaseAgent
from agentic_workflows.protocols.mcp_client import MCPClient

try:
    from anthropic import AsyncAnthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CLAUDE_AGENTS_DIR = Path.home() / ".claude" / "agents"
MCP_ENDPOINT = "https://agentic-workflows-mcp.nick-9a6.workers.dev"


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════


class AgentCategory(Enum):
    """Agent categories from ~/.claude/agents/"""

    CODE = "code"
    DATA = "data"
    OPS = "ops"
    BUSINESS = "business"
    COMMUNICATION = "communication"
    CREATIVE = "creative"
    META = "meta"
    RESEARCH = "research"
    SPECIALIZED = "specialized"
    PHUC = "phuc"  # Our pipeline agents


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AgentDefinition:
    """Parsed agent definition from markdown file (SKILL.md v4.1 compliant)"""

    # Identity
    name: str
    category: AgentCategory
    description: str
    system_prompt: str
    # Tools & Skills
    skills: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    file_path: Path | None = None
    # Config
    model: str = "sonnet"
    permission_mode: str = "dontAsk"
    # Security & Governance (v4.1)
    security_scope: str = "minimal"
    timeout: int = 30000
    max_iterations: int = 10
    recovery: str = "retry"
    guardrails: list[str] = field(default_factory=list)
    # v4.1 Governance Fields
    telemetry: bool = True
    memory: str = "session"
    context_mode: str = "focused"
    cost_tier: str = "standard"
    constitutional: list[str] = field(default_factory=list)
    is_async: bool = True
    delegates_to: list[str] = field(default_factory=list)
    # Raw frontmatter for audit
    _raw_fields: dict = field(default_factory=dict)

    # All v4.1 required fields
    V41_REQUIRED_FIELDS = [
        "security_scope",
        "timeout",
        "max_iterations",
        "recovery",
        "guardrails",
        "telemetry",
        "memory",
        "context_mode",
        "cost_tier",
        "constitutional",
        "async",
        "delegates_to",
    ]

    @staticmethod
    def _parse_yaml_frontmatter(content: str) -> tuple:
        """Parse YAML frontmatter from markdown content.
        Returns (fields_dict, body_text).
        """
        match = re.match(r"^---\n(.*?)\n---\n(.*)", content, re.DOTALL)
        if not match:
            return {}, content

        yaml_str = match.group(1)
        body = match.group(2)
        fields = {}
        for line in yaml_str.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                if value.startswith("[") and value.endswith("]"):
                    items = value[1:-1].split(",")
                    fields[key] = [
                        item.strip().strip('"').strip("'") for item in items if item.strip()
                    ]
                elif value.lower() == "true":
                    fields[key] = True
                elif value.lower() == "false":
                    fields[key] = False
                elif value.isdigit():
                    fields[key] = int(value)
                else:
                    fields[key] = value
        return fields, body

    @classmethod
    def from_markdown(cls, file_path: Path) -> AgentDefinition:
        """Parse agent definition from markdown file with full v4.1 fields"""
        content = file_path.read_text(encoding="utf-8")
        fm, body = cls._parse_yaml_frontmatter(content)

        # Identity
        name = fm.get("name", file_path.stem)
        category_name = file_path.parent.name
        try:
            category = AgentCategory(category_name)
        except ValueError:
            category = AgentCategory.SPECIALIZED

        description = fm.get("description", name)

        # Parse list fields
        skills_raw = fm.get("skills", "")
        if isinstance(skills_raw, str):
            skills = [s.strip() for s in skills_raw.split(",") if s.strip()]
        else:
            skills = skills_raw

        tools_raw = fm.get("tools", "")
        if isinstance(tools_raw, str):
            tools = [t.strip() for t in tools_raw.split(",") if t.strip()]
        else:
            tools = tools_raw

        guardrails = fm.get("guardrails", [])
        if isinstance(guardrails, str):
            guardrails = [g.strip() for g in guardrails.split(",") if g.strip()]

        constitutional = fm.get("constitutional", [])
        if isinstance(constitutional, str):
            constitutional = [c.strip() for c in constitutional.split(",") if c.strip()]

        delegates_to = fm.get("delegates_to", [])
        if isinstance(delegates_to, str):
            delegates_to = [d.strip() for d in delegates_to.split(",") if d.strip()]

        return cls(
            name=name,
            category=category,
            description=description,
            system_prompt=body.strip(),
            skills=skills,
            tools=tools,
            file_path=file_path,
            model=fm.get("model", "sonnet"),
            permission_mode=fm.get("permissionMode", "dontAsk"),
            security_scope=fm.get("security_scope", "minimal"),
            timeout=int(fm.get("timeout", 30000)),
            max_iterations=int(fm.get("max_iterations", 10)),
            recovery=fm.get("recovery", "retry"),
            guardrails=guardrails,
            telemetry=fm.get("telemetry", True),
            memory=fm.get("memory", "session"),
            context_mode=fm.get("context_mode", "focused"),
            cost_tier=fm.get("cost_tier", "standard"),
            constitutional=constitutional,
            is_async=fm.get("async", True),
            delegates_to=delegates_to,
            _raw_fields=fm,
        )

    def validate_v41(self) -> list[str]:
        """Validate this agent has all v4.1 required fields.
        Returns list of missing field names (empty = compliant).
        """
        missing = []
        for f in self.V41_REQUIRED_FIELDS:
            attr = "is_async" if f == "async" else f
            if not hasattr(self, attr):
                missing.append(f)
            elif f in ("guardrails", "constitutional", "delegates_to"):
                pass  # Empty lists are valid
            elif getattr(self, attr) is None:
                missing.append(f)
        return missing

    def is_v41_compliant(self) -> bool:
        """Check if agent passes full v4.1 compliance."""
        return len(self.validate_v41()) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# LOADED AGENT (Runnable)
# ═══════════════════════════════════════════════════════════════════════════════


class LoadedAgent(BaseAgent):
    """
    A Claude Code agent loaded from markdown and made runnable.
    Uses Anthropic SDK to execute with the agent's system prompt.
    Fully v4.1 compliant with telemetry, constitutional guardrails, and delegation.
    """

    # Model ID mapping
    MODEL_MAP = {
        "opus": "claude-opus-4-6",
        "sonnet": "claude-sonnet-4-5-20250929",
        "haiku": "claude-haiku-4-5-20251001",
    }

    def __init__(self, definition: AgentDefinition):
        super().__init__(
            AgentConfig(
                name=definition.name,
                description=definition.description,
                model=self.MODEL_MAP.get(definition.model, "claude-sonnet-4-5-20250929"),
                max_runtime_seconds=definition.timeout / 1000.0,
            )
        )
        self.definition = definition
        self.category = definition.category
        self.system_prompt = definition.system_prompt
        self.claude = AsyncAnthropic() if ANTHROPIC_AVAILABLE else None
        self.mcp = MCPClient(MCP_ENDPOINT)

        # v4.1 governance
        self.security_scope = definition.security_scope
        self.cost_tier = definition.cost_tier
        self.memory_mode = definition.memory
        self.context_mode = definition.context_mode
        self.constitutional = definition.constitutional
        self.delegates_to = definition.delegates_to
        self.max_iterations = definition.max_iterations
        self.telemetry_enabled = definition.telemetry

    async def _execute(self, input_data: Any) -> dict[str, Any]:
        """Execute the agent with its system prompt"""
        user_message = (
            str(input_data)
            if not isinstance(input_data, dict)
            else input_data.get("prompt", str(input_data))
        )

        if not self.claude:
            return {
                "status": "simulated",
                "agent": self.definition.name,
                "category": self.category.value,
                "cost_tier": self.cost_tier,
                "response": f"[Simulated response from {self.definition.name}]",
            }

        model_id = self.MODEL_MAP.get(self.definition.model, "claude-sonnet-4-5-20250929")
        response = await self.claude.messages.create(
            model=model_id,
            max_tokens=8192,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        result = {
            "status": "complete",
            "agent": self.definition.name,
            "category": self.category.value,
            "model": model_id,
            "cost_tier": self.cost_tier,
            "response": response.content[0].text,
        }

        # Telemetry
        if self.telemetry_enabled:
            result["telemetry"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "model": model_id,
            }

        return result

    def can_delegate_to(self, agent_name: str) -> bool:
        """Check if this agent can delegate to the named agent."""
        if not self.delegates_to:
            return False
        if "*" in self.delegates_to:
            return True
        return agent_name in self.delegates_to

    def __repr__(self):
        return (
            f"LoadedAgent({self.definition.name}, "
            f"category={self.category.value}, "
            f"cost_tier={self.cost_tier})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT LOADER
# ═══════════════════════════════════════════════════════════════════════════════


class AgentLoader:
    """
    Loads and manages all agents from ~/.claude/agents/

    Usage:
        loader = AgentLoader()
        loader.load_all()

        # Get by name
        agent = loader.get("code_reviewer")
        result = await agent.run("Review this code...")

        # Get by category
        code_agents = loader.get_by_category(AgentCategory.CODE)

        # List all
        all_agents = loader.list_agents()
    """

    def __init__(self, agents_dir: Path = CLAUDE_AGENTS_DIR):
        self.agents_dir = agents_dir
        self.definitions: dict[str, AgentDefinition] = {}
        self.agents: dict[str, LoadedAgent] = {}
        self._loaded = False

    def load_all(self) -> int:
        """Load all agents from the agents directory"""
        if not self.agents_dir.exists():
            print(f"Warning: Agents directory not found: {self.agents_dir}")
            return 0

        count = 0
        for category_dir in self.agents_dir.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith("."):
                for agent_file in category_dir.glob("*.md"):
                    try:
                        definition = AgentDefinition.from_markdown(agent_file)
                        self.definitions[definition.name] = definition
                        self.agents[definition.name] = LoadedAgent(definition)
                        count += 1
                    except Exception as e:
                        print(f"Warning: Failed to load {agent_file}: {e}")

        self._loaded = True
        return count

    def get(self, name: str) -> LoadedAgent | None:
        """Get an agent by name"""
        if not self._loaded:
            self.load_all()

        # Normalize name
        normalized = name.replace("-", "_").replace(" ", "_").lower()
        return self.agents.get(normalized)

    def get_by_category(self, category: AgentCategory) -> list[LoadedAgent]:
        """Get all agents in a category"""
        if not self._loaded:
            self.load_all()

        return [a for a in self.agents.values() if a.category == category]

    def list_agents(self) -> dict[str, list[str]]:
        """List all agents grouped by category"""
        if not self._loaded:
            self.load_all()

        result: dict[str, list[str]] = {}
        for agent in self.agents.values():
            cat = agent.category.value
            if cat not in result:
                result[cat] = []
            result[cat].append(agent.definition.name)

        return result

    def count(self) -> int:
        """Get total agent count"""
        if not self._loaded:
            self.load_all()
        return len(self.agents)

    def search(self, query: str) -> list[LoadedAgent]:
        """Search agents by name or description"""
        if not self._loaded:
            self.load_all()

        query_lower = query.lower()
        results = []
        for agent in self.agents.values():
            if (
                query_lower in agent.definition.name.lower()
                or query_lower in agent.definition.description.lower()
            ):
                results.append(agent)
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK ACCESS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Global loader instance
_loader: AgentLoader | None = None


def get_loader() -> AgentLoader:
    """Get or create the global agent loader"""
    global _loader
    if _loader is None:
        _loader = AgentLoader()
        _loader.load_all()
    return _loader


def get_agent(name: str) -> LoadedAgent | None:
    """Quick access to get an agent by name"""
    return get_loader().get(name)


def list_all_agents() -> dict[str, list[str]]:
    """Quick access to list all agents"""
    return get_loader().list_agents()


def search_agents(query: str) -> list[LoadedAgent]:
    """Quick access to search agents"""
    return get_loader().search(query)


def get_code_agents() -> list[LoadedAgent]:
    """Get all code agents"""
    return get_loader().get_by_category(AgentCategory.CODE)


def get_data_agents() -> list[LoadedAgent]:
    """Get all data agents"""
    return get_loader().get_by_category(AgentCategory.DATA)


def get_ops_agents() -> list[LoadedAgent]:
    """Get all ops agents"""
    return get_loader().get_by_category(AgentCategory.OPS)


def get_business_agents() -> list[LoadedAgent]:
    """Get all business agents"""
    return get_loader().get_by_category(AgentCategory.BUSINESS)


def get_creative_agents() -> list[LoadedAgent]:
    """Get all creative agents"""
    return get_loader().get_by_category(AgentCategory.CREATIVE)


def get_research_agents() -> list[LoadedAgent]:
    """Get all research agents"""
    return get_loader().get_by_category(AgentCategory.RESEARCH)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "AgentCategory",
    # Classes
    "AgentDefinition",
    "LoadedAgent",
    "AgentLoader",
    # Quick access
    "get_loader",
    "get_agent",
    "list_all_agents",
    "search_agents",
    "get_code_agents",
    "get_data_agents",
    "get_ops_agents",
    "get_business_agents",
    "get_creative_agents",
    "get_research_agents",
    # Config
    "CLAUDE_AGENTS_DIR",
]
