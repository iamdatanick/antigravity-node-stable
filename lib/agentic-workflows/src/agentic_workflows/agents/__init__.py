"""Agent module for agentic workflows."""

from agentic_workflows.agents.agent_hierarchy import (
    AGENT_SKILLS,
    # Config
    CATEGORY_SKILLS,
    ESCALATION_PATHS,
    HANDOFF_ROUTES,
    AgentDomain,
    AgentHierarchy,
    # Classes
    AgentNode,
    HierarchicalAgent,
    HierarchyBuilder,
    # Enums
    HierarchyLevel,
    get_hierarchical_agent,
    # Quick access functions
    get_hierarchy,
    route_task,
)
from agentic_workflows.agents.agent_loader import (
    AgentCategory,
    LoadedAgent,
    get_agent,
    get_loader,
    list_all_agents,
    search_agents,
)
from agentic_workflows.agents.agent_loader import (
    AgentDefinition as LoaderAgentDefinition,
)
from agentic_workflows.agents.agent_loader import (
    AgentLoader as MarkdownAgentLoader,
)
from agentic_workflows.agents.audit import (
    AgentAuditResult,
    AuditSummary,
    get_audit_summary,
    print_audit_report,
    validate_all_agents,
)
from agentic_workflows.agents.base import (
    AgentConfig,
    AgentContext,
    AgentResult,
    AgentState,
    BaseAgent,
    CompositeAgent,
    SimpleAgent,
)
from agentic_workflows.agents.loader import (
    AgentLoader,
    load_agent_from_yaml,
)
from agentic_workflows.agents.registry import (
    AgentDefinition,
    AgentRegistry,
)
from agentic_workflows.agents.skill_loader import (
    AgentDefinition as SkillAgentDefinition,
)
from agentic_workflows.agents.skill_loader import (
    AgentSkillLoader,
    create_agent_loader,
)

# PHUC Pipeline Agents (lazy import to avoid circular dependency)
_phuc_agents_cache = {}


def __getattr__(name):
    phuc_exports = (
        "PipelineRole",
        "SubAgentRole",
        "SubAgent",
        "ArchitectAgent",
        "BuilderAgent",
        "TesterAgent",
        "ShipperAgent",
        "PIPELINE_AGENTS",
        "SUB_AGENTS",
        "get_pipeline_agent",
        "create_pipeline",
    )
    if name in phuc_exports:
        if name not in _phuc_agents_cache:
            from agentic_workflows.agents import phuc_agents

            _phuc_agents_cache["PipelineRole"] = phuc_agents.PipelineRole
            _phuc_agents_cache["SubAgentRole"] = phuc_agents.SubAgentRole
            _phuc_agents_cache["SubAgent"] = phuc_agents.SubAgent
            _phuc_agents_cache["ArchitectAgent"] = phuc_agents.ArchitectAgent
            _phuc_agents_cache["BuilderAgent"] = phuc_agents.BuilderAgent
            _phuc_agents_cache["TesterAgent"] = phuc_agents.TesterAgent
            _phuc_agents_cache["ShipperAgent"] = phuc_agents.ShipperAgent
            _phuc_agents_cache["PIPELINE_AGENTS"] = phuc_agents.PIPELINE_AGENTS
            _phuc_agents_cache["SUB_AGENTS"] = phuc_agents.SUB_AGENTS
            _phuc_agents_cache["get_pipeline_agent"] = phuc_agents.get_pipeline_agent
            _phuc_agents_cache["create_pipeline"] = phuc_agents.create_pipeline
        return _phuc_agents_cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base
    "BaseAgent",
    "SimpleAgent",
    "CompositeAgent",
    "AgentConfig",
    "AgentContext",
    "AgentResult",
    "AgentState",
    # Registry
    "AgentRegistry",
    "AgentDefinition",
    # Loader
    "AgentLoader",
    "load_agent_from_yaml",
    # Skill Loader
    "AgentSkillLoader",
    "SkillAgentDefinition",
    "create_agent_loader",
    # Agent Hierarchy - Enums
    "AgentLevel",
    "AgentDomain",
    "TaskCategory",
    "EscalationReason",
    # Agent Hierarchy - Models
    "HandoffContext",
    "RoutingResult",
    "HierarchyNode",
    # Agent Hierarchy - Classes
    "HierarchicalAgent",
    "AgentHierarchy",
    "HierarchyBuilder",
    # Agent Hierarchy - Constants
    "HANDOFF_ROUTES",
    "ESCALATION_PATHS",
    "CATEGORY_SKILLS",
    "AGENT_SKILLS",
    "AGENT_CATEGORIES",
    # Agent Hierarchy - Functions
    "create_default_hierarchy",
    "get_hierarchy",
    "get_hierarchical_agent",
    "route_task",
    # Audit
    "get_audit_summary",
    "validate_all_agents",
    "print_audit_report",
    "AuditSummary",
    "AgentAuditResult",
    # Markdown Agent Loader (v4.1)
    "LoaderAgentDefinition",
    "LoadedAgent",
    "MarkdownAgentLoader",
    "AgentCategory",
    "get_loader",
    "get_agent",
    "list_all_agents",
    "search_agents",
    # PHUC Pipeline Agents
    "PipelineRole",
    "SubAgentRole",
    "SubAgent",
    "ArchitectAgent",
    "BuilderAgent",
    "TesterAgent",
    "ShipperAgent",
    "PIPELINE_AGENTS",
    "SUB_AGENTS",
    "get_pipeline_agent",
    "create_pipeline",
]
