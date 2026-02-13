"""PHUC Platform - Agentic Workflows Integration.

Production-ready modules for:
- Cloudflare Edge Stack (Workers, D1, R2, Vectorize, Workers AI)
- Analytics (Attribution Engine, Campaign Generation, Reporting)
- Security (Injection Defense, Scope Validation, PII Detection)
- Integrations (UID2/Trade Desk, CAMARA MCP, Kong Gateway)

Agent Architecture:
- 4 Main Agents: Architect, Builder, Tester, Shipper
- 13 Sub-Agents: Designer, Coder, SecurityAuditor, Deployer, etc.
- 5 Workers: D1, R2, AI, Analytics, Security
- 11 MCP Skills across 3 domains
"""

__version__ = "2.0.0"

# Submodule imports
from . import cloudflare
from . import analytics
from . import integrations

# Agent hierarchy
from .phuc_agents import (
    Agent,
    SubAgent,
    AgentRole,
    SubAgentRole,
    AgentStatus,
    AgentConfig,
    SubAgentConfig,
    TaskResult,
    AGENT_HIERARCHY,
    create_agent_hierarchy,
    get_agent,
    get_sub_agent,
    list_agents,
)

# MCP Skills
from .phuc_mcp_skills import (
    MCPSkill,
    MCPSkillConfig,
    MCPSkillRegistry,
    SkillDomain,
    SkillLevel,
    SkillInvocation,
    get_skill_registry,
    get_skill,
    list_skills,
    invoke_skill,
    CLOUDFLARE_SKILLS,
    ANALYTICS_SKILLS,
    SECURITY_SKILLS,
    ALL_SKILLS,
)

# Workers
from .phuc_workers import (
    Worker,
    WorkerType,
    WorkerStatus,
    WorkerConfig,
    WorkerResult,
    WorkerPool,
    WORKERS,
    D1_WORKER,
    R2_WORKER,
    AI_WORKER,
    ANALYTICS_WORKER,
    SECURITY_WORKER,
    get_worker_pool,
    execute_tool,
)

# Orchestrator
from .phuc_orchestrator import (
    PhucOrchestrator,
    PipelineStage,
    OrchestrationStatus,
    OrchestrationConfig,
    StageResult,
    OrchestrationResult,
    get_orchestrator,
    execute_pipeline,
)

__all__ = [
    # Version
    "__version__",

    # Agents
    "Agent",
    "SubAgent",
    "AgentRole",
    "SubAgentRole",
    "AgentStatus",
    "AgentConfig",
    "SubAgentConfig",
    "TaskResult",
    "AGENT_HIERARCHY",
    "create_agent_hierarchy",
    "get_agent",
    "get_sub_agent",
    "list_agents",

    # Skills
    "MCPSkill",
    "MCPSkillConfig",
    "MCPSkillRegistry",
    "SkillDomain",
    "SkillLevel",
    "SkillInvocation",
    "get_skill_registry",
    "get_skill",
    "list_skills",
    "invoke_skill",
    "CLOUDFLARE_SKILLS",
    "ANALYTICS_SKILLS",
    "SECURITY_SKILLS",
    "ALL_SKILLS",

    # Workers
    "Worker",
    "WorkerType",
    "WorkerStatus",
    "WorkerConfig",
    "WorkerResult",
    "WorkerPool",
    "WORKERS",
    "D1_WORKER",
    "R2_WORKER",
    "AI_WORKER",
    "ANALYTICS_WORKER",
    "SECURITY_WORKER",
    "get_worker_pool",
    "execute_tool",

    # Orchestrator
    "PhucOrchestrator",
    "PipelineStage",
    "OrchestrationStatus",
    "OrchestrationConfig",
    "StageResult",
    "OrchestrationResult",
    "get_orchestrator",
    "execute_pipeline",
]
