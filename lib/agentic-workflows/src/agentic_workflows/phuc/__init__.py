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
from . import analytics, cloudflare, integrations

# Agent hierarchy
from .phuc_agents import (
    AGENT_HIERARCHY,
    Agent,
    AgentConfig,
    AgentRole,
    AgentStatus,
    SubAgent,
    SubAgentConfig,
    SubAgentRole,
    TaskResult,
    create_agent_hierarchy,
    get_agent,
    get_sub_agent,
    list_agents,
)

# MCP Skills
from .phuc_mcp_skills import (
    ALL_SKILLS,
    ANALYTICS_SKILLS,
    CLOUDFLARE_SKILLS,
    SECURITY_SKILLS,
    MCPSkill,
    MCPSkillConfig,
    MCPSkillRegistry,
    SkillDomain,
    SkillInvocation,
    SkillLevel,
    get_skill,
    get_skill_registry,
    invoke_skill,
    list_skills,
)

# Orchestrator
from .phuc_orchestrator import (
    OrchestrationConfig,
    OrchestrationResult,
    OrchestrationStatus,
    PhucOrchestrator,
    PipelineStage,
    StageResult,
    execute_pipeline,
    get_orchestrator,
)

# Workers
from .phuc_workers import (
    AI_WORKER,
    ANALYTICS_WORKER,
    D1_WORKER,
    R2_WORKER,
    SECURITY_WORKER,
    WORKERS,
    Worker,
    WorkerConfig,
    WorkerPool,
    WorkerResult,
    WorkerStatus,
    WorkerType,
    execute_tool,
    get_worker_pool,
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
