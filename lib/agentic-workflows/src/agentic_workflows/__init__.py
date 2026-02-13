"""
Agentic Workflows - Production multi-agent workflow toolkit for Claude Code.

This package provides:
- Security: Injection defense, scope validation, rate limiting, kill switch
- Orchestration: Circuit breaker, retry, supervisor, pipeline, parallel execution
- Context: DAG-based context graph with provenance and trust scoring
- Observability: Metrics, tracing, and alerting
- Handoffs: Agent handoffs, checkpoints, and recovery
- Protocols: MCP and A2A client implementations
- Agents: Base agent class, registry, and templates
- Artifacts: Generation, storage, and management utilities
- Skills: Skill registry with progressive loading and dependency resolution
- OpenAI Apps: Widget integration for OpenAI Apps SDK
- Guardrails: Jailbreak, PII, topic filtering
- Hooks: Pre/post tool use lifecycle hooks
- Integrations: AutoGen, CrewAI, DSPy, LlamaIndex, Claude SDK, OpenAI SDK adapters
- MCP: Auth, prompts, resources, sampling, transports
"""

__version__ = "5.1.0"

# PHUC Platform modules
# OpenAI Apps SDK Integration
# A2A Protocol Integration
# New unified modules
from agentic_workflows import a2a, guardrails, hooks, integrations, mcp, openai_apps, phuc
from agentic_workflows.artifacts import (
    Artifact,
    ArtifactGenerator,
    ArtifactManager,
    ArtifactMetadata,
    ArtifactRef,
    ArtifactStorage,
    ArtifactType,
    FileStorage,
    MemoryStorage,
)
from agentic_workflows.context import (
    ContextGraph,
    ContextNode,
    Provenance,
    TrustCalculator,
)
from agentic_workflows.handoffs import (
    CheckpointManager,
    HandoffManager,
    RecoveryOrchestrator,
)
from agentic_workflows.observability import (
    AgentTracer,
    AlertManager,
    MetricsCollector,
    Model,
)
from agentic_workflows.orchestration import (
    CircuitBreaker,
    ParallelExecutor,
    Pipeline,
    Retrier,
    Supervisor,
)
from agentic_workflows.security import (
    KillSwitch,
    PromptInjectionDefense,
    RateLimiter,
    Scope,
    ScopeValidator,
    ThreatLevel,
)

# Skills module
from agentic_workflows.skills import (
    MCP_ENDPOINT,
    SKILLS,
    # PHUC MCP Skills
    MCPSkill,
    SkillDefinition,
    SkillDomain,
    SkillLevel,
    SkillManager,
    SkillRegistry,
    discover_default_skills,
    get_all_tools,
    get_analytics_skills,
    get_cloudflare_skills,
    get_security_skills,
)
from agentic_workflows.skills import (
    get_registry as get_skill_registry,
)

# Workers module
from agentic_workflows.workers import (
    AI_WORKER,
    ANALYTICS_WORKER,
    CAMARA_WORKER,
    D1_WORKER,
    R2_WORKER,
    SECURITY_WORKER,
    WORKERS,
    CloudflareWorker,
    WorkerInfo,
    WorkerPool,
    WorkerPoolStatus,
    WorkerResult,
    WorkerStatus,
    WorkerType,
)

# Pipeline Agents and Orchestrator use lazy imports to avoid circular dependencies
# Access via: agentic_workflows.PipelineRole, agentic_workflows.PhucOrchestrator, etc.
_lazy_imports = {
    # Pipeline Agents
    "PipelineRole": ("agentic_workflows.agents.phuc_agents", "PipelineRole"),
    "SubAgentRole": ("agentic_workflows.agents.phuc_agents", "SubAgentRole"),
    "SubAgent": ("agentic_workflows.agents.phuc_agents", "SubAgent"),
    "PipelineAgent": ("agentic_workflows.agents.phuc_agents", "PipelineAgent"),
    "ArchitectAgent": ("agentic_workflows.agents.phuc_agents", "ArchitectAgent"),
    "BuilderAgent": ("agentic_workflows.agents.phuc_agents", "BuilderAgent"),
    "TesterAgent": ("agentic_workflows.agents.phuc_agents", "TesterAgent"),
    "ShipperAgent": ("agentic_workflows.agents.phuc_agents", "ShipperAgent"),
    "SUB_AGENTS": ("agentic_workflows.agents.phuc_agents", "SUB_AGENTS"),
    "PIPELINE_AGENTS": ("agentic_workflows.agents.phuc_agents", "PIPELINE_AGENTS"),
    "create_agent": ("agentic_workflows.agents.phuc_agents", "create_agent"),
    # Orchestrator
    "PhucOrchestrator": ("agentic_workflows.orchestration.phuc_orchestrator", "PhucOrchestrator"),
    "PhucPipeline": ("agentic_workflows.orchestration.phuc_orchestrator", "PhucPipeline"),
    "SecurityGate": ("agentic_workflows.orchestration.phuc_orchestrator", "SecurityGate"),
    "PipelineResult": ("agentic_workflows.orchestration.phuc_orchestrator", "PipelineResult"),
    "create_orchestrator": (
        "agentic_workflows.orchestration.phuc_orchestrator",
        "create_orchestrator",
    ),
}

_lazy_cache = {}


def __getattr__(name):
    if name in _lazy_imports:
        if name not in _lazy_cache:
            module_name, attr_name = _lazy_imports[name]
            import importlib

            module = importlib.import_module(module_name)
            _lazy_cache[name] = getattr(module, attr_name)
        return _lazy_cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Security
    "PromptInjectionDefense",
    "ScopeValidator",
    "RateLimiter",
    "KillSwitch",
    "ThreatLevel",
    "Scope",
    # Orchestration
    "CircuitBreaker",
    "Retrier",
    "Supervisor",
    "Pipeline",
    "ParallelExecutor",
    # Context
    "ContextGraph",
    "ContextNode",
    "Provenance",
    "TrustCalculator",
    # Observability
    "MetricsCollector",
    "AgentTracer",
    "AlertManager",
    "Model",
    # Handoffs
    "HandoffManager",
    "CheckpointManager",
    "RecoveryOrchestrator",
    # Artifacts
    "ArtifactGenerator",
    "Artifact",
    "ArtifactType",
    "ArtifactMetadata",
    "ArtifactStorage",
    "MemoryStorage",
    "FileStorage",
    "ArtifactManager",
    "ArtifactRef",
    # PHUC Platform
    "phuc",
    # OpenAI Apps SDK
    "openai_apps",
    # A2A Protocol
    "a2a",
    # Guardrails
    "guardrails",
    # Hooks
    "hooks",
    # Integrations
    "integrations",
    # MCP
    "mcp",
    # Skills
    "SkillRegistry",
    "SkillDefinition",
    "SkillLevel",
    "SkillDomain",
    "get_skill_registry",
    "discover_default_skills",
    # PHUC MCP Skills
    "MCPSkill",
    "SkillManager",
    "SKILLS",
    "get_cloudflare_skills",
    "get_analytics_skills",
    "get_security_skills",
    "get_all_tools",
    "MCP_ENDPOINT",
    # Workers
    "WorkerType",
    "WorkerStatus",
    "WorkerResult",
    "WorkerInfo",
    "WorkerPoolStatus",
    "CloudflareWorker",
    "WorkerPool",
    "D1_WORKER",
    "R2_WORKER",
    "AI_WORKER",
    "ANALYTICS_WORKER",
    "SECURITY_WORKER",
    "CAMARA_WORKER",
    "WORKERS",
    # Pipeline Agents
    "PipelineRole",
    "SubAgentRole",
    "SubAgent",
    "PipelineAgent",
    "ArchitectAgent",
    "BuilderAgent",
    "TesterAgent",
    "ShipperAgent",
    "SUB_AGENTS",
    "PIPELINE_AGENTS",
    "create_agent",
    # Orchestrator
    "PhucOrchestrator",
    "PhucPipeline",
    "SecurityGate",
    "PipelineResult",
    "create_orchestrator",
]
