#!/usr/bin/env python3
"""
PHUC Platform - Unified Orchestrator

Wires together: Agents, Sub-Agents, Workers, MCP, Skills

Integrates ALL SDKs:
- anthropic (0.75.0) - Claude API for agent reasoning
- mcp (1.12.4) - Model Context Protocol for skill invocation
- openai (2.15.0) - OpenAI API (optional fallback)
- pydantic (2.12.5) - Data validation and serialization
- httpx (0.28.1) - Async HTTP client
- opentelemetry (1.38.0) - Distributed tracing and observability

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                           PHUC ORCHESTRATOR                                  │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │  Agent Layer:     Architect ─► Builder ─► Tester ─► Shipper                 │
    │                       │          │         │         │                       │
    │  Sub-Agent Layer: Designer   Coder    UnitTest  Deployer                    │
    │                   Researcher Documenter IntegTest Monitor                    │
    │                   Planner    Refactor  SecAudit  Rollback                   │
    │                                        PIICheck                              │
    │                       │          │         │         │                       │
    │  Worker Layer:    D1 Worker, R2 Worker, AI Worker, Analytics, Security      │
    │                       │          │         │         │                       │
    │  MCP Layer:       https://agentic-workflows-mcp.nick-9a6.workers.dev        │
    │                       │          │         │         │                       │
    │  Skill Layer:     Cloudflare (5) + Analytics (3) + Security (3) = 11 skills │
    └─────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

import anthropic
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, Field

# MCP imports
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# OpenAI imports (optional)
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Local imports
from .phuc_agents import (
    AGENT_HIERARCHY,
    AgentRole,
    AgentStatus,
    SubAgentRole,
    TaskResult,
    get_agent,
    get_sub_agent,
)
from .phuc_mcp_skills import (
    SkillInvocation,
    get_skill_registry,
    invoke_skill,
)
from .phuc_workers import (
    WorkerResult,
    get_worker_pool,
)

# Import existing SDK modules
try:
    from ..context import ContextGraph
    from ..observability import AlertManager, MetricsCollector
    from ..orchestration import CircuitBreaker, Pipeline, RetryHandler
    from ..security import KillSwitch, PromptInjectionDefense, RateLimiter, ScopeValidator

    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

# Tracer
tracer = trace.get_tracer("phuc.orchestrator")

# Constants
MCP_ENDPOINT = "https://agentic-workflows-mcp.nick-9a6.workers.dev/mcp/sse"
DEPLOYMENT_ID = "d5b2201e-072a-4367-a7a5-099b3d0c9ca7"


class PipelineStage(str, Enum):
    """Stages in the orchestration pipeline."""

    SECURITY = "security"
    ARCHITECT = "architect"
    BUILDER = "builder"
    TESTER = "tester"
    SHIPPER = "shipper"


class OrchestrationStatus(str, Enum):
    """Status of orchestration."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class OrchestrationConfig(BaseModel):
    """Configuration for the orchestrator."""

    mcp_endpoint: str = MCP_ENDPOINT
    deployment_id: str = DEPLOYMENT_ID
    enable_security: bool = True
    enable_tracing: bool = True
    max_retries: int = 3
    timeout_seconds: int = 300
    model: str = "claude-sonnet-4-20250514"


class StageResult(BaseModel):
    """Result from a pipeline stage."""

    stage: PipelineStage
    status: OrchestrationStatus
    agent: str | None = None
    sub_agent: str | None = None
    output: Any | None = None
    error: str | None = None
    duration_ms: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class OrchestrationResult(BaseModel):
    """Result from full orchestration."""

    task_id: str
    status: OrchestrationStatus
    stages: list[StageResult] = Field(default_factory=list)
    final_output: Any | None = None
    total_duration_ms: int = 0
    tokens_used: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


class PhucOrchestrator:
    """
    Unified orchestrator for PHUC Platform.

    Coordinates:
    - 4 Main Agents (Architect, Builder, Tester, Shipper)
    - 13 Sub-Agents (Designer, Coder, Tester, etc.)
    - 5 Workers (D1, R2, AI, Analytics, Security)
    - 11 MCP Skills (Cloudflare, Analytics, Security)
    """

    def __init__(self, config: OrchestrationConfig | None = None):
        self.config = config or OrchestrationConfig()

        # Claude client
        self.anthropic_client = anthropic.Anthropic()

        # OpenAI client (optional)
        self.openai_client = openai.OpenAI() if OPENAI_AVAILABLE else None

        # Agents
        self.agents = AGENT_HIERARCHY

        # Skills
        self.skill_registry = get_skill_registry(self.config.mcp_endpoint)

        # Workers
        self.worker_pool = get_worker_pool(self.config.mcp_endpoint)

        # Security (from SDK)
        if SDK_AVAILABLE and self.config.enable_security:
            self.injection_defense = PromptInjectionDefense(sensitivity=0.8)
            self.rate_limiter = RateLimiter()
            self.kill_switch = KillSwitch()
        else:
            self.injection_defense = None
            self.rate_limiter = None
            self.kill_switch = None

        # Context
        self.context = ContextGraph() if SDK_AVAILABLE else {}

        # Observability
        if SDK_AVAILABLE:
            self.metrics = MetricsCollector()
            self.alerts = AlertManager()
        else:
            self.metrics = None
            self.alerts = None

        # Pipeline
        self._pipeline_stages: dict[PipelineStage, Callable] = {
            PipelineStage.SECURITY: self._security_stage,
            PipelineStage.ARCHITECT: self._architect_stage,
            PipelineStage.BUILDER: self._builder_stage,
            PipelineStage.TESTER: self._tester_stage,
            PipelineStage.SHIPPER: self._shipper_stage,
        }

    async def execute(self, prompt: str, task_id: str | None = None) -> OrchestrationResult:
        """
        Execute the full orchestration pipeline.

        Pipeline: Security → Architect → Builder → Tester → Shipper
        """
        task_id = task_id or f"task_{int(datetime.now().timestamp())}"
        start_time = datetime.now()

        with tracer.start_as_current_span("orchestrator.execute") as span:
            span.set_attribute("task.id", task_id)
            span.set_attribute("deployment.id", self.config.deployment_id)

            result = OrchestrationResult(task_id=task_id, status=OrchestrationStatus.RUNNING)

            task_data = {"prompt": prompt, "task_id": task_id}

            try:
                # Execute each stage
                for stage in PipelineStage:
                    stage_handler = self._pipeline_stages.get(stage)
                    if not stage_handler:
                        continue

                    stage_result = await stage_handler(task_data)
                    result.stages.append(stage_result)

                    # Check for blocking
                    if stage_result.status == OrchestrationStatus.BLOCKED:
                        result.status = OrchestrationStatus.BLOCKED
                        result.final_output = {
                            "blocked_at": stage.value,
                            "reason": stage_result.error,
                        }
                        break

                    if stage_result.status == OrchestrationStatus.FAILED:
                        result.status = OrchestrationStatus.FAILED
                        result.final_output = {
                            "failed_at": stage.value,
                            "error": stage_result.error,
                        }
                        break

                    # Pass output to next stage
                    if stage_result.output:
                        task_data.update(
                            stage_result.output
                            if isinstance(stage_result.output, dict)
                            else {"output": stage_result.output}
                        )

                # Success
                if result.status == OrchestrationStatus.RUNNING:
                    result.status = OrchestrationStatus.COMPLETED
                    result.final_output = task_data

                result.completed_at = datetime.now()
                result.total_duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

                span.set_status(
                    Status(
                        StatusCode.OK
                        if result.status == OrchestrationStatus.COMPLETED
                        else StatusCode.ERROR
                    )
                )
                return result

            except Exception as e:
                span.record_exception(e)
                result.status = OrchestrationStatus.FAILED
                result.final_output = {"error": str(e)}
                result.completed_at = datetime.now()
                result.total_duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                return result

    async def _security_stage(self, task_data: dict[str, Any]) -> StageResult:
        """Security validation stage."""
        start_time = datetime.now()

        with tracer.start_as_current_span("stage.security") as span:
            prompt = task_data.get("prompt", "")

            # Use SDK injection defense if available
            if self.injection_defense:
                scan_result = self.injection_defense.scan(prompt)
                if not scan_result.is_safe:
                    return StageResult(
                        stage=PipelineStage.SECURITY,
                        status=OrchestrationStatus.BLOCKED,
                        error=f"Security threat detected: {scan_result.threat_level.value}",
                        duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                    )

            # Also use MCP security skill
            try:
                skill_result = await invoke_skill(
                    "injection-defense", "scan_prompt", {"text": prompt}
                )
                if skill_result.error:
                    span.add_event("mcp_skill_error", {"error": skill_result.error})
            except Exception as e:
                span.add_event("mcp_skill_exception", {"error": str(e)})

            return StageResult(
                stage=PipelineStage.SECURITY,
                status=OrchestrationStatus.COMPLETED,
                output={"security_passed": True},
                duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
            )

    async def _architect_stage(self, task_data: dict[str, Any]) -> StageResult:
        """Architect agent stage - designs the solution."""
        start_time = datetime.now()

        with tracer.start_as_current_span("stage.architect") as span:
            agent = get_agent(AgentRole.ARCHITECT)
            if not agent:
                return StageResult(
                    stage=PipelineStage.ARCHITECT,
                    status=OrchestrationStatus.FAILED,
                    error="Architect agent not found",
                    duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                )

            # Execute agent
            result = await agent.execute(task_data)

            return StageResult(
                stage=PipelineStage.ARCHITECT,
                status=OrchestrationStatus.COMPLETED
                if result.status == AgentStatus.COMPLETED
                else OrchestrationStatus.FAILED,
                agent=AgentRole.ARCHITECT.value,
                output={"design": result.output},
                error=result.error,
                duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
            )

    async def _builder_stage(self, task_data: dict[str, Any]) -> StageResult:
        """Builder agent stage - implements the design."""
        start_time = datetime.now()

        with tracer.start_as_current_span("stage.builder") as span:
            agent = get_agent(AgentRole.BUILDER)
            if not agent:
                return StageResult(
                    stage=PipelineStage.BUILDER,
                    status=OrchestrationStatus.FAILED,
                    error="Builder agent not found",
                    duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                )

            # Execute agent
            result = await agent.execute(task_data)

            return StageResult(
                stage=PipelineStage.BUILDER,
                status=OrchestrationStatus.COMPLETED
                if result.status == AgentStatus.COMPLETED
                else OrchestrationStatus.FAILED,
                agent=AgentRole.BUILDER.value,
                output={"code": result.output},
                error=result.error,
                duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
            )

    async def _tester_stage(self, task_data: dict[str, Any]) -> StageResult:
        """Tester agent stage - validates the implementation."""
        start_time = datetime.now()

        with tracer.start_as_current_span("stage.tester") as span:
            agent = get_agent(AgentRole.TESTER)
            if not agent:
                return StageResult(
                    stage=PipelineStage.TESTER,
                    status=OrchestrationStatus.FAILED,
                    error="Tester agent not found",
                    duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                )

            # Use security sub-agents
            security_auditor = get_sub_agent(AgentRole.TESTER, SubAgentRole.SECURITY_AUDITOR)
            pii_checker = get_sub_agent(AgentRole.TESTER, SubAgentRole.PII_CHECKER)

            # Execute security audit via MCP
            code = task_data.get("code", "")
            security_result = await self.worker_pool.execute(
                "security_worker", "scan_prompt", {"text": str(code)}
            )
            pii_result = await self.worker_pool.execute(
                "security_worker", "detect_pii", {"text": str(code)}
            )

            # Execute main agent
            result = await agent.execute(task_data)

            test_passed = result.status == AgentStatus.COMPLETED
            return StageResult(
                stage=PipelineStage.TESTER,
                status=OrchestrationStatus.COMPLETED if test_passed else OrchestrationStatus.FAILED,
                agent=AgentRole.TESTER.value,
                output={
                    "test_passed": test_passed,
                    "security_scan": security_result.result,
                    "pii_scan": pii_result.result,
                },
                error=result.error,
                duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
            )

    async def _shipper_stage(self, task_data: dict[str, Any]) -> StageResult:
        """Shipper agent stage - deploys to production."""
        start_time = datetime.now()

        with tracer.start_as_current_span("stage.shipper") as span:
            # Check if tests passed
            if not task_data.get("test_passed", False):
                return StageResult(
                    stage=PipelineStage.SHIPPER,
                    status=OrchestrationStatus.BLOCKED,
                    error="Cannot deploy - tests did not pass",
                    duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                )

            agent = get_agent(AgentRole.SHIPPER)
            if not agent:
                return StageResult(
                    stage=PipelineStage.SHIPPER,
                    status=OrchestrationStatus.FAILED,
                    error="Shipper agent not found",
                    duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                )

            # Execute agent
            result = await agent.execute(task_data)

            # Track deployment via analytics
            if result.status == AgentStatus.COMPLETED:
                await self.worker_pool.execute(
                    "analytics_worker",
                    "attribution_track",
                    {
                        "event": "deployment",
                        "task_id": task_data.get("task_id"),
                        "timestamp": datetime.now().isoformat(),
                    },
                )

            return StageResult(
                stage=PipelineStage.SHIPPER,
                status=OrchestrationStatus.COMPLETED
                if result.status == AgentStatus.COMPLETED
                else OrchestrationStatus.FAILED,
                agent=AgentRole.SHIPPER.value,
                output={
                    "deployed": result.status == AgentStatus.COMPLETED,
                    "result": result.output,
                },
                error=result.error,
                duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
            )

    # Direct access methods
    async def call_skill(self, skill_name: str, tool: str, args: dict[str, Any]) -> SkillInvocation:
        """Call an MCP skill directly."""
        return await invoke_skill(skill_name, tool, args)

    async def call_worker(self, worker_name: str, tool: str, args: dict[str, Any]) -> WorkerResult:
        """Call a worker directly."""
        return await self.worker_pool.execute(worker_name, tool, args)

    async def delegate_to_agent(self, agent_role: AgentRole, task: dict[str, Any]) -> TaskResult:
        """Delegate task to a specific agent."""
        agent = get_agent(agent_role)
        if not agent:
            return TaskResult(
                task_id=task.get("task_id", "unknown"),
                agent=agent_role.value,
                status=AgentStatus.FAILED,
                error=f"Agent {agent_role.value} not found",
            )
        return await agent.execute(task)

    async def delegate_to_sub_agent(
        self, agent_role: AgentRole, sub_agent_role: SubAgentRole, task: dict[str, Any]
    ) -> TaskResult:
        """Delegate task to a specific sub-agent."""
        agent = get_agent(agent_role)
        if not agent:
            return TaskResult(
                task_id=task.get("task_id", "unknown"),
                agent=agent_role.value,
                status=AgentStatus.FAILED,
                error=f"Agent {agent_role.value} not found",
            )
        return await agent.delegate(task, sub_agent_role)

    def get_status(self) -> dict[str, Any]:
        """Get orchestrator status."""
        return {
            "deployment_id": self.config.deployment_id,
            "mcp_endpoint": self.config.mcp_endpoint,
            "agents": {
                "total": len(self.agents),
                "roles": [r.value for r in self.agents.keys()],
                "sub_agents": sum(len(a.sub_agents) for a in self.agents.values()),
            },
            "workers": self.worker_pool.list_workers(),
            "skills": self.skill_registry.get_stats(),
            "sdks": {
                "anthropic": True,
                "mcp": MCP_AVAILABLE,
                "openai": OPENAI_AVAILABLE,
                "sdk_modules": SDK_AVAILABLE,
            },
        }


# Singleton orchestrator
_orchestrator: PhucOrchestrator | None = None


def get_orchestrator(config: OrchestrationConfig | None = None) -> PhucOrchestrator:
    """Get or create the orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PhucOrchestrator(config=config)
    return _orchestrator


async def execute_pipeline(prompt: str, task_id: str | None = None) -> OrchestrationResult:
    """Convenience function to execute the pipeline."""
    return await get_orchestrator().execute(prompt, task_id)


# Exports
__all__ = [
    "PipelineStage",
    "OrchestrationStatus",
    "OrchestrationConfig",
    "StageResult",
    "OrchestrationResult",
    "PhucOrchestrator",
    "get_orchestrator",
    "execute_pipeline",
]
