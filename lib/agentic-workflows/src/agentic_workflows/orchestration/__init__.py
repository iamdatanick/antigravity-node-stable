"""Orchestration module for agentic workflows."""

from agentic_workflows.orchestration.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitState,
)
from agentic_workflows.orchestration.parallel import (
    ParallelExecutor,
    ParallelResult,
)
from agentic_workflows.orchestration.pipeline import (
    Pipeline,
    PipelineResult,
    PipelineStage,
)
from agentic_workflows.orchestration.retry import (
    Retrier,
    RetryConfig,
    RetryExhausted,
)
from agentic_workflows.orchestration.supervisor import (
    Supervisor,
    Task,
    TaskResult,
    TaskStatus,
)

# PHUC Orchestrator (lazy import to avoid circular dependency)
_phuc_cache = {}


def __getattr__(name):
    if name in (
        "PhucOrchestrator",
        "PhucPipeline",
        "SecurityGate",
        "OrchestrationResult",
        "get_orchestrator",
        "execute_pipeline",
    ):
        if name not in _phuc_cache:
            from agentic_workflows.orchestration import phuc_orchestrator

            _phuc_cache["PhucOrchestrator"] = phuc_orchestrator.PhucOrchestrator
            _phuc_cache["PhucPipeline"] = phuc_orchestrator.PhucPipeline
            _phuc_cache["SecurityGate"] = phuc_orchestrator.SecurityGate
            _phuc_cache["OrchestrationResult"] = phuc_orchestrator.OrchestrationResult
            _phuc_cache["get_orchestrator"] = phuc_orchestrator.get_orchestrator
            _phuc_cache["execute_pipeline"] = phuc_orchestrator.execute_pipeline
        return _phuc_cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpen",
    "CircuitState",
    # Retry
    "Retrier",
    "RetryConfig",
    "RetryExhausted",
    # Supervisor
    "Supervisor",
    "Task",
    "TaskResult",
    "TaskStatus",
    # Pipeline
    "Pipeline",
    "PipelineStage",
    "PipelineResult",
    # Parallel
    "ParallelExecutor",
    "ParallelResult",
    # PHUC Orchestrator
    "PhucOrchestrator",
    "PhucPipeline",
    "SecurityGate",
    "OrchestrationResult",
    "get_orchestrator",
    "execute_pipeline",
]
