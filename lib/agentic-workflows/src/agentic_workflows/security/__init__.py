"""Security module for agentic workflows."""

from agentic_workflows.security.injection_defense import (
    PromptInjectionDefense,
    ScanResult,
    ThreatLevel,
)
from agentic_workflows.security.scope_validator import (
    ScopeValidator,
    Scope,
    ToolPermission,
)
from agentic_workflows.security.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimitExceeded,
    RateLimitStats,
    MultiRateLimiter,
)
from agentic_workflows.security.rate_limiting import (
    RateLimitAlgorithm,
    RateLimitConfig as DistributedRateLimitConfig,
    RateLimitResult,
    RateLimiter as DistributedRateLimiter,
    RedisRateLimiter,
    MemoryRateLimiter,
    create_rate_limiter,
)
from agentic_workflows.security.kill_switch import (
    KillSwitch,
    KillSwitchConfig,
    KillSwitchTriggered,
    KillSwitchState,
    TriggerReason,
)

__all__ = [
    # Injection Defense
    "PromptInjectionDefense",
    "ScanResult",
    "ThreatLevel",
    # Scope Validation
    "ScopeValidator",
    "Scope",
    "ToolPermission",
    # Rate Limiting (legacy in-memory)
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitExceeded",
    "RateLimitStats",
    "MultiRateLimiter",
    # Distributed Rate Limiting (Redis-backed)
    "RateLimitAlgorithm",
    "DistributedRateLimitConfig",
    "RateLimitResult",
    "DistributedRateLimiter",
    "RedisRateLimiter",
    "MemoryRateLimiter",
    "create_rate_limiter",
    # Kill Switch
    "KillSwitch",
    "KillSwitchConfig",
    "KillSwitchTriggered",
    "KillSwitchState",
    "TriggerReason",
]
