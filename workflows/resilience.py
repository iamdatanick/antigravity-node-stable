"""Resilience primitives powered by agentic-workflows.

Provides circuit breakers, rate limiting, retry, and kill switch
for the Antigravity Node orchestrator.
"""

import logging

logger = logging.getLogger("antigravity.resilience")

# Try to import agentic-workflows; fall back to no-op stubs if not installed.
# This keeps the orchestrator functional even without the library.
try:
    from agentic_workflows.orchestration.circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerConfig,
    )
    from agentic_workflows.orchestration.retry import Retrier, RetryConfig
    from agentic_workflows.security.rate_limiter import RateLimiter, RateLimitConfig
    from agentic_workflows.security.kill_switch import (
        KillSwitch,
        KillSwitchConfig,
        TriggerReason,
    )

    _HAS_AGENTIC = True
    logger.info("agentic-workflows loaded: circuit breakers, rate limiting, kill switch active")
except ImportError:
    _HAS_AGENTIC = False
    logger.warning("agentic-workflows not installed — resilience features disabled")


# ---------------------------------------------------------------------------
# Circuit Breakers — isolate failures in external service calls
# ---------------------------------------------------------------------------

if _HAS_AGENTIC:
    _cb_config = CircuitBreakerConfig(
        failure_threshold=5,
        timeout_seconds=30.0,
        success_threshold=2,
    )
    circuit_breaker_ovms = CircuitBreaker(name="ovms", config=_cb_config)
    circuit_breaker_ollama = CircuitBreaker(name="ollama", config=_cb_config)
    circuit_breaker_s3 = CircuitBreaker(name="s3", config=_cb_config)
else:
    circuit_breaker_ovms = None
    circuit_breaker_ollama = None
    circuit_breaker_s3 = None


# ---------------------------------------------------------------------------
# Retrier — exponential backoff with jitter for transient failures
# ---------------------------------------------------------------------------

if _HAS_AGENTIC:
    retrier = Retrier(config=RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        max_delay=15.0,
        jitter=0.5,
    ))
else:
    retrier = None


# ---------------------------------------------------------------------------
# Rate Limiter — protect endpoints from abuse
# ---------------------------------------------------------------------------

if _HAS_AGENTIC:
    rate_limiter = RateLimiter(config=RateLimitConfig(
        requests_per_second=10.0,
        burst_size=100,
    ))
    rate_limiter_strict = RateLimiter(config=RateLimitConfig(
        requests_per_second=2.0,
        burst_size=30,
    ))
else:
    rate_limiter = None
    rate_limiter_strict = None


# ---------------------------------------------------------------------------
# Kill Switch — emergency stop
# ---------------------------------------------------------------------------

if _HAS_AGENTIC:
    kill_switch = KillSwitch(config=KillSwitchConfig(
        max_consecutive_errors=10,
    ))
else:
    kill_switch = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_circuit_states() -> dict:
    """Return current state of all circuit breakers."""
    if not _HAS_AGENTIC:
        return {"enabled": False, "reason": "agentic-workflows not installed"}

    result = {}
    for name, cb in [("ovms", circuit_breaker_ovms), ("ollama", circuit_breaker_ollama), ("s3", circuit_breaker_s3)]:
        result[name] = cb.get_status()
    return {"enabled": True, "circuits": result}


def is_killed() -> bool:
    """Check if the kill switch has been triggered."""
    if kill_switch is None:
        return False
    return kill_switch.is_active


def trigger_kill(reason: str = "Manual admin trigger") -> dict:
    """Trigger the kill switch. Returns status dict."""
    if kill_switch is None:
        return {"status": "unavailable", "message": "agentic-workflows not installed"}
    kill_switch.trigger(reason=TriggerReason.MANUAL, message=reason)
    return {"status": "triggered", "message": reason}
