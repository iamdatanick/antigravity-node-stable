"""
Self-Healing and Recovery Patterns

Provides circuit breakers, retry logic, fallback chains,
and recovery orchestration for resilient agent workflows.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from functools import wraps


T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RecoveryConfig:
    """Configuration for recovery behavior"""
    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff_multiplier: float = 2.0
    retry_max_delay_seconds: float = 30.0

    # Circuit breaker settings
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    half_open_max_calls: int = 3

    # Fallback settings
    fallback_agents: List[str] = field(default_factory=list)
    fallback_timeout_seconds: float = 60.0

    # Escalation settings
    escalate_after_retries: bool = True
    escalate_to: str = "human"
    max_recovery_iterations: int = 5
    require_human_approval_after: int = 3


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt"""
    timestamp: datetime
    strategy: str  # "retry", "fallback", "escalate"
    success: bool
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by failing fast when a service
    is detected as unhealthy.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0

    def can_execute(self) -> bool:
        """Check if circuit allows execution"""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self._transition_to_half_open()
                    return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls

        return False

    def record_success(self):
        """Record a successful execution"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                self._transition_to_closed()
        else:
            self.failure_count = 0

    def record_failure(self, error: str = None):
        """Record a failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        elif self.failure_count >= self.failure_threshold:
            self._transition_to_open()

    def get_status(self) -> Dict:
        """Get circuit breaker status"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None
        }

    def _transition_to_open(self):
        """Transition to open state"""
        self.state = CircuitState.OPEN
        self.success_count = 0

    def _transition_to_half_open(self):
        """Transition to half-open state"""
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.success_count = 0

    def _transition_to_closed(self):
        """Transition to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0


class RetryHandler:
    """
    Retry logic with exponential backoff.
    """

    def __init__(self, config: RecoveryConfig):
        self.config = config

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retry logic"""
        last_error = None
        delay = self.config.retry_delay_seconds

        for attempt in range(self.config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except Exception as e:
                last_error = e

                if attempt < self.config.max_retries:
                    await asyncio.sleep(delay)
                    delay = min(
                        delay * self.config.retry_backoff_multiplier,
                        self.config.retry_max_delay_seconds
                    )

        raise last_error

    def execute_with_retry_sync(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retry logic (synchronous)"""
        last_error = None
        delay = self.config.retry_delay_seconds

        for attempt in range(self.config.max_retries + 1):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_error = e

                if attempt < self.config.max_retries:
                    time.sleep(delay)
                    delay = min(
                        delay * self.config.retry_backoff_multiplier,
                        self.config.retry_max_delay_seconds
                    )

        raise last_error


@dataclass
class FallbackChain:
    """
    Fallback chain for graceful degradation.
    """
    primary: str
    fallbacks: List[str]
    current_index: int = 0
    attempts: List[RecoveryAttempt] = field(default_factory=list)

    def get_current(self) -> str:
        """Get current agent/service in chain"""
        if self.current_index == 0:
            return self.primary
        return self.fallbacks[self.current_index - 1]

    def advance(self, error: str = None) -> Optional[str]:
        """Move to next fallback in chain"""
        self.attempts.append(RecoveryAttempt(
            timestamp=datetime.utcnow(),
            strategy="fallback",
            success=False,
            error=error
        ))

        if self.current_index < len(self.fallbacks):
            self.current_index += 1
            return self.get_current()
        return None

    def has_fallback(self) -> bool:
        """Check if there are more fallbacks available"""
        return self.current_index < len(self.fallbacks)

    def reset(self):
        """Reset chain to primary"""
        self.current_index = 0


class RecoveryOrchestrator:
    """
    Orchestrates recovery strategies for agent workflows.

    Combines circuit breakers, retries, fallbacks, and escalation
    into a unified recovery system.
    """

    def __init__(self, config: RecoveryConfig):
        self.config = config
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handler = RetryHandler(config)
        self.recovery_history: List[RecoveryAttempt] = []
        self.escalation_callbacks: List[Callable] = []

    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a service"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=self.config.failure_threshold,
                recovery_timeout=self.config.recovery_timeout_seconds,
                half_open_max_calls=self.config.half_open_max_calls
            )
        return self.circuit_breakers[name]

    async def execute_with_recovery(
        self,
        agent_id: str,
        func: Callable,
        fallback_chain: FallbackChain = None,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute with full recovery strategy.

        Order of operations:
        1. Check circuit breaker
        2. Try execution with retries
        3. If fails, try fallback chain
        4. If all fail, escalate
        """
        circuit = self.get_circuit_breaker(agent_id)
        start_time = datetime.utcnow()

        # Check circuit breaker
        if not circuit.can_execute():
            if fallback_chain and fallback_chain.has_fallback():
                next_agent = fallback_chain.advance(f"Circuit open for {agent_id}")
                return await self.execute_with_recovery(
                    next_agent, func, fallback_chain, *args, **kwargs
                )
            else:
                await self._escalate(agent_id, "Circuit breaker open, no fallbacks")
                raise RuntimeError(f"Circuit breaker open for {agent_id}")

        # Try with retries
        try:
            result = await self.retry_handler.execute_with_retry(func, *args, **kwargs)
            circuit.record_success()

            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.recovery_history.append(RecoveryAttempt(
                timestamp=start_time,
                strategy="primary",
                success=True,
                duration_ms=duration
            ))

            return result

        except Exception as e:
            circuit.record_failure(str(e))

            # Try fallback chain
            if fallback_chain and fallback_chain.has_fallback():
                next_agent = fallback_chain.advance(str(e))
                return await self.execute_with_recovery(
                    next_agent, func, fallback_chain, *args, **kwargs
                )

            # Escalate
            if self.config.escalate_after_retries:
                await self._escalate(agent_id, str(e))

            self.recovery_history.append(RecoveryAttempt(
                timestamp=start_time,
                strategy="exhausted",
                success=False,
                error=str(e)
            ))

            raise

    def on_escalation(self, callback: Callable):
        """Register escalation callback"""
        self.escalation_callbacks.append(callback)

    async def _escalate(self, agent_id: str, reason: str):
        """Handle escalation"""
        escalation_data = {
            "agent_id": agent_id,
            "reason": reason,
            "escalate_to": self.config.escalate_to,
            "timestamp": datetime.utcnow().isoformat(),
            "recovery_history": [
                {"timestamp": r.timestamp.isoformat(), "strategy": r.strategy, "success": r.success}
                for r in self.recovery_history[-10:]
            ]
        }

        for callback in self.escalation_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(escalation_data)
                else:
                    callback(escalation_data)
            except Exception:
                pass

    def get_health_report(self) -> Dict:
        """Get health report for all circuit breakers"""
        return {
            "circuit_breakers": {
                name: cb.get_status()
                for name, cb in self.circuit_breakers.items()
            },
            "recent_recoveries": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "strategy": r.strategy,
                    "success": r.success,
                    "error": r.error
                }
                for r in self.recovery_history[-20:]
            ],
            "success_rate": self._calculate_success_rate()
        }

    def _calculate_success_rate(self) -> float:
        """Calculate recent success rate"""
        recent = self.recovery_history[-100:]
        if not recent:
            return 1.0
        successes = sum(1 for r in recent if r.success)
        return successes / len(recent)


# Decorators for easy use

def with_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
):
    """Decorator to add retry logic to a function"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            config = RecoveryConfig(
                max_retries=max_retries,
                retry_delay_seconds=delay,
                retry_backoff_multiplier=backoff
            )
            handler = RetryHandler(config)
            return await handler.execute_with_retry(func, *args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            config = RecoveryConfig(
                max_retries=max_retries,
                retry_delay_seconds=delay,
                retry_backoff_multiplier=backoff
            )
            handler = RetryHandler(config)
            return handler.execute_with_retry_sync(func, *args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def with_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0
):
    """Decorator to add circuit breaker to a function"""
    circuit = CircuitBreaker(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout
    )

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not circuit.can_execute():
                raise RuntimeError(f"Circuit breaker '{name}' is open")

            try:
                result = func(*args, **kwargs)
                circuit.record_success()
                return result
            except Exception as e:
                circuit.record_failure(str(e))
                raise

        return wrapper

    return decorator
