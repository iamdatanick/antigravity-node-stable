"""Circuit breaker pattern for failure isolation.

Enhanced with exponential backoff and jitter for improved retry behavior.
"""

from __future__ import annotations

import asyncio
import inspect
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

T = TypeVar("T")


# =============================================================================
# Jitter and Backoff Utilities
# =============================================================================


def calculate_jitter(
    base_delay: float,
    jitter_factor: float = 0.5,
    jitter_type: str = "full",
) -> float:
    """Calculate delay with jitter.

    Jitter types:
    - "full": Delay between 0 and base_delay * (1 + jitter_factor)
    - "equal": Delay between base_delay/2 and base_delay * (1 + jitter_factor/2)
    - "decorrelated": Delay between base_delay and previous * 3 (requires state)

    Args:
        base_delay: Base delay in seconds.
        jitter_factor: Jitter range factor (0-1).
        jitter_type: Type of jitter algorithm.

    Returns:
        Delay with jitter applied.
    """
    if jitter_type == "full":
        # Full jitter: random between 0 and base_delay * (1 + jitter_factor)
        return random.uniform(0, base_delay * (1 + jitter_factor))

    elif jitter_type == "equal":
        # Equal jitter: half fixed, half random
        fixed_part = base_delay * (1 - jitter_factor / 2)
        random_part = random.uniform(0, base_delay * jitter_factor)
        return fixed_part + random_part

    else:
        # Default: just add/subtract small random amount
        jitter_amount = base_delay * jitter_factor
        return base_delay + random.uniform(-jitter_amount, jitter_amount)


def exponential_backoff_with_jitter(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter_factor: float = 0.5,
) -> float:
    """Calculate exponential backoff delay with jitter.

    Formula: min(max_delay, base_delay * exponential_base^attempt) + jitter

    Args:
        attempt: Current attempt number (0-indexed).
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay cap.
        exponential_base: Base for exponential growth.
        jitter_factor: Random jitter factor (0-1).

    Returns:
        Delay in seconds with jitter.

    Example:
        >>> delays = [exponential_backoff_with_jitter(i) for i in range(5)]
        >>> # Approximately: [1.0, 2.0, 4.0, 8.0, 16.0] + jitter
    """
    # Calculate exponential delay
    exp_delay = base_delay * (exponential_base ** attempt)

    # Cap at max_delay
    capped_delay = min(exp_delay, max_delay)

    # Apply jitter
    return calculate_jitter(capped_delay, jitter_factor, "full")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerOpen(Exception):
    """Raised when circuit is open and call is rejected."""

    def __init__(self, name: str, retry_after: float):
        self.name = name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker '{name}' is open. Retry after {retry_after:.1f}s"
        )


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    # Failure threshold to open circuit
    failure_threshold: int = 5

    # Success threshold to close circuit from half-open
    success_threshold: int = 2

    # Time to wait before trying half-open
    timeout_seconds: float = 30.0

    # Optional: time window for failure counting (None = unlimited)
    failure_window_seconds: float | None = 60.0

    # Optional: exclude certain exceptions from counting as failures
    excluded_exceptions: tuple[type[Exception], ...] = ()


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        executed = self.total_calls - self.rejected_calls
        if executed == 0:
            return 1.0
        return self.successful_calls / executed


class CircuitBreaker:
    """Circuit breaker for failure isolation.

    States:
    - CLOSED: Normal operation, tracking failures
    - OPEN: Rejecting all calls, waiting for timeout
    - HALF_OPEN: Allowing test calls to check recovery
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        on_state_change: Callable[[str, CircuitState, CircuitState], None] | None = None,
    ):
        """Initialize circuit breaker.

        Args:
            name: Name for this circuit breaker.
            config: Configuration options.
            on_state_change: Callback for state changes (name, old, new).
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._lock = threading.Lock()
        self._stats = CircuitBreakerStats()

        # Failure tracking
        self._failure_timestamps: list[float] = []
        self._consecutive_failures = 0
        self._consecutive_successes = 0

        # Open state tracking
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        """Get current state."""
        with self._lock:
            return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get statistics."""
        return self._stats

    def call(self, func: Callable[[], T], *args, **kwargs) -> T:
        """Execute function through circuit breaker.

        Args:
            func: Function to call.
            *args: Positional arguments (not used, func should be callable).
            **kwargs: Keyword arguments (not used).

        Returns:
            Function result.

        Raises:
            CircuitBreakerOpen: If circuit is open.
            Exception: Any exception from the function.
        """
        self._stats.total_calls += 1

        # Check state
        with self._lock:
            self._maybe_transition_to_half_open()

            if self._state == CircuitState.OPEN:
                self._stats.rejected_calls += 1
                retry_after = self._get_retry_after()
                raise CircuitBreakerOpen(self.name, retry_after)

        # Execute call
        try:
            result = func()
            self._record_success()
            return result
        except Exception as e:
            # Check if exception should be excluded
            if isinstance(e, self.config.excluded_exceptions):
                self._record_success()
                raise

            self._record_failure()
            raise

    async def call_async(self, func: Callable[[], Any]) -> Any:
        """Async version of call.

        Args:
            func: Async function to call.

        Returns:
            Function result.

        Raises:
            CircuitBreakerOpen: If circuit is open.
            Exception: Any exception from the function.
        """
        self._stats.total_calls += 1

        # Check state
        with self._lock:
            self._maybe_transition_to_half_open()

            if self._state == CircuitState.OPEN:
                self._stats.rejected_calls += 1
                retry_after = self._get_retry_after()
                raise CircuitBreakerOpen(self.name, retry_after)

        # Execute call
        try:
            if inspect.iscoroutinefunction(func):
                result = await func()
            else:
                result = func()
            self._record_success()
            return result
        except Exception as e:
            if isinstance(e, self.config.excluded_exceptions):
                self._record_success()
                raise

            self._record_failure()
            raise

    def _record_success(self) -> None:
        """Record a successful call."""
        self._stats.successful_calls += 1

        with self._lock:
            self._consecutive_failures = 0
            self._consecutive_successes += 1

            if self._state == CircuitState.HALF_OPEN:
                if self._consecutive_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def _record_failure(self) -> None:
        """Record a failed call."""
        self._stats.failed_calls += 1
        now = time.time()

        with self._lock:
            self._consecutive_successes = 0
            self._consecutive_failures += 1
            self._failure_timestamps.append(now)

            # Clean old failures outside window
            if self.config.failure_window_seconds:
                cutoff = now - self.config.failure_window_seconds
                self._failure_timestamps = [
                    t for t in self._failure_timestamps if t > cutoff
                ]

            # Check if should open
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                failures_in_window = len(self._failure_timestamps)
                if failures_in_window >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _maybe_transition_to_half_open(self) -> None:
        """Check if should transition from open to half-open."""
        if self._state != CircuitState.OPEN:
            return

        if self._opened_at is None:
            return

        elapsed = time.time() - self._opened_at
        if elapsed >= self.config.timeout_seconds:
            self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._stats.state_changes += 1

        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
        elif new_state == CircuitState.HALF_OPEN:
            self._consecutive_successes = 0
        elif new_state == CircuitState.CLOSED:
            self._opened_at = None
            self._failure_timestamps = []
            self._consecutive_failures = 0

        if self.on_state_change:
            # Call outside lock
            threading.Thread(
                target=self.on_state_change,
                args=(self.name, old_state, new_state),
                daemon=True,
            ).start()

    def _get_retry_after(self) -> float:
        """Get seconds until retry is allowed."""
        if self._opened_at is None:
            return 0.0

        elapsed = time.time() - self._opened_at
        return max(0, self.config.timeout_seconds - elapsed)

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._opened_at = None
            self._failure_timestamps = []
            self._consecutive_failures = 0
            self._consecutive_successes = 0

    def force_open(self) -> None:
        """Force circuit to open state."""
        with self._lock:
            self._transition_to(CircuitState.OPEN)

    def force_close(self) -> None:
        """Force circuit to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)

    def get_status(self) -> dict[str, Any]:
        """Get detailed status."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "consecutive_failures": self._consecutive_failures,
                "consecutive_successes": self._consecutive_successes,
                "failures_in_window": len(self._failure_timestamps),
                "retry_after": self._get_retry_after() if self._state == CircuitState.OPEN else 0,
                "stats": {
                    "total_calls": self._stats.total_calls,
                    "successful_calls": self._stats.successful_calls,
                    "failed_calls": self._stats.failed_calls,
                    "rejected_calls": self._stats.rejected_calls,
                    "success_rate": self._stats.success_rate,
                    "state_changes": self._stats.state_changes,
                },
            }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        """Initialize registry."""
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker.

        Args:
            name: Circuit breaker name.
            config: Configuration for new breaker.

        Returns:
            Circuit breaker instance.
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        """Get circuit breaker by name."""
        return self._breakers.get(name)

    def remove(self, name: str) -> None:
        """Remove circuit breaker."""
        with self._lock:
            self._breakers.pop(name, None)

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {name: breaker.get_status() for name, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


# =============================================================================
# Retry with Backoff and Jitter
# =============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry with backoff."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter_factor: float = 0.5
    jitter_type: str = "full"
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)


class RetryWithBackoff:
    """Retry wrapper with exponential backoff and jitter.

    Implements the recommended retry pattern with full jitter to
    prevent thundering herd problems.

    Example:
        >>> retry = RetryWithBackoff(RetryConfig(max_attempts=5))
        >>> result = await retry.execute_async(flaky_api_call)

        # With circuit breaker
        >>> retry = RetryWithBackoff(config, circuit_breaker=breaker)
        >>> result = await retry.execute_async(api_call)
    """

    def __init__(
        self,
        config: RetryConfig | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        on_retry: Callable[[int, Exception, float], None] | None = None,
    ):
        """Initialize retry wrapper.

        Args:
            config: Retry configuration.
            circuit_breaker: Optional circuit breaker to use.
            on_retry: Callback on retry (attempt, exception, delay).
        """
        self.config = config or RetryConfig()
        self.circuit_breaker = circuit_breaker
        self.on_retry = on_retry

    def execute(self, func: Callable[[], T]) -> T:
        """Execute function with retry.

        Args:
            func: Function to execute.

        Returns:
            Function result.

        Raises:
            Last exception if all retries exhausted.
        """
        last_exception: Exception | None = None

        for attempt in range(self.config.max_attempts):
            try:
                if self.circuit_breaker:
                    return self.circuit_breaker.call(func)
                return func()

            except CircuitBreakerOpen:
                # Don't retry if circuit is open
                raise

            except self.config.retryable_exceptions as e:
                last_exception = e

                if attempt < self.config.max_attempts - 1:
                    delay = exponential_backoff_with_jitter(
                        attempt=attempt,
                        base_delay=self.config.base_delay,
                        max_delay=self.config.max_delay,
                        exponential_base=self.config.exponential_base,
                        jitter_factor=self.config.jitter_factor,
                    )

                    if self.on_retry:
                        self.on_retry(attempt + 1, e, delay)

                    time.sleep(delay)

        if last_exception:
            raise last_exception
        raise RuntimeError("Retry exhausted without exception")

    async def execute_async(self, func: Callable[[], Any]) -> Any:
        """Execute async function with retry.

        Args:
            func: Async function to execute.

        Returns:
            Function result.

        Raises:
            Last exception if all retries exhausted.
        """
        last_exception: Exception | None = None

        for attempt in range(self.config.max_attempts):
            try:
                if self.circuit_breaker:
                    return await self.circuit_breaker.call_async(func)

                if inspect.iscoroutinefunction(func):
                    return await func()
                return func()

            except CircuitBreakerOpen:
                # Don't retry if circuit is open
                raise

            except self.config.retryable_exceptions as e:
                last_exception = e

                if attempt < self.config.max_attempts - 1:
                    delay = exponential_backoff_with_jitter(
                        attempt=attempt,
                        base_delay=self.config.base_delay,
                        max_delay=self.config.max_delay,
                        exponential_base=self.config.exponential_base,
                        jitter_factor=self.config.jitter_factor,
                    )

                    if self.on_retry:
                        self.on_retry(attempt + 1, e, delay)

                    await asyncio.sleep(delay)

        if last_exception:
            raise last_exception
        raise RuntimeError("Retry exhausted without exception")

    def get_delay_for_attempt(self, attempt: int) -> float:
        """Get the delay for a specific attempt number.

        Args:
            attempt: Attempt number (0-indexed).

        Returns:
            Delay in seconds with jitter.
        """
        return exponential_backoff_with_jitter(
            attempt=attempt,
            base_delay=self.config.base_delay,
            max_delay=self.config.max_delay,
            exponential_base=self.config.exponential_base,
            jitter_factor=self.config.jitter_factor,
        )
