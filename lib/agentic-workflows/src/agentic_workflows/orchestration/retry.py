"""Retry logic with exponential backoff and jitter."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(f"Retry exhausted after {attempts} attempts. Last error: {last_exception}")


@dataclass
class RetryConfig:
    """Retry configuration."""

    # Maximum number of attempts (including first try)
    max_attempts: int = 3

    # Base delay between retries (seconds)
    base_delay: float = 1.0

    # Maximum delay between retries
    max_delay: float = 60.0

    # Exponential backoff multiplier
    backoff_multiplier: float = 2.0

    # Jitter factor (0.0 = no jitter, 1.0 = full jitter)
    jitter: float = 0.1

    # Exceptions to retry on (empty = retry all)
    retry_exceptions: tuple[type[Exception], ...] = ()

    # Exceptions to NOT retry on
    fatal_exceptions: tuple[type[Exception], ...] = ()


@dataclass
class RetryStats:
    """Retry statistics."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_retries: int = 0
    total_delay_seconds: float = 0.0

    @property
    def average_retries(self) -> float:
        """Average retries per call."""
        if self.total_calls == 0:
            return 0.0
        return self.total_retries / self.total_calls


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""

    attempt_number: int
    exception: Exception | None
    delay_seconds: float
    timestamp: float = field(default_factory=time.time)


class Retrier:
    """Retry handler with exponential backoff and jitter.

    Implements retry logic with:
    - Configurable max attempts
    - Exponential backoff with configurable multiplier
    - Jitter to prevent thundering herd
    - Exception filtering (retry_exceptions, fatal_exceptions)
    - Callbacks for retry events
    """

    def __init__(
        self,
        config: RetryConfig | None = None,
        on_retry: Callable[[RetryAttempt], None] | None = None,
    ):
        """Initialize retrier.

        Args:
            config: Retry configuration.
            on_retry: Callback for retry events.
        """
        self.config = config or RetryConfig()
        self.on_retry = on_retry
        self._stats = RetryStats()

    @property
    def stats(self) -> RetryStats:
        """Get retry statistics."""
        return self._stats

    def call(self, func: Callable[[], T]) -> T:
        """Execute function with retry logic.

        Args:
            func: Function to call.

        Returns:
            Function result.

        Raises:
            RetryExhausted: If all attempts fail.
            Exception: Fatal exception that should not be retried.
        """
        self._stats.total_calls += 1
        last_exception: Exception | None = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = func()
                self._stats.successful_calls += 1
                return result
            except Exception as e:
                last_exception = e

                # Check fatal exceptions
                if self.config.fatal_exceptions and isinstance(e, self.config.fatal_exceptions):
                    self._stats.failed_calls += 1
                    raise

                # Check if should retry
                if self.config.retry_exceptions and not isinstance(e, self.config.retry_exceptions):
                    self._stats.failed_calls += 1
                    raise

                # Last attempt?
                if attempt >= self.config.max_attempts:
                    self._stats.failed_calls += 1
                    raise RetryExhausted(attempt, e) from e

                # Calculate delay
                delay = self._calculate_delay(attempt)
                self._stats.total_retries += 1
                self._stats.total_delay_seconds += delay

                # Notify
                retry_attempt = RetryAttempt(
                    attempt_number=attempt,
                    exception=e,
                    delay_seconds=delay,
                )
                if self.on_retry:
                    self.on_retry(retry_attempt)

                # Wait
                time.sleep(delay)

        # Should not reach here
        self._stats.failed_calls += 1
        raise RetryExhausted(self.config.max_attempts, last_exception)

    async def call_async(self, func: Callable[[], Any]) -> Any:
        """Async version of call.

        Args:
            func: Function to call (can be sync or async).

        Returns:
            Function result.

        Raises:
            RetryExhausted: If all attempts fail.
            Exception: Fatal exception that should not be retried.
        """
        self._stats.total_calls += 1
        last_exception: Exception | None = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func()
                else:
                    result = func()
                self._stats.successful_calls += 1
                return result
            except Exception as e:
                last_exception = e

                if self.config.fatal_exceptions and isinstance(e, self.config.fatal_exceptions):
                    self._stats.failed_calls += 1
                    raise

                if self.config.retry_exceptions and not isinstance(e, self.config.retry_exceptions):
                    self._stats.failed_calls += 1
                    raise

                if attempt >= self.config.max_attempts:
                    self._stats.failed_calls += 1
                    raise RetryExhausted(attempt, e) from e

                delay = self._calculate_delay(attempt)
                self._stats.total_retries += 1
                self._stats.total_delay_seconds += delay

                retry_attempt = RetryAttempt(
                    attempt_number=attempt,
                    exception=e,
                    delay_seconds=delay,
                )
                if self.on_retry:
                    self.on_retry(retry_attempt)

                await asyncio.sleep(delay)

        self._stats.failed_calls += 1
        raise RetryExhausted(self.config.max_attempts, last_exception)

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for attempt with exponential backoff and jitter.

        Args:
            attempt: Current attempt number (1-based).

        Returns:
            Delay in seconds.
        """
        # Exponential backoff
        delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))

        # Cap at max delay
        delay = min(delay, self.config.max_delay)

        # Add jitter
        if self.config.jitter > 0:
            jitter_range = delay * self.config.jitter
            delay = delay + random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)  # Ensure non-negative

        return delay

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = RetryStats()


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_multiplier: float = 2.0,
    jitter: float = 0.1,
    retry_exceptions: tuple[type[Exception], ...] = (),
    fatal_exceptions: tuple[type[Exception], ...] = (),
):
    """Decorator for adding retry logic to functions.

    Args:
        max_attempts: Maximum number of attempts.
        base_delay: Base delay between retries.
        max_delay: Maximum delay.
        backoff_multiplier: Exponential backoff multiplier.
        jitter: Jitter factor.
        retry_exceptions: Exceptions to retry on.
        fatal_exceptions: Exceptions to not retry on.

    Returns:
        Decorator function.
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_multiplier=backoff_multiplier,
        jitter=jitter,
        retry_exceptions=retry_exceptions,
        fatal_exceptions=fatal_exceptions,
    )
    retrier = Retrier(config)

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                return await retrier.call_async(lambda: func(*args, **kwargs))

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                return retrier.call(lambda: func(*args, **kwargs))

            return sync_wrapper

    return decorator


class RetryWithFallback:
    """Retry with fallback chain.

    Tries primary function, then falls back to alternatives.
    """

    def __init__(
        self,
        primary: Callable[[], T],
        fallbacks: list[Callable[[], T]],
        config: RetryConfig | None = None,
    ):
        """Initialize retry with fallback.

        Args:
            primary: Primary function to try.
            fallbacks: List of fallback functions.
            config: Retry configuration (applied to each function).
        """
        self.primary = primary
        self.fallbacks = fallbacks
        self.config = config or RetryConfig()
        self._retrier = Retrier(self.config)

    def call(self) -> T:
        """Execute with retry and fallback.

        Returns:
            Result from successful function.

        Raises:
            RetryExhausted: If all functions (including fallbacks) fail.
        """
        all_functions = [self.primary] + self.fallbacks
        last_exception: Exception | None = None

        for i, func in enumerate(all_functions):
            try:
                return self._retrier.call(func)
            except RetryExhausted as e:
                last_exception = e.last_exception
                # Continue to next fallback
                continue

        raise RetryExhausted(
            len(all_functions) * self.config.max_attempts,
            last_exception,
        )

    async def call_async(self) -> T:
        """Async version of call."""
        all_functions = [self.primary] + self.fallbacks
        last_exception: Exception | None = None

        for func in all_functions:
            try:
                return await self._retrier.call_async(func)
            except RetryExhausted as e:
                last_exception = e.last_exception
                continue

        raise RetryExhausted(
            len(all_functions) * self.config.max_attempts,
            last_exception,
        )
