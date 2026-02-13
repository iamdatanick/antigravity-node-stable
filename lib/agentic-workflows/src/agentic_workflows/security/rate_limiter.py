"""Token bucket rate limiter for agent operations."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    requests_per_second: float = 10.0
    burst_size: int = 20
    max_wait: float = 30.0  # Max seconds to wait for a token


@dataclass
class RateLimitStats:
    """Rate limiter statistics."""

    total_requests: int = 0
    allowed_requests: int = 0
    rejected_requests: int = 0
    total_wait_time: float = 0.0

    @property
    def rejection_rate(self) -> float:
        """Calculate rejection rate."""
        if self.total_requests == 0:
            return 0.0
        return self.rejected_requests / self.total_requests


class RateLimiter:
    """Token bucket rate limiter.

    Implements the token bucket algorithm with:
    - Configurable refill rate (tokens per second)
    - Burst capacity for handling spikes
    - Optional waiting for token availability
    - Per-key rate limiting support
    """

    def __init__(
        self,
        config: RateLimitConfig | None = None,
        on_limit: Callable[[str], None] | None = None,
    ):
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration.
            on_limit: Callback when rate limit is hit.
        """
        self.config = config or RateLimitConfig()
        self.on_limit = on_limit

        self._tokens: float = self.config.burst_size
        self._last_update: float = time.monotonic()
        self._lock = Lock()
        self._stats = RateLimitStats()

        # Per-key buckets for multi-tenant scenarios
        self._key_buckets: dict[str, tuple[float, float]] = {}
        self._key_lock = Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now

        # Add tokens based on elapsed time
        new_tokens = elapsed * self.config.requests_per_second
        self._tokens = min(self.config.burst_size, self._tokens + new_tokens)

    def _refill_key(self, key: str) -> tuple[float, float]:
        """Refill tokens for a specific key."""
        now = time.monotonic()

        if key in self._key_buckets:
            tokens, last_update = self._key_buckets[key]
            elapsed = now - last_update
            new_tokens = elapsed * self.config.requests_per_second
            tokens = min(self.config.burst_size, tokens + new_tokens)
        else:
            tokens = self.config.burst_size

        self._key_buckets[key] = (tokens, now)
        return tokens, now

    def try_acquire(self, tokens: int = 1, key: str | None = None) -> bool:
        """Try to acquire tokens without waiting.

        Args:
            tokens: Number of tokens to acquire.
            key: Optional key for per-key limiting.

        Returns:
            True if tokens were acquired, False otherwise.
        """
        self._stats.total_requests += 1

        if key:
            return self._try_acquire_key(tokens, key)

        with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                self._stats.allowed_requests += 1
                return True

            self._stats.rejected_requests += 1
            if self.on_limit:
                self.on_limit(f"Rate limit exceeded: {tokens} tokens requested, {self._tokens:.1f} available")
            return False

    def _try_acquire_key(self, tokens: int, key: str) -> bool:
        """Try to acquire tokens for a specific key."""
        with self._key_lock:
            current_tokens, _ = self._refill_key(key)

            if current_tokens >= tokens:
                self._key_buckets[key] = (current_tokens - tokens, time.monotonic())
                self._stats.allowed_requests += 1
                return True

            self._stats.rejected_requests += 1
            if self.on_limit:
                self.on_limit(f"Rate limit exceeded for key '{key}'")
            return False

    def acquire(self, tokens: int = 1, key: str | None = None) -> float:
        """Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire.
            key: Optional key for per-key limiting.

        Returns:
            Time waited in seconds.

        Raises:
            RateLimitExceeded: If max wait time exceeded.
        """
        start_time = time.monotonic()

        while True:
            if self.try_acquire(tokens, key):
                wait_time = time.monotonic() - start_time
                self._stats.total_wait_time += wait_time
                return wait_time

            # Calculate wait time for next token
            if key:
                with self._key_lock:
                    current_tokens, _ = self._refill_key(key)
            else:
                with self._lock:
                    self._refill()
                    current_tokens = self._tokens

            tokens_needed = tokens - current_tokens
            wait_time = tokens_needed / self.config.requests_per_second

            # Check if we've exceeded max wait
            elapsed = time.monotonic() - start_time
            if elapsed + wait_time > self.config.max_wait:
                raise RateLimitExceeded(
                    f"Rate limit exceeded, would need to wait {wait_time:.1f}s",
                    retry_after=wait_time,
                )

            # Wait and retry
            time.sleep(min(wait_time, 0.1))  # Small increments to stay responsive

    async def acquire_async(self, tokens: int = 1, key: str | None = None) -> float:
        """Async version of acquire.

        Args:
            tokens: Number of tokens to acquire.
            key: Optional key for per-key limiting.

        Returns:
            Time waited in seconds.

        Raises:
            RateLimitExceeded: If max wait time exceeded.
        """
        start_time = time.monotonic()

        while True:
            if self.try_acquire(tokens, key):
                wait_time = time.monotonic() - start_time
                self._stats.total_wait_time += wait_time
                return wait_time

            # Calculate wait time
            if key:
                with self._key_lock:
                    current_tokens, _ = self._refill_key(key)
            else:
                with self._lock:
                    self._refill()
                    current_tokens = self._tokens

            tokens_needed = tokens - current_tokens
            wait_time = tokens_needed / self.config.requests_per_second

            # Check max wait
            elapsed = time.monotonic() - start_time
            if elapsed + wait_time > self.config.max_wait:
                raise RateLimitExceeded(
                    f"Rate limit exceeded, would need to wait {wait_time:.1f}s",
                    retry_after=wait_time,
                )

            await asyncio.sleep(min(wait_time, 0.1))

    def get_tokens_available(self, key: str | None = None) -> float:
        """Get current token count.

        Args:
            key: Optional key for per-key checking.

        Returns:
            Number of tokens currently available.
        """
        if key:
            with self._key_lock:
                tokens, _ = self._refill_key(key)
                return tokens

        with self._lock:
            self._refill()
            return self._tokens

    def reset(self, key: str | None = None) -> None:
        """Reset to full capacity.

        Args:
            key: Optional key to reset (None = global).
        """
        if key:
            with self._key_lock:
                self._key_buckets[key] = (self.config.burst_size, time.monotonic())
        else:
            with self._lock:
                self._tokens = self.config.burst_size
                self._last_update = time.monotonic()

    def get_stats(self) -> RateLimitStats:
        """Get rate limiter statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = RateLimitStats()

    def __enter__(self) -> RateLimiter:
        """Context manager entry - acquire one token."""
        self.acquire(1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        pass


class MultiRateLimiter:
    """Combines multiple rate limiters with different granularities.

    Example:
        limiter = MultiRateLimiter()
        limiter.add_limit("second", RateLimitConfig(requests_per_second=10))
        limiter.add_limit("minute", RateLimitConfig(requests_per_second=100/60, burst_size=100))
        limiter.add_limit("hour", RateLimitConfig(requests_per_second=1000/3600, burst_size=1000))
    """

    def __init__(self):
        """Initialize multi-rate limiter."""
        self._limiters: dict[str, RateLimiter] = {}

    def add_limit(self, name: str, config: RateLimitConfig) -> None:
        """Add a rate limit.

        Args:
            name: Name for this limit.
            config: Rate limit configuration.
        """
        self._limiters[name] = RateLimiter(config)

    def try_acquire(self, tokens: int = 1, key: str | None = None) -> tuple[bool, str | None]:
        """Try to acquire from all limiters.

        Args:
            tokens: Number of tokens.
            key: Optional key for per-key limiting.

        Returns:
            Tuple of (success, failed_limiter_name).
        """
        for name, limiter in self._limiters.items():
            if not limiter.try_acquire(tokens, key):
                return False, name
        return True, None

    def acquire(self, tokens: int = 1, key: str | None = None) -> dict[str, float]:
        """Acquire from all limiters, waiting if needed.

        Args:
            tokens: Number of tokens.
            key: Optional key for per-key limiting.

        Returns:
            Dict of limiter name to wait time.
        """
        wait_times = {}
        for name, limiter in self._limiters.items():
            wait_times[name] = limiter.acquire(tokens, key)
        return wait_times

    async def acquire_async(self, tokens: int = 1, key: str | None = None) -> dict[str, float]:
        """Async acquire from all limiters.

        Args:
            tokens: Number of tokens.
            key: Optional key for per-key limiting.

        Returns:
            Dict of limiter name to wait time.
        """
        wait_times = {}
        for name, limiter in self._limiters.items():
            wait_times[name] = await limiter.acquire_async(tokens, key)
        return wait_times

    def get_stats(self) -> dict[str, RateLimitStats]:
        """Get stats for all limiters."""
        return {name: limiter.get_stats() for name, limiter in self._limiters.items()}
