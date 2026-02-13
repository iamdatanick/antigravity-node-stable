"""Distributed rate limiting with Redis backend.

Provides multiple rate limiting algorithms for protecting against abuse:
- Token bucket: Allows bursts while maintaining average rate
- Sliding window: Precise rate counting
- Fixed window: Simple and efficient
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..storage.redis import RedisStorage, RedisConfig


class RateLimitAlgorithm(Enum):
    """Available rate limiting algorithms."""

    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiter."""

    # Maximum requests allowed
    max_requests: int = 100

    # Time window in seconds
    window_seconds: int = 60

    # For token bucket: tokens added per second
    refill_rate: float | None = None

    # For token bucket: maximum bucket size
    bucket_size: int | None = None

    # Key prefix for Redis
    key_prefix: str = "ratelimit:"

    # Algorithm to use
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW


@dataclass
class RateLimitResult:
    """Result of rate limit check."""

    allowed: bool
    remaining: int
    reset_at: float
    retry_after: float | None = None

    @property
    def headers(self) -> dict[str, str]:
        """Get rate limit headers for HTTP responses."""
        headers = {
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_at)),
        }
        if self.retry_after is not None:
            headers["Retry-After"] = str(int(self.retry_after))
        return headers


class RateLimiter(ABC):
    """Abstract base for rate limiters."""

    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration.
        """
        self.config = config

    @abstractmethod
    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check if request is allowed.

        Args:
            key: Identifier (e.g., user ID, IP address).
            cost: Cost of this request (default 1).

        Returns:
            RateLimitResult indicating if allowed.
        """
        pass

    @abstractmethod
    async def reset(self, key: str) -> None:
        """Reset rate limit for a key.

        Args:
            key: Identifier to reset.
        """
        pass

    @abstractmethod
    async def get_status(self, key: str) -> RateLimitResult:
        """Get current rate limit status without incrementing.

        Args:
            key: Identifier to check.

        Returns:
            Current rate limit status.
        """
        pass


class RedisRateLimiter(RateLimiter):
    """Redis-backed distributed rate limiter."""

    def __init__(
        self,
        config: RateLimitConfig,
        redis_config: RedisConfig | None = None,
        redis_storage: RedisStorage | None = None,
    ):
        """Initialize Redis rate limiter.

        Args:
            config: Rate limit configuration.
            redis_config: Redis connection config (if no storage provided).
            redis_storage: Existing Redis storage instance.
        """
        super().__init__(config)

        if redis_storage:
            self._storage = redis_storage
            self._owns_storage = False
        else:
            self._storage = RedisStorage(redis_config)
            self._owns_storage = True

        self._connected = False

    async def _ensure_connected(self) -> None:
        """Ensure Redis is connected."""
        if not self._connected:
            await self._storage.connect()
            self._connected = True

    def _make_key(self, key: str) -> str:
        """Create full Redis key."""
        return f"{self.config.key_prefix}{key}"

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check if request is allowed using configured algorithm."""
        await self._ensure_connected()

        if self.config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return await self._check_token_bucket(key, cost)
        elif self.config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return await self._check_sliding_window(key, cost)
        else:
            return await self._check_fixed_window(key, cost)

    async def _check_sliding_window(self, key: str, cost: int) -> RateLimitResult:
        """Sliding window rate limiting with Redis sorted sets."""
        full_key = self._make_key(key)
        now = time.time()
        window_start = now - self.config.window_seconds

        client = await self._storage._ensure_connected()

        # Lua script for atomic sliding window check
        script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window_start = tonumber(ARGV[2])
        local max_requests = tonumber(ARGV[3])
        local cost = tonumber(ARGV[4])
        local window_seconds = tonumber(ARGV[5])

        -- Remove old entries
        redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)

        -- Count current requests
        local current = redis.call('ZCARD', key)

        -- Check if allowed
        if current + cost <= max_requests then
            -- Add new entries (one per cost unit)
            for i = 1, cost do
                redis.call('ZADD', key, now, now .. '-' .. i .. '-' .. math.random())
            end
            -- Set expiration
            redis.call('EXPIRE', key, window_seconds + 1)
            return {1, max_requests - current - cost, now + window_seconds}
        else
            -- Get oldest entry for retry-after calculation
            local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
            local retry_after = 0
            if #oldest >= 2 then
                retry_after = oldest[2] + window_seconds - now
            end
            return {0, 0, now + window_seconds, retry_after}
        end
        """

        result = await client.eval(
            script, 1, full_key,
            now, window_start, self.config.max_requests, cost, self.config.window_seconds
        )

        allowed = result[0] == 1
        remaining = int(result[1])
        reset_at = float(result[2])
        retry_after = float(result[3]) if len(result) > 3 and result[3] > 0 else None

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_at=reset_at,
            retry_after=retry_after,
        )

    async def _check_fixed_window(self, key: str, cost: int) -> RateLimitResult:
        """Fixed window rate limiting."""
        now = time.time()
        window_key = f"{key}:{int(now / self.config.window_seconds)}"
        full_key = self._make_key(window_key)

        # Increment counter with TTL
        count = await self._storage.incr_with_ttl(
            window_key, cost, self.config.window_seconds
        )

        window_end = (int(now / self.config.window_seconds) + 1) * self.config.window_seconds
        remaining = max(0, self.config.max_requests - count)
        allowed = count <= self.config.max_requests

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_at=window_end,
            retry_after=window_end - now if not allowed else None,
        )

    async def _check_token_bucket(self, key: str, cost: int) -> RateLimitResult:
        """Token bucket rate limiting."""
        full_key = self._make_key(key)
        now = time.time()

        bucket_size = self.config.bucket_size or self.config.max_requests
        refill_rate = self.config.refill_rate or (self.config.max_requests / self.config.window_seconds)

        client = await self._storage._ensure_connected()

        # Lua script for atomic token bucket
        script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local bucket_size = tonumber(ARGV[2])
        local refill_rate = tonumber(ARGV[3])
        local cost = tonumber(ARGV[4])

        -- Get current state
        local data = redis.call('HMGET', key, 'tokens', 'last_update')
        local tokens = tonumber(data[1]) or bucket_size
        local last_update = tonumber(data[2]) or now

        -- Calculate token refill
        local elapsed = now - last_update
        tokens = math.min(bucket_size, tokens + elapsed * refill_rate)

        -- Check if request is allowed
        if tokens >= cost then
            tokens = tokens - cost
            redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
            redis.call('EXPIRE', key, 86400)  -- 24 hour expiry
            return {1, math.floor(tokens), 0}
        else
            -- Calculate wait time for enough tokens
            local needed = cost - tokens
            local wait_time = needed / refill_rate
            redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
            redis.call('EXPIRE', key, 86400)
            return {0, math.floor(tokens), wait_time}
        end
        """

        result = await client.eval(
            script, 1, full_key,
            now, bucket_size, refill_rate, cost
        )

        allowed = result[0] == 1
        remaining = int(result[1])
        wait_time = float(result[2])

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_at=now + (self.config.window_seconds if not allowed else 0),
            retry_after=wait_time if wait_time > 0 else None,
        )

    async def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        await self._ensure_connected()
        full_key = self._make_key(key)
        await self._storage.delete(full_key)

    async def get_status(self, key: str) -> RateLimitResult:
        """Get current rate limit status without incrementing."""
        await self._ensure_connected()

        if self.config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return await self._get_sliding_window_status(key)
        elif self.config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return await self._get_fixed_window_status(key)
        else:
            return await self._get_token_bucket_status(key)

    async def _get_sliding_window_status(self, key: str) -> RateLimitResult:
        """Get sliding window status."""
        full_key = self._make_key(key)
        now = time.time()
        window_start = now - self.config.window_seconds

        client = await self._storage._ensure_connected()

        # Remove old and count
        await client.zremrangebyscore(full_key, '-inf', window_start)
        count = await client.zcard(full_key)

        remaining = max(0, self.config.max_requests - count)

        return RateLimitResult(
            allowed=remaining > 0,
            remaining=remaining,
            reset_at=now + self.config.window_seconds,
        )

    async def _get_fixed_window_status(self, key: str) -> RateLimitResult:
        """Get fixed window status."""
        now = time.time()
        window_key = f"{key}:{int(now / self.config.window_seconds)}"
        full_key = self._make_key(window_key)

        client = await self._storage._ensure_connected()
        count = await client.get(full_key)
        count = int(count) if count else 0

        window_end = (int(now / self.config.window_seconds) + 1) * self.config.window_seconds
        remaining = max(0, self.config.max_requests - count)

        return RateLimitResult(
            allowed=remaining > 0,
            remaining=remaining,
            reset_at=window_end,
        )

    async def _get_token_bucket_status(self, key: str) -> RateLimitResult:
        """Get token bucket status."""
        full_key = self._make_key(key)
        now = time.time()

        bucket_size = self.config.bucket_size or self.config.max_requests
        refill_rate = self.config.refill_rate or (self.config.max_requests / self.config.window_seconds)

        client = await self._storage._ensure_connected()
        data = await client.hmget(full_key, 'tokens', 'last_update')

        tokens = float(data[0]) if data[0] else bucket_size
        last_update = float(data[1]) if data[1] else now

        elapsed = now - last_update
        tokens = min(bucket_size, tokens + elapsed * refill_rate)

        return RateLimitResult(
            allowed=tokens >= 1,
            remaining=int(tokens),
            reset_at=now + self.config.window_seconds,
        )

    async def close(self) -> None:
        """Close Redis connection if owned."""
        if self._owns_storage and self._connected:
            await self._storage.close()
            self._connected = False


class MemoryRateLimiter(RateLimiter):
    """In-memory rate limiter for testing and single-process deployments."""

    def __init__(self, config: RateLimitConfig):
        """Initialize memory rate limiter."""
        super().__init__(config)
        self._buckets: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check if request is allowed."""
        async with self._lock:
            now = time.time()

            if self.config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                return self._check_token_bucket(key, cost, now)
            elif self.config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                return self._check_sliding_window(key, cost, now)
            else:
                return self._check_fixed_window(key, cost, now)

    def _check_sliding_window(self, key: str, cost: int, now: float) -> RateLimitResult:
        """Sliding window check."""
        window_start = now - self.config.window_seconds

        if key not in self._buckets:
            self._buckets[key] = {"requests": []}

        bucket = self._buckets[key]

        # Remove old requests
        bucket["requests"] = [t for t in bucket["requests"] if t > window_start]

        current = len(bucket["requests"])

        if current + cost <= self.config.max_requests:
            bucket["requests"].extend([now] * cost)
            return RateLimitResult(
                allowed=True,
                remaining=self.config.max_requests - current - cost,
                reset_at=now + self.config.window_seconds,
            )
        else:
            retry_after = None
            if bucket["requests"]:
                oldest = min(bucket["requests"])
                retry_after = oldest + self.config.window_seconds - now
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=now + self.config.window_seconds,
                retry_after=retry_after,
            )

    def _check_fixed_window(self, key: str, cost: int, now: float) -> RateLimitResult:
        """Fixed window check."""
        window_id = int(now / self.config.window_seconds)
        window_key = f"{key}:{window_id}"

        if window_key not in self._buckets:
            self._buckets[window_key] = {"count": 0}

        bucket = self._buckets[window_key]
        new_count = bucket["count"] + cost

        window_end = (window_id + 1) * self.config.window_seconds

        if new_count <= self.config.max_requests:
            bucket["count"] = new_count
            return RateLimitResult(
                allowed=True,
                remaining=self.config.max_requests - new_count,
                reset_at=window_end,
            )
        else:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=window_end,
                retry_after=window_end - now,
            )

    def _check_token_bucket(self, key: str, cost: int, now: float) -> RateLimitResult:
        """Token bucket check."""
        bucket_size = self.config.bucket_size or self.config.max_requests
        refill_rate = self.config.refill_rate or (self.config.max_requests / self.config.window_seconds)

        if key not in self._buckets:
            self._buckets[key] = {"tokens": bucket_size, "last_update": now}

        bucket = self._buckets[key]

        # Refill tokens
        elapsed = now - bucket["last_update"]
        bucket["tokens"] = min(bucket_size, bucket["tokens"] + elapsed * refill_rate)
        bucket["last_update"] = now

        if bucket["tokens"] >= cost:
            bucket["tokens"] -= cost
            return RateLimitResult(
                allowed=True,
                remaining=int(bucket["tokens"]),
                reset_at=now + self.config.window_seconds,
            )
        else:
            needed = cost - bucket["tokens"]
            wait_time = needed / refill_rate
            return RateLimitResult(
                allowed=False,
                remaining=int(bucket["tokens"]),
                reset_at=now + wait_time,
                retry_after=wait_time,
            )

    async def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        async with self._lock:
            # Remove all keys starting with this key
            keys_to_remove = [k for k in self._buckets if k == key or k.startswith(f"{key}:")]
            for k in keys_to_remove:
                del self._buckets[k]

    async def get_status(self, key: str) -> RateLimitResult:
        """Get current status without incrementing."""
        async with self._lock:
            now = time.time()

            if self.config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                window_start = now - self.config.window_seconds
                bucket = self._buckets.get(key, {"requests": []})
                current = len([t for t in bucket.get("requests", []) if t > window_start])
                remaining = max(0, self.config.max_requests - current)
            elif self.config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                window_id = int(now / self.config.window_seconds)
                window_key = f"{key}:{window_id}"
                bucket = self._buckets.get(window_key, {"count": 0})
                remaining = max(0, self.config.max_requests - bucket["count"])
            else:
                bucket_size = self.config.bucket_size or self.config.max_requests
                bucket = self._buckets.get(key, {"tokens": bucket_size, "last_update": now})
                remaining = int(bucket["tokens"])

            return RateLimitResult(
                allowed=remaining > 0,
                remaining=remaining,
                reset_at=now + self.config.window_seconds,
            )


def create_rate_limiter(
    config: RateLimitConfig,
    redis_config: RedisConfig | None = None,
    use_redis: bool = True,
) -> RateLimiter:
    """Factory function to create appropriate rate limiter.

    Args:
        config: Rate limit configuration.
        redis_config: Redis configuration (optional).
        use_redis: Whether to use Redis (falls back to memory if unavailable).

    Returns:
        Configured rate limiter instance.
    """
    if use_redis:
        try:
            return RedisRateLimiter(config, redis_config)
        except ImportError:
            # Redis not available, fall back to memory
            pass

    return MemoryRateLimiter(config)
