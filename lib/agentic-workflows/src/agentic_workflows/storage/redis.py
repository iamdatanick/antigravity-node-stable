"""Redis storage backend for caching and rate limiting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .base import StorageBackend, StorageConfig

# Try to import redis
try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class RedisConfig(StorageConfig):
    """Redis storage configuration."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    ssl: bool = False
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    decode_responses: bool = True

    @property
    def url(self) -> str:
        """Get Redis URL."""
        scheme = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{scheme}://{auth}{self.host}:{self.port}/{self.db}"


class RedisStorage(StorageBackend[Any]):
    """Redis storage backend.

    Features:
    - Fast in-memory caching
    - Native TTL support
    - Atomic operations
    - Distributed rate limiting support

    Best for:
    - Session caching
    - Rate limiting state
    - Circuit breaker state
    - Distributed locks
    """

    def __init__(self, config: RedisConfig | None = None):
        """Initialize Redis storage.

        Args:
            config: Redis configuration.
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis is required for Redis storage. Install with: pip install redis"
            )

        self.redis_config = config or RedisConfig()
        super().__init__(self.redis_config)

        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        """Connect to Redis."""
        self._client = redis.Redis(
            host=self.redis_config.host,
            port=self.redis_config.port,
            db=self.redis_config.db,
            password=self.redis_config.password,
            ssl=self.redis_config.ssl,
            socket_timeout=self.redis_config.socket_timeout,
            socket_connect_timeout=self.redis_config.socket_connect_timeout,
            decode_responses=self.redis_config.decode_responses,
        )
        # Test connection
        await self._client.ping()

    async def _ensure_connected(self) -> redis.Redis:
        """Ensure client is connected."""
        if self._client is None:
            await self.connect()
        return self._client

    async def get(self, key: str) -> Any | None:
        """Retrieve item by key.

        Args:
            key: Item key.

        Returns:
            Item value or None if not found.
        """
        client = await self._ensure_connected()
        full_key = self._make_key(key)

        value = await client.get(full_key)
        if value is None:
            return None

        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            # Return raw value if not JSON
            return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store item with optional TTL.

        Args:
            key: Item key.
            value: Item value (will be JSON serialized).
            ttl: Time-to-live in seconds.
        """
        client = await self._ensure_connected()
        full_key = self._make_key(key)

        # Serialize value
        try:
            serialized = json.dumps(value)
        except (TypeError, ValueError):
            serialized = str(value)

        # Use default TTL if not specified
        effective_ttl = ttl if ttl is not None else self.config.default_ttl

        if effective_ttl is not None:
            await client.setex(full_key, effective_ttl, serialized)
        else:
            await client.set(full_key, serialized)

    async def delete(self, key: str) -> bool:
        """Delete item.

        Args:
            key: Item key.

        Returns:
            True if item existed and was deleted.
        """
        client = await self._ensure_connected()
        full_key = self._make_key(key)

        result = await client.delete(full_key)
        return result > 0

    async def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Item key.

        Returns:
            True if key exists.
        """
        client = await self._ensure_connected()
        full_key = self._make_key(key)

        result = await client.exists(full_key)
        return result > 0

    async def keys(self, pattern: str = "*") -> list[str]:
        """List keys matching pattern.

        Args:
            pattern: Glob pattern.

        Returns:
            List of matching keys (without prefix).
        """
        client = await self._ensure_connected()
        full_pattern = self._make_key(pattern)
        prefix_len = len(self.config.prefix)

        # Use SCAN for large key sets
        matching_keys = []
        async for key in client.scan_iter(match=full_pattern):
            matching_keys.append(key[prefix_len:])

        return matching_keys

    async def clear(self) -> int:
        """Clear all items with the configured prefix.

        Returns:
            Number of items cleared.
        """
        client = await self._ensure_connected()
        pattern = self._make_key("*")

        # Collect keys first
        keys_to_delete = []
        async for key in client.scan_iter(match=pattern):
            keys_to_delete.append(key)

        if keys_to_delete:
            await client.delete(*keys_to_delete)

        return len(keys_to_delete)

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # Redis-specific methods for rate limiting

    async def incr(self, key: str, amount: int = 1) -> int:
        """Increment a counter.

        Args:
            key: Counter key.
            amount: Amount to increment.

        Returns:
            New counter value.
        """
        client = await self._ensure_connected()
        full_key = self._make_key(key)

        return await client.incrby(full_key, amount)

    async def incr_with_ttl(self, key: str, amount: int = 1, ttl: int = 60) -> int:
        """Increment counter and set TTL atomically.

        Args:
            key: Counter key.
            amount: Amount to increment.
            ttl: TTL in seconds.

        Returns:
            New counter value.
        """
        client = await self._ensure_connected()
        full_key = self._make_key(key)

        # Use Lua script for atomicity
        script = """
        local current = redis.call('INCRBY', KEYS[1], ARGV[1])
        if current == tonumber(ARGV[1]) then
            redis.call('EXPIRE', KEYS[1], ARGV[2])
        end
        return current
        """

        return await client.eval(script, 1, full_key, amount, ttl)

    async def get_ttl(self, key: str) -> int:
        """Get remaining TTL for a key.

        Args:
            key: Item key.

        Returns:
            TTL in seconds (-2 if key doesn't exist, -1 if no TTL).
        """
        client = await self._ensure_connected()
        full_key = self._make_key(key)

        return await client.ttl(full_key)

    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL on existing key.

        Args:
            key: Item key.
            ttl: TTL in seconds.

        Returns:
            True if TTL was set.
        """
        client = await self._ensure_connected()
        full_key = self._make_key(key)

        return await client.expire(full_key, ttl)

    async def get_many(self, keys: list[str]) -> dict[str, Any | None]:
        """Get multiple items efficiently.

        Args:
            keys: List of keys.

        Returns:
            Dict of key to value.
        """
        if not keys:
            return {}

        client = await self._ensure_connected()
        full_keys = [self._make_key(k) for k in keys]

        values = await client.mget(full_keys)

        results = {}
        for key, value in zip(keys, values):
            if value is None:
                results[key] = None
            else:
                try:
                    results[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    results[key] = value

        return results

    async def set_many(self, items: dict[str, Any], ttl: int | None = None) -> None:
        """Set multiple items efficiently.

        Args:
            items: Dict of key to value.
            ttl: TTL for all items.
        """
        if not items:
            return

        client = await self._ensure_connected()

        # Serialize values
        serialized = {
            self._make_key(k): json.dumps(v) if not isinstance(v, str) else v
            for k, v in items.items()
        }

        await client.mset(serialized)

        # Set TTL if specified
        if ttl is not None:
            for key in serialized.keys():
                await client.expire(key, ttl)

    async def __aenter__(self) -> RedisStorage:
        """Async context manager entry."""
        await self.connect()
        return self

    def __repr__(self) -> str:
        return f"RedisStorage(host={self.redis_config.host}, port={self.redis_config.port})"
