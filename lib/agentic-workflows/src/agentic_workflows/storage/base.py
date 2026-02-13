"""Abstract storage interface for persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class StorageConfig:
    """Base configuration for storage backends."""

    prefix: str = "agentic:"
    default_ttl: int | None = None  # Default TTL in seconds


class StorageBackend(ABC, Generic[T]):
    """Abstract base for storage backends.

    Provides a simple key-value interface with optional TTL support.
    Subclasses implement specific backends (PostgreSQL, Redis, Memory).
    """

    def __init__(self, config: StorageConfig | None = None):
        """Initialize storage backend.

        Args:
            config: Storage configuration.
        """
        self.config = config or StorageConfig()

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.config.prefix}{key}"

    @abstractmethod
    async def get(self, key: str) -> T | None:
        """Retrieve item by key.

        Args:
            key: Item key.

        Returns:
            Item value or None if not found.
        """
        pass

    @abstractmethod
    async def set(self, key: str, value: T, ttl: int | None = None) -> None:
        """Store item with optional TTL.

        Args:
            key: Item key.
            value: Item value.
            ttl: Time-to-live in seconds (None for no expiration).
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete item.

        Args:
            key: Item key.

        Returns:
            True if item existed and was deleted.
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Item key.

        Returns:
            True if key exists.
        """
        pass

    @abstractmethod
    async def keys(self, pattern: str = "*") -> list[str]:
        """List keys matching pattern.

        Args:
            pattern: Glob pattern (e.g., "agent:*").

        Returns:
            List of matching keys.
        """
        pass

    async def get_many(self, keys: list[str]) -> dict[str, T | None]:
        """Get multiple items.

        Args:
            keys: List of keys.

        Returns:
            Dict of key to value (None for missing keys).
        """
        results = {}
        for key in keys:
            results[key] = await self.get(key)
        return results

    async def set_many(self, items: dict[str, T], ttl: int | None = None) -> None:
        """Set multiple items.

        Args:
            items: Dict of key to value.
            ttl: TTL for all items.
        """
        for key, value in items.items():
            await self.set(key, value, ttl)

    async def delete_many(self, keys: list[str]) -> int:
        """Delete multiple items.

        Args:
            keys: List of keys.

        Returns:
            Number of items deleted.
        """
        count = 0
        for key in keys:
            if await self.delete(key):
                count += 1
        return count

    @abstractmethod
    async def clear(self) -> int:
        """Clear all items with the configured prefix.

        Returns:
            Number of items cleared.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the connection."""
        pass

    async def __aenter__(self) -> StorageBackend[T]:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


class TransactionalStorage(StorageBackend[T], ABC):
    """Storage backend with transaction support."""

    @abstractmethod
    async def begin(self) -> None:
        """Begin a transaction."""
        pass

    @abstractmethod
    async def commit(self) -> None:
        """Commit the current transaction."""
        pass

    @abstractmethod
    async def rollback(self) -> None:
        """Rollback the current transaction."""
        pass
