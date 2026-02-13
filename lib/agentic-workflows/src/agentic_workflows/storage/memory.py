"""In-memory storage backend for testing and development."""

from __future__ import annotations

import asyncio
import fnmatch
import time
from dataclasses import dataclass, field
from typing import Any

from .base import StorageBackend, StorageConfig


@dataclass
class MemoryItem:
    """An item in memory storage."""

    value: Any
    expires_at: float | None = None

    def is_expired(self) -> bool:
        """Check if item is expired."""
        if self.expires_at is None:
            return False
        return time.time() >= self.expires_at


class MemoryStorage(StorageBackend[Any]):
    """In-memory storage backend.

    Useful for:
    - Testing
    - Development
    - Single-process deployments

    Note: Data is not persisted across process restarts.
    """

    def __init__(self, config: StorageConfig | None = None):
        """Initialize memory storage.

        Args:
            config: Storage configuration.
        """
        super().__init__(config)
        self._data: dict[str, MemoryItem] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._closed = False

    async def start_cleanup(self, interval: float = 60.0) -> None:
        """Start background cleanup task for expired items.

        Args:
            interval: Cleanup interval in seconds.
        """
        if self._cleanup_task is not None:
            return

        async def cleanup_loop():
            while not self._closed:
                await asyncio.sleep(interval)
                await self._cleanup_expired()

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def _cleanup_expired(self) -> int:
        """Remove expired items.

        Returns:
            Number of items removed.
        """
        async with self._lock:
            expired = [
                key for key, item in self._data.items()
                if item.is_expired()
            ]
            for key in expired:
                del self._data[key]
            return len(expired)

    async def get(self, key: str) -> Any | None:
        """Retrieve item by key.

        Args:
            key: Item key.

        Returns:
            Item value or None if not found or expired.
        """
        full_key = self._make_key(key)

        async with self._lock:
            item = self._data.get(full_key)
            if item is None:
                return None

            if item.is_expired():
                del self._data[full_key]
                return None

            return item.value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store item with optional TTL.

        Args:
            key: Item key.
            value: Item value.
            ttl: Time-to-live in seconds.
        """
        full_key = self._make_key(key)

        expires_at = None
        if ttl is not None:
            expires_at = time.time() + ttl
        elif self.config.default_ttl is not None:
            expires_at = time.time() + self.config.default_ttl

        async with self._lock:
            self._data[full_key] = MemoryItem(value=value, expires_at=expires_at)

    async def delete(self, key: str) -> bool:
        """Delete item.

        Args:
            key: Item key.

        Returns:
            True if item existed and was deleted.
        """
        full_key = self._make_key(key)

        async with self._lock:
            if full_key in self._data:
                del self._data[full_key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Item key.

        Returns:
            True if key exists and is not expired.
        """
        full_key = self._make_key(key)

        async with self._lock:
            item = self._data.get(full_key)
            if item is None:
                return False

            if item.is_expired():
                del self._data[full_key]
                return False

            return True

    async def keys(self, pattern: str = "*") -> list[str]:
        """List keys matching pattern.

        Args:
            pattern: Glob pattern.

        Returns:
            List of matching keys (without prefix).
        """
        full_pattern = self._make_key(pattern)
        prefix_len = len(self.config.prefix)

        async with self._lock:
            # Clean up expired first
            expired = [
                key for key, item in self._data.items()
                if item.is_expired()
            ]
            for key in expired:
                del self._data[key]

            # Match pattern
            matches = []
            for key in self._data.keys():
                if fnmatch.fnmatch(key, full_pattern):
                    # Remove prefix
                    matches.append(key[prefix_len:])

            return matches

    async def clear(self) -> int:
        """Clear all items with the configured prefix.

        Returns:
            Number of items cleared.
        """
        async with self._lock:
            keys_to_delete = [
                key for key in self._data.keys()
                if key.startswith(self.config.prefix)
            ]
            for key in keys_to_delete:
                del self._data[key]
            return len(keys_to_delete)

    async def close(self) -> None:
        """Close the storage."""
        self._closed = True
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def get_many(self, keys: list[str]) -> dict[str, Any | None]:
        """Get multiple items efficiently.

        Args:
            keys: List of keys.

        Returns:
            Dict of key to value.
        """
        results = {}
        async with self._lock:
            for key in keys:
                full_key = self._make_key(key)
                item = self._data.get(full_key)

                if item is None:
                    results[key] = None
                elif item.is_expired():
                    del self._data[full_key]
                    results[key] = None
                else:
                    results[key] = item.value

        return results

    async def set_many(self, items: dict[str, Any], ttl: int | None = None) -> None:
        """Set multiple items efficiently.

        Args:
            items: Dict of key to value.
            ttl: TTL for all items.
        """
        expires_at = None
        if ttl is not None:
            expires_at = time.time() + ttl
        elif self.config.default_ttl is not None:
            expires_at = time.time() + self.config.default_ttl

        async with self._lock:
            for key, value in items.items():
                full_key = self._make_key(key)
                self._data[full_key] = MemoryItem(value=value, expires_at=expires_at)

    def size(self) -> int:
        """Get number of items in storage."""
        return len(self._data)

    def __repr__(self) -> str:
        return f"MemoryStorage(prefix={self.config.prefix!r}, size={self.size()})"
