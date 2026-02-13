"""State management for OpenAI Apps SDK widgets.

This module provides session state management for stateful widgets:
- WidgetSessionManager: High-level session management
- StateStore: Abstract interface for state persistence
- InMemoryStateStore: In-memory state storage for development
- RedisStateStore: Redis-backed state storage for production
- D1StateStore: Cloudflare D1-backed state storage

Supports TTL-based expiration, state merging, and cleanup.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Awaitable

from .widget_types import WidgetState

logger = logging.getLogger(__name__)


@dataclass
class StateStoreConfig:
    """Configuration for state stores.

    Attributes:
        prefix: Key prefix for namespacing
        default_ttl: Default TTL in seconds (None = no expiration)
        cleanup_interval: Background cleanup interval in seconds
        max_sessions: Maximum number of sessions to track (0 = unlimited)
    """

    prefix: str = "widget_state:"
    default_ttl: int | None = 3600  # 1 hour default
    cleanup_interval: float = 60.0
    max_sessions: int = 0


class StateStore(ABC):
    """Abstract interface for widget state persistence.

    Implementations provide backend-specific storage for widget state,
    supporting get, set, delete, and TTL operations.

    Example:
        >>> store = InMemoryStateStore()
        >>> await store.set("sess_123", {"items": []})
        >>> state = await store.get("sess_123")
        >>> await store.delete("sess_123")
    """

    def __init__(self, config: StateStoreConfig | None = None):
        """Initialize state store.

        Args:
            config: Store configuration.
        """
        self.config = config or StateStoreConfig()

    def _make_key(self, session_id: str) -> str:
        """Create prefixed key for session."""
        return f"{self.config.prefix}{session_id}"

    @abstractmethod
    async def get(self, session_id: str) -> WidgetState | None:
        """Retrieve state by session ID.

        Args:
            session_id: Session identifier.

        Returns:
            WidgetState or None if not found.
        """
        pass

    @abstractmethod
    async def set(
        self, session_id: str, state: WidgetState, ttl: int | None = None
    ) -> None:
        """Store state with optional TTL.

        Args:
            session_id: Session identifier.
            state: Widget state to store.
            ttl: Time-to-live in seconds (None for default).
        """
        pass

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """Delete state by session ID.

        Args:
            session_id: Session identifier.

        Returns:
            True if state existed and was deleted.
        """
        pass

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        """Check if session state exists.

        Args:
            session_id: Session identifier.

        Returns:
            True if session exists.
        """
        pass

    @abstractmethod
    async def list_sessions(self, pattern: str = "*") -> list[str]:
        """List session IDs matching pattern.

        Args:
            pattern: Glob pattern for matching.

        Returns:
            List of matching session IDs.
        """
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Remove expired sessions.

        Returns:
            Number of sessions removed.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the store and release resources."""
        pass

    async def __aenter__(self) -> "StateStore":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


@dataclass
class _MemoryStateItem:
    """Internal storage item for in-memory store."""

    state: WidgetState
    expires_at: float | None = None

    def is_expired(self) -> bool:
        """Check if item is expired."""
        if self.expires_at is None:
            return False
        return time.time() >= self.expires_at


class InMemoryStateStore(StateStore):
    """In-memory state storage for development and testing.

    Provides fast, non-persistent storage with TTL support
    and background cleanup.

    Example:
        >>> async with InMemoryStateStore() as store:
        ...     state = WidgetState(session_id="sess_123")
        ...     state.set("cart", {"items": []})
        ...     await store.set("sess_123", state)
        ...     retrieved = await store.get("sess_123")
    """

    def __init__(self, config: StateStoreConfig | None = None):
        """Initialize in-memory store.

        Args:
            config: Store configuration.
        """
        super().__init__(config)
        self._data: dict[str, _MemoryStateItem] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._closed = False

    async def start_cleanup(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is not None:
            return

        async def cleanup_loop():
            while not self._closed:
                await asyncio.sleep(self.config.cleanup_interval)
                count = await self.cleanup_expired()
                if count > 0:
                    logger.debug(f"Cleaned up {count} expired widget sessions")

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def get(self, session_id: str) -> WidgetState | None:
        """Retrieve state by session ID."""
        key = self._make_key(session_id)

        async with self._lock:
            item = self._data.get(key)
            if item is None:
                return None

            if item.is_expired():
                del self._data[key]
                return None

            return item.state

    async def set(
        self, session_id: str, state: WidgetState, ttl: int | None = None
    ) -> None:
        """Store state with optional TTL."""
        key = self._make_key(session_id)

        expires_at = None
        effective_ttl = ttl if ttl is not None else self.config.default_ttl
        if effective_ttl is not None:
            expires_at = time.time() + effective_ttl

        async with self._lock:
            self._data[key] = _MemoryStateItem(state=state, expires_at=expires_at)

    async def delete(self, session_id: str) -> bool:
        """Delete state by session ID."""
        key = self._make_key(session_id)

        async with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    async def exists(self, session_id: str) -> bool:
        """Check if session state exists."""
        key = self._make_key(session_id)

        async with self._lock:
            item = self._data.get(key)
            if item is None:
                return False
            if item.is_expired():
                del self._data[key]
                return False
            return True

    async def list_sessions(self, pattern: str = "*") -> list[str]:
        """List session IDs matching pattern."""
        import fnmatch

        prefix_len = len(self.config.prefix)

        async with self._lock:
            # Clean expired first
            expired = [k for k, v in self._data.items() if v.is_expired()]
            for key in expired:
                del self._data[key]

            # Match pattern
            sessions = []
            for key in self._data.keys():
                session_id = key[prefix_len:]
                if fnmatch.fnmatch(session_id, pattern):
                    sessions.append(session_id)

            return sessions

    async def cleanup_expired(self) -> int:
        """Remove expired sessions."""
        async with self._lock:
            expired = [k for k, v in self._data.items() if v.is_expired()]
            for key in expired:
                del self._data[key]
            return len(expired)

    async def close(self) -> None:
        """Close the store."""
        self._closed = True
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    def size(self) -> int:
        """Get number of stored sessions."""
        return len(self._data)


class RedisStateStore(StateStore):
    """Redis-backed state storage for production deployments.

    Provides distributed, persistent storage with native TTL support.

    Requires: pip install redis

    Example:
        >>> from agentic_workflows.openai_apps.state import RedisStateStore
        >>> config = StateStoreConfig(prefix="myapp:widget:")
        >>> store = RedisStateStore(config, host="localhost", port=6379)
        >>> await store.connect()
    """

    def __init__(
        self,
        config: StateStoreConfig | None = None,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        ssl: bool = False,
    ):
        """Initialize Redis state store.

        Args:
            config: Store configuration.
            host: Redis host.
            port: Redis port.
            db: Redis database number.
            password: Redis password.
            ssl: Enable SSL/TLS.
        """
        super().__init__(config)
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.ssl = ssl
        self._client = None

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError(
                "redis is required for RedisStateStore. "
                "Install with: pip install redis"
            )

        self._client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            ssl=self.ssl,
            decode_responses=True,
        )
        await self._client.ping()

    async def _ensure_connected(self):
        """Ensure client is connected."""
        if self._client is None:
            await self.connect()
        return self._client

    async def get(self, session_id: str) -> WidgetState | None:
        """Retrieve state by session ID."""
        client = await self._ensure_connected()
        key = self._make_key(session_id)

        data = await client.get(key)
        if data is None:
            return None

        try:
            state_dict = json.loads(data)
            return WidgetState.from_dict(state_dict)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to deserialize state for {session_id}: {e}")
            return None

    async def set(
        self, session_id: str, state: WidgetState, ttl: int | None = None
    ) -> None:
        """Store state with optional TTL."""
        client = await self._ensure_connected()
        key = self._make_key(session_id)

        data = json.dumps(state.to_dict())
        effective_ttl = ttl if ttl is not None else self.config.default_ttl

        if effective_ttl is not None:
            await client.setex(key, effective_ttl, data)
        else:
            await client.set(key, data)

    async def delete(self, session_id: str) -> bool:
        """Delete state by session ID."""
        client = await self._ensure_connected()
        key = self._make_key(session_id)

        result = await client.delete(key)
        return result > 0

    async def exists(self, session_id: str) -> bool:
        """Check if session state exists."""
        client = await self._ensure_connected()
        key = self._make_key(session_id)

        result = await client.exists(key)
        return result > 0

    async def list_sessions(self, pattern: str = "*") -> list[str]:
        """List session IDs matching pattern."""
        client = await self._ensure_connected()
        full_pattern = self._make_key(pattern)
        prefix_len = len(self.config.prefix)

        sessions = []
        async for key in client.scan_iter(match=full_pattern):
            sessions.append(key[prefix_len:])

        return sessions

    async def cleanup_expired(self) -> int:
        """Remove expired sessions (Redis handles this automatically)."""
        # Redis TTL handles expiration, but we can clean up any
        # sessions that are explicitly marked as expired
        return 0

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class D1StateStore(StateStore):
    """Cloudflare D1-backed state storage.

    Uses Cloudflare D1 (SQLite) for serverless state persistence.

    Requires: Cloudflare Workers environment with D1 binding.

    Example:
        >>> # In Cloudflare Worker context
        >>> store = D1StateStore(config, d1_binding=env.D1_DATABASE)
    """

    # SQL schema for D1 table
    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS widget_state (
        session_id TEXT PRIMARY KEY,
        state_data TEXT NOT NULL,
        version INTEGER DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        expires_at TEXT,
        metadata TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_expires_at ON widget_state(expires_at);
    """

    def __init__(
        self,
        config: StateStoreConfig | None = None,
        d1_binding: Any = None,
        api_token: str | None = None,
        account_id: str | None = None,
        database_id: str | None = None,
    ):
        """Initialize D1 state store.

        Can be used either with a direct D1 binding (in Worker) or
        via the REST API (external access).

        Args:
            config: Store configuration.
            d1_binding: D1 binding from Worker environment.
            api_token: Cloudflare API token for REST access.
            account_id: Cloudflare account ID for REST access.
            database_id: D1 database ID for REST access.
        """
        super().__init__(config)
        self.d1_binding = d1_binding
        self.api_token = api_token
        self.account_id = account_id
        self.database_id = database_id
        self._initialized = False

    async def initialize(self) -> None:
        """Create table if it doesn't exist."""
        if self._initialized:
            return

        if self.d1_binding:
            await self.d1_binding.exec(self.CREATE_TABLE_SQL)
        else:
            # Use REST API
            await self._execute_sql(self.CREATE_TABLE_SQL)

        self._initialized = True

    async def _execute_sql(
        self, sql: str, params: list | None = None
    ) -> list[dict] | None:
        """Execute SQL via D1 REST API."""
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for D1StateStore REST API. "
                "Install with: pip install httpx"
            )

        url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{self.account_id}/d1/database/{self.database_id}/query"
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json",
                },
                json={"sql": sql, "params": params or []},
            )
            response.raise_for_status()
            result = response.json()

            if result.get("success") and result.get("result"):
                return result["result"][0].get("results", [])
            return None

    async def get(self, session_id: str) -> WidgetState | None:
        """Retrieve state by session ID."""
        await self.initialize()

        sql = "SELECT * FROM widget_state WHERE session_id = ?"

        if self.d1_binding:
            result = await self.d1_binding.prepare(sql).bind(session_id).first()
        else:
            results = await self._execute_sql(sql, [session_id])
            result = results[0] if results else None

        if not result:
            return None

        # Check expiration
        if result.get("expires_at"):
            expires_at = datetime.fromisoformat(result["expires_at"])
            if datetime.utcnow() >= expires_at:
                await self.delete(session_id)
                return None

        return WidgetState(
            session_id=result["session_id"],
            state_data=json.loads(result["state_data"]),
            version=result.get("version", 1),
            created_at=datetime.fromisoformat(result["created_at"]),
            updated_at=datetime.fromisoformat(result["updated_at"]),
            expires_at=datetime.fromisoformat(result["expires_at"])
            if result.get("expires_at")
            else None,
            metadata=json.loads(result.get("metadata") or "{}"),
        )

    async def set(
        self, session_id: str, state: WidgetState, ttl: int | None = None
    ) -> None:
        """Store state with optional TTL."""
        await self.initialize()

        effective_ttl = ttl if ttl is not None else self.config.default_ttl
        expires_at = None
        if effective_ttl is not None:
            expires_at = (datetime.utcnow() + timedelta(seconds=effective_ttl)).isoformat()

        sql = """
        INSERT OR REPLACE INTO widget_state
        (session_id, state_data, version, created_at, updated_at, expires_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        params = [
            session_id,
            json.dumps(state.state_data),
            state.version,
            state.created_at.isoformat(),
            state.updated_at.isoformat(),
            expires_at,
            json.dumps(state.metadata),
        ]

        if self.d1_binding:
            await self.d1_binding.prepare(sql).bind(*params).run()
        else:
            await self._execute_sql(sql, params)

    async def delete(self, session_id: str) -> bool:
        """Delete state by session ID."""
        await self.initialize()

        sql = "DELETE FROM widget_state WHERE session_id = ?"

        if self.d1_binding:
            result = await self.d1_binding.prepare(sql).bind(session_id).run()
            return result.meta.changes > 0
        else:
            await self._execute_sql(sql, [session_id])
            return True  # REST API doesn't return affected rows easily

    async def exists(self, session_id: str) -> bool:
        """Check if session state exists."""
        state = await self.get(session_id)
        return state is not None

    async def list_sessions(self, pattern: str = "*") -> list[str]:
        """List session IDs matching pattern."""
        await self.initialize()

        # Convert glob pattern to SQL LIKE pattern
        like_pattern = pattern.replace("*", "%").replace("?", "_")

        sql = "SELECT session_id FROM widget_state WHERE session_id LIKE ?"

        if self.d1_binding:
            results = await self.d1_binding.prepare(sql).bind(like_pattern).all()
            return [r["session_id"] for r in results.results]
        else:
            results = await self._execute_sql(sql, [like_pattern])
            return [r["session_id"] for r in (results or [])]

    async def cleanup_expired(self) -> int:
        """Remove expired sessions."""
        await self.initialize()

        now = datetime.utcnow().isoformat()
        sql = "DELETE FROM widget_state WHERE expires_at IS NOT NULL AND expires_at < ?"

        if self.d1_binding:
            result = await self.d1_binding.prepare(sql).bind(now).run()
            return result.meta.changes
        else:
            await self._execute_sql(sql, [now])
            return 0  # REST API doesn't return affected rows easily

    async def close(self) -> None:
        """Close the store (no-op for D1)."""
        pass


class WidgetSessionManager:
    """High-level session management for widgets.

    Provides a convenient interface for managing widget state,
    including creation, retrieval, updates, and cleanup.

    Example:
        >>> manager = WidgetSessionManager()
        >>> session_id = await manager.create_session()
        >>> await manager.update_state(session_id, {"items": [{"id": 1}]})
        >>> state = await manager.get_state(session_id)
        >>> await manager.merge_state(session_id, {"total": 99.99})
    """

    def __init__(
        self,
        store: StateStore | None = None,
        default_ttl: int = 3600,
    ):
        """Initialize session manager.

        Args:
            store: State store backend (default: InMemoryStateStore).
            default_ttl: Default session TTL in seconds.
        """
        self.store = store or InMemoryStateStore()
        self.default_ttl = default_ttl
        self._on_create_callbacks: list[Callable[[str, WidgetState], Awaitable[None]]] = []
        self._on_update_callbacks: list[Callable[[str, WidgetState], Awaitable[None]]] = []

    def on_create(
        self, callback: Callable[[str, WidgetState], Awaitable[None]]
    ) -> None:
        """Register callback for session creation.

        Args:
            callback: Async callback receiving (session_id, state).
        """
        self._on_create_callbacks.append(callback)

    def on_update(
        self, callback: Callable[[str, WidgetState], Awaitable[None]]
    ) -> None:
        """Register callback for session updates.

        Args:
            callback: Async callback receiving (session_id, state).
        """
        self._on_update_callbacks.append(callback)

    async def create_session(
        self,
        session_id: str | None = None,
        initial_state: dict[str, Any] | None = None,
        ttl: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new widget session.

        Args:
            session_id: Optional session ID (auto-generated if not provided).
            initial_state: Initial state data.
            ttl: Session TTL in seconds.
            metadata: Additional metadata.

        Returns:
            Session ID.
        """
        session_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        effective_ttl = ttl if ttl is not None else self.default_ttl

        state = WidgetState(
            session_id=session_id,
            state_data=initial_state or {},
            metadata=metadata or {},
        )

        if effective_ttl:
            state.expires_at = datetime.utcnow() + timedelta(seconds=effective_ttl)

        await self.store.set(session_id, state, effective_ttl)

        # Notify callbacks
        for callback in self._on_create_callbacks:
            await callback(session_id, state)

        logger.debug(f"Created widget session: {session_id}")
        return session_id

    async def get_state(self, session_id: str) -> WidgetState | None:
        """Get session state.

        Args:
            session_id: Session identifier.

        Returns:
            WidgetState or None if not found.
        """
        return await self.store.get(session_id)

    async def update_state(
        self,
        session_id: str,
        new_data: dict[str, Any],
        extend_ttl: bool = True,
    ) -> WidgetState | None:
        """Update session state (full replacement).

        Args:
            session_id: Session identifier.
            new_data: New state data.
            extend_ttl: Extend TTL on update.

        Returns:
            Updated WidgetState or None if session not found.
        """
        state = await self.store.get(session_id)
        if state is None:
            return None

        state.update(new_data)

        ttl = None
        if extend_ttl and state.expires_at:
            ttl = self.default_ttl

        await self.store.set(session_id, state, ttl)

        # Notify callbacks
        for callback in self._on_update_callbacks:
            await callback(session_id, state)

        return state

    async def merge_state(
        self,
        session_id: str,
        partial_data: dict[str, Any],
        deep: bool = False,
        extend_ttl: bool = True,
    ) -> WidgetState | None:
        """Merge partial data into session state.

        Args:
            session_id: Session identifier.
            partial_data: Partial state data to merge.
            deep: Use deep merge for nested objects.
            extend_ttl: Extend TTL on update.

        Returns:
            Updated WidgetState or None if session not found.
        """
        state = await self.store.get(session_id)
        if state is None:
            return None

        if deep:
            state.merge_deep(partial_data)
        else:
            state.merge(partial_data)

        ttl = None
        if extend_ttl and state.expires_at:
            ttl = self.default_ttl

        await self.store.set(session_id, state, ttl)

        # Notify callbacks
        for callback in self._on_update_callbacks:
            await callback(session_id, state)

        return state

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session identifier.

        Returns:
            True if session existed and was deleted.
        """
        result = await self.store.delete(session_id)
        if result:
            logger.debug(f"Deleted widget session: {session_id}")
        return result

    async def session_exists(self, session_id: str) -> bool:
        """Check if session exists.

        Args:
            session_id: Session identifier.

        Returns:
            True if session exists.
        """
        return await self.store.exists(session_id)

    async def list_sessions(self, pattern: str = "*") -> list[str]:
        """List sessions matching pattern.

        Args:
            pattern: Glob pattern.

        Returns:
            List of session IDs.
        """
        return await self.store.list_sessions(pattern)

    async def cleanup(self) -> int:
        """Clean up expired sessions.

        Returns:
            Number of sessions cleaned up.
        """
        count = await self.store.cleanup_expired()
        if count > 0:
            logger.info(f"Cleaned up {count} expired widget sessions")
        return count

    async def close(self) -> None:
        """Close the session manager and its store."""
        await self.store.close()

    async def __aenter__(self) -> "WidgetSessionManager":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
