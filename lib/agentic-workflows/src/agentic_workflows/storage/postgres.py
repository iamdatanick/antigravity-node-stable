"""PostgreSQL storage backend for persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .base import StorageConfig, TransactionalStorage

# Try to import asyncpg
try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False


@dataclass
class PostgresConfig(StorageConfig):
    """PostgreSQL storage configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "agentic_workflows"
    user: str = "postgres"
    password: str = ""
    table_name: str = "kv_store"
    min_pool_size: int = 2
    max_pool_size: int = 10
    ssl: bool = False

    @property
    def dsn(self) -> str:
        """Get connection DSN."""
        ssl_param = "?sslmode=require" if self.ssl else ""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}{ssl_param}"


class PostgresStorage(TransactionalStorage[Any]):
    """PostgreSQL storage backend.

    Features:
    - Persistent storage across restarts
    - JSONB for flexible value storage
    - TTL support with automatic expiration
    - Transaction support
    - Connection pooling

    Schema:
    ```sql
    CREATE TABLE IF NOT EXISTS kv_store (
        key TEXT PRIMARY KEY,
        value JSONB NOT NULL,
        expires_at TIMESTAMP WITH TIME ZONE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_kv_expires ON kv_store (expires_at) WHERE expires_at IS NOT NULL;
    ```
    """

    def __init__(self, config: PostgresConfig | None = None):
        """Initialize PostgreSQL storage.

        Args:
            config: PostgreSQL configuration.
        """
        if not ASYNCPG_AVAILABLE:
            raise ImportError(
                "asyncpg is required for PostgreSQL storage. Install with: pip install asyncpg"
            )

        self.pg_config = config or PostgresConfig()
        super().__init__(self.pg_config)

        self._pool: asyncpg.Pool | None = None
        self._conn: asyncpg.Connection | None = None  # For transactions

    async def connect(self) -> None:
        """Connect to PostgreSQL and create schema if needed."""
        self._pool = await asyncpg.create_pool(
            self.pg_config.dsn,
            min_size=self.pg_config.min_pool_size,
            max_size=self.pg_config.max_pool_size,
        )

        # Create table if not exists
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.pg_config.table_name} (
                    key TEXT PRIMARY KEY,
                    value JSONB NOT NULL,
                    expires_at TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.pg_config.table_name}_expires
                ON {self.pg_config.table_name} (expires_at)
                WHERE expires_at IS NOT NULL
            """)

    async def _get_conn(self) -> asyncpg.Connection:
        """Get connection (transaction or pool)."""
        if self._conn is not None:
            return self._conn
        if self._pool is None:
            await self.connect()
        return await self._pool.acquire()

    async def _release_conn(self, conn: asyncpg.Connection) -> None:
        """Release connection if not in transaction."""
        if self._conn is None and self._pool is not None:
            await self._pool.release(conn)

    async def get(self, key: str) -> Any | None:
        """Retrieve item by key.

        Args:
            key: Item key.

        Returns:
            Item value or None if not found or expired.
        """
        full_key = self._make_key(key)
        conn = await self._get_conn()

        try:
            row = await conn.fetchrow(
                f"""
                SELECT value FROM {self.pg_config.table_name}
                WHERE key = $1 AND (expires_at IS NULL OR expires_at > NOW())
                """,
                full_key,
            )

            if row is None:
                return None

            return json.loads(row["value"])

        finally:
            await self._release_conn(conn)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store item with optional TTL.

        Args:
            key: Item key.
            value: Item value (must be JSON serializable).
            ttl: Time-to-live in seconds.
        """
        full_key = self._make_key(key)
        json_value = json.dumps(value)

        # Calculate expiration
        if ttl is not None:
            expires_at = f"NOW() + INTERVAL '{ttl} seconds'"
        elif self.config.default_ttl is not None:
            expires_at = f"NOW() + INTERVAL '{self.config.default_ttl} seconds'"
        else:
            expires_at = "NULL"

        conn = await self._get_conn()

        try:
            await conn.execute(
                f"""
                INSERT INTO {self.pg_config.table_name} (key, value, expires_at, updated_at)
                VALUES ($1, $2::jsonb, {expires_at}, NOW())
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    expires_at = EXCLUDED.expires_at,
                    updated_at = NOW()
                """,
                full_key,
                json_value,
            )
        finally:
            await self._release_conn(conn)

    async def delete(self, key: str) -> bool:
        """Delete item.

        Args:
            key: Item key.

        Returns:
            True if item existed and was deleted.
        """
        full_key = self._make_key(key)
        conn = await self._get_conn()

        try:
            result = await conn.execute(
                f"DELETE FROM {self.pg_config.table_name} WHERE key = $1",
                full_key,
            )
            return result == "DELETE 1"
        finally:
            await self._release_conn(conn)

    async def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Item key.

        Returns:
            True if key exists and is not expired.
        """
        full_key = self._make_key(key)
        conn = await self._get_conn()

        try:
            result = await conn.fetchval(
                f"""
                SELECT 1 FROM {self.pg_config.table_name}
                WHERE key = $1 AND (expires_at IS NULL OR expires_at > NOW())
                """,
                full_key,
            )
            return result is not None
        finally:
            await self._release_conn(conn)

    async def keys(self, pattern: str = "*") -> list[str]:
        """List keys matching pattern.

        Args:
            pattern: Glob pattern (converted to SQL LIKE).

        Returns:
            List of matching keys (without prefix).
        """
        full_pattern = self._make_key(pattern)
        # Convert glob to SQL LIKE
        sql_pattern = full_pattern.replace("*", "%").replace("?", "_")
        prefix_len = len(self.config.prefix)

        conn = await self._get_conn()

        try:
            rows = await conn.fetch(
                f"""
                SELECT key FROM {self.pg_config.table_name}
                WHERE key LIKE $1 AND (expires_at IS NULL OR expires_at > NOW())
                """,
                sql_pattern,
            )
            return [row["key"][prefix_len:] for row in rows]
        finally:
            await self._release_conn(conn)

    async def clear(self) -> int:
        """Clear all items with the configured prefix.

        Returns:
            Number of items cleared.
        """
        pattern = f"{self.config.prefix}%"
        conn = await self._get_conn()

        try:
            result = await conn.execute(
                f"DELETE FROM {self.pg_config.table_name} WHERE key LIKE $1",
                pattern,
            )
            # Parse "DELETE N"
            count = int(result.split()[1]) if result else 0
            return count
        finally:
            await self._release_conn(conn)

    async def cleanup_expired(self) -> int:
        """Remove expired items.

        Returns:
            Number of items removed.
        """
        conn = await self._get_conn()

        try:
            result = await conn.execute(
                f"DELETE FROM {self.pg_config.table_name} WHERE expires_at IS NOT NULL AND expires_at <= NOW()"
            )
            count = int(result.split()[1]) if result else 0
            return count
        finally:
            await self._release_conn(conn)

    async def begin(self) -> None:
        """Begin a transaction."""
        if self._conn is not None:
            raise RuntimeError("Transaction already in progress")

        if self._pool is None:
            await self.connect()

        self._conn = await self._pool.acquire()
        await self._conn.execute("BEGIN")

    async def commit(self) -> None:
        """Commit the current transaction."""
        if self._conn is None:
            raise RuntimeError("No transaction in progress")

        try:
            await self._conn.execute("COMMIT")
        finally:
            await self._pool.release(self._conn)
            self._conn = None

    async def rollback(self) -> None:
        """Rollback the current transaction."""
        if self._conn is None:
            raise RuntimeError("No transaction in progress")

        try:
            await self._conn.execute("ROLLBACK")
        finally:
            await self._pool.release(self._conn)
            self._conn = None

    async def close(self) -> None:
        """Close the connection pool."""
        if self._conn is not None:
            await self.rollback()

        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def __aenter__(self) -> PostgresStorage:
        """Async context manager entry."""
        await self.connect()
        return self

    def __repr__(self) -> str:
        return f"PostgresStorage(host={self.pg_config.host}, db={self.pg_config.database})"
