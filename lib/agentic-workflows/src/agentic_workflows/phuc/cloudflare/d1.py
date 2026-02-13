"""D1 database operations for PHUC platform."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx


@dataclass
class D1Config:
    """D1 database configuration."""

    account_id: str = field(default_factory=lambda: os.getenv("CF_ACCOUNT_ID", ""))
    api_token: str = field(default_factory=lambda: os.getenv("CF_API_TOKEN", ""))
    database_id: str = "b2443c54-6ece-4d69-8239-fd2004a3861e"  # phucai
    database_name: str = "phucai"


@dataclass
class QueryResult:
    """D1 query result."""

    success: bool
    results: list[dict]
    meta: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def count(self) -> int:
        return len(self.results)

    def first(self) -> dict | None:
        return self.results[0] if self.results else None


class D1Database:
    """D1 SQLite database client."""

    BASE_URL = "https://api.cloudflare.com/client/v4"

    # PHUC Schema
    SCHEMA = {
        "users": [
            "id TEXT PRIMARY KEY",
            "email TEXT UNIQUE NOT NULL",
            "password_hash TEXT NOT NULL",
            "role TEXT DEFAULT 'user'",
            "created_at DATETIME DEFAULT CURRENT_TIMESTAMP",
            "last_login DATETIME",
        ],
        "documents": [
            "id TEXT PRIMARY KEY",
            "user_id TEXT NOT NULL",
            "filename TEXT NOT NULL",
            "r2_key TEXT NOT NULL",
            "status TEXT DEFAULT 'pending'",
            "mime_type TEXT",
            "size_bytes INTEGER",
            "created_at DATETIME DEFAULT CURRENT_TIMESTAMP",
            "FOREIGN KEY (user_id) REFERENCES users(id)",
        ],
        "campaigns": [
            "id TEXT PRIMARY KEY",
            "user_id TEXT NOT NULL",
            "name TEXT NOT NULL",
            "therapeutic_area TEXT",
            "config JSON",
            "metrics JSON",
            "status TEXT DEFAULT 'draft'",
            "created_at DATETIME DEFAULT CURRENT_TIMESTAMP",
            "updated_at DATETIME",
            "FOREIGN KEY (user_id) REFERENCES users(id)",
        ],
        "chat_history": [
            "id TEXT PRIMARY KEY",
            "user_id TEXT NOT NULL",
            "session_id TEXT NOT NULL",
            "role TEXT NOT NULL",
            "content TEXT NOT NULL",
            "created_at DATETIME DEFAULT CURRENT_TIMESTAMP",
            "FOREIGN KEY (user_id) REFERENCES users(id)",
        ],
        "npi_registry": [
            "npi TEXT PRIMARY KEY",
            "name TEXT NOT NULL",
            "specialty TEXT",
            "taxonomy TEXT",
            "address TEXT",
            "city TEXT",
            "state TEXT",
            "zip TEXT",
            "status TEXT DEFAULT 'active'",
        ],
        "ndc_directory": [
            "ndc TEXT PRIMARY KEY",
            "brand_name TEXT",
            "generic_name TEXT",
            "dosage_form TEXT",
            "route TEXT",
            "manufacturer TEXT",
            "active_ingredient TEXT",
        ],
        "attributions": [
            "id TEXT PRIMARY KEY",
            "campaign_id TEXT NOT NULL",
            "npi TEXT",
            "ndc TEXT",
            "touchpoint_type TEXT",
            "attribution_score REAL",
            "attributed_nrx INTEGER DEFAULT 0",
            "attributed_trx INTEGER DEFAULT 0",
            "timestamp DATETIME",
            "created_at DATETIME DEFAULT CURRENT_TIMESTAMP",
            "FOREIGN KEY (campaign_id) REFERENCES campaigns(id)",
        ],
    }

    def __init__(self, config: D1Config = None):
        self.config = config or D1Config()
        self._client: httpx.AsyncClient | None = None

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_token}",
            "Content-Type": "application/json",
        }

    @property
    def base_url(self) -> str:
        return f"{self.BASE_URL}/accounts/{self.config.account_id}/d1/database/{self.config.database_id}"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(headers=self.headers, timeout=30.0)
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def query(self, sql: str, params: list = None) -> QueryResult:
        """Execute SQL query."""
        client = await self._get_client()

        payload = {"sql": sql}
        if params:
            payload["params"] = params

        response = await client.post(f"{self.base_url}/query", json=payload)

        if response.status_code != 200:
            return QueryResult(success=False, results=[], error=response.text)

        data = response.json()
        result = data.get("result", [{}])[0]

        return QueryResult(
            success=result.get("success", False),
            results=result.get("results", []),
            meta=result.get("meta", {}),
        )

    async def execute(self, sql: str, params: list = None) -> QueryResult:
        """Execute SQL statement (INSERT/UPDATE/DELETE)."""
        return await self.query(sql, params)

    async def batch(self, statements: list[tuple[str, list]]) -> list[QueryResult]:
        """Execute batch statements."""
        results = []
        for sql, params in statements:
            result = await self.query(sql, params)
            results.append(result)
        return results

    async def init_schema(self) -> bool:
        """Initialize PHUC database schema."""
        for table_name, columns in self.SCHEMA.items():
            sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
            result = await self.execute(sql)
            if not result.success:
                return False
        return True

    # Convenience methods for PHUC tables
    async def get_user(self, user_id: str) -> dict | None:
        """Get user by ID."""
        result = await self.query("SELECT * FROM users WHERE id = ?", [user_id])
        return result.first()

    async def get_user_by_email(self, email: str) -> dict | None:
        """Get user by email."""
        result = await self.query("SELECT * FROM users WHERE email = ?", [email])
        return result.first()

    async def get_documents(self, user_id: str, status: str = None) -> list[dict]:
        """Get user documents."""
        if status:
            result = await self.query(
                "SELECT * FROM documents WHERE user_id = ? AND status = ? ORDER BY created_at DESC",
                [user_id, status],
            )
        else:
            result = await self.query(
                "SELECT * FROM documents WHERE user_id = ? ORDER BY created_at DESC", [user_id]
            )
        return result.results

    async def get_campaigns(self, user_id: str) -> list[dict]:
        """Get user campaigns."""
        result = await self.query(
            "SELECT * FROM campaigns WHERE user_id = ? ORDER BY created_at DESC", [user_id]
        )
        return result.results

    async def lookup_npi(self, npi: str) -> dict | None:
        """Lookup NPI record."""
        result = await self.query("SELECT * FROM npi_registry WHERE npi = ?", [npi])
        return result.first()

    async def lookup_ndc(self, ndc: str) -> dict | None:
        """Lookup NDC record."""
        result = await self.query("SELECT * FROM ndc_directory WHERE ndc = ?", [ndc])
        return result.first()

    async def get_attribution_summary(self, campaign_id: str) -> dict:
        """Get attribution summary for campaign."""
        result = await self.query(
            """
            SELECT 
                COUNT(*) as touchpoints,
                SUM(attributed_nrx) as total_nrx,
                SUM(attributed_trx) as total_trx,
                AVG(attribution_score) as avg_score
            FROM attributions 
            WHERE campaign_id = ?
        """,
            [campaign_id],
        )
        return result.first() or {}
