"""StarRocks specialist agent for OLAP analytics.

Handles real-time analytics and SQL queries on large datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import SpecialistAgent, SpecialistConfig, SpecialistCapability


@dataclass
class StarRocksConfig(SpecialistConfig):
    """StarRocks-specific configuration."""

    host: str = "localhost"
    port: int = 9030
    user: str = "root"
    password: str = ""
    database: str = "default"
    http_port: int = 8030


class StarRocksAgent(SpecialistAgent):
    """Specialist agent for StarRocks OLAP database.

    Capabilities:
    - SQL query execution
    - Real-time analytics
    - Data ingestion
    - Schema management
    """

    def __init__(self, config: StarRocksConfig | None = None, **kwargs):
        self.sr_config = config or StarRocksConfig()
        super().__init__(config=self.sr_config, **kwargs)

        self._connection = None

        self.register_handler("query", self._execute_query)
        self.register_handler("insert", self._insert_data)
        self.register_handler("create_table", self._create_table)
        self.register_handler("list_tables", self._list_tables)
        self.register_handler("describe_table", self._describe_table)
        self.register_handler("stream_load", self._stream_load)

    @property
    def capabilities(self) -> list[SpecialistCapability]:
        return [
            SpecialistCapability.OLAP_QUERY,
            SpecialistCapability.REAL_TIME_ANALYTICS,
            SpecialistCapability.SQL_QUERY,
        ]

    @property
    def service_name(self) -> str:
        return "StarRocks"

    async def _connect(self) -> None:
        """Connect to StarRocks."""
        try:
            import pymysql

            self._connection = pymysql.connect(
                host=self.sr_config.host,
                port=self.sr_config.port,
                user=self.sr_config.user,
                password=self.sr_config.password,
                database=self.sr_config.database,
                cursorclass=pymysql.cursors.DictCursor,
            )
        except ImportError:
            self.logger.warning("pymysql not installed")

    async def _disconnect(self) -> None:
        """Disconnect from StarRocks."""
        if self._connection:
            self._connection.close()
            self._connection = None

    async def _health_check(self) -> bool:
        """Check StarRocks health."""
        if self._connection is None:
            return False
        try:
            with self._connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            return True
        except Exception:
            return False

    async def _execute_query(
        self,
        sql: str,
        params: tuple | dict | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Execute a SQL query.

        Args:
            sql: SQL query.
            params: Query parameters.
            limit: Maximum rows.

        Returns:
            Query results.
        """
        if self._connection is None:
            return {"error": "Not connected"}

        try:
            with self._connection.cursor() as cursor:
                if limit and "LIMIT" not in sql.upper():
                    sql = f"{sql} LIMIT {limit}"
                cursor.execute(sql, params)
                rows = cursor.fetchall()

            return {
                "rows": rows,
                "row_count": len(rows),
                "columns": [desc[0] for desc in cursor.description] if cursor.description else [],
            }
        except Exception as e:
            return {"error": str(e)}

    async def _insert_data(
        self,
        table: str,
        data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Insert data into a table.

        Args:
            table: Table name.
            data: List of row dictionaries.

        Returns:
            Insert result.
        """
        if self._connection is None or not data:
            return {"error": "Not connected or no data"}

        try:
            columns = list(data[0].keys())
            placeholders = ", ".join(["%s"] * len(columns))
            col_str = ", ".join(columns)

            sql = f"INSERT INTO {table} ({col_str}) VALUES ({placeholders})"

            with self._connection.cursor() as cursor:
                for row in data:
                    values = [row.get(col) for col in columns]
                    cursor.execute(sql, values)

            self._connection.commit()

            return {"inserted": len(data), "table": table}
        except Exception as e:
            return {"error": str(e)}

    async def _create_table(
        self,
        name: str,
        columns: list[dict[str, str]],
        distributed_by: list[str] | None = None,
        properties: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create a new table.

        Args:
            name: Table name.
            columns: Column definitions.
            distributed_by: Distribution columns.
            properties: Table properties.

        Returns:
            Creation result.
        """
        if self._connection is None:
            return {"error": "Not connected"}

        col_defs = ", ".join([f"{c['name']} {c['type']}" for c in columns])
        dist_clause = ""
        if distributed_by:
            dist_clause = f"DISTRIBUTED BY HASH({', '.join(distributed_by)})"

        props_clause = ""
        if properties:
            props = ", ".join([f'"{k}" = "{v}"' for k, v in properties.items()])
            props_clause = f"PROPERTIES ({props})"

        sql = f"CREATE TABLE IF NOT EXISTS {name} ({col_defs}) {dist_clause} {props_clause}"

        try:
            with self._connection.cursor() as cursor:
                cursor.execute(sql)
            self._connection.commit()
            return {"table": name, "created": True}
        except Exception as e:
            return {"table": name, "created": False, "error": str(e)}

    async def _list_tables(self, database: str | None = None) -> list[str]:
        """List all tables."""
        if self._connection is None:
            return []

        db = database or self.sr_config.database

        with self._connection.cursor() as cursor:
            cursor.execute(f"SHOW TABLES FROM {db}")
            return [row[list(row.keys())[0]] for row in cursor.fetchall()]

    async def _describe_table(self, table: str) -> dict[str, Any]:
        """Get table schema."""
        if self._connection is None:
            return {"error": "Not connected"}

        with self._connection.cursor() as cursor:
            cursor.execute(f"DESCRIBE {table}")
            columns = cursor.fetchall()

        return {
            "table": table,
            "columns": columns,
        }

    async def _stream_load(
        self,
        table: str,
        data: str | bytes,
        format: str = "json",
        columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Stream load data via HTTP.

        Args:
            table: Target table.
            data: Data to load.
            format: Data format (json, csv).
            columns: Column mapping.

        Returns:
            Load result.
        """
        import aiohttp

        url = f"http://{self.sr_config.host}:{self.sr_config.http_port}/api/{self.sr_config.database}/{table}/_stream_load"

        headers = {
            "Content-Type": "application/json" if format == "json" else "text/csv",
            "format": format,
        }

        if columns:
            headers["columns"] = ", ".join(columns)

        auth = aiohttp.BasicAuth(self.sr_config.user, self.sr_config.password)

        async with aiohttp.ClientSession() as session:
            async with session.put(url, data=data, headers=headers, auth=auth) as resp:
                result = await resp.json()
                return result
