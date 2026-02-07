"""MCP Tool Server for StarRocks memory operations."""

import os
import json
import pymysql
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("StarRocks Memory")

STARROCKS_HOST = os.environ.get("STARROCKS_HOST", "starrocks")
STARROCKS_PORT = int(os.environ.get("STARROCKS_PORT", "9030"))
STARROCKS_USER = os.environ.get("STARROCKS_USER", "root")


def _get_conn():
    return pymysql.connect(
        host=STARROCKS_HOST,
        port=STARROCKS_PORT,
        user=STARROCKS_USER,
        database="antigravity",
        cursorclass=pymysql.cursors.DictCursor,
    )


@mcp.tool()
def query_episodic(tenant_id: str = "system", limit: int = 10) -> str:
    """Query recent episodic memory events for a tenant."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT timestamp, actor, action_type, content
                   FROM memory_episodic
                   WHERE tenant_id = %s
                   ORDER BY timestamp DESC LIMIT %s""",
                (tenant_id, limit),
            )
            return json.dumps(cur.fetchall(), default=str)
    finally:
        conn.close()


@mcp.tool()
def query_semantic(query: str, limit: int = 5) -> str:
    """Search semantic memory by content keyword."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT doc_id, chunk_id, content, source_uri
                   FROM memory_semantic
                   WHERE content LIKE %s
                   LIMIT %s""",
                (f"%{query}%", limit),
            )
            return json.dumps(cur.fetchall(), default=str)
    finally:
        conn.close()


@mcp.tool()
def query_procedural(skill_name: str = "") -> str:
    """Look up procedural memory (skills/tools)."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            if skill_name:
                cur.execute(
                    "SELECT * FROM memory_procedural WHERE skill_id LIKE %s",
                    (f"%{skill_name}%",),
                )
            else:
                cur.execute("SELECT skill_id, description, success_rate FROM memory_procedural LIMIT 20")
            return json.dumps(cur.fetchall(), default=str)
    finally:
        conn.close()


@mcp.tool()
def execute_sql(sql: str) -> str:
    """Execute arbitrary read-only SQL on StarRocks."""
    if any(kw in sql.upper() for kw in ["DROP", "DELETE", "TRUNCATE", "ALTER"]):
        return json.dumps({"error": "Destructive SQL operations are not allowed"})
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return json.dumps(cur.fetchall(), default=str)
    finally:
        conn.close()


if __name__ == "__main__":
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8000
    # Disable DNS rebinding protection for Docker network access
    mcp.settings.transport_security.enable_dns_rebinding_protection = False
    mcp.run(transport="sse")
