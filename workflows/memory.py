"""StarRocks memory push — tenant-partitioned context to 3-layer memory schema."""

import os
import time
import logging
import itertools
import re
import hashlib
import pymysql
from workflows.telemetry import get_tracer

logger = logging.getLogger("antigravity.memory")
tracer = get_tracer("antigravity.memory")

STARROCKS_HOST = os.environ.get("STARROCKS_HOST", "starrocks")
STARROCKS_PORT = int(os.environ.get("STARROCKS_PORT", "9030"))
STARROCKS_USER = os.environ.get("STARROCKS_USER", "root")
STARROCKS_DB = os.environ.get("STARROCKS_DB", "antigravity")

_event_counter = itertools.count(1)


def _get_conn():
    """Get a connection to StarRocks FE."""
    return pymysql.connect(
        host=STARROCKS_HOST,
        port=STARROCKS_PORT,
        user=STARROCKS_USER,
        database=STARROCKS_DB,
        cursorclass=pymysql.cursors.DictCursor,
    )


def push_episodic(
    tenant_id: str,
    session_id: str,
    actor: str,
    action_type: str,
    content: str,
):
    """Write an event to episodic memory."""
    with tracer.start_as_current_span("memory.push_episodic", attributes={"tenant_id": tenant_id, "action_type": action_type}):
        count = next(_event_counter)
        event_id = int(time.time() * 1000) * 1000 + count

        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO memory_episodic
                       (event_id, tenant_id, timestamp, session_id, actor, action_type, content)
                       VALUES (%s, %s, NOW(), %s, %s, %s, %s)""",
                    (event_id, tenant_id, session_id, actor, action_type, content),
                )
            conn.commit()
            logger.info(f"Episodic memory saved: {action_type} by {actor} (tenant={tenant_id})")
        finally:
            conn.close()


def push_semantic(
    doc_id: str,
    tenant_id: str,
    chunk_id: int,
    content: str,
    source_uri: str,
):
    """Write a knowledge chunk to semantic memory."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO memory_semantic
                   (doc_id, tenant_id, chunk_id, content, source_uri)
                   VALUES (%s, %s, %s, %s, %s)""",
                (doc_id, tenant_id, chunk_id, content, source_uri),
            )
        conn.commit()
        logger.info(f"Semantic memory saved: doc={doc_id} chunk={chunk_id}")
    finally:
        conn.close()


def recall_experience(goal: str, tenant_id: str, limit: int = 10) -> list:
    """Query episodic memory BEFORE generating a new plan."""
    with tracer.start_as_current_span("memory.recall_experience", attributes={"tenant_id": tenant_id, "limit": limit}):
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT action_type, content, timestamp
                       FROM memory_episodic
                       WHERE tenant_id = %s
                       ORDER BY timestamp DESC
                       LIMIT %s""",
                    (tenant_id, limit),
                )
                return cur.fetchall()
        finally:
            conn.close()


def query(sql: str) -> list:
    """Execute read-only SQL on StarRocks memory tables."""
    # Create a hash of the SQL for tracing (to avoid PII exposure)
    sql_hash = hashlib.sha256(sql.encode()).hexdigest()[:16]
    with tracer.start_as_current_span("memory.query", attributes={"sql_hash": sql_hash, "sql_length": len(sql)}):
        # Allow-list: only SELECT queries permitted
        normalized = sql.strip().upper()
        if not normalized.startswith("SELECT"):
            raise ValueError("Only SELECT queries are permitted")
        
        # Remove SQL comments before checking for forbidden keywords
        # Remove single-line comments (--) and multi-line comments (/* */)
        # Note: This handles simple comment patterns but may not catch all edge cases
        # Remove /* */ style comments
        normalized = re.sub(r'/\*.*?\*/', ' ', normalized, flags=re.DOTALL)
        # Remove -- style comments
        normalized = re.sub(r'--[^\n]*', ' ', normalized)
        
        forbidden = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE", "GRANT", "REVOKE"]
        for keyword in forbidden:
            # Check for keyword as a standalone word (not part of column names)
            # Use word boundaries and handle various whitespace characters
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, normalized):
                raise ValueError(f"Forbidden SQL keyword: {keyword}")
        
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetchall()
        finally:
            conn.close()
