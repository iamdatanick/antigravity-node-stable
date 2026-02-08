"""StarRocks memory push — tenant-partitioned context to 3-layer memory schema."""

import hashlib
import itertools
import logging
import os
import re
import threading
import time

import pymysql
from dbutils.pooled_db import PooledDB

from workflows.telemetry import get_tracer

logger = logging.getLogger("antigravity.memory")
tracer = get_tracer("antigravity.memory")

STARROCKS_HOST = os.environ.get("STARROCKS_HOST", "starrocks")
STARROCKS_PORT = int(os.environ.get("STARROCKS_PORT", "9030"))
STARROCKS_USER = os.environ.get("STARROCKS_USER", "root")
STARROCKS_PASSWORD = os.environ.get("STARROCKS_PASSWORD", "")
STARROCKS_DB = os.environ.get("STARROCKS_DB", "antigravity")

_event_counter = itertools.count(1)
_pool = None
_pool_lock = threading.Lock()


def _get_pool():
    """Get or create connection pool for StarRocks FE."""
    global _pool
    if _pool is None:
        with _pool_lock:
            # Double-check locking pattern
            if _pool is None:
                pool_kwargs = {
                    "creator": pymysql,
                    "maxconnections": 10,
                    "mincached": 2,
                    "maxcached": 5,
                    "blocking": True,
                    "host": STARROCKS_HOST,
                    "port": STARROCKS_PORT,
                    "user": STARROCKS_USER,
                    "database": STARROCKS_DB,
                    "cursorclass": pymysql.cursors.DictCursor,
                }
                # Only add password if it's non-empty
                if STARROCKS_PASSWORD:
                    pool_kwargs["password"] = STARROCKS_PASSWORD
                _pool = PooledDB(**pool_kwargs)
    return _pool


def _get_conn():
    """Get a pooled connection to StarRocks FE."""
    return _get_pool().connection()


def push_episodic(
    tenant_id: str,
    session_id: str,
    actor: str,
    action_type: str,
    content: str,
):
    """Write an event to episodic memory."""
    with tracer.start_as_current_span(
        "memory.push_episodic", attributes={"tenant_id": tenant_id, "action_type": action_type}
    ):
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
        normalized = re.sub(r"/\*.*?\*/", " ", normalized, flags=re.DOTALL)
        # Remove -- style comments
        normalized = re.sub(r"--[^\n]*", " ", normalized)

        forbidden = [
            "DROP",
            "DELETE",
            "INSERT",
            "UPDATE",
            "ALTER",
            "CREATE",
            "TRUNCATE",
            "GRANT",
            "REVOKE",
            "INTO OUTFILE",
            "INTO DUMPFILE",
            "LOAD",
            "SET",
            "EXEC",
        ]
        for keyword in forbidden:
            # Check for keyword as a standalone word (not part of column names)
            # Use word boundaries and handle various whitespace characters
            pattern = r"\b" + keyword + r"\b"
            if re.search(pattern, normalized):
                raise ValueError(f"Forbidden SQL keyword: {keyword}")

        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(normalized)
                return cur.fetchall()
        finally:
            conn.close()
