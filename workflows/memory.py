"""StarRocks memory push — tenant-partitioned context to 3-layer memory schema."""

import os
import time
import logging
import itertools
import pymysql

logger = logging.getLogger("antigravity.memory")

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
    # Allow-list: only SELECT queries permitted
    normalized = sql.strip().upper()
    if not normalized.startswith("SELECT"):
        raise ValueError("Only SELECT queries are permitted")
    
    forbidden = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE", "GRANT", "REVOKE"]
    for keyword in forbidden:
        # Check for keyword as a standalone word (not part of column names)
        if f" {keyword} " in f" {normalized} ":
            raise ValueError(f"Forbidden SQL keyword: {keyword}")
    
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()
    finally:
        conn.close()
