"""AsyncDAGEngine — Lightweight task orchestrator for Antigravity Node v14.1.

Replaces Argo Workflows (BLK-028) with a Python-native DAG engine.
Uses etcd3 for distributed locking/state and aioboto3 for S3 artifact persistence.

CG-102: Core orchestration engine.
"""

import json
import logging
import os
import uuid
from datetime import UTC, datetime

logger = logging.getLogger("antigravity.engine")


class AsyncDAGEngine:
    """Async DAG-based task engine backed by etcd3 + S3."""

    def __init__(self):
        self.etcd_host = os.environ.get("ETCD_HOST", "etcd")
        self.etcd_port = int(os.environ.get("ETCD_PORT", "2379"))
        self.s3_endpoint = os.environ.get("S3_ENDPOINT_URL", "http://ceph-demo:8000")
        self.s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID", "antigravity")
        self.s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "antigravity_secret")
        self.s3_region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self._etcd = None
        self._s3_session = None
        logger.info(
            "AsyncDAGEngine initialized (etcd=%s:%d, s3=%s)",
            self.etcd_host,
            self.etcd_port,
            self.s3_endpoint,
        )

    def _ensure_etcd(self):
        """Lazy-connect to etcd."""
        if self._etcd is None:
            import etcd3

            self._etcd = etcd3.client(
                host=self.etcd_host,
                port=self.etcd_port,
            )
        return self._etcd

    async def _ensure_s3(self):
        """Lazy-create async S3 session for Ceph RGW."""
        if self._s3_session is None:
            import aioboto3

            self._s3_session = aioboto3.Session()
        return self._s3_session

    async def submit_task(self, name: str, payload: dict, tenant_id: str = "system") -> str:
        """Submit a task to the DAG engine.

        Stores task metadata in etcd, returns task ID.
        """
        task_id = str(uuid.uuid4())
        task_record = {
            "id": task_id,
            "name": name,
            "tenant_id": tenant_id,
            "payload": payload,
            "status": "pending",
            "created_at": datetime.now(UTC).isoformat(),
        }
        etcd = self._ensure_etcd()
        key = f"/antigravity/tasks/{tenant_id}/{task_id}"
        etcd.put(key, json.dumps(task_record))
        logger.info("Task submitted: %s (%s) for tenant %s", task_id, name, tenant_id)
        return task_id

    async def get_task_status(self, task_id: str, tenant_id: str = "system") -> dict:
        """Get task status from etcd."""
        etcd = self._ensure_etcd()
        key = f"/antigravity/tasks/{tenant_id}/{task_id}"
        value, _ = etcd.get(key)
        if value is None:
            return {"status": "not_found", "id": task_id}
        return json.loads(value)

    async def update_task_status(self, task_id: str, status: str, tenant_id: str = "system"):
        """Update task status in etcd."""
        record = await self.get_task_status(task_id, tenant_id)
        if record.get("status") == "not_found":
            raise ValueError(f"Task {task_id} not found")
        record["status"] = status
        record["updated_at"] = datetime.now(UTC).isoformat()
        etcd = self._ensure_etcd()
        key = f"/antigravity/tasks/{tenant_id}/{task_id}"
        etcd.put(key, json.dumps(record))

    async def store_artifact(self, task_id: str, key: str, data: bytes, tenant_id: str = "system"):
        """Store a task artifact in Ceph S3 via aioboto3."""
        session = await self._ensure_s3()
        bucket = f"{tenant_id}"
        s3_key = f"artifacts/{task_id}/{key}"
        async with session.client(
            "s3",
            endpoint_url=self.s3_endpoint,
            aws_access_key_id=self.s3_access_key,
            aws_secret_access_key=self.s3_secret_key,
            region_name=self.s3_region,
        ) as client:
            await client.put_object(Bucket=bucket, Key=s3_key, Body=data)
        logger.info("Artifact stored: s3://%s/%s (%d bytes)", bucket, s3_key, len(data))

    async def acquire_lock(self, lock_name: str, ttl: int = 30):
        """Acquire a distributed lock via etcd lease."""
        etcd = self._ensure_etcd()
        lease = etcd.lease(ttl)
        key = f"/antigravity/locks/{lock_name}"
        success, _ = etcd.transaction(
            compare=[etcd.transactions.create(key) == 0],
            success=[etcd.transactions.put(key, "locked", lease)],
            failure=[],
        )
        if success:
            logger.info("Lock acquired: %s (TTL=%ds)", lock_name, ttl)
            return lease
        logger.warning("Lock contention: %s", lock_name)
        return None

    async def release_lock(self, lock_name: str, lease):
        """Release a distributed lock."""
        if lease:
            lease.revoke()
            logger.info("Lock released: %s", lock_name)

    async def list_tasks(self, tenant_id: str = "system") -> list[dict]:
        """List all tasks for a tenant."""
        etcd = self._ensure_etcd()
        prefix = f"/antigravity/tasks/{tenant_id}/"
        tasks = []
        for value, _metadata in etcd.get_prefix(prefix):
            tasks.append(json.loads(value))
        return tasks
