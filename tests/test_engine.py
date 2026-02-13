"""Tests for AsyncDAGEngine (CG-102)."""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestAsyncDAGEngine:
    """Unit tests for the orchestrator engine."""

    def test_engine_imports(self):
        """Engine module should be importable."""
        from src.orchestrator.engine import AsyncDAGEngine

        assert AsyncDAGEngine is not None

    def test_engine_init(self):
        """Engine should initialize with etcd and S3 config from env."""
        with patch.dict(
            os.environ,
            {
                "ETCD_HOST": "localhost",
                "ETCD_PORT": "2379",
                "S3_ENDPOINT_URL": "http://localhost:8000",
                "AWS_ACCESS_KEY_ID": "test",
                "AWS_SECRET_ACCESS_KEY": "test",
            },
        ):
            from src.orchestrator.engine import AsyncDAGEngine

            engine = AsyncDAGEngine()
            assert engine.etcd_host == "localhost"
            assert engine.etcd_port == 2379
            assert engine.s3_endpoint == "http://localhost:8000"

    @pytest.mark.asyncio
    async def test_submit_task(self):
        """Engine should accept task submissions."""
        with patch.dict(
            os.environ,
            {
                "ETCD_HOST": "localhost",
                "ETCD_PORT": "2379",
                "S3_ENDPOINT_URL": "http://localhost:8000",
                "AWS_ACCESS_KEY_ID": "test",
                "AWS_SECRET_ACCESS_KEY": "test",
            },
        ):
            from src.orchestrator.engine import AsyncDAGEngine

            engine = AsyncDAGEngine()
            # Mock etcd client
            engine._etcd = MagicMock()
            engine._etcd.put = MagicMock()
            task_id = await engine.submit_task(
                name="test-task",
                payload={"input": "hello"},
                tenant_id="tenant-a",
            )
            assert task_id is not None
            assert isinstance(task_id, str)

    @pytest.mark.asyncio
    async def test_get_task_status(self):
        """Engine should return task status from etcd."""
        with patch.dict(
            os.environ,
            {
                "ETCD_HOST": "localhost",
                "ETCD_PORT": "2379",
                "S3_ENDPOINT_URL": "http://localhost:8000",
                "AWS_ACCESS_KEY_ID": "test",
                "AWS_SECRET_ACCESS_KEY": "test",
            },
        ):
            from src.orchestrator.engine import AsyncDAGEngine

            engine = AsyncDAGEngine()
            engine._etcd = MagicMock()
            engine._etcd.get = MagicMock(return_value=(b'{"status":"running"}', None))
            status = await engine.get_task_status("task-123")
            assert status["status"] == "running"
