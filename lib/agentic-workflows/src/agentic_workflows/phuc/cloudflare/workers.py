"""Cloudflare Workers management for PHUC platform."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum

import httpx


class WorkerStatus(Enum):
    ACTIVE = "active"
    DEPLOYING = "deploying"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class WorkerConfig:
    """Cloudflare Worker configuration."""

    account_id: str = field(default_factory=lambda: os.getenv("CF_ACCOUNT_ID", ""))
    api_token: str = field(default_factory=lambda: os.getenv("CF_API_TOKEN", ""))
    worker_name: str = "phuc-ai-v2-backend"

    def __post_init__(self):
        if not self.account_id:
            raise ValueError("CF_ACCOUNT_ID required")
        if not self.api_token:
            raise ValueError("CF_API_TOKEN required")


@dataclass
class WorkerMetrics:
    """Worker performance metrics."""

    requests: int = 0
    errors: int = 0
    cpu_time_ms: float = 0.0
    wall_time_ms: float = 0.0
    status: WorkerStatus = WorkerStatus.ACTIVE


class CloudflareWorkers:
    """Cloudflare Workers API client."""

    BASE_URL = "https://api.cloudflare.com/client/v4"

    def __init__(self, config: WorkerConfig):
        self.config = config
        self._client: httpx.AsyncClient | None = None

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_token}",
            "Content-Type": "application/json",
        }

    @property
    def base_url(self) -> str:
        return f"{self.BASE_URL}/accounts/{self.config.account_id}"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(headers=self.headers, timeout=30.0)
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def deploy(self, script: str, bindings: list[dict] = None) -> dict:
        """Deploy worker script."""
        client = await self._get_client()

        # Prepare multipart form
        files = {
            "script": ("worker.js", script, "application/javascript"),
        }

        if bindings:
            files["metadata"] = (
                "metadata.json",
                json.dumps({"bindings": bindings}),
                "application/json",
            )

        response = await client.put(
            f"{self.base_url}/workers/scripts/{self.config.worker_name}", files=files
        )
        response.raise_for_status()
        return response.json()

    async def get_worker(self) -> dict:
        """Get worker details."""
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/workers/scripts/{self.config.worker_name}")
        response.raise_for_status()
        return response.json()

    async def list_workers(self) -> list[dict]:
        """List all workers in account."""
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/workers/scripts")
        response.raise_for_status()
        return response.json().get("result", [])

    async def get_metrics(self, since: str = "-1h") -> WorkerMetrics:
        """Get worker analytics."""
        client = await self._get_client()

        query = """
        query {
          viewer {
            accounts(filter: {accountTag: "%s"}) {
              workersInvocationsAdaptive(
                filter: {scriptName: "%s"}
                limit: 1000
              ) {
                sum {
                  requests
                  errors
                }
                quantiles {
                  cpuTimeP50
                  durationP50
                }
              }
            }
          }
        }
        """ % (self.config.account_id, self.config.worker_name)

        response = await client.post(f"{self.BASE_URL}/graphql", json={"query": query})

        if response.status_code != 200:
            return WorkerMetrics()

        data = response.json()
        try:
            invocations = data["data"]["viewer"]["accounts"][0]["workersInvocationsAdaptive"][0]
            return WorkerMetrics(
                requests=invocations["sum"]["requests"],
                errors=invocations["sum"]["errors"],
                cpu_time_ms=invocations["quantiles"]["cpuTimeP50"],
                wall_time_ms=invocations["quantiles"]["durationP50"],
            )
        except (KeyError, IndexError):
            return WorkerMetrics()

    async def get_logs(self, limit: int = 100) -> list[dict]:
        """Get recent worker logs (requires tail session)."""
        # Note: Real-time logs require WebSocket connection
        # This returns recent logged events from analytics
        client = await self._get_client()
        response = await client.get(
            f"{self.base_url}/workers/scripts/{self.config.worker_name}/tail"
        )
        if response.status_code == 200:
            return response.json().get("result", [])[:limit]
        return []

    async def delete_worker(self) -> bool:
        """Delete worker."""
        client = await self._get_client()
        response = await client.delete(f"{self.base_url}/workers/scripts/{self.config.worker_name}")
        return response.status_code == 200


# Convenience function
async def get_worker_status(config: WorkerConfig = None) -> WorkerMetrics:
    """Quick status check for PHUC backend worker."""
    config = config or WorkerConfig()
    workers = CloudflareWorkers(config)
    try:
        return await workers.get_metrics()
    finally:
        await workers.close()
