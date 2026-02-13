#!/usr/bin/env python3
"""
PHUC Platform - Cloudflare Worker Integration

Workers wrap MCP skill groups for simplified access to deployed skills.

Integrates:
- mcp SDK (1.12.4) - Model Context Protocol for skill invocation
- httpx (0.28.1) - Async HTTP for Worker API calls
- pydantic (2.12.5) - Data validation
- opentelemetry (1.38.0) - Distributed tracing

Workers:
    d1_worker: D1 database operations
    r2_worker: R2 object storage
    ai_worker: AI inference + Vectorize embeddings
    analytics_worker: Attribution, Campaign, Reporting
    security_worker: Injection defense, Scope validation, PII detection
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json

from pydantic import BaseModel, Field
import httpx
from opentelemetry import trace

# MCP imports
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Tracer
tracer = trace.get_tracer("phuc.workers")

# MCP Server Endpoint
MCP_ENDPOINT = "https://agentic-workflows-mcp.nick-9a6.workers.dev/mcp/sse"
WORKER_API = "https://agentic-workflows-mcp.nick-9a6.workers.dev"


class WorkerType(str, Enum):
    """Types of Cloudflare Workers."""
    D1 = "d1"
    R2 = "r2"
    AI = "ai"
    VECTORIZE = "vectorize"
    ANALYTICS = "analytics"
    SECURITY = "security"


class WorkerStatus(str, Enum):
    """Worker execution status."""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


class WorkerConfig(BaseModel):
    """Configuration for a Worker."""
    name: str
    worker_type: WorkerType
    skills: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    endpoint: str = MCP_ENDPOINT
    timeout: int = 30


class WorkerResult(BaseModel):
    """Result from a Worker execution."""
    worker: str
    tool: str
    status: WorkerStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


@dataclass
class Worker:
    """Cloudflare Worker that wraps MCP skills."""
    config: WorkerConfig
    _http_client: Optional[httpx.AsyncClient] = None
    _executions: List[WorkerResult] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def worker_type(self) -> WorkerType:
        return self.config.worker_type

    @property
    def skills(self) -> List[str]:
        return self.config.skills

    @property
    def tools(self) -> List[str]:
        return self.config.tools

    async def execute(self, tool: str, args: Optional[Dict[str, Any]] = None) -> WorkerResult:
        """Execute a tool via MCP."""
        start_time = datetime.now()
        args = args or {}

        with tracer.start_as_current_span(f"worker.{self.name}.{tool}") as span:
            span.set_attribute("worker.name", self.name)
            span.set_attribute("worker.type", self.worker_type.value)
            span.set_attribute("tool.name", tool)

            result = WorkerResult(
                worker=self.name,
                tool=tool,
                status=WorkerStatus.RUNNING
            )

            # Validate tool
            if tool not in self.tools:
                result.status = WorkerStatus.ERROR
                result.error = f"Tool '{tool}' not available in worker '{self.name}'. Available: {self.tools}"
                result.duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                self._executions.append(result)
                return result

            if not MCP_AVAILABLE:
                # Fallback to HTTP API
                return await self._execute_http(tool, args, result, start_time, span)

            try:
                async with sse_client(self.config.endpoint) as (read, write):
                    session = ClientSession(read, write)
                    await session.initialize()

                    mcp_result = await session.call_tool(f"skill_{tool}", args)

                    result.result = mcp_result.content[0].text if mcp_result.content else None
                    result.status = WorkerStatus.SUCCESS
                    result.duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            except Exception as e:
                span.record_exception(e)
                result.status = WorkerStatus.ERROR
                result.error = str(e)
                result.duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            self._executions.append(result)
            return result

    async def _execute_http(
        self,
        tool: str,
        args: Dict[str, Any],
        result: WorkerResult,
        start_time: datetime,
        span: Any
    ) -> WorkerResult:
        """Execute via HTTP API when MCP is not available."""
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    f"{WORKER_API}/api/tools/{tool}",
                    json=args,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    result.result = response.json()
                    result.status = WorkerStatus.SUCCESS
                else:
                    result.status = WorkerStatus.ERROR
                    result.error = f"HTTP {response.status_code}: {response.text}"

        except Exception as e:
            span.record_exception(e)
            result.status = WorkerStatus.ERROR
            result.error = str(e)

        result.duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        self._executions.append(result)
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self._executions:
            return {"total": 0}

        successful = [e for e in self._executions if e.status == WorkerStatus.SUCCESS]
        failed = [e for e in self._executions if e.status == WorkerStatus.ERROR]

        return {
            "worker": self.name,
            "total_executions": len(self._executions),
            "successful": len(successful),
            "failed": len(failed),
            "avg_duration_ms": sum(e.duration_ms for e in self._executions) / len(self._executions),
            "tools_used": list(set(e.tool for e in self._executions))
        }


# Pre-configured Workers
D1_WORKER = Worker(config=WorkerConfig(
    name="d1_worker",
    worker_type=WorkerType.D1,
    skills=["d1"],
    tools=[
        "d1_query",
        "d1_execute",
        "d1_batch",
        "d1_migrate",
        "d1_backup",
        "d1_schema"
    ]
))

R2_WORKER = Worker(config=WorkerConfig(
    name="r2_worker",
    worker_type=WorkerType.R2,
    skills=["r2"],
    tools=[
        "r2_get",
        "r2_put",
        "r2_list",
        "r2_delete",
        "r2_presign",
        "r2_copy"
    ]
))

AI_WORKER = Worker(config=WorkerConfig(
    name="ai_worker",
    worker_type=WorkerType.AI,
    skills=["ai", "vectorize"],
    tools=[
        "ai_generate",
        "ai_embed",
        "ai_classify",
        "ai_summarize",
        "ai_translate",
        "ai_image",
        "vectorize_insert",
        "vectorize_query",
        "vectorize_delete",
        "vectorize_upsert"
    ]
))

ANALYTICS_WORKER = Worker(config=WorkerConfig(
    name="analytics_worker",
    worker_type=WorkerType.ANALYTICS,
    skills=["attribution", "campaign", "reporting"],
    tools=[
        "attribution_track",
        "attribution_query",
        "attribution_report",
        "attribution_model",
        "campaign_create",
        "campaign_update",
        "campaign_analyze",
        "campaign_optimize",
        "report_generate",
        "report_schedule",
        "report_export",
        "dashboard_create"
    ]
))

SECURITY_WORKER = Worker(config=WorkerConfig(
    name="security_worker",
    worker_type=WorkerType.SECURITY,
    skills=["injection-defense", "scope-validator", "pii-detector"],
    tools=[
        "scan_prompt",
        "check_threat",
        "sanitize_input",
        "block_pattern",
        "validate_scope",
        "check_permissions",
        "enforce_policy",
        "audit_access",
        "detect_pii",
        "mask_pii",
        "audit_pii",
        "classify_data"
    ]
))

# Worker registry
WORKERS: Dict[str, Worker] = {
    "d1_worker": D1_WORKER,
    "r2_worker": R2_WORKER,
    "ai_worker": AI_WORKER,
    "analytics_worker": ANALYTICS_WORKER,
    "security_worker": SECURITY_WORKER,
}


class WorkerPool:
    """
    Manages all Cloudflare Workers.

    Provides:
    - Worker discovery and listing
    - Tool execution routing
    - Parallel execution
    - Statistics aggregation
    """

    def __init__(self, endpoint: str = MCP_ENDPOINT):
        self.endpoint = endpoint
        self.workers = WORKERS.copy()

    def get_worker(self, name: str) -> Optional[Worker]:
        """Get a worker by name."""
        return self.workers.get(name)

    def get_worker_for_skill(self, skill: str) -> Optional[Worker]:
        """Find the worker that handles a skill."""
        for worker in self.workers.values():
            if skill in worker.skills:
                return worker
        return None

    def get_worker_for_tool(self, tool: str) -> Optional[Worker]:
        """Find the worker that handles a tool."""
        for worker in self.workers.values():
            if tool in worker.tools:
                return worker
        return None

    async def execute(self, worker_name: str, tool: str, args: Optional[Dict[str, Any]] = None) -> WorkerResult:
        """Execute a tool on a specific worker."""
        worker = self.get_worker(worker_name)
        if not worker:
            return WorkerResult(
                worker=worker_name,
                tool=tool,
                status=WorkerStatus.ERROR,
                error=f"Worker '{worker_name}' not found"
            )
        return await worker.execute(tool, args)

    async def execute_tool(self, tool: str, args: Optional[Dict[str, Any]] = None) -> WorkerResult:
        """Execute a tool, automatically routing to the correct worker."""
        worker = self.get_worker_for_tool(tool)
        if not worker:
            return WorkerResult(
                worker="unknown",
                tool=tool,
                status=WorkerStatus.ERROR,
                error=f"No worker found for tool '{tool}'"
            )
        return await worker.execute(tool, args)

    async def execute_parallel(self, calls: List[Dict[str, Any]]) -> List[WorkerResult]:
        """
        Execute multiple tools in parallel.

        Args:
            calls: List of {"worker": str, "tool": str, "args": dict}
        """
        tasks = [
            self.execute(
                call.get("worker", ""),
                call.get("tool", ""),
                call.get("args")
            )
            for call in calls
        ]
        return await asyncio.gather(*tasks)

    def list_workers(self) -> List[Dict[str, Any]]:
        """List all workers with their configurations."""
        return [
            {
                "name": w.name,
                "type": w.worker_type.value,
                "skills": w.skills,
                "tools": w.tools
            }
            for w in self.workers.values()
        ]

    def list_all_tools(self) -> List[str]:
        """List all available tools across all workers."""
        tools = []
        for worker in self.workers.values():
            tools.extend(worker.tools)
        return sorted(set(tools))

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics for all workers."""
        stats = {
            "total_workers": len(self.workers),
            "total_tools": len(self.list_all_tools()),
            "workers": {}
        }

        for name, worker in self.workers.items():
            stats["workers"][name] = worker.get_stats()

        return stats


# Singleton pool
_pool: Optional[WorkerPool] = None


def get_worker_pool(endpoint: str = MCP_ENDPOINT) -> WorkerPool:
    """Get or create the worker pool singleton."""
    global _pool
    if _pool is None:
        _pool = WorkerPool(endpoint=endpoint)
    return _pool


async def execute_tool(tool: str, args: Optional[Dict[str, Any]] = None) -> WorkerResult:
    """Convenience function to execute a tool."""
    return await get_worker_pool().execute_tool(tool, args)


# Exports
__all__ = [
    "WorkerType",
    "WorkerStatus",
    "WorkerConfig",
    "WorkerResult",
    "Worker",
    "WorkerPool",
    "WORKERS",
    "D1_WORKER",
    "R2_WORKER",
    "AI_WORKER",
    "ANALYTICS_WORKER",
    "SECURITY_WORKER",
    "get_worker_pool",
    "execute_tool",
]
