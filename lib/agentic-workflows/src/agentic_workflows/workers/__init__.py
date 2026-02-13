"""
Workers module for Agentic Workflows v5.0.

Workers wrap MCP skill groups for simplified access to deployed skills.

Components:
- WorkerPool: Manages all workers
- CloudflareWorker: Base worker class wrapping MCP skills
- D1_WORKER, R2_WORKER, AI_WORKER, ANALYTICS_WORKER, SECURITY_WORKER, CAMARA_WORKER
"""

from agentic_workflows.workers.phuc_workers import (
    AI_WORKER,
    ANALYTICS_WORKER,
    CAMARA_ENDPOINT,
    CAMARA_WORKER,
    # Pre-configured worker instances
    D1_WORKER,
    # Config
    MCP_ENDPOINT,
    R2_WORKER,
    SECURITY_WORKER,
    # Registry
    WORKERS,
    AIWorker,
    AnalyticsWorker,
    CAMARAWorker,
    # Base class
    CloudflareWorker,
    # Concrete worker classes
    D1Worker,
    R2Worker,
    SecurityWorker,
    WorkerInfo,
    # Pool
    WorkerPool,
    WorkerPoolStatus,
    # Models
    WorkerResult,
    WorkerStatus,
    # Enums
    WorkerType,
)

__all__ = [
    # Enums
    "WorkerType",
    "WorkerStatus",
    # Models
    "WorkerResult",
    "WorkerInfo",
    "WorkerPoolStatus",
    # Base class
    "CloudflareWorker",
    # Concrete worker classes
    "D1Worker",
    "R2Worker",
    "AIWorker",
    "AnalyticsWorker",
    "SecurityWorker",
    "CAMARAWorker",
    # Pre-configured worker instances
    "D1_WORKER",
    "R2_WORKER",
    "AI_WORKER",
    "ANALYTICS_WORKER",
    "SECURITY_WORKER",
    "CAMARA_WORKER",
    # Registry
    "WORKERS",
    # Pool
    "WorkerPool",
    # Config
    "MCP_ENDPOINT",
    "CAMARA_ENDPOINT",
]

__version__ = "5.0.0"
