"""
Workers module for Agentic Workflows v5.0.

Workers wrap MCP skill groups for simplified access to deployed skills.

Components:
- WorkerPool: Manages all workers
- CloudflareWorker: Base worker class wrapping MCP skills
- D1_WORKER, R2_WORKER, AI_WORKER, ANALYTICS_WORKER, SECURITY_WORKER, CAMARA_WORKER
"""

from agentic_workflows.workers.phuc_workers import (
    # Enums
    WorkerType,
    WorkerStatus,
    # Models
    WorkerResult,
    WorkerInfo,
    WorkerPoolStatus,
    # Base class
    CloudflareWorker,
    # Concrete worker classes
    D1Worker,
    R2Worker,
    AIWorker,
    AnalyticsWorker,
    SecurityWorker,
    CAMARAWorker,
    # Pre-configured worker instances
    D1_WORKER,
    R2_WORKER,
    AI_WORKER,
    ANALYTICS_WORKER,
    SECURITY_WORKER,
    CAMARA_WORKER,
    # Registry
    WORKERS,
    # Pool
    WorkerPool,
    # Config
    MCP_ENDPOINT,
    CAMARA_ENDPOINT,
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
