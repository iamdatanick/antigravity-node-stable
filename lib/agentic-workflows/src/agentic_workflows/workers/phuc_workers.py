"""
PHUC Workers - Cloudflare Worker Wrappers
==========================================

Workers wrap MCP skill groups for simplified access.
Each worker maps to one or more deployed skills.

Integrates:
- agentic-workflows 5.0.0
- mcp 1.12.4
- pydantic 2.12.5
- opentelemetry 1.38.0

Workers:
- d1_worker: D1 database operations
- r2_worker: R2 object storage
- ai_worker: AI inference + Vectorize
- analytics_worker: Attribution + Campaign + Reporting
- security_worker: Injection defense + Scope validator + PII detector
- camara_worker: CAMARA telecom APIs

Location: C:\\Users\\NickV\\agentic-workflows\\agentic-workflows\\src\\agentic_workflows\\workers\\phuc_workers.py
"""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from abc import ABC, abstractmethod

# ═══════════════════════════════════════════════════════════════════════════════
# SDK IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

# MCP SDK
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Pydantic
from pydantic import BaseModel, Field

# OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    tracer = trace.get_tracer("phuc.workers")
except ImportError:
    tracer = None

# Agentic-workflows
from agentic_workflows.observability import MetricsCollector
from agentic_workflows.security import RateLimiter
from agentic_workflows.orchestration import CircuitBreaker

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MCP_ENDPOINT = "https://agentic-workflows-mcp.nick-9a6.workers.dev"
CAMARA_ENDPOINT = "https://mcp.camaramcp.com/sse"
DEPLOYMENT_VERSION = "d5b2201e-072a-4367-a7a5-099b3d0c9ca7"


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class WorkerType(Enum):
    """Worker types by domain"""
    D1 = "d1"
    R2 = "r2"
    AI = "ai"
    VECTORIZE = "vectorize"
    ANALYTICS = "analytics"
    SECURITY = "security"
    CAMARA = "camara"


class WorkerStatus(Enum):
    """Worker status"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    DISABLED = "disabled"


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class WorkerResult(BaseModel):
    """Worker execution result"""
    worker: str
    tool: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    retries: int = 0


class WorkerInfo(BaseModel):
    """Worker information"""
    name: str
    worker_type: str
    skills: List[str]
    tools: List[str]
    endpoint: str
    status: str = "idle"
    call_count: int = 0


class WorkerPoolStatus(BaseModel):
    """Worker pool status"""
    total_workers: int
    active_workers: int
    total_calls: int
    total_errors: int
    workers: List[WorkerInfo]


# ═══════════════════════════════════════════════════════════════════════════════
# BASE WORKER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CloudflareWorker(ABC):
    """
    Base worker that wraps MCP skills.
    Provides simplified interface to skill groups.
    """
    name: str
    worker_type: WorkerType
    skills: List[str]
    tools: List[str]
    endpoint: str = MCP_ENDPOINT
    description: str = ""
    active: bool = True
    _status: WorkerStatus = WorkerStatus.IDLE
    _call_count: int = 0
    _error_count: int = 0
    _rate_limiter: Optional[RateLimiter] = None
    _circuit_breaker: Optional[CircuitBreaker] = None
    _metrics: Optional[MetricsCollector] = None
    
    def __post_init__(self):
        self._rate_limiter = RateLimiter()
        self._circuit_breaker = CircuitBreaker(name=f"worker-{self.name}")
        self._metrics = MetricsCollector()
    
    def info(self) -> WorkerInfo:
        """Get worker info"""
        return WorkerInfo(
            name=self.name,
            worker_type=self.worker_type.value,
            skills=self.skills,
            tools=self.tools,
            endpoint=self.endpoint,
            status=self._status.value,
            call_count=self._call_count
        )
    
    async def execute(self, tool: str, args: Dict[str, Any] = None) -> WorkerResult:
        """Execute a tool from the worker's skills"""
        start_time = time.time()
        self._call_count += 1
        self._status = WorkerStatus.BUSY
        
        # Validate tool
        if tool not in self.tools:
            self._status = WorkerStatus.ERROR
            return WorkerResult(
                worker=self.name,
                tool=tool,
                success=False,
                error=f"Tool '{tool}' not available in {self.name}"
            )
        
        # Check rate limit
        if self._rate_limiter and not self._rate_limiter.allow(self.name):
            return WorkerResult(
                worker=self.name,
                tool=tool,
                success=False,
                error="Rate limit exceeded"
            )
        
        # Check circuit breaker
        if self._circuit_breaker and not self._circuit_breaker.allow():
            return WorkerResult(
                worker=self.name,
                tool=tool,
                success=False,
                error="Circuit breaker open"
            )
        
        # OpenTelemetry span
        span = None
        if tracer:
            span = tracer.start_span(f"worker.{self.name}.{tool}")
            span.set_attribute("worker.name", self.name)
            span.set_attribute("worker.type", self.worker_type.value)
            span.set_attribute("tool", tool)
        
        try:
            # Execute via MCP
            result = await self._call_mcp(tool, args or {})
            
            self._status = WorkerStatus.IDLE
            if span:
                span.set_status(Status(StatusCode.OK))
            
            # Record metrics
            if self._metrics:
                self._metrics.record_tool_call(tool, success=True)
            
            return WorkerResult(
                worker=self.name,
                tool=tool,
                success=True,
                result=result,
                duration_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            self._error_count += 1
            self._status = WorkerStatus.ERROR
            
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
            
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()
            
            return WorkerResult(
                worker=self.name,
                tool=tool,
                success=False,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
        finally:
            if span:
                span.end()
    
    async def _call_mcp(self, tool: str, args: Dict[str, Any]) -> Any:
        """Call MCP server"""
        if not MCP_AVAILABLE:
            return {"simulated": True, "tool": tool, "args": args}
        
        async with sse_client(self.endpoint) as (read, write):
            session = ClientSession(read, write)
            await session.initialize()
            result = await session.call_tool(tool, args)
            return result.content[0].text if result.content else None


# ═══════════════════════════════════════════════════════════════════════════════
# CONCRETE WORKERS (6)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class D1Worker(CloudflareWorker):
    """
    D1 Worker - Cloudflare D1 SQL database operations
    Skill: d1
    """
    name: str = "d1_worker"
    worker_type: WorkerType = WorkerType.D1
    skills: List[str] = field(default_factory=lambda: ["d1"])
    tools: List[str] = field(default_factory=lambda: [
        "d1_query", "d1_execute", "d1_batch", "d1_migrate", "d1_backup", "d1_restore"
    ])
    description: str = "Cloudflare D1 serverless SQL database"
    
    # Convenience methods
    async def query(self, sql: str, params: List = None) -> WorkerResult:
        return await self.execute("d1_query", {"sql": sql, "params": params or []})
    
    async def execute_sql(self, sql: str, params: List = None) -> WorkerResult:
        return await self.execute("d1_execute", {"sql": sql, "params": params or []})
    
    async def batch(self, statements: List[Dict]) -> WorkerResult:
        return await self.execute("d1_batch", {"statements": statements})
    
    async def backup(self) -> WorkerResult:
        return await self.execute("d1_backup", {})


@dataclass
class R2Worker(CloudflareWorker):
    """
    R2 Worker - Cloudflare R2 object storage
    Skill: r2
    """
    name: str = "r2_worker"
    worker_type: WorkerType = WorkerType.R2
    skills: List[str] = field(default_factory=lambda: ["r2"])
    tools: List[str] = field(default_factory=lambda: [
        "r2_get", "r2_put", "r2_list", "r2_delete", "r2_multipart", "r2_presign", "r2_copy"
    ])
    description: str = "Cloudflare R2 object storage"
    
    async def get(self, key: str) -> WorkerResult:
        return await self.execute("r2_get", {"key": key})
    
    async def put(self, key: str, value: Any, content_type: str = None) -> WorkerResult:
        return await self.execute("r2_put", {
            "key": key, "value": value, "content_type": content_type
        })
    
    async def list(self, prefix: str = "", limit: int = 100) -> WorkerResult:
        return await self.execute("r2_list", {"prefix": prefix, "limit": limit})
    
    async def delete(self, key: str) -> WorkerResult:
        return await self.execute("r2_delete", {"key": key})
    
    async def presign(self, key: str, expires_in: int = 3600) -> WorkerResult:
        return await self.execute("r2_presign", {"key": key, "expires_in": expires_in})


@dataclass
class AIWorker(CloudflareWorker):
    """
    AI Worker - Cloudflare Workers AI + Vectorize
    Skills: ai, vectorize
    """
    name: str = "ai_worker"
    worker_type: WorkerType = WorkerType.AI
    skills: List[str] = field(default_factory=lambda: ["ai", "vectorize"])
    tools: List[str] = field(default_factory=lambda: [
        # AI tools
        "ai_generate", "ai_embed", "ai_classify", "ai_summarize", "ai_translate", "ai_image", "ai_speech",
        # Vectorize tools
        "vectorize_insert", "vectorize_query", "vectorize_delete", "vectorize_upsert", "vectorize_info"
    ])
    description: str = "Cloudflare Workers AI inference and vector embeddings"
    
    async def generate(self, prompt: str, model: str = None, max_tokens: int = 1024) -> WorkerResult:
        return await self.execute("ai_generate", {
            "prompt": prompt, "model": model, "max_tokens": max_tokens
        })
    
    async def embed(self, text: str, model: str = None) -> WorkerResult:
        return await self.execute("ai_embed", {"text": text, "model": model})
    
    async def classify(self, text: str, labels: List[str]) -> WorkerResult:
        return await self.execute("ai_classify", {"text": text, "labels": labels})
    
    async def summarize(self, text: str, max_length: int = 200) -> WorkerResult:
        return await self.execute("ai_summarize", {"text": text, "max_length": max_length})
    
    async def vector_query(self, vector: List[float], top_k: int = 10) -> WorkerResult:
        return await self.execute("vectorize_query", {"vector": vector, "top_k": top_k})
    
    async def vector_insert(self, id: str, vector: List[float], metadata: Dict = None) -> WorkerResult:
        return await self.execute("vectorize_insert", {
            "id": id, "vector": vector, "metadata": metadata
        })


@dataclass
class AnalyticsWorker(CloudflareWorker):
    """
    Analytics Worker - Attribution, Campaign, Reporting
    Skills: attribution, campaign, reporting
    """
    name: str = "analytics_worker"
    worker_type: WorkerType = WorkerType.ANALYTICS
    skills: List[str] = field(default_factory=lambda: ["attribution", "campaign", "reporting"])
    tools: List[str] = field(default_factory=lambda: [
        # Attribution
        "attribution_track", "attribution_query", "attribution_report", "attribution_model", "attribution_window",
        # Campaign
        "campaign_create", "campaign_update", "campaign_analyze", "campaign_optimize", "campaign_budget",
        # Reporting
        "report_generate", "report_schedule", "report_export", "report_template", "report_share"
    ])
    description: str = "Analytics, attribution, and campaign management"
    
    async def track(self, event: str, properties: Dict = None) -> WorkerResult:
        return await self.execute("attribution_track", {
            "event": event, "properties": properties or {}
        })
    
    async def query_attribution(self, start_date: str, end_date: str, **filters) -> WorkerResult:
        return await self.execute("attribution_query", {
            "start_date": start_date, "end_date": end_date, **filters
        })
    
    async def create_campaign(self, name: str, config: Dict) -> WorkerResult:
        return await self.execute("campaign_create", {"name": name, "config": config})
    
    async def analyze_campaign(self, campaign_id: str) -> WorkerResult:
        return await self.execute("campaign_analyze", {"campaign_id": campaign_id})
    
    async def generate_report(self, report_type: str, params: Dict = None) -> WorkerResult:
        return await self.execute("report_generate", {
            "type": report_type, "params": params or {}
        })


@dataclass
class SecurityWorker(CloudflareWorker):
    """
    Security Worker - Injection defense, Scope validation, PII detection
    Skills: injection-defense, scope-validator, pii-detector
    """
    name: str = "security_worker"
    worker_type: WorkerType = WorkerType.SECURITY
    skills: List[str] = field(default_factory=lambda: [
        "injection-defense", "scope-validator", "pii-detector"
    ])
    tools: List[str] = field(default_factory=lambda: [
        # Injection defense
        "scan_prompt", "check_threat", "sanitize_input", "block_pattern", "audit_log",
        # Scope validator
        "validate_scope", "check_permissions", "enforce_policy", "grant_scope", "revoke_scope",
        # PII detector
        "detect_pii", "mask_pii", "audit_pii", "classify_data", "redact_document"
    ])
    description: str = "Security scanning, validation, and PII protection"
    
    async def scan_prompt(self, text: str, sensitivity: float = 0.8) -> WorkerResult:
        return await self.execute("scan_prompt", {"text": text, "sensitivity": sensitivity})
    
    async def check_threat(self, text: str) -> WorkerResult:
        return await self.execute("check_threat", {"text": text})
    
    async def validate_scope(self, scope: str, user_id: str = None) -> WorkerResult:
        return await self.execute("validate_scope", {"scope": scope, "user_id": user_id})
    
    async def detect_pii(self, text: str) -> WorkerResult:
        return await self.execute("detect_pii", {"text": text})
    
    async def mask_pii(self, text: str, mask_char: str = "*") -> WorkerResult:
        return await self.execute("mask_pii", {"text": text, "mask_char": mask_char})
    
    async def redact_document(self, content: str, pii_types: List[str] = None) -> WorkerResult:
        return await self.execute("redact_document", {
            "content": content, "pii_types": pii_types or ["ssn", "email", "phone"]
        })


@dataclass
class CAMARAWorker(CloudflareWorker):
    """
    CAMARA Worker - Telecom network APIs
    Connected to: https://mcp.camaramcp.com/sse
    """
    name: str = "camara_worker"
    worker_type: WorkerType = WorkerType.CAMARA
    skills: List[str] = field(default_factory=lambda: ["camara"])
    tools: List[str] = field(default_factory=lambda: [
        "checkSimSwap", "checkDeviceSwap", "checkDeviceLocation", 
        "checkRoamingStatus", "verifyKycMatch"
    ])
    endpoint: str = CAMARA_ENDPOINT
    description: str = "CAMARA telecom verification APIs"
    
    async def check_sim_swap(self, phone_number: str, operator: str = "telefonica", max_age: int = 24) -> WorkerResult:
        return await self.execute("checkSimSwap", {
            "phoneNumber": phone_number, "operator": operator, "maxAge": max_age
        })
    
    async def check_device_swap(self, phone_number: str, operator: str = "telefonica", max_age: int = 24) -> WorkerResult:
        return await self.execute("checkDeviceSwap", {
            "phoneNumber": phone_number, "operator": operator, "maxAge": max_age
        })
    
    async def check_location(self, phone_number: str, latitude: float, longitude: float, 
                             accuracy: float = 10, operator: str = "telefonica") -> WorkerResult:
        return await self.execute("checkDeviceLocation", {
            "phoneNumber": phone_number, "latitude": latitude, "longitude": longitude,
            "accuracy": accuracy, "operator": operator
        })
    
    async def check_roaming(self, phone_number: str, operator: str = "telefonica") -> WorkerResult:
        return await self.execute("checkRoamingStatus", {
            "phoneNumber": phone_number, "operator": operator
        })


# ═══════════════════════════════════════════════════════════════════════════════
# WORKER REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

# Singleton instances
D1_WORKER = D1Worker()
R2_WORKER = R2Worker()
AI_WORKER = AIWorker()
ANALYTICS_WORKER = AnalyticsWorker()
SECURITY_WORKER = SecurityWorker()
CAMARA_WORKER = CAMARAWorker()

WORKERS: Dict[str, CloudflareWorker] = {
    "d1_worker": D1_WORKER,
    "r2_worker": R2_WORKER,
    "ai_worker": AI_WORKER,
    "analytics_worker": ANALYTICS_WORKER,
    "security_worker": SECURITY_WORKER,
    "camara_worker": CAMARA_WORKER,
}


# ═══════════════════════════════════════════════════════════════════════════════
# WORKER POOL
# ═══════════════════════════════════════════════════════════════════════════════

class WorkerPool:
    """
    Manages all workers and routes requests.
    Provides unified interface to all Cloudflare workers.
    """
    
    def __init__(self):
        self.workers = WORKERS.copy()
        self._metrics = MetricsCollector()
        self._total_calls = 0
        self._total_errors = 0
    
    def get_worker(self, name: str) -> Optional[CloudflareWorker]:
        """Get worker by name"""
        return self.workers.get(name)
    
    def get_worker_for_skill(self, skill_name: str) -> Optional[CloudflareWorker]:
        """Find worker that handles a specific skill"""
        for worker in self.workers.values():
            if skill_name in worker.skills:
                return worker
        return None
    
    def get_worker_for_tool(self, tool: str) -> Optional[CloudflareWorker]:
        """Find worker that has a specific tool"""
        for worker in self.workers.values():
            if tool in worker.tools:
                return worker
        return None
    
    async def execute(self, worker_name: str, tool: str, args: Dict = None) -> WorkerResult:
        """Execute a worker tool"""
        self._total_calls += 1
        
        worker = self.workers.get(worker_name)
        if not worker:
            return WorkerResult(
                worker=worker_name,
                tool=tool,
                success=False,
                error=f"Worker '{worker_name}' not found"
            )
        
        if not worker.active:
            return WorkerResult(
                worker=worker_name,
                tool=tool,
                success=False,
                error=f"Worker '{worker_name}' is disabled"
            )
        
        result = await worker.execute(tool, args or {})
        if not result.success:
            self._total_errors += 1
        
        return result
    
    async def execute_by_tool(self, tool: str, args: Dict = None) -> WorkerResult:
        """Execute by finding the worker that has the tool"""
        worker = self.get_worker_for_tool(tool)
        if not worker:
            return WorkerResult(
                worker="unknown",
                tool=tool,
                success=False,
                error=f"No worker found for tool '{tool}'"
            )
        return await worker.execute(tool, args or {})
    
    async def execute_parallel(self, calls: List[Dict]) -> List[WorkerResult]:
        """Execute multiple worker calls in parallel"""
        tasks = [
            self.execute(call["worker"], call["tool"], call.get("args", {}))
            for call in calls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            r if isinstance(r, WorkerResult) else WorkerResult(
                worker="unknown", tool="unknown", success=False, error=str(r)
            )
            for r in results
        ]
    
    def status(self) -> WorkerPoolStatus:
        """Get pool status"""
        worker_infos = [w.info() for w in self.workers.values()]
        active = sum(1 for w in self.workers.values() if w.active)
        
        return WorkerPoolStatus(
            total_workers=len(self.workers),
            active_workers=active,
            total_calls=self._total_calls,
            total_errors=self._total_errors,
            workers=worker_infos
        )
    
    def list_workers(self) -> Dict[str, Dict]:
        """List all workers"""
        return {
            name: {
                "type": w.worker_type.value,
                "skills": w.skills,
                "tools": w.tools,
                "active": w.active,
                "description": w.description
            }
            for name, w in self.workers.items()
        }
    
    def list_all_tools(self) -> Dict[str, List[str]]:
        """List all tools by worker"""
        return {name: w.tools for name, w in self.workers.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "WorkerType",
    "WorkerStatus",
    # Models
    "WorkerResult",
    "WorkerInfo",
    "WorkerPoolStatus",
    # Base
    "CloudflareWorker",
    # Concrete workers
    "D1Worker",
    "R2Worker",
    "AIWorker",
    "AnalyticsWorker",
    "SecurityWorker",
    "CAMARAWorker",
    # Instances
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
