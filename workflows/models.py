"""Pydantic request/response models for Antigravity Node v13.0 A2A endpoints."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


# Constants for validation limits
MAX_GOAL_LENGTH = 10000  # Maximum characters for task goal (balance between flexibility and DoS protection)
MAX_CONTEXT_LENGTH = 50000  # Maximum characters for task context (allows detailed context without unbounded payloads)


# --- /task endpoint ---
class TaskRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=MAX_GOAL_LENGTH, description="The goal or task to accomplish")
    context: Optional[str] = Field(default="", max_length=MAX_CONTEXT_LENGTH, description="Additional context for the task")
    session_id: Optional[str] = Field(default=None, description="Session ID for continuity; auto-generated if omitted")


class TaskResponse(BaseModel):
    status: str
    session_id: str
    tenant_id: str
    history_count: int


# --- /handoff endpoint ---
class HandoffRequest(BaseModel):
    target_agent: str = Field(..., min_length=1, description="Target agent identifier for handoff")
    payload: Optional[Dict[str, Any]] = Field(default={}, description="Payload to pass to the target agent")


class HandoffResponse(BaseModel):
    status: str
    target: str


# --- /webhook endpoint ---
class WebhookPayload(BaseModel):
    task_id: str = Field(default="unknown", description="Argo workflow task ID")
    status: str = Field(default="unknown", description="Workflow status (Succeeded, Failed, etc.)")
    message: Optional[str] = Field(default="", description="Status message or error details")


class WebhookResponse(BaseModel):
    ack: bool


# --- /v1/chat/completions endpoint ---
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1, description="List of chat messages")
    model: Optional[str] = Field(default=None, description="Model to use; defaults to GOOSE_MODEL env var or gpt-4o")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=128000, description="Maximum tokens to generate")


# --- /upload endpoint ---
class UploadResponse(BaseModel):
    status: str
    key: str
    size: int


# --- /health endpoint ---
class HealthCheck(BaseModel):
    name: str
    healthy: bool
    error: Optional[str] = None


class HealthLevel(BaseModel):
    level: str
    name: str
    checks: List[HealthCheck]


class HealthResponse(BaseModel):
    status: str
    levels: List[HealthLevel]


# --- /tools endpoint ---
class ToolInfo(BaseModel):
    name: str
    server: str
    description: Optional[str] = None
    status: Optional[str] = None
    transport: Optional[str] = None
    url: Optional[str] = None


class ToolsResponse(BaseModel):
    tools: List[ToolInfo]
    total: int


# --- /capabilities endpoint ---
class CapabilitiesResponse(BaseModel):
    node: str
    protocols: List[str]
    endpoints: Dict[str, str]
    mcp_servers: Dict[str, Any]
    memory: Dict[str, str]
    budget: Dict[str, Any]


# --- /v1/inference endpoint ---
class InferenceRequest(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=256, description="Name of the model deployed in OVMS")
    input_data: Dict[str, Any] = Field(..., description="Mapping of input tensor names to values (e.g. {\"input\": [[1.0, 2.0]]})")


class InferenceResponse(BaseModel):
    status: str
    model: Optional[str] = None
    outputs: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    latency_ms: Optional[float] = None


# --- /api/budget/history endpoint ---
class BudgetHistoryResponse(BaseModel):
    current_spend: float = Field(default=0.0, description="Current total spend in dollars")
    max_daily: float = Field(default=10.0, description="Daily budget limit in dollars")
    currency: str = Field(default="USD", description="Currency code")
    hourly_spend: List[float] = Field(default_factory=lambda: [0.0] * 24, description="24-point hourly spend array (0=midnight)")


# --- /api/memory endpoint ---
class MemoryEntry(BaseModel):
    event_id: Optional[int] = None
    tenant_id: str
    timestamp: Optional[str] = None
    session_id: Optional[str] = None
    actor: Optional[str] = None
    action_type: Optional[str] = None
    content: Optional[str] = None


class MemoryListResponse(BaseModel):
    entries: List[MemoryEntry]
    total: int
    limit: int
    offset: int


# --- /query endpoint (Phase 9: Monaco Editor SQL executor) ---
class QueryRequest(BaseModel):
    sql: str = Field(..., min_length=1, max_length=2000, description="Read-only SQL query to execute against StarRocks")


class QueryResponse(BaseModel):
    columns: List[str] = Field(default_factory=list, description="Column names from the result set")
    rows: List[List[Any]] = Field(default_factory=list, description="Result rows as lists of values")
    row_count: int = Field(default=0, description="Number of rows returned")
    truncated: bool = Field(default=False, description="Whether results were truncated at the 200-row limit")


# --- /workflows endpoint (Phase 9: Cytoscape DAG visualizer) ---
class WorkflowNode(BaseModel):
    id: str = Field(..., description="Unique node ID within the workflow")
    name: str = Field(..., description="Node display name")
    type: str = Field(default="Pod", description="Node type (e.g. Pod, DAG, Steps)")
    phase: str = Field(default="Pending", description="Node execution phase")
    dependencies: List[str] = Field(default_factory=list, description="List of node IDs this node depends on")


class WorkflowInfo(BaseModel):
    name: str = Field(..., description="Workflow name")
    phase: str = Field(default="Unknown", description="Workflow phase (e.g. Running, Succeeded, Failed)")
    started_at: str = Field(default="", description="Workflow start timestamp (ISO 8601)")
    finished_at: Optional[str] = Field(default=None, description="Workflow finish timestamp (ISO 8601) or null if still running")
    nodes: List[WorkflowNode] = Field(default_factory=list, description="DAG nodes for Cytoscape visualization")


class WorkflowListResponse(BaseModel):
    workflows: List[WorkflowInfo] = Field(default_factory=list, description="List of recent Argo workflows")
