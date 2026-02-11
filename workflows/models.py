"""Pydantic request/response models for Antigravity Node v13.0 A2A endpoints."""

from typing import Any

from pydantic import BaseModel, Field

# Constants for validation limits
MAX_GOAL_LENGTH = 10000  # Maximum characters for task goal (balance between flexibility and DoS protection)
MAX_CONTEXT_LENGTH = 50000  # Maximum characters for task context (allows detailed context without unbounded payloads)


# --- /task endpoint ---
class TaskRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=MAX_GOAL_LENGTH, description="The goal or task to accomplish")
    context: str | None = Field(
        default="", max_length=MAX_CONTEXT_LENGTH, description="Additional context for the task"
    )
    session_id: str | None = Field(default=None, description="Session ID for continuity; auto-generated if omitted")


class TaskResponse(BaseModel):
    status: str
    session_id: str
    tenant_id: str
    history_count: int


# --- /handoff endpoint ---
class HandoffRequest(BaseModel):
    target_agent: str = Field(..., min_length=1, description="Target agent identifier for handoff")
    payload: dict[str, Any] | None = Field(default={}, description="Payload to pass to the target agent")


class HandoffResponse(BaseModel):
    status: str
    target: str


# --- /webhook endpoint ---
class WebhookPayload(BaseModel):
    task_id: str = Field(default="unknown", description="Argo workflow task ID")
    status: str = Field(default="unknown", description="Workflow status (Succeeded, Failed, etc.)")
    message: str | None = Field(default="", description="Status message or error details")


class WebhookResponse(BaseModel):
    ack: bool


# --- /v1/chat/completions endpoint ---
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, description="List of chat messages")
    model: str | None = Field(default=None, description="Model to use; defaults to GOOSE_MODEL env var or gpt-4o")
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int | None = Field(default=2048, ge=1, le=128000, description="Maximum tokens to generate")


# --- /upload endpoint ---
class UploadResponse(BaseModel):
    status: str
    key: str
    size: int


# --- /health endpoint ---
class HealthCheck(BaseModel):
    name: str
    healthy: bool
    error: str | None = None


class HealthLevel(BaseModel):
    level: str
    name: str
    checks: list[HealthCheck]


class HealthResponse(BaseModel):
    status: str
    levels: list[HealthLevel]


# --- /tools endpoint ---
class ToolInfo(BaseModel):
    name: str
    server: str
    description: str | None = None
    status: str | None = None
    transport: str | None = None
    url: str | None = None


class ToolsResponse(BaseModel):
    tools: list[ToolInfo]
    total: int


# --- /capabilities endpoint ---
class CapabilitiesResponse(BaseModel):
    node: str
    protocols: list[str]
    endpoints: dict[str, str]
    mcp_servers: dict[str, Any]
    memory: dict[str, str]
    budget: dict[str, Any]


# --- /v1/inference endpoint ---
class InferenceRequest(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=256, description="Name of the model deployed in OVMS")
    input_data: dict[str, Any] = Field(
        ..., description='Mapping of input tensor names to values (e.g. {"input": [[1.0, 2.0]]})'
    )


class InferenceResponse(BaseModel):
    status: str
    model: str | None = None
    outputs: dict[str, Any] | None = None
    message: str | None = None
    latency_ms: float | None = None


# --- /api/budget/history endpoint ---
class BudgetHistoryResponse(BaseModel):
    current_spend: float = Field(default=0.0, description="Current total spend in dollars")
    max_daily: float = Field(default=10.0, description="Daily budget limit in dollars")
    currency: str = Field(default="USD", description="Currency code")
    hourly_spend: list[float] = Field(
        default_factory=lambda: [0.0] * 24, description="24-point hourly spend array (0=midnight)"
    )


# --- /api/memory endpoint ---
class MemoryEntry(BaseModel):
    event_id: int | None = None
    tenant_id: str
    timestamp: str | None = None
    session_id: str | None = None
    actor: str | None = None
    action_type: str | None = None
    content: str | None = None


class MemoryListResponse(BaseModel):
    entries: list[MemoryEntry]
    total: int
    limit: int
    offset: int


# --- /query endpoint (Phase 9: Monaco Editor SQL executor) ---
class QueryRequest(BaseModel):
    sql: str = Field(..., min_length=1, max_length=2000, description="Read-only SQL query to execute against StarRocks")


class QueryResponse(BaseModel):
    columns: list[str] = Field(default_factory=list, description="Column names from the result set")
    rows: list[list[Any]] = Field(default_factory=list, description="Result rows as lists of values")
    row_count: int = Field(default=0, description="Number of rows returned")
    truncated: bool = Field(default=False, description="Whether results were truncated at the 200-row limit")


# --- /workflows endpoint (Phase 9: Cytoscape DAG visualizer) ---
class WorkflowNode(BaseModel):
    id: str = Field(..., description="Unique node ID within the workflow")
    name: str = Field(..., description="Node display name")
    type: str = Field(default="Pod", description="Node type (e.g. Pod, DAG, Steps)")
    phase: str = Field(default="Pending", description="Node execution phase")
    dependencies: list[str] = Field(default_factory=list, description="List of node IDs this node depends on")


class WorkflowInfo(BaseModel):
    name: str = Field(..., description="Workflow name")
    phase: str = Field(default="Unknown", description="Workflow phase (e.g. Running, Succeeded, Failed)")
    started_at: str = Field(default="", description="Workflow start timestamp (ISO 8601)")
    finished_at: str | None = Field(
        default=None, description="Workflow finish timestamp (ISO 8601) or null if still running"
    )
    nodes: list[WorkflowNode] = Field(default_factory=list, description="DAG nodes for Cytoscape visualization")


class WorkflowListResponse(BaseModel):
    workflows: list[WorkflowInfo] = Field(default_factory=list, description="List of recent Argo workflows")


# --- /api/settings/keys endpoint ---
class ApiKeyRequest(BaseModel):
    provider: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9_-]+$",
        description="Provider slug: openai, anthropic, google, mistral",
    )
    api_key: str = Field(..., min_length=1, max_length=256, description="The API key value")


class ApiKeyEntry(BaseModel):
    provider: str
    masked_key: str
    configured: bool = True


class ApiKeyListResponse(BaseModel):
    keys: list[ApiKeyEntry]
