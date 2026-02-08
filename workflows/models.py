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
