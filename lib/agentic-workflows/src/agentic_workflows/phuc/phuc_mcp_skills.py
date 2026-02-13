#!/usr/bin/env python3
"""
PHUC Platform - MCP Skill Registry

11 Skills across Cloudflare, Analytics, Security domains

Integrates:
- mcp SDK (1.12.4) - Model Context Protocol client/server
- anthropic SDK (0.75.0) - Claude API for skill execution
- httpx (0.28.1) - Async HTTP for MCP SSE transport
- pydantic (2.12.5) - Data validation for skill schemas
- opentelemetry (1.38.0) - Distributed tracing

Skills:
    Cloudflare (5): d1, r2, workers, vectorize, ai
    Analytics (3): attribution, campaign, reporting
    Security (3): injection-defense, scope-validator, pii-detector
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

from pydantic import BaseModel, Field
import httpx
from opentelemetry import trace

# MCP imports
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Tracer
tracer = trace.get_tracer("phuc.mcp_skills")


class SkillDomain(str, Enum):
    """Skill domain categories."""
    CLOUDFLARE = "cloudflare"
    ANALYTICS = "analytics"
    SECURITY = "security"


class SkillLevel(str, Enum):
    """Skill complexity levels."""
    L1_BASIC = "L1_basic"
    L2_INTERMEDIATE = "L2_intermediate"
    L3_ADVANCED = "L3_advanced"
    L4_EXPERT = "L4_expert"


class MCPSkillConfig(BaseModel):
    """Configuration for an MCP skill."""
    name: str
    domain: SkillDomain
    description: str
    level: SkillLevel = SkillLevel.L2_INTERMEDIATE
    tools: List[str] = Field(default_factory=list)
    allowed_claude_tools: List[str] = Field(default_factory=list)
    endpoint: str = ""
    defer_loading: bool = True
    requires: List[str] = Field(default_factory=list)
    optional: List[str] = Field(default_factory=list)


class SkillInvocation(BaseModel):
    """Record of a skill invocation."""
    skill_name: str
    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: int = 0
    tokens_used: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# MCP Server Endpoints
CENTILLION_MCP = "https://agentic-workflows-mcp.nick-9a6.workers.dev/mcp/sse"
CAMARA_MCP = "https://mcp.camaramcp.com/sse"


@dataclass
class MCPSkill:
    """MCP Skill with full SDK integration."""
    config: MCPSkillConfig
    _session: Optional[Any] = None
    _tools: List[Any] = field(default_factory=list)
    _invocations: List[SkillInvocation] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def domain(self) -> SkillDomain:
        return self.config.domain

    @property
    def tools(self) -> List[str]:
        return self.config.tools

    @property
    def endpoint(self) -> str:
        return self.config.endpoint

    async def connect(self) -> Dict[str, Any]:
        """Connect to MCP server and initialize session."""
        if not MCP_AVAILABLE:
            return {"error": "MCP SDK not available", "skill": self.name}

        with tracer.start_as_current_span(f"skill.{self.name}.connect") as span:
            span.set_attribute("skill.name", self.name)
            span.set_attribute("skill.domain", self.domain.value)

            try:
                async with sse_client(self.endpoint) as (read, write):
                    self._session = ClientSession(read, write)
                    await self._session.initialize()
                    tools_response = await self._session.list_tools()
                    self._tools = tools_response.tools

                    return {
                        "connected": True,
                        "skill": self.name,
                        "tools_count": len(self._tools),
                        "tools": [t.name for t in self._tools]
                    }
            except Exception as e:
                span.record_exception(e)
                return {"error": str(e), "skill": self.name}

    async def invoke(self, tool_name: str, args: Dict[str, Any]) -> SkillInvocation:
        """Invoke a tool on this skill."""
        start_time = datetime.now()

        with tracer.start_as_current_span(f"skill.{self.name}.invoke") as span:
            span.set_attribute("skill.name", self.name)
            span.set_attribute("tool.name", tool_name)

            invocation = SkillInvocation(
                skill_name=self.name,
                tool_name=tool_name,
                args=args
            )

            if not MCP_AVAILABLE:
                invocation.error = "MCP SDK not available"
                invocation.duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                self._invocations.append(invocation)
                return invocation

            try:
                async with sse_client(self.endpoint) as (read, write):
                    session = ClientSession(read, write)
                    await session.initialize()
                    result = await session.call_tool(tool_name, args)

                    invocation.result = result.content[0].text if result.content else None
                    invocation.duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            except Exception as e:
                span.record_exception(e)
                invocation.error = str(e)
                invocation.duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            self._invocations.append(invocation)
            return invocation

    def get_invocation_stats(self) -> Dict[str, Any]:
        """Get statistics about skill invocations."""
        if not self._invocations:
            return {"total": 0}

        successful = [i for i in self._invocations if i.error is None]
        failed = [i for i in self._invocations if i.error is not None]

        return {
            "total": len(self._invocations),
            "successful": len(successful),
            "failed": len(failed),
            "avg_duration_ms": sum(i.duration_ms for i in self._invocations) / len(self._invocations),
            "total_tokens": sum(i.tokens_used for i in self._invocations)
        }


class MCPSkillRegistry:
    """Registry of all MCP skills."""

    def __init__(self, endpoint: str = CENTILLION_MCP):
        self.endpoint = endpoint
        self.skills: Dict[str, MCPSkill] = {}
        self._initialize_skills()

    def _initialize_skills(self):
        """Initialize all 11 skills."""

        # CLOUDFLARE DOMAIN (5 skills)
        cloudflare_skills = [
            MCPSkillConfig(
                name="d1",
                domain=SkillDomain.CLOUDFLARE,
                description="Query and manage Cloudflare D1 SQLite databases. Use when storing data, running SQL queries, or managing schemas.",
                level=SkillLevel.L2_INTERMEDIATE,
                tools=["d1_query", "d1_execute", "d1_batch", "d1_schema"],
                allowed_claude_tools=["Read", "Grep", "Bash"],
                endpoint=self.endpoint,
            ),
            MCPSkillConfig(
                name="r2",
                domain=SkillDomain.CLOUDFLARE,
                description="Store and retrieve objects from Cloudflare R2 storage. Use when managing files, documents, or binary data.",
                level=SkillLevel.L2_INTERMEDIATE,
                tools=["r2_get", "r2_put", "r2_list", "r2_delete", "r2_presign"],
                allowed_claude_tools=["Read", "Grep", "Bash"],
                endpoint=self.endpoint,
            ),
            MCPSkillConfig(
                name="workers",
                domain=SkillDomain.CLOUDFLARE,
                description="Deploy and manage Cloudflare Workers for edge computing. Use when building APIs, serverless functions, or edge applications.",
                level=SkillLevel.L2_INTERMEDIATE,
                tools=["workers_deploy", "workers_list", "workers_delete", "workers_logs", "workers_tail"],
                allowed_claude_tools=["Read", "Write", "Bash"],
                endpoint=self.endpoint,
            ),
            MCPSkillConfig(
                name="vectorize",
                domain=SkillDomain.CLOUDFLARE,
                description="Create and query vector embeddings with Cloudflare Vectorize. Use when building semantic search, RAG, or similarity matching.",
                level=SkillLevel.L3_ADVANCED,
                tools=["vectorize_insert", "vectorize_query", "vectorize_delete", "vectorize_upsert"],
                allowed_claude_tools=["Read", "Grep", "Bash"],
                endpoint=self.endpoint,
                requires=["ai"],
            ),
            MCPSkillConfig(
                name="ai",
                domain=SkillDomain.CLOUDFLARE,
                description="Run AI models with Workers AI inference. Use when generating text, embeddings, images, or classifications.",
                level=SkillLevel.L3_ADVANCED,
                tools=["ai_generate", "ai_embed", "ai_classify", "ai_summarize", "ai_translate"],
                allowed_claude_tools=["Read", "Bash"],
                endpoint=self.endpoint,
            ),
        ]

        # ANALYTICS DOMAIN (3 skills)
        analytics_skills = [
            MCPSkillConfig(
                name="attribution",
                domain=SkillDomain.ANALYTICS,
                description="Calculate marketing attribution from campaign touchpoints to conversions. Use when measuring ROI, optimizing spend, or analyzing customer journeys.",
                level=SkillLevel.L3_ADVANCED,
                tools=["attribution_track", "attribution_query", "attribution_report", "attribution_model"],
                allowed_claude_tools=["Read", "Grep", "Glob", "Bash"],
                endpoint=self.endpoint,
            ),
            MCPSkillConfig(
                name="campaign",
                domain=SkillDomain.ANALYTICS,
                description="Generate and optimize marketing campaigns. Use when planning media mix, targeting audiences, or analyzing campaign performance.",
                level=SkillLevel.L3_ADVANCED,
                tools=["campaign_create", "campaign_update", "campaign_analyze", "campaign_optimize"],
                allowed_claude_tools=["Read", "Write", "Grep"],
                endpoint=self.endpoint,
            ),
            MCPSkillConfig(
                name="reporting",
                domain=SkillDomain.ANALYTICS,
                description="Generate executive dashboards and marketing reports. Use when creating presentations, tracking KPIs, or summarizing performance.",
                level=SkillLevel.L2_INTERMEDIATE,
                tools=["report_generate", "report_schedule", "report_export", "dashboard_create"],
                allowed_claude_tools=["Read", "Write", "Grep"],
                endpoint=self.endpoint,
            ),
        ]

        # SECURITY DOMAIN (3 skills)
        security_skills = [
            MCPSkillConfig(
                name="injection-defense",
                domain=SkillDomain.SECURITY,
                description="Multi-layer prompt injection detection and prevention. Use when processing untrusted input, validating prompts, or scanning for threats.",
                level=SkillLevel.L2_INTERMEDIATE,
                tools=["scan_prompt", "check_threat", "sanitize_input", "get_threat_report"],
                allowed_claude_tools=["Read"],
                endpoint=self.endpoint,
            ),
            MCPSkillConfig(
                name="scope-validator",
                domain=SkillDomain.SECURITY,
                description="Validate operations against security scope permissions. Use when enforcing access control, checking permissions, or auditing actions.",
                level=SkillLevel.L2_INTERMEDIATE,
                tools=["validate_scope", "check_permissions", "enforce_policy", "audit_access"],
                allowed_claude_tools=["Read"],
                endpoint=self.endpoint,
            ),
            MCPSkillConfig(
                name="pii-detector",
                domain=SkillDomain.SECURITY,
                description="Detect and redact personally identifiable information (PII). Use when processing sensitive data, anonymizing outputs, or compliance checking.",
                level=SkillLevel.L2_INTERMEDIATE,
                tools=["detect_pii", "mask_pii", "audit_pii", "get_pii_report"],
                allowed_claude_tools=["Read", "Grep"],
                endpoint=self.endpoint,
            ),
        ]

        # Register all skills
        all_skills = cloudflare_skills + analytics_skills + security_skills
        for config in all_skills:
            self.skills[config.name] = MCPSkill(config=config)

    def get_skill(self, name: str) -> Optional[MCPSkill]:
        """Get a skill by name."""
        return self.skills.get(name)

    def get_skills_by_domain(self, domain: SkillDomain) -> List[MCPSkill]:
        """Get all skills for a domain."""
        return [s for s in self.skills.values() if s.domain == domain]

    def list_skills(self) -> List[Dict[str, Any]]:
        """List all skills with their metadata."""
        return [
            {
                "name": s.name,
                "domain": s.domain.value,
                "description": s.config.description,
                "level": s.config.level.value,
                "tools": s.tools,
                "endpoint": s.endpoint
            }
            for s in self.skills.values()
        ]

    def search_skills(self, query: str, domain: Optional[SkillDomain] = None) -> List[MCPSkill]:
        """Search skills by query string."""
        query_lower = query.lower()
        results = []

        for skill in self.skills.values():
            if domain and skill.domain != domain:
                continue

            # Search in name and description
            if query_lower in skill.name.lower() or query_lower in skill.config.description.lower():
                results.append(skill)
                continue

            # Search in tools
            for tool in skill.tools:
                if query_lower in tool.lower():
                    results.append(skill)
                    break

        return results

    async def invoke_skill(self, skill_name: str, tool_name: str, args: Dict[str, Any]) -> SkillInvocation:
        """Invoke a tool on a skill."""
        skill = self.get_skill(skill_name)
        if not skill:
            return SkillInvocation(
                skill_name=skill_name,
                tool_name=tool_name,
                args=args,
                error=f"Skill '{skill_name}' not found"
            )

        return await skill.invoke(tool_name, args)

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_skills": len(self.skills),
            "by_domain": {
                domain.value: len(self.get_skills_by_domain(domain))
                for domain in SkillDomain
            },
            "endpoint": self.endpoint,
            "mcp_available": MCP_AVAILABLE
        }


# Singleton registry
_registry: Optional[MCPSkillRegistry] = None


def get_skill_registry(endpoint: str = CENTILLION_MCP) -> MCPSkillRegistry:
    """Get or create the skill registry singleton."""
    global _registry
    if _registry is None:
        _registry = MCPSkillRegistry(endpoint=endpoint)
    return _registry


def get_skill(name: str) -> Optional[MCPSkill]:
    """Get a skill by name from the global registry."""
    return get_skill_registry().get_skill(name)


def list_skills() -> List[Dict[str, Any]]:
    """List all skills from the global registry."""
    return get_skill_registry().list_skills()


async def invoke_skill(skill_name: str, tool_name: str, args: Dict[str, Any]) -> SkillInvocation:
    """Invoke a skill tool from the global registry."""
    return await get_skill_registry().invoke_skill(skill_name, tool_name, args)


# Convenience exports
CLOUDFLARE_SKILLS = ["d1", "r2", "workers", "vectorize", "ai"]
ANALYTICS_SKILLS = ["attribution", "campaign", "reporting"]
SECURITY_SKILLS = ["injection-defense", "scope-validator", "pii-detector"]
ALL_SKILLS = CLOUDFLARE_SKILLS + ANALYTICS_SKILLS + SECURITY_SKILLS
