"""
PHUC MCP Skills - All 11 Deployed Skills
=========================================

Integrates with MCP SDK 1.12.4 and agentic-workflows 5.0.0

Skills:
- Cloudflare (5): d1, r2, workers, vectorize, ai
- Analytics (3): attribution, campaign, reporting
- Security (3): injection-defense, scope-validator, pii-detector

Location: C:\\Users\\NickV\\agentic-workflows\\agentic-workflows\\src\\agentic_workflows\\skills\\phuc_mcp_skills.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════════
# MCP SDK IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════════
# OPENTELEMETRY IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from opentelemetry import trace

    tracer = trace.get_tracer("phuc.skills")
except ImportError:
    tracer = None

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MCP_ENDPOINT = "https://agentic-workflows-mcp.nick-9a6.workers.dev"
CAMARA_ENDPOINT = "https://mcp.camaramcp.com/sse"


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════


class SkillDomain(Enum):
    """Skill domains"""

    CLOUDFLARE = "cloudflare"
    ANALYTICS = "analytics"
    SECURITY = "security"


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class ToolCall(BaseModel):
    """Tool call request"""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Tool call result"""

    tool: str
    success: bool
    result: Any | None = None
    error: str | None = None
    duration_ms: float = 0.0


class SkillInfo(BaseModel):
    """Skill information"""

    name: str
    domain: str
    description: str
    tools: list[str]
    endpoint: str
    version: str = "1.0.0"


# ═══════════════════════════════════════════════════════════════════════════════
# SKILL CLASS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MCPSkill:
    """
    MCP Skill definition with tools.
    Connects to MCP server for tool execution.
    """

    name: str
    domain: SkillDomain
    description: str
    tools: list[str]
    endpoint: str = MCP_ENDPOINT
    version: str = "1.0.0"
    _session: ClientSession | None = field(default=None, repr=False)

    def info(self) -> SkillInfo:
        """Get skill info"""
        return SkillInfo(
            name=self.name,
            domain=self.domain.value,
            description=self.description,
            tools=self.tools,
            endpoint=self.endpoint,
            version=self.version,
        )

    async def call(self, tool: str, args: dict[str, Any] = None) -> ToolResult:
        """Call a tool in this skill"""
        import time

        start = time.time()

        if tool not in self.tools:
            return ToolResult(
                tool=tool, success=False, error=f"Tool '{tool}' not in skill '{self.name}'"
            )

        if not MCP_AVAILABLE:
            return ToolResult(
                tool=tool,
                success=True,
                result={"simulated": True, "tool": tool, "args": args},
                duration_ms=(time.time() - start) * 1000,
            )

        try:
            async with sse_client(self.endpoint) as (read, write):
                session = ClientSession(read, write)
                await session.initialize()
                result = await session.call_tool(tool, args or {})

                return ToolResult(
                    tool=tool,
                    success=True,
                    result=result.content[0].text if result.content else None,
                    duration_ms=(time.time() - start) * 1000,
                )
        except Exception as e:
            return ToolResult(
                tool=tool, success=False, error=str(e), duration_ms=(time.time() - start) * 1000
            )

    async def list_tools(self) -> list[str]:
        """List all tools in this skill"""
        return self.tools


# ═══════════════════════════════════════════════════════════════════════════════
# CLOUDFLARE SKILLS (5)
# ═══════════════════════════════════════════════════════════════════════════════

D1_SKILL = MCPSkill(
    name="d1",
    domain=SkillDomain.CLOUDFLARE,
    description="Cloudflare D1 serverless SQL database",
    tools=[
        "d1_query",  # Execute SELECT queries
        "d1_execute",  # Execute INSERT/UPDATE/DELETE
        "d1_batch",  # Batch operations
        "d1_migrate",  # Run migrations
        "d1_backup",  # Create backups
        "d1_restore",  # Restore from backup
    ],
)

R2_SKILL = MCPSkill(
    name="r2",
    domain=SkillDomain.CLOUDFLARE,
    description="Cloudflare R2 object storage",
    tools=[
        "r2_get",  # Get object
        "r2_put",  # Put object
        "r2_list",  # List objects
        "r2_delete",  # Delete object
        "r2_multipart",  # Multipart upload
        "r2_presign",  # Generate presigned URL
        "r2_copy",  # Copy object
    ],
)

WORKERS_SKILL = MCPSkill(
    name="workers",
    domain=SkillDomain.CLOUDFLARE,
    description="Cloudflare Workers deployment and management",
    tools=[
        "workers_deploy",  # Deploy worker
        "workers_list",  # List workers
        "workers_delete",  # Delete worker
        "workers_logs",  # Get worker logs
        "workers_tail",  # Tail logs in real-time
        "workers_secrets",  # Manage secrets
        "workers_routes",  # Manage routes
    ],
)

VECTORIZE_SKILL = MCPSkill(
    name="vectorize",
    domain=SkillDomain.CLOUDFLARE,
    description="Cloudflare Vectorize for vector embeddings",
    tools=[
        "vectorize_insert",  # Insert vectors
        "vectorize_query",  # Query similar vectors
        "vectorize_delete",  # Delete vectors
        "vectorize_upsert",  # Upsert vectors
        "vectorize_info",  # Get index info
        "vectorize_create",  # Create index
    ],
)

AI_SKILL = MCPSkill(
    name="ai",
    domain=SkillDomain.CLOUDFLARE,
    description="Cloudflare Workers AI inference",
    tools=[
        "ai_generate",  # Text generation
        "ai_embed",  # Generate embeddings
        "ai_classify",  # Classification
        "ai_summarize",  # Summarization
        "ai_translate",  # Translation
        "ai_image",  # Image generation
        "ai_speech",  # Speech-to-text
    ],
)

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICS SKILLS (3)
# ═══════════════════════════════════════════════════════════════════════════════

ATTRIBUTION_SKILL = MCPSkill(
    name="attribution",
    domain=SkillDomain.ANALYTICS,
    description="Campaign attribution and conversion tracking",
    tools=[
        "attribution_track",  # Track conversion event
        "attribution_query",  # Query attribution data
        "attribution_report",  # Generate attribution report
        "attribution_model",  # Set attribution model
        "attribution_window",  # Configure attribution window
        "attribution_export",  # Export attribution data
    ],
)

CAMPAIGN_SKILL = MCPSkill(
    name="campaign",
    domain=SkillDomain.ANALYTICS,
    description="Campaign management and optimization",
    tools=[
        "campaign_create",  # Create campaign
        "campaign_update",  # Update campaign
        "campaign_analyze",  # Analyze performance
        "campaign_optimize",  # Optimization suggestions
        "campaign_budget",  # Budget management
        "campaign_schedule",  # Schedule campaigns
    ],
)

REPORTING_SKILL = MCPSkill(
    name="reporting",
    domain=SkillDomain.ANALYTICS,
    description="Analytics reporting and dashboards",
    tools=[
        "report_generate",  # Generate report
        "report_schedule",  # Schedule reports
        "report_export",  # Export report
        "report_template",  # Manage templates
        "report_share",  # Share reports
        "report_dashboard",  # Dashboard widgets
    ],
)

# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY SKILLS (3)
# ═══════════════════════════════════════════════════════════════════════════════

INJECTION_DEFENSE_SKILL = MCPSkill(
    name="injection-defense",
    domain=SkillDomain.SECURITY,
    description="Prompt injection detection and prevention",
    tools=[
        "scan_prompt",  # Scan for injection
        "check_threat",  # Check threat level
        "sanitize_input",  # Sanitize user input
        "block_pattern",  # Block known patterns
        "audit_log",  # Log security events
        "threat_intel",  # Threat intelligence
    ],
)

SCOPE_VALIDATOR_SKILL = MCPSkill(
    name="scope-validator",
    domain=SkillDomain.SECURITY,
    description="Permission scope validation and enforcement",
    tools=[
        "validate_scope",  # Validate permission scope
        "check_permissions",  # Check user permissions
        "enforce_policy",  # Enforce access policy
        "grant_scope",  # Grant scope
        "revoke_scope",  # Revoke scope
        "audit_access",  # Audit access logs
    ],
)

PII_DETECTOR_SKILL = MCPSkill(
    name="pii-detector",
    domain=SkillDomain.SECURITY,
    description="PII detection, masking, and compliance",
    tools=[
        "detect_pii",  # Detect PII in text
        "mask_pii",  # Mask PII
        "audit_pii",  # Audit PII exposure
        "classify_data",  # Classify data sensitivity
        "redact_document",  # Redact PII from document
        "compliance_check",  # Check compliance status
    ],
)


# ═══════════════════════════════════════════════════════════════════════════════
# SKILL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

SKILLS: dict[str, MCPSkill] = {
    # Cloudflare (5)
    "d1": D1_SKILL,
    "r2": R2_SKILL,
    "workers": WORKERS_SKILL,
    "vectorize": VECTORIZE_SKILL,
    "ai": AI_SKILL,
    # Analytics (3)
    "attribution": ATTRIBUTION_SKILL,
    "campaign": CAMPAIGN_SKILL,
    "reporting": REPORTING_SKILL,
    # Security (3)
    "injection-defense": INJECTION_DEFENSE_SKILL,
    "scope-validator": SCOPE_VALIDATOR_SKILL,
    "pii-detector": PII_DETECTOR_SKILL,
}


# ═══════════════════════════════════════════════════════════════════════════════
# SKILL MANAGER
# ═══════════════════════════════════════════════════════════════════════════════


class SkillManager:
    """Manages all MCP skills"""

    def __init__(self, endpoint: str = MCP_ENDPOINT):
        self.endpoint = endpoint
        self.skills = SKILLS
        self._connected = False

    async def connect(self) -> dict[str, Any]:
        """Connect to MCP server and verify skills"""
        if not MCP_AVAILABLE:
            return {"connected": False, "error": "MCP SDK not available"}

        try:
            async with sse_client(self.endpoint) as (read, write):
                session = ClientSession(read, write)
                await session.initialize()
                tools = await session.list_tools()
                self._connected = True
                return {
                    "connected": True,
                    "endpoint": self.endpoint,
                    "tools_available": len(tools.tools),
                    "skills_registered": len(self.skills),
                }
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def get_skill(self, name: str) -> MCPSkill | None:
        """Get skill by name"""
        return self.skills.get(name)

    def get_skills_by_domain(self, domain: SkillDomain) -> list[MCPSkill]:
        """Get all skills in a domain"""
        return [s for s in self.skills.values() if s.domain == domain]

    async def call_tool(self, skill_name: str, tool: str, args: dict = None) -> ToolResult:
        """Call a tool from a specific skill"""
        skill = self.skills.get(skill_name)
        if not skill:
            return ToolResult(tool=tool, success=False, error=f"Skill '{skill_name}' not found")
        return await skill.call(tool, args)

    def list_skills(self) -> dict[str, SkillInfo]:
        """List all skills"""
        return {name: skill.info() for name, skill in self.skills.items()}

    def list_all_tools(self) -> dict[str, list[str]]:
        """List all tools by skill"""
        return {name: skill.tools for name, skill in self.skills.items()}

    def get_tool_skill(self, tool: str) -> str | None:
        """Find which skill contains a tool"""
        for name, skill in self.skills.items():
            if tool in skill.tools:
                return name
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def get_skills_by_domain(domain: SkillDomain) -> dict[str, MCPSkill]:
    """Get skills filtered by domain"""
    return {k: v for k, v in SKILLS.items() if v.domain == domain}


def get_cloudflare_skills() -> dict[str, MCPSkill]:
    """Get all Cloudflare skills"""
    return get_skills_by_domain(SkillDomain.CLOUDFLARE)


def get_analytics_skills() -> dict[str, MCPSkill]:
    """Get all Analytics skills"""
    return get_skills_by_domain(SkillDomain.ANALYTICS)


def get_security_skills() -> dict[str, MCPSkill]:
    """Get all Security skills"""
    return get_skills_by_domain(SkillDomain.SECURITY)


def get_all_tools() -> list[str]:
    """Get flat list of all tools"""
    tools = []
    for skill in SKILLS.values():
        tools.extend(skill.tools)
    return tools


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "SkillDomain",
    # Models
    "ToolCall",
    "ToolResult",
    "SkillInfo",
    # Classes
    "MCPSkill",
    "SkillManager",
    # Skill instances
    "D1_SKILL",
    "R2_SKILL",
    "WORKERS_SKILL",
    "VECTORIZE_SKILL",
    "AI_SKILL",
    "ATTRIBUTION_SKILL",
    "CAMPAIGN_SKILL",
    "REPORTING_SKILL",
    "INJECTION_DEFENSE_SKILL",
    "SCOPE_VALIDATOR_SKILL",
    "PII_DETECTOR_SKILL",
    # Registry
    "SKILLS",
    # Helpers
    "get_skills_by_domain",
    "get_cloudflare_skills",
    "get_analytics_skills",
    "get_security_skills",
    "get_all_tools",
    # Config
    "MCP_ENDPOINT",
]
