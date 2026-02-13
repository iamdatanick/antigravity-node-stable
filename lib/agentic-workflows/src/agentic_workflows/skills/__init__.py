"""
Skills module for Agentic Workflows v5.0.

This module provides skill registration, discovery, and management
for progressive skill loading in Claude Code integrations.

Key Components:
- SkillRegistry: Central registry for skill definitions
- SkillDefinition: Dataclass representing a skill
- SkillLevel: Enum for skill complexity levels
- SkillDomain: Enum for skill domain categories

Quick Start:
    >>> from agentic_workflows.skills import (
    ...     SkillRegistry,
    ...     SkillDefinition,
    ...     get_registry,
    ...     discover_default_skills,
    ... )
    >>>
    >>> # Discover skills from default locations
    >>> count = discover_default_skills()
    >>> print(f"Found {count} skills")
    >>>
    >>> # Get tool definitions for Claude API
    >>> registry = get_registry()
    >>> tools = registry.get_tool_definitions()
    >>>
    >>> # Load specific skill context
    >>> context = registry.load_skill_context("cloudflare-d1")

Version: 5.0.0
"""

from agentic_workflows.skills.phuc_mcp_skills import (
    AI_SKILL,
    ATTRIBUTION_SKILL,
    CAMPAIGN_SKILL,
    # Skill instances
    D1_SKILL,
    INJECTION_DEFENSE_SKILL,
    # Config
    MCP_ENDPOINT,
    PII_DETECTOR_SKILL,
    R2_SKILL,
    REPORTING_SKILL,
    SCOPE_VALIDATOR_SKILL,
    # Registry
    SKILLS,
    VECTORIZE_SKILL,
    WORKERS_SKILL,
    # Classes
    MCPSkill,
    SkillInfo,
    SkillManager,
    # Models
    ToolCall,
    ToolResult,
    get_all_tools,
    get_analytics_skills,
    get_cloudflare_skills,
    get_security_skills,
    # Helpers
    get_skills_by_domain,
)

# PHUC MCP Skills
from agentic_workflows.skills.phuc_mcp_skills import (
    # Enums
    SkillDomain as MCPSkillDomain,
)
from agentic_workflows.skills.registry import (
    SkillDefinition,
    SkillDomain,
    SkillLevel,
    # Core classes
    SkillRegistry,
    discover_default_skills,
    # Global registry functions
    get_registry,
    register_skill,
    set_registry,
)

__all__ = [
    # Core classes
    "SkillRegistry",
    "SkillDefinition",
    "SkillLevel",
    "SkillDomain",
    # Global registry functions
    "get_registry",
    "set_registry",
    "discover_default_skills",
    "register_skill",
    # PHUC MCP Skills - Enums
    "MCPSkillDomain",
    # PHUC MCP Skills - Models
    "ToolCall",
    "ToolResult",
    "SkillInfo",
    # PHUC MCP Skills - Classes
    "MCPSkill",
    "SkillManager",
    # PHUC MCP Skills - Instances
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
    # PHUC MCP Skills - Registry
    "SKILLS",
    # PHUC MCP Skills - Helpers
    "get_skills_by_domain",
    "get_cloudflare_skills",
    "get_analytics_skills",
    "get_security_skills",
    "get_all_tools",
    # PHUC MCP Skills - Config
    "MCP_ENDPOINT",
]

__version__ = "5.0.0"
