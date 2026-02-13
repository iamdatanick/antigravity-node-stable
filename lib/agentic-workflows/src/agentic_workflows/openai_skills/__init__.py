"""
OpenAI Skills SDK Integration Module for Agentic Workflows.

This module provides comprehensive support for the agentskills.io open standard,
enabling skill discovery, installation, and management across both Anthropic
Claude Code and OpenAI Codex environments.

Features:
- Skill Types: Dataclasses for skill manifests, instructions, resources, and tools
- Skill Loader: Parse SKILL.md files and OpenAI Codex format skills
- Skill Installer: Install skills from GitHub URLs and curated catalogs
- Skill Registry: Index, search, and manage skills with context generation
- Compatibility: Convert between Anthropic, OpenAI, and agentskills.io formats

Supported Formats:
- agentskills.io standard (SKILL.md with YAML frontmatter)
- Anthropic Claude Code (~/.claude/skills)
- OpenAI Codex (~/.codex/skills)
- agentic_workflows.skills integration

Quick Start:
    >>> from agentic_workflows.openai_skills import (
    ...     SkillLoader,
    ...     SkillInstaller,
    ...     OpenAISkillRegistry,
    ...     SkillManifest,
    ... )
    >>>
    >>> # Discover skills
    >>> loader = SkillLoader()
    >>> skills = loader.discover_all()
    >>>
    >>> # Install from GitHub
    >>> installer = SkillInstaller()
    >>> result = installer.install_by_name("pdf-processing")
    >>>
    >>> # Search registry
    >>> registry = OpenAISkillRegistry()
    >>> results = registry.search("data extraction")
    >>>
    >>> # Get skill context for agent
    >>> context = registry.get_skill_context("pdf-processing")

Architecture:
    The module follows progressive disclosure principles:
    1. Metadata (~100 tokens) indexed for all skills at startup
    2. Full content (<5000 tokens) loaded when skill is activated
    3. Resources loaded on-demand during execution

Integration:
    Works seamlessly with the existing agentic_workflows.skills module,
    providing cross-platform skill management while maintaining backward
    compatibility.

References:
    - agentskills.io specification: https://agentskills.io/specification
    - OpenAI Skills repository: https://github.com/openai/skills
    - Anthropic Skills repository: https://github.com/anthropics/skills

Author: Agentic Workflows Contributors
Version: 1.0.0
"""

# Skill Types
from agentic_workflows.openai_skills.skill_types import (
    # Enums
    SkillCategory,
    ResourceType,
    ToolPermission,
    # Dataclasses
    SkillMetadata,
    SkillTool,
    SkillResource,
    SkillInstruction,
    SkillTrigger,
    SkillManifest,
    # Validation
    validate_skill_name,
    # Constants
    MAX_NAME_LENGTH,
    MAX_DESCRIPTION_LENGTH,
    # Type aliases
    SkillDict,
    SkillList,
)

# Skill Loader
from agentic_workflows.openai_skills.loader import (
    # Classes
    SkillPath,
    LoaderConfig,
    SkillLoader,
    # Functions
    create_default_loader,
    discover_skills,
    load_skill_md,
)

# Skill Installer
from agentic_workflows.openai_skills.installer import (
    # Enums
    InstallTarget,
    InstallStatus,
    # Dataclasses
    InstallResult,
    InstallerConfig,
    SkillSource,
    # Classes
    SkillInstaller,
    # Functions
    install_skill,
    uninstall_skill,
    list_installed_skills,
)

# Skill Registry
from agentic_workflows.openai_skills.registry import (
    # Enums
    SearchMode,
    # Dataclasses
    SearchResult,
    RegistryConfig,
    # Classes
    OpenAISkillRegistry,
    # Global functions
    get_openai_skill_registry,
    set_openai_skill_registry,
    search_skills,
    get_skill_context,
)

# Compatibility
from agentic_workflows.openai_skills.compat import (
    # Enums
    SkillFormat,
    # Dataclasses
    ConversionResult,
    # Classes
    FormatDetector,
    SkillConverter,
    CrossPlatformDiscovery,
    # Functions
    detect_format,
    convert_skill,
    discover_cross_platform,
)


__all__ = [
    # === Skill Types ===
    # Enums
    "SkillCategory",
    "ResourceType",
    "ToolPermission",
    # Dataclasses
    "SkillMetadata",
    "SkillTool",
    "SkillResource",
    "SkillInstruction",
    "SkillTrigger",
    "SkillManifest",
    # Validation
    "validate_skill_name",
    # Constants
    "MAX_NAME_LENGTH",
    "MAX_DESCRIPTION_LENGTH",
    # Type aliases
    "SkillDict",
    "SkillList",

    # === Skill Loader ===
    "SkillPath",
    "LoaderConfig",
    "SkillLoader",
    "create_default_loader",
    "discover_skills",
    "load_skill_md",

    # === Skill Installer ===
    "InstallTarget",
    "InstallStatus",
    "InstallResult",
    "InstallerConfig",
    "SkillSource",
    "SkillInstaller",
    "install_skill",
    "uninstall_skill",
    "list_installed_skills",

    # === Skill Registry ===
    "SearchMode",
    "SearchResult",
    "RegistryConfig",
    "OpenAISkillRegistry",
    "get_openai_skill_registry",
    "set_openai_skill_registry",
    "search_skills",
    "get_skill_context",

    # === Compatibility ===
    "SkillFormat",
    "ConversionResult",
    "FormatDetector",
    "SkillConverter",
    "CrossPlatformDiscovery",
    "detect_format",
    "convert_skill",
    "discover_cross_platform",
]

__version__ = "1.0.0"


# Convenience re-exports for common usage patterns
def quick_search(query: str) -> list[SearchResult]:
    """Quick search for skills matching a query.

    This is a convenience function that uses the global registry.

    Args:
        query: Search query text.

    Returns:
        List of search results sorted by relevance.

    Example:
        >>> from agentic_workflows.openai_skills import quick_search
        >>> results = quick_search("pdf extraction tables")
        >>> for r in results:
        ...     print(f"{r.name}: {r.description}")
    """
    return get_openai_skill_registry().search(query)


def quick_install(source: str) -> InstallResult:
    """Quick install a skill from URL or name.

    This is a convenience function for common installation tasks.

    Args:
        source: GitHub URL, local path, or skill name.

    Returns:
        Installation result.

    Example:
        >>> from agentic_workflows.openai_skills import quick_install
        >>> result = quick_install("pdf-processing")
        >>> print(result.status)
    """
    return install_skill(source)


def get_all_skills() -> list[SkillManifest]:
    """Get all discovered skills.

    Returns:
        List of all skill manifests.

    Example:
        >>> from agentic_workflows.openai_skills import get_all_skills
        >>> skills = get_all_skills()
        >>> print(f"Found {len(skills)} skills")
    """
    return list(get_openai_skill_registry())


def get_context_for_skills(names: list[str]) -> dict[str, str]:
    """Get context strings for multiple skills.

    Args:
        names: List of skill names.

    Returns:
        Dictionary mapping names to context strings.

    Example:
        >>> from agentic_workflows.openai_skills import get_context_for_skills
        >>> contexts = get_context_for_skills(["pdf-processing", "data-analysis"])
        >>> for name, ctx in contexts.items():
        ...     print(f"=== {name} ===")
        ...     print(ctx[:200])
    """
    return get_openai_skill_registry().get_multiple_contexts(names)
