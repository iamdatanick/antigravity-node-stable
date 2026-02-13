"""
Unified Skill System for Agentic Workflows.

This module provides a unified skill registry that combines:
- agentic_workflows.skills.SkillRegistry (Anthropic SKILL.md format)
- agentic_workflows.openai_skills.OpenAISkillRegistry (OpenAI Codex format)

Features:
- Cross-format skill discovery and loading
- Skill-to-tool conversion for all agent types
- Unified search across all skill formats
- Progressive disclosure and deferred loading
- Context generation for any agent framework

Example:
    >>> from agentic_workflows.unified import UnifiedSkillRegistry
    >>> registry = UnifiedSkillRegistry()
    >>> registry.discover_all()
    >>>
    >>> # Search across all formats
    >>> results = registry.search("cloudflare database")
    >>>
    >>> # Get tools for different agent types
    >>> claude_tools = registry.get_claude_tools()
    >>> openai_tools = registry.get_openai_tools()
    >>> mcp_tools = registry.get_mcp_tools()
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_workflows.skills import SkillRegistry, SkillDefinition
    from agentic_workflows.openai_skills import OpenAISkillRegistry, SkillManifest


logger = logging.getLogger(__name__)


class SkillFormat(Enum):
    """Supported skill definition formats."""
    ANTHROPIC = "anthropic"      # SKILL.md with YAML frontmatter
    OPENAI_CODEX = "openai"      # OpenAI Codex/Skills format
    MCP_TOOL = "mcp"             # MCP tool definitions
    A2A_SKILL = "a2a"            # A2A protocol skills
    UNIFIED = "unified"          # Internal unified format


class ToolFormat(Enum):
    """Target tool output formats."""
    CLAUDE_API = "claude"        # Claude Messages API tool format
    OPENAI_API = "openai"        # OpenAI function calling format
    MCP_TOOL = "mcp"             # MCP Tool type
    A2A_SKILL = "a2a"            # A2A AgentSkill format


@dataclass
class UnifiedSkill:
    """Unified skill representation across all formats.

    This dataclass provides a common interface for skills regardless
    of their original format (Anthropic SKILL.md, OpenAI Codex, etc.).

    Attributes:
        name: Unique skill identifier.
        description: Human-readable description.
        format: Original skill format.
        domain: Skill domain/category.
        level: Complexity level (L1-L4 or equivalent).
        version: Semantic version string.
        tags: Categorization tags/keywords.
        tools: List of tool names this skill provides.
        requires: Required dependencies.
        defer_loading: Whether to defer full content loading.
        source_path: Path to source file.
        raw_content: Full skill content (loaded on demand).
        metadata: Additional format-specific metadata.
    """
    name: str
    description: str
    format: SkillFormat = SkillFormat.UNIFIED
    domain: str = "core"
    level: str = "L2_intermediate"
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    defer_loading: bool = True
    source_path: Path | None = None
    raw_content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # Internal references to original objects
    _anthropic_skill: Any = None
    _openai_manifest: Any = None

    @property
    def full_name(self) -> str:
        """Get fully qualified name."""
        return f"{self.domain}:{self.name}"

    @property
    def is_loaded(self) -> bool:
        """Check if full content is loaded."""
        return bool(self.raw_content)

    def to_claude_tool(self) -> dict[str, Any]:
        """Convert to Claude API tool format."""
        tool_def = {
            "name": f"skill_{self.name.replace('-', '_')}",
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The action to perform",
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context",
                        "additionalProperties": True,
                    },
                },
                "required": ["action"],
            },
        }

        if self.defer_loading:
            tool_def["defer_loading"] = True
            tool_def["cache_control"] = {"type": "ephemeral"}

        return tool_def

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": f"skill_{self.name.replace('-', '_')}",
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "The action to perform",
                        },
                        "params": {
                            "type": "object",
                            "description": "Action parameters",
                            "additionalProperties": True,
                        },
                    },
                    "required": ["action"],
                },
            },
        }

    def to_mcp_tool(self) -> dict[str, Any]:
        """Convert to MCP Tool format."""
        return {
            "name": f"skill_{self.name.replace('-', '_')}",
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "context": {"type": "object"},
                },
                "required": ["action"],
            },
        }

    def to_a2a_skill(self) -> dict[str, Any]:
        """Convert to A2A AgentSkill format."""
        return {
            "id": self.name,
            "name": self.name.replace("-", " ").title(),
            "description": self.description,
            "inputModes": ["text"],
            "outputModes": ["text"],
        }

    def get_context(self) -> str:
        """Get skill context for agent prompt injection."""
        parts = [
            f"# Skill: {self.name}",
            f"**Domain:** {self.domain}",
            f"**Level:** {self.level}",
            f"**Version:** {self.version}",
            "",
            f"**Description:** {self.description}",
            "",
        ]

        if self.tools:
            parts.append(f"**Tools:** {', '.join(self.tools)}")
            parts.append("")

        if self.requires:
            parts.append(f"**Requires:** {', '.join(self.requires)}")
            parts.append("")

        if self.raw_content:
            parts.append("---")
            parts.append("")
            parts.append(self.raw_content)

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "format": self.format.value,
            "domain": self.domain,
            "level": self.level,
            "version": self.version,
            "tags": self.tags.copy(),
            "tools": self.tools.copy(),
            "requires": self.requires.copy(),
            "defer_loading": self.defer_loading,
            "metadata": self.metadata.copy(),
        }


@dataclass
class SearchResult:
    """Result from unified skill search."""
    skill: UnifiedSkill
    score: float = 0.0
    matched_fields: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.skill.name

    @property
    def description(self) -> str:
        return self.skill.description


@dataclass
class UnifiedSkillConfig:
    """Configuration for the unified skill registry."""
    # Discovery paths
    anthropic_paths: list[Path] = field(default_factory=list)
    openai_paths: list[Path] = field(default_factory=list)

    # Auto-discovery
    auto_discover: bool = True
    include_default_paths: bool = True

    # Loading behavior
    defer_loading: bool = True
    always_loaded: list[str] = field(default_factory=list)

    # Search
    max_search_results: int = 20
    enable_cache: bool = True
    cache_ttl: int = 300


class UnifiedSkillRegistry:
    """Unified registry for all skill formats.

    Combines capabilities from:
    - agentic_workflows.skills.SkillRegistry
    - agentic_workflows.openai_skills.OpenAISkillRegistry

    Provides:
    - Cross-format skill discovery
    - Unified search interface
    - Tool conversion for any agent type
    - Progressive disclosure support

    Example:
        >>> registry = UnifiedSkillRegistry()
        >>> registry.discover_all()
        >>>
        >>> # Search for skills
        >>> results = registry.search("cloudflare")
        >>> for r in results:
        ...     print(f"{r.name}: {r.description}")
        >>>
        >>> # Get tools for different formats
        >>> claude_tools = registry.get_tools(ToolFormat.CLAUDE_API)
        >>> openai_tools = registry.get_tools(ToolFormat.OPENAI_API)
    """

    def __init__(self, config: UnifiedSkillConfig | None = None):
        """Initialize unified skill registry.

        Args:
            config: Registry configuration.
        """
        self.config = config or UnifiedSkillConfig()

        # Unified skill storage
        self._skills: dict[str, UnifiedSkill] = {}
        self._by_domain: dict[str, list[str]] = {}
        self._by_format: dict[SkillFormat, list[str]] = {}
        self._by_tag: dict[str, list[str]] = {}

        # Underlying registries (lazy loaded)
        self._anthropic_registry: SkillRegistry | None = None
        self._openai_registry: OpenAISkillRegistry | None = None

        # Thread safety
        self._lock = threading.RLock()

        # Search cache
        self._cache: dict[str, tuple[float, list[SearchResult]]] = {}

        if self.config.auto_discover:
            self.discover_all()

    def _get_anthropic_registry(self) -> "SkillRegistry":
        """Get or create Anthropic skill registry."""
        if self._anthropic_registry is None:
            try:
                from agentic_workflows.skills import SkillRegistry, get_registry
                self._anthropic_registry = get_registry()
            except ImportError:
                from agentic_workflows.skills.registry import SkillRegistry
                self._anthropic_registry = SkillRegistry()
        return self._anthropic_registry

    def _get_openai_registry(self) -> "OpenAISkillRegistry":
        """Get or create OpenAI skill registry."""
        if self._openai_registry is None:
            try:
                from agentic_workflows.openai_skills import (
                    OpenAISkillRegistry,
                    get_openai_skill_registry,
                )
                self._openai_registry = get_openai_skill_registry()
            except ImportError:
                from agentic_workflows.openai_skills.registry import (
                    OpenAISkillRegistry,
                    RegistryConfig,
                )
                self._openai_registry = OpenAISkillRegistry(
                    RegistryConfig(auto_index=False)
                )
        return self._openai_registry

    def discover_all(self, force: bool = False) -> int:
        """Discover skills from all sources.

        Args:
            force: Force rediscovery even if already discovered.

        Returns:
            Total number of skills discovered.
        """
        if self._skills and not force:
            return len(self._skills)

        with self._lock:
            if force:
                self._skills.clear()
                self._by_domain.clear()
                self._by_format.clear()
                self._by_tag.clear()
                self._cache.clear()

            count = 0

            # Discover Anthropic format skills
            count += self._discover_anthropic_skills()

            # Discover OpenAI format skills
            count += self._discover_openai_skills()

            logger.info(f"Discovered {count} total skills across all formats")
            return count

    def _discover_anthropic_skills(self) -> int:
        """Discover Anthropic SKILL.md format skills."""
        try:
            registry = self._get_anthropic_registry()

            # Add default paths
            if self.config.include_default_paths:
                default_paths = [
                    Path.home() / ".claude" / "skills",
                    Path.cwd() / ".claude" / "skills",
                    Path.cwd() / "skills",
                ]
                for path in default_paths:
                    if path.exists():
                        registry.add_skill_path(path)

            # Add configured paths
            for path in self.config.anthropic_paths:
                registry.add_skill_path(path)

            # Discover
            registry.discover_skills()

            # Import into unified registry
            count = 0
            for skill_def in registry:
                unified = self._from_anthropic_skill(skill_def)
                self._register_skill(unified)
                count += 1

            logger.info(f"Discovered {count} Anthropic format skills")
            return count

        except Exception as e:
            logger.warning(f"Failed to discover Anthropic skills: {e}")
            return 0

    def _discover_openai_skills(self) -> int:
        """Discover OpenAI Codex format skills."""
        try:
            registry = self._get_openai_registry()

            # Index all
            registry.index_all(force=True)

            # Import into unified registry
            count = 0
            for manifest in registry:
                unified = self._from_openai_manifest(manifest)
                self._register_skill(unified)
                count += 1

            logger.info(f"Discovered {count} OpenAI format skills")
            return count

        except Exception as e:
            logger.warning(f"Failed to discover OpenAI skills: {e}")
            return 0

    def _from_anthropic_skill(self, skill_def: "SkillDefinition") -> UnifiedSkill:
        """Convert Anthropic SkillDefinition to UnifiedSkill."""
        return UnifiedSkill(
            name=skill_def.name,
            description=skill_def.description,
            format=SkillFormat.ANTHROPIC,
            domain=skill_def.domain,
            level=skill_def.level,
            version=skill_def.version,
            tags=skill_def.tags.copy(),
            tools=skill_def.allowed_tools.copy(),
            requires=skill_def.requires.copy(),
            defer_loading=skill_def.defer_loading,
            source_path=skill_def.source_path,
            raw_content=skill_def.content,
            metadata={
                "author": skill_def.author,
                "security_scope": skill_def.security_scope,
                "optional": skill_def.optional.copy(),
                "conflicts": skill_def.conflicts.copy(),
                "load_on_demand": skill_def.load_on_demand.copy(),
                "recovery_config": skill_def.recovery_config.copy(),
            },
            _anthropic_skill=skill_def,
        )

    def _from_openai_manifest(self, manifest: "SkillManifest") -> UnifiedSkill:
        """Convert OpenAI SkillManifest to UnifiedSkill."""
        return UnifiedSkill(
            name=manifest.name,
            description=manifest.description,
            format=SkillFormat.OPENAI_CODEX,
            domain=manifest.metadata.extra.get("domain", "general"),
            level="L2_intermediate",  # OpenAI skills don't have levels
            version=manifest.version,
            tags=manifest.metadata.keywords.copy(),
            tools=[t.full_spec for t in manifest.tools] if manifest.tools else [],
            requires=[],  # OpenAI skills don't have explicit dependencies
            defer_loading=not manifest.loaded,
            source_path=manifest.source_path,
            raw_content=manifest.raw_content,
            metadata={
                "category": manifest.category.value if manifest.category else None,
                "author": manifest.metadata.author,
                "triggers": manifest.triggers.__dict__ if manifest.triggers else {},
            },
            _openai_manifest=manifest,
        )

    def _register_skill(self, skill: UnifiedSkill) -> None:
        """Register a unified skill."""
        # Main index
        self._skills[skill.name] = skill

        # Domain index
        if skill.domain not in self._by_domain:
            self._by_domain[skill.domain] = []
        if skill.name not in self._by_domain[skill.domain]:
            self._by_domain[skill.domain].append(skill.name)

        # Format index
        if skill.format not in self._by_format:
            self._by_format[skill.format] = []
        if skill.name not in self._by_format[skill.format]:
            self._by_format[skill.format].append(skill.name)

        # Tag index
        for tag in skill.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = []
            if skill.name not in self._by_tag[tag]:
                self._by_tag[tag].append(skill.name)

        # Apply always_loaded config
        if skill.name in self.config.always_loaded:
            skill.defer_loading = False

    def add_skill(self, skill: UnifiedSkill) -> None:
        """Add a skill to the registry.

        Args:
            skill: Unified skill to add.
        """
        with self._lock:
            self._register_skill(skill)
            self._invalidate_cache()

    def remove_skill(self, name: str) -> bool:
        """Remove a skill from the registry.

        Args:
            name: Skill name to remove.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if name not in self._skills:
                return False

            skill = self._skills[name]
            del self._skills[name]

            # Remove from indexes
            if skill.domain in self._by_domain:
                self._by_domain[skill.domain] = [
                    n for n in self._by_domain[skill.domain] if n != name
                ]

            if skill.format in self._by_format:
                self._by_format[skill.format] = [
                    n for n in self._by_format[skill.format] if n != name
                ]

            for tag in skill.tags:
                if tag in self._by_tag:
                    self._by_tag[tag] = [
                        n for n in self._by_tag[tag] if n != name
                    ]

            self._invalidate_cache()
            return True

    def get_skill(self, name: str) -> UnifiedSkill | None:
        """Get a skill by name.

        Args:
            name: Skill name.

        Returns:
            UnifiedSkill or None if not found.
        """
        with self._lock:
            return self._skills.get(name)

    def get_all_skills(self) -> list[UnifiedSkill]:
        """Get all registered skills.

        Returns:
            List of all unified skills.
        """
        with self._lock:
            return list(self._skills.values())

    def get_by_domain(self, domain: str) -> list[UnifiedSkill]:
        """Get all skills in a domain.

        Args:
            domain: Domain name.

        Returns:
            List of unified skills.
        """
        with self._lock:
            names = self._by_domain.get(domain, [])
            return [self._skills[n] for n in names if n in self._skills]

    def get_by_format(self, format: SkillFormat) -> list[UnifiedSkill]:
        """Get all skills of a specific format.

        Args:
            format: Skill format.

        Returns:
            List of unified skills.
        """
        with self._lock:
            names = self._by_format.get(format, [])
            return [self._skills[n] for n in names if n in self._skills]

    def get_by_tag(self, tag: str) -> list[UnifiedSkill]:
        """Get all skills with a tag.

        Args:
            tag: Tag keyword.

        Returns:
            List of unified skills.
        """
        with self._lock:
            names = self._by_tag.get(tag, [])
            return [self._skills[n] for n in names if n in self._skills]

    def search(
        self,
        query: str,
        domains: list[str] | None = None,
        formats: list[SkillFormat] | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Search for skills across all formats.

        Args:
            query: Search query.
            domains: Optional domain filter.
            formats: Optional format filter.
            limit: Maximum results.

        Returns:
            List of search results sorted by relevance.
        """
        limit = limit or self.config.max_search_results

        # Check cache
        cache_key = f"{query}:{domains}:{formats}"
        if self.config.enable_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached[:limit]

        results: list[SearchResult] = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        with self._lock:
            for skill in self._skills.values():
                # Apply filters
                if domains and skill.domain not in domains:
                    continue
                if formats and skill.format not in formats:
                    continue

                # Calculate score
                score, matched = self._calculate_score(
                    skill, query_lower, query_words
                )

                if score > 0:
                    results.append(SearchResult(
                        skill=skill,
                        score=score,
                        matched_fields=matched,
                    ))

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)

        # Cache
        if self.config.enable_cache:
            self._set_cached(cache_key, results)

        return results[:limit]

    def _calculate_score(
        self,
        skill: UnifiedSkill,
        query_lower: str,
        query_words: set[str],
    ) -> tuple[float, list[str]]:
        """Calculate search relevance score."""
        score = 0.0
        matched: list[str] = []

        name_lower = skill.name.lower()
        desc_lower = skill.description.lower()

        # Name matching
        if query_lower == name_lower:
            score += 100.0
            matched.append("name")
        elif query_lower in name_lower:
            score += 70.0
            matched.append("name")
        elif name_lower in query_lower:
            score += 50.0
            matched.append("name")

        # Description matching
        if query_lower in desc_lower:
            score += 30.0
            matched.append("description")

        # Word matching
        name_words = set(name_lower.replace("-", " ").replace("_", " ").split())
        desc_words = set(desc_lower.split())

        name_overlap = len(query_words & name_words)
        desc_overlap = len(query_words & desc_words)

        if name_overlap > 0:
            score += name_overlap * 20.0
            if "name" not in matched:
                matched.append("name")

        if desc_overlap > 0:
            score += desc_overlap * 10.0
            if "description" not in matched:
                matched.append("description")

        # Tag matching
        tags_lower = {t.lower() for t in skill.tags}
        tag_overlap = len(query_words & tags_lower)
        if tag_overlap > 0:
            score += tag_overlap * 15.0
            matched.append("tags")

        # Domain matching
        if query_lower in skill.domain.lower():
            score += 25.0
            matched.append("domain")

        return score, matched

    def _get_cached(self, key: str) -> list[SearchResult] | None:
        """Get cached search results."""
        import time
        if key not in self._cache:
            return None
        timestamp, results = self._cache[key]
        if time.time() - timestamp > self.config.cache_ttl:
            del self._cache[key]
            return None
        return results

    def _set_cached(self, key: str, results: list[SearchResult]) -> None:
        """Cache search results."""
        import time
        self._cache[key] = (time.time(), results)

    def _invalidate_cache(self) -> None:
        """Invalidate all cached results."""
        self._cache.clear()

    def get_tools(
        self,
        format: ToolFormat = ToolFormat.CLAUDE_API,
        domains: list[str] | None = None,
        include_deferred: bool = True,
    ) -> list[dict[str, Any]]:
        """Get tool definitions in specified format.

        Args:
            format: Target tool format.
            domains: Optional domain filter.
            include_deferred: Include deferred skills.

        Returns:
            List of tool definitions.
        """
        tools = []

        with self._lock:
            for skill in self._skills.values():
                if domains and skill.domain not in domains:
                    continue
                if not include_deferred and skill.defer_loading:
                    continue

                if format == ToolFormat.CLAUDE_API:
                    tools.append(skill.to_claude_tool())
                elif format == ToolFormat.OPENAI_API:
                    tools.append(skill.to_openai_tool())
                elif format == ToolFormat.MCP_TOOL:
                    tools.append(skill.to_mcp_tool())
                elif format == ToolFormat.A2A_SKILL:
                    tools.append(skill.to_a2a_skill())

        return tools

    def get_claude_tools(
        self,
        domains: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get tools in Claude API format."""
        return self.get_tools(ToolFormat.CLAUDE_API, domains)

    def get_openai_tools(
        self,
        domains: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get tools in OpenAI function format."""
        return self.get_tools(ToolFormat.OPENAI_API, domains)

    def get_mcp_tools(
        self,
        domains: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get tools in MCP format."""
        return self.get_tools(ToolFormat.MCP_TOOL, domains)

    def get_a2a_skills(
        self,
        domains: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get skills in A2A format."""
        return self.get_tools(ToolFormat.A2A_SKILL, domains)

    def load_skill_context(self, name: str) -> str:
        """Load full skill context.

        Args:
            name: Skill name.

        Returns:
            Skill context string for prompt injection.
        """
        skill = self.get_skill(name)
        if skill is None:
            return ""

        # Load content if needed
        if not skill.raw_content:
            self._load_skill_content(skill)

        return skill.get_context()

    def _load_skill_content(self, skill: UnifiedSkill) -> None:
        """Load full content for a skill."""
        try:
            if skill.format == SkillFormat.ANTHROPIC and skill._anthropic_skill:
                registry = self._get_anthropic_registry()
                content = registry.load_skill_context(skill.name)
                skill.raw_content = content

            elif skill.format == SkillFormat.OPENAI_CODEX and skill._openai_manifest:
                registry = self._get_openai_registry()
                content = registry.get_skill_context(skill.name)
                skill.raw_content = content

            elif skill.source_path and skill.source_path.exists():
                skill.raw_content = skill.source_path.read_text(encoding="utf-8")

        except Exception as e:
            logger.warning(f"Failed to load content for skill {skill.name}: {e}")

    def load_multiple_contexts(self, names: list[str]) -> dict[str, str]:
        """Load contexts for multiple skills.

        Args:
            names: List of skill names.

        Returns:
            Dictionary mapping names to contexts.
        """
        return {name: self.load_skill_context(name) for name in names}

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            by_format = {
                fmt.value: len(names)
                for fmt, names in self._by_format.items()
            }
            by_domain = {
                domain: len(names)
                for domain, names in self._by_domain.items()
            }

            return {
                "total_skills": len(self._skills),
                "by_format": by_format,
                "by_domain": by_domain,
                "domains": list(self._by_domain.keys()),
                "tags": list(self._by_tag.keys()),
                "deferred_count": sum(
                    1 for s in self._skills.values() if s.defer_loading
                ),
                "cache_entries": len(self._cache),
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._skills)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._skills

    def __iter__(self) -> Iterator[UnifiedSkill]:
        with self._lock:
            return iter(list(self._skills.values()))

    def __repr__(self) -> str:
        return f"UnifiedSkillRegistry(skills={len(self)})"


# Global registry
_global_registry: UnifiedSkillRegistry | None = None


def get_unified_registry() -> UnifiedSkillRegistry:
    """Get or create the global unified skill registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = UnifiedSkillRegistry()
    return _global_registry


def set_unified_registry(registry: UnifiedSkillRegistry) -> None:
    """Set the global unified skill registry."""
    global _global_registry
    _global_registry = registry
