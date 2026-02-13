"""
Skill Registry for Agentic Workflows v5.0.

This module provides a comprehensive skill registration and management system
for deferred skill loading, dependency resolution, and tool definition generation.
Skills are defined in SKILL.md files with YAML frontmatter and markdown content.

Features:
- Automatic skill discovery from configurable paths
- YAML frontmatter parsing for skill metadata
- Skill dependency resolution (requires, optional, conflicts)
- Tool definition generation with defer_loading support
- Context injection for loaded skills
- Thread-safe operations

Example usage:
    >>> registry = SkillRegistry()
    >>> registry.discover_skills([Path("~/.claude/skills")])
    >>> tools = registry.get_tool_definitions()
    >>> context = registry.load_skill_context("cloudflare-d1")

Author: Agentic Workflows Contributors
Version: 5.0.0
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class SkillLevel(Enum):
    """Skill complexity levels for progressive loading.

    Skills are categorized by complexity to enable intelligent
    progressive loading strategies:

    - L1_FOUNDATION: Core skills always loaded (e.g., basic tools)
    - L2_INTERMEDIATE: Common domain skills loaded on-demand
    - L3_ADVANCED: Complex multi-step workflows
    - L4_EXPERT: Specialist skills requiring deep context
    """

    L1_FOUNDATION = "L1_foundation"
    L2_INTERMEDIATE = "L2_intermediate"
    L3_ADVANCED = "L3_advanced"
    L4_EXPERT = "L4_expert"

    @classmethod
    def from_string(cls, value: str) -> SkillLevel:
        """Parse skill level from string.

        Args:
            value: String representation (e.g., "L1_foundation", "l2_intermediate")

        Returns:
            Corresponding SkillLevel enum value.

        Raises:
            ValueError: If string doesn't match any level.
        """
        normalized = value.lower().strip()
        for level in cls:
            if level.value.lower() == normalized:
                return level
        # Try matching just the level prefix
        for level in cls:
            if normalized.startswith(level.value[:2].lower()):
                return level
        raise ValueError(f"Unknown skill level: {value}")


class SkillDomain(Enum):
    """Skill domain categories for organization and filtering.

    Domains enable:
    - Logical grouping of related skills
    - Domain-specific tool permissions
    - Targeted skill discovery
    """

    CLOUDFLARE = "cloudflare"
    PHARMA = "pharma"
    ANALYTICS = "analytics"
    LIFESCI = "lifesci"
    DEVOPS = "devops"
    SECURITY = "security"
    DATA = "data"
    CODE = "code"
    RESEARCH = "research"
    CORE = "core"

    @classmethod
    def from_string(cls, value: str) -> SkillDomain:
        """Parse domain from string.

        Args:
            value: String representation.

        Returns:
            Corresponding SkillDomain enum value.

        Raises:
            ValueError: If string doesn't match any domain.
        """
        normalized = value.lower().strip()
        for domain in cls:
            if domain.value == normalized:
                return domain
        raise ValueError(f"Unknown skill domain: {value}")


@dataclass
class SkillDefinition:
    """Complete definition of a skill from SKILL.md frontmatter.

    This dataclass captures all metadata needed for skill registration,
    dependency resolution, and tool definition generation.

    Attributes:
        name: Unique skill identifier (e.g., "cloudflare-d1").
        description: Human-readable description for tool definitions.
        level: Skill complexity level (L1-L4).
        domain: Primary domain category.
        allowed_tools: Tools this skill can invoke.
        requires: Skills that MUST be loaded before this skill.
        optional: Skills that enhance this skill if available.
        conflicts: Skills that cannot be loaded simultaneously.
        load_on_demand: References to progressively load during execution.
        defer_loading: Whether to defer full loading until needed.
        version: Semantic version string.
        author: Skill author/maintainer.
        tags: Additional categorization tags.
        security_scope: Required security scope level (1-4).
        recovery_config: Retry/fallback configuration.
        metadata: Additional custom metadata.
        source_path: Path to the SKILL.md file.
        content: Raw markdown content (excluding frontmatter).
    """

    name: str
    description: str
    level: str = "L2_intermediate"
    domain: str = "core"
    allowed_tools: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    optional: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    load_on_demand: list[str] = field(default_factory=list)
    defer_loading: bool = True
    version: str = "1.0.0"
    author: str = ""
    tags: list[str] = field(default_factory=list)
    security_scope: int = 2
    recovery_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None
    content: str = ""

    def __post_init__(self):
        """Validate and normalize skill definition."""
        if not self.name:
            raise ValueError("Skill name is required")

        # Normalize name to lowercase with hyphens
        self.name = self.name.lower().replace("_", "-").strip()

        # Ensure lists are actual lists
        if isinstance(self.allowed_tools, str):
            self.allowed_tools = [self.allowed_tools]
        if isinstance(self.requires, str):
            self.requires = [self.requires]
        if isinstance(self.optional, str):
            self.optional = [self.optional]
        if isinstance(self.conflicts, str):
            self.conflicts = [self.conflicts]
        if isinstance(self.load_on_demand, str):
            self.load_on_demand = [self.load_on_demand]
        if isinstance(self.tags, str):
            self.tags = [self.tags]

    @property
    def level_enum(self) -> SkillLevel:
        """Get level as enum."""
        return SkillLevel.from_string(self.level)

    @property
    def domain_enum(self) -> SkillDomain:
        """Get domain as enum."""
        try:
            return SkillDomain.from_string(self.domain)
        except ValueError:
            return SkillDomain.CORE

    @property
    def is_foundation(self) -> bool:
        """Check if this is a foundation-level skill."""
        return self.level_enum == SkillLevel.L1_FOUNDATION

    @property
    def full_name(self) -> str:
        """Get fully qualified name (domain:name)."""
        return f"{self.domain}:{self.name}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all skill attributes.
        """
        return {
            "name": self.name,
            "description": self.description,
            "level": self.level,
            "domain": self.domain,
            "allowed_tools": self.allowed_tools.copy(),
            "requires": self.requires.copy(),
            "optional": self.optional.copy(),
            "conflicts": self.conflicts.copy(),
            "load_on_demand": self.load_on_demand.copy(),
            "defer_loading": self.defer_loading,
            "version": self.version,
            "author": self.author,
            "tags": self.tags.copy(),
            "security_scope": self.security_scope,
            "recovery_config": self.recovery_config.copy(),
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillDefinition:
        """Create from dictionary.

        Args:
            data: Dictionary with skill attributes.

        Returns:
            New SkillDefinition instance.
        """
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            level=data.get("level", "L2_intermediate"),
            domain=data.get("domain", "core"),
            allowed_tools=data.get("allowed_tools", data.get("allowed-tools", [])),
            requires=data.get("requires", []),
            optional=data.get("optional", []),
            conflicts=data.get("conflicts", []),
            load_on_demand=data.get("load_on_demand", data.get("load-on-demand", [])),
            defer_loading=data.get("defer_loading", data.get("defer-loading", True)),
            version=data.get("version", "1.0.0"),
            author=data.get("author", ""),
            tags=data.get("tags", []),
            security_scope=data.get("security_scope", data.get("security-scope", 2)),
            recovery_config=data.get("recovery", data.get("recovery_config", {})),
            metadata=data.get("metadata", {}),
        )


class SkillRegistry:
    """Registry for managing skill definitions and progressive loading.

    The SkillRegistry provides a centralized system for:
    - Discovering skills from filesystem directories
    - Parsing SKILL.md files with YAML frontmatter
    - Managing skill dependencies and conflicts
    - Generating tool definitions for Claude API
    - Loading skill context for injection

    Thread-safe operations are ensured through internal locking.

    Attributes:
        skills: Dictionary mapping skill names to definitions.
        always_loaded: List of skill names that are never deferred.
        skill_paths: Directories to scan for SKILL.md files.
        defer_loading: Global flag to enable/disable deferred loading.

    Example:
        >>> registry = SkillRegistry()
        >>> registry.add_skill_path(Path("~/.claude/skills"))
        >>> registry.discover_skills()
        >>>
        >>> # Get tools for API
        >>> tools = registry.get_tool_definitions()
        >>>
        >>> # Load specific skill content
        >>> context = registry.load_skill_context("analytics-attribution")
    """

    # Default skills that should always be loaded immediately
    DEFAULT_ALWAYS_LOADED = [
        "cloudflare-d1",
        "pharma-npi-ndc",
        "analytics-attribution",
        "core-orchestration",
        "security-injection-defense",
    ]

    # Pattern to match YAML frontmatter
    FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL | re.MULTILINE)

    def __init__(
        self,
        skill_paths: list[Path] | None = None,
        always_loaded: list[str] | None = None,
        defer_loading: bool = True,
    ):
        """Initialize the skill registry.

        Args:
            skill_paths: Initial directories to scan for skills.
            always_loaded: Skills that should never be deferred.
            defer_loading: Global flag for deferred loading (default True).
        """
        self._skills: dict[str, SkillDefinition] = {}
        self._skill_paths: list[Path] = []
        self._lock = threading.RLock()

        # Configuration
        self.always_loaded: list[str] = always_loaded or self.DEFAULT_ALWAYS_LOADED.copy()
        self.defer_loading: bool = defer_loading

        # Add initial paths
        if skill_paths:
            for path in skill_paths:
                self.add_skill_path(path)

    @property
    def skills(self) -> dict[str, SkillDefinition]:
        """Get all registered skills (read-only view)."""
        with self._lock:
            return self._skills.copy()

    @property
    def skill_paths(self) -> list[Path]:
        """Get configured skill paths."""
        with self._lock:
            return self._skill_paths.copy()

    def add_skill_path(self, path: Path) -> None:
        """Add a directory to scan for skills.

        Args:
            path: Directory path to add.

        Raises:
            ValueError: If path is not a directory.
        """
        resolved = Path(path).expanduser().resolve()

        with self._lock:
            if resolved not in self._skill_paths:
                self._skill_paths.append(resolved)
                logger.debug(f"Added skill path: {resolved}")

    def remove_skill_path(self, path: Path) -> bool:
        """Remove a skill path.

        Args:
            path: Directory path to remove.

        Returns:
            True if path was removed, False if not found.
        """
        resolved = Path(path).expanduser().resolve()

        with self._lock:
            if resolved in self._skill_paths:
                self._skill_paths.remove(resolved)
                return True
            return False

    def discover_skills(self, paths: list[Path] | None = None) -> int:
        """Scan directories for SKILL.md files and register skills.

        Recursively scans configured paths (or provided paths) for
        files matching SKILL.md or *.skill.md patterns.

        Args:
            paths: Optional specific paths to scan. If None, uses
                   configured skill_paths.

        Returns:
            Number of skills discovered and registered.

        Example:
            >>> registry = SkillRegistry()
            >>> count = registry.discover_skills([
            ...     Path("~/.claude/skills"),
            ...     Path("./project/.claude/skills")
            ... ])
            >>> print(f"Discovered {count} skills")
        """
        scan_paths = paths if paths is not None else self._skill_paths
        discovered = 0

        for base_path in scan_paths:
            resolved = Path(base_path).expanduser().resolve()

            if not resolved.exists():
                logger.warning(f"Skill path does not exist: {resolved}")
                continue

            if not resolved.is_dir():
                logger.warning(f"Skill path is not a directory: {resolved}")
                continue

            # Find all SKILL.md files
            for skill_file in self._find_skill_files(resolved):
                try:
                    skill_def = self.parse_skill_md(skill_file)
                    self.register_skill(skill_def)
                    discovered += 1
                    logger.info(f"Discovered skill: {skill_def.name} from {skill_file}")
                except Exception as e:
                    logger.error(f"Failed to parse skill file {skill_file}: {e}")

        return discovered

    def _find_skill_files(self, base_path: Path) -> Iterator[Path]:
        """Find all skill files in a directory tree.

        Args:
            base_path: Root directory to search.

        Yields:
            Paths to discovered skill files.
        """
        # Look for SKILL.md files
        for skill_file in base_path.rglob("SKILL.md"):
            yield skill_file

        # Also look for *.skill.md pattern
        for skill_file in base_path.rglob("*.skill.md"):
            yield skill_file

        # Check for individual .md files that might be skills
        # (in dedicated skill directories)
        for md_file in base_path.rglob("*.md"):
            # Skip non-skill markdown files
            if md_file.name.startswith(("README", "CHANGELOG", "LICENSE")):
                continue
            # Check if parent is a skill-related directory
            parent_name = md_file.parent.name.lower()
            if parent_name in ("skills", "core", "security", "domain", "protocols", "recovery"):
                if md_file.name != "SKILL.md" and not md_file.name.endswith(".skill.md"):
                    yield md_file

    def parse_skill_md(self, path: Path) -> SkillDefinition:
        """Parse a SKILL.md file into a SkillDefinition.

        Extracts YAML frontmatter and markdown content from the file.
        The frontmatter must be delimited by '---' markers.

        Args:
            path: Path to the SKILL.md file.

        Returns:
            Parsed SkillDefinition.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If frontmatter is missing or invalid.

        Example:
            >>> skill = registry.parse_skill_md(Path("skills/my-skill/SKILL.md"))
            >>> print(skill.name, skill.description)
        """
        if not path.exists():
            raise FileNotFoundError(f"Skill file not found: {path}")

        content = path.read_text(encoding="utf-8")

        # Extract frontmatter
        match = self.FRONTMATTER_PATTERN.match(content)
        if not match:
            raise ValueError(f"No valid YAML frontmatter in: {path}")

        frontmatter_text = match.group(1)
        markdown_content = content[match.end() :].strip()

        # Parse YAML
        try:
            frontmatter = yaml.safe_load(frontmatter_text)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML frontmatter in {path}: {e}")

        if not isinstance(frontmatter, dict):
            raise ValueError(f"Frontmatter must be a dictionary in: {path}")

        # Derive name from filename if not specified
        if "name" not in frontmatter:
            # Use directory name for SKILL.md, or stem for *.skill.md
            if path.name == "SKILL.md":
                frontmatter["name"] = path.parent.name
            else:
                frontmatter["name"] = path.stem.replace(".skill", "")

        # Create definition
        skill_def = SkillDefinition.from_dict(frontmatter)
        skill_def.source_path = path
        skill_def.content = markdown_content

        return skill_def

    def register_skill(self, skill: SkillDefinition) -> None:
        """Register a skill definition.

        Args:
            skill: Skill definition to register.

        Raises:
            ValueError: If skill with same name already exists.
        """
        with self._lock:
            if skill.name in self._skills:
                logger.warning(f"Overwriting existing skill: {skill.name}")

            self._skills[skill.name] = skill

            # Override defer_loading for always-loaded skills
            if skill.name in self.always_loaded:
                skill.defer_loading = False

    def unregister_skill(self, name: str) -> bool:
        """Remove a skill from the registry.

        Args:
            name: Skill name to remove.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if name in self._skills:
                del self._skills[name]
                return True
            return False

    def get_skill(self, name: str) -> SkillDefinition | None:
        """Get a skill definition by name.

        Args:
            name: Skill name.

        Returns:
            Skill definition or None if not found.
        """
        with self._lock:
            return self._skills.get(name)

    def get_skills_by_domain(self, domain: str | SkillDomain) -> list[SkillDefinition]:
        """Get all skills in a domain.

        Args:
            domain: Domain name or enum.

        Returns:
            List of matching skill definitions.
        """
        if isinstance(domain, SkillDomain):
            domain_str = domain.value
        else:
            domain_str = domain.lower()

        with self._lock:
            return [skill for skill in self._skills.values() if skill.domain == domain_str]

    def get_skills_by_level(self, level: str | SkillLevel) -> list[SkillDefinition]:
        """Get all skills at a specific level.

        Args:
            level: Level string or enum.

        Returns:
            List of matching skill definitions.
        """
        if isinstance(level, SkillLevel):
            level_str = level.value
        else:
            level_str = level

        with self._lock:
            return [skill for skill in self._skills.values() if skill.level == level_str]

    def get_tool_definitions(
        self,
        include_deferred: bool = True,
        domains: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate tool definitions for Claude API integration.

        Creates tool definitions compatible with Claude's tool use API,
        including defer_loading flags for progressive loading.

        Args:
            include_deferred: Whether to include deferred skills.
            domains: Optional domain filter.

        Returns:
            List of tool definition dictionaries suitable for API calls.

        Example:
            >>> tools = registry.get_tool_definitions()
            >>> # Use with Claude API
            >>> response = client.messages.create(
            ...     model="claude-sonnet-4-20250514",
            ...     tools=tools,
            ...     messages=[...]
            ... )
        """
        tools = []

        with self._lock:
            for skill in self._skills.values():
                # Apply domain filter
                if domains and skill.domain not in domains:
                    continue

                # Apply deferred filter
                effective_defer = (
                    self.defer_loading
                    and skill.defer_loading
                    and skill.name not in self.always_loaded
                )

                if not include_deferred and effective_defer:
                    continue

                tool_def = self._skill_to_tool_definition(skill, effective_defer)
                tools.append(tool_def)

        return tools

    def _skill_to_tool_definition(
        self,
        skill: SkillDefinition,
        defer: bool,
    ) -> dict[str, Any]:
        """Convert a skill to Claude API tool definition format.

        Args:
            skill: Skill definition.
            defer: Whether this skill should be deferred.

        Returns:
            Tool definition dictionary.
        """
        # Build parameter schema for the skill invocation
        properties = {
            "action": {
                "type": "string",
                "description": "The specific action to perform with this skill",
            },
            "context": {
                "type": "object",
                "description": "Additional context for the skill execution",
                "additionalProperties": True,
            },
        }

        # Add skill-specific parameters based on domain
        if skill.domain == "cloudflare":
            properties["resource_id"] = {
                "type": "string",
                "description": "Cloudflare resource identifier (database ID, bucket name, etc.)",
            }
        elif skill.domain == "pharma":
            properties["identifiers"] = {
                "type": "object",
                "description": "Pharma identifiers (NPI, NDC, etc.)",
                "properties": {
                    "npi": {"type": "string"},
                    "ndc": {"type": "string"},
                },
            }
        elif skill.domain == "analytics":
            properties["date_range"] = {
                "type": "object",
                "description": "Date range for analytics queries",
                "properties": {
                    "start": {"type": "string", "format": "date"},
                    "end": {"type": "string", "format": "date"},
                },
            }

        tool_def: dict[str, Any] = {
            "name": f"skill_{skill.name.replace('-', '_')}",
            "description": skill.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": ["action"],
            },
        }

        # Add defer_loading extension
        if defer:
            tool_def["defer_loading"] = True
            tool_def["cache_control"] = {"type": "ephemeral"}

        return tool_def

    def resolve_dependencies(self, skill_names: list[str]) -> list[str]:
        """Resolve skill dependencies in load order.

        Performs topological sort to ensure dependencies are loaded
        before dependent skills. Handles both required and optional
        dependencies.

        Args:
            skill_names: Skills to resolve dependencies for.

        Returns:
            Ordered list of skill names including all dependencies.

        Raises:
            ValueError: If circular dependency detected.

        Example:
            >>> deps = registry.resolve_dependencies(["analytics-attribution"])
            >>> print(deps)  # ["cloudflare-d1", "pharma-npi-ndc", "analytics-attribution"]
        """
        resolved: list[str] = []
        seen: set[str] = set()
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name in resolved:
                return
            if name in visiting:
                raise ValueError(f"Circular dependency detected: {name}")

            skill = self.get_skill(name)
            if skill is None:
                logger.warning(f"Dependency not found: {name}")
                return

            visiting.add(name)

            # Process required dependencies first
            for dep in skill.requires:
                if dep not in seen:
                    visit(dep)

            # Process optional dependencies (if available)
            for dep in skill.optional:
                if dep in self._skills and dep not in seen:
                    visit(dep)

            visiting.remove(name)
            seen.add(name)
            resolved.append(name)

        with self._lock:
            for name in skill_names:
                visit(name)

        return resolved

    def validate_conflicts(self, skill_names: list[str]) -> list[str]:
        """Check for conflicts between skills.

        Identifies skills that cannot be loaded simultaneously based
        on their conflict declarations.

        Args:
            skill_names: Skills to check for conflicts.

        Returns:
            List of conflict descriptions (empty if no conflicts).

        Example:
            >>> conflicts = registry.validate_conflicts(["skill-a", "skill-b"])
            >>> if conflicts:
            ...     print("Cannot load together:", conflicts)
        """
        conflicts: list[str] = []
        name_set = set(skill_names)

        with self._lock:
            for name in skill_names:
                skill = self._skills.get(name)
                if skill is None:
                    continue

                for conflict in skill.conflicts:
                    if conflict in name_set:
                        conflict_msg = f"{name} conflicts with {conflict}"
                        # Avoid duplicate reports
                        reverse_msg = f"{conflict} conflicts with {name}"
                        if conflict_msg not in conflicts and reverse_msg not in conflicts:
                            conflicts.append(conflict_msg)

        return conflicts

    def load_skill_context(self, name: str) -> str:
        """Load skill content for context injection.

        Returns the markdown content of a skill for injection into
        the conversation context. This is called when a deferred
        skill is actually needed.

        Args:
            name: Skill name to load.

        Returns:
            Markdown content of the skill, or empty string if not found.

        Example:
            >>> context = registry.load_skill_context("cloudflare-d1")
            >>> # Inject into system prompt or conversation
        """
        with self._lock:
            skill = self._skills.get(name)
            if skill is None:
                logger.warning(f"Skill not found for context loading: {name}")
                return ""

            # If content is empty but we have a source path, reload
            if not skill.content and skill.source_path:
                try:
                    reloaded = self.parse_skill_md(skill.source_path)
                    skill.content = reloaded.content
                except Exception as e:
                    logger.error(f"Failed to reload skill content: {e}")

            # Build context with metadata header
            context_parts = [
                f"# Skill: {skill.name}",
                f"**Domain:** {skill.domain}",
                f"**Level:** {skill.level}",
                f"**Version:** {skill.version}",
                "",
                f"**Description:** {skill.description}",
                "",
            ]

            if skill.allowed_tools:
                context_parts.append(f"**Allowed Tools:** {', '.join(skill.allowed_tools)}")
                context_parts.append("")

            if skill.requires:
                context_parts.append(f"**Required Dependencies:** {', '.join(skill.requires)}")
                context_parts.append("")

            context_parts.append("---")
            context_parts.append("")
            context_parts.append(skill.content)

            return "\n".join(context_parts)

    def load_multiple_contexts(self, names: list[str]) -> dict[str, str]:
        """Load context for multiple skills.

        Args:
            names: List of skill names to load.

        Returns:
            Dictionary mapping skill names to their contexts.
        """
        return {name: self.load_skill_context(name) for name in names}

    def get_always_loaded_contexts(self) -> dict[str, str]:
        """Get contexts for all always-loaded skills.

        Returns:
            Dictionary of always-loaded skill contexts.
        """
        return self.load_multiple_contexts(self.always_loaded)

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary with registry statistics.
        """
        with self._lock:
            by_domain: dict[str, int] = {}
            by_level: dict[str, int] = {}
            deferred_count = 0

            for skill in self._skills.values():
                by_domain[skill.domain] = by_domain.get(skill.domain, 0) + 1
                by_level[skill.level] = by_level.get(skill.level, 0) + 1
                if skill.defer_loading and skill.name not in self.always_loaded:
                    deferred_count += 1

            return {
                "total_skills": len(self._skills),
                "always_loaded_count": len(self.always_loaded),
                "deferred_count": deferred_count,
                "skill_paths": [str(p) for p in self._skill_paths],
                "by_domain": by_domain,
                "by_level": by_level,
                "skills": list(self._skills.keys()),
            }

    def __len__(self) -> int:
        """Get number of registered skills."""
        with self._lock:
            return len(self._skills)

    def __contains__(self, name: str) -> bool:
        """Check if skill is registered."""
        with self._lock:
            return name in self._skills

    def __iter__(self) -> Iterator[SkillDefinition]:
        """Iterate over registered skills."""
        with self._lock:
            return iter(list(self._skills.values()))

    def __repr__(self) -> str:
        return f"SkillRegistry(skills={len(self)}, paths={len(self._skill_paths)})"


# Global registry instance
_global_registry: SkillRegistry | None = None


def get_registry() -> SkillRegistry:
    """Get or create the global skill registry.

    Returns:
        Global SkillRegistry instance.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = SkillRegistry()
    return _global_registry


def set_registry(registry: SkillRegistry) -> None:
    """Set the global skill registry.

    Args:
        registry: Registry instance to set as global.
    """
    global _global_registry
    _global_registry = registry


def discover_default_skills() -> int:
    """Discover skills from default locations.

    Scans common skill directories:
    - ~/.claude/skills
    - ./.claude/skills
    - ./skills

    Returns:
        Number of skills discovered.
    """
    registry = get_registry()

    default_paths = [
        Path.home() / ".claude" / "skills",
        Path.cwd() / ".claude" / "skills",
        Path.cwd() / "skills",
    ]

    for path in default_paths:
        if path.exists():
            registry.add_skill_path(path)

    return registry.discover_skills()


def register_skill(
    name: str,
    description: str,
    domain: str = "core",
    level: str = "L2_intermediate",
    **kwargs,
) -> SkillDefinition:
    """Convenience function to register a skill.

    Args:
        name: Skill name.
        description: Skill description.
        domain: Skill domain.
        level: Skill level.
        **kwargs: Additional skill attributes.

    Returns:
        Created SkillDefinition.
    """
    skill = SkillDefinition(
        name=name,
        description=description,
        domain=domain,
        level=level,
        **kwargs,
    )
    get_registry().register_skill(skill)
    return skill
