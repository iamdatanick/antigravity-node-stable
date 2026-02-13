"""
Skill Types for OpenAI Skills SDK Integration.

This module provides dataclass definitions for skill manifests, instructions,
resources, and tools compatible with both Anthropic SKILL.md and OpenAI Codex
skill formats following the agentskills.io open standard.

Supported Formats:
- Anthropic SKILL.md (Claude Code)
- OpenAI Codex Skills (.codex/skills)
- agentskills.io standard specification

Example:
    >>> from agentic_workflows.openai_skills.skill_types import (
    ...     SkillManifest,
    ...     SkillInstruction,
    ...     SkillResource,
    ...     SkillTool,
    ... )
    >>> manifest = SkillManifest(
    ...     name="pdf-processing",
    ...     description="Extract text and tables from PDF files.",
    ...     version="1.0.0",
    ... )

Author: Agentic Workflows Contributors
Version: 1.0.0
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class SkillCategory(Enum):
    """Skill category for organization.

    Categories follow the OpenAI skills repository structure:
    - SYSTEM: Automatically installed system skills (.system)
    - CURATED: Community-approved skills (.curated)
    - EXPERIMENTAL: Development-stage skills (.experimental)
    - USER: User-defined skills (~/.codex/skills or ~/.claude/skills)
    - PROJECT: Project-specific skills (.codex/skills or .claude/skills)
    """

    SYSTEM = "system"
    CURATED = "curated"
    EXPERIMENTAL = "experimental"
    USER = "user"
    PROJECT = "project"


class ResourceType(Enum):
    """Type of skill resource.

    Following the agentskills.io specification:
    - SCRIPT: Executable code in scripts/ directory
    - REFERENCE: Documentation in references/ directory
    - ASSET: Static resources in assets/ directory
    - TEMPLATE: Configuration or document templates
    - SCHEMA: JSON/YAML schemas
    - DATA: Lookup tables, canned examples
    """

    SCRIPT = "script"
    REFERENCE = "reference"
    ASSET = "asset"
    TEMPLATE = "template"
    SCHEMA = "schema"
    DATA = "data"


class ToolPermission(Enum):
    """Tool permission levels for skills.

    Defines what tools a skill is allowed to invoke.
    """

    NONE = "none"
    READ_ONLY = "read_only"
    WRITE = "write"
    EXECUTE = "execute"
    FULL = "full"


# Name validation pattern per agentskills.io spec
NAME_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024


def validate_skill_name(name: str) -> bool:
    """Validate skill name per agentskills.io specification.

    Rules:
    - Max 64 characters
    - Lowercase letters, numbers, and hyphens only
    - Must not start/end with hyphen
    - No consecutive hyphens

    Args:
        name: Skill name to validate.

    Returns:
        True if valid, False otherwise.
    """
    if not name or len(name) > MAX_NAME_LENGTH:
        return False
    if "--" in name:
        return False
    return bool(NAME_PATTERN.match(name))


@dataclass
class SkillMetadata:
    """Additional metadata for a skill.

    Attributes:
        author: Skill author or organization.
        version: Semantic version string.
        license: License identifier (e.g., Apache-2.0, MIT).
        homepage: URL to skill homepage or repository.
        repository: Git repository URL.
        keywords: Search keywords for discovery.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        extra: Additional arbitrary key-value metadata.
    """

    author: str = ""
    version: str = "1.0.0"
    license: str = ""
    homepage: str = ""
    repository: str = ""
    keywords: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result: dict[str, Any] = {}
        if self.author:
            result["author"] = self.author
        if self.version:
            result["version"] = self.version
        if self.license:
            result["license"] = self.license
        if self.homepage:
            result["homepage"] = self.homepage
        if self.repository:
            result["repository"] = self.repository
        if self.keywords:
            result["keywords"] = self.keywords.copy()
        if self.created_at:
            result["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            result["updated_at"] = self.updated_at.isoformat()
        if self.extra:
            result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillMetadata:
        """Create from dictionary."""
        created = data.get("created_at")
        updated = data.get("updated_at")

        # Parse datetime strings if present
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        if isinstance(updated, str):
            updated = datetime.fromisoformat(updated)

        # Extract known fields, rest goes to extra
        known_fields = {
            "author",
            "version",
            "license",
            "homepage",
            "repository",
            "keywords",
            "created_at",
            "updated_at",
        }
        extra = {k: v for k, v in data.items() if k not in known_fields}

        return cls(
            author=data.get("author", ""),
            version=data.get("version", "1.0.0"),
            license=data.get("license", ""),
            homepage=data.get("homepage", ""),
            repository=data.get("repository", ""),
            keywords=data.get("keywords", []),
            created_at=created,
            updated_at=updated,
            extra=extra,
        )


@dataclass
class SkillTool:
    """Tool definition within a skill.

    Represents a tool that the skill can invoke or provides.

    Attributes:
        name: Tool name (e.g., "Bash", "Read", "Write").
        pattern: Optional glob pattern for tool invocation (e.g., "git:*").
        permission: Permission level for this tool.
        description: What this tool does within the skill context.
        parameters: Tool parameter schema.
    """

    name: str
    pattern: str = "*"
    permission: ToolPermission = ToolPermission.EXECUTE
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def full_spec(self) -> str:
        """Get full tool specification string (e.g., 'Bash(git:*)').

        Following the agentskills.io allowed-tools format.
        """
        if self.pattern and self.pattern != "*":
            return f"{self.name}({self.pattern})"
        return self.name

    @classmethod
    def parse(cls, spec: str) -> SkillTool:
        """Parse a tool specification string.

        Args:
            spec: Tool spec like 'Bash(git:*)' or 'Read'.

        Returns:
            SkillTool instance.
        """
        # Match pattern like "Bash(git:*)"
        match = re.match(r"(\w+)\(([^)]+)\)", spec)
        if match:
            return cls(name=match.group(1), pattern=match.group(2))
        return cls(name=spec)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "pattern": self.pattern,
            "permission": self.permission.value,
            "description": self.description,
            "parameters": self.parameters.copy(),
        }


@dataclass
class SkillResource:
    """Resource associated with a skill.

    Resources are files in scripts/, references/, or assets/ directories
    that support the skill's functionality.

    Attributes:
        path: Relative path from skill root (e.g., "scripts/extract.py").
        resource_type: Type of resource (script, reference, asset).
        description: What this resource provides.
        content: Raw content (loaded lazily).
        encoding: Content encoding (default utf-8).
        executable: Whether the resource is executable (for scripts).
        dependencies: External dependencies required (for scripts).
    """

    path: str
    resource_type: ResourceType = ResourceType.REFERENCE
    description: str = ""
    content: str = ""
    encoding: str = "utf-8"
    executable: bool = False
    dependencies: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Get resource filename."""
        return Path(self.path).name

    @property
    def extension(self) -> str:
        """Get file extension."""
        return Path(self.path).suffix

    @property
    def language(self) -> str:
        """Infer programming language for scripts."""
        ext_map = {
            ".py": "python",
            ".sh": "bash",
            ".bash": "bash",
            ".js": "javascript",
            ".ts": "typescript",
            ".rb": "ruby",
            ".go": "go",
            ".rs": "rust",
        }
        return ext_map.get(self.extension.lower(), "text")

    @classmethod
    def from_path(cls, skill_root: Path, resource_path: Path) -> SkillResource:
        """Create from file path.

        Args:
            skill_root: Root directory of the skill.
            resource_path: Path to the resource file.

        Returns:
            SkillResource instance.
        """
        relative = resource_path.relative_to(skill_root)
        rel_str = str(relative).replace("\\", "/")

        # Determine resource type from directory
        first_dir = relative.parts[0].lower() if relative.parts else ""
        type_map = {
            "scripts": ResourceType.SCRIPT,
            "references": ResourceType.REFERENCE,
            "assets": ResourceType.ASSET,
        }
        resource_type = type_map.get(first_dir, ResourceType.REFERENCE)

        return cls(
            path=rel_str,
            resource_type=resource_type,
            executable=resource_type == ResourceType.SCRIPT,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "path": self.path,
            "type": self.resource_type.value,
            "description": self.description,
            "executable": self.executable,
            "dependencies": self.dependencies.copy(),
        }


@dataclass
class SkillInstruction:
    """Skill-specific instruction or workflow step.

    Instructions define the step-by-step guidance provided in the
    SKILL.md body content.

    Attributes:
        title: Instruction title or heading.
        content: Markdown content of the instruction.
        order: Sequence order for multi-step instructions.
        conditions: When this instruction applies.
        examples: Input/output examples.
        edge_cases: Known edge cases to handle.
    """

    title: str
    content: str
    order: int = 0
    conditions: list[str] = field(default_factory=list)
    examples: list[dict[str, str]] = field(default_factory=list)
    edge_cases: list[str] = field(default_factory=list)

    @classmethod
    def from_markdown_section(cls, heading: str, content: str, order: int = 0) -> SkillInstruction:
        """Parse from a markdown section.

        Args:
            heading: Section heading text.
            content: Section content.
            order: Section order.

        Returns:
            SkillInstruction instance.
        """
        return cls(
            title=heading.strip("#").strip(),
            content=content.strip(),
            order=order,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "title": self.title,
            "content": self.content,
            "order": self.order,
            "conditions": self.conditions.copy(),
            "examples": [e.copy() for e in self.examples],
            "edge_cases": self.edge_cases.copy(),
        }


@dataclass
class SkillTrigger:
    """Trigger condition for skill activation.

    Defines when a skill should be activated based on user input
    or context patterns.

    Attributes:
        keywords: Keywords that trigger the skill.
        patterns: Regex patterns for matching user input.
        file_types: File extensions that activate the skill.
        domains: Domain categories (e.g., "pdf", "data-analysis").
        priority: Trigger priority (higher = more specific).
    """

    keywords: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    file_types: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    priority: int = 50

    def matches(self, text: str) -> bool:
        """Check if text matches any trigger condition.

        Args:
            text: User input or context text.

        Returns:
            True if any trigger matches.
        """
        text_lower = text.lower()

        # Check keywords
        for keyword in self.keywords:
            if keyword.lower() in text_lower:
                return True

        # Check patterns
        for pattern in self.patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "keywords": self.keywords.copy(),
            "patterns": self.patterns.copy(),
            "file_types": self.file_types.copy(),
            "domains": self.domains.copy(),
            "priority": self.priority,
        }

    @classmethod
    def from_description(cls, description: str) -> SkillTrigger:
        """Extract trigger keywords from skill description.

        Per agentskills.io, the description should include keywords
        that help agents identify relevant tasks.

        Args:
            description: Skill description text.

        Returns:
            SkillTrigger with extracted keywords.
        """
        # Common action words and domain terms to extract
        action_patterns = [
            r"\b(extract|parse|convert|create|generate|analyze|validate)\b",
            r"\b(pdf|csv|json|xml|html|markdown)\b",
            r"\b(file|document|data|table|form|report)\b",
        ]

        keywords = []
        for pattern in action_patterns:
            matches = re.findall(pattern, description.lower())
            keywords.extend(matches)

        return cls(keywords=list(set(keywords)))


@dataclass
class SkillManifest:
    """Complete skill manifest combining metadata and content.

    This is the primary dataclass representing a skill, combining
    YAML frontmatter fields and markdown body content per the
    agentskills.io specification.

    Attributes:
        name: Skill name (max 64 chars, lowercase, hyphens allowed).
        description: What the skill does and when to use it (max 1024 chars).
        version: Semantic version string.
        triggers: Trigger conditions for skill activation.
        tools: Tools the skill can invoke (allowed-tools).
        instructions: Parsed instruction sections from body.
        resources: Associated scripts, references, and assets.
        metadata: Additional metadata (author, license, etc.).
        compatibility: Environment requirements.
        category: Skill category (system, curated, experimental, user, project).
        source_path: Path to the SKILL.md file.
        raw_content: Original markdown body content.
        loaded: Whether full content has been loaded.

    Example:
        >>> manifest = SkillManifest(
        ...     name="pdf-processing",
        ...     description="Extract text and tables from PDF files.",
        ...     tools=[SkillTool(name="Bash", pattern="pdftotext:*")],
        ... )
        >>> manifest.is_valid
        True
    """

    name: str
    description: str
    version: str = "1.0.0"
    triggers: SkillTrigger = field(default_factory=SkillTrigger)
    tools: list[SkillTool] = field(default_factory=list)
    instructions: list[SkillInstruction] = field(default_factory=list)
    resources: list[SkillResource] = field(default_factory=list)
    metadata: SkillMetadata = field(default_factory=SkillMetadata)
    compatibility: str = ""
    category: SkillCategory = SkillCategory.USER
    source_path: Path | None = None
    raw_content: str = ""
    loaded: bool = False

    def __post_init__(self):
        """Validate and normalize manifest fields."""
        # Normalize name
        self.name = self.name.lower().strip()

        # Extract triggers from description if not provided
        if not self.triggers.keywords and self.description:
            self.triggers = SkillTrigger.from_description(self.description)

    @property
    def is_valid(self) -> bool:
        """Check if manifest meets agentskills.io requirements."""
        if not validate_skill_name(self.name):
            return False
        if not self.description or len(self.description) > MAX_DESCRIPTION_LENGTH:
            return False
        return True

    @property
    def validation_errors(self) -> list[str]:
        """Get list of validation errors."""
        errors = []

        if not self.name:
            errors.append("Name is required")
        elif len(self.name) > MAX_NAME_LENGTH:
            errors.append(f"Name exceeds {MAX_NAME_LENGTH} characters")
        elif "--" in self.name:
            errors.append("Name cannot contain consecutive hyphens")
        elif not NAME_PATTERN.match(self.name):
            errors.append("Name must be lowercase with hyphens only")

        if not self.description:
            errors.append("Description is required")
        elif len(self.description) > MAX_DESCRIPTION_LENGTH:
            errors.append(f"Description exceeds {MAX_DESCRIPTION_LENGTH} characters")

        return errors

    @property
    def allowed_tools_string(self) -> str:
        """Get allowed-tools field as space-delimited string.

        Per agentskills.io format: "Bash(git:*) Bash(jq:*) Read"
        """
        return " ".join(tool.full_spec for tool in self.tools)

    @property
    def scripts(self) -> list[SkillResource]:
        """Get all script resources."""
        return [r for r in self.resources if r.resource_type == ResourceType.SCRIPT]

    @property
    def references(self) -> list[SkillResource]:
        """Get all reference resources."""
        return [r for r in self.resources if r.resource_type == ResourceType.REFERENCE]

    @property
    def assets(self) -> list[SkillResource]:
        """Get all asset resources."""
        return [r for r in self.resources if r.resource_type == ResourceType.ASSET]

    @property
    def full_name(self) -> str:
        """Get fully qualified name including category."""
        return f"{self.category.value}:{self.name}"

    def get_context(self, include_resources: bool = False) -> str:
        """Generate context string for agent prompt injection.

        Args:
            include_resources: Whether to include resource contents.

        Returns:
            Formatted context string for agent use.
        """
        parts = [
            f"# Skill: {self.name}",
            f"**Description:** {self.description}",
            f"**Version:** {self.version}",
        ]

        if self.compatibility:
            parts.append(f"**Compatibility:** {self.compatibility}")

        if self.tools:
            parts.append(f"**Allowed Tools:** {self.allowed_tools_string}")

        parts.append("")
        parts.append("---")
        parts.append("")

        # Add main content
        if self.raw_content:
            parts.append(self.raw_content)
        elif self.instructions:
            for inst in sorted(self.instructions, key=lambda x: x.order):
                parts.append(f"## {inst.title}")
                parts.append(inst.content)
                parts.append("")

        # Optionally add resources
        if include_resources:
            for resource in self.scripts:
                if resource.content:
                    parts.append(f"\n### Script: {resource.path}")
                    parts.append(f"```{resource.language}")
                    parts.append(resource.content)
                    parts.append("```")

        return "\n".join(parts)

    def to_frontmatter(self) -> dict[str, Any]:
        """Convert to YAML frontmatter dictionary.

        Returns:
            Dictionary suitable for YAML serialization.
        """
        fm: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
        }

        if self.metadata.license:
            fm["license"] = self.metadata.license

        if self.compatibility:
            fm["compatibility"] = self.compatibility

        if self.tools:
            fm["allowed-tools"] = self.allowed_tools_string

        # Add metadata section if present
        meta_dict = self.metadata.to_dict()
        if meta_dict:
            # Remove license as it's top-level
            meta_dict.pop("license", None)
            if meta_dict:
                fm["metadata"] = meta_dict

        return fm

    def to_skill_md(self) -> str:
        """Generate complete SKILL.md file content.

        Returns:
            String with YAML frontmatter and markdown body.
        """
        import yaml

        frontmatter = yaml.dump(
            self.to_frontmatter(),
            default_flow_style=False,
            sort_keys=False,
        )

        return f"---\n{frontmatter}---\n\n{self.raw_content}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to complete dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "triggers": self.triggers.to_dict(),
            "tools": [t.to_dict() for t in self.tools],
            "instructions": [i.to_dict() for i in self.instructions],
            "resources": [r.to_dict() for r in self.resources],
            "metadata": self.metadata.to_dict(),
            "compatibility": self.compatibility,
            "category": self.category.value,
            "source_path": str(self.source_path) if self.source_path else None,
            "loaded": self.loaded,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillManifest:
        """Create from dictionary representation.

        Args:
            data: Dictionary with manifest fields.

        Returns:
            SkillManifest instance.
        """
        # Parse tools
        tools = []
        if "allowed-tools" in data:
            # Parse space-delimited string
            for spec in data["allowed-tools"].split():
                tools.append(SkillTool.parse(spec))
        elif "tools" in data:
            # Parse list of dicts
            for t in data["tools"]:
                if isinstance(t, dict):
                    tools.append(
                        SkillTool(
                            name=t.get("name", ""),
                            pattern=t.get("pattern", "*"),
                            permission=ToolPermission(t.get("permission", "execute")),
                            description=t.get("description", ""),
                            parameters=t.get("parameters", {}),
                        )
                    )
                elif isinstance(t, str):
                    tools.append(SkillTool.parse(t))

        # Parse triggers
        triggers_data = data.get("triggers", {})
        triggers = SkillTrigger(
            keywords=triggers_data.get("keywords", []),
            patterns=triggers_data.get("patterns", []),
            file_types=triggers_data.get("file_types", []),
            domains=triggers_data.get("domains", []),
            priority=triggers_data.get("priority", 50),
        )

        # Parse metadata
        metadata_data = data.get("metadata", {})
        if not isinstance(metadata_data, dict):
            metadata_data = {}

        # Handle top-level metadata fields
        if "author" in data:
            metadata_data.setdefault("author", data["author"])
        if "version" in data:
            metadata_data.setdefault("version", data["version"])
        if "license" in data:
            metadata_data.setdefault("license", data["license"])

        metadata = SkillMetadata.from_dict(metadata_data)

        # Parse category
        category_str = data.get("category", "user")
        try:
            category = SkillCategory(category_str)
        except ValueError:
            category = SkillCategory.USER

        # Parse source path
        source_path = None
        if data.get("source_path"):
            source_path = Path(data["source_path"])

        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", metadata.version),
            triggers=triggers,
            tools=tools,
            metadata=metadata,
            compatibility=data.get("compatibility", ""),
            category=category,
            source_path=source_path,
            raw_content=data.get("raw_content", data.get("content", "")),
            loaded=data.get("loaded", False),
        )


# Type aliases for convenience
SkillDict = dict[str, Any]
SkillList = list[SkillManifest]
