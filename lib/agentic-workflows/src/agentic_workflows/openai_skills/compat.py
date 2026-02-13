"""
Compatibility Layer for OpenAI Skills SDK Integration.

This module provides conversion utilities between different skill formats:
- Anthropic SKILL.md format (Claude Code)
- OpenAI Codex skills format (instructions + scripts)
- agentskills.io open standard
- agentic_workflows.skills.SkillDefinition format

Cross-Platform Support:
- Convert between formats while preserving metadata
- Support discovery from multiple platform directories
- Generate compatible output for different agents

Example:
    >>> from agentic_workflows.openai_skills.compat import SkillConverter
    >>> converter = SkillConverter()
    >>>
    >>> # Convert OpenAI format to SKILL.md
    >>> skill_md = converter.openai_to_skill_md(openai_skill)
    >>>
    >>> # Convert to agentic_workflows format
    >>> skill_def = converter.to_agentic_definition(manifest)

Author: Agentic Workflows Contributors
Version: 1.0.0
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from agentic_workflows.openai_skills.skill_types import (
    ResourceType,
    SkillCategory,
    SkillInstruction,
    SkillManifest,
    SkillMetadata,
    SkillResource,
    SkillTool,
    SkillTrigger,
)


logger = logging.getLogger(__name__)


class SkillFormat(Enum):
    """Supported skill format types."""
    ANTHROPIC_SKILL_MD = "anthropic_skill_md"  # SKILL.md with YAML frontmatter
    OPENAI_CODEX = "openai_codex"              # instructions/ + scripts/ folders
    AGENTSKILLS_IO = "agentskills_io"          # agentskills.io standard
    AGENTIC_WORKFLOWS = "agentic_workflows"    # agentic_workflows.skills format
    CLAUDE_COMMANDS = "claude_commands"        # .claude/commands format
    GENERIC_MARKDOWN = "generic_markdown"      # Plain markdown instructions


@dataclass
class ConversionResult:
    """Result of a format conversion operation.

    Attributes:
        success: Whether conversion succeeded.
        source_format: Original format.
        target_format: Target format.
        manifest: Converted manifest (if successful).
        output_path: Output path (for file conversions).
        errors: List of conversion errors.
        warnings: List of conversion warnings.
    """
    success: bool
    source_format: SkillFormat
    target_format: SkillFormat
    manifest: SkillManifest | None = None
    output_path: Path | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class FormatDetector:
    """Detect skill format from directory or file structure."""

    @staticmethod
    def detect(path: Path) -> SkillFormat | None:
        """Detect skill format from path.

        Args:
            path: Path to skill file or directory.

        Returns:
            Detected format or None if unknown.
        """
        if path.is_file():
            return FormatDetector._detect_from_file(path)
        elif path.is_dir():
            return FormatDetector._detect_from_directory(path)
        return None

    @staticmethod
    def _detect_from_file(path: Path) -> SkillFormat | None:
        """Detect format from a file.

        Args:
            path: File path.

        Returns:
            Detected format.
        """
        if path.name.upper() == "SKILL.MD" or path.name.endswith(".skill.md"):
            # Check if it has YAML frontmatter
            try:
                content = path.read_text(encoding="utf-8")
                if content.startswith("---"):
                    return SkillFormat.AGENTSKILLS_IO
            except Exception:
                pass
            return SkillFormat.GENERIC_MARKDOWN

        if path.suffix.lower() == ".md":
            return SkillFormat.GENERIC_MARKDOWN

        return None

    @staticmethod
    def _detect_from_directory(path: Path) -> SkillFormat | None:
        """Detect format from a directory.

        Args:
            path: Directory path.

        Returns:
            Detected format.
        """
        # Check for SKILL.md (agentskills.io / Anthropic)
        skill_md = path / "SKILL.md"
        if not skill_md.exists():
            for f in path.iterdir():
                if f.name.upper() == "SKILL.MD":
                    skill_md = f
                    break

        if skill_md.exists():
            try:
                content = skill_md.read_text(encoding="utf-8")
                if content.startswith("---"):
                    return SkillFormat.AGENTSKILLS_IO
            except Exception:
                pass
            return SkillFormat.ANTHROPIC_SKILL_MD

        # Check for OpenAI Codex format (instructions/ folder)
        if (path / "instructions").is_dir():
            return SkillFormat.OPENAI_CODEX

        # Check for Claude commands format
        if path.name == "commands" or (path / "commands").is_dir():
            return SkillFormat.CLAUDE_COMMANDS

        return None


class SkillConverter:
    """Converter for transforming between skill formats.

    The SkillConverter handles bidirectional conversion between
    supported skill formats while preserving as much metadata
    as possible.

    Supported Conversions:
    - OpenAI Codex -> SKILL.md (agentskills.io)
    - SKILL.md -> OpenAI Codex
    - Any -> agentic_workflows.skills.SkillDefinition
    - SkillManifest <-> All formats

    Example:
        >>> converter = SkillConverter()
        >>>
        >>> # Convert directory
        >>> result = converter.convert_directory(
        ...     source_dir,
        ...     target_dir,
        ...     SkillFormat.AGENTSKILLS_IO
        ... )
    """

    def __init__(self):
        """Initialize the converter."""
        self.detector = FormatDetector()

    def detect_format(self, path: Path) -> SkillFormat | None:
        """Detect skill format from path.

        Args:
            path: Path to skill.

        Returns:
            Detected format or None.
        """
        return self.detector.detect(path)

    def convert_to_skill_md(
        self,
        manifest: SkillManifest,
        output_dir: Path | None = None,
    ) -> ConversionResult:
        """Convert a skill manifest to SKILL.md format.

        Generates a SKILL.md file following the agentskills.io
        specification with YAML frontmatter and markdown body.

        Args:
            manifest: Skill manifest to convert.
            output_dir: Optional output directory.

        Returns:
            Conversion result.
        """
        errors: list[str] = []
        warnings: list[str] = []

        try:
            # Build frontmatter
            frontmatter: dict[str, Any] = {
                "name": manifest.name,
                "description": manifest.description,
            }

            if manifest.metadata.license:
                frontmatter["license"] = manifest.metadata.license

            if manifest.compatibility:
                frontmatter["compatibility"] = manifest.compatibility

            if manifest.tools:
                frontmatter["allowed-tools"] = manifest.allowed_tools_string

            # Add metadata section
            meta: dict[str, Any] = {}
            if manifest.metadata.author:
                meta["author"] = manifest.metadata.author
            if manifest.version and manifest.version != "1.0.0":
                meta["version"] = manifest.version
            if manifest.metadata.keywords:
                meta["keywords"] = manifest.metadata.keywords

            if meta:
                frontmatter["metadata"] = meta

            # Generate YAML frontmatter
            frontmatter_yaml = yaml.dump(
                frontmatter,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

            # Build body content
            body_parts: list[str] = []

            if manifest.instructions:
                for inst in sorted(manifest.instructions, key=lambda x: x.order):
                    body_parts.append(f"## {inst.title}")
                    body_parts.append("")
                    body_parts.append(inst.content)
                    body_parts.append("")
            elif manifest.raw_content:
                body_parts.append(manifest.raw_content)

            body = "\n".join(body_parts)

            # Combine
            skill_md_content = f"---\n{frontmatter_yaml}---\n\n{body}"

            # Write if output_dir specified
            output_path = None
            if output_dir:
                output_dir = Path(output_dir)
                skill_dir = output_dir / manifest.name
                skill_dir.mkdir(parents=True, exist_ok=True)

                output_path = skill_dir / "SKILL.md"
                output_path.write_text(skill_md_content, encoding="utf-8")

                # Copy resources
                if manifest.source_path:
                    source_dir = manifest.source_path.parent
                    for resource in manifest.resources:
                        src_resource = source_dir / resource.path
                        if src_resource.exists():
                            dst_resource = skill_dir / resource.path
                            dst_resource.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src_resource, dst_resource)

            # Update manifest with new content
            manifest.raw_content = body

            return ConversionResult(
                success=True,
                source_format=SkillFormat.AGENTIC_WORKFLOWS,
                target_format=SkillFormat.AGENTSKILLS_IO,
                manifest=manifest,
                output_path=output_path,
                warnings=warnings,
            )

        except Exception as e:
            errors.append(str(e))
            return ConversionResult(
                success=False,
                source_format=SkillFormat.AGENTIC_WORKFLOWS,
                target_format=SkillFormat.AGENTSKILLS_IO,
                errors=errors,
            )

    def convert_to_openai_codex(
        self,
        manifest: SkillManifest,
        output_dir: Path,
    ) -> ConversionResult:
        """Convert a skill manifest to OpenAI Codex format.

        Creates an instructions/ folder structure with separate
        markdown files for each instruction section.

        Args:
            manifest: Skill manifest to convert.
            output_dir: Output directory.

        Returns:
            Conversion result.
        """
        errors: list[str] = []
        warnings: list[str] = []

        try:
            output_dir = Path(output_dir)
            skill_dir = output_dir / manifest.name
            skill_dir.mkdir(parents=True, exist_ok=True)

            # Create instructions directory
            instructions_dir = skill_dir / "instructions"
            instructions_dir.mkdir(exist_ok=True)

            # Write instructions
            if manifest.instructions:
                for idx, inst in enumerate(sorted(manifest.instructions, key=lambda x: x.order)):
                    filename = f"{idx:02d}-{self._slugify(inst.title)}.md"
                    inst_path = instructions_dir / filename
                    content = f"# {inst.title}\n\n{inst.content}"
                    inst_path.write_text(content, encoding="utf-8")
            elif manifest.raw_content:
                # Single instruction file
                main_path = instructions_dir / "00-main.md"
                main_path.write_text(manifest.raw_content, encoding="utf-8")

            # Create scripts directory if needed
            scripts = [r for r in manifest.resources if r.resource_type == ResourceType.SCRIPT]
            if scripts:
                scripts_dir = skill_dir / "scripts"
                scripts_dir.mkdir(exist_ok=True)

                if manifest.source_path:
                    source_dir = manifest.source_path.parent
                    for resource in scripts:
                        src_path = source_dir / resource.path
                        if src_path.exists():
                            dst_path = scripts_dir / Path(resource.path).name
                            shutil.copy2(src_path, dst_path)
                        elif resource.content:
                            dst_path = scripts_dir / Path(resource.path).name
                            dst_path.write_text(resource.content, encoding="utf-8")

            # Create README.md
            readme_content = f"# {manifest.name}\n\n{manifest.description}\n"
            if manifest.version:
                readme_content += f"\nVersion: {manifest.version}\n"
            if manifest.metadata.author:
                readme_content += f"Author: {manifest.metadata.author}\n"

            readme_path = skill_dir / "README.md"
            readme_path.write_text(readme_content, encoding="utf-8")

            return ConversionResult(
                success=True,
                source_format=SkillFormat.AGENTSKILLS_IO,
                target_format=SkillFormat.OPENAI_CODEX,
                manifest=manifest,
                output_path=skill_dir,
                warnings=warnings,
            )

        except Exception as e:
            errors.append(str(e))
            return ConversionResult(
                success=False,
                source_format=SkillFormat.AGENTSKILLS_IO,
                target_format=SkillFormat.OPENAI_CODEX,
                errors=errors,
            )

    def convert_to_agentic_definition(
        self,
        manifest: SkillManifest,
    ) -> Any:
        """Convert to agentic_workflows.skills.SkillDefinition.

        Args:
            manifest: Skill manifest to convert.

        Returns:
            SkillDefinition instance or dict if module unavailable.
        """
        try:
            from agentic_workflows.skills import SkillDefinition

            return SkillDefinition(
                name=manifest.name,
                description=manifest.description,
                domain=manifest.metadata.extra.get("domain", "core"),
                level="L2_intermediate",
                version=manifest.version,
                author=manifest.metadata.author,
                tags=manifest.metadata.keywords,
                allowed_tools=[t.full_spec for t in manifest.tools],
                defer_loading=not manifest.loaded,
                source_path=manifest.source_path,
                content=manifest.raw_content,
            )

        except ImportError:
            # Return dict representation
            return {
                "name": manifest.name,
                "description": manifest.description,
                "domain": manifest.metadata.extra.get("domain", "core"),
                "level": "L2_intermediate",
                "version": manifest.version,
                "author": manifest.metadata.author,
                "tags": manifest.metadata.keywords,
                "allowed_tools": [t.full_spec for t in manifest.tools],
                "defer_loading": not manifest.loaded,
                "source_path": str(manifest.source_path) if manifest.source_path else None,
                "content": manifest.raw_content,
            }

    def from_agentic_definition(
        self,
        skill_def: Any,
    ) -> SkillManifest:
        """Convert from agentic_workflows.skills.SkillDefinition.

        Args:
            skill_def: SkillDefinition instance or dict.

        Returns:
            SkillManifest instance.
        """
        # Handle both object and dict
        if hasattr(skill_def, "to_dict"):
            data = skill_def.to_dict()
        elif isinstance(skill_def, dict):
            data = skill_def
        else:
            # Try attribute access
            data = {
                "name": getattr(skill_def, "name", ""),
                "description": getattr(skill_def, "description", ""),
                "domain": getattr(skill_def, "domain", "core"),
                "level": getattr(skill_def, "level", "L2_intermediate"),
                "version": getattr(skill_def, "version", "1.0.0"),
                "author": getattr(skill_def, "author", ""),
                "tags": getattr(skill_def, "tags", []),
                "allowed_tools": getattr(skill_def, "allowed_tools", []),
                "content": getattr(skill_def, "content", ""),
            }

        # Parse tools
        tools: list[SkillTool] = []
        for tool_spec in data.get("allowed_tools", []):
            tools.append(SkillTool.parse(str(tool_spec)))

        # Build metadata
        metadata = SkillMetadata(
            author=data.get("author", ""),
            version=data.get("version", "1.0.0"),
            keywords=data.get("tags", []),
            extra={"domain": data.get("domain", "core")},
        )

        # Get source path
        source_path = data.get("source_path")
        if source_path and isinstance(source_path, str):
            source_path = Path(source_path)

        return SkillManifest(
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            tools=tools,
            metadata=metadata,
            source_path=source_path,
            raw_content=data.get("content", ""),
            loaded=bool(data.get("content")),
        )

    def convert_directory(
        self,
        source_dir: Path,
        target_dir: Path,
        target_format: SkillFormat,
    ) -> ConversionResult:
        """Convert a skill directory to another format.

        Args:
            source_dir: Source skill directory.
            target_dir: Target output directory.
            target_format: Target format.

        Returns:
            Conversion result.
        """
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)

        # Detect source format
        source_format = self.detect_format(source_dir)
        if source_format is None:
            return ConversionResult(
                success=False,
                source_format=SkillFormat.GENERIC_MARKDOWN,
                target_format=target_format,
                errors=["Could not detect source format"],
            )

        # Load manifest
        from agentic_workflows.openai_skills.loader import SkillLoader

        loader = SkillLoader()

        # Find SKILL.md or parse Codex format
        skill_md = source_dir / "SKILL.md"
        if not skill_md.exists():
            for f in source_dir.iterdir():
                if f.name.upper() == "SKILL.MD":
                    skill_md = f
                    break

        manifest = None
        if skill_md.exists():
            manifest = loader.load_skill_by_path(skill_md)
        elif source_format == SkillFormat.OPENAI_CODEX:
            # Load from instructions/ folder
            loader.add_path(source_dir.parent, SkillCategory.USER, recursive=False)
            manifests = loader.discover_all()
            if manifests:
                manifest = manifests[0]

        if manifest is None:
            return ConversionResult(
                success=False,
                source_format=source_format,
                target_format=target_format,
                errors=["Failed to load skill from source directory"],
            )

        # Convert to target format
        if target_format == SkillFormat.AGENTSKILLS_IO:
            return self.convert_to_skill_md(manifest, target_dir)
        elif target_format == SkillFormat.OPENAI_CODEX:
            return self.convert_to_openai_codex(manifest, target_dir)
        else:
            return ConversionResult(
                success=False,
                source_format=source_format,
                target_format=target_format,
                errors=[f"Unsupported target format: {target_format}"],
            )

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to URL-friendly slug.

        Args:
            text: Text to slugify.

        Returns:
            Slugified string.
        """
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[\s_]+", "-", text)
        text = re.sub(r"-+", "-", text)
        return text.strip("-")


class CrossPlatformDiscovery:
    """Cross-platform skill discovery across Claude Code and Codex.

    This class provides unified skill discovery from multiple
    platform-specific locations with format normalization.

    Supported Platforms:
    - Claude Code (~/.claude/skills, .claude/skills)
    - OpenAI Codex (~/.codex/skills, .codex/skills)
    - agentskills.io compliant directories

    Example:
        >>> discovery = CrossPlatformDiscovery()
        >>> skills = discovery.discover_all()
        >>> print(f"Found {len(skills)} skills across all platforms")
    """

    # Platform-specific paths
    PLATFORM_PATHS = {
        "claude_user": Path.home() / ".claude" / "skills",
        "claude_project": Path(".claude") / "skills",
        "codex_user": Path.home() / ".codex" / "skills",
        "codex_project": Path(".codex") / "skills",
    }

    def __init__(self):
        """Initialize cross-platform discovery."""
        self.converter = SkillConverter()
        self._manifests: dict[str, SkillManifest] = {}

    def discover_all(self) -> list[SkillManifest]:
        """Discover skills from all platform paths.

        Returns:
            List of discovered manifests (deduplicated by name).
        """
        from agentic_workflows.openai_skills.loader import SkillLoader

        manifests: list[SkillManifest] = []
        seen_names: set[str] = set()

        for platform, path in self.PLATFORM_PATHS.items():
            if not path.exists():
                continue

            logger.debug(f"Scanning {platform}: {path}")

            # Determine category
            if "user" in platform:
                category = SkillCategory.USER
            else:
                category = SkillCategory.PROJECT

            # Create loader for this path
            loader = SkillLoader()
            loader.add_path(path, category)

            try:
                for manifest in loader.discover_all():
                    if manifest.name not in seen_names:
                        manifests.append(manifest)
                        seen_names.add(manifest.name)
                        self._manifests[manifest.name] = manifest
                    else:
                        logger.debug(
                            f"Skipping duplicate: {manifest.name} from {platform}"
                        )
            except Exception as e:
                logger.warning(f"Error scanning {platform}: {e}")

        logger.info(f"Discovered {len(manifests)} skills across platforms")
        return manifests

    def get_skill(self, name: str) -> SkillManifest | None:
        """Get a skill by name.

        Args:
            name: Skill name.

        Returns:
            Skill manifest or None.
        """
        return self._manifests.get(name)

    def get_platform_skills(self, platform: str) -> list[SkillManifest]:
        """Get skills from a specific platform.

        Args:
            platform: Platform key (claude_user, codex_user, etc.)

        Returns:
            List of manifests from that platform.
        """
        path = self.PLATFORM_PATHS.get(platform)
        if path is None or not path.exists():
            return []

        from agentic_workflows.openai_skills.loader import SkillLoader

        loader = SkillLoader()
        loader.add_path(path, SkillCategory.USER)
        return loader.discover_all()

    def sync_between_platforms(
        self,
        source_platform: str,
        target_platform: str,
        skill_names: list[str] | None = None,
    ) -> list[ConversionResult]:
        """Sync skills from one platform to another.

        Args:
            source_platform: Source platform key.
            target_platform: Target platform key.
            skill_names: Optional list of specific skills to sync.

        Returns:
            List of conversion results.
        """
        source_path = self.PLATFORM_PATHS.get(source_platform)
        target_path = self.PLATFORM_PATHS.get(target_platform)

        if source_path is None or target_path is None:
            return []

        results: list[ConversionResult] = []

        # Get source skills
        source_skills = self.get_platform_skills(source_platform)

        # Filter if specific names requested
        if skill_names:
            source_skills = [s for s in source_skills if s.name in skill_names]

        # Determine target format
        if "codex" in target_platform:
            target_format = SkillFormat.OPENAI_CODEX
        else:
            target_format = SkillFormat.AGENTSKILLS_IO

        # Convert each skill
        for manifest in source_skills:
            if target_format == SkillFormat.OPENAI_CODEX:
                result = self.converter.convert_to_openai_codex(manifest, target_path)
            else:
                result = self.converter.convert_to_skill_md(manifest, target_path)

            results.append(result)

        return results


# Convenience functions

def detect_format(path: Path) -> SkillFormat | None:
    """Detect skill format from path.

    Args:
        path: Path to skill.

    Returns:
        Detected format or None.
    """
    return FormatDetector.detect(path)


def convert_skill(
    source: Path,
    target: Path,
    target_format: SkillFormat,
) -> ConversionResult:
    """Convert a skill to another format.

    Args:
        source: Source skill path.
        target: Target output path.
        target_format: Target format.

    Returns:
        Conversion result.
    """
    converter = SkillConverter()
    return converter.convert_directory(source, target, target_format)


def discover_cross_platform() -> list[SkillManifest]:
    """Discover skills from all platforms.

    Returns:
        List of discovered manifests.
    """
    discovery = CrossPlatformDiscovery()
    return discovery.discover_all()
