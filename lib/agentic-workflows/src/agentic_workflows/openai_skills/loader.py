"""
Skill Loader for OpenAI Skills SDK Integration.

This module provides the SkillLoader class that discovers and parses skills
from both Anthropic SKILL.md format (Claude Code) and OpenAI Codex skill
format (instructions + scripts folders).

Supported Formats:
- SKILL.md files with YAML frontmatter (agentskills.io standard)
- OpenAI Codex skills (instructions/, scripts/ folders)
- Claude Code skills (~/.claude/skills)

Discovery Paths:
- ~/.claude/skills (Anthropic Claude Code)
- ~/.codex/skills (OpenAI Codex)
- ./.claude/skills (project-level Claude)
- ./.codex/skills (project-level Codex)

Example:
    >>> from agentic_workflows.openai_skills.loader import SkillLoader
    >>> loader = SkillLoader()
    >>> skills = loader.discover_all()
    >>> print(f"Found {len(skills)} skills")

Author: Agentic Workflows Contributors
Version: 1.0.0
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

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


# Pattern for YAML frontmatter extraction
FRONTMATTER_PATTERN = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n?",
    re.DOTALL | re.MULTILINE
)

# Pattern for markdown headings
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class SkillPath:
    """Configuration for a skill discovery path.

    Attributes:
        path: Directory path to search.
        category: Category to assign to discovered skills.
        recursive: Whether to search subdirectories.
        enabled: Whether this path is active.
    """
    path: Path
    category: SkillCategory = SkillCategory.USER
    recursive: bool = True
    enabled: bool = True

    def exists(self) -> bool:
        """Check if the path exists and is a directory."""
        return self.path.exists() and self.path.is_dir()


@dataclass
class LoaderConfig:
    """Configuration for the SkillLoader.

    Attributes:
        claude_user_path: Claude Code user skills directory.
        codex_user_path: OpenAI Codex user skills directory.
        claude_project_path: Claude Code project-relative path.
        codex_project_path: Codex project-relative path.
        additional_paths: Extra paths to search.
        load_resources: Whether to load resource file contents.
        max_skill_size: Maximum SKILL.md file size in bytes.
        encoding: Text file encoding.
    """
    claude_user_path: Path = field(
        default_factory=lambda: Path.home() / ".claude" / "skills"
    )
    codex_user_path: Path = field(
        default_factory=lambda: Path.home() / ".codex" / "skills"
    )
    claude_project_path: Path = field(
        default_factory=lambda: Path(".claude") / "skills"
    )
    codex_project_path: Path = field(
        default_factory=lambda: Path(".codex") / "skills"
    )
    additional_paths: list[Path] = field(default_factory=list)
    load_resources: bool = True
    max_skill_size: int = 1024 * 1024  # 1MB
    encoding: str = "utf-8"


class SkillLoader:
    """Loader for discovering and parsing skill files.

    The SkillLoader handles discovery from multiple paths and supports
    both Anthropic SKILL.md and OpenAI Codex skill formats.

    Progressive Disclosure:
    - Metadata (~100 tokens) loaded at discovery for all skills
    - Full content (<5000 tokens) loaded when skill is activated
    - Resources loaded on-demand during execution

    Attributes:
        config: Loader configuration.
        skill_paths: Configured discovery paths.
        cached_manifests: Cached skill manifests by path.

    Example:
        >>> loader = SkillLoader()
        >>> loader.add_path(Path("./custom/skills"), SkillCategory.PROJECT)
        >>>
        >>> # Discover all skills (metadata only)
        >>> skills = loader.discover_all()
        >>>
        >>> # Load full skill content
        >>> skill = loader.load_skill("pdf-processing")
    """

    def __init__(self, config: LoaderConfig | None = None):
        """Initialize the skill loader.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or LoaderConfig()
        self.skill_paths: list[SkillPath] = []
        self._cached_manifests: dict[str, SkillManifest] = {}
        self._path_to_name: dict[Path, str] = {}

        # Initialize default paths
        self._setup_default_paths()

    def _setup_default_paths(self) -> None:
        """Configure default skill discovery paths."""
        # Claude Code user skills
        self.skill_paths.append(SkillPath(
            path=self.config.claude_user_path,
            category=SkillCategory.USER,
        ))

        # OpenAI Codex user skills
        self.skill_paths.append(SkillPath(
            path=self.config.codex_user_path,
            category=SkillCategory.USER,
        ))

        # Project-level Claude skills
        cwd = Path.cwd()
        self.skill_paths.append(SkillPath(
            path=cwd / self.config.claude_project_path,
            category=SkillCategory.PROJECT,
        ))

        # Project-level Codex skills
        self.skill_paths.append(SkillPath(
            path=cwd / self.config.codex_project_path,
            category=SkillCategory.PROJECT,
        ))

        # Additional configured paths
        for path in self.config.additional_paths:
            self.skill_paths.append(SkillPath(
                path=path,
                category=SkillCategory.USER,
            ))

    def add_path(
        self,
        path: Path,
        category: SkillCategory = SkillCategory.USER,
        recursive: bool = True,
    ) -> None:
        """Add a discovery path.

        Args:
            path: Directory to search.
            category: Category for discovered skills.
            recursive: Whether to search subdirectories.
        """
        resolved = Path(path).expanduser().resolve()
        self.skill_paths.append(SkillPath(
            path=resolved,
            category=category,
            recursive=recursive,
        ))
        logger.debug(f"Added skill path: {resolved}")

    def remove_path(self, path: Path) -> bool:
        """Remove a discovery path.

        Args:
            path: Path to remove.

        Returns:
            True if removed, False if not found.
        """
        resolved = Path(path).expanduser().resolve()
        for sp in self.skill_paths:
            if sp.path == resolved:
                self.skill_paths.remove(sp)
                return True
        return False

    def discover_all(self, force_reload: bool = False) -> list[SkillManifest]:
        """Discover skills from all configured paths.

        This performs a lightweight discovery that loads only metadata
        (name, description) for each skill, following progressive
        disclosure principles.

        Args:
            force_reload: If True, ignore cached manifests.

        Returns:
            List of discovered skill manifests.
        """
        if not force_reload and self._cached_manifests:
            return list(self._cached_manifests.values())

        manifests: list[SkillManifest] = []

        for skill_path in self.skill_paths:
            if not skill_path.enabled or not skill_path.exists():
                continue

            logger.debug(f"Scanning: {skill_path.path}")

            try:
                for manifest in self._scan_directory(skill_path):
                    if manifest.name in self._cached_manifests:
                        logger.warning(
                            f"Duplicate skill '{manifest.name}' found, "
                            f"keeping first occurrence"
                        )
                        continue

                    self._cached_manifests[manifest.name] = manifest
                    if manifest.source_path:
                        self._path_to_name[manifest.source_path] = manifest.name
                    manifests.append(manifest)

            except Exception as e:
                logger.error(f"Error scanning {skill_path.path}: {e}")

        logger.info(f"Discovered {len(manifests)} skills from {len(self.skill_paths)} paths")
        return manifests

    def _scan_directory(
        self,
        skill_path: SkillPath
    ) -> Iterator[SkillManifest]:
        """Scan a directory for skill definitions.

        Args:
            skill_path: Path configuration to scan.

        Yields:
            Discovered skill manifests.
        """
        base = skill_path.path

        # Look for SKILL.md files
        pattern = "**/*.md" if skill_path.recursive else "*.md"

        for md_file in base.glob(pattern):
            # Check for SKILL.md (case-insensitive)
            if md_file.name.upper() == "SKILL.MD":
                try:
                    manifest = self._parse_skill_md(
                        md_file,
                        skill_path.category
                    )
                    yield manifest
                except Exception as e:
                    logger.error(f"Failed to parse {md_file}: {e}")

            # Also check for *.skill.md pattern
            elif md_file.name.lower().endswith(".skill.md"):
                try:
                    manifest = self._parse_skill_md(
                        md_file,
                        skill_path.category
                    )
                    yield manifest
                except Exception as e:
                    logger.error(f"Failed to parse {md_file}: {e}")

        # Check for OpenAI Codex format (instructions/ folder)
        for dir_path in base.iterdir():
            if not dir_path.is_dir():
                continue

            instructions_dir = dir_path / "instructions"
            if instructions_dir.exists() and instructions_dir.is_dir():
                try:
                    manifest = self._parse_codex_skill(
                        dir_path,
                        skill_path.category
                    )
                    if manifest:
                        yield manifest
                except Exception as e:
                    logger.error(f"Failed to parse Codex skill {dir_path}: {e}")

    def _parse_skill_md(
        self,
        path: Path,
        category: SkillCategory
    ) -> SkillManifest:
        """Parse a SKILL.md file into a manifest.

        Args:
            path: Path to SKILL.md file.
            category: Category to assign.

        Returns:
            Parsed skill manifest.

        Raises:
            ValueError: If file format is invalid.
        """
        # Check file size
        if path.stat().st_size > self.config.max_skill_size:
            raise ValueError(f"Skill file too large: {path}")

        content = path.read_text(encoding=self.config.encoding)

        # Extract frontmatter
        match = FRONTMATTER_PATTERN.match(content)
        if not match:
            raise ValueError(f"No valid YAML frontmatter in: {path}")

        frontmatter_text = match.group(1)
        body_content = content[match.end():].strip()

        # Parse YAML
        try:
            frontmatter = yaml.safe_load(frontmatter_text)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}")

        if not isinstance(frontmatter, dict):
            raise ValueError(f"Frontmatter must be a mapping in: {path}")

        # Extract required fields
        name = frontmatter.get("name", "")
        if not name:
            # Derive from directory or file name
            if path.name.upper() == "SKILL.MD":
                name = path.parent.name
            else:
                name = path.stem.replace(".skill", "")

        description = frontmatter.get("description", "")
        if not description:
            # Try to extract from first paragraph of body
            first_para = body_content.split("\n\n")[0] if body_content else ""
            description = first_para[:500] if first_para else f"Skill: {name}"

        # Parse allowed-tools
        tools: list[SkillTool] = []
        allowed_tools = frontmatter.get("allowed-tools", "")
        if isinstance(allowed_tools, str) and allowed_tools:
            for spec in allowed_tools.split():
                tools.append(SkillTool.parse(spec))
        elif isinstance(allowed_tools, list):
            for spec in allowed_tools:
                if isinstance(spec, str):
                    tools.append(SkillTool.parse(spec))

        # Parse metadata
        metadata_dict = frontmatter.get("metadata", {})
        if not isinstance(metadata_dict, dict):
            metadata_dict = {}

        # Handle top-level metadata fields
        if "author" in frontmatter:
            metadata_dict.setdefault("author", frontmatter["author"])
        if "license" in frontmatter:
            metadata_dict.setdefault("license", frontmatter["license"])
        if "version" in frontmatter:
            metadata_dict.setdefault("version", frontmatter["version"])

        metadata = SkillMetadata.from_dict(metadata_dict)

        # Build manifest
        manifest = SkillManifest(
            name=name,
            description=description,
            version=metadata.version,
            tools=tools,
            metadata=metadata,
            compatibility=frontmatter.get("compatibility", ""),
            category=category,
            source_path=path,
            raw_content=body_content,
            loaded=False,  # Full content not yet processed
        )

        # Discover resources
        if self.config.load_resources:
            manifest.resources = self._discover_resources(path.parent)

        return manifest

    def _parse_codex_skill(
        self,
        path: Path,
        category: SkillCategory
    ) -> SkillManifest | None:
        """Parse an OpenAI Codex format skill.

        Codex skills have an instructions/ folder with markdown files
        and optionally a scripts/ folder.

        Args:
            path: Skill directory path.
            category: Category to assign.

        Returns:
            Parsed manifest or None if not a valid skill.
        """
        instructions_dir = path / "instructions"
        if not instructions_dir.exists():
            return None

        # Collect instruction content
        instructions: list[SkillInstruction] = []
        combined_content: list[str] = []

        for idx, md_file in enumerate(sorted(instructions_dir.glob("*.md"))):
            try:
                content = md_file.read_text(encoding=self.config.encoding)
                combined_content.append(f"# {md_file.stem}\n\n{content}")

                instructions.append(SkillInstruction(
                    title=md_file.stem,
                    content=content,
                    order=idx,
                ))
            except Exception as e:
                logger.warning(f"Failed to read {md_file}: {e}")

        if not instructions:
            return None

        # Extract metadata from first instruction or README
        readme_path = path / "README.md"
        name = path.name
        description = f"Skill: {name}"

        if readme_path.exists():
            readme_content = readme_path.read_text(encoding=self.config.encoding)
            # Try to extract description from first paragraph
            lines = readme_content.split("\n")
            for line in lines:
                if line.strip() and not line.startswith("#"):
                    description = line.strip()[:500]
                    break

        # Build manifest
        manifest = SkillManifest(
            name=name,
            description=description,
            instructions=instructions,
            category=category,
            source_path=path,
            raw_content="\n\n".join(combined_content),
            loaded=True,
        )

        # Discover resources
        if self.config.load_resources:
            manifest.resources = self._discover_resources(path)

        return manifest

    def _discover_resources(self, skill_dir: Path) -> list[SkillResource]:
        """Discover resources in a skill directory.

        Args:
            skill_dir: Skill root directory.

        Returns:
            List of discovered resources.
        """
        resources: list[SkillResource] = []

        # Map directories to resource types
        resource_dirs = {
            "scripts": ResourceType.SCRIPT,
            "references": ResourceType.REFERENCE,
            "assets": ResourceType.ASSET,
        }

        for dir_name, resource_type in resource_dirs.items():
            resource_dir = skill_dir / dir_name
            if not resource_dir.exists():
                continue

            for file_path in resource_dir.rglob("*"):
                if not file_path.is_file():
                    continue

                # Skip hidden files
                if file_path.name.startswith("."):
                    continue

                resource = SkillResource.from_path(skill_dir, file_path)
                resource.resource_type = resource_type
                resource.executable = resource_type == ResourceType.SCRIPT
                resources.append(resource)

        return resources

    def load_skill(
        self,
        name: str,
        include_resources: bool = True
    ) -> SkillManifest | None:
        """Load full skill content by name.

        This method loads the complete skill including parsed instructions
        and optionally resource contents.

        Args:
            name: Skill name to load.
            include_resources: Whether to load resource file contents.

        Returns:
            Fully loaded manifest or None if not found.
        """
        # Check cache
        manifest = self._cached_manifests.get(name)
        if manifest is None:
            logger.warning(f"Skill not found: {name}")
            return None

        if manifest.loaded:
            return manifest

        # Parse instructions from body content
        if manifest.raw_content and not manifest.instructions:
            manifest.instructions = self._parse_instructions(manifest.raw_content)

        # Load resource contents
        if include_resources and manifest.source_path:
            skill_dir = manifest.source_path.parent
            for resource in manifest.resources:
                if not resource.content:
                    resource_path = skill_dir / resource.path
                    if resource_path.exists():
                        try:
                            resource.content = resource_path.read_text(
                                encoding=resource.encoding
                            )
                        except Exception as e:
                            logger.warning(f"Failed to load resource {resource.path}: {e}")

        manifest.loaded = True
        return manifest

    def load_skill_by_path(self, path: Path) -> SkillManifest | None:
        """Load a skill directly from a file path.

        Args:
            path: Path to SKILL.md file.

        Returns:
            Loaded manifest or None if invalid.
        """
        resolved = Path(path).expanduser().resolve()

        # Check if already cached
        if resolved in self._path_to_name:
            return self.load_skill(self._path_to_name[resolved])

        # Parse and cache
        try:
            manifest = self._parse_skill_md(resolved, SkillCategory.USER)
            self._cached_manifests[manifest.name] = manifest
            self._path_to_name[resolved] = manifest.name
            return self.load_skill(manifest.name)
        except Exception as e:
            logger.error(f"Failed to load skill from {path}: {e}")
            return None

    def _parse_instructions(self, content: str) -> list[SkillInstruction]:
        """Parse markdown content into instruction sections.

        Args:
            content: Markdown body content.

        Returns:
            List of parsed instructions.
        """
        instructions: list[SkillInstruction] = []

        # Split by headings
        parts = HEADING_PATTERN.split(content)

        # parts will be: [pre-heading content, heading_level, heading_text, ...]
        if len(parts) < 3:
            # No headings, treat whole content as single instruction
            if content.strip():
                instructions.append(SkillInstruction(
                    title="Instructions",
                    content=content.strip(),
                    order=0,
                ))
            return instructions

        # Process heading pairs
        current_order = 0
        for i in range(1, len(parts), 3):
            if i + 1 >= len(parts):
                break

            heading_text = parts[i + 1]
            # Get content until next heading
            next_content_idx = i + 2
            if next_content_idx < len(parts):
                section_content = parts[next_content_idx].strip()
            else:
                section_content = ""

            instructions.append(SkillInstruction(
                title=heading_text.strip(),
                content=section_content,
                order=current_order,
            ))
            current_order += 1

        return instructions

    def get_skill_context(self, name: str) -> str:
        """Get skill context string for agent prompt injection.

        Args:
            name: Skill name.

        Returns:
            Formatted context string.
        """
        manifest = self.load_skill(name)
        if manifest is None:
            return ""
        return manifest.get_context()

    def find_matching_skills(self, query: str) -> list[SkillManifest]:
        """Find skills matching a search query.

        Uses trigger keywords and description matching.

        Args:
            query: Search query text.

        Returns:
            List of matching manifests, sorted by relevance.
        """
        if not self._cached_manifests:
            self.discover_all()

        matches: list[tuple[int, SkillManifest]] = []
        query_lower = query.lower()

        for manifest in self._cached_manifests.values():
            score = 0

            # Check name
            if query_lower in manifest.name:
                score += 100

            # Check description
            if query_lower in manifest.description.lower():
                score += 50

            # Check trigger keywords
            if manifest.triggers.matches(query):
                score += manifest.triggers.priority

            if score > 0:
                matches.append((score, manifest))

        # Sort by score descending
        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches]

    def get_all_manifests(self) -> dict[str, SkillManifest]:
        """Get all cached manifests.

        Returns:
            Dictionary mapping names to manifests.
        """
        if not self._cached_manifests:
            self.discover_all()
        return self._cached_manifests.copy()

    def clear_cache(self) -> None:
        """Clear the manifest cache."""
        self._cached_manifests.clear()
        self._path_to_name.clear()

    def reload_skill(self, name: str) -> SkillManifest | None:
        """Reload a skill from disk.

        Args:
            name: Skill name to reload.

        Returns:
            Reloaded manifest or None if not found.
        """
        manifest = self._cached_manifests.get(name)
        if manifest is None or manifest.source_path is None:
            return None

        # Remove from cache
        del self._cached_manifests[name]
        if manifest.source_path in self._path_to_name:
            del self._path_to_name[manifest.source_path]

        # Reload
        return self.load_skill_by_path(manifest.source_path)

    def get_stats(self) -> dict[str, Any]:
        """Get loader statistics.

        Returns:
            Dictionary with statistics.
        """
        by_category: dict[str, int] = {}
        loaded_count = 0

        for manifest in self._cached_manifests.values():
            cat = manifest.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
            if manifest.loaded:
                loaded_count += 1

        return {
            "total_skills": len(self._cached_manifests),
            "loaded_skills": loaded_count,
            "pending_load": len(self._cached_manifests) - loaded_count,
            "skill_paths": len(self.skill_paths),
            "active_paths": sum(1 for p in self.skill_paths if p.enabled and p.exists()),
            "by_category": by_category,
            "skill_names": list(self._cached_manifests.keys()),
        }

    def __len__(self) -> int:
        """Get number of discovered skills."""
        return len(self._cached_manifests)

    def __contains__(self, name: str) -> bool:
        """Check if skill exists by name."""
        return name in self._cached_manifests

    def __iter__(self) -> Iterator[SkillManifest]:
        """Iterate over discovered manifests."""
        return iter(self._cached_manifests.values())

    def __repr__(self) -> str:
        return f"SkillLoader(skills={len(self)}, paths={len(self.skill_paths)})"


# Convenience functions

def create_default_loader() -> SkillLoader:
    """Create a skill loader with default configuration.

    Returns:
        Configured SkillLoader instance.
    """
    return SkillLoader()


def discover_skills(paths: list[Path] | None = None) -> list[SkillManifest]:
    """Discover skills from paths.

    Args:
        paths: Optional paths to search. Uses defaults if not provided.

    Returns:
        List of discovered manifests.
    """
    loader = SkillLoader()

    if paths:
        for path in paths:
            loader.add_path(path)

    return loader.discover_all()


def load_skill_md(path: Path) -> SkillManifest:
    """Load a single SKILL.md file.

    Args:
        path: Path to SKILL.md file.

    Returns:
        Parsed skill manifest.

    Raises:
        ValueError: If file format is invalid.
    """
    loader = SkillLoader()
    manifest = loader.load_skill_by_path(path)
    if manifest is None:
        raise ValueError(f"Failed to load skill from {path}")
    return manifest
