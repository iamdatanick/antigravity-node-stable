"""
Skill Installer for OpenAI Skills SDK Integration.

This module provides the SkillInstaller class for installing skills from
GitHub URLs, following the pattern established by the OpenAI skills
repository ($skill-installer).

Supported Sources:
- GitHub repository URLs (e.g., github.com/openai/skills)
- Direct skill directory URLs
- Local file paths
- Skill names from the curated catalog

Installation Targets:
- ~/.claude/skills (Claude Code user skills)
- ~/.codex/skills (OpenAI Codex user skills)
- ./.claude/skills (project-level)
- ./.codex/skills (project-level)

Example:
    >>> from agentic_workflows.openai_skills.installer import SkillInstaller
    >>> installer = SkillInstaller()
    >>> installer.install_from_github("openai/skills", ".curated/address-github-comments")
    >>> installer.install_by_name("pdf-processing")

Author: Agentic Workflows Contributors
Version: 1.0.0
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from agentic_workflows.openai_skills.loader import SkillLoader
from agentic_workflows.openai_skills.skill_types import (
    SkillCategory,
    SkillManifest,
)

logger = logging.getLogger(__name__)


class InstallTarget(Enum):
    """Target location for skill installation."""

    USER_CLAUDE = "user_claude"  # ~/.claude/skills
    USER_CODEX = "user_codex"  # ~/.codex/skills
    PROJECT_CLAUDE = "project_claude"  # ./.claude/skills
    PROJECT_CODEX = "project_codex"  # ./.codex/skills
    CUSTOM = "custom"


class InstallStatus(Enum):
    """Status of an installation operation."""

    SUCCESS = "success"
    ALREADY_EXISTS = "already_exists"
    UPDATED = "updated"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class InstallResult:
    """Result of a skill installation operation.

    Attributes:
        skill_name: Name of the installed skill.
        status: Installation status.
        target_path: Path where skill was installed.
        source: Original source (URL or path).
        message: Status message or error description.
        version: Installed skill version.
        dependencies: List of dependency skill names.
        timestamp: Installation timestamp.
    """

    skill_name: str
    status: InstallStatus
    target_path: Path | None = None
    source: str = ""
    message: str = ""
    version: str = ""
    dependencies: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "skill_name": self.skill_name,
            "status": self.status.value,
            "target_path": str(self.target_path) if self.target_path else None,
            "source": self.source,
            "message": self.message,
            "version": self.version,
            "dependencies": self.dependencies.copy(),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class InstallerConfig:
    """Configuration for the SkillInstaller.

    Attributes:
        default_target: Default installation target.
        user_claude_path: Claude Code user skills directory.
        user_codex_path: OpenAI Codex user skills directory.
        project_claude_path: Claude Code project-relative path.
        project_codex_path: Codex project-relative path.
        catalog_repos: GitHub repositories for skill catalogs.
        auto_resolve_deps: Whether to auto-install dependencies.
        overwrite: Whether to overwrite existing skills.
        verify_signature: Whether to verify skill signatures.
        timeout: Git operation timeout in seconds.
    """

    default_target: InstallTarget = InstallTarget.USER_CLAUDE
    user_claude_path: Path = field(default_factory=lambda: Path.home() / ".claude" / "skills")
    user_codex_path: Path = field(default_factory=lambda: Path.home() / ".codex" / "skills")
    project_claude_path: Path = field(default_factory=lambda: Path.cwd() / ".claude" / "skills")
    project_codex_path: Path = field(default_factory=lambda: Path.cwd() / ".codex" / "skills")
    catalog_repos: list[str] = field(
        default_factory=lambda: [
            "openai/skills",
            "anthropics/skills",
        ]
    )
    auto_resolve_deps: bool = True
    overwrite: bool = False
    verify_signature: bool = False
    timeout: int = 60


@dataclass
class SkillSource:
    """Parsed skill source information.

    Attributes:
        source_type: Type of source (github, local, catalog).
        repo: GitHub repository (owner/repo).
        branch: Git branch name.
        path: Path within repository.
        skill_name: Skill name if from catalog.
        url: Original URL.
        category: Skill category based on path.
    """

    source_type: str
    repo: str = ""
    branch: str = "main"
    path: str = ""
    skill_name: str = ""
    url: str = ""
    category: SkillCategory = SkillCategory.USER

    @classmethod
    def parse(cls, source: str) -> SkillSource:
        """Parse a source string into SkillSource.

        Supports formats:
        - github.com/owner/repo/tree/branch/path
        - owner/repo:.path
        - owner/repo
        - ./local/path
        - skill-name (catalog lookup)

        Args:
            source: Source string to parse.

        Returns:
            Parsed SkillSource.
        """
        # Local path
        if source.startswith(("/", ".", "~")):
            return cls(
                source_type="local",
                path=source,
                url=source,
            )

        # GitHub URL
        github_patterns = [
            # Full URL: github.com/owner/repo/tree/branch/path
            r"(?:https?://)?github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.+)",
            # Full URL without tree: github.com/owner/repo/path
            r"(?:https?://)?github\.com/([^/]+)/([^/]+)/(?!tree)(.+)",
            # Full URL repo only: github.com/owner/repo
            r"(?:https?://)?github\.com/([^/]+)/([^/]+)/?$",
        ]

        for pattern in github_patterns:
            match = re.match(pattern, source)
            if match:
                groups = match.groups()
                if len(groups) == 4:  # With branch
                    owner, repo, branch, path = groups
                elif len(groups) == 3:  # Without branch
                    owner, repo, path = groups
                    branch = "main"
                else:  # Repo only
                    owner, repo = groups[:2]
                    path = ""
                    branch = "main"

                # Determine category from path
                category = SkillCategory.USER
                if ".system" in path:
                    category = SkillCategory.SYSTEM
                elif ".curated" in path:
                    category = SkillCategory.CURATED
                elif ".experimental" in path:
                    category = SkillCategory.EXPERIMENTAL

                return cls(
                    source_type="github",
                    repo=f"{owner}/{repo}",
                    branch=branch,
                    path=path,
                    url=source,
                    category=category,
                )

        # Short format: owner/repo:path
        short_match = re.match(r"([^/]+)/([^:]+):(.+)", source)
        if short_match:
            owner, repo, path = short_match.groups()
            return cls(
                source_type="github",
                repo=f"{owner}/{repo}",
                path=path,
                url=source,
            )

        # Owner/repo only
        repo_match = re.match(r"([^/]+)/([^/]+)$", source)
        if repo_match:
            owner, repo = repo_match.groups()
            return cls(
                source_type="github",
                repo=f"{owner}/{repo}",
                url=source,
            )

        # Assume catalog skill name
        return cls(
            source_type="catalog",
            skill_name=source,
            url=source,
        )


class SkillInstaller:
    """Installer for managing skill installation and updates.

    The SkillInstaller handles downloading skills from GitHub repositories,
    resolving dependencies, and installing to appropriate locations.

    Features:
    - Install from GitHub URLs or skill names
    - Support for curated, experimental, and system skills
    - Automatic dependency resolution
    - Version tracking and updates

    Attributes:
        config: Installer configuration.
        loader: Skill loader for parsing manifests.
        installed: Registry of installed skills.

    Example:
        >>> installer = SkillInstaller()
        >>>
        >>> # Install by name (from catalog)
        >>> result = installer.install_by_name("address-github-comments")
        >>>
        >>> # Install from GitHub URL
        >>> result = installer.install_from_url(
        ...     "https://github.com/openai/skills/tree/main/.curated/pdf-processing"
        ... )
        >>>
        >>> # Check what's installed
        >>> for skill in installer.list_installed():
        ...     print(skill.name)
    """

    def __init__(self, config: InstallerConfig | None = None):
        """Initialize the skill installer.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or InstallerConfig()
        self.loader = SkillLoader()
        self._installed: dict[str, InstallResult] = {}
        self._install_log: list[InstallResult] = []

        # Ensure target directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create installation directories if they don't exist."""
        paths = [
            self.config.user_claude_path,
            self.config.user_codex_path,
        ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

    def get_target_path(self, target: InstallTarget | None = None) -> Path:
        """Get the target directory for installation.

        Args:
            target: Install target. Uses default if not provided.

        Returns:
            Target directory path.
        """
        target = target or self.config.default_target

        paths = {
            InstallTarget.USER_CLAUDE: self.config.user_claude_path,
            InstallTarget.USER_CODEX: self.config.user_codex_path,
            InstallTarget.PROJECT_CLAUDE: self.config.project_claude_path,
            InstallTarget.PROJECT_CODEX: self.config.project_codex_path,
        }

        return paths.get(target, self.config.user_claude_path)

    def install_from_url(
        self,
        url: str,
        target: InstallTarget | None = None,
        skill_name: str | None = None,
    ) -> InstallResult:
        """Install a skill from a URL.

        Args:
            url: GitHub URL or local path.
            target: Installation target location.
            skill_name: Optional override for skill name.

        Returns:
            Installation result.
        """
        source = SkillSource.parse(url)

        if source.source_type == "local":
            return self._install_from_local(source.path, target, skill_name)
        elif source.source_type == "github":
            return self._install_from_github(source, target, skill_name)
        elif source.source_type == "catalog":
            return self.install_by_name(source.skill_name, target)

        return InstallResult(
            skill_name=skill_name or "unknown",
            status=InstallStatus.FAILED,
            source=url,
            message=f"Unknown source type: {source.source_type}",
        )

    def install_by_name(
        self,
        name: str,
        target: InstallTarget | None = None,
    ) -> InstallResult:
        """Install a skill by name from the catalog.

        Searches configured catalog repositories for the skill.

        Args:
            name: Skill name (e.g., "pdf-processing").
            target: Installation target location.

        Returns:
            Installation result.
        """
        logger.info(f"Looking up skill '{name}' in catalogs")

        # Search catalog repos
        for repo in self.config.catalog_repos:
            # Try curated first
            source = SkillSource(
                source_type="github",
                repo=repo,
                path=f".curated/{name}",
                skill_name=name,
                category=SkillCategory.CURATED,
            )

            result = self._install_from_github(source, target, name)
            if result.status in (InstallStatus.SUCCESS, InstallStatus.UPDATED):
                return result

            # Try skills/ directory
            source.path = f"skills/.curated/{name}"
            result = self._install_from_github(source, target, name)
            if result.status in (InstallStatus.SUCCESS, InstallStatus.UPDATED):
                return result

        return InstallResult(
            skill_name=name,
            status=InstallStatus.FAILED,
            message=f"Skill '{name}' not found in any catalog",
        )

    def _install_from_github(
        self,
        source: SkillSource,
        target: InstallTarget | None,
        skill_name: str | None,
    ) -> InstallResult:
        """Install a skill from GitHub.

        Args:
            source: Parsed source information.
            target: Installation target.
            skill_name: Optional skill name override.

        Returns:
            Installation result.
        """
        target_dir = self.get_target_path(target)

        # Create temp directory for cloning
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            try:
                # Sparse checkout for just the skill directory
                if source.path:
                    skill_dir = self._sparse_checkout(
                        source.repo,
                        source.branch,
                        source.path,
                        tmp_path,
                    )
                else:
                    # Clone entire repo
                    skill_dir = self._clone_repo(
                        source.repo,
                        source.branch,
                        tmp_path,
                    )

                if skill_dir is None or not skill_dir.exists():
                    return InstallResult(
                        skill_name=skill_name or source.skill_name or "unknown",
                        status=InstallStatus.FAILED,
                        source=source.url,
                        message="Failed to fetch skill from GitHub",
                    )

                # Find SKILL.md
                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    # Check for case-insensitive match
                    for f in skill_dir.iterdir():
                        if f.name.upper() == "SKILL.MD":
                            skill_md = f
                            break

                if not skill_md.exists():
                    return InstallResult(
                        skill_name=skill_name or source.skill_name or skill_dir.name,
                        status=InstallStatus.FAILED,
                        source=source.url,
                        message="No SKILL.md found in skill directory",
                    )

                # Parse manifest
                manifest = self.loader.load_skill_by_path(skill_md)
                if manifest is None:
                    return InstallResult(
                        skill_name=skill_name or skill_dir.name,
                        status=InstallStatus.FAILED,
                        source=source.url,
                        message="Failed to parse SKILL.md",
                    )

                # Override name if provided
                final_name = skill_name or manifest.name

                # Check for existing installation
                dest_dir = target_dir / final_name
                if dest_dir.exists():
                    if not self.config.overwrite:
                        return InstallResult(
                            skill_name=final_name,
                            status=InstallStatus.ALREADY_EXISTS,
                            target_path=dest_dir,
                            source=source.url,
                            message=f"Skill already installed at {dest_dir}",
                            version=manifest.version,
                        )
                    # Remove existing for update
                    shutil.rmtree(dest_dir)

                # Copy skill directory
                shutil.copytree(skill_dir, dest_dir)

                # Resolve dependencies
                dependencies: list[str] = []
                if self.config.auto_resolve_deps and manifest.metadata.extra.get("requires"):
                    deps = manifest.metadata.extra["requires"]
                    if isinstance(deps, list):
                        for dep in deps:
                            dep_result = self.install_by_name(str(dep), target)
                            if dep_result.status == InstallStatus.SUCCESS:
                                dependencies.append(str(dep))

                result = InstallResult(
                    skill_name=final_name,
                    status=InstallStatus.SUCCESS,
                    target_path=dest_dir,
                    source=source.url,
                    message=f"Successfully installed {final_name}",
                    version=manifest.version,
                    dependencies=dependencies,
                )

                self._installed[final_name] = result
                self._install_log.append(result)

                logger.info(f"Installed skill '{final_name}' to {dest_dir}")
                return result

            except Exception as e:
                logger.error(f"GitHub install failed: {e}")
                return InstallResult(
                    skill_name=skill_name or source.skill_name or "unknown",
                    status=InstallStatus.FAILED,
                    source=source.url,
                    message=str(e),
                )

    def _sparse_checkout(
        self,
        repo: str,
        branch: str,
        path: str,
        dest: Path,
    ) -> Path | None:
        """Perform sparse checkout of a specific directory.

        Args:
            repo: GitHub repository (owner/repo).
            branch: Branch name.
            path: Path within repository.
            dest: Destination directory.

        Returns:
            Path to checked out directory or None on failure.
        """
        repo_url = f"https://github.com/{repo}.git"

        try:
            # Initialize repo
            subprocess.run(
                ["git", "init"],
                cwd=dest,
                check=True,
                capture_output=True,
                timeout=self.config.timeout,
            )

            # Add remote
            subprocess.run(
                ["git", "remote", "add", "origin", repo_url],
                cwd=dest,
                check=True,
                capture_output=True,
                timeout=self.config.timeout,
            )

            # Enable sparse checkout
            subprocess.run(
                ["git", "config", "core.sparseCheckout", "true"],
                cwd=dest,
                check=True,
                capture_output=True,
                timeout=self.config.timeout,
            )

            # Set sparse checkout path
            sparse_file = dest / ".git" / "info" / "sparse-checkout"
            sparse_file.parent.mkdir(parents=True, exist_ok=True)
            sparse_file.write_text(path + "\n")

            # Fetch and checkout
            subprocess.run(
                ["git", "fetch", "--depth=1", "origin", branch],
                cwd=dest,
                check=True,
                capture_output=True,
                timeout=self.config.timeout,
            )

            subprocess.run(
                ["git", "checkout", branch],
                cwd=dest,
                check=True,
                capture_output=True,
                timeout=self.config.timeout,
            )

            return dest / path

        except subprocess.CalledProcessError as e:
            logger.error(f"Git operation failed: {e.stderr.decode() if e.stderr else e}")
            return None
        except subprocess.TimeoutExpired:
            logger.error("Git operation timed out")
            return None

    def _clone_repo(
        self,
        repo: str,
        branch: str,
        dest: Path,
    ) -> Path | None:
        """Clone entire repository.

        Args:
            repo: GitHub repository (owner/repo).
            branch: Branch name.
            dest: Destination directory.

        Returns:
            Path to cloned directory or None on failure.
        """
        repo_url = f"https://github.com/{repo}.git"

        try:
            subprocess.run(
                ["git", "clone", "--depth=1", "-b", branch, repo_url, str(dest)],
                check=True,
                capture_output=True,
                timeout=self.config.timeout,
            )
            return dest

        except subprocess.CalledProcessError as e:
            logger.error(f"Git clone failed: {e.stderr.decode() if e.stderr else e}")
            return None
        except subprocess.TimeoutExpired:
            logger.error("Git clone timed out")
            return None

    def _install_from_local(
        self,
        path: str,
        target: InstallTarget | None,
        skill_name: str | None,
    ) -> InstallResult:
        """Install a skill from a local path.

        Args:
            path: Local directory path.
            target: Installation target.
            skill_name: Optional skill name override.

        Returns:
            Installation result.
        """
        source_path = Path(path).expanduser().resolve()

        if not source_path.exists():
            return InstallResult(
                skill_name=skill_name or source_path.name,
                status=InstallStatus.FAILED,
                source=path,
                message=f"Path does not exist: {source_path}",
            )

        # Find SKILL.md
        if source_path.is_file():
            skill_md = source_path
            source_dir = source_path.parent
        else:
            source_dir = source_path
            skill_md = source_dir / "SKILL.md"
            if not skill_md.exists():
                for f in source_dir.iterdir():
                    if f.name.upper() == "SKILL.MD":
                        skill_md = f
                        break

        if not skill_md.exists():
            return InstallResult(
                skill_name=skill_name or source_dir.name,
                status=InstallStatus.FAILED,
                source=path,
                message="No SKILL.md found",
            )

        # Parse manifest
        manifest = self.loader.load_skill_by_path(skill_md)
        if manifest is None:
            return InstallResult(
                skill_name=skill_name or source_dir.name,
                status=InstallStatus.FAILED,
                source=path,
                message="Failed to parse SKILL.md",
            )

        final_name = skill_name or manifest.name
        target_dir = self.get_target_path(target)
        dest_dir = target_dir / final_name

        # Check existing
        if dest_dir.exists():
            if not self.config.overwrite:
                return InstallResult(
                    skill_name=final_name,
                    status=InstallStatus.ALREADY_EXISTS,
                    target_path=dest_dir,
                    source=path,
                    message=f"Skill already installed at {dest_dir}",
                )
            shutil.rmtree(dest_dir)

        # Copy
        shutil.copytree(source_dir, dest_dir)

        result = InstallResult(
            skill_name=final_name,
            status=InstallStatus.SUCCESS,
            target_path=dest_dir,
            source=path,
            message=f"Successfully installed {final_name}",
            version=manifest.version,
        )

        self._installed[final_name] = result
        self._install_log.append(result)

        logger.info(f"Installed skill '{final_name}' from local path")
        return result

    def uninstall(
        self,
        name: str,
        target: InstallTarget | None = None,
    ) -> InstallResult:
        """Uninstall a skill.

        Args:
            name: Skill name to uninstall.
            target: Target location to uninstall from.

        Returns:
            Uninstall result.
        """
        # Search all targets if not specified
        if target is None:
            targets = [
                InstallTarget.USER_CLAUDE,
                InstallTarget.USER_CODEX,
                InstallTarget.PROJECT_CLAUDE,
                InstallTarget.PROJECT_CODEX,
            ]
        else:
            targets = [target]

        for t in targets:
            target_dir = self.get_target_path(t)
            skill_dir = target_dir / name

            if skill_dir.exists():
                try:
                    shutil.rmtree(skill_dir)

                    if name in self._installed:
                        del self._installed[name]

                    return InstallResult(
                        skill_name=name,
                        status=InstallStatus.SUCCESS,
                        target_path=skill_dir,
                        message=f"Successfully uninstalled {name}",
                    )
                except Exception as e:
                    return InstallResult(
                        skill_name=name,
                        status=InstallStatus.FAILED,
                        target_path=skill_dir,
                        message=f"Failed to uninstall: {e}",
                    )

        return InstallResult(
            skill_name=name,
            status=InstallStatus.FAILED,
            message=f"Skill '{name}' not found in any target location",
        )

    def update(
        self,
        name: str,
        target: InstallTarget | None = None,
    ) -> InstallResult:
        """Update an installed skill.

        Args:
            name: Skill name to update.
            target: Target location.

        Returns:
            Update result.
        """
        # Get current installation info
        current = self._installed.get(name)
        if current is None:
            # Try to find in installed locations
            target_dir = self.get_target_path(target)
            skill_dir = target_dir / name
            if not skill_dir.exists():
                return InstallResult(
                    skill_name=name,
                    status=InstallStatus.FAILED,
                    message=f"Skill '{name}' not installed",
                )

        # Temporarily enable overwrite
        original_overwrite = self.config.overwrite
        self.config.overwrite = True

        try:
            # Reinstall from original source or catalog
            if current and current.source:
                result = self.install_from_url(current.source, target, name)
            else:
                result = self.install_by_name(name, target)

            if result.status == InstallStatus.SUCCESS:
                result.status = InstallStatus.UPDATED
                result.message = f"Successfully updated {name}"

            return result

        finally:
            self.config.overwrite = original_overwrite

    def list_installed(self, target: InstallTarget | None = None) -> list[SkillManifest]:
        """List all installed skills.

        Args:
            target: Target to list from. Lists all if not specified.

        Returns:
            List of installed skill manifests.
        """
        manifests: list[SkillManifest] = []

        if target is None:
            targets = [
                InstallTarget.USER_CLAUDE,
                InstallTarget.USER_CODEX,
            ]
        else:
            targets = [target]

        for t in targets:
            target_dir = self.get_target_path(t)
            if not target_dir.exists():
                continue

            for skill_dir in target_dir.iterdir():
                if not skill_dir.is_dir():
                    continue

                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    for f in skill_dir.iterdir():
                        if f.name.upper() == "SKILL.MD":
                            skill_md = f
                            break

                if skill_md.exists():
                    manifest = self.loader.load_skill_by_path(skill_md)
                    if manifest:
                        manifests.append(manifest)

        return manifests

    def is_installed(self, name: str) -> bool:
        """Check if a skill is installed.

        Args:
            name: Skill name.

        Returns:
            True if installed.
        """
        for target in [InstallTarget.USER_CLAUDE, InstallTarget.USER_CODEX]:
            target_dir = self.get_target_path(target)
            if (target_dir / name).exists():
                return True
        return False

    def get_install_log(self) -> list[InstallResult]:
        """Get the installation log.

        Returns:
            List of all installation results.
        """
        return self._install_log.copy()

    def get_stats(self) -> dict[str, Any]:
        """Get installer statistics.

        Returns:
            Dictionary with statistics.
        """
        installed = self.list_installed()
        by_category: dict[str, int] = {}

        for manifest in installed:
            cat = manifest.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

        return {
            "total_installed": len(installed),
            "install_operations": len(self._install_log),
            "by_category": by_category,
            "installed_names": [m.name for m in installed],
            "catalog_repos": self.config.catalog_repos,
        }

    def __repr__(self) -> str:
        return f"SkillInstaller(installed={len(self._installed)})"


# Convenience functions


def install_skill(
    source: str,
    target: InstallTarget | None = None,
) -> InstallResult:
    """Install a skill from URL or name.

    Args:
        source: GitHub URL, local path, or skill name.
        target: Installation target.

    Returns:
        Installation result.
    """
    installer = SkillInstaller()
    return installer.install_from_url(source, target)


def uninstall_skill(name: str) -> InstallResult:
    """Uninstall a skill by name.

    Args:
        name: Skill name.

    Returns:
        Uninstall result.
    """
    installer = SkillInstaller()
    return installer.uninstall(name)


def list_installed_skills() -> list[SkillManifest]:
    """List all installed skills.

    Returns:
        List of installed skill manifests.
    """
    installer = SkillInstaller()
    return installer.list_installed()
