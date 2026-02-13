"""
Skill Registry for OpenAI Skills SDK Integration.

This module provides the OpenAISkillRegistry class that indexes all available
skills and provides search, context generation, and integration with the
existing agentic_workflows.skills module.

Features:
- Index skills from multiple discovery paths
- Search by name, trigger keywords, domain, and tags
- Generate context strings for agent prompts
- Integrate with existing SkillRegistry from agentic_workflows.skills
- Support progressive disclosure for efficient context management

Example:
    >>> from agentic_workflows.openai_skills.registry import OpenAISkillRegistry
    >>> registry = OpenAISkillRegistry()
    >>> registry.index_all()
    >>>
    >>> # Search for skills
    >>> results = registry.search("pdf extraction")
    >>> for skill in results:
    ...     print(skill.name, skill.description)
    >>>
    >>> # Get context for agent prompt
    >>> context = registry.get_skill_context("pdf-processing")

Author: Agentic Workflows Contributors
Version: 1.0.0
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator

from agentic_workflows.openai_skills.skill_types import (
    SkillCategory,
    SkillManifest,
    SkillTrigger,
)
from agentic_workflows.openai_skills.loader import SkillLoader, LoaderConfig


logger = logging.getLogger(__name__)


class SearchMode(Enum):
    """Search mode for skill discovery."""
    EXACT = "exact"          # Exact name match
    PREFIX = "prefix"        # Name starts with query
    CONTAINS = "contains"    # Name or description contains query
    FUZZY = "fuzzy"          # Fuzzy matching with scoring
    TRIGGER = "trigger"      # Match trigger keywords


@dataclass
class SearchResult:
    """Result of a skill search operation.

    Attributes:
        manifest: Matched skill manifest.
        score: Relevance score (higher is better).
        matched_fields: Fields that matched the query.
    """
    manifest: SkillManifest
    score: float = 0.0
    matched_fields: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Get skill name."""
        return self.manifest.name

    @property
    def description(self) -> str:
        """Get skill description."""
        return self.manifest.description


@dataclass
class RegistryConfig:
    """Configuration for the skill registry.

    Attributes:
        auto_index: Whether to auto-index on initialization.
        enable_caching: Whether to cache search results.
        cache_ttl: Cache time-to-live in seconds.
        max_search_results: Maximum search results to return.
        default_search_mode: Default search mode.
        integrate_agentic: Whether to integrate with agentic_workflows.skills.
    """
    auto_index: bool = True
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    max_search_results: int = 20
    default_search_mode: SearchMode = SearchMode.FUZZY
    integrate_agentic: bool = True


class OpenAISkillRegistry:
    """Registry for indexing and searching skills.

    The OpenAISkillRegistry provides centralized skill management with
    search capabilities, context generation, and integration with the
    existing agentic_workflows.skills module.

    Progressive Disclosure:
    - Registry indexes metadata (~100 tokens per skill) for all skills
    - Full skill content loaded on-demand when accessed
    - Context strings generated for agent prompt injection

    Attributes:
        config: Registry configuration.
        loader: Skill loader for discovery.
        skills: Dictionary mapping names to manifests.
        by_domain: Skills indexed by domain.
        by_category: Skills indexed by category.

    Example:
        >>> registry = OpenAISkillRegistry()
        >>>
        >>> # Search by keywords
        >>> results = registry.search("pdf tables extraction")
        >>>
        >>> # Get skills for a domain
        >>> cloudflare_skills = registry.get_by_domain("cloudflare")
        >>>
        >>> # Generate agent context
        >>> context = registry.get_skill_context("cloudflare-d1")
    """

    def __init__(
        self,
        config: RegistryConfig | None = None,
        loader_config: LoaderConfig | None = None,
    ):
        """Initialize the skill registry.

        Args:
            config: Registry configuration.
            loader_config: Loader configuration.
        """
        self.config = config or RegistryConfig()
        self.loader = SkillLoader(loader_config)

        # Index structures
        self._skills: dict[str, SkillManifest] = {}
        self._by_domain: dict[str, list[str]] = {}
        self._by_category: dict[SkillCategory, list[str]] = {}
        self._by_tag: dict[str, list[str]] = {}
        self._triggers: dict[str, SkillTrigger] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Search cache
        self._search_cache: dict[str, tuple[datetime, list[SearchResult]]] = {}

        # Integration with existing registry
        self._agentic_registry: Any = None

        if self.config.auto_index:
            self.index_all()

        if self.config.integrate_agentic:
            self._setup_agentic_integration()

    def _setup_agentic_integration(self) -> None:
        """Set up integration with agentic_workflows.skills.SkillRegistry."""
        try:
            from agentic_workflows.skills import get_registry

            self._agentic_registry = get_registry()
            logger.info("Integrated with agentic_workflows.skills.SkillRegistry")
        except ImportError:
            logger.debug("agentic_workflows.skills not available for integration")
        except Exception as e:
            logger.warning(f"Failed to integrate with agentic_workflows.skills: {e}")

    def index_all(self, force: bool = False) -> int:
        """Index all skills from discovery paths.

        Args:
            force: If True, force reindex even if already indexed.

        Returns:
            Number of skills indexed.
        """
        if self._skills and not force:
            return len(self._skills)

        with self._lock:
            # Clear existing indexes
            self._skills.clear()
            self._by_domain.clear()
            self._by_category.clear()
            self._by_tag.clear()
            self._triggers.clear()
            self._search_cache.clear()

            # Discover all skills
            manifests = self.loader.discover_all(force_reload=force)

            for manifest in manifests:
                self._index_skill(manifest)

            logger.info(f"Indexed {len(self._skills)} skills")
            return len(self._skills)

    def _index_skill(self, manifest: SkillManifest) -> None:
        """Index a single skill manifest.

        Args:
            manifest: Skill manifest to index.
        """
        name = manifest.name

        # Main index
        self._skills[name] = manifest

        # Domain index
        domain = manifest.metadata.extra.get("domain", "general")
        if domain not in self._by_domain:
            self._by_domain[domain] = []
        if name not in self._by_domain[domain]:
            self._by_domain[domain].append(name)

        # Category index
        category = manifest.category
        if category not in self._by_category:
            self._by_category[category] = []
        if name not in self._by_category[category]:
            self._by_category[category].append(name)

        # Tag index
        for tag in manifest.metadata.keywords:
            if tag not in self._by_tag:
                self._by_tag[tag] = []
            if name not in self._by_tag[tag]:
                self._by_tag[tag].append(name)

        # Trigger index
        self._triggers[name] = manifest.triggers

    def add_skill(self, manifest: SkillManifest) -> None:
        """Add a skill to the registry.

        Args:
            manifest: Skill manifest to add.
        """
        with self._lock:
            self._index_skill(manifest)
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

            manifest = self._skills[name]

            # Remove from main index
            del self._skills[name]

            # Remove from domain index
            domain = manifest.metadata.extra.get("domain", "general")
            if domain in self._by_domain and name in self._by_domain[domain]:
                self._by_domain[domain].remove(name)

            # Remove from category index
            category = manifest.category
            if category in self._by_category and name in self._by_category[category]:
                self._by_category[category].remove(name)

            # Remove from tag index
            for tag in manifest.metadata.keywords:
                if tag in self._by_tag and name in self._by_tag[tag]:
                    self._by_tag[tag].remove(name)

            # Remove triggers
            if name in self._triggers:
                del self._triggers[name]

            self._invalidate_cache()
            return True

    def get_skill(self, name: str) -> SkillManifest | None:
        """Get a skill by name.

        Args:
            name: Skill name.

        Returns:
            Skill manifest or None if not found.
        """
        with self._lock:
            return self._skills.get(name)

    def get_by_domain(self, domain: str) -> list[SkillManifest]:
        """Get all skills in a domain.

        Args:
            domain: Domain name.

        Returns:
            List of skill manifests.
        """
        with self._lock:
            names = self._by_domain.get(domain, [])
            return [self._skills[n] for n in names if n in self._skills]

    def get_by_category(self, category: SkillCategory) -> list[SkillManifest]:
        """Get all skills in a category.

        Args:
            category: Skill category.

        Returns:
            List of skill manifests.
        """
        with self._lock:
            names = self._by_category.get(category, [])
            return [self._skills[n] for n in names if n in self._skills]

    def get_by_tag(self, tag: str) -> list[SkillManifest]:
        """Get all skills with a tag.

        Args:
            tag: Tag keyword.

        Returns:
            List of skill manifests.
        """
        with self._lock:
            names = self._by_tag.get(tag, [])
            return [self._skills[n] for n in names if n in self._skills]

    def search(
        self,
        query: str,
        mode: SearchMode | None = None,
        domains: list[str] | None = None,
        categories: list[SkillCategory] | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Search for skills matching a query.

        Args:
            query: Search query text.
            mode: Search mode. Uses default if not specified.
            domains: Optional domain filter.
            categories: Optional category filter.
            limit: Maximum results. Uses config default if not specified.

        Returns:
            List of search results sorted by relevance.
        """
        mode = mode or self.config.default_search_mode
        limit = limit or self.config.max_search_results

        # Check cache
        cache_key = f"{query}:{mode.value}:{domains}:{categories}"
        if self.config.enable_caching:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached[:limit]

        results: list[SearchResult] = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        with self._lock:
            for name, manifest in self._skills.items():
                # Apply filters
                if domains:
                    skill_domain = manifest.metadata.extra.get("domain", "general")
                    if skill_domain not in domains:
                        continue

                if categories and manifest.category not in categories:
                    continue

                # Calculate match score
                score, matched = self._calculate_match_score(
                    manifest, query_lower, query_words, mode
                )

                if score > 0:
                    results.append(SearchResult(
                        manifest=manifest,
                        score=score,
                        matched_fields=matched,
                    ))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        # Cache results
        if self.config.enable_caching:
            self._set_cached(cache_key, results)

        return results[:limit]

    def _calculate_match_score(
        self,
        manifest: SkillManifest,
        query_lower: str,
        query_words: set[str],
        mode: SearchMode,
    ) -> tuple[float, list[str]]:
        """Calculate match score for a skill.

        Args:
            manifest: Skill manifest.
            query_lower: Lowercase query string.
            query_words: Set of query words.
            mode: Search mode.

        Returns:
            Tuple of (score, matched_fields).
        """
        score = 0.0
        matched: list[str] = []
        name_lower = manifest.name.lower()
        desc_lower = manifest.description.lower()

        if mode == SearchMode.EXACT:
            if name_lower == query_lower:
                score = 100.0
                matched.append("name")

        elif mode == SearchMode.PREFIX:
            if name_lower.startswith(query_lower):
                score = 90.0 - (len(name_lower) - len(query_lower))
                matched.append("name")

        elif mode == SearchMode.CONTAINS:
            if query_lower in name_lower:
                score += 80.0
                matched.append("name")
            if query_lower in desc_lower:
                score += 40.0
                matched.append("description")

        elif mode == SearchMode.FUZZY:
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
            name_words = set(name_lower.replace("-", " ").split())
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

            # Keyword matching
            keywords = set(kw.lower() for kw in manifest.metadata.keywords)
            keyword_overlap = len(query_words & keywords)
            if keyword_overlap > 0:
                score += keyword_overlap * 15.0
                matched.append("keywords")

        elif mode == SearchMode.TRIGGER:
            if manifest.triggers.matches(query_lower):
                score = manifest.triggers.priority + 50.0
                matched.append("trigger")

        return score, matched

    def _get_cached(
        self,
        key: str
    ) -> list[SearchResult] | None:
        """Get cached search results.

        Args:
            key: Cache key.

        Returns:
            Cached results or None if not found/expired.
        """
        if key not in self._search_cache:
            return None

        timestamp, results = self._search_cache[key]
        age = (datetime.now() - timestamp).total_seconds()

        if age > self.config.cache_ttl:
            del self._search_cache[key]
            return None

        return results

    def _set_cached(
        self,
        key: str,
        results: list[SearchResult]
    ) -> None:
        """Cache search results.

        Args:
            key: Cache key.
            results: Results to cache.
        """
        self._search_cache[key] = (datetime.now(), results)

    def _invalidate_cache(self) -> None:
        """Invalidate all cached search results."""
        self._search_cache.clear()

    def get_skill_context(
        self,
        name: str,
        include_resources: bool = False,
    ) -> str:
        """Get skill context string for agent prompt injection.

        This method loads the full skill content and generates a
        formatted context string suitable for injection into agent
        prompts.

        Args:
            name: Skill name.
            include_resources: Whether to include resource contents.

        Returns:
            Formatted context string.
        """
        manifest = self.loader.load_skill(name, include_resources)
        if manifest is None:
            return ""
        return manifest.get_context(include_resources)

    def get_multiple_contexts(
        self,
        names: list[str],
        include_resources: bool = False,
    ) -> dict[str, str]:
        """Get contexts for multiple skills.

        Args:
            names: List of skill names.
            include_resources: Whether to include resource contents.

        Returns:
            Dictionary mapping names to context strings.
        """
        return {
            name: self.get_skill_context(name, include_resources)
            for name in names
        }

    def find_matching_skills(self, user_input: str) -> list[SkillManifest]:
        """Find skills that match user input using triggers.

        This method uses trigger keywords and patterns to find
        relevant skills for a given user request.

        Args:
            user_input: User input or query text.

        Returns:
            List of matching manifests sorted by trigger priority.
        """
        matches: list[tuple[int, SkillManifest]] = []

        with self._lock:
            for name, trigger in self._triggers.items():
                if trigger.matches(user_input):
                    manifest = self._skills.get(name)
                    if manifest:
                        matches.append((trigger.priority, manifest))

        # Sort by priority descending
        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches]

    def get_all_domains(self) -> list[str]:
        """Get all registered domains.

        Returns:
            List of domain names.
        """
        with self._lock:
            return list(self._by_domain.keys())

    def get_all_tags(self) -> list[str]:
        """Get all registered tags.

        Returns:
            List of tag names.
        """
        with self._lock:
            return list(self._by_tag.keys())

    def get_skill_summary(self, name: str) -> dict[str, Any] | None:
        """Get a summary of a skill for display.

        Args:
            name: Skill name.

        Returns:
            Summary dictionary or None if not found.
        """
        manifest = self.get_skill(name)
        if manifest is None:
            return None

        return {
            "name": manifest.name,
            "description": manifest.description,
            "version": manifest.version,
            "category": manifest.category.value,
            "domain": manifest.metadata.extra.get("domain", "general"),
            "keywords": manifest.metadata.keywords,
            "tools_count": len(manifest.tools),
            "resources_count": len(manifest.resources),
            "loaded": manifest.loaded,
        }

    def list_all_summaries(self) -> list[dict[str, Any]]:
        """Get summaries for all registered skills.

        Returns:
            List of skill summary dictionaries.
        """
        summaries = []
        with self._lock:
            for name in self._skills:
                summary = self.get_skill_summary(name)
                if summary:
                    summaries.append(summary)
        return summaries

    def to_agentic_definitions(self) -> list[Any]:
        """Convert to agentic_workflows.skills.SkillDefinition format.

        Returns:
            List of SkillDefinition objects.
        """
        if self._agentic_registry is None:
            return []

        try:
            from agentic_workflows.skills import SkillDefinition

            definitions = []
            for manifest in self._skills.values():
                skill_def = SkillDefinition(
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
                definitions.append(skill_def)

            return definitions

        except ImportError:
            return []

    def sync_to_agentic(self) -> int:
        """Sync skills to the agentic_workflows.skills.SkillRegistry.

        Returns:
            Number of skills synced.
        """
        if self._agentic_registry is None:
            return 0

        definitions = self.to_agentic_definitions()
        synced = 0

        for skill_def in definitions:
            try:
                self._agentic_registry.register_skill(skill_def)
                synced += 1
            except Exception as e:
                logger.warning(f"Failed to sync skill {skill_def.name}: {e}")

        logger.info(f"Synced {synced} skills to agentic_workflows.skills")
        return synced

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary with statistics.
        """
        with self._lock:
            by_category = {
                cat.value: len(names)
                for cat, names in self._by_category.items()
            }

            return {
                "total_skills": len(self._skills),
                "domains": len(self._by_domain),
                "categories": len(self._by_category),
                "tags": len(self._by_tag),
                "by_category": by_category,
                "cache_entries": len(self._search_cache),
                "agentic_integrated": self._agentic_registry is not None,
            }

    def __len__(self) -> int:
        """Get number of registered skills."""
        with self._lock:
            return len(self._skills)

    def __contains__(self, name: str) -> bool:
        """Check if skill exists by name."""
        with self._lock:
            return name in self._skills

    def __iter__(self) -> Iterator[SkillManifest]:
        """Iterate over registered manifests."""
        with self._lock:
            return iter(list(self._skills.values()))

    def __repr__(self) -> str:
        return (
            f"OpenAISkillRegistry(skills={len(self)}, "
            f"domains={len(self._by_domain)})"
        )


# Global registry instance
_global_registry: OpenAISkillRegistry | None = None


def get_openai_skill_registry() -> OpenAISkillRegistry:
    """Get or create the global OpenAI skill registry.

    Returns:
        Global OpenAISkillRegistry instance.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = OpenAISkillRegistry()
    return _global_registry


def set_openai_skill_registry(registry: OpenAISkillRegistry) -> None:
    """Set the global OpenAI skill registry.

    Args:
        registry: Registry instance to set as global.
    """
    global _global_registry
    _global_registry = registry


def search_skills(query: str, **kwargs) -> list[SearchResult]:
    """Search for skills matching a query.

    Args:
        query: Search query.
        **kwargs: Additional search parameters.

    Returns:
        List of search results.
    """
    return get_openai_skill_registry().search(query, **kwargs)


def get_skill_context(name: str) -> str:
    """Get skill context for agent prompt.

    Args:
        name: Skill name.

    Returns:
        Formatted context string.
    """
    return get_openai_skill_registry().get_skill_context(name)
