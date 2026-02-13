"""OpenMetadata specialist agent for data governance.

Handles data catalog, lineage tracking, and metadata management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import SpecialistAgent, SpecialistConfig, SpecialistCapability


@dataclass
class OpenMetadataConfig(SpecialistConfig):
    """OpenMetadata-specific configuration."""

    api_endpoint: str = "http://localhost:8585/api/v1"
    jwt_token: str = ""


class OpenMetadataAgent(SpecialistAgent):
    """Specialist agent for OpenMetadata.

    Capabilities:
    - Data cataloging
    - Lineage tracking
    - Metadata management
    - Data quality
    """

    def __init__(self, config: OpenMetadataConfig | None = None, **kwargs):
        self.om_config = config or OpenMetadataConfig()
        super().__init__(config=self.om_config, **kwargs)

        self._session = None

        self.register_handler("search", self._search)
        self.register_handler("get_table", self._get_table)
        self.register_handler("get_lineage", self._get_lineage)
        self.register_handler("add_tag", self._add_tag)
        self.register_handler("list_databases", self._list_databases)
        self.register_handler("get_glossary_terms", self._get_glossary_terms)

    @property
    def capabilities(self) -> list[SpecialistCapability]:
        return [
            SpecialistCapability.DATA_CATALOG,
            SpecialistCapability.LINEAGE_TRACKING,
            SpecialistCapability.METADATA_MANAGEMENT,
        ]

    @property
    def service_name(self) -> str:
        return "OpenMetadata"

    async def _connect(self) -> None:
        """Connect to OpenMetadata API."""
        import aiohttp

        headers = {"Authorization": f"Bearer {self.om_config.jwt_token}"}
        self._session = aiohttp.ClientSession(headers=headers)

    async def _disconnect(self) -> None:
        """Disconnect from OpenMetadata."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _health_check(self) -> bool:
        """Check OpenMetadata health."""
        if self._session is None:
            return False
        try:
            url = f"{self.om_config.api_endpoint}/system/version"
            async with self._session.get(url) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def _search(
        self,
        query: str,
        entity_type: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search the data catalog.

        Args:
            query: Search query.
            entity_type: Filter by entity type.
            limit: Maximum results.

        Returns:
            Search results.
        """
        if self._session is None:
            return {"error": "Not connected"}

        url = f"{self.om_config.api_endpoint}/search/query"
        params = {"q": query, "size": limit}
        if entity_type:
            params["index"] = entity_type

        async with self._session.get(url, params=params) as resp:
            if resp.status == 200:
                return await resp.json()
            return {"error": f"Search failed: {resp.status}"}

    async def _get_table(self, fqn: str) -> dict[str, Any]:
        """Get table metadata by fully qualified name.

        Args:
            fqn: Fully qualified table name.

        Returns:
            Table metadata.
        """
        if self._session is None:
            return {"error": "Not connected"}

        url = f"{self.om_config.api_endpoint}/tables/name/{fqn}"
        params = {"fields": "columns,tableConstraints,tableProfile,tags,owner"}

        async with self._session.get(url, params=params) as resp:
            if resp.status == 200:
                return await resp.json()
            return {"error": f"Table not found: {resp.status}"}

    async def _get_lineage(
        self,
        entity_type: str,
        fqn: str,
        upstream_depth: int = 3,
        downstream_depth: int = 3,
    ) -> dict[str, Any]:
        """Get entity lineage.

        Args:
            entity_type: Entity type (table, pipeline, etc.).
            fqn: Fully qualified name.
            upstream_depth: Upstream traversal depth.
            downstream_depth: Downstream traversal depth.

        Returns:
            Lineage graph.
        """
        if self._session is None:
            return {"error": "Not connected"}

        url = f"{self.om_config.api_endpoint}/lineage/{entity_type}/name/{fqn}"
        params = {"upstreamDepth": upstream_depth, "downstreamDepth": downstream_depth}

        async with self._session.get(url, params=params) as resp:
            if resp.status == 200:
                return await resp.json()
            return {"error": f"Lineage not found: {resp.status}"}

    async def _add_tag(
        self,
        entity_type: str,
        entity_id: str,
        tag_fqn: str,
    ) -> dict[str, Any]:
        """Add a tag to an entity.

        Args:
            entity_type: Entity type.
            entity_id: Entity ID.
            tag_fqn: Tag fully qualified name.

        Returns:
            Updated entity.
        """
        if self._session is None:
            return {"error": "Not connected"}

        url = f"{self.om_config.api_endpoint}/{entity_type}/{entity_id}"

        payload = [
            {
                "op": "add",
                "path": "/tags/-",
                "value": {"tagFQN": tag_fqn, "source": "Classification"},
            }
        ]

        async with self._session.patch(url, json=payload) as resp:
            if resp.status == 200:
                return await resp.json()
            return {"error": f"Failed to add tag: {resp.status}"}

    async def _list_databases(self, service: str | None = None) -> list[dict[str, Any]]:
        """List all databases.

        Args:
            service: Filter by database service.

        Returns:
            List of databases.
        """
        if self._session is None:
            return []

        url = f"{self.om_config.api_endpoint}/databases"
        params = {}
        if service:
            params["service"] = service

        async with self._session.get(url, params=params) as resp:
            if resp.status == 200:
                result = await resp.json()
                return result.get("data", [])
            return []

    async def _get_glossary_terms(self, glossary: str | None = None) -> list[dict[str, Any]]:
        """Get glossary terms.

        Args:
            glossary: Filter by glossary name.

        Returns:
            List of glossary terms.
        """
        if self._session is None:
            return []

        url = f"{self.om_config.api_endpoint}/glossaryTerms"
        params = {}
        if glossary:
            params["glossary"] = glossary

        async with self._session.get(url, params=params) as resp:
            if resp.status == 200:
                result = await resp.json()
                return result.get("data", [])
            return []
