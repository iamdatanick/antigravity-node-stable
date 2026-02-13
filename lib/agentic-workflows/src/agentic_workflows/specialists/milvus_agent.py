"""Milvus specialist agent for vector database operations.

Handles embedding storage, similarity search, and vector indexing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import SpecialistAgent, SpecialistCapability, SpecialistConfig


@dataclass
class MilvusConfig(SpecialistConfig):
    """Milvus-specific configuration."""

    host: str = "localhost"
    port: int = 19530
    user: str = ""
    password: str = ""
    db_name: str = "default"
    default_collection: str = "embeddings"


class MilvusAgent(SpecialistAgent):
    """Specialist agent for Milvus vector database.

    Capabilities:
    - Vector storage and retrieval
    - Similarity search
    - Collection management
    - Index management
    """

    def __init__(self, config: MilvusConfig | None = None, **kwargs):
        self.milvus_config = config or MilvusConfig()
        super().__init__(config=self.milvus_config, **kwargs)

        self._client = None

        self.register_handler("insert", self._insert_vectors)
        self.register_handler("search", self._search_vectors)
        self.register_handler("query", self._query_vectors)
        self.register_handler("delete", self._delete_vectors)
        self.register_handler("create_collection", self._create_collection)
        self.register_handler("list_collections", self._list_collections)
        self.register_handler("create_index", self._create_index)

    @property
    def capabilities(self) -> list[SpecialistCapability]:
        return [
            SpecialistCapability.VECTOR_SEARCH,
            SpecialistCapability.EMBEDDING_STORAGE,
            SpecialistCapability.SIMILARITY_SEARCH,
        ]

    @property
    def service_name(self) -> str:
        return "Milvus"

    async def _connect(self) -> None:
        """Connect to Milvus."""
        try:
            from pymilvus import connections, utility

            connections.connect(
                alias="default",
                host=self.milvus_config.host,
                port=self.milvus_config.port,
                user=self.milvus_config.user,
                password=self.milvus_config.password,
                db_name=self.milvus_config.db_name,
            )
            self._client = True
        except ImportError:
            self.logger.warning("pymilvus not installed")

    async def _disconnect(self) -> None:
        """Disconnect from Milvus."""
        try:
            from pymilvus import connections

            connections.disconnect("default")
        except Exception:
            pass
        self._client = None

    async def _health_check(self) -> bool:
        """Check Milvus health."""
        try:
            from pymilvus import utility

            return utility.get_server_version() is not None
        except Exception:
            return False

    async def _insert_vectors(
        self,
        collection: str | None = None,
        vectors: list[list[float]] = None,
        ids: list[int | str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Insert vectors into collection.

        Args:
            collection: Collection name.
            vectors: Vector embeddings.
            ids: Optional vector IDs.
            metadata: Optional metadata for each vector.

        Returns:
            Insert result.
        """
        from pymilvus import Collection

        collection = collection or self.milvus_config.default_collection
        col = Collection(collection)

        # Prepare data
        data = [vectors]
        if ids:
            data.insert(0, ids)

        # Add metadata fields if present
        if metadata and col.schema:
            for field in col.schema.fields:
                if field.name not in ("id", "embedding", "vector"):
                    field_values = [m.get(field.name) for m in metadata]
                    data.append(field_values)

        result = col.insert(data)

        return {
            "collection": collection,
            "inserted": result.insert_count,
            "ids": list(result.primary_keys),
        }

    async def _search_vectors(
        self,
        collection: str | None = None,
        query_vectors: list[list[float]] = None,
        top_k: int = 10,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
        metric_type: str = "L2",
    ) -> list[dict[str, Any]]:
        """Search for similar vectors.

        Args:
            collection: Collection name.
            query_vectors: Query embeddings.
            top_k: Number of results per query.
            filter_expr: Filter expression.
            output_fields: Fields to return.
            metric_type: Distance metric (L2, IP, COSINE).

        Returns:
            Search results.
        """
        from pymilvus import Collection

        collection = collection or self.milvus_config.default_collection
        col = Collection(collection)
        col.load()

        search_params = {"metric_type": metric_type, "params": {"nprobe": 10}}

        results = col.search(
            data=query_vectors,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=output_fields or ["*"],
        )

        output = []
        for hits in results:
            query_results = []
            for hit in hits:
                query_results.append(
                    {
                        "id": hit.id,
                        "distance": hit.distance,
                        "entity": hit.entity.to_dict() if hasattr(hit, "entity") else {},
                    }
                )
            output.append(query_results)

        return output

    async def _query_vectors(
        self,
        collection: str | None = None,
        filter_expr: str = "",
        output_fields: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query vectors by filter expression.

        Args:
            collection: Collection name.
            filter_expr: Filter expression.
            output_fields: Fields to return.
            limit: Maximum results.

        Returns:
            Matching vectors.
        """
        from pymilvus import Collection

        collection = collection or self.milvus_config.default_collection
        col = Collection(collection)
        col.load()

        results = col.query(
            expr=filter_expr,
            output_fields=output_fields or ["*"],
            limit=limit,
        )

        return results

    async def _delete_vectors(
        self,
        collection: str | None = None,
        ids: list[int | str] | None = None,
        filter_expr: str | None = None,
    ) -> dict[str, Any]:
        """Delete vectors from collection.

        Args:
            collection: Collection name.
            ids: Vector IDs to delete.
            filter_expr: Filter expression for deletion.

        Returns:
            Deletion result.
        """
        from pymilvus import Collection

        collection = collection or self.milvus_config.default_collection
        col = Collection(collection)

        if ids:
            expr = f"id in {ids}"
        elif filter_expr:
            expr = filter_expr
        else:
            return {"error": "Must provide ids or filter_expr"}

        result = col.delete(expr)

        return {"collection": collection, "deleted": result.delete_count}

    async def _create_collection(
        self,
        name: str,
        dimension: int,
        metric_type: str = "L2",
        description: str = "",
        extra_fields: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create a new collection.

        Args:
            name: Collection name.
            dimension: Vector dimension.
            metric_type: Distance metric.
            description: Collection description.
            extra_fields: Additional schema fields.

        Returns:
            Creation result.
        """
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

        if utility.has_collection(name):
            return {"collection": name, "created": False, "exists": True}

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        ]

        if extra_fields:
            for field in extra_fields:
                dtype = getattr(DataType, field.get("dtype", "VARCHAR"))
                fields.append(
                    FieldSchema(
                        name=field["name"],
                        dtype=dtype,
                        max_length=field.get("max_length", 256),
                    )
                )

        schema = CollectionSchema(fields=fields, description=description)
        Collection(name=name, schema=schema)

        return {"collection": name, "created": True, "dimension": dimension}

    async def _list_collections(self) -> list[str]:
        """List all collections."""
        from pymilvus import utility

        return utility.list_collections()

    async def _create_index(
        self,
        collection: str | None = None,
        field_name: str = "embedding",
        index_type: str = "IVF_FLAT",
        metric_type: str = "L2",
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an index on a field.

        Args:
            collection: Collection name.
            field_name: Field to index.
            index_type: Index type (IVF_FLAT, HNSW, etc.).
            metric_type: Distance metric.
            params: Index parameters.

        Returns:
            Index creation result.
        """
        from pymilvus import Collection

        collection = collection or self.milvus_config.default_collection
        col = Collection(collection)

        index_params = {
            "index_type": index_type,
            "metric_type": metric_type,
            "params": params or {"nlist": 1024},
        }

        col.create_index(field_name, index_params)

        return {"collection": collection, "field": field_name, "indexed": True}
