"""Milvus MCP tool for vector memory search."""

import os
import logging
import json

logger = logging.getLogger("antigravity.tools.milvus")

MILVUS_HOST = os.environ.get("MILVUS_HOST", "milvus")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))


def connect():
    """Connect to Milvus."""
    from pymilvus import connections
    connections.connect(host=MILVUS_HOST, port=str(MILVUS_PORT))
    logger.info(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")


def search_vectors(query_embedding: list, collection_name: str = "semantic_memory", limit: int = 5) -> list:
    """Search for similar vectors in a Milvus collection."""
    from pymilvus import Collection

    try:
        connect()
        collection = Collection(collection_name)
        collection.load()
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=limit,
            output_fields=["content", "source_uri"],
        )
        return [
            {
                "id": hit.id,
                "distance": hit.distance,
                "content": hit.entity.get("content", ""),
                "source": hit.entity.get("source_uri", ""),
            }
            for hits in results
            for hit in hits
        ]
    except Exception as e:
        logger.warning(f"Milvus search failed: {e}")
        return []


def store_vector(doc_id: str, embedding: list, content: str, source_uri: str = "", collection_name: str = "semantic_memory"):
    """Store a vector in Milvus."""
    from pymilvus import Collection

    try:
        connect()
        collection = Collection(collection_name)
        collection.insert([
            [doc_id],
            [embedding],
            [content],
            [source_uri],
        ])
        collection.flush()
        logger.info(f"Stored vector: {doc_id}")
    except Exception as e:
        logger.warning(f"Milvus store failed: {e}")
