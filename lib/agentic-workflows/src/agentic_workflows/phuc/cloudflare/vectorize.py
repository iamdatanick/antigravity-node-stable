"""Vectorize vector database for PHUC RAG pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import httpx
import os
import json


@dataclass
class VectorizeConfig:
    """Vectorize index configuration."""
    account_id: str = field(default_factory=lambda: os.getenv("CF_ACCOUNT_ID", ""))
    api_token: str = field(default_factory=lambda: os.getenv("CF_API_TOKEN", ""))
    index_name: str = "phuc-afancy-field-7cbb"
    dimensions: int = 768  # BGE-base
    metric: str = "cosine"


@dataclass
class VectorMatch:
    """Vector search result."""
    id: str
    score: float
    values: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)


class Vectorize:
    """Cloudflare Vectorize client."""
    
    BASE_URL = "https://api.cloudflare.com/client/v4"
    
    def __init__(self, config: VectorizeConfig = None):
        self.config = config or VectorizeConfig()
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_token}",
            "Content-Type": "application/json"
        }
    
    @property
    def base_url(self) -> str:
        return f"{self.BASE_URL}/accounts/{self.config.account_id}/vectorize/v2/indexes/{self.config.index_name}"
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(headers=self.headers, timeout=30.0)
        return self._client
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def insert(self, vectors: list[dict]) -> dict:
        """Insert vectors with metadata.
        
        Args:
            vectors: List of dicts with 'id', 'values', and optional 'metadata'
        """
        client = await self._get_client()
        
        # Format for Vectorize API
        ndjson_lines = []
        for v in vectors:
            ndjson_lines.append(json.dumps({
                "id": v["id"],
                "values": v["values"],
                "metadata": v.get("metadata", {})
            }))
        
        response = await client.post(
            f"{self.base_url}/insert",
            content="\n".join(ndjson_lines),
            headers={**self.headers, "Content-Type": "application/x-ndjson"}
        )
        response.raise_for_status()
        return response.json()
    
    async def upsert(self, vectors: list[dict]) -> dict:
        """Upsert vectors (insert or update)."""
        client = await self._get_client()
        
        ndjson_lines = []
        for v in vectors:
            ndjson_lines.append(json.dumps({
                "id": v["id"],
                "values": v["values"],
                "metadata": v.get("metadata", {})
            }))
        
        response = await client.post(
            f"{self.base_url}/upsert",
            content="\n".join(ndjson_lines),
            headers={**self.headers, "Content-Type": "application/x-ndjson"}
        )
        response.raise_for_status()
        return response.json()
    
    async def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filter: dict = None,
        return_values: bool = False,
        return_metadata: bool = True
    ) -> list[VectorMatch]:
        """Query similar vectors."""
        client = await self._get_client()
        
        payload = {
            "vector": vector,
            "topK": top_k,
            "returnValues": return_values,
            "returnMetadata": return_metadata
        }
        
        if filter:
            payload["filter"] = filter
        
        response = await client.post(f"{self.base_url}/query", json=payload)
        response.raise_for_status()
        
        matches = []
        for m in response.json().get("result", {}).get("matches", []):
            matches.append(VectorMatch(
                id=m["id"],
                score=m["score"],
                values=m.get("values"),
                metadata=m.get("metadata", {})
            ))
        
        return matches
    
    async def get_by_ids(self, ids: list[str]) -> list[VectorMatch]:
        """Get vectors by IDs."""
        client = await self._get_client()
        
        response = await client.post(
            f"{self.base_url}/getByIds",
            json={"ids": ids}
        )
        response.raise_for_status()
        
        matches = []
        for v in response.json().get("result", {}).get("vectors", []):
            matches.append(VectorMatch(
                id=v["id"],
                score=1.0,
                values=v.get("values"),
                metadata=v.get("metadata", {})
            ))
        
        return matches
    
    async def delete(self, ids: list[str]) -> dict:
        """Delete vectors by ID."""
        client = await self._get_client()
        
        response = await client.post(
            f"{self.base_url}/deleteByIds",
            json={"ids": ids}
        )
        response.raise_for_status()
        return response.json()
    
    async def get_info(self) -> dict:
        """Get index information."""
        client = await self._get_client()
        response = await client.get(self.base_url.rsplit("/", 1)[0])
        response.raise_for_status()
        return response.json().get("result", {})
    
    # PHUC-specific methods
    async def search_documents(
        self,
        query_vector: list[float],
        user_id: str = None,
        doc_type: str = None,
        top_k: int = 5
    ) -> list[VectorMatch]:
        """Search document embeddings with optional filters."""
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if doc_type:
            filters["doc_type"] = doc_type
        
        return await self.query(
            vector=query_vector,
            top_k=top_k,
            filter=filters if filters else None,
            return_metadata=True
        )
    
    async def index_document_chunks(
        self,
        doc_id: str,
        chunks: list[dict],
        embeddings: list[list[float]]
    ) -> dict:
        """Index document chunks with embeddings."""
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": f"{doc_id}_chunk_{i}",
                "values": embedding,
                "metadata": {
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "text": chunk.get("text", "")[:500],  # Limit metadata text
                    **{k: v for k, v in chunk.items() if k != "text"}
                }
            })
        
        return await self.upsert(vectors)
    
    async def delete_document(self, doc_id: str) -> dict:
        """Delete all chunks for a document."""
        # First, find all chunk IDs for this document
        # Note: This requires querying by metadata filter
        # For now, we'll need to track chunk IDs separately
        # This is a limitation of Vectorize's current API
        pass
