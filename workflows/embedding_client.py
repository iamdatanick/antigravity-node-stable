
import os
import httpx
import logging

logger = logging.getLogger("embedding-client")

async def get_embeddings(text: str) -> list[float]:
    """Call budget-proxy for OpenAI embeddings."""
    proxy_url = os.environ.get("LITELLM_BASE_URL", "http://budget-proxy:4055")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                f"{proxy_url}/v1/embeddings",
                json={"model": "text-embedding-3-small", "input": text}
            )
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
            logger.error(f"Embedding failed with status {resp.status_code}: {resp.text}")
            return []
        except Exception as e:
            logger.error(f"Embedding connection failed: {e}")
            return []
