import os

import httpx


async def get_embeddings(text: str) -> list[float]:
    proxy_url = os.environ.get("LITELLM_BASE_URL", "http://budget-proxy:4000")
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{proxy_url}/v1/embeddings", json={"input": text, "model": "text-embedding-3-small"})
        return resp.json()["data"][0]["embedding"]
