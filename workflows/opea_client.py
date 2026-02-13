"""Intel OPEA Gateway connector — triggers RAG pipelines."""

import logging
import os

import httpx

logger = logging.getLogger("antigravity.opea")

OPEA_GATEWAY = os.environ.get("OPEA_GATEWAY", "http://opea-gateway:8888")
ACCELERATOR = os.environ.get("ACCELERATOR", "cpu")
OPEA_HARDWARE_MODE = os.environ.get("OPEA_HARDWARE_MODE", "auto")


async def trigger_rag(goal: str, tenant_id: str, context: str = "") -> dict:
    """Trigger a RAG pipeline via the OPEA Gateway."""
    payload = {
        "query": goal,
        "tenant_id": tenant_id,
        "context": context,
        "accelerator": ACCELERATOR,
        "hardware_mode": OPEA_HARDWARE_MODE,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{OPEA_GATEWAY}/v1/rag", json=payload)
            resp.raise_for_status()
            result = resp.json()
            logger.info(f"OPEA RAG result for tenant={tenant_id}: {result.get('status', 'unknown')}")
            return result
    except httpx.HTTPStatusError as e:
        logger.warning(f"OPEA Gateway returned {e.response.status_code}: {e.response.text}")
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.warning(f"OPEA Gateway unreachable: {e}", exc_info=True)
        return {"status": "unavailable", "message": str(e)}


async def ingest(file_path: str, tenant_id: str = "system") -> dict:
    """Ingest a document into semantic memory via OPEA pipeline."""
    payload = {
        "file_path": file_path,
        "tenant_id": tenant_id,
        "pipeline": "ingest",
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{OPEA_GATEWAY}/v1/ingest", json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning(f"OPEA ingest failed for {file_path}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
