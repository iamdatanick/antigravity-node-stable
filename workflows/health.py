"""5-level health check hierarchy for Antigravity Node v13.0.

Uses httpx (not aiohttp) because aiohttp's DNS resolver conflicts with
uvicorn's event loop, causing all checks to fail with empty errors when
called from FastAPI endpoints.
"""

import asyncio
import logging
import os

import httpx

logger = logging.getLogger("antigravity.health")

# Shared timeout — 5s gives slow-starting services (Keycloak, StarRocks) time
_TIMEOUT = httpx.Timeout(5.0, connect=3.0)


async def _check_url(client: httpx.AsyncClient, name: str, url: str, accept_any: bool = False) -> dict:
    """Check a single service URL, return {name, healthy, error}.

    If accept_any=True, any HTTP response (even 4xx/5xx) counts as healthy
    (service is reachable). Used for services without proper health endpoints.
    """
    try:
        resp = await client.get(url)
        healthy = True if accept_any else (resp.status_code == 200)
        return {"name": name, "healthy": healthy, "error": None}
    except httpx.ConnectError:
        return {"name": name, "healthy": False, "error": "connection refused"}
    except httpx.ConnectTimeout:
        return {"name": name, "healthy": False, "error": "connect timeout"}
    except httpx.ReadTimeout:
        return {"name": name, "healthy": False, "error": "read timeout"}
    except Exception as e:
        return {"name": name, "healthy": False, "error": f"{type(e).__name__}: {e}"}


async def _check_tcp(name: str, host: str, port: int) -> dict:
    """Check TCP connectivity to a service."""
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=5.0)
        writer.close()
        await writer.wait_closed()
        return {"name": name, "healthy": True, "error": None}
    except Exception as e:
        return {"name": name, "healthy": False, "error": f"{type(e).__name__}: {e}"}


async def check_level_0() -> dict:
    """L0 Infrastructure: seaweedfs, nats."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        results = await asyncio.gather(
            _check_url(
                client,
                "seaweedfs",
                f"http://{os.environ.get('SEAWEEDFS_HOST', 'seaweedfs')}:9333/cluster/status",
            ),
            _check_tcp("nats", os.environ.get("NATS_HOST", "nats"), 4222),
        )
    return {"level": "L0", "name": "infrastructure", "checks": list(results)}


async def check_level_1() -> dict:
    """L1 IAM + Lineage: keycloak, marquez."""
    keycloak_url = f"{os.environ.get('KEYCLOAK_URL', 'http://keycloak:8080')}/"
    marquez_url = f"http://{os.environ.get('MARQUEZ_HOST', 'marquez')}:5000/api/v1/namespaces"
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        results = await asyncio.gather(
            _check_url(client, "keycloak", keycloak_url, accept_any=True),
            _check_url(client, "marquez", marquez_url),
        )
    return {"level": "L1", "name": "iam_lineage", "checks": list(results)}


async def check_level_2() -> dict:
    """L2 Memory + Compute: starrocks, milvus, valkey, openbao, ovms, ollama."""
    ovms_rest = os.environ.get("OVMS_REST", "http://ovms:8000")
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        results = await asyncio.gather(
            _check_url(
                client,
                "starrocks",
                f"http://{os.environ.get('STARROCKS_HOST', 'starrocks')}:{os.environ.get('STARROCKS_HTTP_PORT', '8030')}/api/health",
            ),
            _check_url(
                client,
                "milvus",
                f"http://{os.environ.get('MILVUS_HOST', 'milvus')}:9091/healthz",
            ),
            _check_url(
                client,
                "openbao",
                f"{os.environ.get('OPENBAO_ADDR', 'http://openbao:8200')}/v1/sys/health",
            ),
            _check_url(client, "ovms", f"{ovms_rest}/v1/config", accept_any=True),
            _check_url(
                client,
                "ollama",
                f"{os.environ.get('OLLAMA_URL', 'http://ollama:11434')}/api/tags",
            ),
            _check_tcp("valkey", "valkey", 6379),
        )
    return {"level": "L2", "name": "memory_compute", "checks": list(results)}


async def check_level_3() -> dict:
    """L3 Agent: MCP servers (starrocks, filesystem)."""
    results = await asyncio.gather(
        _check_tcp("mcp_starrocks", "mcp-starrocks", 8000),
        _check_tcp("mcp_filesystem", "mcp-filesystem", 8000),
    )
    return {"level": "L3", "name": "agent", "checks": list(results)}


async def check_level_4() -> dict:
    """L4 Observability: budget-proxy, perses, opensearch, fluent-bit."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        results = await asyncio.gather(
            _check_url(client, "budget_proxy", "http://budget-proxy:4000/health"),
            _check_url(client, "perses", "http://perses:8080/api/v1/health"),
            _check_url(client, "opensearch", "http://opensearch:9200/_cluster/health"),
            _check_tcp("fluent_bit", "fluent-bit", 24224),
        )
    return {"level": "L4", "name": "observability", "checks": list(results)}


async def full_health_check() -> dict:
    """Run all 5 levels and compute aggregate status."""
    levels = await asyncio.gather(
        check_level_0(),
        check_level_1(),
        check_level_2(),
        check_level_3(),
        check_level_4(),
    )

    all_healthy = all(check["healthy"] for level in levels for check in level["checks"])

    status = "healthy" if all_healthy else "degraded"
    return {"status": status, "levels": list(levels)}
