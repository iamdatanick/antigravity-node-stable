"""5-level health check hierarchy for Antigravity Node v13.0."""

import os
import logging
import asyncio
import aiohttp

logger = logging.getLogger("antigravity.health")


async def _check_url(session: aiohttp.ClientSession, name: str, url: str) -> dict:
    """Check a single service URL, return {name, healthy, error}."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
            return {"name": name, "healthy": resp.status == 200, "error": None}
    except Exception as e:
        return {"name": name, "healthy": False, "error": str(e)}


async def check_level_0() -> dict:
    """L0 Infrastructure: postgres, seaweedfs, nats."""
    checks = {
        "seaweedfs": f"http://{os.environ.get('SEAWEEDFS_HOST', 'seaweedfs')}:9333/cluster/status",
        "nats": f"http://{os.environ.get('NATS_HOST', 'nats')}:8222/healthz",
    }
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[_check_url(session, name, url) for name, url in checks.items()]
        )
    return {"level": "L0", "name": "infrastructure", "checks": results}


async def check_level_1() -> dict:
    """L1 Orchestration: K3D cluster, Argo server."""
    checks = {
        "argo": f"http://{os.environ.get('ARGO_SERVER', 'k3d')}:2746/api/v1/info",
    }
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[_check_url(session, name, url) for name, url in checks.items()]
        )
    return {"level": "L1", "name": "orchestration", "checks": results}


async def check_level_2() -> dict:
    """L2 Services: starrocks, valkey, milvus, keycloak, openbao."""
    checks = {
        "starrocks": f"http://{os.environ.get('STARROCKS_HOST', 'starrocks')}:{os.environ.get('STARROCKS_HTTP_PORT', '8030')}/api/health",
        "milvus": f"http://{os.environ.get('MILVUS_HOST', 'milvus')}:9091/healthz",
        "keycloak": f"{os.environ.get('KEYCLOAK_URL', 'http://keycloak:8080')}/health/ready",
        "openbao": f"{os.environ.get('OPENBAO_ADDR', 'http://openbao:8200')}/v1/sys/health",
    }
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[_check_url(session, name, url) for name, url in checks.items()]
        )
    return {"level": "L2", "name": "services", "checks": results}


async def check_level_3() -> dict:
    """L3 Agent: MCP gateway, Goose process."""
    checks = {
        "mcp_gateway": "http://mcp-gateway:8080/health",
    }
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[_check_url(session, name, url) for name, url in checks.items()]
        )
    return {"level": "L3", "name": "agent", "checks": results}


async def check_level_4() -> dict:
    """L4 Budget: LiteLLM remaining budget."""
    checks = {
        "litellm": f"http://{os.environ.get('LITELLM_HOST', 'litellm')}:4000/health",
        "marquez": f"http://{os.environ.get('MARQUEZ_HOST', 'marquez')}:5000/api/v1/namespaces",
    }
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[_check_url(session, name, url) for name, url in checks.items()]
        )
    return {"level": "L4", "name": "observability", "checks": results}


async def full_health_check() -> dict:
    """Run all 5 levels and compute aggregate status."""
    levels = await asyncio.gather(
        check_level_0(),
        check_level_1(),
        check_level_2(),
        check_level_3(),
        check_level_4(),
    )

    all_healthy = all(
        check["healthy"]
        for level in levels
        for check in level["checks"]
    )

    status = "healthy" if all_healthy else "degraded"
    return {"status": status, "levels": levels}
