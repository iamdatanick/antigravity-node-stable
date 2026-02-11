"""4-level health check hierarchy for Antigravity Node v14.1.

Uses httpx (not aiohttp) because aiohttp's DNS resolver conflicts with
uvicorn's event loop, causing all checks to fail with empty errors when
called from FastAPI endpoints.
"""

import asyncio
import logging
import os

import httpx

logger = logging.getLogger("antigravity.health")

# Shared timeout — 5s gives slow-starting services time to respond
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
    """L0 Infrastructure: etcd, ceph."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        results = await asyncio.gather(
            _check_url(
                client,
                "etcd",
                f"http://{os.environ.get('ETCD_HOST', 'etcd')}:2379/health",
            ),
            _check_url(
                client,
                "ceph",
                f"http://{os.environ.get('CEPH_HOST', 'ceph-demo')}:8000",
                accept_any=True,
            ),
        )
    return {"level": "L0", "name": "infrastructure", "checks": list(results)}


async def check_level_1() -> dict:
    """L1 Secrets: openbao."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        results = await asyncio.gather(
            _check_url(
                client,
                "openbao",
                f"{os.environ.get('OPENBAO_ADDR', 'http://openbao:8200')}/v1/sys/health",
            ),
        )
    return {"level": "L1", "name": "secrets", "checks": list(results)}


async def check_level_2() -> dict:
    """L2 Inference: ovms."""
    ovms_rest = os.environ.get("OVMS_REST", "http://ovms:9001")
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        results = await asyncio.gather(
            _check_url(client, "ovms", f"{ovms_rest}/v2/health/live"),
        )
    return {"level": "L2", "name": "inference", "checks": list(results)}


async def check_level_3() -> dict:
    """L3 Observability: otel-collector."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        results = await asyncio.gather(
            _check_url(
                client,
                "otel_collector",
                f"http://{os.environ.get('OTEL_COLLECTOR_HOST', 'otel-collector')}:4318",
                accept_any=True,
            ),
        )
    return {"level": "L3", "name": "observability", "checks": list(results)}


async def full_health_check() -> dict:
    """Run all 4 levels and compute aggregate status."""
    levels = await asyncio.gather(
        check_level_0(),
        check_level_1(),
        check_level_2(),
        check_level_3(),
    )

    all_healthy = all(check["healthy"] for level in levels for check in level["checks"])

    status = "healthy" if all_healthy else "degraded"
    return {"status": status, "levels": list(levels)}
