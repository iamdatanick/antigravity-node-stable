"""4-level health check hierarchy for Antigravity Node v14.1."""
import asyncio
import logging
import os
import httpx

logger = logging.getLogger("antigravity.health")
_TIMEOUT = httpx.Timeout(5.0, connect=3.0)

async def _check_url(client: httpx.AsyncClient, name: str, url: str, accept_codes: list = [200]) -> dict:
    try:
        resp = await client.get(url)
        healthy = resp.status_code in accept_codes
        return {"name": name, "healthy": healthy, "error": None}
    except Exception as e:
        return {"name": name, "healthy": False, "error": str(e)}

async def check_level_0():
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        results = await asyncio.gather(
            _check_url(client, "etcd", f"http://{os.environ.get('ETCD_HOST', 'etcd')}:2379/health"),
            _check_url(client, "ceph", f"http://{os.environ.get('CEPH_HOST', 'ceph-demo')}:8000", accept_codes=[200, 404]),
        )
    return {"level": "L0", "name": "infrastructure", "checks": list(results)}

async def check_level_1():
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        results = await asyncio.gather(
            _check_url(client, "openbao", f"{os.environ.get('OPENBAO_ADDR', 'http://openbao:8200')}/v1/sys/health", accept_codes=[200, 473]),
        )
    return {"level": "L1", "name": "secrets", "checks": list(results)}

async def check_level_2():
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        results = await asyncio.gather(
            _check_url(client, "ovms", f"{os.environ.get('OVMS_REST_URL', 'http://ovms:9001')}/v2/health/live"),
        )
    return {"level": "L2", "name": "inference", "checks": list(results)}

async def check_level_3():
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        results = await asyncio.gather(
            _check_url(client, "otel_collector", f"http://{os.environ.get('OTEL_COLLECTOR_HOST', 'otel-collector')}:13133"),
        )
    return {"level": "L3", "name": "observability", "checks": list(results)}

async def full_health_check():
    levels = await asyncio.gather(check_level_0(), check_level_1(), check_level_2(), check_level_3())
    all_healthy = all(check["healthy"] for level in levels for check in level["checks"])
    return {"status": "healthy" if all_healthy else "degraded", "levels": list(levels)}
