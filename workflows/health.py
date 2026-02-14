"""Antigravity Node v14.1 — 4-level health check hierarchy."""

import asyncio

import httpx


async def _check_url(client, name, url, accept_codes=(200,), accept_any=False):
    """Check an HTTP endpoint. Returns dict with name, healthy, error."""
    try:
        resp = await client.get(url, timeout=5.0)
        healthy = accept_any or resp.status_code in accept_codes
        return {"name": name, "healthy": healthy, "error": None}
    except httpx.ConnectTimeout:
        return {"name": name, "healthy": False, "error": "connect timeout"}
    except Exception as exc:
        return {"name": name, "healthy": False, "error": str(exc)[:120]}


async def _check_tcp(name, host, port, timeout=5.0):
    """Check a TCP port. Returns dict with name, healthy, error."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return {"name": name, "healthy": True, "error": None}
    except Exception as exc:
        return {"name": name, "healthy": False, "error": str(exc)[:120]}


async def check_level_0():
    """L0: Infrastructure — etcd + ceph."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        checks = await asyncio.gather(
            _check_url(client, "etcd", "http://etcd:2379/health"),
            _check_url(client, "ceph", "http://ceph:5000"),
        )
    return {"level": "L0", "name": "infrastructure", "checks": list(checks)}


async def check_level_1():
    """L1: Secrets — openbao."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        checks = await asyncio.gather(
            _check_url(client, "openbao", "http://openbao:8200/v1/sys/health", accept_codes=(200, 473)),
        )
    return {"level": "L1", "name": "secrets", "checks": list(checks)}


async def check_level_2():
    """L2: Inference — ovms."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        checks = await asyncio.gather(
            _check_url(client, "ovms", "http://ovms:9001/v1/config"),
        )
    return {"level": "L2", "name": "inference", "checks": list(checks)}


async def check_level_3():
    """L3: Observability — otel-collector."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        checks = await asyncio.gather(
            _check_url(client, "otel_collector", "http://otel-collector:13133/"),
        )
    return {"level": "L3", "name": "observability", "checks": list(checks)}


async def full_health_check():
    """Run all levels and return aggregate status."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        l0 = await asyncio.gather(
            _check_url(client, "etcd", "http://etcd:2379/health"),
            _check_url(client, "ceph", "http://ceph:5000"),
        )
        l1 = await asyncio.gather(
            _check_url(client, "openbao", "http://openbao:8200/v1/sys/health", accept_codes=(200, 473)),
        )
        l2 = await asyncio.gather(
            _check_url(client, "ovms", "http://ovms:9001/v1/config"),
        )
        l3 = await asyncio.gather(
            _check_url(client, "otel_collector", "http://otel-collector:13133/"),
        )
    levels = [
        {"level": "L0", "name": "infrastructure", "checks": list(l0)},
        {"level": "L1", "name": "secrets", "checks": list(l1)},
        {"level": "L2", "name": "inference", "checks": list(l2)},
        {"level": "L3", "name": "observability", "checks": list(l3)},
    ]
    all_h = all(c["healthy"] for lvl in levels for c in lvl["checks"])
    return {"status": "healthy" if all_h else "degraded", "levels": levels}
