
import asyncio
import os
import httpx

async def _check_url(client, name, url, accept_codes=[200]):
    try:
        resp = await client.get(url)
        return {"name": name, "healthy": resp.status_code in accept_codes}
    except Exception:
        return {"name": name, "healthy": False}

async def full_health_check():
    async with httpx.AsyncClient(timeout=3.0) as client:
        l0 = await asyncio.gather(
            _check_url(client, "etcd", "http://etcd:2379/health"),
            _check_url(client, "ceph", "http://ceph-demo:8000", accept_codes=[200, 404])
        )
        l1 = await _check_url(client, "openbao", "http://openbao:8200/v1/sys/health", accept_codes=[200, 473])
        l2 = await _check_url(client, "ovms", "http://ovms:9001/v2/health/live")
        l3 = await _check_url(client, "otel", "http://otel-collector:13133")
    
    levels = [
        {"level": "L0", "checks": l0},
        {"level": "L1", "checks": [l1]},
        {"level": "L2", "checks": [l2]},
        {"level": "L3", "checks": [l3]}
    ]
    all_healthy = all(c["healthy"] for l in levels for c in l["checks"])
    return {"status": "healthy" if all_healthy else "degraded", "levels": levels}
