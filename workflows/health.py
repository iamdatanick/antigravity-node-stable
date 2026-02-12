import asyncio, os, httpx, time
async def _check_url(client, name, url, accept_codes=[200]):
    try:
        # Increased timeout to 10s for RGW specifically
        resp = await client.get(url, timeout=10.0)
        return {"name": name, "healthy": resp.status_code in accept_codes}
    except Exception: return {"name": name, "healthy": False}

async def full_health_check():
    async with httpx.AsyncClient(timeout=5.0) as client:
        l0 = await asyncio.gather(
            _check_url(client, "etcd", "http://etcd:2379/health"),
            # RGW can be slow; allow 404/200/403 (any response is a sign of life)
            _check_url(client, "ceph", "http://ceph-demo:8000", accept_codes=[200, 404, 403])
        )
        l1 = await _check_url(client, "openbao", "http://openbao:8200/v1/sys/health", accept_codes=[200, 473])
        l2 = await _check_url(client, "ovms", "http://ovms:9001/v2/health/live")
        l3 = await _check_url(client, "otel", "http://otel-collector:13133")
    levels = [{"level": "L0", "checks": l0}, {"level": "L1", "checks": [l1]}, {"level": "L2", "checks": [l2]}, {"level": "L3", "checks": [l3]}]
    all_h = all(c["healthy"] for l in levels for c in l["checks"])
    return {"status": "healthy" if all_h else "degraded", "levels": levels}
