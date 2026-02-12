### C:\Users\NickV\OneDrive\Desktop\Antigravity-Node/workflows/health.py
```python
1: import asyncio, os, httpx
2: async def _check_url(client, name, url, accept_codes=[200]):
3:     try:
4:         resp = await client.get(url)
5:         return {"name": name, "healthy": resp.status_code in accept_codes}
6:     except Exception: return {"name": name, "healthy": False}
7: 
8: async def full_health_check():
9:     async with httpx.AsyncClient(timeout=3.0) as client:
10:         l0 = await asyncio.gather(
11:             _check_url(client, "etcd", "http://etcd:2379/health"),
12:             _check_url(client, "ceph", "http://ceph-demo:8000", accept_codes=[200, 404])
13:         )
14:         l1 = await _check_url(client, "openbao", "http://openbao:8200/v1/sys/health", accept_codes=[200, 473])
15:         l2 = await _check_url(client, "ovms", "http://ovms:9001/v2/health/live")
16:         l3 = await _check_url(client, "otel", "http://otel-collector:13133")
17:     levels = [{"level": "L0", "checks": l0}, {"level": "L1", "checks": [l1]}, {"level": "L2", "checks": [l2]}, {"level": "L3", "checks": [l3]}]
18:     all_h = all(c["healthy"] for l in levels for c in l["checks"])
19:     return {"status": "healthy" if all_h else "degraded", "levels": levels}
```
