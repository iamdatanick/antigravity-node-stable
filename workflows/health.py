import asyncio
import httpx
from workflows.resilience import get_circuit_states


async def _check_url(client, name, url, accept_codes=(200,)):
    try:
        resp = await client.get(url, timeout=5.0)
        return {"name": name, "healthy": resp.status_code in accept_codes, "error": None}
    except Exception as exc:
        return {"name": name, "healthy": False, "error": str(exc)[:120]}


async def full_health_check():
    async with httpx.AsyncClient(timeout=5.0) as client:
        l0 = await asyncio.gather(
            _check_url(client, "etcd", "http://etcd:2379/health"),
            _check_url(client, "seaweedfs", "http://seaweedfs:9333/cluster/status"),
        )
        l1 = await asyncio.gather(
            _check_url(client, "openbao", "http://openbao:8200/v1/sys/health", accept_codes=(200, 473)),
            _check_url(client, "keycloak", "http://keycloak:8080", accept_codes=(200, 302, 303)),
        )
        l2 = await asyncio.gather(
            _check_url(client, "ovms", "http://ovms:8000/v1/config"),
            _check_url(client, "starrocks", "http://starrocks:8030/api/health"),
            _check_url(client, "milvus", "http://milvus:9091/healthz"),
        )
        l3 = await asyncio.gather(
            _check_url(client, "marquez", "http://marquez:5000/api/v1/namespaces"),
            _check_url(client, "opensearch", "http://opensearch:9200/_cluster/health"),
        )
    levels = [
        {"level": "L0", "name": "infrastructure", "checks": list(l0)},
        {"level": "L1", "name": "secrets_iam", "checks": list(l1)},
        {"level": "L2", "name": "inference_memory", "checks": list(l2)},
        {"level": "L3", "name": "observability", "checks": list(l3)},
    ]
    all_h = all(c["healthy"] for lvl in levels for c in lvl["checks"])
    return {"status": "healthy" if all_h else "degraded", "levels": levels}
