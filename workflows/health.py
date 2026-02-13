import asyncio
import httpx
from workflows.resilience import get_circuit_states


async def _check_url(client, name, url, accept_codes=(200,), accept_any=False):
    """Check URL endpoint health.
    
    Args:
        client: httpx AsyncClient
        name: Service name
        url: URL to check
        accept_codes: Tuple of acceptable status codes
        accept_any: If True, any response (even 500) counts as healthy (reachable)
    """
    try:
        resp = await client.get(url, timeout=5.0)
        if accept_any:
            return {"name": name, "healthy": True, "error": None}
        return {"name": name, "healthy": resp.status_code in accept_codes, "error": None}
    except httpx.ConnectTimeout as exc:
        return {"name": name, "healthy": False, "error": f"connect timeout: {str(exc)[:100]}"}
    except Exception as exc:
        return {"name": name, "healthy": False, "error": str(exc)[:120]}


async def _check_tcp(name, host, port):
    """Check TCP port health.
    
    Args:
        name: Service name
        host: Hostname
        port: Port number
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=5.0
        )
        writer.close()
        await writer.wait_closed()
        return {"name": name, "healthy": True, "error": None}
    except (ConnectionRefusedError, OSError) as exc:
        return {"name": name, "healthy": False, "error": f"Connection refused: {str(exc)[:100]}"}
    except TimeoutError:
        return {"name": name, "healthy": False, "error": "Timeout"}
    except Exception as exc:
        return {"name": name, "healthy": False, "error": str(exc)[:120]}


async def check_level_0():
    """L0: Infrastructure - etcd + ceph (seaweedfs)."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        checks = await asyncio.gather(
            _check_url(client, "etcd", "http://etcd:2379/health"),
            _check_url(client, "ceph", "http://seaweedfs:9333/cluster/status"),
        )
    return {"level": "L0", "name": "infrastructure", "checks": list(checks)}


async def check_level_1():
    """L1: Secrets - openbao."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        checks = await asyncio.gather(
            _check_url(client, "openbao", "http://openbao:8200/v1/sys/health", accept_codes=(200, 473)),
        )
    return {"level": "L1", "name": "secrets", "checks": list(checks)}


async def check_level_2():
    """L2: Inference - ovms."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        checks = await asyncio.gather(
            _check_url(client, "ovms", "http://ovms:8000/v1/config"),
        )
    return {"level": "L2", "name": "inference", "checks": list(checks)}


async def check_level_3():
    """L3: Observability - otel-collector."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        checks = await asyncio.gather(
            _check_url(client, "otel_collector", "http://otel-collector:13133"),
        )
    return {"level": "L3", "name": "observability", "checks": list(checks)}


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
