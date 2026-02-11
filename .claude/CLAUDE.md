# Antigravity Node v14.1 Phoenix

## Architecture (Cloud)

10 containers, 4 layers:
- **L0 Infra:** etcd v3.5.17, Ceph v18, OpenBao 2.1.0, OTel Collector
- **L2 Inference:** OVMS 2025.4 (REST port 9001, AVX-512)
- **L3 Logic:** Orchestrator (AsyncDAGEngine), Budget Proxy
- **L4 Interface:** UI (React 19 + Nginx)

## Key Paths

| What | Path |
|------|------|
| Cloud compose | deployment/cloud-test/docker-compose.yml |
| Cloud scripts | deployment/cloud-test/scripts/ |
| Terraform | deployment/terraform/ |
| Engine | src/orchestrator/engine.py |
| Entry point | workflows/main.py |
| Health checks | workflows/health.py |
| S3 client | workflows/s3_client.py |
| Inference | workflows/inference.py |
| Phoenix spec | docs/AntigravityNode_v14.1_Phoenix_Spec.xlsx |
| Plan | docs/plans/2026-02-10-v14-phoenix-gcp-deploy.md |

## Validation Gates

Run `/deploy-validate` before any push. Gates:
1. pytest test_validation_gates.py (VG-101 to VG-109)
2. Compose YAML syntax
3. Terraform validate
4. Ruff lint
5. Unit tests
6. Secrets scan
7. Truncation scan

## Docker Compose Rules

- ALL critical services: healthcheck blocks required
- ALL services: `restart: unless-stopped`
- Orchestrator: `depends_on` all 4 infra with `condition: service_healthy`
- Pin every image version (no `:latest`)
- Ceph: MUST have BOTH `./data/ceph:/var/lib/ceph` AND `./data/ceph_conf:/etc/ceph`
- Etcd: MUST have `--data-dir=/etcd-data` AND `./data/etcd:/etcd-data`

## Deployment Order

Phase 0: check_avx512 > fix_perms > pip install > init_models
Phase 1: docker compose up (etcd, ceph, openbao) > setup_tenants
Phase 2: docker compose up (all)
Phase 3: curl health check
Phase 4: E2E Chat UI

Never skip phases. Never reorder.

## License

```
ALLOWED: Apache-2.0, MIT, BSD-3, MPL-2.0
BANNED:  GPL, AGPL, SSPL, BSL, LGPL, custom, VC-backed critical path
```

## Do NOT

- Use v13 services: SeaweedFS, Milvus, StarRocks, Postgres, NATS, Ollama, Keycloak, Marquez
- Reference port 8500 or 8000 for OVMS (v14.1 = REST 9001)
- Add asyncpg, pymilvus, nats-py to cloud requirements
- Skip healthchecks on any service
- Run `docker compose up` without Phase 1 infra first
- Push without `/deploy-validate`
