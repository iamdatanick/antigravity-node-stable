# Antigravity Node v13.0 — Complete Stack Description

## Identity

**Name:** Antigravity Node v13.0 ("The God Node")
**Repository:** `github.com/iamdatanick/Antigravity-Node` (private)
**Branch:** `master`
**Docker Hub Org:** `centillionai`
**License Governance:** Apache-2.0, MIT, BSD-3, MPL-2.0 only — no GPL/AGPL/SSPL/BSL

---

## Architecture Overview

A sovereign, multi-protocol AI agent node composed of **20 Docker containers** (18 long-running services + 2 init containers that exit after boot). Runs entirely via `docker compose` on a single Docker Desktop host. No Kubernetes required.

**Three protocol interfaces:**
1. **A2A (Agent-to-Agent)** — FastAPI HTTP on port 8080 (REST + WebSocket)
2. **MCP (Model Context Protocol)** — SSE transport on port 8000 per MCP server
3. **gRPC** — Intel SuperBuilder middleware on port 8081

**Single entry point:** Nginx reverse proxy on **port 1055** serving a React SPA with backend routing.

---

## Service Inventory (8 Layers)

### Layer 0: Infrastructure
| Service | Image | Port(s) | Purpose | Memory Limit |
|---------|-------|---------|---------|-------------|
| **postgres** | `postgres:16-alpine` | 5432 | Relational DB — hosts `antigravity`, `keycloak`, and `marquez` databases | 1G |
| **nats** | `nats:2.10-alpine` | 4222, 8222 | JetStream message bus for async event distribution | 256M |
| **seaweedfs** | `chrislusf/seaweedfs:3.59` | 8333 (S3), 9333 (master) | S3-compatible object storage for file uploads and context documents | 1G |
| **etcd** | `quay.io/coreos/etcd:v3.5.0` | 2379 | Distributed KV store used as Milvus metadata backend | 256M |

### Layer 1: Lineage + IAM
| Service | Image | Port(s) | Purpose | Memory Limit |
|---------|-------|---------|---------|-------------|
| **marquez** | `marquezproject/marquez:0.48.0` | 5000, 5001 | OpenLineage-compliant data lineage tracking | 512M |
| **keycloak** | `quay.io/keycloak/keycloak:26.0` | 8082→8080 | OIDC/OAuth2 identity and access management | 1G |

### Layer 2: Memory + Vector + Cache
| Service | Image | Port(s) | Purpose | Memory Limit |
|---------|-------|---------|---------|-------------|
| **starrocks** | `starrocks/allin1-ubuntu:latest` | 9030, 8030 | OLAP SQL engine hosting 3-layer memory: `memory_episodic`, `memory_semantic`, `memory_procedural` | 2G |
| **valkey** | `valkey/valkey:7.2-alpine` | 6379 | In-memory cache (Redis-compatible, BSD-3 licensed). 384MB max, LRU eviction | 512M |
| **milvus** | `milvusdb/milvus:v2.4.24` | 19530 (gRPC), 9091 (metrics) | Vector database for embedding similarity search | 2G |
| **openbao** | `openbao/openbao:2.0.0` | 8200 | Secrets management (MPL-2.0, replaces HashiCorp Vault BSL). Dev mode with root token | 256M |
| **ovms** | `openvino/model_server:2024.5` | 9000 (gRPC), 8500 (REST) | Intel OpenVINO Model Server for ML inference. Starts with empty config, watches `/models` for hot-reload | 2G |

### Layer 3: Sidecars
| Service | Image | Purpose | Memory Limit |
|---------|-------|---------|-------------|
| **wasm-worker** | `wasmedge/wasmedge:latest` | WebAssembly runtime placeholder for future edge compute plugins | 256M |

### Layer 4: Observability + Cost Control
| Service | Image | Port(s) | Purpose | Memory Limit |
|---------|-------|---------|---------|-------------|
| **perses** | `persesdev/perses:v0.47.1` | 3055→8080 | Apache-2.0 dashboards (replaces Grafana AGPL) | 256M |
| **budget-proxy** | Custom (`centillionai/budget-proxy:v13.0`) | 4055→4000 | LLM cost control proxy. Enforces $50/day hard budget cap. Routes to OpenAI, Anthropic, or local OVMS. Built on FastAPI | 256M |
| **opensearch** | `opensearchproject/opensearch:2.17.1` | 9200 | Apache-2.0 log aggregation and full-text search | 1G |
| **fluent-bit** | `fluent/fluent-bit:3.1` | — | Apache-2.0 log shipper. Receives on port 24224 (forward protocol), ships to OpenSearch index `antigravity-logs` | 128M |

### Layer 5: Agent Control Plane (MCP Servers)
| Service | Image | Purpose | Memory Limit |
|---------|-------|---------|-------------|
| **mcp-filesystem** | Custom (`centillionai/mcp-filesystem:v13.0`) | MCP server exposing read-only filesystem access to `/data` volume via SSE transport | 128M |
| **mcp-starrocks** | Custom (`centillionai/mcp-starrocks:v13.0`) | MCP server exposing StarRocks memory queries via SSE transport | 256M |

### Layer 6: Brain (Orchestrator)
| Service | Image | Port(s) | Purpose | Memory Limit |
|---------|-------|---------|---------|-------------|
| **orchestrator** | Custom (`centillionai/antigravity-brain:v13.0`) | 8080, 8081 | The brain. Dual-protocol entry point: FastAPI (A2A HTTP) + gRPC (Intel SuperBuilder). Runs "God Mode" background loop for health monitoring and context ingestion. Python 3.11-slim + Goose v1.20.1 CLI. Multi-arch: `linux/amd64` + `linux/arm64` | 1G |

### Layer 8: Portal (UI)
| Service | Image | Port(s) | Purpose | Memory Limit |
|---------|-------|---------|---------|-------------|
| **master-ui** | Custom (`centillionai/antigravity-ui:v13.0`) | 1055→80 | Nginx-served React 19 SPA. Reverse proxy routes `/api/*` to backend services. 9 pages: Dashboard, Chat, Logs, Memory, Query, Workflows, Budget, Services, Settings | 128M |

---

## Custom-Built Images (5 total, built via `docker buildx bake`)

| Target | Dockerfile | Tags | Platforms |
|--------|-----------|------|-----------|
| `orchestrator` | `./Dockerfile` | `centillionai/antigravity-brain:v13.0`, `:latest` | amd64 + arm64 |
| `mcp-starrocks` | `./src/mcp-starrocks/Dockerfile` | `centillionai/mcp-starrocks:v13.0`, `:latest` | amd64 |
| `mcp-filesystem` | `./src/mcp-filesystem/Dockerfile` | `centillionai/mcp-filesystem:v13.0`, `:latest` | amd64 |
| `master-ui` | `./src/master-ui/Dockerfile` | `centillionai/antigravity-ui:v13.0`, `:latest` | amd64 |
| `budget-proxy` | `./src/budget-proxy/Dockerfile` | `centillionai/budget-proxy:v13.0`, `:latest` | amd64 |

Build infra: Docker Build Cloud endpoint `centillionai/phucai`, GHA cache (`type=gha`), registry layer cache for orchestrator.

---

## Networking

- **Single bridge network:** `antigravity_mesh`
- All 20 containers communicate over this flat network using container names as DNS hostnames
- **Docker embedded DNS resolver** used by Nginx: `resolver 127.0.0.11 valid=5s ipv6=off`
- No host networking, no port conflicts between services

**Nginx reverse proxy routes (master-ui port 1055):**

| Frontend Path | Backend Target | Transport |
|--------------|----------------|-----------|
| `/api/ws/*` | `orchestrator:8080` | WebSocket (upgrade) |
| `/api/budget/*` | `budget-proxy:4000` | HTTP |
| `/api/lineage/*` | `marquez:5000` | HTTP |
| `/api/search/*` | `opensearch:9200` | HTTP |
| `/api/*` (all else) | `orchestrator:8080` | HTTP (SSE streaming supported) |
| `/*` (SPA) | Static files | nginx direct |

---

## Persistent Storage (7 named Docker volumes)

| Volume | Mounted By | Purpose |
|--------|-----------|---------|
| `postgres-data` | postgres | PostgreSQL data directory |
| `seaweedfs-data` | seaweedfs | Object storage blobs |
| `milvus-data` | milvus | Vector index data |
| `valkey-data` | valkey | Redis-compatible AOF persistence |
| `openbao-data` | openbao | Secrets vault data |
| `perses-data` | perses | Dashboard definitions |
| `opensearch-data` | opensearch | Log indices |

**Bind mounts:**
- `./config/postgres/init-databases.sh` → postgres (init script creates `keycloak` + `marquez` databases)
- `./config/fluent-bit.conf` → fluent-bit
- `./models` → ovms (read-only model directory)
- `${HOST_DATA_DIR:-./data}` → mcp-filesystem (read-only)

---

## Orchestrator (Brain) Detail

**Base:** Python 3.11-slim + Goose v1.20.1 (Block Inc agentic framework)

**Entry point:** `workflows/main.py` which starts 3 subsystems in parallel:
1. **OpenTelemetry** tracing initialization
2. **gRPC server** on port 8081 (Intel SuperBuilder protobuf service)
3. **God Mode loop** — background async health monitor that runs 50 iterations checking StarRocks, Milvus, Keycloak, OpenBao health + ingesting context files
4. **FastAPI server** on port 8080 (blocking, main thread)

**Python dependencies (key):**
- `fastapi>=0.109`, `uvicorn`, `httpx>=0.27`
- `nats-py>=2.7` (NATS JetStream)
- `asyncpg>=0.29` (PostgreSQL, Apache-2.0 — replaces psycopg2 LGPL)
- `pymilvus>=2.4` (vector search)
- `grpcio>=1.62` + `grpcio-tools` (gRPC stubs compiled at build time)
- `mcp>=1.0`, `fastmcp>=2.0` (Model Context Protocol)
- `openlineage-python>=1.9` (Marquez integration)
- `tenacity>=8.2` (retry/resilience)
- `slowapi>=0.1.9` (rate limiting)
- `python-jose[cryptography]>=3.3` (JWT validation for Keycloak tokens)
- `opentelemetry-*` (distributed tracing: FastAPI, gRPC, aiohttp instrumented)
- `boto3>=1.35` (S3 client for SeaweedFS)
- `pandas>=2.2` (data processing)

**A2A HTTP Endpoints (FastAPI on port 8080):**

| Method | Path | Purpose | Rate Limit | Auth |
|--------|------|---------|------------|------|
| GET | `/health` | 5-level health hierarchy (returns 200 or 503) | — | — |
| GET | `/.well-known/agent.json` | A2A agent descriptor | — | — |
| POST | `/task` | Submit a goal with multi-tenant isolation | 60/min | JWT + x-tenant-id |
| POST | `/handoff` | Agent-to-agent handoff | — | JWT |
| POST | `/upload` | File upload to SeaweedFS (100MB max) | 10/min | JWT + x-tenant-id |
| POST | `/webhook` | Argo exit-handler callback (HMAC-SHA256 verified) | — | Webhook signature |
| POST | `/v1/chat/completions` | OpenAI-compatible chat (routes through budget-proxy) | 30/min | JWT + x-tenant-id |
| GET | `/v1/models` | OpenAI-compatible model list | — | — |
| POST | `/v1/inference` | OVMS model inference (gRPC+REST fallback) | 120/min | JWT + x-tenant-id |
| GET | `/v1/models/ovms` | List OVMS-loaded models | — | — |
| GET | `/capabilities` | Full node capability discovery | — | — |
| GET | `/tools` | MCP tool discovery (probes SSE endpoints) | — | — |
| POST | `/query` | Read-only SQL against StarRocks (SELECT only, forbidden keyword blocklist) | 30/min | JWT + x-tenant-id |
| GET | `/workflows` | List Argo workflows for DAG visualization | 30/min | JWT |
| GET | `/memory` | Paginated episodic memory browser | 60/min | JWT + x-tenant-id |
| WS | `/ws/logs` | Real-time log streaming from OpenSearch | — | — |
| GET | `/budget/history` | 24-hour spend chart data | 30/min | JWT + x-tenant-id |

---

## Memory Architecture (3-Layer)

Stored in **StarRocks** OLAP engine with HASH distribution by `tenant_id`:

| Table | Schema | Purpose |
|-------|--------|---------|
| `memory_episodic` | `event_id BIGINT, tenant_id, timestamp, session_id, actor, action_type, content TEXT, embedding ARRAY<FLOAT>` | Event log — every task, chat, upload, webhook |
| `memory_semantic` | `doc_id, tenant_id, chunk_id INT, content TEXT, source_uri, embedding ARRAY<FLOAT>` | Document chunks for RAG retrieval |
| `memory_procedural` | `skill_id, description TEXT, argo_template_yaml TEXT, success_rate FLOAT, embedding ARRAY<FLOAT>` | Learned skills with Argo workflow templates |

**Milvus** stores the vector embeddings referenced by `embedding ARRAY<FLOAT>` columns for similarity search.

---

## Frontend (React SPA)

**Stack:** React 19, Vite 6, TypeScript 5.7, Tailwind CSS 4, TanStack Query 5
**Runtime:** Nginx Alpine serving static build, reverse-proxying API calls

**9 Pages:**
1. **Dashboard** — Health grid, service count, capabilities summary
2. **Chat** — OpenAI-compatible chat with streaming (SSE), markdown rendering (marked + highlight.js)
3. **Logs** — WebSocket-fed xterm.js terminal with ANSI color rendering
4. **Memory** — TanStack-powered paginated table with search over episodic memory
5. **Query** — CodeMirror 6 SQL editor with syntax highlighting, execute against StarRocks
6. **Workflows** — Cytoscape.js DAG visualizer for Argo workflow nodes
7. **Budget** — Recharts bar chart + progress bar showing daily LLM spend
8. **Services** — Full endpoint + MCP server discovery from `/capabilities`
9. **Settings** — Raw node config display from `/capabilities`

**API Client Pattern:**
- All API calls go through `/api/*` which Nginx routes to backend services
- `apiFetch()` wrapper with exponential backoff retry (2 retries, doubling delay)
- `acceptStatuses` option to treat HTTP 503 as valid data (health endpoint returns 503 when degraded)
- `streamChat()` async generator for SSE streaming chat responses
- Multi-tenant isolation via `X-Tenant-Id` header on every request

---

## CI/CD (GitHub Actions)

**Workflow 1: CI** (triggers on push to master + PRs)
- **lint** — Python: `ruff check .` + `ruff format --check .`
- **test** — Python: `pip install -r requirements.txt`, compile protobufs, `pytest tests/ -v` (218 tests)
- **frontend** — Node 22: `npm ci`, `tsc --noEmit`, `vite build`
- **security** — Trivy filesystem scan (CRITICAL + HIGH)

**Workflow 2: Docker Build Cloud** (triggers on push to master)
- Docker Hub login
- QEMU setup for arm64 cross-compilation
- Docker Build Cloud (endpoint: `centillionai/phucai`) with standard buildx fallback
- `docker/bake-action` builds all 5 targets, pushes to Docker Hub
- GHA layer caching (`type=gha, mode=max`)

---

## Security Posture

| Control | Implementation |
|---------|---------------|
| Container user | Non-root `appuser` in custom images |
| Privilege escalation | `no-new-privileges:true` on 12 containers |
| Read-only filesystems | 8 containers run `read_only: true` with tmpfs for `/tmp` |
| SQL injection prevention | Forbidden keyword blocklist + parameterized queries only |
| File upload validation | Path traversal prevention (`os.path.basename`), 100MB size limit |
| Webhook authentication | HMAC-SHA256 signature verification |
| Rate limiting | slowapi per-endpoint limits (10-120 req/min) |
| JWT validation | Keycloak OIDC tokens validated on all write endpoints |
| CORS | Configurable via `CORS_ORIGINS` env var |
| Secrets | OpenBao (dev mode) for runtime secrets |
| mTLS | SPIRE agent configured (not yet wired to all services) |
| HTTP security headers | X-Frame-Options, X-Content-Type-Options, X-XSS-Protection, Referrer-Policy |
| Trivy scanning | GitHub Actions security job on every push/PR |

---

## Resource Requirements

| Resource | Total Allocated |
|----------|----------------|
| **Memory** | ~14 GB across all containers |
| **CPUs** | ~13.5 cores allocated |
| **Disk (volumes)** | 7 named volumes — size depends on data ingestion |
| **Ports exposed** | 1055 (UI), 8080-8081 (orchestrator), 5432 (postgres), plus ~15 other service ports |

**Minimum recommended host:** 16 GB RAM, 4+ cores, 50 GB disk.

---

## Environment Variables (`.env` file)

| Variable | Default | Purpose |
|----------|---------|---------|
| `POSTGRES_USER` | `antigravity` | PostgreSQL superuser |
| `POSTGRES_PASSWORD` | `antigravity_secure` | PostgreSQL password |
| `POSTGRES_DB` | `antigravity` | Default database |
| `KEYCLOAK_ADMIN_PASSWORD` | `admin` | Keycloak admin console password |
| `OPENBAO_DEV_TOKEN` | `dev-only-token` | OpenBao root token (dev mode) |
| `OPENAI_API_KEY` | — | OpenAI API key for budget-proxy |
| `ANTHROPIC_API_KEY` | — | Anthropic API key for budget-proxy |
| `DAILY_BUDGET_USD` | `50` | Daily LLM spend cap |
| `HOST_DATA_DIR` | `./data` | Host path for MCP filesystem server |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `WEBHOOK_SECRET` | — | HMAC secret for Argo webhook verification |
| `GOD_MODE_ITERATIONS` | `50` | Number of health-check/ingest cycles |

---

## Dependency Topology (Boot Order)

```
postgres ─────┬──→ marquez
              ├──→ keycloak
              └──→ orchestrator
etcd ─────────→ milvus ──→ orchestrator
seaweedfs ────→ orchestrator
starrocks ────→ orchestrator, mcp-starrocks
valkey ───────→ orchestrator
openbao ──────→ orchestrator
opensearch ───→ fluent-bit
orchestrator ─→ master-ui
budget-proxy ─→ master-ui
perses ───────→ master-ui
```

Services with `condition: service_healthy` enforce strict health-check-based boot ordering. The orchestrator waits for 6 healthy dependencies before starting.

---

## Deployment Methods

1. **Local development:** `docker compose up -d` from repo root
2. **Local image build:** `docker buildx bake --load --set "*.platform=linux/amd64"`
3. **CI/CD production:** GitHub Actions → Docker Build Cloud → Docker Hub push → pull from any host
4. **Single-host production:** Pull images from `centillionai/*` on Docker Hub, `docker compose up -d`
