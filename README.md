# Antigravity Node v14.1 "Phoenix"

Production-grade **AI Orchestration Engine** — 21-container Docker Compose stack with dual-protocol runtime (FastAPI + gRPC), circuit breakers, kill switch, and full observability.

## Quick Start

**Prerequisites:** Docker Desktop (running), Git, PowerShell 5.1+

```powershell
# Clone and install
git clone https://github.com/iamdatanick/antigravity-node-stable.git
cd antigravity-node-stable
.\install.ps1

# Or manually:
copy .env.example .env   # Edit with your API keys
docker compose up -d
```

Wait ~2-3 minutes for all health checks to pass, then:

```powershell
curl http://localhost:8080/health
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| **Master UI** | [localhost:1055](http://localhost:1055) | React 19 portal |
| **Orchestrator** | [localhost:8080](http://localhost:8080/health) | FastAPI A2A + health |
| **gRPC** | localhost:8081 | Intel SuperBuilder |
| **Budget Proxy** | localhost:4055 | Cost control + model routing |
| **Perses** | [localhost:3055](http://localhost:3055) | Dashboards |
| **Keycloak** | [localhost:8082](http://localhost:8082) | IAM |
| **OpenBao** | localhost:8200 | Secrets (Vault fork) |
| **SeaweedFS** | localhost:8333/9333 | S3-compatible storage |
| **StarRocks** | localhost:9030 | OLAP memory |
| **Milvus** | localhost:19530 | Vector DB |
| **OpenSearch** | localhost:9200 | Log aggregation |
| **Marquez** | localhost:5000 | OpenLineage |
| **OVMS** | localhost:8500 | OpenVINO inference |
| **Ollama** | localhost:11434 | Local LLM (tinyllama) |

## API Endpoints

```
GET  /health                    — Stack health + circuit breakers
GET  /capabilities              — Node capabilities
GET  /tools                     — MCP tool registry
POST /task                      — Submit task (x-tenant-id header required)
POST /v1/chat/completions       — OpenAI-compatible chat proxy
POST /upload                    — File upload to S3
POST /webhook                   — Workflow status callback
POST /handoff                   — Agent-to-agent handoff
GET  /.well-known/agent.json    — A2A agent descriptor
GET  /v1/models                 — Available models
GET  /admin/circuits            — Circuit breaker status
POST /admin/kill-switch         — Emergency stop
WS   /ws/logs                   — Real-time container logs
```

## Architecture

| Layer | Services |
|-------|----------|
| **L0: Infrastructure** | Postgres 16, SeaweedFS 3.59, NATS 2.10, etcd 3.5 |
| **L1: IAM + Lineage** | Keycloak 26.0, Marquez 0.48 |
| **L2: Memory + Vector** | StarRocks, Valkey 7.2, Milvus 2.4, OpenBao 2.0 |
| **L3: Inference** | OVMS 2024.5, Ollama, WasmEdge |
| **L4: Observability** | Perses 0.47, Budget Proxy, OpenSearch 2.17, Fluent Bit 3.1 |
| **L5: Control Plane** | MCP Filesystem, MCP StarRocks |
| **L6: Brain** | Orchestrator (FastAPI + gRPC + God Mode) |
| **L8: Portal** | Master UI (React 19 + Nginx) |

## Resilience

Built on [agentic-workflows](https://github.com/iamdatanick/agentic-workflows) v5.0:

- **Circuit Breakers** — OVMS, Ollama, S3 (opens after 5 failures, half-open at 30s)
- **Rate Limiter** — Token bucket (10 req/s, burst 100)
- **Kill Switch** — Emergency stop via `/admin/kill-switch`
- **Retrier** — Exponential backoff with jitter

## Cloudflare Tunnel (Optional)

```powershell
# Set token in .env
$env:CLOUDFLARE_TUNNEL_TOKEN = "your-token"

# Start tunnel
docker compose --profile tunnel up -d cloudflare-tunnel
```

## Development

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests/test_a2a_server.py -v
```

## License

Apache-2.0 / MIT / BSD-3 / MPL-2.0 components only. No GPL/AGPL/SSPL/BSL.
