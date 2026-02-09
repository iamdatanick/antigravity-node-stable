# Antigravity Node v13.0 "The God Node"

Antigravity Node is a production-grade **AI Orchestration Engine** designed to run autonomous agents with full observability, data lineage, and enterprise security. It combines a dual-protocol Python runtime (FastAPI + gRPC) with a comprehensive microservices mesh.

**Current Status:** Production Ready (Phases 1-4 Complete)

---

## 🚀 Key Capabilities

*   **Dual-Protocol Brain**:
    *   **FastAPI (A2A)**: HTTP endpoints for Agent-to-Agent communication.
    *   **gRPC (SuperBuilder)**: High-performance inter-process communication with protobuf contracts.
    *   **God Mode**: Background autonomous loop for health monitoring and context ingestion.
*   **Deep Memory Systems**:
    *   **StarRocks**: High-speed OLAP for episodic memory and analytics.
    *   **Milvus**: Vector database for semantic search and long-term recall.
    *   **SeaweedFS**: S3-compatible object storage for artifacts.
*   **Full Observability Stack**:
    *   **OpenLineage + Marquez**: Complete data lineage tracking.
    *   **OpenTelemetry**: Distributed tracing across all services.
    *   **Perses**: Visualization dashboards (Grafana alternative).
    *   **Trace Viewer**: Custom Streamlit app for inspecting agent thought processes.
*   **Enterprise Security**:
    *   **Keycloak**: Identity and Access Management (IAM).
    *   **OpenBao**: Secrets management (Vault fork).
    *   **Falco**: Runtime security monitoring.
    *   **Budget Proxy**: Cost control and rate limiting for LLM calls.

---

## 🛠️ Architecture Layers

The system is organized into functional layers defined in `docker-compose.yml`:

| Layer | Components | Description |
| :--- | :--- | :--- |
| **Layer 0: Core** | Postgres, SeaweedFS, NATS, Etcd | Foundation for storage and messaging. |
| **Layer 1: IAM & Lineage** | Keycloak, Marquez | Identity and data governance. |
| **Layer 1.5: Workflows** | K3D, Argo Workflows | Containerized workflow orchestration. |
| **Layer 2: Memory** | StarRocks, Valkey, Milvus, OpenBao | Hot/Cold storage and vector search. |
| **Layer 3: Sidecars** | WasmEdge, Falco | Runtime extensions and security. |
| **Layer 4: Observability** | Perses, Budget Proxy, OpenSearch | Monitoring, logs, and cost tracking. |
| **Layer 5: Control Plane** | MCP Gateway, MCP Servers | Model Context Protocol (MCP) integration. |
| **Layer 6: The Brain** | **Orchestrator** | Main Python application (FastAPI + gRPC). |
| **Layer 7: Interfaces** | LibreChat, Master UI, Trace Viewer | Human-in-the-loop UIs. |

---

## ⚡ Quick Start

### Prerequisites
*   **Docker Desktop** (with Kubernetes disabled, as K3D is used internally).
*   **Python 3.11** (for local development).
*   **Git**.

### 1. Configure Environment
Create a `.env` file in the root directory (copy from example if available, or set these minimums):
```bash
# API Keys (Required for LLM features)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...

# Security (Change these for production!)
POSTGRES_PASSWORD=secret
KEYCLOAK_ADMIN_PASSWORD=admin
```

### 2. Launch the Stack
```bash
docker compose up -d
```
*Note: The first launch performs heavy initialization (Postgres, Keycloak, StarRocks). Wait ~2-3 minutes for all health checks to pass.*

### 3. Access Interfaces
| Service | URL | Credentials (Default) |
| :--- | :--- | :--- |
| **Master UI** (Gateway) | http://localhost:1055 | N/A |
| **Orchestrator Health** | http://localhost:8080/health | N/A |
| **LibreChat** | http://localhost:3355 | Create account |
| **Trace Viewer** | http://localhost:8655 | N/A |
| **Marquez** (Lineage) | http://localhost:5055 | N/A |
| **Perses** (Dashboards) | http://localhost:3055 | N/A |

---

## 💻 Local Development

### Python Setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### gRPC Compilation
If you modify `workflows/protos/*.proto`, regenerate the Python stubs:
```bash
python -m grpc_tools.protoc \
    -I workflows/protos \
    --python_out=workflows \
    --grpc_python_out=workflows \
    workflows/protos/*.proto
```

### Running Tests
The project includes a comprehensive test suite (Unit + Integration):
```bash
# Run all tests
python -m pytest

# Run specific category
python -m pytest tests/test_grpc_server.py
```

---

## 📦 CI/CD

The repository uses **GitHub Actions** for continuous integration:
1.  **Linting**: Ruff formatting and checks.
2.  **Testing**: Pytest with gRPC stub compilation.
3.  **Security**: Trivy vulnerability scanning.
4.  **Build**: Docker Build Cloud integration.

---

## 🤝 Contributing

1.  Create a feature branch (e.g., `feature/new-agent-tool`).
2.  Add tests for your changes.
3.  Ensure CI passes.
4.  Submit a Pull Request.

**License**: Proprietary / Private (See repository owner for details).
