# Antigravity Node v14.1 Phoenix - Copilot Instructions

## Project Overview

Antigravity Node is a Python 3.11-based orchestration service that combines FastAPI A2A endpoints, a gRPC server, and a Docker Compose stack for multi-tenant AI workflow orchestration. 

## Architecture (v14.1 Phoenix)

- **Infrastructure:** etcd v3.5.17, Ceph RGW (S3), OpenBao 2.1.0
- **Observability:** OpenTelemetry Collector
- **Inference:** OVMS 2025.4 (REST port 9001)
- **Logic:** Orchestrator (FastAPI), Budget Proxy
- **Interface:** React 19 UI (port 1055)

## Tech Stack

- **Backend**: Python 3.11, FastAPI, gRPC
- **Frontend**: React 19, TypeScript, Vite 6, TailwindCSS v4
- **Persistence**: etcd (state), Ceph (objects), OpenBao (secrets)
- **Inference**: OpenVINO Model Server

## Development

- **Full stack**: `docker compose up -d`
- **Linting**: `ruff check .`, `ruff format .`
- **Testing**: `pytest tests/`
