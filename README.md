# Antigravity Node

Antigravity Node v13 is a Python 3.11-based orchestration service (despite the name, it is not a Node.js project) that combines FastAPI A2A endpoints, a gRPC server, and a Docker Compose stack to coordinate workflows, storage, messaging, and UI tooling.

## Repository layout

- `workflows/`: main runtime (FastAPI + gRPC), health checks, Argo workflow definitions, and supporting clients.
- `src/mcp-starrocks/`: MCP tool server for StarRocks.
- `src/trace-viewer/`: Streamlit-based trace viewer service.
- `src/master-ui/`: static UI gateway that aggregates other UIs.
- `config/`: service configuration (LiteLLM, MCP catalog, prompts, SPIRE, Postgres init scripts).
- `docker-compose.yml`: full stack runtime and dependencies.
- `Dockerfile`: orchestrator container build.

## Prerequisites

- Docker + Docker Compose for the full stack.
- Python 3.11 for local development.

## Quick start (Docker Compose)

1. Create a `.env` file (gitignored) with any required API keys, such as `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.
2. Start the stack:
   ```bash
   docker compose up -d
   ```
3. Follow orchestrator logs:
   ```bash
   docker compose logs -f orchestrator
   ```

Key endpoints (after startup):

- Orchestrator HTTP: `http://localhost:8080/health`
- Orchestrator gRPC: `localhost:8081`
- Open WebUI: `http://localhost:3355`
- Grafana: `http://localhost:3055`
- Trace Viewer: `http://localhost:8655`
- Master UI: `http://localhost:1055`

## Local development

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   . .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Generate gRPC stubs when `.proto` files change:
   ```bash
   python -m grpc_tools.protoc \
     -I workflows/protos \
     --python_out=workflows \
     --grpc_python_out=workflows \
     workflows/protos/*.proto
   ```
3. Run the orchestrator:
   ```bash
   python workflows/main.py
   ```
4. Start supporting services with Docker Compose as needed.

## Configuration

- `.env`: runtime secrets for Docker Compose (API keys, model selection).
- `config/litellm/config.yaml`: LiteLLM provider routing.
- `config/mcp-catalog.yaml`: MCP tool catalog.
- `config/prompts/system.txt`: Goose system prompt.
- `config/postgres/init-databases.sh`: database initialization.

## Development workflow

- Update workflow logic in `workflows/` and restart the orchestrator.
- Rebuild containers with `docker compose build` when Dockerfiles change.
- Review `docker-compose.yml` for service ports, dependencies, and environment variables.

## Testing

Automated tests are minimal; the `tests/` directory currently provides shared fixtures without full test coverage.
If you add tests, run them with:

```bash
python -m pytest
```

## Linting

No linting configuration is defined yet.

## Contributing

Use feature branches, keep changes focused, and include any relevant test updates in pull requests.

## License

No license file is currently included in the repository. Confirm licensing expectations with the maintainers before redistribution.
