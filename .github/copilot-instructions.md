# Antigravity Node - Copilot Instructions

## Project Overview

Antigravity Node is a Python 3.11-based orchestration service that combines FastAPI A2A endpoints, a gRPC server, and a Docker Compose stack to coordinate workflows, storage, messaging, and UI tooling. Despite the name, this is **not a Node.js project** - it's Python-based.

## Repository Structure

- `workflows/`: Main runtime with FastAPI + gRPC servers, health checks, Argo workflow definitions, and supporting clients
- `src/mcp-starrocks/`: MCP tool server for StarRocks
- `src/trace-viewer/`: Streamlit-based trace viewer service
- `src/master-ui/`: Static UI gateway that aggregates other UIs
- `config/`: Service configuration (LiteLLM, MCP catalog, prompts, SPIRE, Postgres init scripts)
- `tests/`: Test fixtures and test files
- `docker-compose.yml`: Full stack runtime and dependencies
- `Dockerfile`: Orchestrator container build

## Technology Stack

- **Language**: Python 3.11
- **Web Framework**: FastAPI for A2A HTTP endpoints
- **RPC Framework**: gRPC (Intel SuperBuilder middleware)
- **Workflow Engine**: Argo Workflows with Hera SDK
- **Messaging**: NATS
- **Storage**: S3 (boto3), PostgreSQL, StarRocks
- **Vector Database**: Milvus
- **Cache**: Valkey
- **UI Frameworks**: Streamlit (trace viewer), static HTML (master UI)

## Coding Standards & Conventions

### Python Style

- Target Python 3.11 features and syntax
- Follow standard Python conventions (PEP 8)
- Use type hints where appropriate
- Prefer async/await patterns for I/O operations
- Use descriptive variable names

### gRPC Patterns

- gRPC servers use synchronous handlers with `asyncio.run()` to call async functions
- All gRPC servers should include `grpc_health.v1` health check service for monitoring
- Generate gRPC stubs when `.proto` files change using:
  ```bash
  python -m grpc_tools.protoc -I workflows/protos --python_out=workflows --grpc_python_out=workflows workflows/protos/*.proto
  ```

### Testing Practices

- Tests run with `python -m pytest`
- Use mocking to avoid external dependencies in tests
- Automated tests are minimal; the `tests/` directory provides shared fixtures
- Tests should be independent and not rely on external services

### Error Handling

- Use appropriate exception handling for I/O operations
- Log errors appropriately for debugging
- Include health check endpoints for all services

## Development Workflow

### Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   . .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create a `.env` file (gitignored) with required API keys:
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`

### Running Locally

- **Full stack**: `docker compose up -d`
- **Orchestrator only**: `python workflows/main.py`
- **View logs**: `docker compose logs -f orchestrator`

### Building & Testing

- **Run tests**: `python -m pytest`
- **Rebuild containers**: `docker compose build`
- **Generate gRPC stubs**: See gRPC Patterns section above

### Key Endpoints (after Docker Compose startup)

- Orchestrator HTTP: http://localhost:8080/health
- Orchestrator gRPC: localhost:8081
- Open WebUI: http://localhost:3355
- Perses (Dashboards): http://localhost:3055
- Trace Viewer: http://localhost:8655
- Master UI: http://localhost:1055

## Configuration Files

- `.env`: Runtime secrets for Docker Compose (API keys, model selection) - **gitignored**
- `config/litellm/config.yaml`: LiteLLM provider routing
- `config/mcp-catalog.yaml`: MCP tool catalog
- `config/prompts/system.txt`: Goose system prompt
- `config/postgres/init-databases.sh`: Database initialization

## Important Notes

### What NOT to Commit

- `.env` files (secrets)
- `__pycache__/` directories
- Data directories (`data/postgres/`, `data/starrocks/`, etc.)
- Test artifacts (`tests/syntax_*.txt`, `tests/git_*.txt`)
- IDE configuration (`.vscode/`, `.idea/`)
- Build artifacts (`dist/`, `build/`, `*.egg-info/`)

### Dependencies

- Review and use existing dependencies in `requirements.txt` before adding new ones
- Key dependencies: Hera, FastAPI, gRPC, NATS, boto3, psycopg2, pandas, pymilvus, valkey, MCP

### Security Considerations

- Never commit API keys or secrets to the repository
- Use environment variables for sensitive configuration
- Validate input to prevent injection attacks
- Use secure connections for external services

## Working with Specific Components

### FastAPI (A2A endpoints)

- Located in `workflows/a2a_server.py`
- Follow FastAPI patterns for route handlers
- Use async handlers where appropriate

### gRPC Server

- Located in `workflows/grpc_server.py`
- Synchronous handlers with `asyncio.run()` for async operations
- Include health check service

### Argo Workflows

- Workflow definitions in `workflows/workflow_defs.py`
- Use Hera SDK for defining workflows
- Keep workflow definitions declarative

### MCP Servers

- Tool servers in `src/mcp-*/` directories
- Follow MCP protocol standards
- Use FastMCP for simpler implementations

## Linting & Code Quality

Currently, no linting configuration is defined. When working on code quality improvements:
- Consider adding tools like `black`, `ruff`, or `pylint`
- Maintain consistency with existing code style
- Focus on readability and maintainability

## Contributing Guidelines

- Use feature branches for development
- Keep changes focused and well-scoped
- Update documentation when changing functionality
- Include relevant test updates in pull requests
- Ensure Docker Compose stack still works after changes

## Notes for Copilot Coding Agent

- This is a **Python project**, not Node.js, despite the repository name
- Always verify changes work with `python -m pytest` when tests exist
- Test Docker Compose integration when making infrastructure changes
- Maintain backwards compatibility unless explicitly requested to break it
- Focus on minimal, surgical changes rather than broad refactors
- Preserve existing working code and configurations
