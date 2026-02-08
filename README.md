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

## Documentation

- [User Manual](docs/USER_MANUAL.md): full installation, UI walkthroughs, and troubleshooting.

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
- Perses (Dashboards): `http://localhost:3055`
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

## Task Plan

**Week 1:** PR #1 (critical bugs) → PR #2 (gRPC fix)  
**Week 2:** PR #3 (Pydantic) → PR #4 (CORS/auth/rate limit)  
**Week 3:** PR #5 (connection pooling) → PR #6 (Docker hardening)  
**Week 4:** PR #7 (OpenTelemetry) → PR #8 (OpenLineage)  
**Week 5:** PR #9 (Keycloak JWT) → PR #10 (CI pipeline)  
**Week 6:** PR #11 (tests) → PR #12 (HTMX Master UI upgrade)  
**Week 7:** PR #13 (WebSocket logs) → PR #14 (Chart.js dashboard)  
**Week 8:** PR #15 (Xterm.js) → PR #16 (Memory browser)  
**Week 9:** PR #17 (GridStack) → PR #18 (Monaco editor)  
**Week 10:** PR #19 (Cytoscape workflow viz)

### Antigravity Node v13.0 — Task Plan

#### Execution Phases & PR Breakdown

##### Phase 1: Critical Bug Fixes (P0)

**PR #1 — Fix critical bugs: duplicate templates key, thread-safe counter, SQL injection**

Scope:
- workflows/workflow_defs.py — Merge the duplicate templates key into a single list
- workflows/memory.py — Confirm global _event_counter uses itertools.count() (or equivalent) for thread-safe event IDs
- workflows/memory.py — Confirm query() enforces SELECT-only semantics via a SQL keyword allow-list
- src/trace-viewer/trace_viewer.py — Confirm all SQL queries use parameterized execution (no f-string interpolation of user input)
- Verify unused imports (time from main.py, json from a2a_server.py, subprocess from goose_client.py) remain removed
- Verify unused variables (MAX_RETRIES, BASE_DELAY, MAX_DELAY in main.py) remain removed; tests/test_critical_fixes.py should continue to enforce this

**PR #2 — Fix gRPC server: handler wiring, servicer registration, and proto responses**

Scope:
- workflows/grpc_server.py — Keep the synchronous grpc.server(...) with asyncio.run(...) bridge and ensure ExecuteWorkflow correctly invokes async workflow logic and handles errors
- Generate/import gRPC proto stubs (from workflows/protos/*.proto) and register the workflow servicer implementation with the server so ExecuteWorkflow returns real proto responses instead of placeholders
- Confirm grpc_health.v1 health checking is wired correctly and, optionally, add server reflection so gRPC clients can discover available RPCs
- If proto files aren’t ready, add a clear TODO and make serve_grpc() return immediately without starting a listener so the orchestrator can run without gRPC

##### Phase 2: API Hardening & Validation (P1)

**PR #3 — Add Pydantic request/response models to all FastAPI endpoints**

Scope:
- Audit and extend workflows/models.py so all FastAPI request/response bodies use concrete Pydantic models (e.g., TaskRequest, TaskResponse, HandoffRequest, WebhookPayload, ChatCompletionRequest, etc.)
- Replace any remaining body: dict parameters in a2a_server.py (and related modules) with typed Pydantic models
- Ensure all FastAPI endpoint decorators specify appropriate response_model entries for accurate OpenAPI documentation
- Tighten input validation (e.g., goal max length, allowed model enum values, temperature bounds) across all relevant models and endpoints

**PR #4 — Add CORS middleware + webhook authentication + path cleanup**

Scope:
- Add CORSMiddleware to FastAPI app in a2a_server.py
- Add HMAC signature or bearer token validation to /webhook endpoint
- Fix SYSTEM_PROMPT_PATH default to remove .. path traversal
- Add rate limiting middleware (e.g., slowapi) to /v1/chat/completions, /upload, and /task

##### Phase 3: Infrastructure Reliability (P1-P2)

**PR #5 — Add connection pooling for StarRocks + S3 client singleton**

Scope:
- workflows/memory.py — Replace per-call pymysql.connect() with a connection pool (SQLAlchemy pool or DBUtils.PooledDB)
- workflows/s3_client.py — Create a singleton boto3 client instead of per-call instantiation
- workflows/s3_client.py — Fix ensure_bucket to catch botocore.exceptions.ClientError specifically instead of bare Exception
- src/trace-viewer/trace_viewer.py — Add connection retry/reconnect logic to handle StarRocks restarts (replace stale @st.cache_resource)

**PR #6 — Docker Compose security hardening**

Scope:
- Move hardcoded POSTGRES_USER / POSTGRES_PASSWORD to .env file references with ${VAR} syntax
- Add .env.example with placeholder values for all required secrets
- Add docker-compose.override.yml.example for dev-specific settings
- Add health check start_period to services that take longer to boot
- Add read_only: true and security_opt: [no-new-privileges:true] to containers where possible

##### Phase 4: Observability & Tracing (P2)

**PR #7 — Add OpenTelemetry instrumentation (CNCF)**

Scope:
- Add opentelemetry-api, opentelemetry-sdk, opentelemetry-instrumentation-fastapi, opentelemetry-instrumentation-grpc, opentelemetry-exporter-otlp to requirements.txt
- Initialize OTel tracer provider in main.py
- Instrument FastAPI app with FastAPIInstrumentor
- Add span creation in memory.py, s3_client.py, goose_client.py for cross-service tracing
- Add OTLP exporter config pointing to Jaeger (add to Docker Compose if needed)

**PR #8 — Integrate OpenLineage with Marquez for data lineage**

Scope:
- Add actual OpenLineage emit calls in workflows/workflow_defs.py on workflow submission
- Add lineage events in memory.py for episodic/semantic writes
- Add lineage events in s3_client.py for file uploads
- Configure Marquez connection via environment variables
- Wire up the existing openlineage-python dependency that’s in requirements.txt but unused

##### Phase 5: Auth & Security (P2)

**PR #9 — Integrate Keycloak JWT validation on API endpoints**

Scope:
- Create workflows/auth.py with JWT validation middleware using python-jose or PyJWT
- Add a FastAPI dependency that validates bearer tokens against Keycloak’s JWKS endpoint
- Apply auth dependency to /task, /handoff, /upload, /v1/chat/completions
- Keep /health, /.well-known/agent.json, and /capabilities unauthenticated
- Add python-jose[cryptography] to requirements.txt

##### Phase 6: Testing & CI (P2-P3)

**PR #10 — Add GitHub Actions CI pipeline (lint + test)**

Scope:
- Create .github/workflows/ci.yml with:
  - Python 3.11 setup
  - pip install -r requirements.txt
  - Ruff or Flake8 linting
  - pytest test execution
- Add pyproject.toml or setup.cfg with linting configuration (Ruff recommended)
- Add mypy type checking configuration
- Add Trivy container scanning step for the Dockerfile

**PR #11 — Add unit and integration tests**

Scope:
- Create tests/test_a2a_server.py — test all FastAPI endpoints with httpx.AsyncClient / TestClient
- Create tests/test_memory.py — test push_episodic, recall_experience with mocked PyMySQL
- Create tests/test_s3_client.py — test upload/download with mocked boto3
- Create tests/test_goose_client.py — test tool dispatch with mocked backends
- Create tests/test_workflow_defs.py — test workflow manifest generation
- Create tests/test_health.py — test health check hierarchy with mocked services
- Wire up existing conftest.py fixtures to the new test files
- Add pytest-asyncio, pytest-cov, respx (httpx mocking) to test dependencies

##### Phase 7: Front-End — Immediate Upgrade (P2)

**PR #12 — Upgrade Master UI with HTMX + Alpine.js + Pico CSS (zero build step)**

Scope:
- Replace inline assets with HTMX, Alpine.js, and Pico CSS references for a zero-build UI upgrade

## Contributing

Use feature branches, keep changes focused, and include any relevant test updates in pull requests.

## License

No license file is currently included in the repository. Confirm licensing expectations with the maintainers before redistribution.
