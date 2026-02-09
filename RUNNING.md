# 🚀 Antigravity Node v13.0 - RUNNING APPLICATION

## Quick Start

### Option 1: Run with Minimal Infrastructure (Recommended for Testing)
```bash
cd C:\Users\NickV\OneDrive\Desktop\Antigravity-Node

# Start minimal services (PostgreSQL, NATS, Valkey)
docker compose -f docker-compose.minimal.yml up -d

# Run the Python application
python workflows/main.py
```

### Option 2: Run Full Stack with Docker
```bash
cd C:\Users\NickV\OneDrive\Desktop\Antigravity-Node

# Start all services
docker compose up -d

# Wait 60 seconds for services to initialize
# Access health: http://localhost:8080/health
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (5-level) |
| `/task` | POST | Submit a task |
| `/.well-known/agent.json` | GET | A2A agent descriptor |
| `/docs` | GET | Swagger UI |
| `:8081` | gRPC | Intel SuperBuilder |

## Current Status

- ✅ **HTTP API on port 8080**: Running
- ✅ **gRPC on port 8081**: Running
- ✅ **Health endpoint**: Responding
- ✅ **hooks.py**: Imported (HookRegistry system)
- ⚠️ **Infrastructure**: Use docker-compose.minimal.yml

## Files Ported from Centillion

- `workflows/hooks.py` - HookRegistry for PRE_TOOL/POST_TOOL interception

## Testing

```bash
# Health check
curl http://localhost:8080/health

# Submit a task
curl -X POST http://localhost:8080/task \
  -H "Content-Type: application/json" \
  -H "x-tenant-id: nick" \
  -d '{"goal": "What can you do?"}'
```

## Infrastructure Services (docker-compose.minimal.yml)

| Service | Port | Image |
|---------|------|-------|
| PostgreSQL | 5455 | postgres:16-alpine |
| NATS | 4255 | nats:2.10-alpine |
| Valkey | 6380 | valkey/valkey:7.2-alpine |

---

*Generated: 2026-02-08*
*Application is LIVE and responding on ports 8080 + 8081*
