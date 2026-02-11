# v14.1 Phoenix Closeout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all remaining gaps from the v14.1 Phoenix deployment — broken tests, missing API endpoint, dead v13 nginx routes, and lint — then create the PR to merge `feature/v14-phoenix` into `master`.

**Architecture:** Six surgical fixes: (1) align validation gate tests with actual compose state, (2) add `/budget/history` endpoint to budget-proxy so the UI budget HUD works, (3) strip dead v13 proxy routes from nginx, (4) run lint + fix, (5) rebuild + push updated images to GAR, (6) create PR.

**Tech Stack:** Python 3.11/FastAPI (budget-proxy), nginx, Docker Compose, Google Artifact Registry, pytest, ruff

**Branch:** `feature/v14-phoenix` (29 commits ahead of `master`)

---

### Task 1: Fix validation gate tests to match actual compose

**Files:**
- Modify: `tests/test_validation_gates.py:76-83` (restart policy test)
- Modify: `tests/test_validation_gates.py:128-141` (depends_on test)

**Context:** Two tests will fail against the current `docker-compose.yml`:
1. `test_orchestrator_depends_on_healthy_infra` checks for `ceph-demo` in `depends_on`, but we intentionally removed it (Ceph is optional — graceful degradation).
2. `test_all_services_restart_policy` currently passes (all services have `restart: unless-stopped`), but the original plan had a UI exception that was never applied. No change needed here.

The depends_on test must be updated to reflect the design decision: orchestrator depends on `etcd`, `ovms`, `openbao` only — NOT `ceph-demo`.

**Step 1: Update the depends_on test**

In `tests/test_validation_gates.py`, change `TestComposeDependsOn`:

```python
class TestComposeDependsOn:
    """Boot order: orchestrator waits for critical infra to be healthy."""

    def test_orchestrator_depends_on_healthy_infra(self):
        compose_path = os.path.join(CLOUD_DIR, "docker-compose.yml")
        with open(compose_path) as f:
            compose = yaml.safe_load(f)
        deps = compose["services"]["orchestrator"]["depends_on"]
        # Ceph is intentionally excluded — graceful degradation (ACT-110)
        for svc in ["etcd", "ovms", "openbao"]:
            assert svc in deps, f"Orchestrator must depend on {svc}"
            assert deps[svc]["condition"] == "service_healthy", (
                f"Orchestrator must wait for {svc} to be healthy"
            )
        # Verify ceph-demo is NOT a hard dependency
        assert "ceph-demo" not in deps, (
            "ceph-demo must NOT be in depends_on (graceful degradation)"
        )
```

**Step 2: Run tests to verify they pass**

Run (from project root):
```
pytest tests/test_validation_gates.py -v
```
Expected: ALL PASS (10 tests)

**Step 3: Commit**

```bash
git -C "C:\Users\NickV\OneDrive\Desktop\Antigravity-Node" add tests/test_validation_gates.py
git -C "C:\Users\NickV\OneDrive\Desktop\Antigravity-Node" commit -m "fix: align validation gate tests with graceful Ceph degradation"
```

---

### Task 2: Add `/budget/history` endpoint to budget-proxy

**Files:**
- Modify: `src/budget-proxy/proxy.py`
- Create: `tests/test_budget_proxy.py`

**Context:** The UI's `useBudget()` hook (in `src/master-ui/src/api/budget.ts`) calls `GET /budget/history` first, falling back to `GET /budget/health`. The budget-proxy only has `/health` and `/v1/chat/completions`. The UI gets a 404 on `/budget/history`, then falls back to `/health` — which works, but the `/budget/history` endpoint returns richer data (hourly spend breakdown).

The nginx config routes `/api/budget/*` to `budget-proxy:4055` with the `/api/budget/` prefix stripped. So the budget-proxy receives `GET /budget/history` as-is.

**Step 1: Write the failing test**

Create `tests/test_budget_proxy.py`:

```python
"""Tests for budget-proxy API endpoints."""
import importlib
import os
import sys
from unittest.mock import patch

import pytest


@pytest.fixture
def budget_app():
    """Import and return the budget-proxy FastAPI app."""
    # Ensure src/budget-proxy is importable
    proxy_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "src", "budget-proxy",
    )
    if proxy_dir not in sys.path:
        sys.path.insert(0, proxy_dir)
    import proxy
    importlib.reload(proxy)
    return proxy.app


class TestBudgetHistory:
    """GET /budget/history must return hourly spend data."""

    def test_budget_history_returns_200(self, budget_app):
        from fastapi.testclient import TestClient
        client = TestClient(budget_app)
        resp = client.get("/budget/history")
        assert resp.status_code == 200

    def test_budget_history_has_required_fields(self, budget_app):
        from fastapi.testclient import TestClient
        client = TestClient(budget_app)
        data = client.get("/budget/history").json()
        assert "current_spend" in data
        assert "max_daily" in data
        assert "currency" in data
        assert "hourly_spend" in data
        assert isinstance(data["hourly_spend"], list)
        assert len(data["hourly_spend"]) == 24

    def test_budget_history_spend_matches_health(self, budget_app):
        from fastapi.testclient import TestClient
        client = TestClient(budget_app)
        history = client.get("/budget/history").json()
        health = client.get("/health").json()
        assert history["current_spend"] == health["daily_spend_usd"]
        assert history["max_daily"] == health["daily_budget_usd"]


class TestHealthEndpoint:
    """GET /health must return budget status."""

    def test_health_returns_ok(self, budget_app):
        from fastapi.testclient import TestClient
        client = TestClient(budget_app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "daily_spend_usd" in data
        assert "daily_budget_usd" in data
        assert "remaining_usd" in data
```

**Step 2: Run test to verify it fails**

Run:
```
pytest tests/test_budget_proxy.py::TestBudgetHistory -v
```
Expected: FAIL — `/budget/history` returns 404 (endpoint doesn't exist yet)

**Step 3: Add the endpoint to proxy.py**

In `src/budget-proxy/proxy.py`, add after the `/health` endpoint (after line 125):

```python
# Hourly spend tracking (24-hour rolling window)
_hourly_spend: list[float] = [0.0] * 24
_current_hour: int = datetime.now(UTC).hour


def _update_hourly(cost: float):
    """Track spend per hour for the budget history chart."""
    global _current_hour, _hourly_spend
    now_hour = datetime.now(UTC).hour
    if now_hour != _current_hour:
        # Zero out hours we skipped (handles gaps)
        if now_hour > _current_hour:
            for h in range(_current_hour + 1, now_hour + 1):
                _hourly_spend[h] = 0.0
        else:
            # Day wrapped
            for h in range(_current_hour + 1, 24):
                _hourly_spend[h] = 0.0
            for h in range(0, now_hour + 1):
                _hourly_spend[h] = 0.0
        _current_hour = now_hour
    _hourly_spend[now_hour] += cost


@app.get("/budget/history")
async def budget_history():
    """Return hourly spend breakdown for the UI budget chart."""
    _reset_if_new_day()
    return {
        "current_spend": round(_daily_spend, 4),
        "max_daily": DAILY_BUDGET_USD,
        "currency": "USD",
        "hourly_spend": [round(h, 6) for h in _hourly_spend],
    }
```

Also update the `chat_completions` function to call `_update_hourly(cost)` after adding to `_daily_spend`. Add this line inside the `async with _spend_lock:` block at the end of `chat_completions`, right after `_daily_spend += cost`:

```python
    async with _spend_lock:
        _daily_spend += cost
        _update_hourly(cost)
        current_spend = _daily_spend
```

**Step 4: Run tests to verify they pass**

Run:
```
pytest tests/test_budget_proxy.py -v
```
Expected: ALL PASS (5 tests)

**Step 5: Commit**

```bash
git -C "C:\Users\NickV\OneDrive\Desktop\Antigravity-Node" add src/budget-proxy/proxy.py tests/test_budget_proxy.py
git -C "C:\Users\NickV\OneDrive\Desktop\Antigravity-Node" commit -m "feat: add /budget/history endpoint with hourly spend tracking"
```

---

### Task 3: Remove dead v13 nginx proxy routes

**Files:**
- Modify: `src/master-ui/nginx.conf:65-83`

**Context:** nginx still routes `/api/lineage/` to `marquez:5000` and `/api/search/` to `opensearch:9200`. Neither Marquez nor OpenSearch exist in v14.1. These routes cause silent 502 errors and log noise. Remove them.

**Step 1: Remove the dead location blocks**

Delete lines 65-83 from `src/master-ui/nginx.conf` (the `location /api/lineage/` and `location /api/search/` blocks):

```nginx
        # Lineage (Marquez)              ← DELETE THIS BLOCK
        location /api/lineage/ {          ← DELETE
            ...                           ← DELETE
        }                                 ← DELETE

        # Search (OpenSearch)             ← DELETE THIS BLOCK
        location /api/search/ {           ← DELETE
            ...                           ← DELETE
        }                                 ← DELETE
```

The remaining routes are:
- `/` — static SPA
- `/api/ws/` — WebSocket to orchestrator
- `/api/budget/` — budget-proxy
- `/api/` — catch-all to orchestrator

**Step 2: Verify nginx config syntax**

Run:
```
docker run --rm -v "C:\Users\NickV\OneDrive\Desktop\Antigravity-Node\src\master-ui\nginx.conf:/etc/nginx/nginx.conf:ro" nginx:alpine nginx -t
```
Expected: `nginx: the configuration file /etc/nginx/nginx.conf syntax is ok`

**Step 3: Commit**

```bash
git -C "C:\Users\NickV\OneDrive\Desktop\Antigravity-Node" add src/master-ui/nginx.conf
git -C "C:\Users\NickV\OneDrive\Desktop\Antigravity-Node" commit -m "fix: remove dead v13 Marquez and OpenSearch nginx routes"
```

---

### Task 4: Run lint and fix violations

**Files:**
- Modify: any Python files with lint issues

**Step 1: Run ruff check**

Run (from project root):
```
ruff check src/budget-proxy/proxy.py workflows/ src/orchestrator/ tests/test_budget_proxy.py tests/test_validation_gates.py
```
Expected: Either clean or fixable violations

**Step 2: Auto-fix if needed**

Run:
```
ruff check --fix src/budget-proxy/proxy.py workflows/ src/orchestrator/ tests/test_budget_proxy.py tests/test_validation_gates.py
```

**Step 3: Run format**

Run:
```
ruff format src/budget-proxy/proxy.py workflows/ src/orchestrator/ tests/test_budget_proxy.py tests/test_validation_gates.py
```

**Step 4: Run full test suite**

Run:
```
pytest tests/test_validation_gates.py tests/test_budget_proxy.py tests/test_cloud_scripts.py tests/test_engine.py -v
```
Expected: ALL PASS

**Step 5: Commit (if changes)**

```bash
git -C "C:\Users\NickV\OneDrive\Desktop\Antigravity-Node" add -p
git -C "C:\Users\NickV\OneDrive\Desktop\Antigravity-Node" commit -m "style: lint and format fixes"
```

---

### Task 5: Rebuild and push updated images to GAR

**Files:** None (Docker operations)

**Context:** Tasks 2 and 3 modified `proxy.py` (budget-proxy) and `nginx.conf` (UI). The orchestrator image is unchanged. We need to rebuild these 2 images and push to GAR.

**Step 1: Rebuild budget-proxy and UI**

Run (from project root):
```
docker build -t us-central1-docker.pkg.dev/agentic1111/antigravity/budget-proxy:v14.1 -f src/budget-proxy/Dockerfile .
docker build -t us-central1-docker.pkg.dev/agentic1111/antigravity/ui:v14.1 -f src/master-ui/Dockerfile .
```

**Step 2: Push to GAR**

Run:
```
docker push us-central1-docker.pkg.dev/agentic1111/antigravity/budget-proxy:v14.1
docker push us-central1-docker.pkg.dev/agentic1111/antigravity/ui:v14.1
```

**Step 3: Deploy to VM**

Run:
```
gcloud compute ssh antigravity-v14-pilot --zone=us-central1-a --project=agentic1111 --command="bash /home/NickV/docker-login-gar.sh && cd /home/ubuntu/antigravity/repo/deployment/cloud-test && sudo docker compose pull budget-proxy ui && sudo docker compose up -d"
```

**Step 4: Health check**

Run:
```
gcloud compute ssh antigravity-v14-pilot --zone=us-central1-a --project=agentic1111 --command="curl -s http://localhost:4055/budget/history && echo && curl -s http://localhost:8080/health"
```
Expected: `/budget/history` returns JSON with `current_spend`, `max_daily`, `hourly_spend[24]`. `/health` returns 200.

---

### Task 6: Create PR

**Files:** None (git/GitHub operations)

**Step 1: Push all commits**

Run:
```
git -C "C:\Users\NickV\OneDrive\Desktop\Antigravity-Node" push origin feature/v14-phoenix
```

**Step 2: Create PR**

Run:
```
gh pr create --repo iamdatanick/Antigravity-Node --base master --head feature/v14-phoenix --title "feat: Antigravity Node v14.1 Phoenix — Cloud Deployment" --body "$(cat <<'EOF'
## Summary

- **30 → 8 containers**: Removed SeaweedFS, Milvus, StarRocks, Postgres, NATS, Ollama, Keycloak, Marquez, OpenSearch. Replaced with Ceph (S3), etcd3 (state), OpenBao (secrets), OVMS 2025.4 (inference), OTel (observability).
- **AsyncDAGEngine**: Python-native DAG engine replaces Argo Workflows. etcd3 for distributed locking, aioboto3 for Ceph S3 artifacts.
- **Budget Proxy**: Lightweight LLM router with $50/day cost controls, OpenBao vault key fetching, hourly spend tracking.
- **Consciousness Chamber UI**: Futuristic chat interface with particle fields, plasma orb AI presence, glassmorphism message cards.
- **GCP Deployment**: Terraform → c2-standard-8 (AVX-512), Docker images via Google Artifact Registry, deterministic boot sequence.
- **Validation Gates**: VG-101 through VG-109 test suite for spec compliance.

## Infrastructure

| Layer | Service | Image |
|-------|---------|-------|
| L0 | etcd v3.5.17 | `quay.io/coreos/etcd:v3.5.17` |
| L0 | Ceph v18 | `quay.io/ceph/demo:latest` |
| L0 | OpenBao 2.1.0 | `openbao/openbao:2.1.0` |
| L0 | OTel Collector | `otel/opentelemetry-collector-contrib:0.95.0` |
| L2 | OVMS 2025.4 | `openvino/model_server:2025.4` |
| L3 | Orchestrator | GAR `orchestrator:v14.1` |
| L3 | Budget Proxy | GAR `budget-proxy:v14.1` |
| L4 | UI | GAR `ui:v14.1` |

## Test plan

- [ ] `pytest tests/test_validation_gates.py -v` — all 10 gates pass
- [ ] `pytest tests/test_budget_proxy.py -v` — budget API tests pass
- [ ] `pytest tests/test_engine.py -v` — AsyncDAGEngine tests pass
- [ ] `ruff check .` — lint clean
- [ ] VM health: `curl http://34.170.105.203:8080/health` returns 200
- [ ] VM UI: `http://34.170.105.203:1055` loads Consciousness Chamber
- [ ] Budget history: `curl http://34.170.105.203:4055/budget/history` returns hourly data

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Step 3: Return PR URL**

Expected: PR URL printed to stdout.

---

## Summary: File Manifest

| Action | File | Task |
|--------|------|------|
| MODIFY | `tests/test_validation_gates.py:128-141` | 1 |
| MODIFY | `src/budget-proxy/proxy.py` | 2 |
| CREATE | `tests/test_budget_proxy.py` | 2 |
| MODIFY | `src/master-ui/nginx.conf:65-83` | 3 |
| MODIFY | Various (lint fixes) | 4 |
| DOCKER | budget-proxy + ui images | 5 |
| GIT | PR `feature/v14-phoenix` → `master` | 6 |
