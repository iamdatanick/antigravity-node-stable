# UI Operational Plan — Make Antigravity Node Usable

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the Antigravity Node UI functional end-to-end — models load, navigation works, broken pages are handled, and the system feels like a single operational product (Goose-like UX).

**Architecture:** Fix the model loading chain (budget-proxy `/v1/models` endpoint missing), update stale v13 references to v14.1, disable pages that depend on removed services, auto-refresh models when API keys change, and update the capabilities endpoint to reflect v14.1 reality.

**Tech Stack:** Python/FastAPI (budget-proxy, orchestrator), React 19 + TypeScript + TailwindCSS v4 + Zustand 5 + TanStack Query 5 (UI)

---

## Phase 1: Fix Model Loading (Root Cause)

### Task 1: Add /v1/models endpoint to budget-proxy

**Files:**
- Modify: `src/budget-proxy/proxy.py`
- Test: `tests/test_budget_proxy.py`

**Step 1: Write the failing test**

Add to `tests/test_budget_proxy.py`:

```python
class TestModelsEndpoint:
    """Tests for GET /v1/models model listing."""

    @pytest.mark.asyncio
    async def test_models_returns_200(self, client):
        """Models endpoint returns 200 with OpenAI-compatible format."""
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert isinstance(body["data"], list)
        assert len(body["data"]) > 0

    @pytest.mark.asyncio
    async def test_models_include_cost_table_entries(self, client):
        """Every model in COST_TABLE appears in the models list."""
        from proxy import COST_TABLE
        resp = await client.get("/v1/models")
        model_ids = {m["id"] for m in resp.json()["data"]}
        for model_name in COST_TABLE:
            if model_name != "local":
                assert model_name in model_ids

    @pytest.mark.asyncio
    async def test_models_have_required_fields(self, client):
        """Each model entry has id, object, owned_by fields."""
        resp = await client.get("/v1/models")
        for m in resp.json()["data"]:
            assert "id" in m
            assert "object" in m
            assert "owned_by" in m
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_budget_proxy.py::TestModelsEndpoint -v`
Expected: FAIL with 404

**Step 3: Write minimal implementation**

Add to `src/budget-proxy/proxy.py` before the `if __name__` block:

```python
@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model list derived from COST_TABLE."""
    models = []
    for model_name in COST_TABLE:
        if model_name == "local":
            continue
        models.append({
            "id": model_name,
            "object": "model",
            "owned_by": "antigravity",
        })
    # Add local model entry if LOCAL_LLM_URL is configured
    if LOCAL_LLM_URL:
        models.append({
            "id": "local/default",
            "object": "model",
            "owned_by": "local",
        })
    return {"object": "list", "data": models}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_budget_proxy.py::TestModelsEndpoint -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_budget_proxy.py src/budget-proxy/proxy.py
git commit -m "feat: add /v1/models endpoint to budget-proxy"
```

---

### Task 2: Auto-refresh models on API key save

**Files:**
- Modify: `src/master-ui/src/api/settings.ts`

**Step 1: Add queryClient invalidation for models**

In `useSaveApiKey()`, update the `onSuccess` handler to also invalidate the `["models"]` query:

```typescript
export function useSaveApiKey() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: SaveApiKeyRequest) =>
      apiFetch("/api/settings/keys", {
        method: "POST",
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["api-keys"] });
      qc.invalidateQueries({ queryKey: ["models"] });
    },
  });
}
```

Also update `useDeleteApiKey()`:

```typescript
export function useDeleteApiKey() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (provider: string) =>
      apiFetch(`/api/settings/keys/${provider}`, { method: "DELETE" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["api-keys"] });
      qc.invalidateQueries({ queryKey: ["models"] });
    },
  });
}
```

**Step 2: Commit**

```bash
git add src/master-ui/src/api/settings.ts
git commit -m "feat: auto-refresh models list when API keys change"
```

---

### Task 3: Expand AmbientHUD model fallback

**Files:**
- Modify: `src/master-ui/src/components/chat/AmbientHUD.tsx`

**Step 1: Replace single-option fallback with full COST_TABLE set**

In `AmbientHUD.tsx`, change the fallback `<option>` block (line 77):

From:
```tsx
<option value="gpt-4o" className="bg-[#0f172a]">gpt-4o</option>
```

To:
```tsx
<>
  <option value="gpt-4o" className="bg-[#0f172a]">GPT-4o</option>
  <option value="gpt-4o-mini" className="bg-[#0f172a]">GPT-4o Mini</option>
  <option value="claude-sonnet-4-20250514" className="bg-[#0f172a]">Claude Sonnet 4</option>
  <option value="claude-haiku-4-5-20251001" className="bg-[#0f172a]">Claude Haiku 4.5</option>
</>
```

**Step 2: Commit**

```bash
git add src/master-ui/src/components/chat/AmbientHUD.tsx
git commit -m "fix: expand model fallback list in AmbientHUD"
```

---

## Phase 2: Fix Navigation

### Task 4: Update version strings from v13 to v14.1

**Files:**
- Modify: `src/master-ui/src/components/layout/Sidebar.tsx`
- Modify: `workflows/a2a_server.py`

**Step 1: Update Sidebar version**

In `Sidebar.tsx` line 48, change:
```tsx
<div className="text-[10px] text-[var(--color-text-muted)]">v13.0 — The God Node</div>
```
To:
```tsx
<div className="text-[10px] text-[var(--color-text-muted)]">v14.1 — Phoenix</div>
```

**Step 2: Update a2a_server.py version strings**

In `workflows/a2a_server.py`:

- Line 69: `app = FastAPI(title="Antigravity Node v13.0", version="13.0.0")`
  → `app = FastAPI(title="Antigravity Node v14.1", version="14.1.0")`

- Line 414: `_system_prompt_cache = "You are the Antigravity Node v13.0, a sovereign AI agent."`
  → `_system_prompt_cache = "You are the Antigravity Node v14.1 Phoenix, a sovereign AI agent."`

- Line 704: `"node": "Antigravity Node v13.0",`
  → `"node": "Antigravity Node v14.1 Phoenix",`

**Step 3: Commit**

```bash
git add src/master-ui/src/components/layout/Sidebar.tsx workflows/a2a_server.py
git commit -m "fix: update version strings v13.0 → v14.1 Phoenix"
```

---

### Task 5: Disable broken nav items and create Unavailable page

**Files:**
- Create: `src/master-ui/src/pages/Unavailable.tsx`
- Modify: `src/master-ui/src/components/layout/Sidebar.tsx`
- Modify: `src/master-ui/src/App.tsx`

**Step 1: Create Unavailable.tsx**

```tsx
import { Construction } from "lucide-react";

interface UnavailableProps {
  feature: string;
  reason: string;
}

export default function Unavailable({ feature, reason }: UnavailableProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-4 text-center px-8">
      <Construction size={48} className="text-[var(--color-text-muted)] opacity-40" />
      <h1 className="text-lg font-bold text-[var(--color-text-primary)]">{feature}</h1>
      <p className="text-sm text-[var(--color-text-muted)] max-w-md">{reason}</p>
    </div>
  );
}
```

**Step 2: Update App.tsx routes**

Replace broken-service routes with Unavailable:

```tsx
import Unavailable from "./pages/Unavailable";

// In Routes:
<Route path="/logs" element={<Unavailable feature="Logs" reason="Log streaming requires OpenSearch, which was removed in v14.1 Phoenix. Logs are available via docker compose logs." />} />
<Route path="/memory" element={<Unavailable feature="Memory" reason="Episodic memory requires StarRocks, which was removed in v14.1 Phoenix. Memory is stored in etcd." />} />
<Route path="/query" element={<Unavailable feature="Query" reason="SQL query requires StarRocks, which was removed in v14.1 Phoenix." />} />
<Route path="/workflows" element={<Unavailable feature="Workflows" reason="Workflow visualization requires Argo, which was replaced by AsyncDAGEngine in v14.1 Phoenix." />} />
```

**Step 3: Dim broken nav items in Sidebar**

Add an `available` flag to the NAV array:

```tsx
const NAV = [
  { to: "/", icon: LayoutDashboard, label: "Dashboard", available: true },
  { to: "/chat", icon: MessageSquare, label: "Chat", available: true },
  { to: "/logs", icon: ScrollText, label: "Logs", available: false },
  { to: "/memory", icon: Database, label: "Memory", available: false },
  { to: "/query", icon: Terminal, label: "Query", available: false },
  { to: "/workflows", icon: Workflow, label: "Workflows", available: false },
  { to: "/budget", icon: DollarSign, label: "Budget", available: true },
  { to: "/services", icon: Server, label: "Services", available: true },
  { to: "/settings", icon: Settings, label: "Settings", available: true },
] as const;
```

In the nav link rendering, add opacity for unavailable items:

```tsx
<NavLink
  key={to}
  to={to}
  end={to === "/"}
  className={({ isActive }) =>
    `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
      !available ? "opacity-30 pointer-events-none" :
      isActive
        ? "bg-[var(--color-accent-dim)] text-[var(--color-accent)] font-medium"
        : "text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)] hover:text-[var(--color-text-primary)]"
    }`
  }
>
```

**Step 4: Run build to verify**

Run: `cd src/master-ui && npm run build`
Expected: Build succeeds with no errors

**Step 5: Commit**

```bash
git add src/master-ui/src/pages/Unavailable.tsx src/master-ui/src/App.tsx src/master-ui/src/components/layout/Sidebar.tsx
git commit -m "feat: disable broken nav items + Unavailable page for removed v13 services"
```

---

## Phase 3: Update Capabilities to v14.1 Reality

### Task 6: Fix /capabilities endpoint

**Files:**
- Modify: `workflows/a2a_server.py`

**Step 1: Update the capabilities endpoint**

Replace the capabilities dict (lines 703-737) to reflect v14.1:

```python
@app.get("/capabilities", response_model=CapabilitiesResponse)
async def capabilities():
    """Return full node capabilities for A2A discovery."""
    return {
        "node": "Antigravity Node v14.1 Phoenix",
        "protocols": ["a2a", "mcp", "openai-compatible"],
        "endpoints": {
            "health": "/health",
            "task": "/task",
            "upload": "/upload",
            "handoff": "/handoff",
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "inference": "/v1/inference",
            "ovms_models": "/v1/models/ovms",
            "tools": "/tools",
            "capabilities": "/capabilities",
            "budget_history": "/budget/history",
            "agent_descriptor": "/.well-known/agent.json",
        },
        "mcp_servers": {},
        "memory": {
            "episodic": "etcd KV store",
        },
        "budget": {
            "proxy": "budget-proxy",
            "max_daily": os.environ.get("DAILY_BUDGET_USD", "$50.00"),
            "model": os.environ.get("GOOSE_MODEL", "gpt-4o"),
        },
    }
```

**Step 2: Commit**

```bash
git add workflows/a2a_server.py
git commit -m "fix: update /capabilities to reflect v14.1 Phoenix architecture"
```

---

### Task 7: Fix stale tool references in /tools endpoint

**Files:**
- Modify: `workflows/a2a_server.py`

**Step 1: Update the /tools endpoint**

Replace the builtin tools list and MCP server references (lines 611-665) to remove StarRocks, Milvus, Argo references:

```python
@app.get("/tools", response_model=ToolsResponse)
async def list_tools():
    """List all available tools."""
    tools = [
        {"name": "chat", "server": "budget-proxy", "description": "LLM chat via budget-proxy with cost controls"},
        {"name": "upload", "server": "orchestrator", "description": "Upload files to Ceph S3-compatible storage"},
        {"name": "inference", "server": "ovms", "description": "Run inference on OVMS-served OpenVINO models"},
    ]
    return {"tools": tools, "total": len(tools)}
```

**Step 2: Commit**

```bash
git add workflows/a2a_server.py
git commit -m "fix: remove stale StarRocks/Argo/Milvus tool references"
```

---

## Phase 4: Fix Budget Proxy Base URL

### Task 8: Fix LITELLM_BASE default URL

**Files:**
- Modify: `workflows/a2a_server.py`

**Step 1: Fix the budget-proxy base URL default**

Line 360: `LITELLM_BASE = os.environ.get("LITELLM_URL", os.environ.get("LITELLM_BASE_URL", "http://budget-proxy:4000"))`

Port 4000 is wrong — budget-proxy runs on 4055. Change to:

```python
LITELLM_BASE = os.environ.get("LITELLM_URL", os.environ.get("LITELLM_BASE_URL", "http://budget-proxy:4055"))
```

**Step 2: Commit**

```bash
git add workflows/a2a_server.py
git commit -m "fix: correct budget-proxy default port 4000 → 4055"
```

---

## Phase 5: Lint, Test, Build

### Task 9: Run full validation suite

**Step 1: Lint Python**

Run: `ruff check src/budget-proxy/proxy.py workflows/a2a_server.py`
Expected: No errors

**Step 2: Run all tests**

Run: `pytest tests/ -q`
Expected: All pass (including new TestModelsEndpoint)

**Step 3: Build UI**

Run: `cd src/master-ui && npm run build`
Expected: Build succeeds

**Step 4: Lint UI**

Run: `cd src/master-ui && npx tsc --noEmit`
Expected: No type errors

**Step 5: Commit any format fixes**

```bash
ruff format src/budget-proxy/proxy.py workflows/a2a_server.py
git add -p  # stage only formatting changes
git commit -m "style: format Python files"
```

---

## Phase 6: Rebuild, Push, Deploy

### Task 10: Build and push Docker images

**Step 1: Build budget-proxy**

```powershell
docker build -t us-central1-docker.pkg.dev/agentic1111/antigravity/budget-proxy:v14.1 -f src/budget-proxy/Dockerfile .
```

**Step 2: Build UI**

```powershell
docker build -t us-central1-docker.pkg.dev/agentic1111/antigravity/ui:v14.1 -f src/master-ui/Dockerfile .
```

**Step 3: Build orchestrator**

```powershell
docker build -t us-central1-docker.pkg.dev/agentic1111/antigravity/orchestrator:v14.1 -f src/orchestrator/Dockerfile.cloud .
```

**Step 4: Push all 3 to GAR**

```powershell
docker push us-central1-docker.pkg.dev/agentic1111/antigravity/budget-proxy:v14.1
docker push us-central1-docker.pkg.dev/agentic1111/antigravity/ui:v14.1
docker push us-central1-docker.pkg.dev/agentic1111/antigravity/orchestrator:v14.1
```

**Step 5: Push git**

```bash
git push origin feature/v14-phoenix
```

---

### Task 11: Deploy to VM and verify

**Step 1: Deploy**

```powershell
gcloud compute ssh antigravity-v14-pilot --zone=us-central1-a --project=agentic1111 --command="cd /opt/antigravity && docker compose pull && docker compose up -d"
```

**Step 2: Verify model loading**

```bash
curl http://34.170.105.203:1055/api/v1/models
```
Expected: JSON with `{"object": "list", "data": [{"id": "gpt-4o", ...}, {"id": "gpt-4o-mini", ...}, ...]}` — 5+ models

**Step 3: Verify health**

```bash
curl http://34.170.105.203:1055/api/health
```
Expected: 200 with service health

**Step 4: Verify capabilities (v14.1)**

```bash
curl http://34.170.105.203:1055/api/capabilities
```
Expected: `"node": "Antigravity Node v14.1 Phoenix"` — no StarRocks/Milvus references

**Step 5: Browser test**

Navigate to `http://34.170.105.203:1055`:
- Sidebar shows "v14.1 — Phoenix"
- Chat page loads with model dropdown populated
- Logs/Memory/Query/Workflows nav items are dimmed
- Budget and Settings pages work
- Clicking a dimmed nav item shows Unavailable page with explanation

---

## Summary

| Phase | Tasks | What It Fixes |
|-------|-------|---------------|
| 1 | 1-3 | Models actually load in the dropdown |
| 2 | 4-5 | Navigation is honest — broken pages say so |
| 3 | 6-7 | Capabilities and tools reflect v14.1 reality |
| 4 | 8 | Chat requests actually reach budget-proxy |
| 5 | 9 | Everything passes lint/test/build |
| 6 | 10-11 | Deployed and verified on GCP |

**Total: 11 tasks, 6 phases.**
