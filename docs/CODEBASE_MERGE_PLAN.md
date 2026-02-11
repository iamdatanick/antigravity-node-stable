# 🔀 Codebase Consolidation & Merge Plan

**Generated:** 2026-02-08
**Target:** Create unified codebase at `Antigravity-Node`

---

## 📊 Project Inventory

### Desktop Projects
| Project | Location | Type | Key Components |
|---------|----------|------|----------------|
| **Antigravity-Node** | Desktop/ | Target | .github, config, workflows, src, tests |
| **agentic-cdp-review** | Desktop/ | Review | agentic-cdp subfolder |
| **agentic-cdp-source** | Desktop/ | Source | Raw extracted |
| **goose** | Desktop/ | FORTRESS | Security microservice |
| **intel-zero-trust-audit** | Desktop/ | Security | Audit patterns |
| **zero-trust-gate** | Desktop/ | Security | Gate implementation |

### Claude-Desktop Codebases
| Project | Subfolders | Unique Assets |
|---------|------------|---------------|
| **agentic-cdp** | agent_runner, agents, bootstrap_loop_review, cdp-agent-sdk, cdp-services, goose-runner, mcp-servers | hooks.py, orchestrator.py, main.py |
| **centillion-ai-platform** | Same + data-loader, intel-superbuilder, monitoring, recipes, zerotrust-gate | gate.py, data loaders, recipes |
| **bootstrap loop agent deploys** | bootstrap-loop-cdp, OVMS-Bootstrap-Deployer, sentinel-bootstrap-extracted, skills | Deployment configs |

### Downloads Codebases  
| Project | Subfolders | Unique Assets |
|---------|------------|---------------|
| **bootstrap-loop-cdp** | 1-hardware-layer, 2-data-layer, 3-agentic-layer, api, intel-ai-super-builder, tools, ui, workbooks | Layer architecture |
| **agentic-workflows** | src, templates, examples, docs | Workflow definitions |
| **agentic-workflows-v3** | - | V3 workflows |
| **agentic-workflows-v4.1.skill** | - | Latest skill |
| **mcp-audience-segmentation** | - | MCP segmentation |

---

## 🎯 Merge Strategy: Unified Antigravity-Node

### Target Structure
```
Antigravity-Node/
├── .github/
│   └── workflows/
│       ├── ci.yml                    # FROM: centillion-ai-platform
│       └── docker.yml                # FROM: centillion-ai-platform
├── config/
│   ├── mcp-catalog.yaml              ✅ EXISTS
│   ├── grafana/                      ✅ EXISTS
│   ├── litellm/                      ✅ EXISTS
│   └── spire/                        ✅ EXISTS
├── docs/
│   ├── INTEGRATION_PLAN.md           ✅ EXISTS
│   └── MERGE_SPECIFICATION.md        ✅ EXISTS
├── src/
│   ├── agent_runner/                 🆕 FROM: agentic-cdp
│   │   ├── main.py                   # Core orchestration
│   │   ├── hooks.py                  # HookRegistry
│   │   ├── orchestrator.py           # Plan→Act→Critique
│   │   ├── bootstrap.py              # Bootstrap logic
│   │   ├── vault_secrets.py          # Secrets management
│   │   └── telemetry.py              # OpenLineage
│   ├── cdp-services/                 🆕 FROM: agentic-cdp
│   ├── data-loader/                  🆕 FROM: centillion-ai-platform
│   ├── intel-superbuilder/           🆕 FROM: centillion-ai-platform
│   ├── mcp-servers/                  🆕 FROM: agentic-cdp
│   ├── zerotrust-gate/               🆕 FROM: centillion-ai-platform
│   ├── master-ui/                    ✅ EXISTS
│   ├── mcp-filesystem/               ✅ EXISTS
│   ├── mcp-starrocks/                ✅ EXISTS
│   ├── security/                     ✅ EXISTS
│   └── trace-viewer/                 ✅ EXISTS
├── workflows/
│   ├── main.py                       ✅ EXISTS (enhance)
│   ├── a2a_server.py                 ✅ EXISTS
│   ├── goose_client.py               ✅ EXISTS (merge goose_block.py)
│   ├── hooks.py                      🆕 FROM: agent_runner/hooks.py
│   ├── orchestrator.py               🆕 FROM: agent_runner/orchestrator.py
│   ├── context.py                    🆕 FROM: bootstrap_loop_review/context_graph_enhanced.py
│   └── zerotrust_gate.py             🆕 FROM: zerotrust-gate/gate.py
├── skills/                           🆕 FROM: Claude-Desktop/skills + Downloads
├── recipes/                          🆕 FROM: centillion-ai-platform
├── tests/
│   ├── conftest.py                   ✅ EXISTS
│   ├── test_main.py                  🆕 CREATE
│   └── test_hooks.py                 🆕 CREATE
├── Dockerfile                        ✅ EXISTS
├── docker-compose.yml                ✅ EXISTS
├── pyproject.toml                    🆕 CREATE
└── requirements.txt                  ✅ EXISTS (update)
```

---

## 📋 Merge Phases

### Phase 1: Core Agent Infrastructure
**Source:** `Claude-Desktop/agentic-cdp/agent_runner/`
**Target:** `Antigravity-Node/src/agent_runner/`

| File | LOC | Priority | Description |
|------|-----|----------|-------------|
| main.py | 2,568 | P0 | Core orchestration, OIDC, SecurityLayer |
| hooks.py | 746 | P0 | HookRegistry, PRE_TOOL, POST_TOOL |
| orchestrator.py | 743 | P0 | Plan→Act→Critique, LangGraph |
| bootstrap.py | ~400 | P1 | Bootstrap sequences |
| vault_secrets.py | ~300 | P1 | HashiCorp Vault integration |
| telemetry.py | ~400 | P1 | OpenLineage integration |

### Phase 2: Security Components
**Source:** `Claude-Desktop/centillion-ai-platform/zerotrust-gate/`
**Target:** `Antigravity-Node/src/zerotrust-gate/`

| File | LOC | Priority | Description |
|------|-----|----------|-------------|
| gate.py | 796 | P0 | GateMode, TOOL_SENSITIVITY, CAMARA |
| goose_block.py | 746 | P1 | MCPClient, enhanced Goose patterns |

### Phase 3: Data Infrastructure
**Source:** `Claude-Desktop/centillion-ai-platform/data-loader/`
**Target:** `Antigravity-Node/src/data-loader/`

| File | LOC | Priority | Description |
|------|-----|----------|-------------|
| main.py | 887 | P1 | Document ingestion, embeddings |
| splitters.py | ~300 | P2 | Text chunking strategies |

### Phase 4: Context & Workflows
**Source:** `Claude-Desktop/agentic-cdp/bootstrap_loop_review/`
**Target:** `Antigravity-Node/workflows/`

| File | LOC | Priority | Description |
|------|-----|----------|-------------|
| context_graph_enhanced.py | 1,187 | P1 | ContextNode, ContextEdge, handoff |
| workflow_executor.py | ~500 | P2 | Workflow execution |

### Phase 5: Skills & Recipes
**Source:** Multiple locations
**Target:** `Antigravity-Node/skills/` and `Antigravity-Node/recipes/`

| Source | Files | Priority |
|--------|-------|----------|
| Claude-Desktop/skills/ | *.skill | P2 |
| Downloads/agentic-workflows-v4.1.skill/ | Latest | P2 |
| centillion-ai-platform/recipes/ | *.yaml | P2 |

---

## 🔧 Merge Commands

### Step 1: Create Target Directories
```powershell
$target = "C:\Users\NickV\OneDrive\Desktop\Antigravity-Node"
New-Item -ItemType Directory -Force -Path "$target\src\agent_runner"
New-Item -ItemType Directory -Force -Path "$target\src\zerotrust-gate"
New-Item -ItemType Directory -Force -Path "$target\src\data-loader"
New-Item -ItemType Directory -Force -Path "$target\skills"
New-Item -ItemType Directory -Force -Path "$target\recipes"
```

### Step 2: Copy Agent Runner
```powershell
$source = "C:\Users\NickV\OneDrive\Desktop\Claude-Desktop\agentic-cdp\agent_runner"
$dest = "C:\Users\NickV\OneDrive\Desktop\Antigravity-Node\src\agent_runner"
Copy-Item -Path "$source\*.py" -Destination $dest -Force
```

### Step 3: Copy Zero Trust Gate
```powershell
$source = "C:\Users\NickV\OneDrive\Desktop\Claude-Desktop\centillion-ai-platform\zerotrust-gate"
$dest = "C:\Users\NickV\OneDrive\Desktop\Antigravity-Node\src\zerotrust-gate"
Copy-Item -Path "$source\*" -Destination $dest -Recurse -Force
```

### Step 4: Copy Data Loader
```powershell
$source = "C:\Users\NickV\OneDrive\Desktop\Claude-Desktop\centillion-ai-platform\data-loader"
$dest = "C:\Users\NickV\OneDrive\Desktop\Antigravity-Node\src\data-loader"
Copy-Item -Path "$source\*" -Destination $dest -Recurse -Force
```

### Step 5: Copy Goose Block to Workflows
```powershell
$source = "C:\Users\NickV\OneDrive\Desktop\Claude-Desktop\centillion-ai-platform\goose-runner\goose_block.py"
$dest = "C:\Users\NickV\OneDrive\Desktop\Antigravity-Node\workflows\goose_block.py"
Copy-Item -Path $source -Destination $dest -Force
```

### Step 6: Copy CI/CD
```powershell
$source = "C:\Users\NickV\OneDrive\Desktop\Claude-Desktop\centillion-ai-platform\.github"
$dest = "C:\Users\NickV\OneDrive\Desktop\Antigravity-Node\.github"
Copy-Item -Path "$source\*" -Destination $dest -Recurse -Force
```

---

## 📊 Component Deduplication Map

### Duplicate Projects to Consolidate

| Keep (Primary) | Delete/Archive | Reason |
|----------------|----------------|--------|
| Desktop/Antigravity-Node | Desktop/antigravity-node-cleanup | Primary active |
| | Desktop/Antigravity_Node | Underscore variant |
| | Knowledge-Base/.../Antigravity-Node | Analysis copy |
| Claude-Desktop/centillion-ai-platform | Claude-Desktop/agentic-cdp | Centillion is superset |
| Downloads/agentic-workflows-v4.1.skill | Downloads/agentic-workflows | Latest version |
| | Downloads/agentic-workflows-v3 | Old version |
| | Downloads/agentic-workflows-fixed.skill | Intermediate |
| Desktop/013026-bootstrap-loop-cdp | Downloads/bootstrap-loop-cdp | Desktop is newer |
| | Claude-Desktop/bootstrap loop agent deploys | Intermediate |

### Similar Codebases Analysis

| Codebase A | Codebase B | Similarity | Merge Action |
|------------|------------|------------|--------------|
| agentic-cdp | centillion-ai-platform | 85% | Centillion absorbs agentic-cdp |
| bootstrap-loop-cdp (Desktop) | bootstrap-loop-cdp (Downloads) | 70% | Keep Desktop, reference Downloads layers |
| Antigravity-Node (Desktop) | Antigravity-Node (KB) | 60% | Desktop is target, KB is reference |

---

## 🎯 Post-Merge Cleanup

### Delete After Merge
```powershell
# After verifying merge success:
# Remove-Item -Path "C:\Users\NickV\OneDrive\Desktop\antigravity-node-cleanup" -Recurse
# Remove-Item -Path "C:\Users\NickV\OneDrive\Desktop\Antigravity_Node" -Recurse  
# Remove-Item -Path "C:\Users\NickV\Downloads\agentic-workflows" -Recurse
# Remove-Item -Path "C:\Users\NickV\Downloads\agentic-workflows-v3" -Recurse
```

### Archive (Move to Knowledge-Base)
```powershell
# Move-Item "C:\Users\NickV\OneDrive\Desktop\agentic-cdp-review" "C:\Users\NickV\OneDrive\Desktop\Knowledge-Base\03-Resources\Archive\"
# Move-Item "C:\Users\NickV\Downloads\bootstrap-loop-cdp" "C:\Users\NickV\OneDrive\Desktop\Knowledge-Base\03-Resources\Archive\"
```

---

## 📈 Expected Outcome

### Before Merge
- 8+ scattered codebases
- ~500k+ lines of code with 60%+ duplication
- No unified entry point
- Inconsistent patterns

### After Merge
- 1 unified Antigravity-Node codebase
- ~50k lines of curated code
- Single entry point with modular components
- Consistent patterns:
  - Plan→Act→Critique orchestration
  - HookRegistry for tool interception
  - Zero-Trust gate for security
  - OpenLineage for telemetry

---

*Execute Phase 1-2 first, validate, then proceed with Phase 3-5*
