# Antigravity Node v13.0 — User Manual

**"The God Node"** — A Sovereign AI Agent Platform

---

## Table of Contents

1. [What Is the Antigravity Node?](#1-what-is-the-antigravity-node)
2. [System Requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [First Boot](#4-first-boot)
5. [The Five UIs — Your Control Panels](#5-the-five-uis--your-control-panels)
6. [Chatting with the Agent (Open WebUI)](#6-chatting-with-the-agent-open-webui)
7. [Uploading Data](#7-uploading-data)
8. [Memory System — Episodic, Semantic, Procedural](#8-memory-system--episodic-semantic-procedural)
9. [Querying Tables (StarRocks SQL)](#9-querying-tables-starrocks-sql)
10. [Saving & Managing Tables](#10-saving--managing-tables)
11. [Thought Trace Viewer](#11-thought-trace-viewer)
12. [Grafana Dashboards & Logs](#12-grafana-dashboards--logs)
13. [Data Lineage (Marquez)](#13-data-lineage-marquez)
14. [File Operations (MCP Filesystem)](#14-file-operations-mcp-filesystem)
15. [Object Storage (SeaweedFS S3)](#15-object-storage-seaweedfs-s3)
16. [Workflows (Argo)](#16-workflows-argo)
17. [Health Monitoring](#17-health-monitoring)
18. [Budget & Cost Control](#18-budget--cost-control)
19. [Security Features](#19-security-features)
20. [API Reference](#20-api-reference)
21. [50+ Use Cases & Sample Workflows](#21-50-use-cases--sample-workflows)
22. [Troubleshooting](#22-troubleshooting)
23. [Architecture Reference](#23-architecture-reference)
24. [Port Reference](#24-port-reference)

---

## 1. What Is the Antigravity Node?

The Antigravity Node is a **self-contained AI agent platform** running ~30 Docker containers. It provides:

- **An AI Chat Interface** — Talk to an AI agent that remembers past conversations
- **3-Layer Memory** — The agent has episodic (events), semantic (knowledge), and procedural (skills) memory
- **File Ingestion** — Upload CSVs, PDFs, JSON, Excel files and the agent can read them
- **Data Lineage** — Track where every piece of data came from
- **Self-Healing** — If a service crashes, the system can auto-recover
- **Cost Control** — Hard $10/day budget cap on AI API calls
- **Visual Dashboards** — Grafana for metrics, Streamlit for thought traces

### Who Is This For?

- **Data analysts** who want an AI assistant with persistent memory
- **Developers** building agentic AI workflows
- **Teams** needing a private, self-hosted AI platform
- **Anyone** who wants an AI that learns from uploaded documents

---

## 2. System Requirements

### Minimum Hardware
| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 16 GB | 32 GB |
| CPU | 4 cores | 8 cores |
| Disk | 20 GB free | 50 GB free |
| OS | Windows 10/11, macOS, Linux | Windows 11 with WSL2 |

### Required Software
| Software | Version | Purpose |
|----------|---------|---------|
| **Docker Desktop** or **Rancher Desktop** | Latest | Container runtime |
| **Git** | 2.x+ | Clone the repository |
| **Web browser** | Chrome/Edge/Firefox | Access the UIs |

### API Keys (At Least One Required)
| Provider | Key Name | Get It At |
|----------|----------|-----------|
| OpenAI | `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| Anthropic | `ANTHROPIC_API_KEY` | https://console.anthropic.com/ |

> **Note:** You need at least ONE API key. The system routes through LiteLLM which supports both providers.

---

## 3. Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/iamdatanick/Antigravity-Node.git
cd Antigravity-Node
```

### Step 2: Create Your `.env` File

Create a file named `.env` in the project root:

```env
# Required: At least one API key
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Optional: Choose your default model
GOOSE_PROVIDER=openai
GOOSE_MODEL=gpt-4o

# Optional: Budget (default $10/day)
LITELLM_LOG_LEVEL=INFO
```

> **Security:** Never share your `.env` file. It contains your API keys.

### Step 3: Create Required Directories

```bash
# Windows (PowerShell)
New-Item -ItemType Directory -Force -Path data\starrocks, data\spire, models, config\grafana\provisioning\datasources

# macOS/Linux
mkdir -p data/starrocks data/spire models config/grafana/provisioning/datasources
```

### Step 4: Configure Your Downloads Folder Mount

The system mounts your local Downloads folder as read-only context. Edit `docker-compose.yml` line ~372 to point to YOUR Downloads folder:

```yaml
# Find this line in the orchestrator service:
- C:\Users\NickV\Downloads:/app/context:ro

# Change to YOUR path:
- C:\Users\YOUR_USERNAME\Downloads:/app/context:ro
# or on macOS:
- /Users/YOUR_USERNAME/Downloads:/app/context:ro
# or on Linux:
- /home/YOUR_USERNAME/Downloads:/app/context:ro
```

Also update the `mcp-filesystem` service (~line 346):
```yaml
# Change to YOUR path:
- C:\Users\YOUR_USERNAME\Downloads:/data:ro
```

### Step 5: Start Docker Desktop

1. Open **Docker Desktop** (or Rancher Desktop)
2. Wait until the Docker engine icon shows "Running"
3. Verify: `docker info` should return system info

---

## 4. First Boot

### Start Everything

```bash
docker compose up -d
```

This will:
1. Pull ~20 container images (first time takes 5-10 minutes)
2. Start services in dependency order (8 layers)
3. Initialize databases and memory tables

### Watch the Boot Progress

```bash
# See all containers starting
docker compose ps

# Watch live logs
docker compose logs -f --tail=20
```

### Verify Health

```bash
# Quick health check
curl http://localhost:8080/health

# Expected response:
# {"status":"healthy","levels":[...]}
```

### Boot Timeline

| Time | What Happens |
|------|-------------|
| 0-10s | volume-fixer sets permissions, exits |
| 10-30s | PostgreSQL, SeaweedFS, NATS start |
| 30-60s | Keycloak, Marquez, Loki, StarRocks start |
| 60-90s | K3D cluster bootstraps (may take longer first time) |
| 90-120s | MCP Gateway, tool servers, observability stack |
| 120-150s | Orchestrator starts, God Mode loop begins |
| 150-180s | Open WebUI, Trace Viewer, Master UI go live |

### Open the Master UI

Once boot is complete, open your browser to:

> **http://localhost:1055**

This is your **unified portal** — all 5 sub-UIs are accessible from here.

### First-Time Open WebUI Setup

1. Navigate to **http://localhost:3355** (or click "Chat" in Master UI)
2. If prompted with "What's New" dialog, click **"Okay, Let's Go!"**
3. You're ready to chat!

---

## 5. The Five UIs — Your Control Panels

| UI | Direct URL | Master UI Path | What It Does |
|----|-----------|----------------|-------------|
| **Chat** | http://localhost:3355 | /chat/ | Talk to the AI agent |
| **Workflows** | http://localhost:2755 | /argo/ | Monitor Argo workflows |
| **Lineage** | http://localhost:5155 | /lineage/ | Track data provenance |
| **Dashboards** | http://localhost:3055 | /grafana/ | Metrics & log dashboards |
| **Thought Trace** | http://localhost:8655 | /trace/ | See the agent's memory |

### Master UI Navigation

The Master UI at **http://localhost:1055** has a dark navigation bar at the top:

```
⚛ Antigravity v13.0  |  💬 Chat  |  🔄 Workflows  |  🔗 Lineage  |  📊 Dashboards  |  🧠 Thought Trace
```

Click any link to load that sub-UI in the main content area (iframe). You never need to leave this page.

---

## 6. Chatting with the Agent (Open WebUI)

### How to Start a Conversation

1. Open **http://localhost:3355** (or click Chat in Master UI)
2. Select a model from the dropdown (e.g., `gpt-4o`)
3. Type your message in the text box at the bottom
4. Press **Enter** or click the send button

### What Happens Behind the Scenes

When you send a message:

```
You type: "Analyze my sales data"
    ↓
Open WebUI → POST /v1/chat/completions → Orchestrator
    ↓
Orchestrator records your message in EPISODIC MEMORY (StarRocks)
    ↓
Orchestrator recalls your last 5 interactions for context
    ↓
Enriched prompt → LiteLLM proxy → OpenAI/Anthropic API
    ↓
AI response recorded in EPISODIC MEMORY
    ↓
Response displayed in Open WebUI
```

### Chat Features

| Feature | How to Use |
|---------|-----------|
| **New conversation** | Click "+" button in sidebar |
| **Switch models** | Dropdown at top of chat (gpt-4o, claude-sonnet-4-20250514) |
| **View history** | Left sidebar shows past conversations |
| **Upload a file** | Click the 📎 paperclip icon |
| **Code blocks** | AI responses render code with syntax highlighting |
| **Markdown** | AI responses render tables, lists, headers |
| **Copy response** | Hover over a response → click copy icon |
| **Regenerate** | Click the regenerate (↻) button on any response |

### Example Conversations

**Simple question:**
```
You: What is the capital of France?
AI: The capital of France is Paris...
```

**Ask about your data:**
```
You: What files do I have in my Downloads folder?
AI: Let me check your context files...
```

**Multi-turn conversation:**
```
You: I'm working on a marketing campaign for Q2
AI: I'd be happy to help with your Q2 campaign...
You: What were the results from last quarter?
AI: Based on my memory, in our previous session...
```

> **Memory:** The agent remembers past conversations! Each message is stored in StarRocks episodic memory and can be recalled later.

---

## 7. Uploading Data

### Method 1: Drag and Drop in Open WebUI

1. Open the Chat UI
2. Click the **📎 paperclip** icon or drag a file into the chat
3. The file is sent with your next message as context

### Method 2: HTTP Upload Endpoint (API)

Upload files directly via the REST API:

```bash
# Upload a CSV file
curl -X POST http://localhost:8080/upload \
  -F "file=@/path/to/your/data.csv" \
  -H "x-tenant-id: my-project"

# Expected response:
# {"status":"uploaded","key":"context/my-project/data.csv","size":1234}
```

```bash
# Upload a PDF
curl -X POST http://localhost:8080/upload \
  -F "file=@/path/to/report.pdf" \
  -H "x-tenant-id: research"

# Upload an Excel file
curl -X POST http://localhost:8080/upload \
  -F "file=@/path/to/financials.xlsx"
```

### Method 3: Place Files in Downloads Folder

Any files in your Downloads folder are automatically available:

1. Save/copy files to your **Downloads** folder
2. The orchestrator's God Mode loop scans for: `*.csv`, `*.pdf`, `*.json`, `*CANONICAL*.xlsx`
3. Files appear in the MCP filesystem tool

### Method 4: SeaweedFS S3 (Programmatic)

```python
import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:8455",
    aws_access_key_id="admin",
    aws_secret_access_key="admin",
    region_name="us-east-1",
)

# Create bucket
s3.create_bucket(Bucket="my-data")

# Upload file
s3.upload_file("/path/to/data.csv", "my-data", "data.csv")

# List files
response = s3.list_objects_v2(Bucket="my-data")
for obj in response.get("Contents", []):
    print(obj["Key"], obj["Size"])
```

### Supported File Types

| Type | Extension | Max Size |
|------|-----------|----------|
| CSV | .csv | 100 MB |
| JSON | .json | 100 MB |
| PDF | .pdf | 100 MB |
| Excel | .xlsx, .xls | 100 MB |
| Markdown | .md | 100 MB |
| Text | .txt | 100 MB |

---

## 8. Memory System — Episodic, Semantic, Procedural

The Antigravity Node has a **3-layer memory system** stored in StarRocks:

### Layer 1: Episodic Memory — "What Happened?"

Every interaction is recorded as an event:

| Field | Description | Example |
|-------|-------------|---------|
| `event_id` | Unique ID (timestamp-based) | 1707321600000001 |
| `tenant_id` | Project/user isolation | "my-project" |
| `timestamp` | When it happened | 2026-02-07 14:30:00 |
| `session_id` | Conversation session | "abc123" |
| `actor` | Who did it | "User" or "Goose" |
| `action_type` | What type of event | "TASK_REQUEST", "RESPONSE" |
| `content` | The actual content | "Analyze sales data" |

**How it's used:** Before answering any question, the agent recalls your last 5 interactions to maintain context across conversations.

### Layer 2: Semantic Memory — "What Do I Know?"

Document chunks stored for knowledge retrieval:

| Field | Description | Example |
|-------|-------------|---------|
| `doc_id` | Document identifier | "report-2026-q1" |
| `chunk_id` | Chunk within document | 3 |
| `content` | Text content | "Revenue increased 15%..." |
| `source_uri` | Where it came from | "s3://antigravity/reports/q1.pdf" |
| `tenant_id` | Tenant isolation | "finance" |

### Layer 3: Procedural Memory — "How Do I Do It?"

Skills and workflow templates:

| Field | Description | Example |
|-------|-------------|---------|
| `skill_id` | Skill name | "data-pipeline-etl" |
| `description` | What the skill does | "Extract, transform, load CSV data" |
| `argo_template_yaml` | Argo workflow YAML | (workflow definition) |
| `success_rate` | Historical success % | 0.95 |

---

## 9. Querying Tables (StarRocks SQL)

### Method 1: Through the Chat UI

Just ask in natural language:

```
You: Show me the last 10 things that happened in the system
You: What events were recorded today?
You: How many unique sessions are in my episodic memory?
```

### Method 2: Direct SQL via MCP Tool

Use the `execute_sql` MCP tool:

```bash
# Via curl to the orchestrator
curl -X POST http://localhost:8080/task \
  -H "Content-Type: application/json" \
  -H "x-tenant-id: system" \
  -d '{"goal": "Run SQL: SELECT COUNT(*) as total FROM memory_episodic", "context": ""}'
```

### Method 3: MySQL Client (Direct Connection)

StarRocks is MySQL-protocol compatible:

```bash
# Connect with any MySQL client
mysql -h 127.0.0.1 -P 9055 -u root

# Or with docker
docker compose exec starrocks mysql -h127.0.0.1 -P9030 -uroot
```

### Essential SQL Queries

#### View all databases
```sql
SHOW DATABASES;
```

#### Use the memory database
```sql
USE antigravity;
```

#### Show all tables
```sql
SHOW TABLES;
-- Returns: memory_episodic, memory_semantic, memory_procedural
```

#### View table schema
```sql
DESCRIBE memory_episodic;
DESCRIBE memory_semantic;
DESCRIBE memory_procedural;
```

#### Query episodic memory
```sql
-- Last 20 events
SELECT timestamp, actor, action_type, content
FROM memory_episodic
ORDER BY timestamp DESC
LIMIT 20;

-- Events for a specific tenant
SELECT * FROM memory_episodic
WHERE tenant_id = 'my-project'
ORDER BY timestamp DESC;

-- Count events by type
SELECT action_type, COUNT(*) as count
FROM memory_episodic
GROUP BY action_type
ORDER BY count DESC;

-- Events in the last hour
SELECT * FROM memory_episodic
WHERE timestamp > DATE_SUB(NOW(), INTERVAL 1 HOUR);
```

#### Query semantic memory
```sql
-- Search knowledge base
SELECT doc_id, chunk_id, content, source_uri
FROM memory_semantic
WHERE content LIKE '%revenue%';

-- List all documents
SELECT DISTINCT doc_id, source_uri
FROM memory_semantic;

-- Count chunks per document
SELECT doc_id, COUNT(*) as chunks
FROM memory_semantic
GROUP BY doc_id;
```

#### Query procedural memory
```sql
-- List all skills
SELECT skill_id, description, success_rate
FROM memory_procedural
ORDER BY success_rate DESC;

-- Find skills by name
SELECT * FROM memory_procedural
WHERE skill_id LIKE '%pipeline%';
```

---

## 10. Saving & Managing Tables

### Create a New Table

```sql
-- Connect via MySQL client first
USE antigravity;

CREATE TABLE IF NOT EXISTS my_sales_data (
    sale_id BIGINT,
    date DATE,
    product VARCHAR(128),
    amount DECIMAL(10,2),
    region VARCHAR(64)
) ENGINE=OLAP
PRIMARY KEY (sale_id)
DISTRIBUTED BY HASH(sale_id);
```

### Insert Data

```sql
INSERT INTO my_sales_data VALUES
(1, '2026-01-15', 'Widget A', 299.99, 'North'),
(2, '2026-01-16', 'Widget B', 149.50, 'South'),
(3, '2026-01-17', 'Widget A', 299.99, 'East'),
(4, '2026-01-18', 'Widget C', 499.00, 'West');
```

### Load CSV Data (Stream Load)

```bash
# Upload CSV directly to StarRocks via Stream Load
curl --location-trusted -u root: \
  -H "label:my_load_$(date +%s)" \
  -H "column_separator:," \
  -H "columns: sale_id, date, product, amount, region" \
  -T /path/to/sales.csv \
  http://localhost:8055/api/antigravity/my_sales_data/_stream_load
```

### Export Data

```sql
-- Export to CSV format (copy from query results)
SELECT * FROM my_sales_data
INTO OUTFILE "s3://antigravity/exports/sales.csv"
FORMAT AS CSV
PROPERTIES (
    "aws.s3.endpoint" = "http://seaweedfs:8333",
    "aws.s3.access_key" = "admin",
    "aws.s3.secret_key" = "admin"
);
```

### Drop a Table

```sql
-- WARNING: This permanently deletes the table and all data
DROP TABLE IF EXISTS my_sales_data;
```

> **Safety:** The MCP `execute_sql` tool blocks DROP, DELETE, TRUNCATE, and ALTER commands. You must use a direct MySQL client for destructive operations.

---

## 11. Thought Trace Viewer

The Trace Viewer at **http://localhost:8655** lets you visualize the agent's memory in real time.

### Features

#### Sidebar Filters
- **Actor**: Filter by "All", "Goose", or "User"
- **Action Type**: Filter by "All", "THOUGHT", "TOOL_USE", "RESPONSE", "TASK_REQUEST", "FILE_UPLOAD"
- **Max Records**: Slider from 10 to 500 (default: 100)

#### Summary Metrics
At the top of the page:
- **Total Events** — How many memory entries exist
- **Unique Sessions** — How many separate conversations
- **Unique Actors** — How many distinct actors (User, Goose, etc.)

#### Event Log Table
A sortable, scrollable table showing:
| Column | Description |
|--------|-------------|
| timestamp | When the event occurred |
| actor | Who performed the action |
| action_type | Category of action |
| content | The actual message/data |
| session_id | Which conversation session |
| tenant_id | Which project/tenant |

#### Thought Timeline
A Plotly interactive chart showing events over time:
- **X-axis**: Time
- **Y-axis**: Actor
- **Color**: Action type
- **Hover**: Shows content preview
- **Controls**: Zoom, pan, download as PNG

#### Export
Click **"Download as CSV"** to export the filtered data.

### How to Read the Thought Trace

```
Timeline Example:
───────────────────────────────────────────────
  14:00  [User]    TASK_REQUEST  "Analyze Q1 sales"
  14:00  [Goose]   THOUGHT       "Checking memory for context..."
  14:01  [Goose]   TOOL_USE      "query_starrocks: SELECT..."
  14:01  [Goose]   RESPONSE      "Based on the data, Q1 sales..."
  14:05  [User]    TASK_REQUEST  "Compare to Q4"
  14:05  [Goose]   RESPONSE      "Compared to Q4, revenue is up..."
───────────────────────────────────────────────
```

---

## 12. Grafana Dashboards & Logs

### Access Grafana

Open **http://localhost:3055** (or click Dashboards in Master UI).

Default credentials: **admin / admin** (anonymous access is enabled by default).

### Viewing Logs with Loki

1. Click **Explore** (compass icon) in the left sidebar
2. Select **Loki** as the datasource
3. Enter a LogQL query:

```logql
# All orchestrator logs
{container="antigravity_brain"}

# Errors only
{container="antigravity_brain"} |= "ERROR"

# God Mode loop status
{container="antigravity_brain"} |= "GOD MODE"

# Search across all containers
{job="docker"} |= "error" | json
```

### Useful Log Queries

| What You Want | LogQL Query |
|--------------|-------------|
| All errors | `{job="docker"} \|= "ERROR"` |
| Orchestrator activity | `{container="antigravity_brain"}` |
| Health checks | `{container="antigravity_brain"} \|= "health"` |
| File uploads | `{container="antigravity_brain"} \|= "uploaded"` |
| Memory operations | `{container="antigravity_brain"} \|= "memory"` |
| SeaweedFS activity | `{container_name=~".*seaweedfs.*"}` |
| MCP tool calls | `{container_name=~".*mcp.*"}` |

### Creating a Dashboard

1. Click **Dashboards** in the left sidebar
2. Click **New** → **New Dashboard**
3. Click **Add visualization**
4. Select **Loki** as datasource
5. Enter your LogQL query
6. Choose visualization type (time series, table, stat, etc.)
7. Click **Apply**
8. Click **Save** (💾 icon)

---

## 13. Data Lineage (Marquez)

### What Is Data Lineage?

Data lineage tracks **where data came from and where it went**. Every file the agent processes, every query it runs, and every workflow it triggers is recorded.

### Access Marquez

- **Web UI**: http://localhost:5155
- **API**: http://localhost:5055

### Using the Marquez UI

1. Open **http://localhost:5155** (or click Lineage in Master UI)
2. Browse **Namespaces** — each project/tenant is a namespace
3. Click a **Job** to see its inputs and outputs
4. Click a **Dataset** to see its lineage graph

### Marquez API Examples

```bash
# List all namespaces
curl http://localhost:5055/api/v1/namespaces

# List jobs in a namespace
curl http://localhost:5055/api/v1/namespaces/antigravity/jobs

# List datasets
curl http://localhost:5055/api/v1/namespaces/antigravity/datasets

# Get specific job run
curl http://localhost:5055/api/v1/namespaces/antigravity/jobs/my-job/runs
```

---

## 14. File Operations (MCP Filesystem)

The MCP Filesystem server provides tools to browse and read files from your Downloads folder.

### List Files

```bash
# List all files
curl -X POST http://localhost:8080/task \
  -H "Content-Type: application/json" \
  -H "x-tenant-id: system" \
  -d '{"goal": "List all files in my context folder", "context": ""}'
```

Or ask in the Chat UI:
```
You: What files are in my Downloads folder?
You: List all CSV files I have
You: Show me PDF files in my context
```

### Search Files

```
You: Find files containing "sales" in the name
You: Search for Excel files about marketing
```

### Read a File

```
You: Read the file customers.csv
You: Show me the contents of report.json
You: What's in the README.md file?
```

### File Info

```
You: How big is the data.csv file?
You: When was report.pdf last modified?
```

---

## 15. Object Storage (SeaweedFS S3)

SeaweedFS provides S3-compatible storage for artifacts, uploads, and exports.

### Access

- **S3 Endpoint**: http://localhost:8455
- **Master UI**: http://localhost:9355

### Using with AWS CLI

```bash
# Configure AWS CLI for SeaweedFS
aws configure set aws_access_key_id admin
aws configure set aws_secret_access_key admin

# List buckets
aws --endpoint-url http://localhost:8455 s3 ls

# Create a bucket
aws --endpoint-url http://localhost:8455 s3 mb s3://my-bucket

# Upload a file
aws --endpoint-url http://localhost:8455 s3 cp data.csv s3://my-bucket/

# Download a file
aws --endpoint-url http://localhost:8455 s3 cp s3://my-bucket/data.csv ./

# List objects in a bucket
aws --endpoint-url http://localhost:8455 s3 ls s3://my-bucket/
```

### Using with Python (boto3)

```python
import boto3

s3 = boto3.client("s3",
    endpoint_url="http://localhost:8455",
    aws_access_key_id="admin",
    aws_secret_access_key="admin"
)

# List buckets
print(s3.list_buckets()["Buckets"])

# Upload
s3.put_object(Bucket="antigravity", Key="test.txt", Body=b"Hello!")

# Download
obj = s3.get_object(Bucket="antigravity", Key="test.txt")
print(obj["Body"].read().decode())
```

---

## 16. Workflows (Argo)

### Access Argo UI

Open **http://localhost:2755** (or click Workflows in Master UI).

> **Note:** Argo runs inside K3D (Kubernetes-in-Docker). The UI may show 502 if K3D hasn't fully bootstrapped.

### Submit a Workflow via API

```bash
curl -X POST http://localhost:8080/task \
  -H "Content-Type: application/json" \
  -H "x-tenant-id: system" \
  -d '{
    "goal": "Submit a data processing workflow",
    "context": "Process the Q1 sales data"
  }'
```

### Submit via Chat

```
You: Run a data pipeline workflow for the uploaded CSV
You: Submit a workflow to process the sales data
```

---

## 17. Health Monitoring

### Quick Health Check

```bash
curl http://localhost:8080/health | python -m json.tool
```

### Response Format

```json
{
  "status": "healthy",
  "levels": [
    {
      "level": "L0",
      "name": "infrastructure",
      "checks": [
        {"name": "seaweedfs", "healthy": true, "error": null},
        {"name": "nats", "healthy": true, "error": null}
      ]
    },
    {
      "level": "L1",
      "name": "orchestration",
      "checks": [
        {"name": "argo", "healthy": false, "error": "Connection refused"}
      ]
    },
    {
      "level": "L2",
      "name": "services",
      "checks": [
        {"name": "starrocks", "healthy": true, "error": null},
        {"name": "milvus", "healthy": true, "error": null},
        {"name": "openbao", "healthy": true, "error": null},
        {"name": "keycloak", "healthy": true, "error": null}
      ]
    },
    {
      "level": "L3",
      "name": "agent",
      "checks": [
        {"name": "mcp_gateway", "healthy": true, "error": null}
      ]
    },
    {
      "level": "L4",
      "name": "observability",
      "checks": [
        {"name": "litellm", "healthy": true, "error": null},
        {"name": "marquez", "healthy": true, "error": null}
      ]
    }
  ]
}
```

### Health Levels Explained

| Level | Name | What It Checks |
|-------|------|---------------|
| **L0** | Infrastructure | SeaweedFS, NATS — core storage & messaging |
| **L1** | Orchestration | Argo Workflows — workflow engine |
| **L2** | Services | StarRocks, Milvus, OpenBao, Keycloak — data & security |
| **L3** | Agent | MCP Gateway — AI agent tooling |
| **L4** | Observability | LiteLLM, Marquez — monitoring & lineage |

### Check Individual Services

```bash
# SeaweedFS cluster status
curl http://localhost:9355/cluster/status

# StarRocks health
curl http://localhost:8055/api/health

# Grafana health
curl http://localhost:3055/api/health

# LiteLLM health
curl http://localhost:4055/health/readiness

# Marquez namespaces
curl http://localhost:5055/api/v1/namespaces
```

---

## 18. Budget & Cost Control

### How It Works

All AI API calls are routed through **LiteLLM** (port 4055), which enforces a **$10/day hard budget cap**.

```
Chat message → Orchestrator → LiteLLM Proxy → OpenAI/Anthropic API
                                    ↓
                           Budget check: $spent < $10?
                                    ↓
                            YES → Forward request
                            NO  → Return HTTP 429
```

### Check Budget Status

```bash
# LiteLLM health/readiness
curl http://localhost:4055/health/readiness

# List available models
curl http://localhost:4055/v1/models
```

### What Happens When Budget Is Exceeded

1. LiteLLM returns HTTP 429 (Too Many Requests)
2. The orchestrator catches the error
3. Chat UI shows: "I've reached my daily API budget limit. The budget resets in X hours."
4. Budget resets automatically every 24 hours

### Changing the Budget

Edit `docker-compose.yml`, find the `litellm` service:

```yaml
litellm:
  environment:
    - LITELLM_BUDGET_MAX=10.00    # Change this (dollars per day)
    - LITELLM_BUDGET_DURATION=24h  # Reset interval
```

Then restart LiteLLM:
```bash
docker compose restart litellm
```

---

## 19. Security Features

### Multi-Tenant Isolation

Every request includes an `x-tenant-id` header. Data is isolated per tenant:

```bash
# Tenant A's data
curl -X POST http://localhost:8080/task \
  -H "x-tenant-id: project-alpha" \
  -d '{"goal": "Show my data"}'

# Tenant B's data (separate)
curl -X POST http://localhost:8080/task \
  -H "x-tenant-id: project-beta" \
  -d '{"goal": "Show my data"}'
```

### SQL Injection Protection

The `execute_sql` MCP tool blocks destructive SQL:
- ❌ `DROP TABLE ...`
- ❌ `DELETE FROM ...`
- ❌ `TRUNCATE TABLE ...`
- ❌ `ALTER TABLE ...`
- ✅ `SELECT`, `INSERT`, `CREATE TABLE`, `SHOW` — all allowed

### Path Traversal Protection

The filesystem MCP server validates all paths:
```python
# This is blocked:
read_file("../../etc/passwd")  # "Path traversal denied"

# This is allowed:
read_file("data.csv")  # Reads from /data/data.csv only
```

### File Upload Limits

- Maximum file size: **100 MB**
- Files stored with tenant isolation: `context/{tenant_id}/{filename}`

### Secrets Management

OpenBao (HashiCorp Vault fork) stores sensitive data:
- Access: http://localhost:8255
- Dev token: `dev-only-token`
- Used by the agent for secure credential retrieval

### Identity & Access (Keycloak)

- Access: http://localhost:8355
- Admin: `admin / admin`
- Provides OAuth2/OIDC for production deployments

### Runtime Security (Falco)

Falco monitors container behavior for suspicious activity (running in read-only mode).

---

## 20. API Reference

### Base URL: `http://localhost:8080`

### GET Endpoints

| Endpoint | Description | Auth |
|----------|-------------|------|
| `GET /health` | 5-level health check | None |
| `GET /.well-known/agent.json` | Agent descriptor | None |
| `GET /v1/models` | Available AI models | None |
| `GET /tools` | List MCP tools | None |
| `GET /capabilities` | Full node capabilities | None |

### POST Endpoints

| Endpoint | Description | Headers | Body |
|----------|-------------|---------|------|
| `POST /task` | Submit task | `x-tenant-id` | `{goal, context, session_id?}` |
| `POST /handoff` | Agent handoff | `x-tenant-id?` | `{target_agent, payload}` |
| `POST /upload` | Upload file | `x-tenant-id?` | multipart/form-data |
| `POST /webhook` | Argo callback | None | `{task_id, status, message}` |
| `POST /v1/chat/completions` | Chat (OpenAI format) | `x-tenant-id?` | OpenAI chat format |

### Chat Completions Format

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-tenant-id: my-project" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "temperature": 0.7,
    "max_tokens": 1000
  }'
```

### Response Format

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1707321600,
  "model": "gpt-4o",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "2 + 2 = 4"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "total_tokens": 30
  }
}
```

---

## 21. 50+ Use Cases & Sample Workflows

### Category 1: Basic Chat & Q&A (Use Cases 1-8)

#### UC-1: Ask a General Question
```
You: What are the best practices for data governance?
AI: Data governance best practices include...
```

#### UC-2: Get a Code Snippet
```
You: Write a Python function to calculate moving average
AI: Here's a moving average function...
```

#### UC-3: Explain a Concept
```
You: Explain what vector databases are in simple terms
AI: A vector database stores data as mathematical representations...
```

#### UC-4: Summarize Text
```
You: Summarize this article: [paste long text]
AI: The key points are: 1) ... 2) ... 3) ...
```

#### UC-5: Translate Content
```
You: Translate "Hello, how are you?" to Spanish, French, and Japanese
AI: Spanish: "Hola, ¿cómo estás?"...
```

#### UC-6: Compare Options
```
You: Compare PostgreSQL vs MySQL for OLAP workloads
AI: For OLAP workloads, here's a comparison...
```

#### UC-7: Generate a Checklist
```
You: Create a deployment checklist for a production release
AI: ☐ Run unit tests...
```

#### UC-8: Draft an Email
```
You: Draft a professional email declining a meeting request
AI: Subject: Re: Meeting Request...
```

---

### Category 2: Data Analysis (Use Cases 9-18)

#### UC-9: Analyze a CSV File
```
1. Upload your CSV: curl -X POST http://localhost:8080/upload -F "file=@sales.csv"
2. Chat: "Analyze the sales.csv file I just uploaded"
3. AI reads the file and provides insights
```

#### UC-10: Query Episodic Memory
```
You: What happened in my last 5 interactions?
AI: Based on your episodic memory...
```

SQL equivalent:
```sql
SELECT timestamp, actor, action_type, content
FROM memory_episodic
WHERE tenant_id = 'system'
ORDER BY timestamp DESC LIMIT 5;
```

#### UC-11: Count Events by Type
```
You: How many different types of events are in my memory?
```

SQL equivalent:
```sql
SELECT action_type, COUNT(*) as count
FROM memory_episodic
GROUP BY action_type
ORDER BY count DESC;
```

#### UC-12: Find Events in a Time Range
```
You: What happened between 2pm and 3pm today?
```

SQL equivalent:
```sql
SELECT * FROM memory_episodic
WHERE timestamp BETWEEN '2026-02-07 14:00:00' AND '2026-02-07 15:00:00'
ORDER BY timestamp;
```

#### UC-13: Search Knowledge Base
```
You: Search my semantic memory for anything about "revenue"
```

SQL equivalent:
```sql
SELECT doc_id, content, source_uri
FROM memory_semantic
WHERE content LIKE '%revenue%';
```

#### UC-14: Create a Custom Table
```sql
-- Connect via MySQL client to port 9055
USE antigravity;

CREATE TABLE customer_feedback (
    feedback_id BIGINT,
    date DATE,
    customer VARCHAR(128),
    rating INT,
    comment TEXT,
    category VARCHAR(64)
) ENGINE=OLAP
PRIMARY KEY (feedback_id)
DISTRIBUTED BY HASH(feedback_id);

INSERT INTO customer_feedback VALUES
(1, '2026-01-10', 'Acme Corp', 5, 'Excellent service', 'support'),
(2, '2026-01-11', 'Globex', 3, 'Average experience', 'product'),
(3, '2026-01-12', 'Initech', 4, 'Good but room for improvement', 'onboarding');
```

#### UC-15: Cross-Table Analysis
```sql
-- Join episodic memory with custom table
SELECT e.timestamp, e.content, c.rating
FROM memory_episodic e
JOIN customer_feedback c ON e.content LIKE CONCAT('%', c.customer, '%')
ORDER BY e.timestamp DESC;
```

#### UC-16: Aggregate Statistics
```
You: Give me daily counts of my interactions this week
```

SQL equivalent:
```sql
SELECT DATE(timestamp) as day, COUNT(*) as events
FROM memory_episodic
WHERE timestamp > DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY DATE(timestamp)
ORDER BY day;
```

#### UC-17: Export Query Results
```
You: Export my episodic memory to CSV format
```

Via MySQL client:
```sql
SELECT timestamp, actor, action_type, content
FROM memory_episodic
ORDER BY timestamp DESC
INTO OUTFILE '/tmp/memory_export.csv'
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n';
```

Or use the Trace Viewer's **"Download as CSV"** button.

#### UC-18: Data Quality Check
```
You: Are there any null or empty values in my episodic memory?
```

SQL equivalent:
```sql
SELECT
    SUM(CASE WHEN content IS NULL OR content = '' THEN 1 ELSE 0 END) as empty_content,
    SUM(CASE WHEN actor IS NULL THEN 1 ELSE 0 END) as null_actors,
    SUM(CASE WHEN session_id IS NULL THEN 1 ELSE 0 END) as null_sessions
FROM memory_episodic;
```

---

### Category 3: File Management (Use Cases 19-26)

#### UC-19: List All Available Files
```
You: What files do I have available?
```

API:
```bash
curl http://localhost:8080/tools
# Shows filesystem tools available
```

#### UC-20: Search for Specific Files
```
You: Find all files with "report" in the name
```

#### UC-21: Read a JSON File
```
You: Show me the contents of config.json
```

#### UC-22: Get File Metadata
```
You: How big is the data.csv file and when was it last modified?
```

#### UC-23: Upload Multiple Files
```bash
# Upload multiple files in sequence
for file in sales.csv marketing.csv finance.csv; do
  curl -X POST http://localhost:8080/upload \
    -F "file=@$file" \
    -H "x-tenant-id: quarterly-report"
done
```

#### UC-24: Organize Files by Tenant
```bash
# Marketing files
curl -X POST http://localhost:8080/upload \
  -F "file=@campaign.csv" -H "x-tenant-id: marketing"

# Finance files
curl -X POST http://localhost:8080/upload \
  -F "file=@budget.csv" -H "x-tenant-id: finance"

# Each tenant's files are isolated in S3
```

#### UC-25: Store Artifacts in S3
```python
import boto3

s3 = boto3.client("s3",
    endpoint_url="http://localhost:8455",
    aws_access_key_id="admin",
    aws_secret_access_key="admin"
)

# Store a processed result
s3.put_object(
    Bucket="antigravity",
    Key="results/analysis-2026-02-07.json",
    Body=b'{"revenue": 1500000, "growth": 0.15}'
)
```

#### UC-26: Download Artifacts
```python
# Retrieve stored results
obj = s3.get_object(Bucket="antigravity", Key="results/analysis-2026-02-07.json")
data = obj["Body"].read().decode()
print(data)
```

---

### Category 4: Monitoring & Observability (Use Cases 27-34)

#### UC-27: Check System Health
```bash
curl http://localhost:8080/health | python -m json.tool
```

Or in chat:
```
You: Is the system healthy?
```

#### UC-28: View Container Logs
```bash
# All logs
docker compose logs -f --tail=50

# Specific service
docker compose logs orchestrator --tail=100
docker compose logs starrocks --tail=50
docker compose logs mcp-starrocks --tail=50
```

#### UC-29: Search Logs in Grafana
1. Open http://localhost:3055
2. Go to **Explore**
3. Select **Loki** datasource
4. Query: `{container="antigravity_brain"} |= "ERROR"`

#### UC-30: Monitor God Mode Loop
```bash
docker compose logs orchestrator | grep "GOD MODE"
```

In Grafana:
```logql
{container="antigravity_brain"} |= "GOD MODE LOOP"
```

#### UC-31: Track Memory Usage
In Grafana Explore with Loki:
```logql
{container="antigravity_brain"} |= "memory saved"
```

#### UC-32: Monitor API Budget
```bash
curl http://localhost:4055/health/readiness
```

#### UC-33: View Thought Trace Timeline
1. Open http://localhost:8655
2. Set filters in sidebar
3. Scroll the interactive Plotly timeline
4. Hover over events for details

#### UC-34: Export Thought Trace
1. Open http://localhost:8655
2. Apply desired filters
3. Click **"Download as CSV"**
4. Open in Excel/Google Sheets

---

### Category 5: Multi-Tenant Workflows (Use Cases 35-40)

#### UC-35: Create Isolated Project Spaces
```bash
# Project Alpha
curl -X POST http://localhost:8080/task \
  -H "x-tenant-id: project-alpha" \
  -d '{"goal": "Initialize project workspace", "context": "Marketing campaign Q2"}'

# Project Beta
curl -X POST http://localhost:8080/task \
  -H "x-tenant-id: project-beta" \
  -d '{"goal": "Initialize project workspace", "context": "Product launch"}'
```

#### UC-36: Upload Data Per Tenant
```bash
# Marketing team uploads
curl -X POST http://localhost:8080/upload \
  -F "file=@campaign_metrics.csv" \
  -H "x-tenant-id: marketing"

# Sales team uploads
curl -X POST http://localhost:8080/upload \
  -F "file=@pipeline.csv" \
  -H "x-tenant-id: sales"
```

#### UC-37: Query Tenant-Specific Memory
```sql
-- Only marketing events
SELECT * FROM memory_episodic
WHERE tenant_id = 'marketing'
ORDER BY timestamp DESC LIMIT 10;

-- Only sales events
SELECT * FROM memory_episodic
WHERE tenant_id = 'sales'
ORDER BY timestamp DESC LIMIT 10;
```

#### UC-38: Cross-Tenant Analysis (Admin)
```sql
-- Compare activity across tenants
SELECT tenant_id, COUNT(*) as events, MAX(timestamp) as last_active
FROM memory_episodic
GROUP BY tenant_id
ORDER BY events DESC;
```

#### UC-39: Agent Handoff Between Tenants
```bash
curl -X POST http://localhost:8080/handoff \
  -H "Content-Type: application/json" \
  -H "x-tenant-id: marketing" \
  -d '{
    "target_agent": "sales-agent",
    "payload": {"lead": "Acme Corp", "score": 85}
  }'
```

#### UC-40: Tenant Cleanup
```sql
-- Delete all events for a specific tenant (use MySQL client)
DELETE FROM memory_episodic WHERE tenant_id = 'old-project';
```

---

### Category 6: Integration & Automation (Use Cases 41-48)

#### UC-41: Build a Python Client
```python
import requests

class AntigravityClient:
    def __init__(self, base_url="http://localhost:8080", tenant="system"):
        self.base_url = base_url
        self.tenant = tenant
        self.headers = {"x-tenant-id": tenant, "Content-Type": "application/json"}

    def chat(self, message, model="gpt-4o"):
        resp = requests.post(f"{self.base_url}/v1/chat/completions",
            headers=self.headers,
            json={"model": model, "messages": [{"role": "user", "content": message}]}
        )
        return resp.json()["choices"][0]["message"]["content"]

    def upload(self, filepath):
        with open(filepath, "rb") as f:
            resp = requests.post(f"{self.base_url}/upload",
                headers={"x-tenant-id": self.tenant},
                files={"file": f}
            )
        return resp.json()

    def health(self):
        return requests.get(f"{self.base_url}/health").json()

# Usage
client = AntigravityClient(tenant="my-project")
print(client.chat("What files do I have?"))
print(client.upload("data.csv"))
print(client.health())
```

#### UC-42: Use with curl Scripts
```bash
#!/bin/bash
# Daily data ingestion script

ENDPOINT="http://localhost:8080"
TENANT="daily-pipeline"

# 1. Upload today's data
curl -X POST "$ENDPOINT/upload" \
  -F "file=@/data/daily_$(date +%Y%m%d).csv" \
  -H "x-tenant-id: $TENANT"

# 2. Ask agent to analyze
curl -X POST "$ENDPOINT/task" \
  -H "Content-Type: application/json" \
  -H "x-tenant-id: $TENANT" \
  -d "{\"goal\": \"Analyze today's data and compare to yesterday\"}"

# 3. Check health
curl "$ENDPOINT/health"
```

#### UC-43: OpenAI SDK Compatibility
```python
from openai import OpenAI

# Point OpenAI client at Antigravity Node
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # Auth handled by LiteLLM
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Analyze my last 5 sessions"}]
)

print(response.choices[0].message.content)
```

#### UC-44: Scheduled Ingestion (cron)
```bash
# Add to crontab: run every hour
0 * * * * curl -X POST http://localhost:8080/task \
  -H "Content-Type: application/json" \
  -H "x-tenant-id: hourly-check" \
  -d '{"goal": "Run hourly health check and record results"}'
```

#### UC-45: Webhook Integration
```bash
# Send Argo workflow completion to a Slack webhook
curl -X POST http://localhost:8080/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "data-pipeline-abc123",
    "status": "Succeeded",
    "message": "Pipeline completed in 45s"
  }'
```

#### UC-46: Batch File Processing
```bash
#!/bin/bash
# Upload and process all CSVs in a directory

for file in /data/batch/*.csv; do
  echo "Uploading: $file"
  curl -s -X POST http://localhost:8080/upload \
    -F "file=@$file" \
    -H "x-tenant-id: batch-$(date +%Y%m%d)"

  sleep 1  # Rate limit
done

echo "All files uploaded. Asking agent to process..."
curl -X POST http://localhost:8080/task \
  -H "Content-Type: application/json" \
  -H "x-tenant-id: batch-$(date +%Y%m%d)" \
  -d '{"goal": "Process all uploaded CSV files and create a summary report"}'
```

#### UC-47: Connect External MCP Clients
```bash
# Claude Desktop can connect to the MCP Gateway
# Add to Claude Desktop's MCP config:
# {
#   "servers": {
#     "antigravity": {
#       "transport": "sse",
#       "url": "http://localhost:7755/sse"
#     }
#   }
# }
```

#### UC-48: Stream Logs to External System
```bash
# Forward Loki logs to an external system
curl "http://localhost:3155/loki/api/v1/query_range" \
  --data-urlencode 'query={container="antigravity_brain"}' \
  --data-urlencode "start=$(date -d '1 hour ago' +%s)000000000" \
  --data-urlencode "end=$(date +%s)000000000" \
  --data-urlencode "limit=100"
```

---

### Category 7: Advanced Workflows (Use Cases 49-55)

#### UC-49: Multi-Step Research Workflow
```
You: I need to research competitive pricing for Widget A.
     Step 1: Check our historical sales data
     Step 2: Analyze pricing trends
     Step 3: Generate a pricing recommendation report

AI: I'll work through this step by step...
    Step 1: Checking episodic memory for Widget A sales...
    Step 2: Analyzing the data I found...
    Step 3: Here's my recommendation report...
```

#### UC-50: Data Pipeline with Memory
```bash
# Step 1: Upload raw data
curl -X POST http://localhost:8080/upload \
  -F "file=@raw_sales.csv" -H "x-tenant-id: etl-pipeline"

# Step 2: Ask agent to transform
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-tenant-id: etl-pipeline" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Read raw_sales.csv, clean the data, and create a SQL INSERT statement for a StarRocks table"}]
  }'

# Step 3: The agent remembers what it did (check trace viewer)
```

#### UC-51: Self-Healing Workflow
```
Scenario: Milvus crashes during a vector search

1. Agent tries: search_vectors("revenue projections")
2. Milvus returns connection error
3. Agent's self-healing activates:
   a. Triggers restart-milvus Argo workflow
   b. Waits 10 seconds
   c. Retries the search
4. Search succeeds on retry
5. All logged in episodic memory
```

#### UC-52: Knowledge Base Building
```bash
# Upload multiple documents to build a knowledge base
for doc in company_handbook.pdf product_specs.pdf pricing_guide.pdf; do
  curl -X POST http://localhost:8080/upload \
    -F "file=@$doc" \
    -H "x-tenant-id: knowledge-base"
done

# Then ask questions against the knowledge
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-tenant-id: knowledge-base" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Based on the uploaded documents, what is our return policy?"}]
  }'
```

#### UC-53: Audit Trail Query
```sql
-- Complete audit trail for a tenant
SELECT
    timestamp,
    actor,
    action_type,
    SUBSTRING(content, 1, 100) as content_preview,
    session_id
FROM memory_episodic
WHERE tenant_id = 'finance'
ORDER BY timestamp DESC;
```

#### UC-54: Session Replay
```sql
-- Replay an entire conversation session
SELECT timestamp, actor, action_type, content
FROM memory_episodic
WHERE session_id = 'abc123'
ORDER BY timestamp ASC;
```

#### UC-55: Performance Monitoring Dashboard
Create in Grafana:
1. Panel 1: Events per minute (Loki query)
2. Panel 2: Active tenants (StarRocks query via API)
3. Panel 3: Error rate (Loki `|= "ERROR"` count)
4. Panel 4: API response times
5. Panel 5: Memory table sizes

---

## 22. Troubleshooting

### Common Issues

#### "Cannot connect to Docker daemon"
```bash
# Make sure Docker Desktop is running
# On Windows: Check system tray for Docker icon
# On macOS: Open Docker Desktop app
docker info  # Should show system info
```

#### "Port already in use"
```bash
# Find what's using the port
# Windows:
netstat -ano | findstr :8080

# macOS/Linux:
lsof -i :8080

# Kill the process or change the port in docker-compose.yml
```

#### "SeaweedFS: no free volumes left"
This was fixed in v13.0. If it recurs:
```yaml
# In docker-compose.yml, increase volume.max:
seaweedfs:
  command: "server -s3 ... -volume.max=50 ..."
```
Then: `docker compose restart seaweedfs`

#### "MCP tool server 421 Misdirected Request"
Fixed in v13.0. If it recurs, verify DNS rebinding protection is disabled in the MCP server files.

#### Container won't start
```bash
# Check why it failed
docker compose logs <service-name>

# Force recreate
docker compose up -d --force-recreate <service-name>
```

#### StarRocks tables don't exist
```bash
# Manually run the init SQL
docker compose exec starrocks mysql -h127.0.0.1 -P9030 -uroot < config/starrocks/init-memory.sql
```

#### Chat returns "Budget exhausted"
Wait for the 24-hour reset, or increase the budget:
```yaml
# In docker-compose.yml:
litellm:
  environment:
    - LITELLM_BUDGET_MAX=25.00  # Increase to $25/day
```

#### K3D shows unhealthy
K3D requires Docker-in-Docker privileges. On some systems:
```bash
# Restart K3D
docker compose restart k3d

# Check K3D logs
docker compose logs k3d --tail=50
```

### Reset Everything

```bash
# Stop all containers
docker compose down

# Remove all data (DESTRUCTIVE!)
docker compose down -v

# Fresh start
docker compose up -d
```

### Partial Reset (Keep Data)

```bash
# Restart all containers but keep volumes
docker compose down
docker compose up -d
```

---

## 23. Architecture Reference

### System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    MASTER UI (port 1055)                     │
│    ┌──────┐  ┌──────┐  ┌───────┐  ┌───────┐  ┌──────┐     │
│    │ Chat │  │ Argo │  │Lineage│  │Grafana│  │Trace │     │
│    │ 3355 │  │ 2755 │  │ 5155  │  │ 3055  │  │ 8655 │     │
│    └──┬───┘  └──────┘  └───────┘  └───┬───┘  └──┬───┘     │
│       │                               │          │          │
│  ┌────▼────────────────────────────────▼──────────▼────┐    │
│  │              ORCHESTRATOR (port 8080/8081)           │    │
│  │         FastAPI (A2A) + gRPC (SuperBuilder)         │    │
│  │                  + God Mode Loop                     │    │
│  └────┬──────────────┬──────────────┬─────────────┬────┘    │
│       │              │              │             │          │
│  ┌────▼────┐   ┌─────▼─────┐  ┌────▼────┐  ┌────▼────┐    │
│  │ LiteLLM │   │ StarRocks │  │ Milvus  │  │SeaweedFS│    │
│  │  $10/d  │   │  Memory   │  │ Vectors │  │   S3    │    │
│  │  4055   │   │ 9055/8055 │  │  19555  │  │8455/9355│    │
│  └─────────┘   └───────────┘  └─────────┘  └─────────┘    │
│                                                             │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌────────────┐   │
│  │ Marquez │  │ Keycloak │  │ OpenBao │  │   Valkey   │   │
│  │  5055   │  │   8355   │  │  8255   │  │    6355    │   │
│  └─────────┘  └──────────┘  └─────────┘  └────────────┘   │
│                                                             │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌────────────┐   │
│  │  NATS   │  │  SPIRE   │  │  OVMS   │  │   Falco    │   │
│  │  4255   │  │   8155   │  │  8555   │  │ (monitor)  │   │
│  └─────────┘  └──────────┘  └─────────┘  └────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Memory Architecture

```
┌──────────────────────────────────────────────┐
│              AGENT MEMORY SYSTEM              │
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │    EPISODIC — "What happened?"         │  │
│  │    StarRocks: memory_episodic          │  │
│  │    Events, conversations, actions      │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │    SEMANTIC — "What do I know?"        │  │
│  │    StarRocks: memory_semantic          │  │
│  │    Document chunks, knowledge base     │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │    PROCEDURAL — "How do I do it?"      │  │
│  │    StarRocks: memory_procedural        │  │
│  │    Skills, workflows, success rates    │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │    VECTOR — Semantic Search            │  │
│  │    Milvus: embeddings for similarity   │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
```

---

## 24. Port Reference

| Port | Service | Protocol | Purpose |
|------|---------|----------|---------|
| **1055** | Master UI | HTTP | Unified portal (start here!) |
| 2755 | Argo UI | HTTP | Workflow visualization |
| **3055** | Grafana | HTTP | Dashboards & log viewer |
| 3155 | Loki | HTTP | Log aggregation API |
| **3355** | Open WebUI | HTTP | Chat interface |
| 4055 | LiteLLM | HTTP | LLM API proxy & budget |
| 4255 | NATS | TCP | JetStream messaging |
| 5055 | Marquez API | HTTP | Lineage REST API |
| **5155** | Marquez UI | HTTP | Lineage web interface |
| 5455 | PostgreSQL | TCP | Shared database |
| 6355 | Valkey | TCP | In-memory cache |
| 6755 | K3D API | HTTPS | Kubernetes API |
| 7755 | MCP Gateway | SSE | MCP tool router |
| **8055** | StarRocks HTTP | HTTP | StarRocks management |
| **8080** | Orchestrator | HTTP | A2A REST API |
| 8081 | Orchestrator | gRPC | Intel SuperBuilder |
| 8155 | SPIRE | gRPC | mTLS identity |
| 8255 | OpenBao | HTTP | Secrets vault |
| 8355 | Keycloak | HTTP | IAM / SSO |
| 8455 | SeaweedFS S3 | HTTP | Object storage |
| 8555 | OVMS REST | HTTP | ML inference |
| **8655** | Trace Viewer | HTTP | Streamlit thought trace |
| **9055** | StarRocks FE | MySQL | SQL query interface |
| 9155 | Milvus Metrics | HTTP | Vector DB metrics |
| 9255 | OVMS gRPC | gRPC | ML inference |
| 9355 | SeaweedFS Master | HTTP | Storage cluster mgmt |
| 19555 | Milvus gRPC | gRPC | Vector DB operations |

**Bold** = Most commonly used ports for human interaction.

---

## Quick Start Cheat Sheet

```bash
# Start everything
docker compose up -d

# Open the portal
open http://localhost:1055

# Chat with the agent
open http://localhost:3355

# Upload a file
curl -X POST http://localhost:8080/upload -F "file=@mydata.csv"

# Check health
curl http://localhost:8080/health

# Query memory (MySQL client)
mysql -h 127.0.0.1 -P 9055 -u root -e "SELECT * FROM antigravity.memory_episodic ORDER BY timestamp DESC LIMIT 10"

# View logs
docker compose logs -f orchestrator

# Stop everything
docker compose down

# Nuclear reset (deletes all data!)
docker compose down -v
```

---

*Antigravity Node v13.0 "The God Node" — Built with 100% open-source software (Apache-2.0, MIT, BSD-3, MPL-2.0)*

*Version: 13.0.0 | Last Updated: February 2026*
