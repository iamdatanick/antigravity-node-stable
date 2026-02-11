---
name: cloud-deploy
description: GCP deployment runbook for Antigravity Node v14.1 Phoenix
disable-model-invocation: true
---

# Cloud Deploy — Phoenix v14.1

Guided deployment to GCP. Walks through Deployment_Order.csv phases 0-4 with verification at each checkpoint.

## Prerequisites

- Terraform provisioned GCP instance (deployment/terraform/)
- SSH access to instance
- Repo cloned to ~/antigravity on the instance
- Working directory: ~/antigravity/deployment/cloud-test/

## Phase 0: Pre-Flight

### Step 1: Hardware Check
```bash
bash scripts/check_avx512.sh
```
MUST see: `HARDWARE PASS: AVX-512 detected.`
If FAIL: Wrong instance type. Need c2-standard-8 (Cascade Lake+).

### Step 2: Permission Fix
```bash
sudo bash scripts/fix_perms.sh
```
MUST see: `PERMISSIONS FIXED` with correct UID ownership.

### Step 3: Dependencies
```bash
pip install -r requirements.txt
python -c "import etcd3; import aioboto3; import grpc; print('DEPENDENCY SMOKE PASS')"
```
MUST see: `DEPENDENCY SMOKE PASS` with no ImportError.

### Step 4: Model Hydration
```bash
bash scripts/init_models.sh
```
MUST see: `MODEL HYDRATION PASS` with TinyLlama size.

**CHECKPOINT: All 4 pre-flight steps passed? Proceed to Phase 1.**

## Phase 1: Infrastructure

### Step 5: Boot Infra
```bash
docker compose up -d ceph-demo etcd openbao
```
Wait for healthy (up to 2 minutes):
```bash
watch -n 5 'docker compose ps'
```
MUST see: All 3 services healthy.

### Step 6: Tenant Setup
```bash
bash scripts/setup_tenants.sh
```
MUST see: `TENANT SETUP PASS` with buckets: antigravity, tenant-a, artifacts.

**CHECKPOINT: 3 infra containers healthy + buckets created.**

## Phase 2: Full Stack

### Step 7: Boot All
```bash
docker compose up -d
```
Wait 30s, then verify:
```bash
docker compose ps
```
MUST see: All containers Up with healthy status.

## Phase 3: Health

### Step 8: Health Check
```bash
curl -s http://localhost:8080/health | python3 -m json.tool
```
MUST see: `{"status": "healthy"}`

**CHECKPOINT: Orchestrator responding. Stack is live.**

## Phase 4: E2E

### Step 9: End-to-End
1. Browser: `http://<INSTANCE_IP>:1055`
2. Send: "Why is the sky blue?"
3. MUST see: Response from local OVMS (no external API calls)

Verify OVMS directly:
```bash
curl -s http://localhost:9001/v2/health/live
```

## Post-Deploy Validation

```bash
# VG-106: Resilience (Restart)
docker compose restart
sleep 30
curl -s http://localhost:8080/health

# VG-107: Resilience (Recreate)
docker compose down && docker compose up -d
sleep 60
curl -s http://localhost:8080/health
```

Both must return healthy with data persisted.

## Rollback

```bash
docker compose down
sudo bash scripts/fix_perms.sh
docker compose up -d ceph-demo etcd openbao  # Restart from Phase 1
```
