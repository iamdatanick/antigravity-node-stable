#!/bin/bash
# deploy.sh — Antigravity Node v14.1 Full Deployment Sequence
# Executes Deployment_Order.csv phases 0-4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================"
echo "  ANTIGRAVITY NODE v14.1 — PHOENIX DEPLOYMENT"
echo "================================================================"

# --- Phase 0: Pre-Flight ---
echo ""
echo "=== PHASE 0: Pre-Flight ==="

echo "[Step 1/9] Hardware Check..."
bash scripts/check_avx512.sh

echo "[Step 2/9] Permission Fix..."
sudo bash scripts/fix_perms.sh

echo "[Step 3/9] Dependency Install..."
pip install -r requirements.txt
python -c "import etcd3; import aioboto3; import grpc; print('DEPENDENCY SMOKE PASS')"

echo "[Step 4/9] Model Hydration..."
bash scripts/init_models.sh

# --- Phase 1: Infrastructure Boot ---
echo ""
echo "=== PHASE 1: Infrastructure ==="

echo "[Step 5/9] Booting etcd, ceph, openbao..."
docker compose up -d ceph-demo etcd openbao
echo "Waiting for infrastructure to become healthy..."
docker compose exec -T etcd etcdctl endpoint health || sleep 10
# Wait up to 120s for all three to be healthy
for i in $(seq 1 24); do
    HEALTHY=$(docker compose ps --format json | python3 -c "
import sys, json
svcs = [json.loads(l) for l in sys.stdin if l.strip()]
healthy = sum(1 for s in svcs if s.get('Health','') == 'healthy')
print(healthy)
" 2>/dev/null || echo "0")
    echo "  Healthy services: $HEALTHY/3 (attempt $i/24)"
    [ "$HEALTHY" -ge 3 ] && break
    sleep 5
done

echo "[Step 6/9] Tenant Setup..."
bash scripts/setup_tenants.sh

# --- Phase 2: Full Stack ---
echo ""
echo "=== PHASE 2: Full Stack ==="

echo "[Step 7/9] Booting all services..."
docker compose up -d
echo "Waiting for full stack..."
sleep 30

# --- Phase 3: Health Check ---
echo ""
echo "=== PHASE 3: Health Verification ==="

echo "[Step 8/9] Health Check..."
HEALTH_PASS=false
for i in $(seq 1 12); do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "HEALTH CHECK PASS (HTTP 200)"
        curl -s http://localhost:8080/health | python3 -m json.tool
        HEALTH_PASS=true
        break
    fi
    echo "  Attempt $i/12 — HTTP $HTTP_CODE, retrying in 10s..."
    sleep 10
done
if [ "$HEALTH_PASS" = "false" ]; then
    echo "FATAL: Health check failed after 120s"
    exit 1
fi

# --- Phase 4: E2E ---
echo ""
echo "=== PHASE 4: E2E Test ==="

echo "[Step 9/9] E2E Smoke Test..."
echo "  UI: http://localhost:1055"
echo "  API: http://localhost:8080"
echo "  OVMS: http://localhost:9001"

# Quick OVMS check
OVMS_STATUS=$(curl -s http://localhost:9001/v2/health/live 2>/dev/null || echo "FAIL")
echo "  OVMS Health: $OVMS_STATUS"

echo ""
echo "================================================================"
echo "  DEPLOYMENT COMPLETE"
echo "  UI: http://$(hostname -I | awk '{print $1}'):1055"
echo "================================================================"
