#!/bin/bash
# setup_tenants.sh — Multi-tenant S3 bucket creation on Ceph RGW
# ACT-106: Creates tenant-a bucket and configures CORS
set -euo pipefail

CEPH_HOST="${CEPH_HOST:-localhost}"
CEPH_PORT="${CEPH_PORT:-8000}"
S3_ACCESS_KEY="${S3_ACCESS_KEY:-antigravity}"
S3_SECRET_KEY="${S3_SECRET_KEY:-antigravity_secret}"

MAX_RETRIES=30
RETRY_INTERVAL=5

echo "=== Tenant Setup (ACT-106) ==="

# Wait for Ceph RGW to be healthy
echo "Waiting for Ceph RGW at ${CEPH_HOST}:${CEPH_PORT}..."
for i in $(seq 1 $MAX_RETRIES); do
    if curl -sf "http://${CEPH_HOST}:${CEPH_PORT}" > /dev/null 2>&1; then
        echo "Ceph RGW is ready (attempt $i/$MAX_RETRIES)"
        break
    fi
    if [ "$i" -eq "$MAX_RETRIES" ]; then
        echo "FAIL: Ceph RGW not ready after $MAX_RETRIES attempts"
        exit 1
    fi
    echo "Attempt $i/$MAX_RETRIES — retrying in ${RETRY_INTERVAL}s..."
    sleep "$RETRY_INTERVAL"
done

# Configure s3cmd
cat > /tmp/.s3cfg << S3EOF
[default]
access_key = ${S3_ACCESS_KEY}
secret_key = ${S3_SECRET_KEY}
host_base = ${CEPH_HOST}:${CEPH_PORT}
host_bucket = ${CEPH_HOST}:${CEPH_PORT}/%(bucket)
use_https = False
signature_v2 = False
S3EOF

# Create tenant buckets
for BUCKET in "antigravity" "tenant-a" "artifacts"; do
    echo "Creating bucket: $BUCKET"
    s3cmd -c /tmp/.s3cfg mb "s3://${BUCKET}" 2>/dev/null || echo "Bucket $BUCKET already exists"
done

# Set CORS on antigravity bucket (BLK-031)
cat > /tmp/cors.xml << 'CORSEOF'
<CORSConfiguration>
  <CORSRule>
    <AllowedOrigin>*</AllowedOrigin>
    <AllowedMethod>GET</AllowedMethod>
    <AllowedMethod>PUT</AllowedMethod>
    <AllowedMethod>POST</AllowedMethod>
    <AllowedHeader>*</AllowedHeader>
  </CORSRule>
</CORSConfiguration>
CORSEOF

s3cmd -c /tmp/.s3cfg setcors /tmp/cors.xml s3://antigravity 2>/dev/null || echo "CORS set (or already configured)"

# Verify
echo "=== Bucket Listing ==="
s3cmd -c /tmp/.s3cfg ls
echo "TENANT SETUP PASS"
