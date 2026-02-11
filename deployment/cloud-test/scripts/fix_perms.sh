#!/bin/bash
# fix_perms.sh — Pre-boot permission fix for container data directories
# CG-108: Solves Permission Drift between host and container UIDs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${BASE_DIR}/data"

echo "=== Permission Fix (CG-108) ==="
echo "Base directory: $BASE_DIR"
echo "Data directory: $DATA_DIR"

# Ensure directories exist
mkdir -p "$DATA_DIR/ceph" "$DATA_DIR/ceph_conf" "$DATA_DIR/etcd"

# Ceph runs as ceph user (UID 167, GID 167)
echo "Setting Ceph ownership (167:167)..."
chown -R 167:167 "$DATA_DIR/ceph"
chown -R 167:167 "$DATA_DIR/ceph_conf"

# Etcd runs as etcd user (UID 1001, GID 1001 in coreos image)
echo "Setting Etcd ownership (1001:1001)..."
chown -R 1001:1001 "$DATA_DIR/etcd"

echo "PERMISSIONS FIXED"
ls -la "$DATA_DIR/"
