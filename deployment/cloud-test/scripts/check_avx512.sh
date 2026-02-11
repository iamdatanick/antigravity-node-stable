#!/bin/bash
# check_avx512.sh — Fail-fast hardware guard for Antigravity Node v14.1
# CG-107: Exits 1 if AVX-512 is not detected (unless FORCE_NO_AVX=1)
set -euo pipefail

echo "=== AVX-512 Hardware Check ==="

if grep -q avx512 /proc/cpuinfo 2>/dev/null; then
    echo "HARDWARE PASS: AVX-512 detected."
    exit 0
fi

# AVX-512 not found
if [ "${FORCE_NO_AVX:-0}" = "1" ]; then
    echo "WARNING: AVX-512 NOT detected. FORCE_NO_AVX=1 override active."
    echo "Proceeding at reduced performance. OVMS will use AVX2 fallback."
    exit 0
fi

echo "HARDWARE FAIL: AVX-512 NOT detected."
echo "This instance does not support AVX-512 instructions."
echo "Set FORCE_NO_AVX=1 to override (not recommended for production)."
exit 1
