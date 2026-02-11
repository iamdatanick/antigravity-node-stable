---
name: deploy-validate
description: Run all v14.1 Phoenix validation gates before push
disable-model-invocation: true
---

# Deploy Validate

Run ALL validation gates for Antigravity Node v14.1 Phoenix. This is the pre-push safety net. Execute gates in order, report PASS/FAIL for each.

## Gate 1: Validation Gates Test Suite (VG-101 to VG-109)

```
cd C:\Users\NickV\OneDrive\Desktop\Antigravity-Node
python -m pytest tests/test_validation_gates.py -v --tb=short
```

Covers: AVX-512 script, Ceph volumes, OpenBao healthcheck, etcd persistence, banned packages, AVX guard.

## Gate 2: Docker Compose YAML Syntax

```
python -c "import yaml; yaml.safe_load(open('deployment/cloud-test/docker-compose.yml')); print('COMPOSE: PASS')"
```

## Gate 3: Terraform Validate

```
terraform -chdir=deployment/terraform validate
```

If terraform not installed, report SKIP (not FAIL).

## Gate 4: Python Lint

```
python -m ruff check workflows/ src/orchestrator/ --select E,F,W,I,N,UP,B,SIM
```

## Gate 5: Unit Tests

```
python -m pytest tests/test_engine.py tests/test_cloud_scripts.py -v --tb=short
```

## Gate 6: Secrets Scan

Scan all staged files for patterns: `sk-ant-`, `AKIA`, `ghp_`, `eyJ`, `xoxb-`

```
git -C "C:\Users\NickV\OneDrive\Desktop\Antigravity-Node" diff --cached -- "*.py" "*.yml" "*.tf" "*.txt" "*.sh"
```

Search output for secret patterns. If ANY match: FAIL.

## Gate 7: Truncation Scan

Search `src/orchestrator/` and `deployment/` for placeholder patterns:
- `# TODO: implement`
- `# ... remaining`
- `pass  # TODO`
- `raise NotImplementedError`

If found: FAIL — complete the implementation first.

## Output

```
========================================
  DEPLOY VALIDATE — Phoenix v14.1
========================================
Gate 1 (VG-101-109):  PASS / FAIL
Gate 2 (Compose):     PASS / FAIL
Gate 3 (Terraform):   PASS / SKIP
Gate 4 (Lint):        PASS / FAIL
Gate 5 (Unit Tests):  PASS / FAIL
Gate 6 (Secrets):     PASS / FAIL
Gate 7 (Truncation):  PASS / FAIL

OVERALL: X/7 PASS
========================================
```

If ANY gate fails, list specific failures with file paths and line numbers. Do NOT suggest skipping or disabling gates. Fix the issue.
