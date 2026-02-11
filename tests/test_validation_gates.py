"""Validation Gates — Acceptance tests for Antigravity Node v14.1 (VG-101 to VG-109).

These tests validate the deployment spec, not live services.
Live integration tests run on the GCP instance.
"""
import os

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLOUD_DIR = os.path.join(PROJECT_ROOT, "deployment", "cloud-test")


class TestVG101HardwareCompat:
    """VG-101: Hardware Compatibility."""

    def test_check_avx512_script_exists(self):
        script = os.path.join(CLOUD_DIR, "scripts", "check_avx512.sh")
        assert os.path.exists(script), "check_avx512.sh must exist"

    def test_check_avx512_is_executable_content(self):
        script = os.path.join(CLOUD_DIR, "scripts", "check_avx512.sh")
        with open(script) as f:
            content = f.read()
        assert "avx512" in content, "Script must check for avx512"
        assert "exit 1" in content, "Script must fail-fast on missing AVX-512"
        assert "FORCE_NO_AVX" in content, "Script must support override"


class TestVG104DataSovereignty:
    """VG-104: Ceph S3 bucket isolation."""

    def test_compose_has_ceph_volumes(self):
        compose = os.path.join(CLOUD_DIR, "docker-compose.yml")
        with open(compose) as f:
            content = f.read()
        assert "./data/ceph:/var/lib/ceph" in content, "Ceph data volume required"
        assert "./data/ceph_conf:/etc/ceph" in content, "Ceph config volume required (keyring persistence)"

    def test_setup_tenants_creates_tenant_bucket(self):
        script = os.path.join(CLOUD_DIR, "scripts", "setup_tenants.sh")
        with open(script) as f:
            content = f.read()
        assert "tenant-a" in content, "Must create tenant-a bucket"


class TestVG105Security:
    """VG-105: OpenBao secrets."""

    def test_compose_has_openbao_healthcheck(self):
        compose_path = os.path.join(CLOUD_DIR, "docker-compose.yml")
        with open(compose_path) as f:
            compose = yaml.safe_load(f)
        openbao = compose["services"]["openbao"]
        assert "healthcheck" in openbao, "OpenBao must have healthcheck"
        assert "IPC_LOCK" in openbao.get("cap_add", []), "OpenBao needs IPC_LOCK"


class TestVG106Resilience:
    """VG-106: Data persists across restart."""

    def test_etcd_has_data_dir_flag(self):
        compose_path = os.path.join(CLOUD_DIR, "docker-compose.yml")
        with open(compose_path) as f:
            compose = yaml.safe_load(f)
        etcd_cmd = compose["services"]["etcd"]["command"]
        assert "--data-dir=/etcd-data" in etcd_cmd, "Etcd must use --data-dir flag"

    def test_etcd_has_volume(self):
        compose_path = os.path.join(CLOUD_DIR, "docker-compose.yml")
        with open(compose_path) as f:
            compose = yaml.safe_load(f)
        volumes = compose["services"]["etcd"]["volumes"]
        assert "./data/etcd:/etcd-data" in volumes, "Etcd must have persistent volume"

    def test_all_services_restart_policy(self):
        compose_path = os.path.join(CLOUD_DIR, "docker-compose.yml")
        with open(compose_path) as f:
            compose = yaml.safe_load(f)
        for name, svc in compose["services"].items():
            if name == "ui":
                continue  # UI doesn't need restart policy
            assert svc.get("restart") == "unless-stopped", (
                f"Service {name} must have restart: unless-stopped"
            )


class TestVG108DependencySmoke:
    """VG-108: Dependency Smoke Test."""

    def test_requirements_no_banned_packages(self):
        req_path = os.path.join(CLOUD_DIR, "requirements.txt")
        with open(req_path) as f:
            content = f.read().lower()
        banned = ["asyncpg", "pymilvus", "nats-py", "psycopg2"]
        for pkg in banned:
            assert pkg not in content, f"Banned package {pkg} found in cloud requirements"

    def test_requirements_has_etcd3(self):
        req_path = os.path.join(CLOUD_DIR, "requirements.txt")
        with open(req_path) as f:
            content = f.read()
        assert "python-etcd3" in content, "python-etcd3 required (not etcd3)"


class TestVG109AVXGuard:
    """VG-109: AVX-512 guard fails on non-compatible hardware."""

    def test_script_exits_nonzero_without_avx(self):
        script = os.path.join(CLOUD_DIR, "scripts", "check_avx512.sh")
        with open(script) as f:
            content = f.read()
        assert "exit 1" in content, "Must exit 1 when AVX-512 missing"


class TestComposeHealthchecks:
    """CG-105: All critical services must have healthchecks."""

    def test_critical_services_have_healthchecks(self):
        compose_path = os.path.join(CLOUD_DIR, "docker-compose.yml")
        with open(compose_path) as f:
            compose = yaml.safe_load(f)
        critical = ["etcd", "ceph-demo", "openbao", "ovms", "orchestrator"]
        for name in critical:
            assert name in compose["services"], f"Service {name} missing from compose"
            svc = compose["services"][name]
            assert "healthcheck" in svc, f"Service {name} must have a healthcheck"


class TestComposeDependsOn:
    """Boot order: orchestrator waits for all infra to be healthy."""

    def test_orchestrator_depends_on_healthy_infra(self):
        compose_path = os.path.join(CLOUD_DIR, "docker-compose.yml")
        with open(compose_path) as f:
            compose = yaml.safe_load(f)
        deps = compose["services"]["orchestrator"]["depends_on"]
        for svc in ["etcd", "ceph-demo", "ovms", "openbao"]:
            assert svc in deps, f"Orchestrator must depend on {svc}"
            assert deps[svc]["condition"] == "service_healthy", (
                f"Orchestrator must wait for {svc} to be healthy"
            )
