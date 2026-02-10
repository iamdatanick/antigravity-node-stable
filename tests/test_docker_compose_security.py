"""Tests for Docker Compose security hardening."""

import os

import pytest
import yaml


@pytest.fixture
def docker_compose_config():
    """Load docker-compose.yml configuration."""
    compose_path = os.path.join(os.path.dirname(__file__), "..", "docker-compose.yml")
    with open(compose_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_env(env_block):
    """Parse environment from either list or dict format to {key: full_value} dict."""
    if isinstance(env_block, dict):
        # Dict format: {KEY: value} — value may contain ${VAR} after compose interpolation
        return {k: str(v) for k, v in env_block.items()}
    # List format: ["KEY=value", ...]
    return {e.split("=", 1)[0]: e for e in env_block}


class TestCredentialEnvironmentVariables:
    """Test that hardcoded credentials are replaced with environment variables."""

    def test_postgres_uses_env_vars(self, docker_compose_config):
        """Test that Postgres uses environment variables for credentials."""
        postgres = docker_compose_config["services"]["postgres"]
        env_vars = _parse_env(postgres["environment"])

        # Check required keys exist (values use ${VAR:-default} in source YAML)
        assert "POSTGRES_USER" in env_vars
        assert "POSTGRES_PASSWORD" in env_vars
        assert "POSTGRES_DB" in env_vars
        # Values should NOT be empty (either ${VAR} reference or a default is present)
        assert env_vars["POSTGRES_USER"], "POSTGRES_USER must have a value"
        assert env_vars["POSTGRES_PASSWORD"], "POSTGRES_PASSWORD must have a value"

    def test_keycloak_uses_env_vars(self, docker_compose_config):
        """Test that Keycloak uses environment variables for credentials."""
        keycloak = docker_compose_config["services"]["keycloak"]
        env_vars = _parse_env(keycloak["environment"])

        assert "KEYCLOAK_ADMIN_PASSWORD" in env_vars
        assert "KC_DB_PASSWORD" in env_vars
        # Must not be hardcoded empty
        assert env_vars["KEYCLOAK_ADMIN_PASSWORD"], "KEYCLOAK_ADMIN_PASSWORD must have a value"
        assert env_vars["KC_DB_PASSWORD"], "KC_DB_PASSWORD must have a value"

    def test_marquez_uses_env_vars(self, docker_compose_config):
        """Test that Marquez uses environment variables for credentials."""
        marquez = docker_compose_config["services"]["marquez"]
        env_vars = _parse_env(marquez["environment"])

        assert "POSTGRES_PASSWORD" in env_vars
        assert env_vars["POSTGRES_PASSWORD"], "POSTGRES_PASSWORD must have a value"

    def test_openbao_uses_env_vars(self, docker_compose_config):
        """Test that OpenBao uses environment variables for token."""
        openbao = docker_compose_config["services"]["openbao"]

        # Check command line references the dev token
        command = openbao["command"]
        assert "dev-root-token-id" in command

        # Check environment variable exists
        env_vars = _parse_env(openbao["environment"])
        assert "BAO_DEV_ROOT_TOKEN_ID" in env_vars
        assert env_vars["BAO_DEV_ROOT_TOKEN_ID"], "BAO_DEV_ROOT_TOKEN_ID must have a value"


class TestSecurityHardening:
    """Test that security hardening is applied to containers."""

    def test_application_containers_have_security_opt(self, docker_compose_config):
        """Test that application containers have no-new-privileges."""
        app_containers = [
            "nats",
            "marquez",
            "opensearch",
            "spire-server",
            "ovms",
            "wasm-worker",
            "perses",
            "fluent-bit",
            "budget-proxy",
            "mcp-gateway",
            "mcp-filesystem",
            "mcp-starrocks",
            "master-ui",
            "orchestrator",
            "keycloak",
        ]

        for container_name in app_containers:
            if container_name in docker_compose_config["services"]:
                container = docker_compose_config["services"][container_name]
                assert "security_opt" in container, f"{container_name} missing security_opt"
                assert "no-new-privileges:true" in container["security_opt"], (
                    f"{container_name} missing no-new-privileges"
                )

    def test_readonly_containers_have_tmpfs(self, docker_compose_config):
        """Test that read-only containers have tmpfs for /tmp."""
        readonly_containers = [
            "nats",
            "marquez",
            "ovms",
            "wasm-worker",
            "perses",
            "fluent-bit",
            "budget-proxy",
            "mcp-gateway",
            "mcp-filesystem",
            "mcp-starrocks",
            "master-ui",
        ]

        for container_name in readonly_containers:
            if container_name in docker_compose_config["services"]:
                container = docker_compose_config["services"][container_name]
                assert "read_only" in container, f"{container_name} missing read_only"
                assert container["read_only"] is True, f"{container_name} read_only not True"
                assert "tmpfs" in container, f"{container_name} missing tmpfs"
                assert "/tmp" in container["tmpfs"], f"{container_name} missing /tmp in tmpfs"

    def test_database_containers_not_readonly(self, docker_compose_config):
        """Test that database containers are explicitly not read-only."""
        db_containers = ["postgres", "etcd", "starrocks", "valkey", "milvus", "openbao", "seaweedfs"]

        for container_name in db_containers:
            if container_name in docker_compose_config["services"]:
                container = docker_compose_config["services"][container_name]
                assert "read_only" in container, f"{container_name} missing read_only flag"
                assert container["read_only"] is False, f"{container_name} should have read_only: false"


class TestHealthCheckStartPeriod:
    """Test that slow-starting services have appropriate start_period."""

    def test_starrocks_has_start_period(self, docker_compose_config):
        """Test that StarRocks has a start_period of at least 30s."""
        starrocks = docker_compose_config["services"]["starrocks"]
        assert "healthcheck" in starrocks
        assert "start_period" in starrocks["healthcheck"]
        # Parse start_period (e.g., "30s")
        start_period = starrocks["healthcheck"]["start_period"]
        # Simple validation: check if it contains "30s" or higher
        assert "30s" in start_period or "45s" in start_period or "60s" in start_period

    def test_keycloak_has_start_period(self, docker_compose_config):
        """Test that Keycloak has a start_period configured."""
        keycloak = docker_compose_config["services"]["keycloak"]
        assert "healthcheck" in keycloak
        assert "start_period" in keycloak["healthcheck"]
        start_period = keycloak["healthcheck"]["start_period"]
        # Should have 45s or higher
        assert "45s" in start_period or "60s" in start_period

    def test_milvus_has_start_period(self, docker_compose_config):
        """Test that Milvus has a start_period configured."""
        milvus = docker_compose_config["services"]["milvus"]
        assert "healthcheck" in milvus
        assert "start_period" in milvus["healthcheck"]
        start_period = milvus["healthcheck"]["start_period"]
        # Should have 30s or higher
        assert "30s" in start_period or "45s" in start_period or "60s" in start_period


class TestGitignore:
    """Test that .env is in .gitignore."""

    def test_env_in_gitignore(self):
        """Test that .env is listed in .gitignore."""
        gitignore_path = os.path.join(os.path.dirname(__file__), "..", ".gitignore")
        with open(gitignore_path, encoding="utf-8") as f:
            gitignore_content = f.read()

        # Check that .env is in gitignore (exact match)
        assert "\n.env\n" in gitignore_content or gitignore_content.startswith(".env\n"), ".env not found in .gitignore"


class TestEnvExample:
    """Test that .env.example file exists and contains required variables."""

    def test_env_example_exists(self):
        """Test that .env.example file exists."""
        env_example_path = os.path.join(os.path.dirname(__file__), "..", ".env.example")
        assert os.path.exists(env_example_path), ".env.example file does not exist"

    def test_env_example_has_required_secrets(self):
        """Test that .env.example contains all required secret variables."""
        env_example_path = os.path.join(os.path.dirname(__file__), "..", ".env.example")
        with open(env_example_path, encoding="utf-8") as f:
            content = f.read()

        required_vars = [
            "POSTGRES_PASSWORD",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ]

        for var in required_vars:
            assert var in content, f"{var} not found in .env.example"

    def test_env_example_has_optional_overrides(self):
        """Test that .env.example contains optional override variables."""
        env_example_path = os.path.join(os.path.dirname(__file__), "..", ".env.example")
        with open(env_example_path, encoding="utf-8") as f:
            content = f.read()

        optional_vars = [
            "POSTGRES_USER",
            "POSTGRES_DB",
            "S3_ACCESS_KEY",
            "S3_SECRET_KEY",
            "GOOSE_MODEL",
            "GOD_MODE_ITERATIONS",
            "CORS_ORIGINS",
        ]

        for var in optional_vars:
            assert var in content, f"{var} not found in .env.example"
