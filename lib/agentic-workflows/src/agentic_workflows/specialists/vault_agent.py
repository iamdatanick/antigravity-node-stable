"""HashiCorp Vault specialist agent for secrets management.

Handles secret storage, encryption, and dynamic credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import SpecialistAgent, SpecialistConfig, SpecialistCapability


@dataclass
class VaultConfig(SpecialistConfig):
    """Vault-specific configuration."""

    addr: str = "http://localhost:8200"
    token: str = ""
    namespace: str = ""
    mount_point: str = "secret"


class VaultAgent(SpecialistAgent):
    """Specialist agent for HashiCorp Vault.

    Capabilities:
    - Secret management
    - Encryption/decryption
    - Dynamic credentials
    - Token management
    """

    def __init__(self, config: VaultConfig | None = None, **kwargs):
        self.vault_config = config or VaultConfig()
        super().__init__(config=self.vault_config, **kwargs)

        self._client = None

        self.register_handler("read_secret", self._read_secret)
        self.register_handler("write_secret", self._write_secret)
        self.register_handler("delete_secret", self._delete_secret)
        self.register_handler("list_secrets", self._list_secrets)
        self.register_handler("encrypt", self._encrypt)
        self.register_handler("decrypt", self._decrypt)
        self.register_handler("get_database_credentials", self._get_database_credentials)

    @property
    def capabilities(self) -> list[SpecialistCapability]:
        return [
            SpecialistCapability.SECRET_MANAGEMENT,
            SpecialistCapability.ENCRYPTION,
        ]

    @property
    def service_name(self) -> str:
        return "HashiCorp Vault"

    async def _connect(self) -> None:
        """Connect to Vault."""
        try:
            import hvac

            self._client = hvac.Client(
                url=self.vault_config.addr,
                token=self.vault_config.token,
                namespace=self.vault_config.namespace or None,
            )
        except ImportError:
            self.logger.warning("hvac not installed")

    async def _disconnect(self) -> None:
        """Disconnect from Vault."""
        self._client = None

    async def _health_check(self) -> bool:
        """Check Vault health."""
        if self._client is None:
            return False
        try:
            return self._client.is_authenticated()
        except Exception:
            return False

    async def _read_secret(
        self,
        path: str,
        mount_point: str | None = None,
        version: int | None = None,
    ) -> dict[str, Any]:
        """Read a secret.

        Args:
            path: Secret path.
            mount_point: KV mount point.
            version: Specific version (KV v2).

        Returns:
            Secret data.
        """
        if self._client is None:
            return {"error": "Not connected"}

        mount = mount_point or self.vault_config.mount_point

        try:
            response = self._client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=mount,
                version=version,
            )
            return {
                "data": response.get("data", {}).get("data", {}),
                "metadata": response.get("data", {}).get("metadata", {}),
            }
        except Exception as e:
            return {"error": str(e)}

    async def _write_secret(
        self,
        path: str,
        data: dict[str, Any],
        mount_point: str | None = None,
    ) -> dict[str, Any]:
        """Write a secret.

        Args:
            path: Secret path.
            data: Secret data.
            mount_point: KV mount point.

        Returns:
            Write result.
        """
        if self._client is None:
            return {"error": "Not connected"}

        mount = mount_point or self.vault_config.mount_point

        try:
            response = self._client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=data,
                mount_point=mount,
            )
            return {
                "path": path,
                "version": response.get("data", {}).get("version"),
                "written": True,
            }
        except Exception as e:
            return {"error": str(e)}

    async def _delete_secret(
        self,
        path: str,
        mount_point: str | None = None,
        versions: list[int] | None = None,
    ) -> dict[str, Any]:
        """Delete a secret.

        Args:
            path: Secret path.
            mount_point: KV mount point.
            versions: Specific versions to delete.

        Returns:
            Deletion result.
        """
        if self._client is None:
            return {"error": "Not connected"}

        mount = mount_point or self.vault_config.mount_point

        try:
            if versions:
                self._client.secrets.kv.v2.delete_secret_versions(
                    path=path,
                    versions=versions,
                    mount_point=mount,
                )
            else:
                self._client.secrets.kv.v2.delete_latest_version_of_secret(
                    path=path,
                    mount_point=mount,
                )
            return {"path": path, "deleted": True}
        except Exception as e:
            return {"error": str(e)}

    async def _list_secrets(
        self,
        path: str = "",
        mount_point: str | None = None,
    ) -> list[str]:
        """List secrets at a path.

        Args:
            path: Path to list.
            mount_point: KV mount point.

        Returns:
            List of secret keys.
        """
        if self._client is None:
            return []

        mount = mount_point or self.vault_config.mount_point

        try:
            response = self._client.secrets.kv.v2.list_secrets(
                path=path,
                mount_point=mount,
            )
            return response.get("data", {}).get("keys", [])
        except Exception:
            return []

    async def _encrypt(
        self,
        plaintext: str,
        key_name: str,
        mount_point: str = "transit",
    ) -> dict[str, Any]:
        """Encrypt data using transit engine.

        Args:
            plaintext: Data to encrypt (base64 encoded).
            key_name: Encryption key name.
            mount_point: Transit mount point.

        Returns:
            Encrypted ciphertext.
        """
        if self._client is None:
            return {"error": "Not connected"}

        try:
            import base64

            # Encode plaintext
            encoded = base64.b64encode(plaintext.encode()).decode()

            response = self._client.secrets.transit.encrypt_data(
                name=key_name,
                plaintext=encoded,
                mount_point=mount_point,
            )
            return {"ciphertext": response.get("data", {}).get("ciphertext")}
        except Exception as e:
            return {"error": str(e)}

    async def _decrypt(
        self,
        ciphertext: str,
        key_name: str,
        mount_point: str = "transit",
    ) -> dict[str, Any]:
        """Decrypt data using transit engine.

        Args:
            ciphertext: Encrypted data.
            key_name: Encryption key name.
            mount_point: Transit mount point.

        Returns:
            Decrypted plaintext.
        """
        if self._client is None:
            return {"error": "Not connected"}

        try:
            import base64

            response = self._client.secrets.transit.decrypt_data(
                name=key_name,
                ciphertext=ciphertext,
                mount_point=mount_point,
            )
            plaintext_b64 = response.get("data", {}).get("plaintext", "")
            plaintext = base64.b64decode(plaintext_b64).decode()
            return {"plaintext": plaintext}
        except Exception as e:
            return {"error": str(e)}

    async def _get_database_credentials(
        self,
        role: str,
        mount_point: str = "database",
    ) -> dict[str, Any]:
        """Get dynamic database credentials.

        Args:
            role: Database role name.
            mount_point: Database secrets mount point.

        Returns:
            Database credentials.
        """
        if self._client is None:
            return {"error": "Not connected"}

        try:
            response = self._client.secrets.database.generate_credentials(
                name=role,
                mount_point=mount_point,
            )
            return {
                "username": response.get("data", {}).get("username"),
                "password": response.get("data", {}).get("password"),
                "lease_id": response.get("lease_id"),
                "lease_duration": response.get("lease_duration"),
            }
        except Exception as e:
            return {"error": str(e)}
