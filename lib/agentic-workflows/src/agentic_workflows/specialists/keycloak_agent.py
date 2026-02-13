"""Keycloak specialist agent for identity and access management.

Handles authentication, authorization, and user management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import SpecialistAgent, SpecialistCapability, SpecialistConfig


@dataclass
class KeycloakConfig(SpecialistConfig):
    """Keycloak-specific configuration."""

    server_url: str = "http://localhost:8080"
    realm: str = "master"
    client_id: str = "admin-cli"
    client_secret: str = ""
    admin_username: str = "admin"
    admin_password: str = "admin"


class KeycloakAgent(SpecialistAgent):
    """Specialist agent for Keycloak IAM.

    Capabilities:
    - User management
    - Role management
    - Token operations
    - Realm configuration
    """

    def __init__(self, config: KeycloakConfig | None = None, **kwargs):
        self.kc_config = config or KeycloakConfig()
        super().__init__(config=self.kc_config, **kwargs)

        self._admin = None
        self._session = None

        self.register_handler("get_user", self._get_user)
        self.register_handler("create_user", self._create_user)
        self.register_handler("update_user", self._update_user)
        self.register_handler("delete_user", self._delete_user)
        self.register_handler("list_users", self._list_users)
        self.register_handler("assign_role", self._assign_role)
        self.register_handler("get_token", self._get_token)
        self.register_handler("validate_token", self._validate_token)

    @property
    def capabilities(self) -> list[SpecialistCapability]:
        return [
            SpecialistCapability.AUTHENTICATION,
            SpecialistCapability.AUTHORIZATION,
        ]

    @property
    def service_name(self) -> str:
        return "Keycloak"

    async def _connect(self) -> None:
        """Connect to Keycloak."""
        try:
            from keycloak import KeycloakAdmin, KeycloakOpenIDConnect

            self._admin = KeycloakAdmin(
                server_url=self.kc_config.server_url,
                username=self.kc_config.admin_username,
                password=self.kc_config.admin_password,
                realm_name=self.kc_config.realm,
                client_id=self.kc_config.client_id,
                client_secret_key=self.kc_config.client_secret,
                verify=True,
            )
        except ImportError:
            self.logger.warning("python-keycloak not installed")

        import aiohttp

        self._session = aiohttp.ClientSession()

    async def _disconnect(self) -> None:
        """Disconnect from Keycloak."""
        self._admin = None
        if self._session:
            await self._session.close()
            self._session = None

    async def _health_check(self) -> bool:
        """Check Keycloak health."""
        if self._session is None:
            return False
        try:
            url = f"{self.kc_config.server_url}/health"
            async with self._session.get(url) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def _get_user(self, user_id: str) -> dict[str, Any]:
        """Get user by ID.

        Args:
            user_id: User identifier.

        Returns:
            User data.
        """
        if self._admin is None:
            return {"error": "Not connected"}

        try:
            return self._admin.get_user(user_id)
        except Exception as e:
            return {"error": str(e)}

    async def _create_user(
        self,
        username: str,
        email: str,
        password: str | None = None,
        first_name: str = "",
        last_name: str = "",
        enabled: bool = True,
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new user.

        Args:
            username: Username.
            email: Email address.
            password: Optional initial password.
            first_name: First name.
            last_name: Last name.
            enabled: Whether user is enabled.
            attributes: Custom attributes.

        Returns:
            Created user info.
        """
        if self._admin is None:
            return {"error": "Not connected"}

        try:
            user_data = {
                "username": username,
                "email": email,
                "firstName": first_name,
                "lastName": last_name,
                "enabled": enabled,
                "attributes": attributes or {},
            }

            if password:
                user_data["credentials"] = [
                    {
                        "type": "password",
                        "value": password,
                        "temporary": False,
                    }
                ]

            user_id = self._admin.create_user(user_data)
            return {"user_id": user_id, "username": username, "created": True}

        except Exception as e:
            return {"error": str(e)}

    async def _update_user(
        self,
        user_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Update a user.

        Args:
            user_id: User identifier.
            updates: Fields to update.

        Returns:
            Update result.
        """
        if self._admin is None:
            return {"error": "Not connected"}

        try:
            self._admin.update_user(user_id, updates)
            return {"user_id": user_id, "updated": True}
        except Exception as e:
            return {"error": str(e)}

    async def _delete_user(self, user_id: str) -> dict[str, Any]:
        """Delete a user.

        Args:
            user_id: User identifier.

        Returns:
            Deletion result.
        """
        if self._admin is None:
            return {"error": "Not connected"}

        try:
            self._admin.delete_user(user_id)
            return {"user_id": user_id, "deleted": True}
        except Exception as e:
            return {"error": str(e)}

    async def _list_users(
        self,
        search: str | None = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """List users.

        Args:
            search: Optional search query.
            max_results: Maximum results.

        Returns:
            List of users.
        """
        if self._admin is None:
            return []

        try:
            query = {"max": max_results}
            if search:
                query["search"] = search
            return self._admin.get_users(query)
        except Exception:
            return []

    async def _assign_role(
        self,
        user_id: str,
        role_name: str,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        """Assign a role to a user.

        Args:
            user_id: User identifier.
            role_name: Role name.
            client_id: Client ID for client roles.

        Returns:
            Assignment result.
        """
        if self._admin is None:
            return {"error": "Not connected"}

        try:
            if client_id:
                client = self._admin.get_client_id(client_id)
                role = self._admin.get_client_role(client, role_name)
                self._admin.assign_client_role(user_id, client, [role])
            else:
                role = self._admin.get_realm_role(role_name)
                self._admin.assign_realm_roles(user_id, [role])

            return {"user_id": user_id, "role": role_name, "assigned": True}
        except Exception as e:
            return {"error": str(e)}

    async def _get_token(
        self,
        username: str,
        password: str,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        """Get access token for user.

        Args:
            username: Username.
            password: Password.
            client_id: Client ID.

        Returns:
            Token response.
        """
        if self._session is None:
            return {"error": "Not connected"}

        url = f"{self.kc_config.server_url}/realms/{self.kc_config.realm}/protocol/openid-connect/token"

        data = {
            "grant_type": "password",
            "client_id": client_id or self.kc_config.client_id,
            "username": username,
            "password": password,
        }

        if self.kc_config.client_secret:
            data["client_secret"] = self.kc_config.client_secret

        async with self._session.post(url, data=data) as resp:
            if resp.status == 200:
                return await resp.json()
            return {"error": f"Token request failed: {resp.status}"}

    async def _validate_token(self, token: str) -> dict[str, Any]:
        """Validate an access token.

        Args:
            token: Access token.

        Returns:
            Token info.
        """
        if self._session is None:
            return {"error": "Not connected"}

        url = f"{self.kc_config.server_url}/realms/{self.kc_config.realm}/protocol/openid-connect/userinfo"

        headers = {"Authorization": f"Bearer {token}"}

        async with self._session.get(url, headers=headers) as resp:
            if resp.status == 200:
                user_info = await resp.json()
                return {"valid": True, "user_info": user_info}
            return {"valid": False, "error": f"Validation failed: {resp.status}"}
