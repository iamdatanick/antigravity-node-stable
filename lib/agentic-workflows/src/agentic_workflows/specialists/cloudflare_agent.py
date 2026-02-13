"""Cloudflare specialist agent for CDN and security.

Handles CDN operations, WAF, DDoS protection, and edge compute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import SpecialistAgent, SpecialistConfig, SpecialistCapability


@dataclass
class CloudflareConfig(SpecialistConfig):
    """Cloudflare-specific configuration."""

    api_token: str = ""
    api_key: str = ""
    api_email: str = ""
    account_id: str = ""
    zone_id: str = ""


class CloudflareAgent(SpecialistAgent):
    """Specialist agent for Cloudflare.

    Capabilities:
    - CDN management
    - WAF rules
    - DDoS protection
    - DNS management
    - Workers deployment
    """

    def __init__(self, config: CloudflareConfig | None = None, **kwargs):
        self.cf_config = config or CloudflareConfig()
        super().__init__(config=self.cf_config, **kwargs)

        self._session = None
        self._base_url = "https://api.cloudflare.com/client/v4"

        self.register_handler("purge_cache", self._purge_cache)
        self.register_handler("get_zone_settings", self._get_zone_settings)
        self.register_handler("update_zone_setting", self._update_zone_setting)
        self.register_handler("list_dns_records", self._list_dns_records)
        self.register_handler("create_dns_record", self._create_dns_record)
        self.register_handler("get_waf_rules", self._get_waf_rules)
        self.register_handler("get_analytics", self._get_analytics)
        self.register_handler("deploy_worker", self._deploy_worker)

    @property
    def capabilities(self) -> list[SpecialistCapability]:
        return [
            SpecialistCapability.CDN,
            SpecialistCapability.WAF,
            SpecialistCapability.DDOS_PROTECTION,
        ]

    @property
    def service_name(self) -> str:
        return "Cloudflare"

    async def _connect(self) -> None:
        """Connect to Cloudflare API."""
        import aiohttp

        headers = {"Content-Type": "application/json"}

        if self.cf_config.api_token:
            headers["Authorization"] = f"Bearer {self.cf_config.api_token}"
        elif self.cf_config.api_key and self.cf_config.api_email:
            headers["X-Auth-Key"] = self.cf_config.api_key
            headers["X-Auth-Email"] = self.cf_config.api_email

        self._session = aiohttp.ClientSession(headers=headers)

    async def _disconnect(self) -> None:
        """Disconnect from Cloudflare."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _health_check(self) -> bool:
        """Check Cloudflare API health."""
        if self._session is None:
            return False
        try:
            url = f"{self._base_url}/user/tokens/verify"
            async with self._session.get(url) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def _purge_cache(
        self,
        zone_id: str | None = None,
        purge_everything: bool = False,
        files: list[str] | None = None,
        tags: list[str] | None = None,
        hosts: list[str] | None = None,
    ) -> dict[str, Any]:
        """Purge CDN cache.

        Args:
            zone_id: Zone identifier.
            purge_everything: Purge all cached content.
            files: Specific URLs to purge.
            tags: Cache tags to purge.
            hosts: Hostnames to purge.

        Returns:
            Purge result.
        """
        if self._session is None:
            return {"error": "Not connected"}

        zone = zone_id or self.cf_config.zone_id
        url = f"{self._base_url}/zones/{zone}/purge_cache"

        if purge_everything:
            payload = {"purge_everything": True}
        else:
            payload = {}
            if files:
                payload["files"] = files
            if tags:
                payload["tags"] = tags
            if hosts:
                payload["hosts"] = hosts

        async with self._session.post(url, json=payload) as resp:
            result = await resp.json()
            return result

    async def _get_zone_settings(self, zone_id: str | None = None) -> dict[str, Any]:
        """Get zone settings.

        Args:
            zone_id: Zone identifier.

        Returns:
            Zone settings.
        """
        if self._session is None:
            return {"error": "Not connected"}

        zone = zone_id or self.cf_config.zone_id
        url = f"{self._base_url}/zones/{zone}/settings"

        async with self._session.get(url) as resp:
            return await resp.json()

    async def _update_zone_setting(
        self,
        setting: str,
        value: Any,
        zone_id: str | None = None,
    ) -> dict[str, Any]:
        """Update a zone setting.

        Args:
            setting: Setting name.
            value: Setting value.
            zone_id: Zone identifier.

        Returns:
            Update result.
        """
        if self._session is None:
            return {"error": "Not connected"}

        zone = zone_id or self.cf_config.zone_id
        url = f"{self._base_url}/zones/{zone}/settings/{setting}"

        async with self._session.patch(url, json={"value": value}) as resp:
            return await resp.json()

    async def _list_dns_records(
        self,
        zone_id: str | None = None,
        record_type: str | None = None,
        name: str | None = None,
    ) -> list[dict[str, Any]]:
        """List DNS records.

        Args:
            zone_id: Zone identifier.
            record_type: Filter by record type.
            name: Filter by record name.

        Returns:
            List of DNS records.
        """
        if self._session is None:
            return []

        zone = zone_id or self.cf_config.zone_id
        url = f"{self._base_url}/zones/{zone}/dns_records"

        params = {}
        if record_type:
            params["type"] = record_type
        if name:
            params["name"] = name

        async with self._session.get(url, params=params) as resp:
            result = await resp.json()
            return result.get("result", [])

    async def _create_dns_record(
        self,
        record_type: str,
        name: str,
        content: str,
        zone_id: str | None = None,
        ttl: int = 1,
        proxied: bool = True,
    ) -> dict[str, Any]:
        """Create a DNS record.

        Args:
            record_type: Record type (A, CNAME, etc.).
            name: Record name.
            content: Record content.
            zone_id: Zone identifier.
            ttl: Time to live.
            proxied: Whether to proxy through Cloudflare.

        Returns:
            Created record.
        """
        if self._session is None:
            return {"error": "Not connected"}

        zone = zone_id or self.cf_config.zone_id
        url = f"{self._base_url}/zones/{zone}/dns_records"

        payload = {
            "type": record_type,
            "name": name,
            "content": content,
            "ttl": ttl,
            "proxied": proxied,
        }

        async with self._session.post(url, json=payload) as resp:
            return await resp.json()

    async def _get_waf_rules(
        self,
        zone_id: str | None = None,
        package_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get WAF rules.

        Args:
            zone_id: Zone identifier.
            package_id: WAF package ID.

        Returns:
            List of WAF rules.
        """
        if self._session is None:
            return []

        zone = zone_id or self.cf_config.zone_id
        url = f"{self._base_url}/zones/{zone}/firewall/waf/packages"

        async with self._session.get(url) as resp:
            result = await resp.json()
            packages = result.get("result", [])

            all_rules = []
            for package in packages:
                if package_id and package.get("id") != package_id:
                    continue

                rules_url = f"{self._base_url}/zones/{zone}/firewall/waf/packages/{package['id']}/rules"
                async with self._session.get(rules_url) as rules_resp:
                    rules_result = await rules_resp.json()
                    all_rules.extend(rules_result.get("result", []))

            return all_rules

    async def _get_analytics(
        self,
        zone_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """Get zone analytics.

        Args:
            zone_id: Zone identifier.
            since: Start time (ISO format).
            until: End time (ISO format).

        Returns:
            Analytics data.
        """
        if self._session is None:
            return {"error": "Not connected"}

        zone = zone_id or self.cf_config.zone_id
        url = f"{self._base_url}/zones/{zone}/analytics/dashboard"

        params = {}
        if since:
            params["since"] = since
        if until:
            params["until"] = until

        async with self._session.get(url, params=params) as resp:
            return await resp.json()

    async def _deploy_worker(
        self,
        script_name: str,
        script_content: str,
        account_id: str | None = None,
    ) -> dict[str, Any]:
        """Deploy a Cloudflare Worker.

        Args:
            script_name: Worker script name.
            script_content: JavaScript content.
            account_id: Account identifier.

        Returns:
            Deployment result.
        """
        if self._session is None:
            return {"error": "Not connected"}

        account = account_id or self.cf_config.account_id
        url = f"{self._base_url}/accounts/{account}/workers/scripts/{script_name}"

        # Workers require multipart form data
        import aiohttp

        form = aiohttp.FormData()
        form.add_field(
            "script",
            script_content,
            content_type="application/javascript",
        )

        # Need to remove Content-Type for multipart
        headers = dict(self._session.headers)
        headers.pop("Content-Type", None)

        async with self._session.put(url, data=form, headers=headers) as resp:
            return await resp.json()
