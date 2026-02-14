"""Apache Unomi specialist agent for Customer Data Platform.

Handles HCP profiles, segmentation, and event tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import SpecialistAgent, SpecialistCapability, SpecialistConfig


@dataclass
class UnomiConfig(SpecialistConfig):
    """Unomi-specific configuration."""

    endpoint: str = "http://localhost:8181"
    username: str = "karaf"
    password: str = "karaf"
    scope: str = "phuc"


class UnomiAgent(SpecialistAgent):
    """Specialist agent for Apache Unomi CDP.

    Capabilities:
    - Profile management (HCP profiles)
    - Segmentation
    - Event tracking
    - Personalization rules
    """

    def __init__(self, config: UnomiConfig | None = None, **kwargs):
        self.unomi_config = config or UnomiConfig()
        super().__init__(config=self.unomi_config, **kwargs)

        self._session = None

        self.register_handler("get_profile", self._get_profile)
        self.register_handler("create_profile", self._create_profile)
        self.register_handler("update_profile", self._update_profile)
        self.register_handler("track_event", self._track_event)
        self.register_handler("get_segments", self._get_segments)
        self.register_handler("create_segment", self._create_segment)
        self.register_handler("evaluate_profile", self._evaluate_profile)

    @property
    def capabilities(self) -> list[SpecialistCapability]:
        return [
            SpecialistCapability.PROFILE_MANAGEMENT,
            SpecialistCapability.SEGMENTATION,
            SpecialistCapability.EVENT_TRACKING,
        ]

    @property
    def service_name(self) -> str:
        return "Apache Unomi"

    async def _connect(self) -> None:
        """Connect to Unomi."""
        import aiohttp

        auth = aiohttp.BasicAuth(self.unomi_config.username, self.unomi_config.password)
        self._session = aiohttp.ClientSession(auth=auth)

    async def _disconnect(self) -> None:
        """Disconnect from Unomi."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _health_check(self) -> bool:
        """Check Unomi health."""
        if self._session is None:
            return False
        try:
            async with self._session.get(f"{self.unomi_config.endpoint}/cxs/cluster") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def _get_profile(self, profile_id: str) -> dict[str, Any]:
        """Get a profile by ID.

        Args:
            profile_id: Profile identifier.

        Returns:
            Profile data.
        """
        if self._session is None:
            return {"error": "Not connected"}

        url = f"{self.unomi_config.endpoint}/cxs/profiles/{profile_id}"
        async with self._session.get(url) as resp:
            if resp.status == 200:
                return await resp.json()
            return {"error": f"Profile not found: {resp.status}"}

    async def _create_profile(
        self,
        profile_id: str,
        properties: dict[str, Any],
        segments: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new profile.

        Args:
            profile_id: Profile identifier.
            properties: Profile properties.
            segments: Initial segments.

        Returns:
            Created profile.
        """
        if self._session is None:
            return {"error": "Not connected"}

        profile = {
            "itemId": profile_id,
            "itemType": "profile",
            "properties": properties,
            "segments": segments or [],
            "scope": self.unomi_config.scope,
        }

        url = f"{self.unomi_config.endpoint}/cxs/profiles"
        async with self._session.post(url, json=profile) as resp:
            if resp.status in (200, 201):
                return await resp.json()
            return {"error": f"Failed to create: {resp.status}"}

    async def _update_profile(
        self,
        profile_id: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        """Update profile properties.

        Args:
            profile_id: Profile identifier.
            properties: Properties to update.

        Returns:
            Updated profile.
        """
        if self._session is None:
            return {"error": "Not connected"}

        # Get current profile
        current = await self._get_profile(profile_id)
        if "error" in current:
            return current

        # Merge properties
        current["properties"].update(properties)

        url = f"{self.unomi_config.endpoint}/cxs/profiles/{profile_id}"
        async with self._session.put(url, json=current) as resp:
            if resp.status == 200:
                return await resp.json()
            return {"error": f"Failed to update: {resp.status}"}

    async def _track_event(
        self,
        profile_id: str,
        event_type: str,
        properties: dict[str, Any] | None = None,
        source: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Track an event for a profile.

        Args:
            profile_id: Profile identifier.
            event_type: Type of event.
            properties: Event properties.
            source: Event source info.

        Returns:
            Event tracking result.
        """
        if self._session is None:
            return {"error": "Not connected"}

        import time
        import uuid

        event = {
            "eventType": event_type,
            "scope": self.unomi_config.scope,
            "profileId": profile_id,
            "properties": properties or {},
            "source": source or {"itemType": "site", "scope": self.unomi_config.scope},
            "timeStamp": int(time.time() * 1000),
            "itemId": str(uuid.uuid4()),
        }

        url = f"{self.unomi_config.endpoint}/cxs/events"
        async with self._session.post(url, json=event) as resp:
            if resp.status in (200, 201):
                return {"tracked": True, "event_id": event["itemId"]}
            return {"error": f"Failed to track: {resp.status}"}

    async def _get_segments(self, scope: str | None = None) -> list[dict[str, Any]]:
        """Get all segments.

        Args:
            scope: Optional scope filter.

        Returns:
            List of segments.
        """
        if self._session is None:
            return []

        url = f"{self.unomi_config.endpoint}/cxs/segments"
        async with self._session.get(url) as resp:
            if resp.status == 200:
                segments = await resp.json()
                if scope:
                    segments = [s for s in segments if s.get("scope") == scope]
                return segments
            return []

    async def _create_segment(
        self,
        segment_id: str,
        name: str,
        condition: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a new segment.

        Args:
            segment_id: Segment identifier.
            name: Segment name.
            condition: Segment condition.

        Returns:
            Created segment.
        """
        if self._session is None:
            return {"error": "Not connected"}

        segment = {
            "itemId": segment_id,
            "itemType": "segment",
            "metadata": {
                "id": segment_id,
                "name": name,
                "scope": self.unomi_config.scope,
            },
            "condition": condition,
            "scope": self.unomi_config.scope,
        }

        url = f"{self.unomi_config.endpoint}/cxs/segments"
        async with self._session.post(url, json=segment) as resp:
            if resp.status in (200, 201):
                return await resp.json()
            return {"error": f"Failed to create segment: {resp.status}"}

    async def _evaluate_profile(
        self,
        profile_id: str,
        segment_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate profile against segments.

        Args:
            profile_id: Profile identifier.
            segment_ids: Specific segments to check.

        Returns:
            Evaluation results.
        """
        if self._session is None:
            return {"error": "Not connected"}

        profile = await self._get_profile(profile_id)
        if "error" in profile:
            return profile

        return {
            "profile_id": profile_id,
            "segments": profile.get("segments", []),
            "scores": profile.get("scores", {}),
        }
