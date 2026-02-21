"""CAMARA Network API Integration for PHUC platform."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from agentic_workflows.protocols.mcp_client import MCPClient, MCPServerConfig, MCPTransport


class Operator(Enum):
    """Supported telecom operators."""

    TELEFONICA = "telefonica"
    DEUTSCHE_TELEKOM = "deutsche-telekom"
    VODAFONE = "vodafone"


@dataclass
class VerificationResult:
    """CAMARA verification result."""

    verified: bool
    confidence: float = 1.0
    details: dict = field(default_factory=dict)
    error: str | None = None


class CAMARAClient:
    """CAMARA MCP client for network verification."""

    MCP_URL = "https://mcp.camaramcp.com/sse"

    def __init__(self, mcp_url: str = None, mcp_client: MCPClient | None = None):
        self.mcp_url = mcp_url or self.MCP_URL
        self._client = mcp_client
        self._connected = False

    async def _get_client(self) -> MCPClient:
        if self._client is None:
            config = MCPServerConfig(
                name="camara",
                url=self.mcp_url,
                transport=MCPTransport.SSE,
            )
            self._client = MCPClient(config)

        if not self._connected:
            await self._client.connect()
            self._connected = True

        return self._client

    async def close(self):
        if self._client:
            if hasattr(self._client, "close"):
                await self._client.close()  # type: ignore[func-returns-value]
            self._client = None
            self._connected = False

    def _normalize_result(self, result: dict | str | None, base_details: dict, default_confidence: float) -> VerificationResult:
        """Normalize MCP result into VerificationResult."""
        try:
            if isinstance(result, str):
                parsed = json.loads(result)
            elif isinstance(result, dict):
                parsed = result
            elif result is None:
                parsed = {}
            else:
                parsed = {"result": result}
        except json.JSONDecodeError:
            parsed = {"raw": result}

        details = {**base_details, **parsed}
        verified = bool(details.get("verified", details.get("status", False)))
        confidence = float(details.get("confidence", default_confidence))
        error = details.get("error")

        return VerificationResult(
            verified=verified,
            confidence=confidence,
            details=details,
            error=error,
        )

    async def _call_camara(self, tool: str, payload: dict, default_confidence: float) -> VerificationResult:
        try:
            client = await self._get_client()
            result = await client.call_tool(tool, payload)
            return self._normalize_result(result, payload, default_confidence)
        except Exception as exc:
            return VerificationResult(
                verified=False,
                confidence=0.0,
                details=payload,
                error=str(exc),
            )

    async def check_sim_swap(
        self, phone_number: str, operator: Operator = Operator.TELEFONICA, max_age_hours: int = 24
    ) -> VerificationResult:
        """Check if SIM was swapped in last N hours.

        Use case: Fraud prevention before sensitive transactions.
        """
        payload = {
            "phone_number": phone_number,
            "operator": operator.value,
            "max_age_hours": max_age_hours,
        }
        return await self._call_camara("check_sim_swap", payload, default_confidence=0.95)

    async def check_device_swap(
        self, phone_number: str, operator: Operator = Operator.TELEFONICA, max_age_hours: int = 24
    ) -> VerificationResult:
        """Check if device was changed in last N hours."""
        payload = {
            "phone_number": phone_number,
            "operator": operator.value,
            "max_age_hours": max_age_hours,
        }
        return await self._call_camara("check_device_swap", payload, default_confidence=0.9)

    async def check_roaming_status(
        self, phone_number: str, operator: Operator = Operator.TELEFONICA
    ) -> VerificationResult:
        """Check if device is roaming."""
        payload = {
            "phone_number": phone_number,
            "operator": operator.value,
        }
        return await self._call_camara("check_roaming_status", payload, default_confidence=0.95)

    async def verify_location(
        self,
        phone_number: str,
        latitude: float,
        longitude: float,
        accuracy_km: int = 50,
        operator: Operator = Operator.TELEFONICA,
    ) -> VerificationResult:
        """Verify device is at expected location."""
        payload = {
            "phone_number": phone_number,
            "operator": operator.value,
            "latitude": latitude,
            "longitude": longitude,
            "accuracy_km": accuracy_km,
        }
        return await self._call_camara("verify_location", payload, default_confidence=0.85)

    async def verify_kyc(
        self,
        phone_number: str,
        first_name: str = None,
        last_name: str = None,
        date_of_birth: str = None,
        address: str = None,
        id_number: str = None,
        operator: Operator = Operator.TELEFONICA,
    ) -> VerificationResult:
        """Verify KYC information matches carrier records."""
        fields_matched = []
        if first_name:
            fields_matched.append("first_name")
        if last_name:
            fields_matched.append("last_name")
        if date_of_birth:
            fields_matched.append("date_of_birth")
        if address:
            fields_matched.append("address")
        if id_number:
            fields_matched.append("id_number")

        payload = {
            "phone_number": phone_number,
            "operator": operator.value,
            "first_name": first_name,
            "last_name": last_name,
            "date_of_birth": date_of_birth,
            "address": address,
            "id_number": id_number,
            "fields_matched": fields_matched,
        }
        return await self._call_camara("verify_kyc", payload, default_confidence=0.9)

    async def verify_hcp_identity(
        self, phone_number: str, npi: str, name: str, operator: Operator = Operator.TELEFONICA
    ) -> VerificationResult:
        """PHUC-specific: Verify HCP identity via carrier data."""
        payload = {
            "phone_number": phone_number,
            "operator": operator.value,
            "npi": npi,
            "name": name,
        }
        return await self._call_camara("verify_hcp_identity", payload, default_confidence=0.9)
