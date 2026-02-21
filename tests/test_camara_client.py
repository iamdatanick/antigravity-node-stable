import pytest

from agentic_workflows.phuc.integrations.camara import CAMARAClient, Operator


class _DummyMCP:
    def __init__(self, result):
        self.result = result
        self.connected = False
        self.calls: list[tuple[str, dict]] = []

    async def connect(self):
        self.connected = True
        return True

    async def call_tool(self, name: str, args: dict):
        self.calls.append((name, args))
        return self.result


@pytest.mark.asyncio
async def test_camara_invokes_mcp_tools():
    mcp = _DummyMCP({"verified": True, "confidence": 0.77, "swapped": False})
    client = CAMARAClient(mcp_client=mcp)

    result = await client.check_sim_swap("+1555", Operator.VODAFONE, 12)

    assert mcp.connected is True
    assert mcp.calls == [
        ("check_sim_swap", {"phone_number": "+1555", "operator": "vodafone", "max_age_hours": 12})
    ]
    assert result.verified is True
    assert result.details["operator"] == "vodafone"
    assert result.details.get("swapped") is False
