import pytest

from agentic_workflows.openai_agents import create_agent
from agentic_workflows.openai_agents.runner import Runner


class _StubChoice:
    def __init__(self, content: str):
        self.message = type("Msg", (), {"content": content, "tool_calls": []})
        self.finish_reason = "stop"


class _StubCompletions:
    async def create(
        self,
        model,
        messages,
        tools=None,
        temperature=None,
        max_tokens=None,
        **kwargs,
    ):
        return type("Resp", (), {"choices": [_StubChoice("ack") ]})


class _StubChat:
    def __init__(self):
        self.completions = _StubCompletions()


class _StubClient:
    def __init__(self):
        self.chat = _StubChat()


@pytest.mark.asyncio
async def test_runner_requires_llm_client():
    agent = create_agent(name="assistant", instructions="You are helpful.")
    runner = Runner(llm_client=None)

    result = await runner.run(agent, "hello")

    assert result.success is False
    assert "LLM client" in (result.error or "")


@pytest.mark.asyncio
async def test_runner_uses_provided_llm_client():
    agent = create_agent(name="assistant", instructions="You are helpful.")
    runner = Runner(llm_client=_StubClient())

    result = await runner.run(agent, "hello")

    assert result.success is True
    assert result.output == "ack"
