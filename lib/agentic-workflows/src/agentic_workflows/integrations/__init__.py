"""SDK Integrations for Agentic Workflows.

Provides adapters and integrations for various AI/ML frameworks.

Usage:
    from agentic_workflows.integrations import (
        ClaudeSDK,
        OpenAISDK,
        CrewAIAdapter,
        AutoGenAdapter,
        DSPyOptimizer,
        LlamaIndexAdapter,
    )
"""

from agentic_workflows.integrations.autogen_adapter import AutoGenAdapter, AutoGenAgent
from agentic_workflows.integrations.claude_sdk import ClaudeAgent, ClaudeSDK
from agentic_workflows.integrations.crewai_adapter import CrewAIAdapter, CrewAIAgent
from agentic_workflows.integrations.dspy_optimizer import DSPyModule, DSPyOptimizer
from agentic_workflows.integrations.llama_index import LlamaIndexAdapter, LlamaIndexAgent
from agentic_workflows.integrations.openai_sdk import OpenAIAgent, OpenAISDK

__all__ = [
    # Claude SDK
    "ClaudeSDK",
    "ClaudeAgent",
    # OpenAI SDK
    "OpenAISDK",
    "OpenAIAgent",
    # CrewAI
    "CrewAIAdapter",
    "CrewAIAgent",
    # AutoGen
    "AutoGenAdapter",
    "AutoGenAgent",
    # DSPy
    "DSPyOptimizer",
    "DSPyModule",
    # LlamaIndex
    "LlamaIndexAdapter",
    "LlamaIndexAgent",
]
