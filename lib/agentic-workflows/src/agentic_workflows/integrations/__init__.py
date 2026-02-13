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

from agentic_workflows.integrations.claude_sdk import ClaudeSDK, ClaudeAgent
from agentic_workflows.integrations.openai_sdk import OpenAISDK, OpenAIAgent
from agentic_workflows.integrations.crewai_adapter import CrewAIAdapter, CrewAIAgent
from agentic_workflows.integrations.autogen_adapter import AutoGenAdapter, AutoGenAgent
from agentic_workflows.integrations.dspy_optimizer import DSPyOptimizer, DSPyModule
from agentic_workflows.integrations.llama_index import LlamaIndexAdapter, LlamaIndexAgent

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
