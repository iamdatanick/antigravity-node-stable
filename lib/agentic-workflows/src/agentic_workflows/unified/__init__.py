"""
Unified Cross-System Integration Module for Agentic Workflows.

This module provides a unified integration layer that ties together all SDKs
and protocols in agentic_workflows, enabling seamless interoperability between:

- A2A Protocol (agent-to-agent communication)
- OpenAI Agents SDK (handoffs, guardrails)
- MCP Servers and Tools
- MCP-UI Widgets
- OpenAI Apps SDK
- Skills (Anthropic SKILL.md, OpenAI Codex formats)

Components:
-----------

**UnifiedAgent** (agent.py):
    Agent class that works with all protocols and skill formats.
    - Methods: run(), run_with_handoffs(), expose_as_a2a(), expose_as_mcp()
    - Support for tools, handoffs, guardrails, and skills

**UnifiedServer** (server.py):
    Server that exposes agents via multiple protocols simultaneously.
    - A2A protocol endpoints
    - MCP protocol endpoints
    - MCP-UI widget endpoints
    - OpenAI Apps SDK endpoints
    - HTTP REST API

**UnifiedSkillRegistry** (skills.py):
    Combined registry for all skill formats.
    - Cross-format skill discovery
    - Skill-to-tool conversion
    - Progressive disclosure support

**UnifiedMCPServer** (mcp.py):
    Enhanced MCP server with full feature support.
    - Standard MCP tools, resources, prompts
    - MCP-UI widget resources
    - A2A compatibility

**Protocol Bridge** (bridge.py):
    Adapters for bidirectional protocol conversion.
    - A2AtoMCP, MCPtoA2A
    - OpenAIAgentsToA2A
    - Tool/skill format translation

Quick Start:
------------

Create a unified agent:
```python
from agentic_workflows.unified import create_unified_agent

agent = create_unified_agent(
    name="assistant",
    instructions="You are a helpful assistant.",
    model="claude-sonnet-4",
)

# Add a tool
@agent.tool("search")
async def search(query: str) -> str:
    '''Search for information.'''
    return f"Results for: {query}"

# Run the agent
result = await agent.run("Search for Python tutorials")
print(result.output)
```

Create a unified server:
```python
from agentic_workflows.unified import (
    create_unified_agent,
    create_unified_server,
)

# Create agents
helper = create_unified_agent(name="helper", instructions="...")
specialist = create_unified_agent(name="specialist", instructions="...")

# Create server exposing all protocols
server = create_unified_server(
    agents=[helper, specialist],
    name="my-server",
    port=8000,
)

# Run server (serves A2A, MCP, REST, etc.)
await server.run()
```

Create an MCP server with skills:
```python
from agentic_workflows.unified import create_unified_mcp_server

server = create_unified_mcp_server("skill-server")

# Skills are auto-loaded
# Add custom tools
@server.tool("hello")
async def hello(name: str = "World") -> str:
    return f"Hello, {name}!"

await server.run()
```

Protocol bridging:
```python
from agentic_workflows.unified import (
    convert_tool,
    convert_message,
    ProtocolType,
)

# Convert A2A skill to OpenAI function
openai_func = convert_tool(
    a2a_skill,
    ProtocolType.A2A,
    ProtocolType.OPENAI_FUNCTIONS,
)

# Convert OpenAI message to Claude format
claude_msg = convert_message(
    openai_msg,
    ProtocolType.OPENAI_FUNCTIONS,
    ProtocolType.CLAUDE_TOOLS,
)
```

Author: Agentic Workflows Contributors
Version: 1.0.0
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Agentic Workflows Contributors"

# =============================================================================
# Unified Skill System
# =============================================================================

# =============================================================================
# Unified Agent
# =============================================================================
from agentic_workflows.unified.agent import (
    # Enums
    AgentCapability,
    AgentGuardrail,
    AgentHandoff,
    AgentResult,
    AgentState,
    # Types
    AgentTool,
    # Core classes
    UnifiedAgent,
    UnifiedAgentConfig,
    # Factory
    create_unified_agent,
)

# =============================================================================
# Protocol Bridge
# =============================================================================
from agentic_workflows.unified.bridge import (
    # Adapters
    A2AtoMCPAdapter,
    A2AtoOpenAIAdapter,
    ClaudeToOpenAIAdapter,
    MCPtoA2AAdapter,
    MessageRole,
    OpenAIAgentsToA2AAdapter,
    OpenAIToClaudeAdapter,
    ProtocolAdapter,
    # Core classes
    ProtocolBridge,
    # Enums
    ProtocolType,
    # Universal types
    UniversalMessage,
    UniversalTool,
    # Convenience functions
    convert_message,
    convert_tool,
    # Global bridge
    get_protocol_bridge,
)

# =============================================================================
# Unified MCP Server
# =============================================================================
from agentic_workflows.unified.mcp import (
    # Enums
    MCPCapability,
    MCPPromptDefinition,
    MCPResourceDefinition,
    # Definitions
    MCPToolDefinition,
    UnifiedMCPConfig,
    # Core classes
    UnifiedMCPServer,
    # Factory
    create_unified_mcp_server,
)
from agentic_workflows.unified.skills import (
    # Search
    SearchResult,
    # Enums
    SkillFormat,
    ToolFormat,
    UnifiedSkill,
    UnifiedSkillConfig,
    # Core classes
    UnifiedSkillRegistry,
    # Global registry
    get_unified_registry,
    set_unified_registry,
)

# Alias for convenience
AgentConfig = UnifiedAgentConfig

# =============================================================================
# Unified Server
# =============================================================================

from agentic_workflows.unified.server import (
    # Types
    AgentRegistration,
    ServerEndpoint,
    # Enums
    ServerProtocol,
    # Core classes
    UnifiedServer,
    UnifiedServerConfig,
    # Factory
    create_unified_server,
)

# =============================================================================
# Convenience Exports
# =============================================================================


def create_agent(
    name: str = "agent",
    instructions: str = "",
    model: str = "claude-sonnet-4",
    tools: list = None,
    handoffs: list = None,
    **kwargs,
) -> UnifiedAgent:
    """Create a unified agent (alias for create_unified_agent).

    Args:
        name: Agent name.
        instructions: System instructions.
        model: LLM model.
        tools: Tool functions.
        handoffs: Handoff targets.
        **kwargs: Additional config.

    Returns:
        Configured UnifiedAgent.
    """
    return create_unified_agent(
        name=name,
        instructions=instructions,
        model=model,
        tools=tools,
        handoffs=handoffs,
        **kwargs,
    )


def create_server(
    agents: list[UnifiedAgent] = None,
    name: str = "server",
    host: str = "127.0.0.1",
    port: int = 8000,
    **kwargs,
) -> UnifiedServer:
    """Create a unified server (alias for create_unified_server).

    Args:
        agents: Agents to register.
        name: Server name.
        host: Host to bind.
        port: Port to listen.
        **kwargs: Additional config.

    Returns:
        Configured UnifiedServer.
    """
    return create_unified_server(
        agents=agents,
        name=name,
        host=host,
        port=port,
        **kwargs,
    )


def create_mcp_server(
    name: str = "mcp-server",
    auto_load_skills: bool = True,
    **kwargs,
) -> UnifiedMCPServer:
    """Create an MCP server (alias for create_unified_mcp_server).

    Args:
        name: Server name.
        auto_load_skills: Auto-load skills.
        **kwargs: Additional config.

    Returns:
        Configured UnifiedMCPServer.
    """
    return create_unified_mcp_server(
        name=name,
        auto_load_skills=auto_load_skills,
        **kwargs,
    )


def get_skill_registry() -> UnifiedSkillRegistry:
    """Get the global unified skill registry."""
    return get_unified_registry()


def get_bridge() -> ProtocolBridge:
    """Get the global protocol bridge."""
    return get_protocol_bridge()


# =============================================================================
# All Exports
# =============================================================================

__all__ = [
    # Version
    "__version__",
    "__author__",
    # =========================================================================
    # Unified Skill System
    # =========================================================================
    "UnifiedSkillRegistry",
    "UnifiedSkill",
    "UnifiedSkillConfig",
    "SkillFormat",
    "ToolFormat",
    "SearchResult",
    "get_unified_registry",
    "set_unified_registry",
    "get_skill_registry",
    # =========================================================================
    # Unified MCP Server
    # =========================================================================
    "UnifiedMCPServer",
    "UnifiedMCPConfig",
    "MCPToolDefinition",
    "MCPResourceDefinition",
    "MCPPromptDefinition",
    "MCPCapability",
    "create_unified_mcp_server",
    "create_mcp_server",
    # =========================================================================
    # Protocol Bridge
    # =========================================================================
    "ProtocolBridge",
    "ProtocolAdapter",
    "A2AtoMCPAdapter",
    "MCPtoA2AAdapter",
    "OpenAIAgentsToA2AAdapter",
    "A2AtoOpenAIAdapter",
    "ClaudeToOpenAIAdapter",
    "OpenAIToClaudeAdapter",
    "UniversalMessage",
    "UniversalTool",
    "ProtocolType",
    "MessageRole",
    "get_protocol_bridge",
    "get_bridge",
    "convert_message",
    "convert_tool",
    # =========================================================================
    # Unified Agent
    # =========================================================================
    "UnifiedAgent",
    "UnifiedAgentConfig",
    "AgentConfig",  # Alias
    "AgentTool",
    "AgentHandoff",
    "AgentGuardrail",
    "AgentResult",
    "AgentCapability",
    "AgentState",
    "create_unified_agent",
    "create_agent",
    # =========================================================================
    # Unified Server
    # =========================================================================
    "UnifiedServer",
    "UnifiedServerConfig",
    "AgentRegistration",
    "ServerEndpoint",
    "ServerProtocol",
    "create_unified_server",
    "create_server",
]


# =============================================================================
# Module Info
# =============================================================================


def get_version() -> str:
    """Get module version."""
    return __version__


def get_info() -> dict:
    """Get module information."""
    return {
        "name": "agentic_workflows.unified",
        "version": __version__,
        "author": __author__,
        "description": "Unified cross-system integration for agentic workflows",
        "components": {
            "agent": "UnifiedAgent - Multi-protocol agent",
            "server": "UnifiedServer - Multi-protocol server",
            "mcp": "UnifiedMCPServer - Enhanced MCP server",
            "skills": "UnifiedSkillRegistry - Cross-format skills",
            "bridge": "ProtocolBridge - Protocol conversion",
        },
        "protocols": [
            "A2A (Agent-to-Agent)",
            "MCP (Model Context Protocol)",
            "MCP-UI (MCP User Interface)",
            "OpenAI Agents SDK",
            "OpenAI Apps SDK",
            "Claude Tools API",
        ],
    }
