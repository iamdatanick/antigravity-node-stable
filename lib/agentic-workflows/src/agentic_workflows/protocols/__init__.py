"""Protocol implementations for agentic workflows.

This module provides implementations for agent communication protocols:

- MCP (Model Context Protocol): For tool and resource integration
- A2A (Agent-to-Agent): For inter-agent communication

See also:
    - agentic_workflows.a2a: Comprehensive A2A protocol support
"""

from agentic_workflows.a2a import (
    # Server
    A2AServer,
    create_a2a_server,
)
from agentic_workflows.a2a import (
    AgentCard as A2AAgentCard,
)

# Import the new A2A module for convenience
from agentic_workflows.a2a import (
    # Core types
    AgentSkill as A2AAgentSkill,
)
from agentic_workflows.a2a import (
    Task as A2ATaskFull,
)
from agentic_workflows.protocols.a2a_client import (
    A2AClient,
    A2AMessage,
    A2ATask,
)
from agentic_workflows.protocols.agent_card import (
    AgentCard,
    Capability,
    Constraint,
)
from agentic_workflows.protocols.mcp_client import (
    MCPClient,
    MCPServerConfig,
    MCPTool,
)
from agentic_workflows.protocols.mcp_server import (
    AgenticWorkflowsMCPServer,
)
from agentic_workflows.protocols.mcp_server import (
    serve as mcp_serve,
)

__all__ = [
    # MCP Client
    "MCPClient",
    "MCPTool",
    "MCPServerConfig",
    # MCP Server
    "AgenticWorkflowsMCPServer",
    "mcp_serve",
    # A2A (legacy)
    "A2AClient",
    "A2AMessage",
    "A2ATask",
    # A2A (new module)
    "A2AAgentSkill",
    "A2AAgentCard",
    "A2ATaskFull",
    "A2AServer",
    "create_a2a_server",
    # Agent Card
    "AgentCard",
    "Capability",
    "Constraint",
]
