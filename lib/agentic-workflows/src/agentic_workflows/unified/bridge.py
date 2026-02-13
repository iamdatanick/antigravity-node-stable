"""
Protocol Bridge for Agentic Workflows.

This module provides adapters for bidirectional conversion between:
- A2A protocol <-> MCP protocol
- OpenAI Agents SDK <-> A2A protocol
- MCP <-> OpenAI function calling
- All protocols <-> Internal agent format

Features:
- Bidirectional message/event conversion
- Tool/skill format translation
- Session and state management bridging
- Error handling and fallback strategies

Example:
    >>> from agentic_workflows.unified.bridge import (
    ...     A2AtoMCPAdapter,
    ...     MCPtoA2AAdapter,
    ...     OpenAIAgentsToA2AAdapter,
    ... )
    >>>
    >>> # Convert A2A message to MCP format
    >>> mcp_msg = A2AtoMCPAdapter.convert_message(a2a_message)
    >>>
    >>> # Bridge A2A server to MCP client
    >>> adapter = A2AtoMCPAdapter(a2a_server)
    >>> mcp_tools = adapter.get_mcp_tools()
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ProtocolType(Enum):
    """Supported protocol types."""
    A2A = "a2a"
    MCP = "mcp"
    OPENAI_AGENTS = "openai_agents"
    OPENAI_FUNCTIONS = "openai_functions"
    CLAUDE_TOOLS = "claude_tools"
    INTERNAL = "internal"


class MessageRole(Enum):
    """Universal message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    AGENT = "agent"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class UniversalMessage:
    """Universal message format for cross-protocol communication."""
    role: MessageRole
    content: str
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Protocol-specific data
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def to_a2a_message(self) -> dict[str, Any]:
        """Convert to A2A message format."""
        parts = [{"type": "text", "text": self.content}]

        for artifact in self.artifacts:
            parts.append({
                "type": "data",
                "data": artifact,
            })

        return {
            "role": "user" if self.role == MessageRole.USER else "agent",
            "parts": parts,
            "metadata": self.metadata,
        }

    def to_mcp_message(self) -> dict[str, Any]:
        """Convert to MCP message format."""
        return {
            "role": self.role.value,
            "content": {
                "type": "text",
                "text": self.content,
            },
        }

    def to_openai_message(self) -> dict[str, Any]:
        """Convert to OpenAI message format."""
        msg = {
            "role": self._map_role_to_openai(),
            "content": self.content,
        }

        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls

        return msg

    def to_claude_message(self) -> dict[str, Any]:
        """Convert to Claude message format."""
        content = [{"type": "text", "text": self.content}]

        if self.tool_results:
            for result in self.tool_results:
                content.append({
                    "type": "tool_result",
                    "tool_use_id": result.get("id", ""),
                    "content": result.get("content", ""),
                })

        return {
            "role": self._map_role_to_claude(),
            "content": content,
        }

    def _map_role_to_openai(self) -> str:
        """Map role to OpenAI format."""
        mapping = {
            MessageRole.USER: "user",
            MessageRole.ASSISTANT: "assistant",
            MessageRole.AGENT: "assistant",
            MessageRole.SYSTEM: "system",
            MessageRole.TOOL: "tool",
        }
        return mapping.get(self.role, "user")

    def _map_role_to_claude(self) -> str:
        """Map role to Claude format."""
        mapping = {
            MessageRole.USER: "user",
            MessageRole.ASSISTANT: "assistant",
            MessageRole.AGENT: "assistant",
            MessageRole.SYSTEM: "user",  # Claude uses system in first message
            MessageRole.TOOL: "user",
        }
        return mapping.get(self.role, "user")

    @classmethod
    def from_a2a(cls, data: dict[str, Any]) -> "UniversalMessage":
        """Create from A2A message."""
        role_str = data.get("role", "user")
        role = MessageRole.USER if role_str == "user" else MessageRole.AGENT

        # Extract text content
        content_parts = []
        artifacts = []
        for part in data.get("parts", []):
            if part.get("type") == "text":
                content_parts.append(part.get("text", ""))
            elif part.get("type") == "data":
                artifacts.append(part.get("data", {}))

        return cls(
            role=role,
            content="\n".join(content_parts),
            metadata=data.get("metadata", {}),
            artifacts=artifacts,
        )

    @classmethod
    def from_openai(cls, data: dict[str, Any]) -> "UniversalMessage":
        """Create from OpenAI message."""
        role_map = {
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "system": MessageRole.SYSTEM,
            "tool": MessageRole.TOOL,
        }
        role = role_map.get(data.get("role", "user"), MessageRole.USER)

        return cls(
            role=role,
            content=data.get("content", ""),
            tool_calls=data.get("tool_calls", []),
        )

    @classmethod
    def from_claude(cls, data: dict[str, Any]) -> "UniversalMessage":
        """Create from Claude message."""
        role = MessageRole.USER if data.get("role") == "user" else MessageRole.ASSISTANT

        content_parts = []
        tool_results = []

        content_data = data.get("content", [])
        if isinstance(content_data, str):
            content_parts.append(content_data)
        else:
            for block in content_data:
                if block.get("type") == "text":
                    content_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    tool_results.append(block)

        return cls(
            role=role,
            content="\n".join(content_parts),
            tool_results=tool_results,
        )


@dataclass
class UniversalTool:
    """Universal tool format for cross-protocol use."""
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    handler: Callable | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_a2a_skill(self) -> dict[str, Any]:
        """Convert to A2A skill format."""
        return {
            "id": self.name,
            "name": self.name.replace("_", " ").title(),
            "description": self.description,
            "inputModes": ["text"],
            "outputModes": ["text"],
        }

    def to_mcp_tool(self) -> dict[str, Any]:
        """Convert to MCP tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters,
        }

    def to_openai_function(self) -> dict[str, Any]:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_claude_tool(self) -> dict[str, Any]:
        """Convert to Claude tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    @classmethod
    def from_a2a_skill(cls, skill: dict[str, Any]) -> "UniversalTool":
        """Create from A2A skill."""
        return cls(
            name=skill.get("id", skill.get("name", "")),
            description=skill.get("description", ""),
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Input text"},
                },
            },
        )

    @classmethod
    def from_mcp_tool(cls, tool: dict[str, Any]) -> "UniversalTool":
        """Create from MCP tool."""
        return cls(
            name=tool.get("name", ""),
            description=tool.get("description", ""),
            parameters=tool.get("inputSchema", {}),
        )

    @classmethod
    def from_openai_function(cls, func: dict[str, Any]) -> "UniversalTool":
        """Create from OpenAI function."""
        func_def = func.get("function", func)
        return cls(
            name=func_def.get("name", ""),
            description=func_def.get("description", ""),
            parameters=func_def.get("parameters", {}),
        )

    @classmethod
    def from_claude_tool(cls, tool: dict[str, Any]) -> "UniversalTool":
        """Create from Claude tool."""
        return cls(
            name=tool.get("name", ""),
            description=tool.get("description", ""),
            parameters=tool.get("input_schema", {}),
        )


class ProtocolAdapter(ABC):
    """Base class for protocol adapters."""

    @property
    @abstractmethod
    def source_protocol(self) -> ProtocolType:
        """Get source protocol type."""
        pass

    @property
    @abstractmethod
    def target_protocol(self) -> ProtocolType:
        """Get target protocol type."""
        pass

    @abstractmethod
    def convert_message(self, message: Any) -> Any:
        """Convert message between protocols."""
        pass

    @abstractmethod
    def convert_tool(self, tool: Any) -> Any:
        """Convert tool definition between protocols."""
        pass


class A2AtoMCPAdapter(ProtocolAdapter):
    """Adapter to convert A2A protocol to MCP protocol.

    Enables using A2A agents as MCP tool providers.

    Example:
        >>> adapter = A2AtoMCPAdapter()
        >>>
        >>> # Convert A2A message to MCP
        >>> mcp_msg = adapter.convert_message(a2a_message)
        >>>
        >>> # Convert A2A skill to MCP tool
        >>> mcp_tool = adapter.convert_tool(a2a_skill)
    """

    @property
    def source_protocol(self) -> ProtocolType:
        return ProtocolType.A2A

    @property
    def target_protocol(self) -> ProtocolType:
        return ProtocolType.MCP

    def convert_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Convert A2A message to MCP format."""
        universal = UniversalMessage.from_a2a(message)
        return universal.to_mcp_message()

    def convert_tool(self, skill: dict[str, Any]) -> dict[str, Any]:
        """Convert A2A skill to MCP tool."""
        universal = UniversalTool.from_a2a_skill(skill)
        return universal.to_mcp_tool()

    def convert_task_to_tool_call(
        self,
        task: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert A2A task to MCP tool call result."""
        # Extract message content
        messages = task.get("history", [])
        results = []

        for msg in messages:
            if msg.get("role") == "agent":
                for part in msg.get("parts", []):
                    if part.get("type") == "text":
                        results.append(part.get("text", ""))

        return {
            "content": [{
                "type": "text",
                "text": "\n".join(results),
            }],
        }

    def convert_agent_card(
        self,
        card: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Convert A2A agent card to MCP tools list."""
        tools = []
        for skill in card.get("skills", []):
            tools.append(self.convert_tool(skill))
        return tools

    @staticmethod
    def create_mcp_resource_from_artifact(
        artifact: dict[str, Any],
    ) -> dict[str, Any]:
        """Create MCP resource from A2A artifact."""
        return {
            "uri": f"a2a://{artifact.get('id', uuid.uuid4())}",
            "name": artifact.get("name", "Artifact"),
            "description": f"A2A artifact: {artifact.get('name', '')}",
            "mimeType": artifact.get("mimeType", "text/plain"),
            "text": str(artifact.get("content", "")),
        }


class MCPtoA2AAdapter(ProtocolAdapter):
    """Adapter to convert MCP protocol to A2A protocol.

    Enables exposing MCP servers as A2A agents.

    Example:
        >>> adapter = MCPtoA2AAdapter()
        >>>
        >>> # Convert MCP tool to A2A skill
        >>> a2a_skill = adapter.convert_tool(mcp_tool)
        >>>
        >>> # Create A2A agent card from MCP server
        >>> card = adapter.create_agent_card(mcp_server_info)
    """

    @property
    def source_protocol(self) -> ProtocolType:
        return ProtocolType.MCP

    @property
    def target_protocol(self) -> ProtocolType:
        return ProtocolType.A2A

    def convert_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Convert MCP message to A2A format."""
        # MCP doesn't have direct message format, interpret as content
        content = message.get("content", {})
        if isinstance(content, dict):
            text = content.get("text", "")
        else:
            text = str(content)

        return {
            "role": "user",
            "parts": [{"type": "text", "text": text}],
        }

    def convert_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Convert MCP tool to A2A skill."""
        universal = UniversalTool.from_mcp_tool(tool)
        return universal.to_a2a_skill()

    def create_agent_card(
        self,
        server_name: str,
        server_version: str,
        tools: list[dict[str, Any]],
        base_url: str = "",
    ) -> dict[str, Any]:
        """Create A2A agent card from MCP server info."""
        skills = [self.convert_tool(t) for t in tools]

        return {
            "name": server_name,
            "description": f"MCP Server: {server_name}",
            "url": base_url,
            "version": server_version,
            "protocolVersion": "0.3",
            "capabilities": {
                "streaming": False,
                "tasks": True,
            },
            "skills": skills,
            "defaultInputModes": ["text"],
            "defaultOutputModes": ["text"],
        }

    def convert_tool_result(
        self,
        result: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Convert MCP tool result to A2A message."""
        parts = []
        for item in result:
            if item.get("type") == "text":
                parts.append({"type": "text", "text": item.get("text", "")})
            elif item.get("type") == "resource":
                resource = item.get("resource", {})
                parts.append({
                    "type": "data",
                    "data": {
                        "uri": resource.get("uri"),
                        "content": resource.get("text"),
                    },
                })

        return {
            "role": "agent",
            "parts": parts,
        }


class OpenAIAgentsToA2AAdapter(ProtocolAdapter):
    """Adapter to expose OpenAI Agents SDK agents as A2A agents.

    Enables interoperability between OpenAI Agents patterns and
    the A2A protocol.

    Example:
        >>> from agentic_workflows.openai_agents import create_agent
        >>> agent = create_agent(name="helper", instructions="...")
        >>>
        >>> adapter = OpenAIAgentsToA2AAdapter(agent)
        >>> a2a_card = adapter.get_agent_card()
        >>> a2a_server = adapter.create_server()
    """

    def __init__(self, agent: Any = None):
        """Initialize adapter with optional agent.

        Args:
            agent: OpenAI-style agent to adapt.
        """
        self._agent = agent

    @property
    def source_protocol(self) -> ProtocolType:
        return ProtocolType.OPENAI_AGENTS

    @property
    def target_protocol(self) -> ProtocolType:
        return ProtocolType.A2A

    def convert_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI message to A2A format."""
        universal = UniversalMessage.from_openai(message)
        return universal.to_a2a_message()

    def convert_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI function to A2A skill."""
        universal = UniversalTool.from_openai_function(tool)
        return universal.to_a2a_skill()

    def get_agent_card(self, base_url: str = "") -> dict[str, Any]:
        """Get A2A agent card for the OpenAI agent."""
        if self._agent is None:
            return {}

        # Extract agent info
        name = getattr(self._agent, "name", "agent")
        description = getattr(self._agent, "instructions", "")[:200]
        tools = getattr(self._agent, "tools", [])

        # Convert tools to skills
        skills = []
        for tool in tools:
            if callable(tool):
                # Function tool
                skills.append({
                    "id": getattr(tool, "__name__", "tool"),
                    "name": getattr(tool, "__name__", "Tool"),
                    "description": getattr(tool, "__doc__", "")[:100],
                    "inputModes": ["text"],
                    "outputModes": ["text"],
                })
            elif isinstance(tool, dict):
                skills.append(self.convert_tool(tool))

        # Add handoffs as skills
        handoffs = getattr(self._agent, "handoffs", [])
        for handoff in handoffs:
            handoff_name = getattr(handoff, "name", "handoff")
            skills.append({
                "id": f"handoff_to_{handoff_name}",
                "name": f"Handoff to {handoff_name}",
                "description": f"Transfer conversation to {handoff_name}",
                "inputModes": ["text"],
                "outputModes": ["text"],
            })

        return {
            "name": name,
            "description": description,
            "url": base_url,
            "version": "1.0.0",
            "protocolVersion": "0.3",
            "capabilities": {
                "streaming": True,
                "tasks": True,
                "handoffs": len(handoffs) > 0,
            },
            "skills": skills,
            "defaultInputModes": ["text"],
            "defaultOutputModes": ["text"],
        }

    def convert_run_result(
        self,
        result: Any,
    ) -> dict[str, Any]:
        """Convert OpenAI run result to A2A task."""
        output = getattr(result, "output", "")
        if callable(output):
            output = str(output)

        return {
            "id": str(uuid.uuid4()),
            "status": {"state": "completed"},
            "history": [{
                "role": "agent",
                "parts": [{"type": "text", "text": str(output)}],
            }],
        }

    async def handle_a2a_task(
        self,
        task_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle an incoming A2A task using the OpenAI agent.

        Args:
            task_params: A2A task parameters.

        Returns:
            A2A task result.
        """
        if self._agent is None:
            return {
                "id": task_params.get("id", str(uuid.uuid4())),
                "status": {"state": "failed", "error": "No agent configured"},
            }

        try:
            # Extract input from A2A message
            message = task_params.get("message", {})
            parts = message.get("parts", [])
            text_parts = [p.get("text", "") for p in parts if p.get("type") == "text"]
            user_input = "\n".join(text_parts)

            # Run the agent
            from agentic_workflows.openai_agents import run

            result = await run(self._agent, user_input)

            return self.convert_run_result(result)

        except Exception as e:
            return {
                "id": task_params.get("id", str(uuid.uuid4())),
                "status": {"state": "failed", "error": str(e)},
            }


class A2AtoOpenAIAdapter(ProtocolAdapter):
    """Adapter to use A2A agents with OpenAI-style patterns.

    Enables calling A2A agents using OpenAI function calling conventions.
    """

    def __init__(self, a2a_client: Any = None):
        """Initialize adapter.

        Args:
            a2a_client: A2A client for remote agent calls.
        """
        self._client = a2a_client

    @property
    def source_protocol(self) -> ProtocolType:
        return ProtocolType.A2A

    @property
    def target_protocol(self) -> ProtocolType:
        return ProtocolType.OPENAI_AGENTS

    def convert_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Convert A2A message to OpenAI format."""
        universal = UniversalMessage.from_a2a(message)
        return universal.to_openai_message()

    def convert_tool(self, skill: dict[str, Any]) -> dict[str, Any]:
        """Convert A2A skill to OpenAI function."""
        universal = UniversalTool.from_a2a_skill(skill)
        return universal.to_openai_function()

    def convert_agent_to_function(
        self,
        card: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert A2A agent card to OpenAI function definition.

        The function represents calling the remote A2A agent.
        """
        return {
            "type": "function",
            "function": {
                "name": f"call_agent_{card.get('name', 'agent').replace(' ', '_').lower()}",
                "description": card.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to send to the agent",
                        },
                        "skill_id": {
                            "type": "string",
                            "description": "Specific skill to invoke",
                            "enum": [s.get("id") for s in card.get("skills", [])],
                        },
                    },
                    "required": ["message"],
                },
            },
        }


class ClaudeToOpenAIAdapter(ProtocolAdapter):
    """Adapter between Claude tools and OpenAI functions.

    Enables using Claude tool definitions with OpenAI-style code.
    """

    @property
    def source_protocol(self) -> ProtocolType:
        return ProtocolType.CLAUDE_TOOLS

    @property
    def target_protocol(self) -> ProtocolType:
        return ProtocolType.OPENAI_FUNCTIONS

    def convert_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Convert Claude message to OpenAI format."""
        universal = UniversalMessage.from_claude(message)
        return universal.to_openai_message()

    def convert_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Convert Claude tool to OpenAI function."""
        universal = UniversalTool.from_claude_tool(tool)
        return universal.to_openai_function()


class OpenAIToClaudeAdapter(ProtocolAdapter):
    """Adapter between OpenAI functions and Claude tools.

    Enables using OpenAI function definitions with Claude.
    """

    @property
    def source_protocol(self) -> ProtocolType:
        return ProtocolType.OPENAI_FUNCTIONS

    @property
    def target_protocol(self) -> ProtocolType:
        return ProtocolType.CLAUDE_TOOLS

    def convert_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI message to Claude format."""
        universal = UniversalMessage.from_openai(message)
        return universal.to_claude_message()

    def convert_tool(self, func: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI function to Claude tool."""
        universal = UniversalTool.from_openai_function(func)
        return universal.to_claude_tool()


class ProtocolBridge:
    """Central bridge for all protocol conversions.

    Provides a unified interface for converting between any
    supported protocols.

    Example:
        >>> bridge = ProtocolBridge()
        >>>
        >>> # Convert A2A to MCP
        >>> mcp_tool = bridge.convert_tool(
        ...     a2a_skill,
        ...     ProtocolType.A2A,
        ...     ProtocolType.MCP,
        ... )
        >>>
        >>> # Convert OpenAI to Claude
        >>> claude_msg = bridge.convert_message(
        ...     openai_msg,
        ...     ProtocolType.OPENAI_FUNCTIONS,
        ...     ProtocolType.CLAUDE_TOOLS,
        ... )
    """

    def __init__(self):
        """Initialize protocol bridge."""
        self._adapters: dict[tuple[ProtocolType, ProtocolType], ProtocolAdapter] = {}
        self._register_default_adapters()

    def _register_default_adapters(self) -> None:
        """Register default adapters."""
        self._adapters[(ProtocolType.A2A, ProtocolType.MCP)] = A2AtoMCPAdapter()
        self._adapters[(ProtocolType.MCP, ProtocolType.A2A)] = MCPtoA2AAdapter()
        self._adapters[(ProtocolType.OPENAI_AGENTS, ProtocolType.A2A)] = OpenAIAgentsToA2AAdapter()
        self._adapters[(ProtocolType.A2A, ProtocolType.OPENAI_AGENTS)] = A2AtoOpenAIAdapter()
        self._adapters[(ProtocolType.CLAUDE_TOOLS, ProtocolType.OPENAI_FUNCTIONS)] = ClaudeToOpenAIAdapter()
        self._adapters[(ProtocolType.OPENAI_FUNCTIONS, ProtocolType.CLAUDE_TOOLS)] = OpenAIToClaudeAdapter()

    def register_adapter(
        self,
        source: ProtocolType,
        target: ProtocolType,
        adapter: ProtocolAdapter,
    ) -> None:
        """Register a custom adapter.

        Args:
            source: Source protocol.
            target: Target protocol.
            adapter: Adapter instance.
        """
        self._adapters[(source, target)] = adapter

    def get_adapter(
        self,
        source: ProtocolType,
        target: ProtocolType,
    ) -> ProtocolAdapter | None:
        """Get adapter for protocol pair.

        Args:
            source: Source protocol.
            target: Target protocol.

        Returns:
            Adapter or None if not found.
        """
        return self._adapters.get((source, target))

    def convert_message(
        self,
        message: dict[str, Any],
        source: ProtocolType,
        target: ProtocolType,
    ) -> dict[str, Any]:
        """Convert message between protocols.

        Args:
            message: Source message.
            source: Source protocol.
            target: Target protocol.

        Returns:
            Converted message.

        Raises:
            ValueError: If no adapter found.
        """
        adapter = self.get_adapter(source, target)
        if adapter is None:
            raise ValueError(f"No adapter for {source.value} -> {target.value}")
        return adapter.convert_message(message)

    def convert_tool(
        self,
        tool: dict[str, Any],
        source: ProtocolType,
        target: ProtocolType,
    ) -> dict[str, Any]:
        """Convert tool definition between protocols.

        Args:
            tool: Source tool definition.
            source: Source protocol.
            target: Target protocol.

        Returns:
            Converted tool definition.

        Raises:
            ValueError: If no adapter found.
        """
        adapter = self.get_adapter(source, target)
        if adapter is None:
            raise ValueError(f"No adapter for {source.value} -> {target.value}")
        return adapter.convert_tool(tool)

    def convert_tools(
        self,
        tools: list[dict[str, Any]],
        source: ProtocolType,
        target: ProtocolType,
    ) -> list[dict[str, Any]]:
        """Convert multiple tools between protocols.

        Args:
            tools: Source tool definitions.
            source: Source protocol.
            target: Target protocol.

        Returns:
            List of converted tool definitions.
        """
        return [self.convert_tool(t, source, target) for t in tools]

    def can_convert(
        self,
        source: ProtocolType,
        target: ProtocolType,
    ) -> bool:
        """Check if conversion is supported.

        Args:
            source: Source protocol.
            target: Target protocol.

        Returns:
            True if adapter exists.
        """
        return (source, target) in self._adapters


# Global bridge instance
_global_bridge: ProtocolBridge | None = None


def get_protocol_bridge() -> ProtocolBridge:
    """Get or create the global protocol bridge."""
    global _global_bridge
    if _global_bridge is None:
        _global_bridge = ProtocolBridge()
    return _global_bridge


def convert_message(
    message: dict[str, Any],
    source: ProtocolType | str,
    target: ProtocolType | str,
) -> dict[str, Any]:
    """Convert message between protocols.

    Convenience function using global bridge.
    """
    if isinstance(source, str):
        source = ProtocolType(source)
    if isinstance(target, str):
        target = ProtocolType(target)
    return get_protocol_bridge().convert_message(message, source, target)


def convert_tool(
    tool: dict[str, Any],
    source: ProtocolType | str,
    target: ProtocolType | str,
) -> dict[str, Any]:
    """Convert tool between protocols.

    Convenience function using global bridge.
    """
    if isinstance(source, str):
        source = ProtocolType(source)
    if isinstance(target, str):
        target = ProtocolType(target)
    return get_protocol_bridge().convert_tool(tool, source, target)
