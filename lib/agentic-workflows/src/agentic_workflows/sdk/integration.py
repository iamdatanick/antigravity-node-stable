"""
Main SDK Integration Module for Agentic Workflows.

Provides the AgenticWorkflowsSDK class that integrates all components
for use with Claude Agent SDK.

Usage:
    from agentic_workflows.sdk import AgenticWorkflowsSDK

    # Initialize SDK with agentic workflows
    sdk = AgenticWorkflowsSDK()

    # Get options for ClaudeSDKClient
    options = sdk.get_claude_options()

    # Use with claude_agent_sdk
    async with ClaudeSDKClient(options=options) as client:
        result = await client.query("Analyze this codebase")
"""

from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import os


class IntegrationMode(Enum):
    """Integration modes for the SDK."""
    FULL = "full"  # All features enabled
    MINIMAL = "minimal"  # Core features only
    SECURITY = "security"  # Security-focused
    AUDIT = "audit"  # Audit/compliance focused


@dataclass
class WorkflowConfig:
    """Configuration for workflow sessions."""
    security_scope: int = 2  # Default: Tool Access
    enable_persistence: bool = True
    enable_audit: bool = True
    enable_telemetry: bool = True
    max_iterations: int = 20
    timeout_seconds: int = 300
    model_preference: str = "sonnet"  # Default model
    allowed_tools: list[str] = field(default_factory=list)
    blocked_tools: list[str] = field(default_factory=list)


@dataclass
class WorkflowSession:
    """Active workflow session with context."""
    session_id: str
    config: WorkflowConfig
    context: dict[str, Any] = field(default_factory=dict)
    scratchpad: Any = None  # Scratchpad instance
    learning_graph: Any = None  # LearningContextGraph instance
    context_graph: Any = None  # ContextGraph instance


class AgenticWorkflowsSDK:
    """Main SDK class for Agentic Workflows integration.

    Provides a unified interface for:
    - Agent definitions
    - MCP tools
    - Hooks
    - Context persistence
    - Workflow orchestration

    Example:
        sdk = AgenticWorkflowsSDK(mode=IntegrationMode.FULL)
        options = sdk.get_claude_options()

        # Custom configuration
        sdk = AgenticWorkflowsSDK(
            security_scope=3,
            enable_expert_panel=True
        )
    """

    def __init__(
        self,
        mode: IntegrationMode = IntegrationMode.FULL,
        security_scope: int = 2,
        enable_expert_panel: bool = True,
        enable_persistence: bool = True,
        custom_agents: dict[str, dict] | None = None,
        custom_tools: list[dict] | None = None,
    ):
        """Initialize the SDK.

        Args:
            mode: Integration mode (full, minimal, security, audit)
            security_scope: Maximum security scope (1-4)
            enable_expert_panel: Include expert panel agents
            enable_persistence: Enable context persistence
            custom_agents: Additional custom agent definitions
            custom_tools: Additional custom tool definitions
        """
        self.mode = mode
        self.security_scope = security_scope
        self.enable_expert_panel = enable_expert_panel
        self.enable_persistence = enable_persistence
        self.custom_agents = custom_agents or {}
        self.custom_tools = custom_tools or []

        # Active sessions
        self._sessions: dict[str, WorkflowSession] = {}

    def get_claude_options(self) -> dict[str, Any]:
        """Get ClaudeAgentOptions-compatible configuration.

        Returns:
            Dictionary with agents, mcp_servers, allowed_tools, hooks

        Usage:
            options = sdk.get_claude_options()
            # Pass to ClaudeAgentOptions(**options)
        """
        from .agents import ALL_SDK_AGENTS, EXPERT_PANEL_AGENTS
        from .tools import create_agentic_mcp_server, get_tool_names
        from .hooks import get_all_hooks, create_scoped_hooks

        # Select agents based on mode
        if self.mode == IntegrationMode.MINIMAL:
            agents = {
                "expert-analyst": ALL_SDK_AGENTS["expert-analyst"],
                "checker": ALL_SDK_AGENTS["checker"],
            }
        elif self.mode == IntegrationMode.SECURITY:
            agents = {
                "security-auditor": ALL_SDK_AGENTS["security-auditor"],
                "risk-assessor": ALL_SDK_AGENTS["risk-assessor"],
                "checker": ALL_SDK_AGENTS["checker"],
            }
        elif self.mode == IntegrationMode.AUDIT:
            agents = {
                "note-taker": ALL_SDK_AGENTS["note-taker"],
                "checker": ALL_SDK_AGENTS["checker"],
                "code-reviewer": ALL_SDK_AGENTS["code-reviewer"],
            }
        else:  # FULL mode
            agents = dict(ALL_SDK_AGENTS)

        # Add expert panel if enabled
        if self.enable_expert_panel and self.mode == IntegrationMode.FULL:
            agents.update(EXPERT_PANEL_AGENTS)

        # Add custom agents
        agents.update(self.custom_agents)

        # Select hooks based on security scope
        hooks = create_scoped_hooks(max_scope=self.security_scope)

        # Build allowed tools list
        allowed_tools = [
            # Core Claude Code tools
            "Task", "Read", "Write", "Edit", "Glob", "Grep",
            "Bash", "WebSearch", "WebFetch",
        ]

        # Add MCP tools
        allowed_tools.extend(get_tool_names())

        # Filter based on security scope
        if self.security_scope < 3:
            # Remove autonomous tools
            allowed_tools = [t for t in allowed_tools if t not in ["Bash", "Task"]]

        return {
            "agents": agents,
            "mcp_servers": {
                "agentic": create_agentic_mcp_server(),
            },
            "allowed_tools": allowed_tools,
            "hooks": hooks,
        }

    def get_agent_definitions(self) -> dict[str, dict[str, Any]]:
        """Get all agent definitions.

        Returns:
            Dictionary of agent name to AgentDefinition
        """
        from .agents import ALL_SDK_AGENTS

        agents = dict(ALL_SDK_AGENTS)
        agents.update(self.custom_agents)
        return agents

    def get_expert_panel_config(self) -> dict[str, Any]:
        """Get Expert Panel workflow configuration.

        Returns:
            Configuration for running Expert Panel workflow
        """
        return {
            "phases": [
                {
                    "name": "Analysis",
                    "agents": ["expert-analyst"],
                    "model": "opus",
                    "output": "analysis_report",
                },
                {
                    "name": "Architecture",
                    "agents": ["expert-architect"],
                    "model": "opus",
                    "output": "design_document",
                },
                {
                    "name": "Risk Assessment",
                    "agents": ["risk-assessor"],
                    "model": "sonnet",
                    "output": "risk_matrix",
                },
                {
                    "name": "Execution",
                    "agents": ["checker", "corrector"],
                    "model": "sonnet",
                    "output": "validation_report",
                },
            ],
            "handoff_format": "structured",
            "enable_debate": True,
            "quality_gates": True,
        }

    def create_session(
        self,
        session_id: str | None = None,
        config: WorkflowConfig | None = None,
    ) -> WorkflowSession:
        """Create a new workflow session with context tracking.

        Args:
            session_id: Unique session identifier (auto-generated if not provided)
            config: Session configuration

        Returns:
            WorkflowSession instance
        """
        import uuid
        from agentic_workflows.core.scratchpad import Scratchpad
        from agentic_workflows.core.context_graph import LearningContextGraph

        session_id = session_id or str(uuid.uuid4())[:8]
        config = config or WorkflowConfig(security_scope=self.security_scope)

        session = WorkflowSession(
            session_id=session_id,
            config=config,
            scratchpad=Scratchpad(),
            learning_graph=LearningContextGraph(),
        )

        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> WorkflowSession | None:
        """Get an existing session by ID.

        Args:
            session_id: Session identifier

        Returns:
            WorkflowSession or None if not found
        """
        return self._sessions.get(session_id)

    def save_session(self, session_id: str) -> bool:
        """Save session context to persistent storage.

        Args:
            session_id: Session to save

        Returns:
            True if saved successfully
        """
        if not self.enable_persistence:
            return False

        session = self._sessions.get(session_id)
        if not session:
            return False

        try:
            from agentic_workflows.core.persistence import FileContextPersistence

            persistence = FileContextPersistence()

            if session.scratchpad:
                persistence.save_scratchpad(session_id, session.scratchpad)

            if session.learning_graph:
                persistence.save_learning_graph(session_id, session.learning_graph)

            if session.context_graph:
                persistence.save_context_graph(session_id, session.context_graph)

            return True
        except Exception:
            return False

    def load_session(self, session_id: str) -> WorkflowSession | None:
        """Load a session from persistent storage.

        Args:
            session_id: Session to load

        Returns:
            WorkflowSession or None if not found
        """
        if not self.enable_persistence:
            return None

        try:
            from agentic_workflows.core.persistence import FileContextPersistence

            persistence = FileContextPersistence()

            scratchpad = persistence.load_scratchpad(session_id)
            learning_graph = persistence.load_learning_graph(session_id)
            context_graph = persistence.load_context_graph(session_id)

            if scratchpad or learning_graph or context_graph:
                session = WorkflowSession(
                    session_id=session_id,
                    config=WorkflowConfig(security_scope=self.security_scope),
                    scratchpad=scratchpad,
                    learning_graph=learning_graph,
                    context_graph=context_graph,
                )
                self._sessions[session_id] = session
                return session

            return None
        except Exception:
            return None


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def get_agent_definitions() -> dict[str, dict[str, Any]]:
    """Get all agent definitions for SDK registration.

    Returns:
        Dictionary of agent definitions
    """
    from .agents import ALL_SDK_AGENTS
    return dict(ALL_SDK_AGENTS)


def get_hooks_config(security_scope: int = 2) -> dict[str, list]:
    """Get hooks configuration for SDK.

    Args:
        security_scope: Maximum security scope (1-4)

    Returns:
        Dictionary of hook type to callbacks
    """
    from .hooks import create_scoped_hooks
    return create_scoped_hooks(max_scope=security_scope)


def create_workflow_session(
    session_id: str | None = None,
    security_scope: int = 2,
    enable_persistence: bool = True,
) -> WorkflowSession:
    """Create a workflow session with default configuration.

    Args:
        session_id: Optional session identifier
        security_scope: Security scope level (1-4)
        enable_persistence: Enable context persistence

    Returns:
        WorkflowSession instance
    """
    sdk = AgenticWorkflowsSDK(
        security_scope=security_scope,
        enable_persistence=enable_persistence,
    )
    return sdk.create_session(session_id)


def get_mcp_server_config() -> dict[str, Any]:
    """Get MCP server configuration for claude settings.

    Returns:
        MCP server configuration dictionary

    Usage in .claude/settings.json:
        {
            "mcpServers": {
                "agentic-workflows": get_mcp_server_config()
            }
        }
    """
    from .tools import create_agentic_mcp_server
    return create_agentic_mcp_server()


def create_plugin_config() -> dict[str, Any]:
    """Generate plugin.json configuration for marketplace.

    Returns:
        Plugin configuration dictionary
    """
    from .agents import ALL_SDK_AGENTS
    from .tools import ALL_TOOLS

    return {
        "name": "agentic-workflows",
        "version": "4.1.0",
        "description": "Intelligent multi-agent workflows with Expert Panel, security controls, and context persistence",
        "author": "Agentic Workflows",
        "license": "MIT",
        "repository": "https://github.com/agentic-workflows/agentic-workflows",
        "keywords": [
            "agents", "workflows", "multi-agent", "expert-panel",
            "security", "orchestration", "mcp", "claude-code"
        ],
        "engines": {
            "claude-code": ">=1.0.0"
        },
        "main": "src/agentic_workflows/__init__.py",
        "agents": list(ALL_SDK_AGENTS.keys()),
        "tools": [t["name"] for t in ALL_TOOLS],
        "capabilities": [
            "multi-agent-orchestration",
            "expert-panel-workflow",
            "security-validation",
            "context-persistence",
            "audit-logging",
            "scope-enforcement",
        ],
        "mcp_server": {
            "command": "python",
            "args": ["-m", "agentic_workflows.mcp_server"]
        },
        "settings_schema": {
            "type": "object",
            "properties": {
                "security_scope": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 4,
                    "default": 2,
                    "description": "Maximum security scope (1=stateless, 2=tool_access, 3=autonomous, 4=full)"
                },
                "enable_persistence": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable context persistence between sessions"
                },
                "enable_audit": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable audit logging"
                },
            }
        }
    }
