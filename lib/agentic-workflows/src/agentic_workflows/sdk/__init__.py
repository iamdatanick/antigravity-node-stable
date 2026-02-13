"""
Agentic Workflows SDK Integration.

Provides seamless integration with Claude Agent SDK for:
- Programmatic agent definitions
- Custom MCP tools via @tool decorator
- SDK-compatible hooks
- Session management with ClaudeSDKClient

Usage:
    from agentic_workflows.sdk import (
        AgenticWorkflowsSDK,
        get_agent_definitions,
        get_sdk_tools,
        get_hooks_config,
    )

    # Initialize SDK with agentic workflows
    sdk = AgenticWorkflowsSDK()
    options = sdk.get_claude_options()

    # Use with claude_agent_sdk
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Run expert panel analysis")
"""

from .agents import (
    ALL_SDK_AGENTS,
    CHECKER_AGENT,
    CORRECTOR_AGENT,
    EXPERT_ANALYST_AGENT,
    EXPERT_ARCHITECT_AGENT,
    NOTE_TAKER_AGENT,
    ORCHESTRATOR_AGENT,
    RISK_ASSESSOR_AGENT,
)
from .hooks import (
    audit_post_tool_hook,
    context_save_hook,
    get_all_hooks,
    security_pre_tool_hook,
)
from .integration import (
    AgenticWorkflowsSDK,
    create_workflow_session,
    get_agent_definitions,
    get_hooks_config,
)
from .tools import (
    audit_agents_tool,
    # Individual tools
    check_injection_tool,
    create_agentic_mcp_server,
    get_agent_tool,
    get_sdk_tools,
    list_agents_tool,
    load_context_tool,
    redact_sensitive_tool,
    run_expert_panel_tool,
    save_context_tool,
    validate_scope_tool,
)

__all__ = [
    # Main SDK class
    "AgenticWorkflowsSDK",
    # Factory functions
    "get_agent_definitions",
    "get_sdk_tools",
    "get_hooks_config",
    "create_workflow_session",
    "create_agentic_mcp_server",
    # Agent definitions
    "EXPERT_ANALYST_AGENT",
    "EXPERT_ARCHITECT_AGENT",
    "RISK_ASSESSOR_AGENT",
    "NOTE_TAKER_AGENT",
    "CHECKER_AGENT",
    "CORRECTOR_AGENT",
    "ORCHESTRATOR_AGENT",
    "ALL_SDK_AGENTS",
    # Individual tools
    "check_injection_tool",
    "redact_sensitive_tool",
    "validate_scope_tool",
    "list_agents_tool",
    "get_agent_tool",
    "audit_agents_tool",
    "save_context_tool",
    "load_context_tool",
    "run_expert_panel_tool",
    # Hooks
    "security_pre_tool_hook",
    "audit_post_tool_hook",
    "context_save_hook",
    "get_all_hooks",
]
