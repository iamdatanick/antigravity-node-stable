"""
SDK-Compatible Tools using @tool decorator.

Provides MCP tools compatible with Claude Agent SDK's create_sdk_mcp_server().

Usage:
    from agentic_workflows.sdk.tools import create_agentic_mcp_server
    from claude_agent_sdk import ClaudeAgentOptions

    # Create MCP server with all tools
    server = create_agentic_mcp_server()

    options = ClaudeAgentOptions(
        mcp_servers={"agentic": server},
        allowed_tools=[
            "mcp__agentic__check_injection",
            "mcp__agentic__list_agents",
            # ... etc
        ]
    )
"""

from typing import Any
import json

# Note: In production, these would use the actual claude_agent_sdk imports
# For now, we provide the tool definitions and factory functions


# =============================================================================
# TOOL DEFINITIONS (SDK @tool decorator format)
# =============================================================================

def _make_tool_schema(name: str, description: str, input_schema: dict) -> dict:
    """Create a tool schema compatible with SDK format."""
    return {
        "name": name,
        "description": description,
        "input_schema": input_schema,
    }


# Security Tools
check_injection_tool = _make_tool_schema(
    name="check_injection",
    description="Check text for prompt injection attacks. Returns safety status and reason.",
    input_schema={
        "text": str,
    }
)

redact_sensitive_tool = _make_tool_schema(
    name="redact_sensitive",
    description="Redact sensitive data (PII, PHI, PCI, credentials) from text.",
    input_schema={
        "text": str,
        "categories": list,  # Optional: ["pii", "phi", "pci", "credentials"]
    }
)

validate_scope_tool = _make_tool_schema(
    name="validate_scope",
    description="Validate if an operation is allowed within a security scope (1-4).",
    input_schema={
        "tool_name": str,
        "scope": int,  # 1-4
    }
)

# Agent Tools
list_agents_tool = _make_tool_schema(
    name="list_agents",
    description="List all 118 agent templates with categories and security scopes.",
    input_schema={
        "category": str,  # Optional filter
        "scope": int,  # Optional filter
    }
)

get_agent_tool = _make_tool_schema(
    name="get_agent",
    description="Get detailed information about a specific agent template.",
    input_schema={
        "name": str,
    }
)

audit_agents_tool = _make_tool_schema(
    name="audit_agents",
    description="Run comprehensive audit of all agent templates for SKILL.md v4.1 compliance.",
    input_schema={}
)

# Context/Persistence Tools
save_context_tool = _make_tool_schema(
    name="save_context",
    description="Save scratchpad, learning graph, or context graph to persistent storage.",
    input_schema={
        "context_type": str,  # "scratchpad", "learning", "context"
        "identifier": str,
        "data": dict,
    }
)

load_context_tool = _make_tool_schema(
    name="load_context",
    description="Load previously saved context from persistent storage.",
    input_schema={
        "context_type": str,
        "identifier": str,
    }
)

# Workflow Tools
run_expert_panel_tool = _make_tool_schema(
    name="run_expert_panel",
    description="Execute Expert Panel workflow with analyst, architect, and risk assessor for comprehensive task analysis.",
    input_schema={
        "task": str,
        "context": dict,  # Optional additional context
    }
)

# Metrics Tools
record_usage_tool = _make_tool_schema(
    name="record_usage",
    description="Record token usage and cost for an agent API call.",
    input_schema={
        "agent_id": str,
        "model": str,  # "opus", "sonnet", "haiku"
        "input_tokens": int,
        "output_tokens": int,
    }
)

get_cost_summary_tool = _make_tool_schema(
    name="get_cost_summary",
    description="Get usage summary with cost breakdown by model and agent.",
    input_schema={}
)


# =============================================================================
# TOOL HANDLERS
# =============================================================================

async def handle_check_injection(args: dict[str, Any]) -> dict[str, Any]:
    """Handle check_injection tool call."""
    from agentic_workflows.skills.security import check_injection

    text = args.get("text", "")
    is_safe, reason = check_injection(text)

    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "is_safe": is_safe,
                "reason": reason,
                "input_length": len(text),
            }, indent=2)
        }]
    }


async def handle_redact_sensitive(args: dict[str, Any]) -> dict[str, Any]:
    """Handle redact_sensitive tool call."""
    from agentic_workflows.skills.security import SensitiveDataFilter

    text = args.get("text", "")
    filter = SensitiveDataFilter()
    matches = filter.scan(text)
    redacted = filter.redact(text)

    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "redacted_text": redacted,
                "findings_count": len(matches),
                "categories_found": list(set(m.category.value for m in matches)) if matches else [],
            }, indent=2)
        }]
    }


async def handle_validate_scope(args: dict[str, Any]) -> dict[str, Any]:
    """Handle validate_scope tool call."""
    from agentic_workflows.skills.security import ScopeValidator, SecurityScope

    tool_name = args.get("tool_name", "")
    scope = args.get("scope", 2)

    scope_map = {
        1: SecurityScope.STATELESS,
        2: SecurityScope.TOOL_ACCESS,
        3: SecurityScope.AUTONOMOUS,
        4: SecurityScope.FULL_AUTONOMY,
    }

    validator = ScopeValidator()
    security_scope = scope_map.get(scope, SecurityScope.TOOL_ACCESS)
    allowed, reason = validator.validate_tool_call(tool_name, security_scope)

    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "allowed": allowed,
                "reason": reason,
                "tool": tool_name,
                "scope": scope,
            }, indent=2)
        }]
    }


async def handle_list_agents(args: dict[str, Any]) -> dict[str, Any]:
    """Handle list_agents tool call."""
    from agentic_workflows.agents.templates import (
        list_agent_templates,
        get_agents_by_category,
        get_agents_by_scope,
    )

    category = args.get("category")
    scope = args.get("scope")

    if category:
        agents = get_agents_by_category(category)
    elif scope:
        agents = get_agents_by_scope(scope)
    else:
        agents = list_agent_templates()

    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "count": len(agents),
                "agents": agents,
            }, indent=2)
        }]
    }


async def handle_get_agent(args: dict[str, Any]) -> dict[str, Any]:
    """Handle get_agent tool call."""
    from agentic_workflows.agents.templates import get_agent_template, ALL_AGENTS

    name = args.get("name", "")
    template = get_agent_template(name)

    if template:
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "found": True,
                    "agent": template,
                }, indent=2)
            }]
        }
    else:
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "found": False,
                    "error": f"Agent '{name}' not found",
                    "available_agents": list(ALL_AGENTS.keys())[:20],
                }, indent=2)
            }]
        }


async def handle_audit_agents(args: dict[str, Any]) -> dict[str, Any]:
    """Handle audit_agents tool call."""
    from agentic_workflows.agents.templates import get_audit_summary

    summary = get_audit_summary()

    return {
        "content": [{
            "type": "text",
            "text": json.dumps(summary, indent=2)
        }]
    }


async def handle_save_context(args: dict[str, Any]) -> dict[str, Any]:
    """Handle save_context tool call."""
    from agentic_workflows.core.persistence import FileContextPersistence
    from agentic_workflows.core.scratchpad import Scratchpad
    from agentic_workflows.core.context_graph import LearningContextGraph
    from agentic_workflows.context.graph import ContextGraph

    persistence = FileContextPersistence()
    context_type = args.get("context_type", "")
    identifier = args.get("identifier", "")
    data = args.get("data", {})

    if context_type == "scratchpad":
        scratchpad = Scratchpad()
        scratchpad.import_data(data)
        filepath = persistence.save_scratchpad(identifier, scratchpad)
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "saved": True,
                    "type": "scratchpad",
                    "identifier": identifier,
                    "filepath": filepath,
                }, indent=2)
            }]
        }
    elif context_type == "learning":
        graph = LearningContextGraph()
        graph.import_data(data)
        filepath = persistence.save_learning_graph(identifier, graph)
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "saved": True,
                    "type": "learning_graph",
                    "identifier": identifier,
                    "filepath": filepath,
                }, indent=2)
            }]
        }
    elif context_type == "context":
        graph = ContextGraph.from_dict(data)
        filepath = persistence.save_context_graph(identifier, graph)
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "saved": True,
                    "type": "context_graph",
                    "identifier": identifier,
                    "filepath": filepath,
                }, indent=2)
            }]
        }
    else:
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "error": f"Unknown context type: {context_type}",
                    "valid_types": ["scratchpad", "learning", "context"],
                }, indent=2)
            }],
            "is_error": True
        }


async def handle_load_context(args: dict[str, Any]) -> dict[str, Any]:
    """Handle load_context tool call."""
    from agentic_workflows.core.persistence import FileContextPersistence

    persistence = FileContextPersistence()
    context_type = args.get("context_type", "")
    identifier = args.get("identifier", "")

    if context_type == "scratchpad":
        scratchpad = persistence.load_scratchpad(identifier)
        if scratchpad:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "found": True,
                        "type": "scratchpad",
                        "summary": scratchpad.get_summary(),
                        "data": scratchpad.export(),
                    }, indent=2)
                }]
            }
    elif context_type == "learning":
        graph = persistence.load_learning_graph(identifier)
        if graph:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "found": True,
                        "type": "learning_graph",
                        "statistics": graph.get_statistics(),
                        "data": graph.export(),
                    }, indent=2)
                }]
            }
    elif context_type == "context":
        graph = persistence.load_context_graph(identifier)
        if graph:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "found": True,
                        "type": "context_graph",
                        "stats": graph.get_stats(),
                        "data": graph.to_dict(),
                    }, indent=2)
                }]
            }

    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "found": False,
                "type": context_type,
                "identifier": identifier,
            }, indent=2)
        }]
    }


async def handle_run_expert_panel(args: dict[str, Any]) -> dict[str, Any]:
    """Handle run_expert_panel tool call."""
    task = args.get("task", "")
    context = args.get("context", {})

    # This would integrate with the actual ExpertPanelWorkflow
    # For SDK usage, we return a structured prompt for the orchestrator
    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "workflow": "expert_panel",
                "phases": [
                    {
                        "phase": 1,
                        "name": "Expert Analysis",
                        "agents": ["expert-analyst", "expert-architect", "risk-assessor"],
                        "status": "pending"
                    },
                    {
                        "phase": 2,
                        "name": "Task Execution",
                        "agents": ["note-taker", "checker", "corrector"],
                        "status": "pending"
                    },
                    {
                        "phase": 3,
                        "name": "Handoff Generation",
                        "status": "pending"
                    }
                ],
                "task": task,
                "context": context,
                "instructions": "Use the orchestrator agent to coordinate this workflow. Start with expert-analyst for deep analysis, then expert-architect for design, then risk-assessor for validation.",
            }, indent=2)
        }]
    }


# =============================================================================
# TOOL REGISTRY
# =============================================================================

TOOL_HANDLERS = {
    "check_injection": handle_check_injection,
    "redact_sensitive": handle_redact_sensitive,
    "validate_scope": handle_validate_scope,
    "list_agents": handle_list_agents,
    "get_agent": handle_get_agent,
    "audit_agents": handle_audit_agents,
    "save_context": handle_save_context,
    "load_context": handle_load_context,
    "run_expert_panel": handle_run_expert_panel,
}

ALL_TOOLS = [
    check_injection_tool,
    redact_sensitive_tool,
    validate_scope_tool,
    list_agents_tool,
    get_agent_tool,
    audit_agents_tool,
    save_context_tool,
    load_context_tool,
    run_expert_panel_tool,
    record_usage_tool,
    get_cost_summary_tool,
]


def get_sdk_tools() -> list[dict]:
    """Get all tool definitions in SDK format.

    Returns:
        List of tool definition dictionaries
    """
    return ALL_TOOLS


def create_agentic_mcp_server() -> dict:
    """Create MCP server configuration for SDK.

    This returns a configuration that can be used with
    claude_agent_sdk's mcp_servers option.

    Returns:
        MCP server configuration dict

    Usage:
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(
            mcp_servers={"agentic": create_agentic_mcp_server()},
            allowed_tools=["mcp__agentic__check_injection", ...]
        )
    """
    return {
        "type": "stdio",
        "command": "python",
        "args": ["-m", "agentic_workflows.mcp_server"],
        "env": {},
    }


def get_tool_names(prefix: str = "mcp__agentic__") -> list[str]:
    """Get all tool names with MCP prefix.

    Args:
        prefix: MCP tool prefix (default: "mcp__agentic__")

    Returns:
        List of prefixed tool names
    """
    return [f"{prefix}{tool['name']}" for tool in ALL_TOOLS]
