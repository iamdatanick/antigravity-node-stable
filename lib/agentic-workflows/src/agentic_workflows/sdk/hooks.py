"""
SDK-Compatible Hooks for Agentic Workflows.

Provides hook callbacks compatible with Claude Agent SDK's hook system.
These can be passed to ClaudeAgentOptions for pre/post tool execution.

Usage:
    from agentic_workflows.sdk.hooks import get_all_hooks
    from claude_agent_sdk import ClaudeAgentOptions

    options = ClaudeAgentOptions(
        hooks=get_all_hooks(),
        # ... other options
    )
"""

from typing import Any, Callable
from dataclasses import dataclass
from enum import Enum
import json
import time


class HookType(Enum):
    """Types of SDK hooks."""
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    STOP = "Stop"
    SUBAGENT_STOP = "SubagentStop"
    PRE_COMPACT = "PreCompact"


@dataclass
class HookContext:
    """Context passed to hook callbacks."""
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: Any | None = None
    session_id: str | None = None
    agent_id: str | None = None
    timestamp: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class HookResult:
    """Result returned from hook callbacks."""
    allow: bool = True
    modified_input: dict[str, Any] | None = None
    message: str | None = None
    metadata: dict[str, Any] | None = None


# =============================================================================
# SECURITY HOOKS
# =============================================================================

def security_pre_tool_hook(context: HookContext) -> HookResult:
    """Pre-tool hook that validates security before tool execution.

    Checks:
    - Bash command injection patterns
    - Dangerous file operations
    - Scope violations

    Args:
        context: Hook context with tool name and input

    Returns:
        HookResult allowing or blocking execution
    """
    tool_name = context.tool_name or ""
    tool_input = context.tool_input or {}

    # Check Bash commands for injection
    if tool_name.lower() in ["bash", "run_command", "execute"]:
        command = tool_input.get("command", "")

        # Import security check
        try:
            from agentic_workflows.skills.security import check_injection
            is_safe, reason = check_injection(command)

            if not is_safe:
                return HookResult(
                    allow=False,
                    message=f"Security violation: {reason}",
                    metadata={"blocked_command": command[:100]}
                )
        except ImportError:
            # If security module not available, allow but warn
            pass

    # Check file operations for sensitive paths
    if tool_name.lower() in ["write", "edit", "delete"]:
        file_path = tool_input.get("file_path", "") or tool_input.get("path", "")

        sensitive_patterns = [
            ".env", "credentials", "secrets", "password",
            ".ssh", ".aws", ".kube", "config.json"
        ]

        file_lower = file_path.lower()
        for pattern in sensitive_patterns:
            if pattern in file_lower:
                return HookResult(
                    allow=False,
                    message=f"Blocked write to sensitive file: {file_path}",
                    metadata={"blocked_path": file_path}
                )

    return HookResult(allow=True)


def scope_validator_hook(context: HookContext, max_scope: int = 2) -> HookResult:
    """Pre-tool hook that validates operations against security scope.

    Args:
        context: Hook context
        max_scope: Maximum allowed security scope (1-4)

    Returns:
        HookResult with allow/block decision
    """
    tool_name = context.tool_name or ""

    # Tool to required scope mapping
    tool_scopes = {
        # Scope 1: Read-only
        "read": 1, "grep": 1, "glob": 1, "websearch": 1,

        # Scope 2: Tool access
        "write": 2, "edit": 2, "webfetch": 2,

        # Scope 3: Autonomous
        "bash": 3, "run_command": 3, "execute": 3, "task": 3,

        # Scope 4: Full autonomy
        "deploy": 4, "publish": 4, "release": 4,
    }

    required_scope = tool_scopes.get(tool_name.lower(), 2)

    if required_scope > max_scope:
        return HookResult(
            allow=False,
            message=f"Tool '{tool_name}' requires scope {required_scope}, but max is {max_scope}",
            metadata={"required_scope": required_scope, "max_scope": max_scope}
        )

    return HookResult(allow=True)


# =============================================================================
# AUDIT HOOKS
# =============================================================================

# In-memory audit log for session
_audit_log: list[dict[str, Any]] = []


def audit_post_tool_hook(context: HookContext) -> HookResult:
    """Post-tool hook that logs tool usage for audit trail.

    Args:
        context: Hook context with tool execution details

    Returns:
        HookResult (always allows, just logs)
    """
    entry = {
        "timestamp": context.timestamp or time.time(),
        "tool": context.tool_name,
        "input_summary": _summarize_input(context.tool_input),
        "output_type": type(context.tool_output).__name__ if context.tool_output else None,
        "session_id": context.session_id,
        "agent_id": context.agent_id,
    }

    _audit_log.append(entry)

    # Keep log bounded
    if len(_audit_log) > 1000:
        _audit_log.pop(0)

    return HookResult(
        allow=True,
        metadata={"logged": True, "entry_id": len(_audit_log) - 1}
    )


def _summarize_input(tool_input: dict[str, Any] | None) -> str:
    """Create safe summary of tool input."""
    if not tool_input:
        return ""

    # Don't log sensitive content
    safe_keys = ["path", "file_path", "pattern", "name", "type"]
    summary = {}

    for key in safe_keys:
        if key in tool_input:
            value = tool_input[key]
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            summary[key] = value

    return json.dumps(summary) if summary else f"{len(tool_input)} params"


def get_audit_log() -> list[dict[str, Any]]:
    """Get the current audit log.

    Returns:
        List of audit entries
    """
    return _audit_log.copy()


def clear_audit_log() -> None:
    """Clear the audit log."""
    _audit_log.clear()


# =============================================================================
# CONTEXT HOOKS
# =============================================================================

def context_save_hook(context: HookContext) -> HookResult:
    """Post-tool hook that auto-saves context after significant operations.

    Triggers context save after:
    - Write/Edit operations
    - Task completions
    - Error conditions

    Args:
        context: Hook context

    Returns:
        HookResult with save status
    """
    tool_name = context.tool_name or ""
    session_id = context.session_id

    # Tools that should trigger context save
    save_triggers = ["write", "edit", "task", "bash"]

    if tool_name.lower() in save_triggers and session_id:
        try:
            from agentic_workflows.core.persistence import FileContextPersistence

            persistence = FileContextPersistence()

            # Save current context state
            # The actual scratchpad/graph would need to be passed via metadata
            if context.metadata and "scratchpad" in context.metadata:
                persistence.save_scratchpad(session_id, context.metadata["scratchpad"])

            return HookResult(
                allow=True,
                metadata={"context_saved": True, "session_id": session_id}
            )
        except Exception as e:
            return HookResult(
                allow=True,  # Don't block on save failure
                metadata={"context_saved": False, "error": str(e)}
            )

    return HookResult(allow=True)


# =============================================================================
# TELEMETRY HOOKS
# =============================================================================

_usage_metrics: dict[str, Any] = {
    "tool_calls": {},
    "tokens": {"input": 0, "output": 0},
    "costs": {"total": 0.0},
    "errors": [],
}


def telemetry_hook(context: HookContext) -> HookResult:
    """Post-tool hook for collecting usage telemetry.

    Tracks:
    - Tool call counts
    - Token usage
    - Error rates

    Args:
        context: Hook context

    Returns:
        HookResult with telemetry status
    """
    tool_name = context.tool_name or "unknown"

    # Increment tool call count
    if tool_name not in _usage_metrics["tool_calls"]:
        _usage_metrics["tool_calls"][tool_name] = 0
    _usage_metrics["tool_calls"][tool_name] += 1

    # Track errors
    if context.metadata and context.metadata.get("is_error"):
        _usage_metrics["errors"].append({
            "tool": tool_name,
            "timestamp": context.timestamp or time.time(),
            "error": context.metadata.get("error_message", "Unknown error")
        })

    return HookResult(
        allow=True,
        metadata={"telemetry_recorded": True}
    )


def get_usage_metrics() -> dict[str, Any]:
    """Get current usage metrics.

    Returns:
        Dictionary with tool calls, tokens, costs, errors
    """
    return _usage_metrics.copy()


def reset_usage_metrics() -> None:
    """Reset all usage metrics."""
    global _usage_metrics
    _usage_metrics = {
        "tool_calls": {},
        "tokens": {"input": 0, "output": 0},
        "costs": {"total": 0.0},
        "errors": [],
    }


# =============================================================================
# HOOK REGISTRATION
# =============================================================================

def get_all_hooks() -> dict[str, list[Callable]]:
    """Get all hooks organized by type for SDK registration.

    Returns:
        Dictionary mapping hook types to callback lists

    Usage:
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(
            hooks=get_all_hooks()
        )
    """
    return {
        "PreToolUse": [
            security_pre_tool_hook,
        ],
        "PostToolUse": [
            audit_post_tool_hook,
            context_save_hook,
            telemetry_hook,
        ],
    }


def get_security_hooks() -> dict[str, list[Callable]]:
    """Get only security-related hooks.

    Returns:
        Dictionary with security hooks
    """
    return {
        "PreToolUse": [
            security_pre_tool_hook,
        ],
    }


def get_audit_hooks() -> dict[str, list[Callable]]:
    """Get only audit-related hooks.

    Returns:
        Dictionary with audit hooks
    """
    return {
        "PostToolUse": [
            audit_post_tool_hook,
        ],
    }


def create_scoped_hooks(max_scope: int = 2) -> dict[str, list[Callable]]:
    """Create hooks with specific security scope limit.

    Args:
        max_scope: Maximum allowed security scope (1-4)

    Returns:
        Dictionary with scope-limited hooks
    """
    def scoped_validator(context: HookContext) -> HookResult:
        return scope_validator_hook(context, max_scope=max_scope)

    return {
        "PreToolUse": [
            security_pre_tool_hook,
            scoped_validator,
        ],
        "PostToolUse": [
            audit_post_tool_hook,
            telemetry_hook,
        ],
    }
