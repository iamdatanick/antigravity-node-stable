"""Hooks system for Agentic Workflows.

Provides pre/post tool execution hooks, event handling, and workflow control.
Based on Claude Code hooks specification.

Usage:
    from agentic_workflows.hooks import HookRegistry, HookEvent, HookContext

    registry = HookRegistry()
    registry.register(HookEvent.PRE_TOOL_USE, my_hook)
    result = await registry.execute(HookEvent.PRE_TOOL_USE, context)
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class HookEvent(Enum):
    """Hook event types matching Claude Code SDK specification (12 events)."""

    # Tool lifecycle
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    POST_TOOL_USE_FAILURE = "PostToolUseFailure"

    # User interaction
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    NOTIFICATION = "Notification"

    # Session lifecycle
    SESSION_START = "SessionStart"
    SESSION_END = "SessionEnd"

    # Agent control
    STOP = "Stop"
    SUBAGENT_START = "SubagentStart"
    SUBAGENT_STOP = "SubagentStop"

    # System events
    PRE_COMPACT = "PreCompact"
    PERMISSION_REQUEST = "PermissionRequest"

    # Legacy/extended events
    ERROR = "Error"
    TIMEOUT = "Timeout"


class HookType(Enum):
    """Hook execution types."""

    COMMAND = "command"  # Shell command
    MCP = "mcp"  # MCP tool call
    PROMPT = "prompt"  # Prompt template
    AGENT = "agent"  # Invoke another agent
    FUNCTION = "function"  # Python function


class HookDecision(Enum):
    """Hook decision outcomes."""

    APPROVE = "approve"  # Allow the operation
    DENY = "deny"  # Block the operation
    MODIFY = "modify"  # Allow with modifications
    ASK = "ask"  # Prompt user for decision
    SKIP = "skip"  # Skip this hook, continue to next


@dataclass
class HookConfig:
    """Configuration for a hook."""

    matcher: str  # Pattern to match (glob or regex)
    type: HookType = HookType.COMMAND
    command: Optional[str] = None  # For command type
    mcp_tool: Optional[str] = None  # For MCP type
    prompt: Optional[str] = None  # For prompt type
    agent_name: Optional[str] = None  # For agent type
    function: Optional[Callable] = None  # For function type
    once: bool = False  # Only run once per session
    timeout_ms: int = 30000  # Timeout in milliseconds
    enabled: bool = True  # Whether hook is active
    priority: int = 0  # Higher priority runs first

    def matches(self, value: str) -> bool:
        """Check if matcher pattern matches the value."""
        # Exact match first (highest priority)
        if value == self.matcher:
            return True
        # Check if matcher has glob characters
        if any(c in self.matcher for c in "*?[]"):
            if fnmatch.fnmatch(value, self.matcher):
                return True
        # Try regex (only if it looks like a regex pattern)
        if any(c in self.matcher for c in "^$+.(){}|\\"):
            try:
                if re.fullmatch(self.matcher, value):
                    return True
            except re.error:
                pass
        return False


@dataclass
class HookSpecificOutput:
    """Claude SDK hookSpecificOutput fields for middleware control."""

    # Permission control
    permission_decision: Optional[str] = None  # "allow" | "deny" | "ask"
    permission_decision_reason: Optional[str] = None

    # Input modification (middleware pattern)
    updated_input: Optional[Dict[str, Any]] = None

    # Output suppression
    suppress_output: bool = False

    # System message injection
    system_message: Optional[str] = None


@dataclass
class HookResult:
    """Result of hook execution."""

    decision: HookDecision = HookDecision.APPROVE
    modified_input: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    error: Optional[str] = None
    output: Optional[str] = None
    duration_ms: float = 0.0

    # Claude SDK specific output
    hook_specific_output: Optional[HookSpecificOutput] = None

    @property
    def should_proceed(self) -> bool:
        """Whether the operation should proceed."""
        return self.decision in (HookDecision.APPROVE, HookDecision.MODIFY, HookDecision.SKIP)

    @property
    def permission_decision(self) -> Optional[str]:
        """Get permission decision from hook specific output."""
        if self.hook_specific_output:
            return self.hook_specific_output.permission_decision
        return None


@dataclass
class HookMatcher:
    """Claude SDK HookMatcher for pattern-based hook registration."""

    matcher: str  # Regex pattern to match tool names
    hooks: List["HookCallback"] = field(default_factory=list)
    timeout: int = 60  # Timeout in seconds

    def matches(self, value: str) -> bool:
        """Check if matcher pattern matches the value."""
        # Wildcard match
        if self.matcher == "*":
            return True
        # Exact match first
        if value == self.matcher:
            return True
        # Check if matcher has glob characters
        if any(c in self.matcher for c in "*?[]"):
            if fnmatch.fnmatch(value, self.matcher):
                return True
        # Try regex
        try:
            if re.fullmatch(self.matcher, value):
                return True
        except re.error:
            pass
        return False


# Type alias for hook callback functions
HookCallback = Callable[[Any], Union["HookResult", Any]]


@dataclass
class HookContext:
    """Context passed to hooks."""

    event: HookEvent
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Any] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_env_vars(self) -> Dict[str, str]:
        """Convert context to environment variables for shell hooks."""
        env = {
            "HOOK_EVENT": self.event.value,
            "HOOK_TOOL_NAME": self.tool_name or "",
            "HOOK_AGENT_ID": self.agent_id or "",
            "HOOK_SESSION_ID": self.session_id or "",
        }
        if self.tool_input:
            import json
            env["HOOK_TOOL_INPUT"] = json.dumps(self.tool_input)
        if self.tool_output:
            import json
            env["HOOK_TOOL_OUTPUT"] = json.dumps(self.tool_output) if isinstance(self.tool_output, dict) else str(self.tool_output)
        return env


class HookRegistry:
    """Registry for managing and executing hooks.

    Example:
        registry = HookRegistry()

        # Register a command hook
        registry.register(
            HookEvent.PRE_TOOL_USE,
            HookConfig(matcher="Bash", type=HookType.COMMAND, command="./validate.sh")
        )

        # Register a function hook
        async def my_hook(context: HookContext) -> HookResult:
            if "dangerous" in str(context.tool_input):
                return HookResult(decision=HookDecision.DENY, reason="Dangerous input")
            return HookResult(decision=HookDecision.APPROVE)

        registry.register_function(HookEvent.PRE_TOOL_USE, "Write", my_hook)

        # Execute hooks
        result = await registry.execute(HookEvent.PRE_TOOL_USE, context)
    """

    def __init__(self):
        self._hooks: Dict[HookEvent, List[HookConfig]] = {event: [] for event in HookEvent}
        self._executed_once: set = set()
        self._functions: Dict[str, Callable] = {}

    def register(self, event: HookEvent, config: HookConfig) -> None:
        """Register a hook configuration.

        Args:
            event: The event type to hook.
            config: Hook configuration.
        """
        self._hooks[event].append(config)
        # Sort by priority (higher first)
        self._hooks[event].sort(key=lambda h: -h.priority)
        logger.debug(f"Registered hook for {event.value}: {config.matcher}")

    def register_function(
        self,
        event: HookEvent,
        matcher: str,
        func: Callable[[HookContext], Union[HookResult, asyncio.coroutine]],
        priority: int = 0,
        once: bool = False,
    ) -> None:
        """Register a Python function as a hook.

        Args:
            event: The event type to hook.
            matcher: Pattern to match.
            func: Async function that takes HookContext and returns HookResult.
            priority: Hook priority (higher runs first).
            once: Only run once per session.
        """
        func_id = f"{event.value}:{matcher}:{id(func)}"
        self._functions[func_id] = func
        config = HookConfig(
            matcher=matcher,
            type=HookType.FUNCTION,
            function=func,
            priority=priority,
            once=once,
        )
        self.register(event, config)

    def register_from_yaml(self, hooks_config: Dict[str, Any]) -> None:
        """Register hooks from YAML configuration.

        Args:
            hooks_config: Dictionary matching Claude Code hooks.json format.

        Example:
            {
                "PreToolUse": [
                    {"matcher": "Bash", "hooks": [{"type": "command", "command": "./validate.sh"}]}
                ]
            }
        """
        for event_name, matchers in hooks_config.items():
            try:
                event = HookEvent(event_name)
            except ValueError:
                logger.warning(f"Unknown hook event: {event_name}")
                continue

            for matcher_config in matchers:
                matcher = matcher_config.get("matcher", "*")
                for hook_def in matcher_config.get("hooks", []):
                    hook_type_str = hook_def.get("type", "command")
                    try:
                        hook_type = HookType(hook_type_str)
                    except ValueError:
                        logger.warning(f"Unknown hook type: {hook_type_str}")
                        continue

                    config = HookConfig(
                        matcher=matcher,
                        type=hook_type,
                        command=hook_def.get("command"),
                        mcp_tool=hook_def.get("mcp_tool"),
                        prompt=hook_def.get("prompt"),
                        agent_name=hook_def.get("agent"),
                        once=hook_def.get("once", False),
                        timeout_ms=hook_def.get("timeout_ms", 30000),
                        enabled=hook_def.get("enabled", True),
                        priority=hook_def.get("priority", 0),
                    )
                    self.register(event, config)

    async def execute(self, event: HookEvent, context: HookContext) -> HookResult:
        """Execute all matching hooks for an event.

        Args:
            event: The event type.
            context: The hook context.

        Returns:
            Combined hook result.
        """
        matching_hooks = self._get_matching_hooks(event, context)

        if not matching_hooks:
            return HookResult(decision=HookDecision.APPROVE)

        last_result: Optional[HookResult] = None
        total_duration = 0.0

        for hook in matching_hooks:
            # Skip if already executed and once=True
            hook_id = f"{event.value}:{hook.matcher}"
            if hook.once and hook_id in self._executed_once:
                continue

            try:
                result = await self._execute_hook(hook, context)
                total_duration += result.duration_ms

                if hook.once:
                    self._executed_once.add(hook_id)

                # Stop on deny or ask
                if result.decision in (HookDecision.DENY, HookDecision.ASK):
                    return result

                # Track last result (for MODIFY)
                last_result = result

                # Update context with modifications
                if result.decision == HookDecision.MODIFY and result.modified_input:
                    context.tool_input = result.modified_input

            except asyncio.TimeoutError:
                logger.warning(f"Hook timeout: {hook.matcher}")
                return HookResult(
                    decision=HookDecision.DENY,
                    error=f"Hook timeout after {hook.timeout_ms}ms",
                    duration_ms=total_duration,
                )
            except Exception as e:
                logger.error(f"Hook error: {e}")
                return HookResult(decision=HookDecision.DENY, error=str(e), duration_ms=total_duration)

        # Return last result if it was MODIFY, otherwise APPROVE with accumulated duration
        if last_result and last_result.decision == HookDecision.MODIFY:
            last_result.duration_ms = total_duration
            return last_result

        return HookResult(decision=HookDecision.APPROVE, duration_ms=total_duration, output=last_result.output if last_result else None)

    def _get_matching_hooks(self, event: HookEvent, context: HookContext) -> List[HookConfig]:
        """Get hooks matching the event and context."""
        hooks = self._hooks.get(event, [])
        matching = []

        for hook in hooks:
            if not hook.enabled:
                continue

            # Match against tool name or event value
            match_value = context.tool_name or event.value
            if hook.matches(match_value):
                matching.append(hook)

        return matching

    async def _execute_hook(self, hook: HookConfig, context: HookContext) -> HookResult:
        """Execute a single hook."""
        import time
        start = time.time()

        try:
            if hook.type == HookType.COMMAND:
                result = await self._execute_command_hook(hook, context)
            elif hook.type == HookType.FUNCTION:
                result = await self._execute_function_hook(hook, context)
            elif hook.type == HookType.MCP:
                result = await self._execute_mcp_hook(hook, context)
            elif hook.type == HookType.AGENT:
                result = await self._execute_agent_hook(hook, context)
            else:
                result = HookResult(decision=HookDecision.APPROVE)

            result.duration_ms = (time.time() - start) * 1000
            return result

        except Exception as e:
            return HookResult(
                decision=HookDecision.DENY,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    async def _execute_command_hook(self, hook: HookConfig, context: HookContext) -> HookResult:
        """Execute a shell command hook."""
        if not hook.command:
            return HookResult(decision=HookDecision.APPROVE)

        env = {**context.to_env_vars()}
        timeout = hook.timeout_ms / 1000

        try:
            proc = await asyncio.create_subprocess_shell(
                hook.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            output = stdout.decode() if stdout else ""
            error = stderr.decode() if stderr else ""

            if proc.returncode == 0:
                return HookResult(decision=HookDecision.APPROVE, output=output)
            elif proc.returncode == 2:  # Special code for "ask"
                return HookResult(decision=HookDecision.ASK, output=output, reason=error)
            else:
                return HookResult(decision=HookDecision.DENY, output=output, reason=error)

        except asyncio.TimeoutError:
            return HookResult(
                decision=HookDecision.DENY,
                error=f"Command timeout after {hook.timeout_ms}ms",
            )
        except Exception as e:
            return HookResult(decision=HookDecision.DENY, error=str(e))

    async def _execute_function_hook(self, hook: HookConfig, context: HookContext) -> HookResult:
        """Execute a Python function hook."""
        if not hook.function:
            return HookResult(decision=HookDecision.APPROVE)

        result = hook.function(context)
        if asyncio.iscoroutine(result):
            result = await result

        if isinstance(result, HookResult):
            return result
        elif isinstance(result, bool):
            return HookResult(decision=HookDecision.APPROVE if result else HookDecision.DENY)
        else:
            return HookResult(decision=HookDecision.APPROVE)

    async def _execute_mcp_hook(self, hook: HookConfig, context: HookContext) -> HookResult:
        """Execute an MCP tool hook."""
        # Placeholder - would integrate with MCPClient
        logger.info(f"MCP hook: {hook.mcp_tool}")
        return HookResult(decision=HookDecision.APPROVE)

    async def _execute_agent_hook(self, hook: HookConfig, context: HookContext) -> HookResult:
        """Execute an agent hook."""
        # Placeholder - would invoke another agent
        logger.info(f"Agent hook: {hook.agent_name}")
        return HookResult(decision=HookDecision.APPROVE)

    def clear(self, event: Optional[HookEvent] = None) -> None:
        """Clear registered hooks.

        Args:
            event: If specified, only clear hooks for this event.
        """
        if event:
            self._hooks[event] = []
        else:
            self._hooks = {e: [] for e in HookEvent}
            self._executed_once.clear()

    def get_hooks(self, event: Optional[HookEvent] = None) -> Dict[HookEvent, List[HookConfig]]:
        """Get registered hooks.

        Args:
            event: If specified, only return hooks for this event.
        """
        if event:
            return {event: self._hooks.get(event, [])}
        return self._hooks.copy()


# Convenience exports
__all__ = [
    "HookEvent",
    "HookType",
    "HookDecision",
    "HookConfig",
    "HookResult",
    "HookContext",
    "HookRegistry",
    # Claude SDK additions
    "HookSpecificOutput",
    "HookMatcher",
    "HookCallback",
]
