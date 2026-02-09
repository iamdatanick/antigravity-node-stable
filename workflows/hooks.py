"""
Hooks module for the Agentic CDP Agent Runner.
Implements PRE_TOOL and POST_TOOL hooks with async execution, priority ordering,
glob pattern matching, and built-in security/audit hooks.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import re
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger("agent-runner.hooks")


# =============================================================================
# ENUMS AND TYPES
# =============================================================================


class HookType(Enum):
    """Hook event types for tool execution lifecycle."""

    PRE_TOOL = "pre_tool"  # Before tool execution
    POST_TOOL = "post_tool"  # After tool execution


class HookDecision(Enum):
    """Decisions that hooks can return to control tool execution flow."""

    APPROVE = "approve"  # Allow execution to proceed
    DENY = "deny"  # Block execution
    MODIFY = "modify"  # Modify the input/output and proceed
    SKIP = "skip"  # Skip this hook, continue to next


# Type alias for async hook functions
HookFunction = Callable[["HookContext"], Awaitable["HookResult"]]


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class HookContext:
    """Context passed to hook functions during execution."""

    hook_type: HookType
    tool_name: str
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_output: Any = None
    agent_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary for logging/serialization."""
        return {
            "hook_type": self.hook_type.value,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_output": str(self.tool_output)[:500] if self.tool_output else None,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class HookResult:
    """Result returned by hook functions."""

    decision: HookDecision
    modified_input: dict[str, Any] | None = None
    modified_output: Any = None
    reason: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def approve(cls, reason: str = None, metadata: dict = None) -> HookResult:
        """Factory for approval result."""
        return cls(decision=HookDecision.APPROVE, reason=reason, metadata=metadata or {})

    @classmethod
    def deny(cls, reason: str, error: str = None, metadata: dict = None) -> HookResult:
        """Factory for denial result."""
        return cls(decision=HookDecision.DENY, reason=reason, error=error, metadata=metadata or {})

    @classmethod
    def modify(cls, modified_input: dict = None, modified_output: Any = None, reason: str = None) -> HookResult:
        """Factory for modification result."""
        return cls(
            decision=HookDecision.MODIFY, modified_input=modified_input, modified_output=modified_output, reason=reason
        )


@dataclass
class RegisteredHook:
    """Internal representation of a registered hook."""

    name: str
    hook_type: HookType
    func: HookFunction
    pattern: str = "*"  # Glob pattern for tool name matching
    priority: int = 100  # Lower = higher priority (runs first)
    enabled: bool = True
    once: bool = False  # Fire only once per session
    timeout_ms: int = 30000  # Timeout for hook execution

    def matches(self, tool_name: str) -> bool:
        """Check if this hook matches the given tool name using glob patterns."""
        return fnmatch.fnmatch(tool_name.lower(), self.pattern.lower())


# =============================================================================
# HOOK REGISTRY
# =============================================================================


class HookRegistry:
    """
    Central registry for managing and executing hooks.

    Features:
    - Register hooks with glob pattern matching for tool names
    - Priority-based execution ordering
    - Async hook execution with timeouts
    - Support for one-time hooks (fire once per session)
    - Built-in hooks for audit logging, rate limiting, and security scanning
    """

    def __init__(self):
        self._hooks: dict[HookType, list[RegisteredHook]] = {hook_type: [] for hook_type in HookType}
        self._fired_once: set[str] = set()
        self._execution_stats: dict[str, dict] = defaultdict(
            lambda: {"call_count": 0, "total_time_ms": 0, "errors": 0, "denials": 0}
        )
        self._rate_limit_state: dict[str, list[float]] = defaultdict(list)

    def register(
        self,
        name: str,
        hook_type: HookType,
        func: HookFunction,
        pattern: str = "*",
        priority: int = 100,
        enabled: bool = True,
        once: bool = False,
        timeout_ms: int = 30000,
    ) -> None:
        """
        Register a hook function.

        Args:
            name: Unique identifier for the hook
            hook_type: PRE_TOOL or POST_TOOL
            func: Async function that takes HookContext and returns HookResult
            pattern: Glob pattern to match tool names (e.g., "search_*", "*_query")
            priority: Execution priority (lower runs first, default 100)
            enabled: Whether hook is active
            once: If True, hook fires only once per session
            timeout_ms: Maximum execution time in milliseconds
        """
        # Check for duplicate names within same hook type
        existing = [h for h in self._hooks[hook_type] if h.name == name]
        if existing:
            logger.warning(f"Hook '{name}' already registered for {hook_type.value}, replacing")
            self._hooks[hook_type] = [h for h in self._hooks[hook_type] if h.name != name]

        hook = RegisteredHook(
            name=name,
            hook_type=hook_type,
            func=func,
            pattern=pattern,
            priority=priority,
            enabled=enabled,
            once=once,
            timeout_ms=timeout_ms,
        )

        self._hooks[hook_type].append(hook)
        # Sort by priority
        self._hooks[hook_type].sort(key=lambda h: h.priority)

        logger.info(f"Registered hook: {name} [{hook_type.value}] pattern='{pattern}' priority={priority}")

    def unregister(self, name: str, hook_type: HookType | None = None) -> bool:
        """
        Unregister a hook by name.

        Args:
            name: Hook name to remove
            hook_type: Optional specific hook type, or remove from all types

        Returns:
            True if hook was found and removed
        """
        removed = False
        types_to_check = [hook_type] if hook_type else list(HookType)

        for ht in types_to_check:
            original_len = len(self._hooks[ht])
            self._hooks[ht] = [h for h in self._hooks[ht] if h.name != name]
            if len(self._hooks[ht]) < original_len:
                removed = True
                logger.info(f"Unregistered hook: {name} from {ht.value}")

        return removed

    def enable(self, name: str, enabled: bool = True) -> bool:
        """Enable or disable a hook by name."""
        for hook_list in self._hooks.values():
            for hook in hook_list:
                if hook.name == name:
                    hook.enabled = enabled
                    logger.info(f"Hook '{name}' {'enabled' if enabled else 'disabled'}")
                    return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a hook by name."""
        return self.enable(name, enabled=False)

    async def execute(self, hook_type: HookType, context: HookContext) -> HookResult:
        """
        Execute all matching hooks for the given context.

        Hooks are executed in priority order. Execution stops if any hook
        returns DENY. MODIFY results update the context for subsequent hooks.

        Args:
            hook_type: PRE_TOOL or POST_TOOL
            context: The hook context with tool information

        Returns:
            Final HookResult (APPROVE if all pass, or first DENY)
        """
        hooks = self._hooks.get(hook_type, [])
        matching_hooks = [h for h in hooks if h.enabled and h.matches(context.tool_name)]

        if not matching_hooks:
            return HookResult.approve(reason="No matching hooks")

        logger.debug(f"Executing {len(matching_hooks)} {hook_type.value} hooks for tool: {context.tool_name}")

        for hook in matching_hooks:
            # Check once flag
            hook_id = f"{hook_type.value}:{hook.name}:{context.session_id}"
            if hook.once and hook_id in self._fired_once:
                logger.debug(f"Skipping one-time hook '{hook.name}' (already fired)")
                continue

            # Execute with timeout
            start_time = time.time()
            try:
                result = await asyncio.wait_for(hook.func(context), timeout=hook.timeout_ms / 1000)

                # Update stats
                elapsed_ms = (time.time() - start_time) * 1000
                self._execution_stats[hook.name]["call_count"] += 1
                self._execution_stats[hook.name]["total_time_ms"] += elapsed_ms

                # Mark as fired if once
                if hook.once:
                    self._fired_once.add(hook_id)

                # Handle result
                if result.decision == HookDecision.DENY:
                    self._execution_stats[hook.name]["denials"] += 1
                    logger.warning(f"Hook '{hook.name}' DENIED tool '{context.tool_name}': {result.reason}")
                    return result

                elif result.decision == HookDecision.MODIFY:
                    # Update context for subsequent hooks
                    if result.modified_input is not None:
                        context.tool_input = result.modified_input
                    if result.modified_output is not None:
                        context.tool_output = result.modified_output
                    logger.debug(f"Hook '{hook.name}' modified context for '{context.tool_name}'")

                elif result.decision == HookDecision.SKIP:
                    logger.debug(f"Hook '{hook.name}' skipped for '{context.tool_name}'")
                    continue

                # APPROVE continues to next hook

            except TimeoutError:
                self._execution_stats[hook.name]["errors"] += 1
                logger.error(f"Hook '{hook.name}' timed out after {hook.timeout_ms}ms")
                return HookResult.deny(
                    reason=f"Hook '{hook.name}' timed out", error=f"Timeout after {hook.timeout_ms}ms"
                )

            except Exception as e:
                self._execution_stats[hook.name]["errors"] += 1
                logger.error(f"Hook '{hook.name}' failed: {e}")
                # Continue to next hook on error (fail-open)
                continue

        return HookResult.approve(reason="All hooks passed")

    def clear(self, hook_type: HookType | None = None) -> None:
        """Clear all hooks of a specific type, or all hooks if type is None."""
        if hook_type:
            self._hooks[hook_type] = []
            logger.info(f"Cleared all {hook_type.value} hooks")
        else:
            for ht in HookType:
                self._hooks[ht] = []
            logger.info("Cleared all hooks")
        self._fired_once.clear()

    def list_hooks(self, hook_type: HookType | None = None) -> list[dict[str, Any]]:
        """List all registered hooks with their configuration."""
        result = []
        types_to_list = [hook_type] if hook_type else list(HookType)

        for ht in types_to_list:
            for hook in self._hooks[ht]:
                result.append(
                    {
                        "name": hook.name,
                        "type": ht.value,
                        "pattern": hook.pattern,
                        "priority": hook.priority,
                        "enabled": hook.enabled,
                        "once": hook.once,
                        "timeout_ms": hook.timeout_ms,
                        "stats": dict(self._execution_stats.get(hook.name, {})),
                    }
                )

        return result

    def get_stats(self) -> dict[str, dict]:
        """Get execution statistics for all hooks."""
        return dict(self._execution_stats)


# =============================================================================
# BUILT-IN HOOKS
# =============================================================================


async def audit_logger_hook(context: HookContext) -> HookResult:
    """
    Built-in hook that logs all tool executions for audit purposes.
    Logs to the application logger and can be extended to send to external systems.
    """
    log_entry = {
        "type": "tool_audit",
        "hook_type": context.hook_type.value,
        "tool_name": context.tool_name,
        "agent_id": context.agent_id,
        "session_id": context.session_id,
        "user_id": context.user_id,
        "timestamp": context.timestamp.isoformat(),
    }

    if context.hook_type == HookType.PRE_TOOL:
        log_entry["input_preview"] = str(context.tool_input)[:200]
        logger.info(f"[AUDIT] PRE_TOOL: {context.tool_name} | agent={context.agent_id} | session={context.session_id}")
    else:
        log_entry["output_preview"] = str(context.tool_output)[:200] if context.tool_output else None
        logger.info(f"[AUDIT] POST_TOOL: {context.tool_name} | agent={context.agent_id} | status=completed")

    return HookResult.approve(reason="Audit logged", metadata={"audit_entry": log_entry})


class RateLimiter:
    """
    Rate limiter hook factory.
    Creates hooks that enforce rate limits on tool executions.
    """

    def __init__(self, requests_per_minute: int = 60, burst_limit: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self._request_times: dict[str, list[float]] = defaultdict(list)

    def _get_key(self, context: HookContext) -> str:
        """Generate rate limit key from context."""
        return f"{context.session_id or 'global'}:{context.tool_name}"

    def _cleanup_old_requests(self, key: str, current_time: float) -> None:
        """Remove request timestamps older than 1 minute."""
        cutoff = current_time - 60
        self._request_times[key] = [t for t in self._request_times[key] if t > cutoff]

    async def __call__(self, context: HookContext) -> HookResult:
        """Rate limiter hook function."""
        current_time = time.time()
        key = self._get_key(context)

        # Cleanup old requests
        self._cleanup_old_requests(key, current_time)

        request_count = len(self._request_times[key])

        # Check burst limit (last second)
        recent_requests = [t for t in self._request_times[key] if t > current_time - 1]
        if len(recent_requests) >= self.burst_limit:
            return HookResult.deny(
                reason=f"Burst rate limit exceeded ({self.burst_limit} requests/second)", error="RATE_LIMIT_BURST"
            )

        # Check minute limit
        if request_count >= self.requests_per_minute:
            return HookResult.deny(
                reason=f"Rate limit exceeded ({self.requests_per_minute} requests/minute)", error="RATE_LIMIT_MINUTE"
            )

        # Record this request
        self._request_times[key].append(current_time)

        return HookResult.approve(
            reason="Rate limit check passed",
            metadata={"requests_in_window": request_count + 1, "limit": self.requests_per_minute},
        )


class SecurityScanner:
    """
    Security scanner hook factory.
    Scans tool inputs for potential security threats like injection attacks.
    """

    # Default patterns to detect common injection attempts
    DEFAULT_PATTERNS = [
        # SQL Injection
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|TRUNCATE)\b.*\b(FROM|INTO|TABLE|WHERE)\b)",
        r"(--|\#|;)\s*(SELECT|DROP|DELETE|UPDATE)",
        r"'\s*(OR|AND)\s*'?\s*[0-9a-zA-Z]*\s*=",
        # Command Injection
        r"(\||;|`|\$\(|\))\s*(cat|ls|rm|mv|cp|wget|curl|bash|sh|python|perl|ruby)",
        r"&{2}\s*(rm|cat|wget|curl)",
        # Path Traversal
        r"\.\./",
        r"\.\.\\",
        # Script Injection
        r"<script[^>]*>",
        r"javascript:",
        r"on(click|load|error|mouseover)=",
    ]

    def __init__(self, patterns: list[str] | None = None, block_on_detection: bool = True, scan_output: bool = False):
        self.patterns = patterns or self.DEFAULT_PATTERNS
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.patterns]
        self.block_on_detection = block_on_detection
        self.scan_output = scan_output

    def _scan_value(self, value: Any) -> list[str]:
        """Scan a value for security threats."""
        threats = []
        str_value = str(value)

        for pattern in self.compiled_patterns:
            if pattern.search(str_value):
                threats.append(pattern.pattern)

        return threats

    def _scan_dict(self, data: dict[str, Any]) -> dict[str, list[str]]:
        """Recursively scan a dictionary for threats."""
        results = {}

        for key, value in data.items():
            if isinstance(value, dict):
                nested = self._scan_dict(value)
                if nested:
                    results[key] = nested
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    threats = self._scan_value(item)
                    if threats:
                        results[f"{key}[{i}]"] = threats
            else:
                threats = self._scan_value(value)
                if threats:
                    results[key] = threats

        return results

    async def __call__(self, context: HookContext) -> HookResult:
        """Security scanner hook function."""
        threats_found = {}

        # Scan input
        if context.tool_input:
            input_threats = self._scan_dict(context.tool_input)
            if input_threats:
                threats_found["input"] = input_threats

        # Scan output (if enabled and available)
        if self.scan_output and context.tool_output:
            if isinstance(context.tool_output, dict):
                output_threats = self._scan_dict(context.tool_output)
            else:
                output_threats = self._scan_value(context.tool_output)
            if output_threats:
                threats_found["output"] = output_threats

        if threats_found:
            logger.warning(f"[SECURITY] Threats detected in tool '{context.tool_name}': {threats_found}")

            if self.block_on_detection:
                return HookResult.deny(
                    reason="Security threat detected in tool input",
                    error="SECURITY_THREAT_DETECTED",
                    metadata={"threats": threats_found},
                )
            else:
                # Log but allow
                return HookResult.approve(
                    reason="Security threats detected but not blocked",
                    metadata={"threats": threats_found, "action": "logged_only"},
                )

        return HookResult.approve(reason="Security scan passed", metadata={"scanned": True})


# =============================================================================
# DECORATORS FOR HOOK REGISTRATION
# =============================================================================


def pre_tool(pattern: str = "*", priority: int = 100, once: bool = False, timeout_ms: int = 30000):
    """
    Decorator to mark a function as a PRE_TOOL hook.

    Usage:
        @pre_tool(pattern="search_*", priority=50)
        async def my_pre_hook(context: HookContext) -> HookResult:
            return HookResult.approve()
    """

    def decorator(func: HookFunction):
        func._hook_metadata = {
            "hook_type": HookType.PRE_TOOL,
            "pattern": pattern,
            "priority": priority,
            "once": once,
            "timeout_ms": timeout_ms,
        }
        return func

    return decorator


def post_tool(pattern: str = "*", priority: int = 100, once: bool = False, timeout_ms: int = 30000):
    """
    Decorator to mark a function as a POST_TOOL hook.

    Usage:
        @post_tool(pattern="*_query", priority=150)
        async def my_post_hook(context: HookContext) -> HookResult:
            return HookResult.approve()
    """

    def decorator(func: HookFunction):
        func._hook_metadata = {
            "hook_type": HookType.POST_TOOL,
            "pattern": pattern,
            "priority": priority,
            "once": once,
            "timeout_ms": timeout_ms,
        }
        return func

    return decorator


def register_decorated_hooks(registry: HookRegistry, *funcs: HookFunction) -> None:
    """
    Register functions decorated with @pre_tool or @post_tool to a registry.

    Usage:
        register_decorated_hooks(registry, my_hook1, my_hook2)
    """
    for func in funcs:
        metadata = getattr(func, "_hook_metadata", None)
        if metadata:
            registry.register(
                name=func.__name__,
                hook_type=metadata["hook_type"],
                func=func,
                pattern=metadata["pattern"],
                priority=metadata["priority"],
                once=metadata["once"],
                timeout_ms=metadata["timeout_ms"],
            )
        else:
            logger.warning(f"Function '{func.__name__}' has no hook metadata, skipping")


# =============================================================================
# FACTORY FUNCTIONS FOR BUILT-IN HOOKS
# =============================================================================


def create_default_registry(
    enable_audit: bool = True,
    enable_rate_limiter: bool = True,
    enable_security_scanner: bool = True,
    rate_limit_rpm: int = 60,
    rate_limit_burst: int = 10,
) -> HookRegistry:
    """
    Create a HookRegistry pre-configured with built-in hooks.

    Args:
        enable_audit: Enable audit logging hook
        enable_rate_limiter: Enable rate limiting hook
        enable_security_scanner: Enable security scanning hook
        rate_limit_rpm: Requests per minute for rate limiter
        rate_limit_burst: Burst limit for rate limiter

    Returns:
        Configured HookRegistry instance
    """
    registry = HookRegistry()

    if enable_audit:
        # Audit logger - runs on all tools with high priority
        registry.register(
            name="audit_logger",
            hook_type=HookType.PRE_TOOL,
            func=audit_logger_hook,
            pattern="*",
            priority=10,  # Run early
        )
        registry.register(
            name="audit_logger_post", hook_type=HookType.POST_TOOL, func=audit_logger_hook, pattern="*", priority=10
        )

    if enable_rate_limiter:
        # Rate limiter - runs before other checks
        rate_limiter = RateLimiter(requests_per_minute=rate_limit_rpm, burst_limit=rate_limit_burst)
        registry.register(name="rate_limiter", hook_type=HookType.PRE_TOOL, func=rate_limiter, pattern="*", priority=20)

    if enable_security_scanner:
        # Security scanner - runs after rate limit but before execution
        scanner = SecurityScanner(block_on_detection=True)
        registry.register(name="security_scanner", hook_type=HookType.PRE_TOOL, func=scanner, pattern="*", priority=30)

    logger.info(f"Created default hook registry with {len(registry.list_hooks())} hooks")
    return registry


# =============================================================================
# AGENTIC-WORKFLOWS BRIDGE
# =============================================================================
# Integrates the shared agentic-workflows security layer (injection defense,
# scope validation, rate limiting, kill switch) into this hook registry.


def create_secured_registry(
    enable_audit: bool = True,
    enable_rate_limiter: bool = True,
    enable_security_scanner: bool = True,
    enable_injection_defense: bool = True,
    rate_limit_rpm: int = 60,
    security_scope: str = "elevated",
) -> HookRegistry:
    """
    Create a HookRegistry with both built-in hooks AND agentic-workflows
    security layer (injection defense, scope validation, kill switch).

    This is the recommended way to create a registry for production use.
    Falls back gracefully if agentic-workflows is not installed.
    """
    # Start with built-in hooks
    registry = create_default_registry(
        enable_audit=enable_audit,
        enable_rate_limiter=enable_rate_limiter,
        enable_security_scanner=enable_security_scanner,
        rate_limit_rpm=rate_limit_rpm,
    )

    # Layer in agentic-workflows security (optional dependency)
    try:
        from agentic_workflows.bridge import PlatformHookConfig, create_platform_hooks

        aw_config = PlatformHookConfig(
            enable_audit=False,  # Already have built-in audit
            enable_rate_limiter=False,  # Already have built-in rate limiter
            enable_injection_defense=enable_injection_defense,
            security_scope=security_scope,
        )
        aw_registry = create_platform_hooks(aw_config)
        registry._aw_registry = aw_registry  # Store for runtime access
        logger.info("agentic-workflows bridge loaded — injection defense + scope validation active")
    except ImportError:
        logger.info("agentic-workflows not installed — using built-in hooks only")

    return registry


# =============================================================================
# DEFAULT INSTANCE
# =============================================================================

# Default registry instance for convenience
default_registry = HookRegistry()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "HookType",
    "HookDecision",
    # Data classes
    "HookContext",
    "HookResult",
    "RegisteredHook",
    # Registry
    "HookRegistry",
    # Built-in hooks
    "audit_logger_hook",
    "RateLimiter",
    "SecurityScanner",
    # Decorators
    "pre_tool",
    "post_tool",
    "register_decorated_hooks",
    # Factory functions
    "create_default_registry",
    "create_secured_registry",
    # Default instance
    "default_registry",
]
