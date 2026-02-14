"""Scope validation for tool and resource access control."""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class Scope(Enum):
    """Security scope levels (AWS-style 4-tier)."""

    MINIMAL = "minimal"  # Read-only, no execution
    STANDARD = "standard"  # Read + limited write
    ELEVATED = "elevated"  # Full local access
    ADMIN = "admin"  # Full access including network


@dataclass
class ToolPermission:
    """Tool permission configuration."""

    name: str
    allowed_scopes: list[Scope]
    requires_approval: bool = False
    rate_limit: int | None = None  # Max calls per minute
    description: str = ""


class ScopeValidator:
    """Validates tool calls against security scope."""

    # Default tool permissions
    DEFAULT_PERMISSIONS: dict[str, ToolPermission] = {
        # Read-only tools
        "Read": ToolPermission(
            name="Read",
            allowed_scopes=[Scope.MINIMAL, Scope.STANDARD, Scope.ELEVATED, Scope.ADMIN],
            description="Read file contents",
        ),
        "Glob": ToolPermission(
            name="Glob",
            allowed_scopes=[Scope.MINIMAL, Scope.STANDARD, Scope.ELEVATED, Scope.ADMIN],
            description="Find files by pattern",
        ),
        "Grep": ToolPermission(
            name="Grep",
            allowed_scopes=[Scope.MINIMAL, Scope.STANDARD, Scope.ELEVATED, Scope.ADMIN],
            description="Search file contents",
        ),
        # Write tools
        "Write": ToolPermission(
            name="Write",
            allowed_scopes=[Scope.STANDARD, Scope.ELEVATED, Scope.ADMIN],
            description="Write file contents",
        ),
        "Edit": ToolPermission(
            name="Edit",
            allowed_scopes=[Scope.STANDARD, Scope.ELEVATED, Scope.ADMIN],
            description="Edit file contents",
        ),
        # Execution tools
        "Bash": ToolPermission(
            name="Bash",
            allowed_scopes=[Scope.ELEVATED, Scope.ADMIN],
            requires_approval=True,
            description="Execute shell commands",
        ),
        "Task": ToolPermission(
            name="Task",
            allowed_scopes=[Scope.STANDARD, Scope.ELEVATED, Scope.ADMIN],
            description="Launch sub-agents",
        ),
        # Network tools
        "WebFetch": ToolPermission(
            name="WebFetch",
            allowed_scopes=[Scope.ADMIN],
            requires_approval=True,
            rate_limit=10,
            description="Fetch web content",
        ),
        "WebSearch": ToolPermission(
            name="WebSearch",
            allowed_scopes=[Scope.ADMIN],
            requires_approval=True,
            rate_limit=5,
            description="Search the web",
        ),
    }

    # File access patterns by scope
    FILE_PATTERNS: dict[Scope, list[str]] = {
        Scope.MINIMAL: [
            "*.md",
            "*.txt",
            "*.json",
            "*.yaml",
            "*.yml",
            "*.py",
            "*.js",
            "*.ts",
            "*.go",
            "*.rs",
            "*.java",
            "*.html",
            "*.css",
            "*.xml",
        ],
        Scope.STANDARD: [
            "*",  # All files except blocked
        ],
        Scope.ELEVATED: [
            "*",
        ],
        Scope.ADMIN: [
            "*",
        ],
    }

    # Blocked patterns (never allowed regardless of scope)
    BLOCKED_PATTERNS: list[str] = [
        "*.pem",
        "*.key",
        "*.p12",
        "*.pfx",  # Private keys
        "**/id_rsa*",
        "**/id_ed25519*",  # SSH keys
        "**/.env*",
        "**/secrets*",  # Environment/secrets
        "**/credentials*",
        "**/password*",  # Credentials
        "**/.aws/*",
        "**/.ssh/*",  # Cloud/SSH configs
    ]

    # Dangerous bash commands
    DANGEROUS_COMMANDS: list[str] = [
        "rm -rf",
        "mkfs",
        "dd if=",
        ":(){",
        "chmod -R 777",
        "curl.*|.*sh",
        "wget.*|.*sh",  # Pipe to shell
        "nc -l",
        "netcat",  # Network listeners
        "> /dev/sd",  # Disk overwrites
    ]

    def __init__(
        self,
        permissions: dict[str, ToolPermission] | None = None,
        approval_callback: Callable[[str, str], bool] | None = None,
    ):
        """Initialize validator.

        Args:
            permissions: Custom tool permissions (merged with defaults).
            approval_callback: Called when tool requires approval.
                               Receives (tool_name, context) and returns bool.
        """
        self.permissions = {**self.DEFAULT_PERMISSIONS}
        if permissions:
            self.permissions.update(permissions)
        self.approval_callback = approval_callback
        self._call_counts: dict[str, list[float]] = {}

    def validate_tool_call(
        self,
        tool_name: str,
        scope: Scope,
        context: str = "",
    ) -> tuple[bool, str]:
        """Validate a tool call against scope.

        Args:
            tool_name: Name of the tool.
            scope: Current security scope.
            context: Additional context for approval.

        Returns:
            Tuple of (allowed, reason).
        """
        # Check if tool exists
        permission = self.permissions.get(tool_name)
        if not permission:
            return False, f"Unknown tool: {tool_name}"

        # Check scope
        if scope not in permission.allowed_scopes:
            allowed_str = ", ".join(s.value for s in permission.allowed_scopes)
            return False, (
                f"Tool '{tool_name}' not allowed in {scope.value} scope. Requires: {allowed_str}"
            )

        # Check rate limit
        if permission.rate_limit:
            import time

            now = time.time()
            calls = self._call_counts.get(tool_name, [])
            # Remove calls older than 1 minute
            calls = [t for t in calls if now - t < 60]
            if len(calls) >= permission.rate_limit:
                return False, (
                    f"Rate limit exceeded for '{tool_name}': {permission.rate_limit}/minute"
                )
            calls.append(now)
            self._call_counts[tool_name] = calls

        # Check approval requirement
        if permission.requires_approval:
            if self.approval_callback:
                if not self.approval_callback(tool_name, context):
                    return False, f"Approval denied for '{tool_name}'"
            else:
                return False, f"Tool '{tool_name}' requires approval but no callback set"

        return True, "Allowed"

    def validate_file_access(
        self,
        file_path: str,
        scope: Scope,
        write: bool = False,
    ) -> tuple[bool, str]:
        """Validate file access against scope.

        Args:
            file_path: Path to the file.
            scope: Current security scope.
            write: Whether this is a write operation.

        Returns:
            Tuple of (allowed, reason).
        """
        # Normalize path
        normalized = file_path.replace("\\", "/").lower()

        # Check blocked patterns first
        for pattern in self.BLOCKED_PATTERNS:
            if fnmatch.fnmatch(normalized, pattern.lower()):
                return False, f"Access to '{file_path}' is blocked (matches: {pattern})"

        # Check write permissions
        if write and scope == Scope.MINIMAL:
            return False, "Write operations not allowed in minimal scope"

        # Check file type patterns for minimal scope
        if scope == Scope.MINIMAL:
            allowed_patterns = self.FILE_PATTERNS[Scope.MINIMAL]
            if not any(fnmatch.fnmatch(normalized, p.lower()) for p in allowed_patterns):
                return False, (f"File type not allowed in minimal scope: {file_path}")

        return True, "Allowed"

    def validate_bash_command(
        self,
        command: str,
        scope: Scope,
    ) -> tuple[bool, str]:
        """Validate bash command against scope.

        Args:
            command: Command to validate.
            scope: Current security scope.

        Returns:
            Tuple of (allowed, reason).
        """
        # Check basic scope
        if scope not in [Scope.ELEVATED, Scope.ADMIN]:
            return False, "Bash commands require elevated or admin scope"

        # Check dangerous patterns
        command_lower = command.lower()
        for pattern in self.DANGEROUS_COMMANDS:
            if re.search(pattern, command_lower):
                if scope != Scope.ADMIN:
                    return False, (
                        f"Dangerous command pattern detected: {pattern}. Requires admin scope."
                    )

        # Check network commands in elevated scope
        network_patterns = ["curl", "wget", "ssh", "scp", "nc", "netcat"]
        if scope == Scope.ELEVATED:
            for pattern in network_patterns:
                if pattern in command_lower:
                    return False, (f"Network command '{pattern}' requires admin scope")

        return True, "Allowed"

    def validate_network_access(
        self,
        url: str,
        scope: Scope,
    ) -> tuple[bool, str]:
        """Validate network access against scope.

        Args:
            url: URL to access.
            scope: Current security scope.

        Returns:
            Tuple of (allowed, reason).
        """
        if scope != Scope.ADMIN:
            return False, "Network access requires admin scope"

        # Block internal/localhost
        internal_patterns = [
            r"localhost",
            r"127\.0\.0\.",
            r"0\.0\.0\.0",
            r"192\.168\.",
            r"10\.",
            r"172\.(1[6-9]|2[0-9]|3[01])\.",
            r"::1",
            r"fe80:",
        ]

        url_lower = url.lower()
        for pattern in internal_patterns:
            if re.search(pattern, url_lower):
                return False, f"Access to internal addresses is blocked: {url}"

        return True, "Allowed"

    def get_allowed_tools(self, scope: Scope) -> list[str]:
        """Get list of tools allowed for a scope.

        Args:
            scope: Security scope.

        Returns:
            List of allowed tool names.
        """
        return [name for name, perm in self.permissions.items() if scope in perm.allowed_scopes]

    def explain_scope(self, scope: Scope) -> str:
        """Get human-readable explanation of scope capabilities.

        Args:
            scope: Security scope to explain.

        Returns:
            Explanation string.
        """
        explanations = {
            Scope.MINIMAL: (
                "MINIMAL scope: Read-only access to code and documentation files. "
                "No execution or network access."
            ),
            Scope.STANDARD: (
                "STANDARD scope: Read and write access to project files. "
                "Can spawn sub-agents. No direct command execution or network."
            ),
            Scope.ELEVATED: (
                "ELEVATED scope: Full local access including command execution. "
                "No network access. Dangerous commands require approval."
            ),
            Scope.ADMIN: (
                "ADMIN scope: Full access including network operations. "
                "All dangerous operations require explicit approval."
            ),
        }

        tools = self.get_allowed_tools(scope)
        return f"{explanations[scope]}\nAllowed tools: {', '.join(tools)}"
