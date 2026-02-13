"""
Bridge module for integrating agentic-workflows with platform services.

Adapts the agentic-workflows HookRegistry, security, and orchestration
into the HookType.PRE_TOOL/POST_TOOL pattern used by Antigravity-Node,
centillion-ai-platform, and bootstrap-loop-cdp.

Usage:
    from agentic_workflows.bridge import (
        create_platform_hooks,
        create_orchestrator_config,
    )

    # Wire hooks into your agent runner
    registry = create_platform_hooks()

    # Get unified orchestrator config
    orch = create_orchestrator_config()
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentic_workflows.hooks import (
    HookRegistry,
    HookEvent,
    HookDecision,
    HookContext as AWHookContext,
    HookResult as AWHookResult,
    HookConfig,
    HookType as AWHookType,
)
from agentic_workflows.security import (
    PromptInjectionDefense,
    ScopeValidator,
    RateLimiter as AWRateLimiter,
    KillSwitch,
    ThreatLevel,
    Scope,
)
from agentic_workflows.security.rate_limiter import RateLimitConfig

logger = logging.getLogger("agentic_workflows.bridge")


# =============================================================================
# PLATFORM HOOK ADAPTER
# =============================================================================

@dataclass
class PlatformHookConfig:
    """Configuration for platform-level hooks."""
    enable_audit: bool = True
    enable_rate_limiter: bool = True
    enable_security_scanner: bool = True
    enable_injection_defense: bool = True
    rate_limit_rpm: int = 60
    rate_limit_burst: int = 10
    security_scope: str = "standard"  # minimal, standard, elevated, admin
    kill_switch_enabled: bool = True


def create_platform_hooks(config: Optional[PlatformHookConfig] = None) -> HookRegistry:
    """
    Create a HookRegistry pre-configured with production hooks that bridge
    the agentic-workflows security layer into the platform hook pattern.

    This replaces the per-repo HookRegistry implementations with a single
    source from the agentic-workflows package.
    """
    if config is None:
        config = PlatformHookConfig()

    registry = HookRegistry()
    injection_defense = PromptInjectionDefense()
    scope_validator = ScopeValidator()
    kill_switch = KillSwitch()

    # Audit logger (priority 0 = highest, runs first)
    if config.enable_audit:
        async def audit_hook(ctx: AWHookContext) -> AWHookResult:
            logger.info(
                f"[AUDIT] {ctx.event.value}: tool={ctx.tool_name} "
                f"agent={ctx.agent_id} session={ctx.session_id}"
            )
            return AWHookResult(decision=HookDecision.APPROVE)

        registry.register_function(HookEvent.PRE_TOOL_USE, "*", audit_hook, priority=100)
        registry.register_function(HookEvent.POST_TOOL_USE, "*", audit_hook, priority=100)

    # Kill switch (priority 90)
    if config.kill_switch_enabled:
        async def kill_switch_hook(ctx: AWHookContext) -> AWHookResult:
            if kill_switch.is_active:
                return AWHookResult(
                    decision=HookDecision.DENY,
                    reason="Kill switch activated — all operations halted",
                )
            return AWHookResult(decision=HookDecision.APPROVE)

        registry.register_function(HookEvent.PRE_TOOL_USE, "*", kill_switch_hook, priority=90)

    # Injection defense (priority 80)
    if config.enable_injection_defense:
        async def injection_hook(ctx: AWHookContext) -> AWHookResult:
            if ctx.tool_input:
                text = str(ctx.tool_input)
                result = injection_defense.scan(text)
                if result.threat_level.value >= ThreatLevel.HIGH.value:
                    return AWHookResult(
                        decision=HookDecision.DENY,
                        reason=f"Injection threat detected: {result.threat_level.name}",
                    )
            return AWHookResult(decision=HookDecision.APPROVE)

        registry.register_function(HookEvent.PRE_TOOL_USE, "*", injection_hook, priority=80)

    # Rate limiter (priority 70)
    if config.enable_rate_limiter:
        rate_limiter = AWRateLimiter(
            config=RateLimitConfig(
                requests_per_second=config.rate_limit_rpm / 60.0,
                burst_size=config.rate_limit_burst,
            )
        )

        async def rate_limit_hook(ctx: AWHookContext) -> AWHookResult:
            key = f"{ctx.session_id or 'global'}:{ctx.tool_name or 'any'}"
            allowed = rate_limiter.try_acquire(key=key)
            if not allowed:
                return AWHookResult(
                    decision=HookDecision.DENY,
                    reason=f"Rate limit exceeded for {key}",
                )
            return AWHookResult(decision=HookDecision.APPROVE)

        registry.register_function(HookEvent.PRE_TOOL_USE, "*", rate_limit_hook, priority=70)

    # Scope validator (priority 60)
    scope_map = {
        "minimal": Scope.MINIMAL,
        "standard": Scope.STANDARD,
        "elevated": Scope.ELEVATED,
        "admin": Scope.ADMIN,
    }
    max_scope = scope_map.get(config.security_scope, Scope.STANDARD)

    async def scope_hook(ctx: AWHookContext) -> AWHookResult:
        tool = ctx.tool_name or ""
        allowed, reason = scope_validator.validate_tool_call(tool, max_scope)
        if not allowed:
            return AWHookResult(
                decision=HookDecision.DENY,
                reason=reason,
            )
        return AWHookResult(decision=HookDecision.APPROVE)

    registry.register_function(HookEvent.PRE_TOOL_USE, "*", scope_hook, priority=60)

    logger.info(
        f"Platform hooks created: audit={config.enable_audit} "
        f"rate_limit={config.enable_rate_limiter} "
        f"injection={config.enable_injection_defense} "
        f"scope={config.security_scope}"
    )
    return registry


# =============================================================================
# ORCHESTRATOR CONFIGURATION
# =============================================================================

@dataclass
class OrchestratorConfig:
    """Unified orchestrator configuration for all platform services."""

    # Service endpoints (auto-discovered from env or explicit)
    starrocks_host: str = field(default_factory=lambda: os.getenv("STARROCKS_HOST", "starrocks"))
    starrocks_port: int = field(default_factory=lambda: int(os.getenv("STARROCKS_HTTP_PORT", "8030")))
    milvus_host: str = field(default_factory=lambda: os.getenv("MILVUS_HOST", "milvus"))
    milvus_port: int = field(default_factory=lambda: int(os.getenv("MILVUS_PORT", "19530")))
    nats_url: str = field(default_factory=lambda: os.getenv("NATS_URL", "nats://nats:4222"))
    valkey_url: str = field(default_factory=lambda: os.getenv("VALKEY_URL", "redis://valkey:6379"))
    keycloak_url: str = field(default_factory=lambda: os.getenv("KEYCLOAK_URL", "http://keycloak:8080"))
    openbao_addr: str = field(default_factory=lambda: os.getenv("OPENBAO_ADDR", "http://openbao:8200"))
    marquez_url: str = field(default_factory=lambda: os.getenv("OPENLINEAGE_URL", "http://marquez:5000"))
    ovms_rest: str = field(default_factory=lambda: os.getenv("OVMS_REST", "http://ovms:8000"))
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://ollama:11434"))
    budget_proxy_url: str = field(default_factory=lambda: os.getenv("LITELLM_URL", "http://budget-proxy:4000"))
    zerotrust_url: str = field(default_factory=lambda: os.getenv("ZEROTRUST_GATE_URL", "http://zerotrust-gate:8150"))

    # MCP servers
    mcp_starrocks: str = field(default_factory=lambda: os.getenv("STARROCKS_MCP_URL", "http://starrocks-mcp:8101"))
    mcp_milvus: str = field(default_factory=lambda: os.getenv("MILVUS_MCP_URL", "http://milvus-mcp:8102"))
    mcp_camara: str = field(default_factory=lambda: os.getenv("CAMARA_MCP_URL", "http://camara-mcp:8103"))
    mcp_uid2: str = field(default_factory=lambda: os.getenv("UID2_MCP_URL", "http://uid2-mcp:8104"))

    # LLM
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"))

    # Security
    security_scope: str = "elevated"
    max_iterations: int = 50

    def get_mcp_servers(self) -> Dict[str, str]:
        """Return MCP server URLs for GooseBlockRunner."""
        return {
            "starrocks": self.mcp_starrocks,
            "milvus": self.mcp_milvus,
            "camara": self.mcp_camara,
            "uid2": self.mcp_uid2,
            "marquez": self.marquez_url,
        }

    def to_env(self) -> Dict[str, str]:
        """Export as environment variables."""
        return {
            "STARROCKS_HOST": self.starrocks_host,
            "STARROCKS_HTTP_PORT": str(self.starrocks_port),
            "MILVUS_HOST": self.milvus_host,
            "MILVUS_PORT": str(self.milvus_port),
            "NATS_URL": self.nats_url,
            "VALKEY_URL": self.valkey_url,
            "KEYCLOAK_URL": self.keycloak_url,
            "OPENBAO_ADDR": self.openbao_addr,
            "OPENLINEAGE_URL": self.marquez_url,
            "OVMS_REST": self.ovms_rest,
            "OLLAMA_URL": self.ollama_url,
            "LITELLM_URL": self.budget_proxy_url,
            "ZEROTRUST_GATE_URL": self.zerotrust_url,
            "LLM_PROVIDER": self.llm_provider,
            "LLM_MODEL": self.llm_model,
        }


def create_orchestrator_config() -> OrchestratorConfig:
    """Create OrchestratorConfig from environment variables."""
    return OrchestratorConfig()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "PlatformHookConfig",
    "create_platform_hooks",
    "OrchestratorConfig",
    "create_orchestrator_config",
]
