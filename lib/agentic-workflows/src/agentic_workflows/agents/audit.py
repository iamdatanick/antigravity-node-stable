"""
Agent Audit Module — SKILL.md v4.1 Compliance Validation
=========================================================

Provides functions to validate all agent templates against the v4.1 spec,
generate compliance reports, and identify issues.

Usage:
    from agentic_workflows.agents.audit import get_audit_summary, validate_all_agents

    summary = get_audit_summary()
    valid, issues = validate_all_agents()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic_workflows.agents.agent_loader import (
    CLAUDE_AGENTS_DIR,
    AgentDefinition,
)

# v4.1 required fields
V41_REQUIRED_FIELDS = [
    "name",
    "description",
    "tools",
    "model",
    "skills",
    "permissionMode",
    "security_scope",
    "timeout",
    "max_iterations",
    "recovery",
    "guardrails",
    "telemetry",
    "memory",
    "context_mode",
    "cost_tier",
    "constitutional",
    "async",
    "delegates_to",
]

# Valid values for each field
VALID_SECURITY_SCOPES = {"minimal", "standard", "elevated", "admin"}
VALID_COST_TIERS = {"low", "standard", "premium"}
VALID_MEMORY_MODES = {"session", "conversation", "accumulated", "persistent"}
VALID_CONTEXT_MODES = {"focused", "full", "accumulated"}
VALID_RECOVERY = {"retry", "fallback", "checkpoint", "escalate"}
VALID_PERMISSION_MODES = {"dontAsk", "acceptEdits", "plan", "default"}
VALID_MODELS = {"opus", "sonnet", "haiku"}

# Security scope constraints
SCOPE_REQUIRES_FALLBACK = {"elevated", "admin"}
SCOPE_REQUIRES_KILL_SWITCH = {"admin"}


@dataclass
class AgentAuditResult:
    """Result of auditing a single agent."""

    name: str
    category: str
    file_path: str
    compliant: bool
    missing_fields: list[str] = field(default_factory=list)
    invalid_values: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    field_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "compliant": self.compliant,
            "missing_fields": self.missing_fields,
            "invalid_values": self.invalid_values,
            "warnings": self.warnings,
            "field_count": self.field_count,
        }


@dataclass
class AuditSummary:
    """Summary of a full agent audit."""

    total_agents: int = 0
    compliant_agents: int = 0
    non_compliant_agents: int = 0
    all_valid: bool = False
    total_fields: int = 0
    missing_field_count: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    by_security_scope: dict[str, int] = field(default_factory=dict)
    by_cost_tier: dict[str, int] = field(default_factory=dict)
    by_model: dict[str, int] = field(default_factory=dict)
    by_memory: dict[str, int] = field(default_factory=dict)
    delegation_graph: dict[str, list[str]] = field(default_factory=dict)
    issues: dict[str, list[str]] = field(default_factory=dict)
    results: list[AgentAuditResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_agents": self.total_agents,
            "compliant_agents": self.compliant_agents,
            "non_compliant_agents": self.non_compliant_agents,
            "all_valid": self.all_valid,
            "total_fields": self.total_fields,
            "missing_field_count": self.missing_field_count,
            "by_category": self.by_category,
            "by_security_scope": self.by_security_scope,
            "by_cost_tier": self.by_cost_tier,
            "by_model": self.by_model,
            "by_memory": self.by_memory,
            "delegation_graph": self.delegation_graph,
            "issues": self.issues,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _load_agent_files(agents_dir: Path | None = None) -> list[tuple[Path, dict, str]]:
    """Load all agent markdown files and parse their frontmatter.

    Returns list of (file_path, frontmatter_dict, body) tuples.
    """
    if agents_dir is None:
        # Try the agentic-workflows repo first, fall back to .claude/agents
        aw_dir = Path.home() / "agentic-workflows" / "agentic-workflows" / "agents"
        if aw_dir.exists():
            agents_dir = aw_dir
        else:
            agents_dir = CLAUDE_AGENTS_DIR

    results = []
    for md_file in sorted(agents_dir.rglob("*.md")):
        if md_file.name == "MANIFEST.md":
            continue
        try:
            fm, body = AgentDefinition._parse_yaml_frontmatter(md_file.read_text(encoding="utf-8"))
            if fm:
                results.append((md_file, fm, body))
        except Exception:
            pass
    return results


def audit_agent(file_path: Path, fm: dict) -> AgentAuditResult:
    """Audit a single agent's frontmatter against v4.1 spec."""
    name = fm.get("name", file_path.stem)
    parent = file_path.parent.name
    category = parent if parent != "agents" else "expert-panel"

    missing = []
    invalid = []
    warnings = []

    # Check required fields
    for f in V41_REQUIRED_FIELDS:
        if f not in fm:
            missing.append(f)

    # Validate field values
    scope = fm.get("security_scope", "")
    if scope and scope not in VALID_SECURITY_SCOPES:
        invalid.append(f"security_scope '{scope}' not in {VALID_SECURITY_SCOPES}")

    cost = fm.get("cost_tier", "")
    if cost and cost not in VALID_COST_TIERS:
        invalid.append(f"cost_tier '{cost}' not in {VALID_COST_TIERS}")

    mem = fm.get("memory", "")
    if mem and mem not in VALID_MEMORY_MODES:
        invalid.append(f"memory '{mem}' not in {VALID_MEMORY_MODES}")

    ctx = fm.get("context_mode", "")
    if ctx and ctx not in VALID_CONTEXT_MODES:
        invalid.append(f"context_mode '{ctx}' not in {VALID_CONTEXT_MODES}")

    rec = fm.get("recovery", "")
    if rec and rec not in VALID_RECOVERY:
        invalid.append(f"recovery '{rec}' not in {VALID_RECOVERY}")

    model = fm.get("model", "")
    if model and model not in VALID_MODELS:
        invalid.append(f"model '{model}' not in {VALID_MODELS}")

    perm = fm.get("permissionMode", "")
    if perm and perm not in VALID_PERMISSION_MODES:
        invalid.append(f"permissionMode '{perm}' not in {VALID_PERMISSION_MODES}")

    # Security scope constraints
    if scope in SCOPE_REQUIRES_FALLBACK:
        recovery = fm.get("recovery", "")
        if recovery not in ("fallback", "checkpoint", "escalate"):
            warnings.append(
                f"scope '{scope}' should use fallback/checkpoint recovery, not '{recovery}'"
            )

    # Model-cost tier alignment
    model_cost_map = {"opus": "premium", "sonnet": "standard", "haiku": "low"}
    expected_cost = model_cost_map.get(model, "")
    if expected_cost and cost and cost != expected_cost:
        warnings.append(
            f"model '{model}' typically maps to cost_tier '{expected_cost}', got '{cost}'"
        )

    # Injection defense check
    guardrails = fm.get("guardrails", [])
    if isinstance(guardrails, list) and "injection-defense" not in guardrails:
        warnings.append("missing injection-defense in guardrails")

    compliant = len(missing) == 0 and len(invalid) == 0

    return AgentAuditResult(
        name=name,
        category=category,
        file_path=str(file_path),
        compliant=compliant,
        missing_fields=missing,
        invalid_values=invalid,
        warnings=warnings,
        field_count=len([f for f in V41_REQUIRED_FIELDS if f in fm]),
    )


def validate_all_agents(
    agents_dir: Path | None = None,
) -> tuple[bool, dict[str, list[str]]]:
    """Validate all agents against v4.1 spec.

    Returns:
        (all_valid, issues_dict) where issues_dict maps agent names
        to lists of issue descriptions.
    """
    agent_files = _load_agent_files(agents_dir)
    all_valid = True
    issues: dict[str, list[str]] = {}

    for file_path, fm, _body in agent_files:
        result = audit_agent(file_path, fm)
        if not result.compliant:
            all_valid = False
            agent_issues = []
            if result.missing_fields:
                agent_issues.append(f"missing: {', '.join(result.missing_fields)}")
            if result.invalid_values:
                agent_issues.extend(result.invalid_values)
            issues[result.name] = agent_issues

    return all_valid, issues


def get_audit_summary(agents_dir: Path | None = None) -> AuditSummary:
    """Get comprehensive audit summary of all agents.

    Returns AuditSummary with totals, breakdowns, delegation graph, and issues.
    """
    agent_files = _load_agent_files(agents_dir)
    summary = AuditSummary()
    summary.total_agents = len(agent_files)

    for file_path, fm, _body in agent_files:
        result = audit_agent(file_path, fm)
        summary.results.append(result)

        if result.compliant:
            summary.compliant_agents += 1
        else:
            summary.non_compliant_agents += 1
            issues = []
            if result.missing_fields:
                issues.append(f"missing: {', '.join(result.missing_fields)}")
                summary.missing_field_count += len(result.missing_fields)
            if result.invalid_values:
                issues.extend(result.invalid_values)
            summary.issues[result.name] = issues

        summary.total_fields += result.field_count

        # Count by category
        cat = result.category
        summary.by_category[cat] = summary.by_category.get(cat, 0) + 1

        # Count by security scope
        scope = fm.get("security_scope", "unknown")
        summary.by_security_scope[scope] = summary.by_security_scope.get(scope, 0) + 1

        # Count by cost tier
        tier = fm.get("cost_tier", "unknown")
        summary.by_cost_tier[tier] = summary.by_cost_tier.get(tier, 0) + 1

        # Count by model
        model = fm.get("model", "unknown")
        summary.by_model[model] = summary.by_model.get(model, 0) + 1

        # Count by memory
        mem = fm.get("memory", "unknown")
        summary.by_memory[mem] = summary.by_memory.get(mem, 0) + 1

        # Build delegation graph
        delegates = fm.get("delegates_to", [])
        if isinstance(delegates, list) and delegates:
            summary.delegation_graph[result.name] = delegates

    summary.all_valid = summary.non_compliant_agents == 0
    return summary


def print_audit_report(agents_dir: Path | None = None) -> AuditSummary:
    """Print a formatted audit report and return the summary."""
    summary = get_audit_summary(agents_dir)

    print("=" * 70)
    print("  AGENT AUDIT REPORT — SKILL.md v4.1 Compliance")
    print("=" * 70)
    print()
    print(f"  Total agents:      {summary.total_agents}")
    print(f"  Compliant:         {summary.compliant_agents}")
    print(f"  Non-compliant:     {summary.non_compliant_agents}")
    print(f"  All valid:         {'YES' if summary.all_valid else 'NO'}")
    print(f"  Total fields:      {summary.total_fields}")
    print()

    print("  By Category:")
    for cat, count in sorted(summary.by_category.items()):
        print(f"    {cat:25s} {count}")
    print()

    print("  By Security Scope:")
    for scope, count in sorted(summary.by_security_scope.items()):
        print(f"    {scope:25s} {count}")
    print()

    print("  By Cost Tier:")
    for tier, count in sorted(summary.by_cost_tier.items()):
        print(f"    {tier:25s} {count}")
    print()

    print("  By Model:")
    for model, count in sorted(summary.by_model.items()):
        print(f"    {model:25s} {count}")
    print()

    print("  By Memory Mode:")
    for mem, count in sorted(summary.by_memory.items()):
        print(f"    {mem:25s} {count}")
    print()

    if summary.delegation_graph:
        print("  Delegation Graph:")
        for agent, delegates in sorted(summary.delegation_graph.items()):
            print(f"    {agent} -> {', '.join(delegates)}")
        print()

    if summary.issues:
        print("  ISSUES:")
        for agent, issues in sorted(summary.issues.items()):
            for issue in issues:
                print(f"    [{agent}] {issue}")
        print()
    else:
        print("  No issues found. All agents are v4.1 compliant.")
        print()

    print("=" * 70)
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    agents_dir = None
    if len(sys.argv) > 1:
        agents_dir = Path(sys.argv[1])

    summary = print_audit_report(agents_dir)
    sys.exit(0 if summary.all_valid else 1)
