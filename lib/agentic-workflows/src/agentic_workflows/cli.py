"""CLI for Agentic Workflows.

Usage:
    agentic-cli discover          # Discover and list all skills
    agentic-cli search <query>    # Search for skills by keyword
    agentic-cli info <skill>      # Show detailed skill information
    agentic-cli validate          # Validate all skills and agents
    agentic-cli mcp               # Start MCP server (stdio)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from agentic_workflows.skills import (
    SkillRegistry,
    discover_default_skills,
    get_registry,
)
from agentic_workflows.tools import ToolSearchTool, create_tool_search_config
from agentic_workflows.agents import AgentSkillLoader, create_agent_loader


def cmd_discover(args):
    """Discover and list all skills."""
    count = discover_default_skills()
    registry = get_registry()

    print(f"Discovered {count} skills\n")

    # Group by domain
    by_domain: dict[str, list] = {}
    for name, skill in registry._skills.items():
        domain = skill.domain or "unknown"
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append((name, skill))

    for domain in sorted(by_domain.keys()):
        skills = by_domain[domain]
        print(f"[{domain}] ({len(skills)} skills)")
        for name, skill in sorted(skills, key=lambda x: x[0]):
            level = skill.level or "L1"
            desc = (skill.description[:60] + "...") if skill.description and len(skill.description) > 60 else (skill.description or "")
            print(f"  - {name} ({level}): {desc}")
        print()


def cmd_search(args):
    """Search for skills by keyword."""
    discover_default_skills()
    registry = get_registry()

    config = create_tool_search_config()
    tool = ToolSearchTool(config, skill_registry=registry)

    results = tool.search(args.query, args.category)

    if not results:
        print(f"No skills found matching '{args.query}'")
        return

    print(f"Found {len(results)} skills matching '{args.query}':\n")
    for r in results:
        print(f"  [{r.get('category', 'unknown')}] {r['name']}")
        if r.get('description'):
            print(f"      {r['description'][:80]}")
        print()


def cmd_info(args):
    """Show detailed skill information."""
    discover_default_skills()
    registry = get_registry()

    skill = registry.get_skill(args.skill)
    if not skill:
        print(f"Skill '{args.skill}' not found")
        sys.exit(1)

    print(f"Skill: {skill.name}")
    print(f"Level: {skill.level}")
    print(f"Domain: {skill.domain}")
    print(f"Description: {skill.description}")
    print(f"Allowed Tools: {', '.join(skill.allowed_tools) if skill.allowed_tools else 'None'}")
    print(f"Requires: {', '.join(skill.requires) if skill.requires else 'None'}")
    print(f"Optional: {', '.join(skill.optional) if skill.optional else 'None'}")
    print(f"Conflicts: {', '.join(skill.conflicts) if skill.conflicts else 'None'}")
    print(f"Defer Loading: {skill.defer_loading}")

    if args.show_context:
        print("\n--- Skill Context ---")
        context = registry.load_skill_context(args.skill)
        if context:
            print(context[:2000])
            if len(context) > 2000:
                print(f"\n... ({len(context) - 2000} more characters)")


def cmd_validate(args):
    """Validate all skills and agents."""
    print("Validating skills...")
    count = discover_default_skills()
    registry = get_registry()

    errors = []
    warnings = []

    # Validate skills
    for name, skill in registry._skills.items():
        # Check required fields
        if not skill.description:
            warnings.append(f"Skill '{name}' has no description")

        if not skill.allowed_tools:
            warnings.append(f"Skill '{name}' has no allowed_tools")

        # Check dependencies exist
        for dep in skill.requires:
            if dep not in registry._skills:
                errors.append(f"Skill '{name}' requires non-existent skill '{dep}'")

        # Check conflicts
        for conflict in skill.conflicts:
            if conflict not in registry._skills:
                warnings.append(f"Skill '{name}' conflicts with non-existent skill '{conflict}'")

    # Validate agents
    print("Validating agents...")
    loader = create_agent_loader()
    agent_count = loader.discover_agents()

    for name, agent in loader.agents.items():
        # Check skills exist
        for skill_name in agent.skills:
            if skill_name not in registry._skills:
                errors.append(f"Agent '{name}' references non-existent skill '{skill_name}'")

    # Report
    print(f"\nValidated {count} skills and {agent_count} agents\n")

    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    if not errors and not warnings:
        print("All validations passed!")

    sys.exit(1 if errors else 0)


def cmd_mcp(args):
    """Start MCP server."""
    from agentic_workflows.protocols.mcp_server import main as mcp_main
    mcp_main()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="agentic-cli",
        description="Agentic Workflows CLI - Manage skills, agents, and orchestration"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # discover command
    discover_parser = subparsers.add_parser("discover", help="Discover and list all skills")
    discover_parser.set_defaults(func=cmd_discover)

    # search command
    search_parser = subparsers.add_parser("search", help="Search for skills")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--category", "-c", help="Filter by category")
    search_parser.set_defaults(func=cmd_search)

    # info command
    info_parser = subparsers.add_parser("info", help="Show skill information")
    info_parser.add_argument("skill", help="Skill name")
    info_parser.add_argument("--show-context", "-s", action="store_true", help="Show skill context")
    info_parser.set_defaults(func=cmd_info)

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate skills and agents")
    validate_parser.set_defaults(func=cmd_validate)

    # mcp command
    mcp_parser = subparsers.add_parser("mcp", help="Start MCP server")
    mcp_parser.set_defaults(func=cmd_mcp)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
