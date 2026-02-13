#!/usr/bin/env python3
"""
PHUC Platform - Agent + Sub-Agent Hierarchy

Integrates:
- anthropic SDK (0.75.0) - Claude API
- mcp SDK (1.12.4) - Model Context Protocol
- openai SDK (2.15.0) - OpenAI API
- pydantic (2.12.5) - Data validation
- opentelemetry (1.38.0) - Observability

Architecture:
    Agent Layer:     Architect -> Builder -> Tester -> Shipper
                         |          |         |         |
    Sub-Agent Layer: Designer   Coder    UnitTest  Deployer
                     Researcher Documenter IntegTest Monitor
                     Planner    Refactor  SecAudit  Rollback
                                          PIICheck
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import anthropic
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, Field

# MCP imports
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Tracer
tracer = trace.get_tracer("phuc.agents")


class AgentRole(str, Enum):
    """Main agent roles in the PHUC pipeline."""

    ARCHITECT = "architect"
    BUILDER = "builder"
    TESTER = "tester"
    SHIPPER = "shipper"


class SubAgentRole(str, Enum):
    """Sub-agent roles under each main agent."""

    # Architect sub-agents
    DESIGNER = "designer"
    RESEARCHER = "researcher"
    PLANNER = "planner"

    # Builder sub-agents
    CODER = "coder"
    DOCUMENTER = "documenter"
    REFACTORER = "refactorer"

    # Tester sub-agents
    UNIT_TESTER = "unit_tester"
    INTEGRATION_TESTER = "integration_tester"
    SECURITY_AUDITOR = "security_auditor"
    PII_CHECKER = "pii_checker"

    # Shipper sub-agents
    DEPLOYER = "deployer"
    MONITOR = "monitor"
    ROLLBACKER = "rollbacker"


class AgentStatus(str, Enum):
    """Agent execution status."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class SubAgentConfig(BaseModel):
    """Configuration for a sub-agent."""

    role: SubAgentRole
    parent: AgentRole
    skills: list[str] = Field(default_factory=list)
    mcp_tools: list[str] = Field(default_factory=list)
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.7
    system_prompt: str | None = None


class AgentConfig(BaseModel):
    """Configuration for a main agent."""

    role: AgentRole
    sub_agents: list[SubAgentConfig] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    mcp_tools: list[str] = Field(default_factory=list)
    handoff_to: list[AgentRole] = Field(default_factory=list)
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 8192
    temperature: float = 0.5


class TaskResult(BaseModel):
    """Result from agent/sub-agent execution."""

    task_id: str
    agent: str
    sub_agent: str | None = None
    status: AgentStatus
    output: Any = None
    error: str | None = None
    duration_ms: int = 0
    tokens_used: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


@dataclass
class SubAgent:
    """Sub-agent that performs specialized tasks."""

    config: SubAgentConfig
    client: anthropic.Anthropic | None = None
    status: AgentStatus = AgentStatus.IDLE

    def __post_init__(self):
        if self.client is None:
            self.client = anthropic.Anthropic()

    @property
    def role(self) -> SubAgentRole:
        return self.config.role

    @property
    def parent(self) -> AgentRole:
        return self.config.parent

    async def execute(self, task: dict[str, Any]) -> TaskResult:
        """Execute a task using Claude API."""
        task_id = task.get("task_id", f"task_{int(datetime.now().timestamp())}")
        start_time = datetime.now()

        with tracer.start_as_current_span(f"sub_agent.{self.role.value}") as span:
            span.set_attribute("sub_agent.role", self.role.value)
            span.set_attribute("sub_agent.parent", self.parent.value)
            span.set_attribute("task.id", task_id)

            self.status = AgentStatus.RUNNING

            try:
                # Build system prompt
                system = self.config.system_prompt or self._default_system_prompt()

                # Call Claude API
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=system,
                    messages=[{"role": "user", "content": task.get("prompt", str(task))}],
                )

                # Extract response
                output = response.content[0].text if response.content else None
                tokens_used = response.usage.input_tokens + response.usage.output_tokens

                self.status = AgentStatus.COMPLETED
                span.set_status(Status(StatusCode.OK))

                return TaskResult(
                    task_id=task_id,
                    agent=self.parent.value,
                    sub_agent=self.role.value,
                    status=AgentStatus.COMPLETED,
                    output=output,
                    tokens_used=tokens_used,
                    duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                )

            except Exception as e:
                self.status = AgentStatus.FAILED
                span.set_status(Status(StatusCode.ERROR, str(e)))

                return TaskResult(
                    task_id=task_id,
                    agent=self.parent.value,
                    sub_agent=self.role.value,
                    status=AgentStatus.FAILED,
                    error=str(e),
                    duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                )

    def _default_system_prompt(self) -> str:
        """Generate default system prompt based on role."""
        prompts = {
            SubAgentRole.DESIGNER: "You are a system designer. Create architecture diagrams, UI/UX designs, and technical specifications.",
            SubAgentRole.RESEARCHER: "You are a technical researcher. Analyze requirements, research solutions, and provide recommendations.",
            SubAgentRole.PLANNER: "You are a project planner. Break down tasks, estimate effort, and create implementation plans.",
            SubAgentRole.CODER: "You are an expert coder. Write clean, efficient, production-ready code.",
            SubAgentRole.DOCUMENTER: "You are a technical writer. Create clear documentation, READMEs, and API docs.",
            SubAgentRole.REFACTORER: "You are a code refactoring expert. Improve code quality, performance, and maintainability.",
            SubAgentRole.UNIT_TESTER: "You are a unit testing expert. Write comprehensive unit tests with high coverage.",
            SubAgentRole.INTEGRATION_TESTER: "You are an integration testing expert. Test system interactions and API contracts.",
            SubAgentRole.SECURITY_AUDITOR: "You are a security auditor. Identify vulnerabilities, audit code, and recommend fixes.",
            SubAgentRole.PII_CHECKER: "You are a PII detection expert. Identify and flag personally identifiable information.",
            SubAgentRole.DEPLOYER: "You are a deployment expert. Deploy applications to Cloudflare Workers and other platforms.",
            SubAgentRole.MONITOR: "You are a monitoring expert. Set up observability, alerts, and performance tracking.",
            SubAgentRole.ROLLBACKER: "You are a rollback expert. Safely revert deployments and recover from failures.",
        }
        return prompts.get(self.role, "You are a helpful assistant.")


@dataclass
class Agent:
    """Main agent that coordinates sub-agents."""

    config: AgentConfig
    sub_agents: dict[SubAgentRole, SubAgent] = field(default_factory=dict)
    client: anthropic.Anthropic | None = None
    status: AgentStatus = AgentStatus.IDLE
    mcp_session: Any | None = None

    def __post_init__(self):
        if self.client is None:
            self.client = anthropic.Anthropic()

        # Initialize sub-agents
        for sub_config in self.config.sub_agents:
            self.sub_agents[sub_config.role] = SubAgent(config=sub_config, client=self.client)

    @property
    def role(self) -> AgentRole:
        return self.config.role

    async def connect_mcp(self, endpoint: str) -> dict[str, Any]:
        """Connect to MCP server."""
        if not MCP_AVAILABLE:
            return {"error": "MCP SDK not available"}

        try:
            async with sse_client(endpoint) as (read, write):
                self.mcp_session = ClientSession(read, write)
                await self.mcp_session.initialize()
                tools = await self.mcp_session.list_tools()
                return {"connected": True, "tools": len(tools.tools)}
        except Exception as e:
            return {"error": str(e)}

    async def delegate(self, task: dict[str, Any], sub_agent_role: SubAgentRole) -> TaskResult:
        """Delegate task to a sub-agent."""
        sub_agent = self.sub_agents.get(sub_agent_role)
        if not sub_agent:
            return TaskResult(
                task_id=task.get("task_id", "unknown"),
                agent=self.role.value,
                status=AgentStatus.FAILED,
                error=f"Sub-agent {sub_agent_role.value} not found",
            )

        with tracer.start_as_current_span(f"agent.{self.role.value}.delegate") as span:
            span.set_attribute("agent.role", self.role.value)
            span.set_attribute("sub_agent.target", sub_agent_role.value)

            return await sub_agent.execute(task)

    async def execute(self, task: dict[str, Any]) -> TaskResult:
        """Execute task directly (without sub-agent delegation)."""
        task_id = task.get("task_id", f"task_{int(datetime.now().timestamp())}")
        start_time = datetime.now()

        with tracer.start_as_current_span(f"agent.{self.role.value}") as span:
            span.set_attribute("agent.role", self.role.value)
            span.set_attribute("task.id", task_id)

            self.status = AgentStatus.RUNNING

            try:
                system = self._default_system_prompt()

                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=system,
                    messages=[{"role": "user", "content": task.get("prompt", str(task))}],
                )

                output = response.content[0].text if response.content else None
                tokens_used = response.usage.input_tokens + response.usage.output_tokens

                self.status = AgentStatus.COMPLETED
                span.set_status(Status(StatusCode.OK))

                return TaskResult(
                    task_id=task_id,
                    agent=self.role.value,
                    status=AgentStatus.COMPLETED,
                    output=output,
                    tokens_used=tokens_used,
                    duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                )

            except Exception as e:
                self.status = AgentStatus.FAILED
                span.set_status(Status(StatusCode.ERROR, str(e)))

                return TaskResult(
                    task_id=task_id,
                    agent=self.role.value,
                    status=AgentStatus.FAILED,
                    error=str(e),
                    duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                )

    async def handoff(self, task: dict[str, Any], target: AgentRole) -> dict[str, Any]:
        """Hand off task to another agent."""
        if target not in self.config.handoff_to:
            return {"error": f"Cannot hand off to {target.value}"}

        return {"handoff": True, "from": self.role.value, "to": target.value, "task": task}

    def _default_system_prompt(self) -> str:
        """Generate default system prompt based on role."""
        prompts = {
            AgentRole.ARCHITECT: """You are the Architect agent. Your responsibilities:
- Design system architecture and technical specifications
- Review requirements and create implementation plans
- Coordinate with Designer, Researcher, and Planner sub-agents
- Hand off to Builder when design is complete""",
            AgentRole.BUILDER: """You are the Builder agent. Your responsibilities:
- Write production-ready code following best practices
- Create documentation and API specifications
- Coordinate with Coder, Documenter, and Refactorer sub-agents
- Hand off to Tester when implementation is complete""",
            AgentRole.TESTER: """You are the Tester agent. Your responsibilities:
- Validate code quality and security
- Run unit tests, integration tests, and security audits
- Coordinate with UnitTester, IntegrationTester, SecurityAuditor, PIIChecker sub-agents
- Hand off to Shipper when tests pass, or back to Builder if fixes needed""",
            AgentRole.SHIPPER: """You are the Shipper agent. Your responsibilities:
- Deploy applications to production (Cloudflare Workers)
- Monitor performance and health
- Coordinate with Deployer, Monitor, and Rollbacker sub-agents
- Handle rollbacks if issues are detected""",
        }
        return prompts.get(self.role, "You are a helpful assistant.")


# Pre-configured agent hierarchy
def create_agent_hierarchy() -> dict[AgentRole, Agent]:
    """Create the full PHUC agent hierarchy with all sub-agents."""

    # Architect Agent
    architect_config = AgentConfig(
        role=AgentRole.ARCHITECT,
        sub_agents=[
            SubAgentConfig(
                role=SubAgentRole.DESIGNER,
                parent=AgentRole.ARCHITECT,
                skills=["frontend-design", "web-artifacts-builder"],
                mcp_tools=["skill_cloudflare-workers", "skill_cloudflare-ai"],
            ),
            SubAgentConfig(
                role=SubAgentRole.RESEARCHER,
                parent=AgentRole.ARCHITECT,
                skills=["mcp-builder", "agentic-workflows"],
                mcp_tools=["skill_search"],
            ),
            SubAgentConfig(
                role=SubAgentRole.PLANNER,
                parent=AgentRole.ARCHITECT,
                skills=["agentic-workflows"],
                mcp_tools=["skill_analytics-reporting"],
            ),
        ],
        skills=["frontend-design", "mcp-builder", "web-artifacts-builder", "agentic-workflows"],
        mcp_tools=["skill_cloudflare-workers", "skill_cloudflare-ai", "skill_search"],
        handoff_to=[AgentRole.BUILDER],
    )

    # Builder Agent
    builder_config = AgentConfig(
        role=AgentRole.BUILDER,
        sub_agents=[
            SubAgentConfig(
                role=SubAgentRole.CODER,
                parent=AgentRole.BUILDER,
                skills=["agentic-workflows", "frontend-design"],
                mcp_tools=[
                    "skill_cloudflare-workers",
                    "skill_cloudflare-d1",
                    "skill_cloudflare-r2",
                    "skill_cloudflare-vectorize",
                    "skill_cloudflare-ai",
                ],
            ),
            SubAgentConfig(
                role=SubAgentRole.DOCUMENTER,
                parent=AgentRole.BUILDER,
                skills=["agentic-workflows"],
                mcp_tools=["skill_cloudflare-r2"],
            ),
            SubAgentConfig(
                role=SubAgentRole.REFACTORER,
                parent=AgentRole.BUILDER,
                skills=["agentic-workflows"],
                mcp_tools=["skill_cloudflare-workers"],
            ),
        ],
        skills=["agentic-workflows", "frontend-design"],
        mcp_tools=[
            "skill_cloudflare-workers",
            "skill_cloudflare-d1",
            "skill_cloudflare-r2",
            "skill_cloudflare-vectorize",
            "skill_cloudflare-ai",
        ],
        handoff_to=[AgentRole.TESTER],
    )

    # Tester Agent
    tester_config = AgentConfig(
        role=AgentRole.TESTER,
        sub_agents=[
            SubAgentConfig(
                role=SubAgentRole.UNIT_TESTER,
                parent=AgentRole.TESTER,
                skills=["agentic-workflows"],
                mcp_tools=["skill_cloudflare-workers"],
            ),
            SubAgentConfig(
                role=SubAgentRole.INTEGRATION_TESTER,
                parent=AgentRole.TESTER,
                skills=["agentic-workflows"],
                mcp_tools=["skill_cloudflare-d1", "skill_cloudflare-r2"],
            ),
            SubAgentConfig(
                role=SubAgentRole.SECURITY_AUDITOR,
                parent=AgentRole.TESTER,
                skills=["agentic-workflows"],
                mcp_tools=["skill_injection-defense", "skill_scope-validator"],
            ),
            SubAgentConfig(
                role=SubAgentRole.PII_CHECKER,
                parent=AgentRole.TESTER,
                skills=["agentic-workflows"],
                mcp_tools=["skill_pii-detector"],
            ),
        ],
        skills=["agentic-workflows"],
        mcp_tools=["skill_injection-defense", "skill_scope-validator", "skill_pii-detector"],
        handoff_to=[AgentRole.SHIPPER, AgentRole.BUILDER],
    )

    # Shipper Agent
    shipper_config = AgentConfig(
        role=AgentRole.SHIPPER,
        sub_agents=[
            SubAgentConfig(
                role=SubAgentRole.DEPLOYER,
                parent=AgentRole.SHIPPER,
                skills=["agentic-workflows"],
                mcp_tools=["skill_cloudflare-workers"],
            ),
            SubAgentConfig(
                role=SubAgentRole.MONITOR,
                parent=AgentRole.SHIPPER,
                skills=["agentic-workflows"],
                mcp_tools=["skill_analytics-reporting", "skill_analytics-attribution"],
            ),
            SubAgentConfig(
                role=SubAgentRole.ROLLBACKER,
                parent=AgentRole.SHIPPER,
                skills=["agentic-workflows"],
                mcp_tools=["skill_cloudflare-workers"],
            ),
        ],
        skills=["agentic-workflows"],
        mcp_tools=[
            "skill_cloudflare-workers",
            "skill_analytics-reporting",
            "skill_analytics-attribution",
            "skill_analytics-campaign",
        ],
        handoff_to=[],
    )

    return {
        AgentRole.ARCHITECT: Agent(config=architect_config),
        AgentRole.BUILDER: Agent(config=builder_config),
        AgentRole.TESTER: Agent(config=tester_config),
        AgentRole.SHIPPER: Agent(config=shipper_config),
    }


# Singleton hierarchy
AGENT_HIERARCHY = create_agent_hierarchy()


def get_agent(role: AgentRole) -> Agent:
    """Get an agent by role."""
    return AGENT_HIERARCHY.get(role)


def get_sub_agent(agent_role: AgentRole, sub_role: SubAgentRole) -> SubAgent | None:
    """Get a sub-agent by agent role and sub-agent role."""
    agent = AGENT_HIERARCHY.get(agent_role)
    if agent:
        return agent.sub_agents.get(sub_role)
    return None


def list_agents() -> list[dict[str, Any]]:
    """List all agents and their sub-agents."""
    result = []
    for role, agent in AGENT_HIERARCHY.items():
        result.append(
            {
                "role": role.value,
                "sub_agents": [sa.role.value for sa in agent.sub_agents.values()],
                "skills": agent.config.skills,
                "mcp_tools": agent.config.mcp_tools,
                "handoff_to": [h.value for h in agent.config.handoff_to],
            }
        )
    return result
