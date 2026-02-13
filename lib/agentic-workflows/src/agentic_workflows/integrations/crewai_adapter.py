"""CrewAI Adapter for Agentic Workflows.

Provides integration with CrewAI framework.

Usage:
    from agentic_workflows.integrations.crewai_adapter import CrewAIAdapter, CrewAIAgent

    adapter = CrewAIAdapter()
    crew_agent = adapter.to_crewai(our_agent)
    crew = adapter.create_crew([agent1, agent2], tasks)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CrewAIAgent:
    """CrewAI-compatible agent definition.

    Maps to crewai.Agent interface.
    """

    role: str
    goal: str
    backstory: str = ""
    tools: List[Any] = field(default_factory=list)
    llm: Optional[str] = None
    verbose: bool = False
    allow_delegation: bool = True
    memory: bool = True
    max_iter: int = 15
    max_rpm: Optional[int] = None


@dataclass
class CrewAITask:
    """CrewAI-compatible task definition."""

    description: str
    agent: CrewAIAgent
    expected_output: str = ""
    context: Optional[List["CrewAITask"]] = None
    tools: Optional[List[Any]] = None
    async_execution: bool = False


class CrewAIAdapter:
    """Adapter for CrewAI framework.

    Example:
        adapter = CrewAIAdapter()

        # Convert our agent to CrewAI agent
        crew_agent = adapter.to_crewai(our_agent)

        # Create a crew
        crew = adapter.create_crew(
            agents=[agent1, agent2],
            tasks=[task1, task2],
        )

        # Run crew
        result = adapter.run_crew(crew)
    """

    def __init__(self, verbose: bool = False):
        """Initialize adapter.

        Args:
            verbose: Enable verbose output.
        """
        self.verbose = verbose
        self._crewai_available = self._check_crewai()

    def _check_crewai(self) -> bool:
        """Check if CrewAI is available."""
        try:
            import crewai
            return True
        except ImportError:
            logger.warning("crewai package not installed")
            return False

    def to_crewai(self, agent: Any) -> CrewAIAgent:
        """Convert agent to CrewAI format.

        Args:
            agent: Our agent definition.

        Returns:
            CrewAIAgent compatible with CrewAI.
        """
        # Extract from our agent format
        name = getattr(agent, "name", "agent")
        description = getattr(agent, "description", "")
        instructions = getattr(agent, "instructions", "")

        # Map to CrewAI structure
        return CrewAIAgent(
            role=name,
            goal=description or f"Execute tasks as {name}",
            backstory=instructions,
            tools=getattr(agent, "tools", []),
            verbose=self.verbose,
            allow_delegation=True,
            memory=True,
        )

    def from_crewai(self, crew_agent: Any) -> Dict[str, Any]:
        """Convert CrewAI agent to our format.

        Args:
            crew_agent: CrewAI Agent instance.

        Returns:
            Dictionary in our agent format.
        """
        return {
            "name": getattr(crew_agent, "role", "agent"),
            "description": getattr(crew_agent, "goal", ""),
            "instructions": getattr(crew_agent, "backstory", ""),
            "tools": getattr(crew_agent, "tools", []),
        }

    def create_crew(
        self,
        agents: List[Any],
        tasks: List[Dict[str, Any]],
        process: str = "sequential",
        verbose: bool = False,
    ) -> Any:
        """Create a CrewAI crew.

        Args:
            agents: List of agents (our format or CrewAI).
            tasks: List of task definitions.
            process: Process type (sequential/hierarchical).
            verbose: Enable verbose output.

        Returns:
            CrewAI Crew instance.
        """
        if not self._crewai_available:
            raise RuntimeError("crewai package not installed")

        from crewai import Agent, Crew, Task, Process

        # Convert agents
        crew_agents = []
        for agent in agents:
            if isinstance(agent, CrewAIAgent):
                crew_agents.append(Agent(
                    role=agent.role,
                    goal=agent.goal,
                    backstory=agent.backstory,
                    tools=agent.tools,
                    verbose=agent.verbose,
                    allow_delegation=agent.allow_delegation,
                    memory=agent.memory,
                ))
            elif hasattr(agent, "role"):
                # Already a CrewAI agent
                crew_agents.append(agent)
            else:
                # Convert from our format
                ca = self.to_crewai(agent)
                crew_agents.append(Agent(
                    role=ca.role,
                    goal=ca.goal,
                    backstory=ca.backstory,
                    tools=ca.tools,
                    verbose=ca.verbose,
                ))

        # Create tasks
        crew_tasks = []
        agent_map = {a.role: a for a in crew_agents}

        for task_def in tasks:
            agent_role = task_def.get("agent", crew_agents[0].role)
            task_agent = agent_map.get(agent_role, crew_agents[0])

            crew_tasks.append(Task(
                description=task_def.get("description", ""),
                agent=task_agent,
                expected_output=task_def.get("expected_output", ""),
            ))

        # Create crew
        process_type = Process.sequential if process == "sequential" else Process.hierarchical

        return Crew(
            agents=crew_agents,
            tasks=crew_tasks,
            process=process_type,
            verbose=verbose,
        )

    def run_crew(self, crew: Any, inputs: Optional[Dict[str, Any]] = None) -> str:
        """Run a CrewAI crew.

        Args:
            crew: CrewAI Crew instance.
            inputs: Optional inputs for the crew.

        Returns:
            Crew execution result.
        """
        if not self._crewai_available:
            raise RuntimeError("crewai package not installed")

        return crew.kickoff(inputs=inputs or {})


__all__ = [
    "CrewAIAgent",
    "CrewAITask",
    "CrewAIAdapter",
]
