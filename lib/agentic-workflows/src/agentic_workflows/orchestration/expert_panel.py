"""
Expert Panel Orchestration System.

Implements a sophisticated multi-agent workflow where:
1. Expert Panel (highest reasoning models) analyzes requests first
2. Creates multi-agent task list with assignments, checkers, correctors
3. Each agent has note-taker logging to scratchpad
4. Produces aggregated LLM handoff document at end

Architecture:
    Request
        |
        v
    +-------------------+
    | EXPERT PANEL      |  (Opus/highest reasoning)
    | - Analyst         |  Understands request deeply
    | - Architect       |  Designs approach
    | - Risk Assessor   |  Identifies risks
    +-------------------+
        |
        v
    +-------------------+
    | TASK DECOMPOSITION|
    | - Task List       |
    | - Assignments     |
    | - Dependencies    |
    | - Checkers        |
    | - Correctors      |
    +-------------------+
        |
        v
    +-------------------+
    | EXECUTION ENGINE  |  (Each agent has note-taker)
    | Agent + NoteTaker |---> Scratchpad
    | Agent + NoteTaker |---> Scratchpad
    | ...               |
    +-------------------+
        |
        v
    +-------------------+
    | QUALITY GATES     |
    | - Checker Review  |
    | - Corrector Fix   |
    | - Approval Gate   |
    +-------------------+
        |
        v
    +-------------------+
    | HANDOFF GENERATOR |
    | - Aggregate all   |
    | - Create document |
    +-------------------+
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
import asyncio
import json
import uuid

from ..core.scratchpad import Scratchpad, ThoughtType
from ..core.artifact import ArtifactBuilder, ArtifactType


class AgentRole(Enum):
    """Roles in the expert panel workflow."""
    # Expert Panel (Analysis Phase)
    ANALYST = "analyst"           # Understands the request deeply
    ARCHITECT = "architect"       # Designs the approach
    RISK_ASSESSOR = "risk_assessor"  # Identifies risks and constraints

    # Execution Phase
    WORKER = "worker"             # Executes assigned tasks
    NOTE_TAKER = "note_taker"     # Logs everything for the worker

    # Quality Phase
    CHECKER = "checker"           # Reviews work for correctness
    CORRECTOR = "corrector"       # Fixes issues found by checker

    # Coordination
    COORDINATOR = "coordinator"   # Manages workflow
    AGGREGATOR = "aggregator"     # Creates final handoff document


class TaskStatus(Enum):
    """Status of a task in the workflow."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    CHECKING = "checking"
    CORRECTING = "correcting"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class AgentAssignment:
    """Assignment of an agent to a task."""
    agent_id: str
    role: AgentRole
    model: str  # haiku, sonnet, opus
    tools: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)


@dataclass
class TaskDefinition:
    """A task in the multi-agent workflow."""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM

    # Assignments
    worker: Optional[AgentAssignment] = None
    note_taker: Optional[AgentAssignment] = None
    checker: Optional[AgentAssignment] = None
    corrector: Optional[AgentAssignment] = None

    # Dependencies
    depends_on: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)

    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    attempts: int = 0
    max_attempts: int = 3

    # Results
    result: Optional[Any] = None
    error: Optional[str] = None

    # Notes from note-taker
    scratchpad: Optional[Scratchpad] = None

    # Checker feedback
    checker_feedback: Optional[str] = None
    checker_approved: bool = False

    # Corrector changes
    corrections: list[str] = field(default_factory=list)


@dataclass
class ExpertAnalysis:
    """Analysis from the expert panel."""
    # From Analyst
    request_understanding: str
    key_requirements: list[str]
    implicit_requirements: list[str]
    clarifications_needed: list[str]

    # From Architect
    approach: str
    task_breakdown: list[dict]
    agent_assignments: dict[str, AgentAssignment]
    execution_order: list[str]

    # From Risk Assessor
    risks: list[dict]  # {risk, severity, mitigation}
    constraints: list[str]
    approval_gates: list[str]  # Points requiring human approval

    # Consensus
    confidence: float  # 0.0 to 1.0
    dissenting_opinions: list[str]


@dataclass
class WorkflowResult:
    """Final result of the expert panel workflow."""
    success: bool
    request: str

    # Analysis phase
    expert_analysis: ExpertAnalysis

    # Execution phase
    tasks: list[TaskDefinition]
    completed_tasks: int
    failed_tasks: int

    # Quality phase
    checker_issues_found: int
    corrections_made: int

    # Aggregated output
    final_output: Any
    handoff_document: str

    # Metrics
    total_tokens: int = 0
    total_cost: float = 0.0
    duration_seconds: float = 0.0

    # Scratchpads (for audit)
    all_scratchpads: dict[str, dict] = field(default_factory=dict)


class NoteTaker:
    """
    Note-taker that accompanies each agent, logging everything to scratchpad.

    Every significant action, decision, observation, and reasoning is captured.
    """

    def __init__(self, task_id: str, agent_id: str):
        self.task_id = task_id
        self.agent_id = agent_id
        self.scratchpad = Scratchpad(max_entries=500)
        self.notes: list[dict] = []  # Track all notes for summary
        self._start_time = datetime.now()

    def log_start(self, task_description: str, context: dict):
        """Log task start."""
        self.scratchpad.add_goal(f"Task: {task_description}", priority=8)
        self.scratchpad.add(
            ThoughtType.CONTEXT,
            f"Initial context: {json.dumps(context, indent=2)}",
            priority=7,
            tags=["context"]
        )
        self.notes.append({"type": "start", "task": task_description, "context": context})

    def log_observation(self, observation: str, importance: int = 5):
        """Log an observation."""
        self.scratchpad.add_observation(observation, tags=["observation", f"importance_{importance}"])
        self.notes.append({"type": "observation", "content": observation, "importance": importance})

    def log_decision(self, decision: str, reasoning: str, alternatives: list[str] = None):
        """Log a decision with full reasoning."""
        full_reasoning = reasoning
        if alternatives:
            full_reasoning += f" | Alternatives: {', '.join(alternatives)}"
        self.scratchpad.add_decision(decision=decision, reasoning=full_reasoning)
        self.notes.append({"type": "decision", "decision": decision, "reasoning": reasoning, "alternatives": alternatives})

    def log_action(self, action: str, params: dict = None):
        """Log an action taken."""
        content = f"Action: {action}"
        if params:
            content += f"\nParameters: {json.dumps(params, indent=2)}"
        self.scratchpad.add_progress(content)
        self.notes.append({"type": "action", "action": action, "params": params})

    def log_result(self, result: Any, success: bool):
        """Log a result."""
        status = "SUCCESS" if success else "FAILURE"
        thought_type = ThoughtType.DONE if success else ThoughtType.WARNING
        self.scratchpad.add(
            thought_type,
            f"Result ({status}): {str(result)[:500]}",
            priority=8 if not success else 6,
            tags=["result", status.lower()]
        )
        self.notes.append({"type": "result", "result": str(result)[:500], "success": success})

    def log_error(self, error: str, context: str = None):
        """Log an error."""
        content = f"ERROR: {error}"
        if context:
            content += f"\nContext: {context}"
        self.scratchpad.add_blocker(content, severity="high")
        self.notes.append({"type": "error", "error": error, "context": context})

    def log_hypothesis(self, hypothesis: str, confidence: float):
        """Log a hypothesis."""
        self.scratchpad.add_hypothesis(hypothesis=hypothesis, confidence=confidence)
        self.notes.append({"type": "hypothesis", "hypothesis": hypothesis, "confidence": confidence})

    def log_question(self, question: str, for_whom: str = "self"):
        """Log a question."""
        self.scratchpad.add_question(f"[For {for_whom}] {question}", priority=7)
        self.notes.append({"type": "question", "question": question, "for_whom": for_whom})

    def log_blocker(self, blocker: str, severity: str = "medium"):
        """Log a blocker."""
        self.scratchpad.add_blocker(blocker, severity=severity)
        self.notes.append({"type": "blocker", "blocker": blocker, "severity": severity})

    def log_handoff_note(self, note: str, for_agent: str):
        """Log a note for the next agent in the chain."""
        self.scratchpad.add(
            ThoughtType.NOTE,
            f"[HANDOFF to {for_agent}] {note}",
            priority=8,
            tags=["handoff", for_agent]
        )
        self.notes.append({"type": "handoff", "note": note, "for_agent": for_agent})

    def get_summary(self) -> dict:
        """Get summary of all notes."""
        return self.scratchpad.get_summary()

    def export(self) -> dict:
        """Export all notes for aggregation."""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "duration_seconds": (datetime.now() - self._start_time).total_seconds(),
            "entries": self.scratchpad.export(),
            "summary": self.get_summary()
        }


class ExpertPanel:
    """
    Expert Panel that analyzes requests before execution.

    Uses highest reasoning models (Opus) to:
    1. Deeply understand the request
    2. Design the approach
    3. Identify risks and constraints
    4. Create task assignments with checkers and correctors
    """

    def __init__(
        self,
        analyst_model: str = "opus",
        architect_model: str = "opus",
        risk_model: str = "sonnet"
    ):
        self.analyst_model = analyst_model
        self.architect_model = architect_model
        self.risk_model = risk_model
        self.scratchpad = Scratchpad(max_entries=200)

    async def analyze(
        self,
        request: str,
        context: dict = None,
        available_agents: list[str] = None
    ) -> ExpertAnalysis:
        """
        Perform expert panel analysis of the request.

        Returns a complete analysis with task breakdown and assignments.
        """
        context = context or {}
        available_agents = available_agents or self._get_default_agents()

        # Log the analysis start
        self.scratchpad.add_goal(
            f"Analyze request: {request[:100]}...",
            priority=10,
            tags=["analysis_start"]
        )

        # Phase 1: Analyst - Deep understanding
        analyst_result = await self._run_analyst(request, context)

        # Phase 2: Architect - Design approach
        architect_result = await self._run_architect(
            request, analyst_result, available_agents
        )

        # Phase 3: Risk Assessor - Identify risks
        risk_result = await self._run_risk_assessor(
            request, analyst_result, architect_result
        )

        # Combine into ExpertAnalysis
        analysis = ExpertAnalysis(
            # Analyst
            request_understanding=analyst_result["understanding"],
            key_requirements=analyst_result["key_requirements"],
            implicit_requirements=analyst_result["implicit_requirements"],
            clarifications_needed=analyst_result["clarifications"],

            # Architect
            approach=architect_result["approach"],
            task_breakdown=architect_result["tasks"],
            agent_assignments=architect_result["assignments"],
            execution_order=architect_result["order"],

            # Risk Assessor
            risks=risk_result["risks"],
            constraints=risk_result["constraints"],
            approval_gates=risk_result["approval_gates"],

            # Consensus
            confidence=self._calculate_confidence(
                analyst_result, architect_result, risk_result
            ),
            dissenting_opinions=[]
        )

        self.scratchpad.add_entry(
            type=ThoughtType.DONE,
            content=f"Analysis complete. Confidence: {analysis.confidence:.2f}",
            priority=10,
            tags=["analysis_complete"]
        )

        return analysis

    async def _run_analyst(self, request: str, context: dict) -> dict:
        """Run the analyst to understand the request deeply."""
        self.scratchpad.add_entry(
            type=ThoughtType.PROGRESS,
            content="Running Analyst (deep understanding)",
            priority=8,
            tags=["analyst"]
        )

        # In production, this would call the LLM
        # For now, return structured analysis
        return {
            "understanding": f"User wants to: {request}",
            "key_requirements": self._extract_requirements(request),
            "implicit_requirements": self._infer_implicit_requirements(request),
            "clarifications": []
        }

    async def _run_architect(
        self,
        request: str,
        analyst_result: dict,
        available_agents: list[str]
    ) -> dict:
        """Run the architect to design the approach."""
        self.scratchpad.add_entry(
            type=ThoughtType.PROGRESS,
            content="Running Architect (designing approach)",
            priority=8,
            tags=["architect"]
        )

        # Create task breakdown
        tasks = self._create_task_breakdown(request, analyst_result)

        # Assign agents to tasks
        assignments = self._assign_agents(tasks, available_agents)

        # Determine execution order
        order = self._determine_order(tasks)

        return {
            "approach": f"Multi-agent approach with {len(tasks)} tasks",
            "tasks": tasks,
            "assignments": assignments,
            "order": order
        }

    async def _run_risk_assessor(
        self,
        request: str,
        analyst_result: dict,
        architect_result: dict
    ) -> dict:
        """Run the risk assessor to identify risks."""
        self.scratchpad.add_entry(
            type=ThoughtType.PROGRESS,
            content="Running Risk Assessor (identifying risks)",
            priority=8,
            tags=["risk_assessor"]
        )

        return {
            "risks": self._identify_risks(request, architect_result),
            "constraints": self._identify_constraints(request),
            "approval_gates": self._identify_approval_gates(architect_result)
        }

    def _get_default_agents(self) -> list[str]:
        """Get default available agents."""
        return [
            "code-reviewer", "debugger", "test-writer", "security-auditor",
            "data-analyst", "sql-expert", "researcher", "technical-writer",
            "devops", "architect", "planner", "orchestrator"
        ]

    def _extract_requirements(self, request: str) -> list[str]:
        """Extract key requirements from request."""
        requirements = []

        # Simple keyword extraction (in production, use LLM)
        if "secure" in request.lower() or "security" in request.lower():
            requirements.append("Security review required")
        if "test" in request.lower():
            requirements.append("Testing required")
        if "document" in request.lower() or "docs" in request.lower():
            requirements.append("Documentation required")
        if "fast" in request.lower() or "performance" in request.lower():
            requirements.append("Performance optimization")

        if not requirements:
            requirements.append("Complete the requested task")

        return requirements

    def _infer_implicit_requirements(self, request: str) -> list[str]:
        """Infer implicit requirements."""
        implicit = []

        # Always implied
        implicit.append("Maintain code quality standards")
        implicit.append("Follow security best practices")
        implicit.append("Document decisions and reasoning")

        return implicit

    def _create_task_breakdown(self, request: str, analyst_result: dict) -> list[dict]:
        """Break down the request into tasks."""
        tasks = []

        # Always start with analysis
        tasks.append({
            "id": "task_1",
            "description": "Analyze and understand the request",
            "type": "analysis",
            "priority": TaskPriority.CRITICAL.value
        })

        # Main work
        tasks.append({
            "id": "task_2",
            "description": f"Execute: {request[:100]}",
            "type": "execution",
            "priority": TaskPriority.HIGH.value,
            "depends_on": ["task_1"]
        })

        # Review
        tasks.append({
            "id": "task_3",
            "description": "Review and validate work",
            "type": "review",
            "priority": TaskPriority.HIGH.value,
            "depends_on": ["task_2"]
        })

        # Documentation
        tasks.append({
            "id": "task_4",
            "description": "Document changes and create handoff",
            "type": "documentation",
            "priority": TaskPriority.MEDIUM.value,
            "depends_on": ["task_3"]
        })

        return tasks

    def _assign_agents(
        self,
        tasks: list[dict],
        available_agents: list[str]
    ) -> dict[str, AgentAssignment]:
        """Assign agents to tasks with checkers and correctors."""
        assignments = {}

        for task in tasks:
            task_id = task["id"]
            task_type = task.get("type", "execution")

            # Assign worker based on task type
            if task_type == "analysis":
                worker_agent = "planner"
                worker_model = "opus"
            elif task_type == "review":
                worker_agent = "code-reviewer"
                worker_model = "sonnet"
            elif task_type == "documentation":
                worker_agent = "technical-writer"
                worker_model = "haiku"
            else:
                worker_agent = "orchestrator"
                worker_model = "sonnet"

            assignments[task_id] = {
                "worker": AgentAssignment(
                    agent_id=f"{worker_agent}_{task_id}",
                    role=AgentRole.WORKER,
                    model=worker_model,
                    tools=["Read", "Write", "Edit", "Glob", "Grep"]
                ),
                "note_taker": AgentAssignment(
                    agent_id=f"note_taker_{task_id}",
                    role=AgentRole.NOTE_TAKER,
                    model="haiku",  # Cheap model for logging
                    tools=[]
                ),
                "checker": AgentAssignment(
                    agent_id=f"checker_{task_id}",
                    role=AgentRole.CHECKER,
                    model="sonnet",  # Good model for review
                    tools=["Read", "Glob", "Grep"]
                ),
                "corrector": AgentAssignment(
                    agent_id=f"corrector_{task_id}",
                    role=AgentRole.CORRECTOR,
                    model="sonnet",
                    tools=["Read", "Write", "Edit"]
                )
            }

        return assignments

    def _determine_order(self, tasks: list[dict]) -> list[str]:
        """Determine execution order based on dependencies."""
        # Simple topological sort
        order = []
        remaining = {t["id"]: t for t in tasks}
        completed = set()

        while remaining:
            # Find tasks with all dependencies met
            ready = [
                tid for tid, task in remaining.items()
                if all(dep in completed for dep in task.get("depends_on", []))
            ]

            if not ready:
                # Circular dependency or error
                order.extend(remaining.keys())
                break

            # Add ready tasks to order
            order.extend(sorted(ready))
            for tid in ready:
                completed.add(tid)
                del remaining[tid]

        return order

    def _identify_risks(self, request: str, architect_result: dict) -> list[dict]:
        """Identify risks in the approach."""
        risks = []

        # Check for security-related requests
        if any(word in request.lower() for word in ["delete", "remove", "drop", "truncate"]):
            risks.append({
                "risk": "Destructive operation requested",
                "severity": "high",
                "mitigation": "Require explicit human approval before execution"
            })

        # Check for external interactions
        if any(word in request.lower() for word in ["api", "external", "send", "post"]):
            risks.append({
                "risk": "External service interaction",
                "severity": "medium",
                "mitigation": "Use circuit breaker and rate limiting"
            })

        return risks

    def _identify_constraints(self, request: str) -> list[str]:
        """Identify constraints on execution."""
        constraints = [
            "Must complete within budget limits",
            "Must not expose sensitive data",
            "Must maintain audit trail"
        ]
        return constraints

    def _identify_approval_gates(self, architect_result: dict) -> list[str]:
        """Identify points requiring human approval."""
        gates = []

        for task in architect_result.get("tasks", []):
            if task.get("type") == "execution":
                gates.append(f"Before executing {task['id']}: Review approach")

        return gates

    def _calculate_confidence(
        self,
        analyst_result: dict,
        architect_result: dict,
        risk_result: dict
    ) -> float:
        """Calculate confidence in the analysis."""
        confidence = 0.8  # Base confidence

        # Reduce confidence for high risks
        high_risks = sum(1 for r in risk_result["risks"] if r["severity"] == "high")
        confidence -= high_risks * 0.1

        # Reduce confidence for clarifications needed
        clarifications = len(analyst_result.get("clarifications", []))
        confidence -= clarifications * 0.05

        return max(0.3, min(1.0, confidence))


class ExpertPanelWorkflow:
    """
    Complete Expert Panel Workflow orchestration.

    Coordinates:
    1. Expert Panel analysis
    2. Task execution with note-takers
    3. Checker/Corrector quality gates
    4. Aggregated handoff document generation
    """

    def __init__(
        self,
        expert_panel: ExpertPanel = None,
        max_correction_cycles: int = 2
    ):
        self.expert_panel = expert_panel or ExpertPanel()
        self.max_correction_cycles = max_correction_cycles
        self.tasks: dict[str, TaskDefinition] = {}
        self.note_takers: dict[str, NoteTaker] = {}
        self.artifact_builder = ArtifactBuilder(agent_id="expert_panel_workflow")
        self._start_time: Optional[datetime] = None

    async def execute(
        self,
        request: str,
        context: dict = None,
        executor: Callable = None
    ) -> WorkflowResult:
        """
        Execute the complete expert panel workflow.

        Args:
            request: The user's request
            context: Additional context
            executor: Function to execute individual tasks

        Returns:
            WorkflowResult with all outputs and handoff document
        """
        self._start_time = datetime.now()
        context = context or {}

        # Phase 1: Expert Panel Analysis
        analysis = await self.expert_panel.analyze(request, context)

        # Phase 2: Create Task Definitions
        self._create_tasks_from_analysis(analysis)

        # Phase 3: Execute Tasks with Note-Takers
        execution_results = await self._execute_all_tasks(executor)

        # Phase 4: Run Quality Gates
        quality_results = await self._run_quality_gates(executor)

        # Phase 5: Generate Aggregated Handoff Document
        handoff_doc = self._generate_handoff_document(
            request, analysis, execution_results, quality_results
        )

        # Compile final result
        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)

        duration = (datetime.now() - self._start_time).total_seconds()

        return WorkflowResult(
            success=failed == 0,
            request=request,
            expert_analysis=analysis,
            tasks=list(self.tasks.values()),
            completed_tasks=completed,
            failed_tasks=failed,
            checker_issues_found=quality_results["issues_found"],
            corrections_made=quality_results["corrections_made"],
            final_output=execution_results.get("final_output"),
            handoff_document=handoff_doc,
            duration_seconds=duration,
            all_scratchpads={
                tid: nt.export() for tid, nt in self.note_takers.items()
            }
        )

    def _create_tasks_from_analysis(self, analysis: ExpertAnalysis):
        """Create TaskDefinitions from expert analysis."""
        for task_data in analysis.task_breakdown:
            task_id = task_data["id"]
            assignments = analysis.agent_assignments.get(task_id, {})

            task = TaskDefinition(
                id=task_id,
                description=task_data["description"],
                priority=TaskPriority(task_data.get("priority", 3)),
                depends_on=task_data.get("depends_on", []),
                worker=assignments.get("worker"),
                note_taker=assignments.get("note_taker"),
                checker=assignments.get("checker"),
                corrector=assignments.get("corrector"),
                scratchpad=Scratchpad(max_entries=100)
            )

            self.tasks[task_id] = task

            # Create note-taker for this task
            self.note_takers[task_id] = NoteTaker(
                task_id=task_id,
                agent_id=task.worker.agent_id if task.worker else "unknown"
            )

    async def _execute_all_tasks(self, executor: Callable) -> dict:
        """Execute all tasks in order with note-taking."""
        results = {}

        for task_id in self._get_execution_order():
            task = self.tasks[task_id]
            note_taker = self.note_takers[task_id]

            # Log task start
            note_taker.log_start(task.description, {"task_id": task_id})

            # Check dependencies
            if not self._dependencies_met(task):
                note_taker.log_blocker(
                    f"Dependencies not met: {task.depends_on}",
                    severity="high"
                )
                task.status = TaskStatus.BLOCKED
                continue

            # Execute task
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now()

            try:
                if executor:
                    result = await executor(task, note_taker)
                else:
                    result = await self._default_executor(task, note_taker)

                task.result = result
                task.status = TaskStatus.CHECKING  # Move to quality gate
                note_taker.log_result(result, success=True)
                results[task_id] = result

            except Exception as e:
                task.error = str(e)
                task.status = TaskStatus.FAILED
                note_taker.log_error(str(e), context=f"Task: {task.description}")

        return {"results": results, "final_output": results.get(list(results.keys())[-1]) if results else None}

    async def _run_quality_gates(self, executor: Callable) -> dict:
        """Run checker and corrector on completed tasks."""
        issues_found = 0
        corrections_made = 0

        for task_id, task in self.tasks.items():
            if task.status != TaskStatus.CHECKING:
                continue

            note_taker = self.note_takers[task_id]

            # Run checker
            note_taker.log_action("Running checker", {"task_id": task_id})
            checker_result = await self._run_checker(task, note_taker)

            if checker_result["approved"]:
                task.checker_approved = True
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                note_taker.log_result("Checker approved", success=True)
            else:
                issues_found += len(checker_result["issues"])
                task.checker_feedback = checker_result["feedback"]

                # Run corrector
                for cycle in range(self.max_correction_cycles):
                    note_taker.log_action(
                        f"Running corrector (cycle {cycle + 1})",
                        {"issues": checker_result["issues"]}
                    )

                    task.status = TaskStatus.CORRECTING
                    correction_result = await self._run_corrector(
                        task, checker_result["issues"], note_taker
                    )

                    if correction_result["fixed"]:
                        corrections_made += len(correction_result["corrections"])
                        task.corrections.extend(correction_result["corrections"])

                        # Re-check
                        checker_result = await self._run_checker(task, note_taker)
                        if checker_result["approved"]:
                            task.checker_approved = True
                            task.status = TaskStatus.COMPLETED
                            task.completed_at = datetime.now()
                            break

                if not task.checker_approved:
                    task.status = TaskStatus.FAILED
                    note_taker.log_error(
                        "Failed quality gate after max correction cycles",
                        context=checker_result["feedback"]
                    )

        return {
            "issues_found": issues_found,
            "corrections_made": corrections_made
        }

    async def _run_checker(self, task: TaskDefinition, note_taker: NoteTaker) -> dict:
        """Run checker agent on task result."""
        # In production, this would invoke the checker agent
        # For now, simulate approval
        return {
            "approved": True,
            "issues": [],
            "feedback": "Looks good"
        }

    async def _run_corrector(
        self,
        task: TaskDefinition,
        issues: list,
        note_taker: NoteTaker
    ) -> dict:
        """Run corrector agent to fix issues."""
        # In production, this would invoke the corrector agent
        return {
            "fixed": True,
            "corrections": [f"Fixed: {issue}" for issue in issues]
        }

    async def _default_executor(
        self,
        task: TaskDefinition,
        note_taker: NoteTaker
    ) -> Any:
        """Default task executor (placeholder)."""
        note_taker.log_observation("Using default executor")
        note_taker.log_decision(
            decision="Execute task as-is",
            reasoning="No custom executor provided"
        )
        return {"status": "completed", "task_id": task.id}

    def _get_execution_order(self) -> list[str]:
        """Get tasks in execution order."""
        # Simple topological sort based on depends_on
        order = []
        remaining = set(self.tasks.keys())
        completed = set()

        while remaining:
            # Find tasks with dependencies met
            ready = [
                tid for tid in remaining
                if all(dep in completed for dep in self.tasks[tid].depends_on)
            ]

            if not ready:
                order.extend(sorted(remaining))
                break

            order.extend(sorted(ready))
            for tid in ready:
                completed.add(tid)
                remaining.remove(tid)

        return order

    def _dependencies_met(self, task: TaskDefinition) -> bool:
        """Check if all dependencies are completed."""
        for dep_id in task.depends_on:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True

    def _generate_handoff_document(
        self,
        request: str,
        analysis: ExpertAnalysis,
        execution_results: dict,
        quality_results: dict
    ) -> str:
        """Generate aggregated handoff document."""
        doc_parts = []

        # Header
        doc_parts.append("=" * 80)
        doc_parts.append("EXPERT PANEL WORKFLOW - AGGREGATED HANDOFF DOCUMENT")
        doc_parts.append("=" * 80)
        doc_parts.append("")

        # Request Summary
        doc_parts.append("## ORIGINAL REQUEST")
        doc_parts.append(request)
        doc_parts.append("")

        # Expert Analysis
        doc_parts.append("## EXPERT PANEL ANALYSIS")
        doc_parts.append(f"Confidence: {analysis.confidence:.2f}")
        doc_parts.append("")
        doc_parts.append("### Understanding")
        doc_parts.append(analysis.request_understanding)
        doc_parts.append("")
        doc_parts.append("### Key Requirements")
        for req in analysis.key_requirements:
            doc_parts.append(f"  - {req}")
        doc_parts.append("")
        doc_parts.append("### Approach")
        doc_parts.append(analysis.approach)
        doc_parts.append("")

        # Risks
        if analysis.risks:
            doc_parts.append("### Identified Risks")
            for risk in analysis.risks:
                doc_parts.append(f"  - [{risk['severity'].upper()}] {risk['risk']}")
                doc_parts.append(f"    Mitigation: {risk['mitigation']}")
            doc_parts.append("")

        # Task Execution
        doc_parts.append("## TASK EXECUTION SUMMARY")
        for task_id, task in self.tasks.items():
            status_emoji = {
                TaskStatus.COMPLETED: "[DONE]",
                TaskStatus.FAILED: "[FAIL]",
                TaskStatus.BLOCKED: "[BLOCK]"
            }.get(task.status, "[????]")

            doc_parts.append(f"{status_emoji} {task_id}: {task.description}")

            if task.checker_feedback:
                doc_parts.append(f"       Checker: {task.checker_feedback}")
            if task.corrections:
                doc_parts.append(f"       Corrections: {len(task.corrections)} made")
        doc_parts.append("")

        # Quality Summary
        doc_parts.append("## QUALITY GATE SUMMARY")
        doc_parts.append(f"Issues Found: {quality_results['issues_found']}")
        doc_parts.append(f"Corrections Made: {quality_results['corrections_made']}")
        doc_parts.append("")

        # Scratchpad Summaries
        doc_parts.append("## NOTE-TAKER SUMMARIES")
        for task_id, note_taker in self.note_takers.items():
            summary = note_taker.get_summary()
            doc_parts.append(f"### {task_id}")
            doc_parts.append(f"  Entries: {summary.get('total_entries', 0)}")
            doc_parts.append(f"  Decisions: {summary.get('by_type', {}).get('decision', 0)}")
            doc_parts.append(f"  Warnings: {summary.get('by_type', {}).get('warning', 0)}")
        doc_parts.append("")

        # Final Output
        doc_parts.append("## FINAL OUTPUT")
        doc_parts.append(str(execution_results.get("final_output", "N/A")))
        doc_parts.append("")

        # Footer
        doc_parts.append("=" * 80)
        doc_parts.append(f"Generated: {datetime.now().isoformat()}")
        doc_parts.append("=" * 80)

        return "\n".join(doc_parts)


# Convenience function
async def run_expert_panel_workflow(
    request: str,
    context: dict = None,
    executor: Callable = None
) -> WorkflowResult:
    """
    Run the complete expert panel workflow.

    Example:
        result = await run_expert_panel_workflow(
            request="Review and secure the authentication module",
            context={"codebase": "/path/to/code"}
        )

        print(result.handoff_document)
    """
    workflow = ExpertPanelWorkflow()
    return await workflow.execute(request, context, executor)
