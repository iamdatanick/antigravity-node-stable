"""Intelligent Agentic Orchestrator.

The main orchestrator that ties together all intelligent components:
- Multi-LLM routing with hierarchical review
- Learning context graph
- Persistent scratchpad
- Self-debate system
- Artifact generation for handoffs
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .artifact import (
    AgentArtifact,
    ArtifactBuilder,
    create_handoff_prompt,
)
from .context_graph import LearningContextGraph, LearningInsight
from .debate import DebateOutcome, DebateResult, DebateSystem
from .multi_llm import LLMResponse, ModelTier, MultiLLMRouter
from .scratchpad import Scratchpad, ThoughtType


class TaskComplexity(Enum):
    """Complexity levels for task routing."""

    TRIVIAL = "trivial"  # Simple lookup or formatting
    SIMPLE = "simple"  # Single-step task
    MODERATE = "moderate"  # Multi-step with some reasoning
    COMPLEX = "complex"  # Requires planning and iteration
    EXPERT = "expert"  # Requires deep expertise and validation


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    # LLM settings
    anthropic_api_key: str | None = None
    default_model_tier: ModelTier = ModelTier.BALANCED
    enable_hierarchical_review: bool = True
    review_threshold: float = 0.7  # Review responses below this confidence

    # Debate settings
    enable_debate: bool = True
    debate_complexity_threshold: TaskComplexity = TaskComplexity.COMPLEX
    max_debate_rounds: int = 3

    # Context settings
    enable_learning: bool = True
    max_context_nodes: int = 1000
    context_retrieval_count: int = 5

    # Scratchpad settings
    max_scratchpad_entries: int = 100
    persist_scratchpad: bool = True

    # Artifact settings
    auto_checkpoint: bool = True
    checkpoint_interval_steps: int = 10

    # Execution settings
    max_iterations: int = 50
    timeout_seconds: float = 300.0


@dataclass
class ExecutionStep:
    """A single step in task execution."""

    step_id: str
    step_type: str
    content: str
    result: Any = None
    success: bool = True
    error: str | None = None
    timestamp: float = field(default_factory=time.time)
    model_used: str | None = None
    tokens_used: int = 0
    duration_ms: float = 0


@dataclass
class ExecutionResult:
    """Result of task execution."""

    task_id: str
    success: bool
    final_output: str
    steps: list[ExecutionStep]
    total_tokens: int
    total_duration_ms: float
    model_tiers_used: list[str]
    debate_result: DebateResult | None = None
    insights_applied: list[LearningInsight] = field(default_factory=list)
    artifact: AgentArtifact | None = None


class AgenticOrchestrator:
    """Intelligent orchestrator for agentic workflows.

    Provides:
    - Smart task routing based on complexity
    - Multi-LLM orchestration with hierarchical review
    - Learning from past executions
    - Self-debate for complex decisions
    - Persistent working memory
    - Artifact generation for handoffs
    """

    def __init__(self, config: OrchestratorConfig | None = None):
        """Initialize the orchestrator.

        Args:
            config: Orchestrator configuration
        """
        self.config = config or OrchestratorConfig()
        self.session_id = str(uuid.uuid4())[:8]

        # Initialize components
        self.llm_router = (
            MultiLLMRouter(
                anthropic_api_key=self.config.anthropic_api_key,
                enable_hierarchical_review=self.config.enable_hierarchical_review,
            )
            if self.config.anthropic_api_key
            else None
        )

        self.context_graph = LearningContextGraph(
            max_nodes=self.config.max_context_nodes,
        )

        self.scratchpad = Scratchpad(
            max_entries=self.config.max_scratchpad_entries,
        )

        self.debate_system = DebateSystem(
            llm_router=self.llm_router,
            max_rounds=self.config.max_debate_rounds,
        )

        self.artifact_builder = ArtifactBuilder(
            agent_id=f"orchestrator_{self.session_id}",
        )

        # Execution state
        self._current_task_id: str | None = None
        self._step_count = 0
        self._tool_registry: dict[str, Callable] = {}

    def register_tool(self, name: str, func: Callable) -> None:
        """Register a tool for the orchestrator to use.

        Args:
            name: Tool name
            func: Tool function (sync or async)
        """
        self._tool_registry[name] = func

    async def execute_task(
        self,
        task: str,
        context: str | None = None,
        tools: list[dict] | None = None,
        force_complexity: TaskComplexity | None = None,
    ) -> ExecutionResult:
        """Execute a task with intelligent orchestration.

        Args:
            task: Task description
            context: Additional context
            tools: Available tools
            force_complexity: Override complexity detection

        Returns:
            Execution result with all details
        """
        task_id = f"task_{self.session_id}_{int(time.time())}"
        self._current_task_id = task_id
        self._step_count = 0
        start_time = time.time()
        steps: list[ExecutionStep] = []
        model_tiers_used: set[str] = set()
        total_tokens = 0

        # Record task start in scratchpad
        goal_entry = self.scratchpad.add_goal(task)
        self.scratchpad.add(
            ThoughtType.CONTEXT,
            context or "No additional context provided",
        )

        # Record in context graph
        self.context_graph.record_task_start(
            task_id=task_id,
            task_content=task,
            approach="intelligent_orchestration",
            context={"user_context": context} if context else None,
        )

        try:
            # Step 1: Analyze complexity
            complexity = force_complexity or await self._analyze_complexity(task, context)
            self.scratchpad.add(
                ThoughtType.OBSERVATION,
                f"Task complexity assessed as: {complexity.value}",
            )
            steps.append(
                ExecutionStep(
                    step_id=f"{task_id}_complexity",
                    step_type="complexity_analysis",
                    content=f"Complexity: {complexity.value}",
                )
            )

            # Step 2: Retrieve relevant insights
            insights = []
            if self.config.enable_learning:
                insights = self.context_graph.get_relevant_insights(
                    task,
                    top_k=self.config.context_retrieval_count,
                )
                if insights:
                    self.scratchpad.add(
                        ThoughtType.INSIGHT,
                        f"Retrieved {len(insights)} relevant insights from past experience",
                    )
                    for insight in insights[:3]:
                        self.scratchpad.add(
                            ThoughtType.OBSERVATION,
                            f"Past insight: {insight.content}",
                            metadata={"confidence": insight.confidence},
                        )

            # Step 3: Generate approach
            approach = await self._generate_approach(task, context, complexity, insights)
            self.scratchpad.add_plan(
                f"Approach: {approach['summary']}",
                steps=approach.get("steps", []),
            )
            steps.append(
                ExecutionStep(
                    step_id=f"{task_id}_approach",
                    step_type="approach_generation",
                    content=approach["summary"],
                )
            )

            # Step 4: Debate approach if complex enough
            debate_result = None
            if (
                self.config.enable_debate
                and complexity.value >= self.config.debate_complexity_threshold.value
            ):
                debate_result = await self._debate_approach(
                    task,
                    approach["summary"],
                    context,
                )
                self.scratchpad.add(
                    ThoughtType.DECISION,
                    f"Debate outcome: {debate_result.outcome.value}",
                    metadata={"confidence": debate_result.confidence},
                )
                steps.append(
                    ExecutionStep(
                        step_id=f"{task_id}_debate",
                        step_type="self_debate",
                        content=f"Outcome: {debate_result.outcome.value}",
                        result=debate_result.to_dict(),
                    )
                )

                # Adjust approach if needed
                if debate_result.outcome == DebateOutcome.SYNTHESIS:
                    approach["summary"] = debate_result.synthesized_approach or approach["summary"]
                    self.scratchpad.add(
                        ThoughtType.PLAN,
                        f"Revised approach: {approach['summary']}",
                    )

            # Step 5: Execute with appropriate model tier
            model_tier = self._select_model_tier(complexity)
            model_tiers_used.add(model_tier.value)

            response = await self._execute_with_llm(
                task=task,
                approach=approach,
                context=context,
                insights=insights,
                tools=tools,
                model_tier=model_tier,
            )

            total_tokens += response.total_tokens
            model_tiers_used.add(response.model)

            steps.append(
                ExecutionStep(
                    step_id=f"{task_id}_execution",
                    step_type="llm_execution",
                    content=response.content[:500],
                    model_used=response.model,
                    tokens_used=response.total_tokens,
                )
            )

            # Step 6: Hierarchical review if needed
            if (
                self.config.enable_hierarchical_review
                and model_tier != ModelTier.POWERFUL
                and complexity.value >= TaskComplexity.COMPLEX.value
            ):
                reviewed = await self._hierarchical_review(
                    task,
                    response.content,
                )
                if reviewed:
                    response = reviewed
                    model_tiers_used.add(reviewed.model)
                    total_tokens += reviewed.total_tokens
                    steps.append(
                        ExecutionStep(
                            step_id=f"{task_id}_review",
                            step_type="hierarchical_review",
                            content="Response reviewed by higher-tier model",
                            model_used=reviewed.model,
                        )
                    )

            # Mark goal complete
            self.scratchpad.complete_todo(goal_entry.id, "Task completed successfully")

            # Record success in context graph
            self.context_graph.record_task_outcome(
                task_id=task_id,
                success=True,
                outcome_content=response.content[:500],
                quality_score=debate_result.confidence if debate_result else 0.8,
            )

            # Create checkpoint artifact if configured
            artifact = None
            if self.config.auto_checkpoint:
                artifact = self._create_checkpoint_artifact(
                    task_id=task_id,
                    task=task,
                    result=response.content,
                    steps=steps,
                )

            duration_ms = (time.time() - start_time) * 1000

            return ExecutionResult(
                task_id=task_id,
                success=True,
                final_output=response.content,
                steps=steps,
                total_tokens=total_tokens,
                total_duration_ms=duration_ms,
                model_tiers_used=list(model_tiers_used),
                debate_result=debate_result,
                insights_applied=insights,
                artifact=artifact,
            )

        except Exception as e:
            # Record failure
            self.scratchpad.add_blocker(f"Task failed: {str(e)}", severity="high")
            self.context_graph.record_task_outcome(
                task_id=task_id,
                success=False,
                outcome_content=str(e),
                quality_score=0.0,
            )

            duration_ms = (time.time() - start_time) * 1000

            return ExecutionResult(
                task_id=task_id,
                success=False,
                final_output=f"Error: {str(e)}",
                steps=steps,
                total_tokens=total_tokens,
                total_duration_ms=duration_ms,
                model_tiers_used=list(model_tiers_used),
            )

    async def _analyze_complexity(
        self,
        task: str,
        context: str | None,
    ) -> TaskComplexity:
        """Analyze task complexity for routing decisions."""
        if not self.llm_router:
            return TaskComplexity.MODERATE

        prompt = f"""Analyze the complexity of this task.

TASK: {task}

{f"CONTEXT: {context}" if context else ""}

Rate complexity as one of:
- TRIVIAL: Simple lookup or formatting
- SIMPLE: Single-step task
- MODERATE: Multi-step with some reasoning
- COMPLEX: Requires planning and iteration
- EXPERT: Requires deep expertise and validation

Respond with just the complexity level (e.g., "MODERATE")."""

        response = await self.llm_router.call(
            messages=[{"role": "user", "content": prompt}],
            force_tier=ModelTier.FAST,
            max_tokens=50,
        )

        complexity_str = response.content.strip().upper()
        for level in TaskComplexity:
            if level.value.upper() in complexity_str:
                return level

        return TaskComplexity.MODERATE

    async def _generate_approach(
        self,
        task: str,
        context: str | None,
        complexity: TaskComplexity,
        insights: list[LearningInsight],
    ) -> dict:
        """Generate an approach for the task."""
        if not self.llm_router:
            return {"summary": task, "steps": []}

        insights_text = ""
        if insights:
            insights_text = "\n\nRELEVANT PAST INSIGHTS:\n" + "\n".join(
                f"- {i.content} (confidence: {i.confidence:.2f})" for i in insights[:5]
            )

        prompt = f"""Generate an approach for this task.

TASK: {task}
COMPLEXITY: {complexity.value}

{f"CONTEXT: {context}" if context else ""}
{insights_text}

Provide:
1. A brief summary of the approach (1-2 sentences)
2. Key steps to execute (3-7 steps)

Format:
SUMMARY: [approach summary]
STEPS:
1. [step 1]
2. [step 2]
..."""

        response = await self.llm_router.call(
            messages=[{"role": "user", "content": prompt}],
            force_tier=ModelTier.FAST,
            max_tokens=500,
        )

        # Parse response
        summary = ""
        steps = []

        for line in response.content.split("\n"):
            line = line.strip()
            if line.startswith("SUMMARY:"):
                summary = line[8:].strip()
            elif line and line[0].isdigit() and "." in line:
                step_text = line.split(".", 1)[-1].strip()
                if step_text:
                    steps.append(step_text)

        return {
            "summary": summary or task,
            "steps": steps,
        }

    async def _debate_approach(
        self,
        task: str,
        approach: str,
        context: str | None,
    ) -> DebateResult:
        """Run self-debate on the approach."""
        return await self.debate_system.debate_approach(
            task=task,
            proposed_approach=approach,
            context=context,
        )

    def _select_model_tier(self, complexity: TaskComplexity) -> ModelTier:
        """Select appropriate model tier based on complexity."""
        tier_map = {
            TaskComplexity.TRIVIAL: ModelTier.FAST,
            TaskComplexity.SIMPLE: ModelTier.FAST,
            TaskComplexity.MODERATE: ModelTier.BALANCED,
            TaskComplexity.COMPLEX: ModelTier.BALANCED,
            TaskComplexity.EXPERT: ModelTier.POWERFUL,
        }
        return tier_map.get(complexity, self.config.default_model_tier)

    async def _execute_with_llm(
        self,
        task: str,
        approach: dict,
        context: str | None,
        insights: list[LearningInsight],
        tools: list[dict] | None,
        model_tier: ModelTier,
    ) -> LLMResponse:
        """Execute task with LLM."""
        if not self.llm_router:
            raise ValueError("LLM router not configured")

        # Build system prompt with scratchpad
        scratchpad_str = self.scratchpad.format_for_prompt(max_tokens=1000)

        system = f"""You are an intelligent agent executing a task.

{scratchpad_str}

APPROACH TO FOLLOW:
{approach["summary"]}

STEPS:
{chr(10).join(f"{i + 1}. {s}" for i, s in enumerate(approach.get("steps", [])))}

Execute the task thoroughly and provide a complete response."""

        # Build user message
        user_content = f"TASK: {task}"
        if context:
            user_content += f"\n\nCONTEXT: {context}"

        messages = [{"role": "user", "content": user_content}]

        # Use tools if available
        if tools and self._tool_registry:
            return await self.llm_router.call_with_tools(
                messages=messages,
                tools=tools,
                tool_executor=self._execute_tool,
                system=system,
                force_tier=model_tier,
            )
        else:
            return await self.llm_router.call(
                messages=messages,
                system=system,
                force_tier=model_tier,
            )

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a registered tool."""
        if tool_name not in self._tool_registry:
            return f"Error: Unknown tool '{tool_name}'"

        func = self._tool_registry[tool_name]

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(**tool_input)
            else:
                result = func(**tool_input)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    async def _hierarchical_review(
        self,
        task: str,
        response: str,
    ) -> LLMResponse | None:
        """Have a higher-tier model review the response."""
        if not self.llm_router:
            return None

        prompt = f"""Review this response for correctness and completeness.

ORIGINAL TASK: {task}

RESPONSE TO REVIEW:
{response}

If the response is correct and complete, respond with:
APPROVED: [brief confirmation]

If corrections are needed, respond with:
CORRECTED: [corrected response]

Be concise."""

        reviewed = await self.llm_router.call(
            messages=[{"role": "user", "content": prompt}],
            force_tier=ModelTier.POWERFUL,
            max_tokens=2000,
        )

        # Check if corrections were made
        if "CORRECTED:" in reviewed.content:
            corrected_content = reviewed.content.split("CORRECTED:", 1)[-1].strip()
            reviewed.content = corrected_content
            return reviewed

        return None

    def _create_checkpoint_artifact(
        self,
        task_id: str,
        task: str,
        result: str,
        steps: list[ExecutionStep],
    ) -> AgentArtifact:
        """Create a checkpoint artifact."""
        return self.artifact_builder.create_task_handoff(
            task_description=task,
            current_state=f"Completed with {len(steps)} steps",
            scratchpad=self.scratchpad,
            context_graph=self.context_graph,
            next_steps=[],
            key_decisions=[
                {
                    "decision": s.content,
                    "step_type": s.step_type,
                }
                for s in steps
                if s.step_type in ("approach_generation", "self_debate", "hierarchical_review")
            ],
        )

    async def stream_task(
        self,
        task: str,
        context: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream task execution with real-time output.

        Args:
            task: Task description
            context: Additional context

        Yields:
            Streaming content chunks
        """
        if not self.llm_router:
            yield "Error: LLM router not configured"
            return

        # Analyze complexity first
        complexity = await self._analyze_complexity(task, context)
        yield f"[Complexity: {complexity.value}]\n\n"

        # Get insights
        insights = self.context_graph.get_relevant_insights(task, top_k=3)
        if insights:
            yield "[Applying past insights...]\n\n"

        # Generate approach
        approach = await self._generate_approach(task, context, complexity, insights)
        yield f"[Approach: {approach['summary']}]\n\n"

        # Stream execution
        system = f"""Execute this task following this approach: {approach["summary"]}"""

        async for chunk in self.llm_router.stream(
            messages=[{"role": "user", "content": f"TASK: {task}\n\n{context or ''}"}],
            system=system,
            task_description=task,
            force_tier=self._select_model_tier(complexity),
        ):
            yield chunk

    def create_handoff(
        self,
        receiving_agent: str = "assistant",
    ) -> tuple[AgentArtifact, str]:
        """Create a handoff package for another agent.

        Args:
            receiving_agent: Role description for receiving agent

        Returns:
            Tuple of (artifact, handoff_prompt)
        """
        # Get current state summary
        summary = self.scratchpad.get_summary()
        unresolved = self.scratchpad.get_unresolved()

        # Create handoff artifact
        artifact = self.artifact_builder.create_task_handoff(
            task_description="Continuing previous work session",
            current_state=f"Session with {summary['total_entries']} entries, {summary['unresolved_count']} pending items",
            scratchpad=self.scratchpad,
            context_graph=self.context_graph,
            next_steps=[e.content for e in unresolved if e.thought_type == ThoughtType.TODO][:5],
            open_questions=[
                e.content for e in unresolved if e.thought_type == ThoughtType.QUESTION
            ][:5],
            warnings=[e.content for e in self.scratchpad.get_by_type(ThoughtType.WARNING)][:3],
        )

        # Create handoff prompt
        prompt = create_handoff_prompt(artifact, receiving_agent)

        return artifact, prompt

    def export_state(self) -> dict:
        """Export complete orchestrator state for persistence."""
        return {
            "session_id": self.session_id,
            "scratchpad": self.scratchpad.export(),
            "context_graph": self.context_graph.export(),
            "config": {
                "default_model_tier": self.config.default_model_tier.value,
                "enable_debate": self.config.enable_debate,
                "enable_learning": self.config.enable_learning,
            },
        }

    def import_state(self, state: dict) -> None:
        """Import orchestrator state from persistence."""
        if "scratchpad" in state:
            self.scratchpad.import_data(state["scratchpad"])
        if "context_graph" in state:
            self.context_graph.import_data(state["context_graph"])
        if "session_id" in state:
            self.session_id = state["session_id"]

    def get_status(self) -> dict:
        """Get current orchestrator status."""
        scratchpad_summary = self.scratchpad.get_summary()
        context_summary = self.context_graph.get_statistics()

        return {
            "session_id": self.session_id,
            "llm_available": self.llm_router is not None,
            "scratchpad": scratchpad_summary,
            "context_graph": context_summary,
            "tools_registered": list(self._tool_registry.keys()),
            "current_task": self._current_task_id,
            "steps_executed": self._step_count,
        }


# Convenience function for quick setup
def create_orchestrator(
    anthropic_api_key: str,
    enable_debate: bool = True,
    enable_learning: bool = True,
) -> AgenticOrchestrator:
    """Create an orchestrator with sensible defaults.

    Args:
        anthropic_api_key: Anthropic API key
        enable_debate: Enable self-debate for complex tasks
        enable_learning: Enable learning from past tasks

    Returns:
        Configured orchestrator
    """
    config = OrchestratorConfig(
        anthropic_api_key=anthropic_api_key,
        enable_debate=enable_debate,
        enable_learning=enable_learning,
    )
    return AgenticOrchestrator(config)
