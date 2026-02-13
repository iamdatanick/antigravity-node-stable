"""Artifact System for Agent Handoffs.

Provides structured serialization of agent state, context, and work products
for seamless handoffs between agents or LLM sessions.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .scratchpad import Scratchpad, ThoughtType
from .context_graph import LearningContextGraph


class ArtifactType(Enum):
    """Types of artifacts that can be generated."""

    # State artifacts
    SCRATCHPAD = "scratchpad"           # Working memory state
    CONTEXT_GRAPH = "context_graph"     # Learning/context state
    FULL_STATE = "full_state"           # Complete agent state

    # Work product artifacts
    CODE = "code"                       # Generated code
    DOCUMENT = "document"               # Generated document
    ANALYSIS = "analysis"               # Analysis results
    PLAN = "plan"                       # Execution plan
    DECISION = "decision"               # Decision record

    # Handoff artifacts
    TASK_HANDOFF = "task_handoff"       # Task transfer to another agent
    CHECKPOINT = "checkpoint"           # Resumable checkpoint
    SUMMARY = "summary"                 # Condensed summary for context


@dataclass
class ArtifactMetadata:
    """Metadata for an artifact."""

    artifact_id: str
    artifact_type: ArtifactType
    created_at: float
    created_by: str  # Agent or session ID
    version: str = "1.0"
    checksum: str = ""
    compressed: bool = False
    encryption_key_id: str | None = None
    tags: list[str] = field(default_factory=list)
    parent_artifact_id: str | None = None
    ttl_seconds: int | None = None  # Time to live

    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "version": self.version,
            "checksum": self.checksum,
            "compressed": self.compressed,
            "encryption_key_id": self.encryption_key_id,
            "tags": self.tags,
            "parent_artifact_id": self.parent_artifact_id,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ArtifactMetadata:
        return cls(
            artifact_id=data["artifact_id"],
            artifact_type=ArtifactType(data["artifact_type"]),
            created_at=data["created_at"],
            created_by=data["created_by"],
            version=data.get("version", "1.0"),
            checksum=data.get("checksum", ""),
            compressed=data.get("compressed", False),
            encryption_key_id=data.get("encryption_key_id"),
            tags=data.get("tags", []),
            parent_artifact_id=data.get("parent_artifact_id"),
            ttl_seconds=data.get("ttl_seconds"),
        )


@dataclass
class AgentArtifact:
    """A complete artifact for agent handoffs.

    Artifacts encapsulate all the state and context needed for
    another agent or LLM session to continue work seamlessly.
    """

    metadata: ArtifactMetadata
    content: dict[str, Any]

    # Optional structured sections
    task_description: str | None = None
    current_state: str | None = None
    next_steps: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    key_decisions: list[dict] = field(default_factory=list)
    relevant_context: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "content": self.content,
            "task_description": self.task_description,
            "current_state": self.current_state,
            "next_steps": self.next_steps,
            "open_questions": self.open_questions,
            "key_decisions": self.key_decisions,
            "relevant_context": self.relevant_context,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AgentArtifact:
        return cls(
            metadata=ArtifactMetadata.from_dict(data["metadata"]),
            content=data["content"],
            task_description=data.get("task_description"),
            current_state=data.get("current_state"),
            next_steps=data.get("next_steps", []),
            open_questions=data.get("open_questions", []),
            key_decisions=data.get("key_decisions", []),
            relevant_context=data.get("relevant_context", []),
            warnings=data.get("warnings", []),
        )

    def format_for_prompt(self, max_tokens: int = 2000) -> str:
        """Format artifact for inclusion in an LLM prompt.

        Creates a structured, readable representation optimized
        for LLM comprehension.
        """
        sections = []

        sections.append("<artifact>")
        sections.append(f"Type: {self.metadata.artifact_type.value}")
        sections.append(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.metadata.created_at))}")
        sections.append(f"By: {self.metadata.created_by}")

        if self.task_description:
            sections.append(f"\n## Task\n{self.task_description}")

        if self.current_state:
            sections.append(f"\n## Current State\n{self.current_state}")

        if self.next_steps:
            sections.append("\n## Next Steps")
            for i, step in enumerate(self.next_steps[:10], 1):
                sections.append(f"{i}. {step}")

        if self.open_questions:
            sections.append("\n## Open Questions")
            for q in self.open_questions[:5]:
                sections.append(f"- {q}")

        if self.key_decisions:
            sections.append("\n## Key Decisions Made")
            for d in self.key_decisions[:5]:
                decision = d.get("decision", "")
                reasoning = d.get("reasoning", "")
                sections.append(f"- {decision}")
                if reasoning:
                    sections.append(f"  Reasoning: {reasoning}")

        if self.relevant_context:
            sections.append("\n## Relevant Context")
            for ctx in self.relevant_context[:5]:
                sections.append(f"- {ctx}")

        if self.warnings:
            sections.append("\n## Warnings")
            for w in self.warnings:
                sections.append(f"! {w}")

        sections.append("</artifact>")

        result = "\n".join(sections)

        # Truncate if needed
        max_chars = max_tokens * 4
        if len(result) > max_chars:
            result = result[:max_chars - 100] + "\n...[truncated]</artifact>"

        return result


class ArtifactBuilder:
    """Builder for creating artifacts."""

    def __init__(self, agent_id: str):
        """Initialize builder.

        Args:
            agent_id: ID of the agent creating artifacts
        """
        self.agent_id = agent_id
        self._counter = 0

    def _generate_id(self) -> str:
        """Generate unique artifact ID."""
        self._counter += 1
        timestamp = int(time.time() * 1000)
        return f"art_{self.agent_id}_{timestamp}_{self._counter}"

    def _compute_checksum(self, content: dict) -> str:
        """Compute checksum of content."""
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def create_scratchpad_artifact(
        self,
        scratchpad: Scratchpad,
        task_description: str | None = None,
    ) -> AgentArtifact:
        """Create artifact from scratchpad state."""
        content = scratchpad.export()
        summary = scratchpad.get_summary()

        # Extract next steps from todos
        next_steps = []
        open_questions = []

        for entry in scratchpad.get_unresolved():
            if entry.thought_type == ThoughtType.TODO:
                next_steps.append(entry.content)
            elif entry.thought_type == ThoughtType.QUESTION:
                open_questions.append(entry.content)

        metadata = ArtifactMetadata(
            artifact_id=self._generate_id(),
            artifact_type=ArtifactType.SCRATCHPAD,
            created_at=time.time(),
            created_by=self.agent_id,
            checksum=self._compute_checksum(content),
            tags=["working_memory", "state"],
        )

        return AgentArtifact(
            metadata=metadata,
            content=content,
            task_description=task_description,
            current_state=f"Scratchpad with {summary['total_entries']} entries, {summary['unresolved_count']} unresolved",
            next_steps=next_steps[:10],
            open_questions=open_questions[:5],
        )

    def create_context_graph_artifact(
        self,
        context_graph: LearningContextGraph,
        task_description: str | None = None,
    ) -> AgentArtifact:
        """Create artifact from context graph state."""
        content = context_graph.export()

        # Extract insights as relevant context
        relevant_context = []
        if task_description:
            insights = context_graph.get_relevant_insights(task_description, top_k=5)
            relevant_context = [
                f"[{i.insight_type}] {i.content} (confidence: {i.confidence:.2f})"
                for i in insights
            ]

        metadata = ArtifactMetadata(
            artifact_id=self._generate_id(),
            artifact_type=ArtifactType.CONTEXT_GRAPH,
            created_at=time.time(),
            created_by=self.agent_id,
            checksum=self._compute_checksum(content),
            tags=["learning", "context", "state"],
        )

        return AgentArtifact(
            metadata=metadata,
            content=content,
            task_description=task_description,
            current_state=f"Context graph with {len(content.get('nodes', []))} nodes",
            relevant_context=relevant_context,
        )

    def create_task_handoff(
        self,
        task_description: str,
        current_state: str,
        scratchpad: Scratchpad | None = None,
        context_graph: LearningContextGraph | None = None,
        next_steps: list[str] | None = None,
        open_questions: list[str] | None = None,
        key_decisions: list[dict] | None = None,
        relevant_context: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> AgentArtifact:
        """Create a task handoff artifact for transferring work to another agent."""
        content = {
            "task": task_description,
            "state": current_state,
            "handoff_time": time.time(),
        }

        # Include state snapshots if provided
        if scratchpad:
            content["scratchpad"] = scratchpad.export()
        if context_graph:
            content["context_graph"] = context_graph.export()

        # Auto-extract from scratchpad if available
        if scratchpad and not next_steps:
            next_steps = [
                e.content for e in scratchpad.get_unresolved()
                if e.thought_type == ThoughtType.TODO
            ][:10]

        if scratchpad and not open_questions:
            open_questions = [
                e.content for e in scratchpad.get_unresolved()
                if e.thought_type == ThoughtType.QUESTION
            ][:5]

        metadata = ArtifactMetadata(
            artifact_id=self._generate_id(),
            artifact_type=ArtifactType.TASK_HANDOFF,
            created_at=time.time(),
            created_by=self.agent_id,
            checksum=self._compute_checksum(content),
            tags=["handoff", "task_transfer"],
        )

        return AgentArtifact(
            metadata=metadata,
            content=content,
            task_description=task_description,
            current_state=current_state,
            next_steps=next_steps or [],
            open_questions=open_questions or [],
            key_decisions=key_decisions or [],
            relevant_context=relevant_context or [],
            warnings=warnings or [],
        )

    def create_checkpoint(
        self,
        task_description: str,
        current_state: str,
        full_state: dict[str, Any],
        resumption_instructions: str,
    ) -> AgentArtifact:
        """Create a checkpoint artifact for resuming later."""
        content = {
            "full_state": full_state,
            "resumption_instructions": resumption_instructions,
            "checkpoint_time": time.time(),
        }

        metadata = ArtifactMetadata(
            artifact_id=self._generate_id(),
            artifact_type=ArtifactType.CHECKPOINT,
            created_at=time.time(),
            created_by=self.agent_id,
            checksum=self._compute_checksum(content),
            compressed=True,  # Checkpoints should be compressed
            tags=["checkpoint", "resumable"],
        )

        return AgentArtifact(
            metadata=metadata,
            content=content,
            task_description=task_description,
            current_state=current_state,
            next_steps=[resumption_instructions],
        )

    def create_analysis_artifact(
        self,
        task_description: str,
        findings: list[str],
        recommendations: list[str],
        data: dict[str, Any] | None = None,
        confidence: float = 0.8,
    ) -> AgentArtifact:
        """Create an analysis result artifact."""
        content = {
            "findings": findings,
            "recommendations": recommendations,
            "confidence": confidence,
            "data": data or {},
        }

        metadata = ArtifactMetadata(
            artifact_id=self._generate_id(),
            artifact_type=ArtifactType.ANALYSIS,
            created_at=time.time(),
            created_by=self.agent_id,
            checksum=self._compute_checksum(content),
            tags=["analysis", "work_product"],
        )

        return AgentArtifact(
            metadata=metadata,
            content=content,
            task_description=task_description,
            current_state=f"Analysis complete with {len(findings)} findings",
            relevant_context=findings[:5],
            next_steps=recommendations[:5],
        )

    def create_decision_artifact(
        self,
        decision: str,
        reasoning: str,
        alternatives_considered: list[str],
        risks: list[str] | None = None,
        confidence: float = 0.8,
    ) -> AgentArtifact:
        """Create a decision record artifact."""
        content = {
            "decision": decision,
            "reasoning": reasoning,
            "alternatives": alternatives_considered,
            "risks": risks or [],
            "confidence": confidence,
        }

        metadata = ArtifactMetadata(
            artifact_id=self._generate_id(),
            artifact_type=ArtifactType.DECISION,
            created_at=time.time(),
            created_by=self.agent_id,
            checksum=self._compute_checksum(content),
            tags=["decision", "work_product"],
        )

        return AgentArtifact(
            metadata=metadata,
            content=content,
            task_description=f"Decision: {decision}",
            current_state="Decision made",
            key_decisions=[{
                "decision": decision,
                "reasoning": reasoning,
                "confidence": confidence,
            }],
            warnings=risks or [],
        )


def serialize_for_handoff(
    artifact: AgentArtifact,
    compress: bool = True,
) -> str:
    """Serialize an artifact for handoff.

    Args:
        artifact: The artifact to serialize
        compress: Whether to compress the output

    Returns:
        Base64-encoded serialized artifact
    """
    data = artifact.to_dict()
    json_str = json.dumps(data, separators=(',', ':'))

    if compress:
        compressed = gzip.compress(json_str.encode('utf-8'))
        return base64.b64encode(compressed).decode('ascii')
    else:
        return base64.b64encode(json_str.encode('utf-8')).decode('ascii')


def deserialize_from_handoff(
    serialized: str,
    compressed: bool = True,
) -> AgentArtifact:
    """Deserialize an artifact from handoff.

    Args:
        serialized: Base64-encoded serialized artifact
        compressed: Whether the data is compressed

    Returns:
        Deserialized artifact
    """
    raw_bytes = base64.b64decode(serialized)

    if compressed:
        json_str = gzip.decompress(raw_bytes).decode('utf-8')
    else:
        json_str = raw_bytes.decode('utf-8')

    data = json.loads(json_str)
    return AgentArtifact.from_dict(data)


def create_handoff_prompt(
    artifact: AgentArtifact,
    receiving_agent_role: str = "assistant",
) -> str:
    """Create a complete prompt for handing off to another agent.

    Args:
        artifact: The handoff artifact
        receiving_agent_role: Role description for receiving agent

    Returns:
        Complete prompt for the receiving agent
    """
    prompt_parts = [
        f"You are a {receiving_agent_role} continuing work from a previous session.",
        "",
        "## Handoff Context",
        artifact.format_for_prompt(),
        "",
        "## Instructions",
        "1. Review the task description and current state",
        "2. Note any open questions - address them if possible",
        "3. Follow the next steps outlined above",
        "4. Be aware of the warnings provided",
        "",
        "Continue the work from where it was left off.",
    ]

    return "\n".join(prompt_parts)
