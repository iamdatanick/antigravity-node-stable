"""Persistent Scratchpad for Agent Working Memory.

Maintains a structured working memory that persists across
interactions and can be serialized for agent handoffs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ThoughtType(Enum):
    """Types of thoughts in the scratchpad."""

    GOAL = "goal"                       # Current goal/objective
    SUBGOAL = "subgoal"                 # Decomposed sub-goal
    OBSERVATION = "observation"         # Observed fact
    HYPOTHESIS = "hypothesis"           # Working hypothesis
    QUESTION = "question"               # Open question
    CONSTRAINT = "constraint"           # Known constraint
    ASSUMPTION = "assumption"           # Working assumption
    DECISION = "decision"               # Decision made
    PLAN = "plan"                       # Planned action
    PROGRESS = "progress"               # Progress update
    BLOCKER = "blocker"                 # Blocking issue
    INSIGHT = "insight"                 # New insight
    TODO = "todo"                       # Action item
    DONE = "done"                       # Completed item
    NOTE = "note"                       # General note
    WARNING = "warning"                 # Warning/concern
    CONTEXT = "context"                 # Important context


@dataclass
class ScratchpadEntry:
    """An entry in the scratchpad."""

    id: str
    thought_type: ThoughtType
    content: str
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher = more important
    parent_id: str | None = None  # For hierarchical thoughts
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False  # For questions, blockers, todos

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "thought_type": self.thought_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "parent_id": self.parent_id,
            "tags": self.tags,
            "metadata": self.metadata,
            "resolved": self.resolved,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ScratchpadEntry:
        return cls(
            id=data["id"],
            thought_type=ThoughtType(data["thought_type"]),
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            priority=data.get("priority", 0),
            parent_id=data.get("parent_id"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            resolved=data.get("resolved", False),
        )


class Scratchpad:
    """Persistent working memory for agents.

    Features:
    - Hierarchical thought organization
    - Priority-based ordering
    - Tag-based filtering
    - Automatic summarization
    - Serialization for handoffs
    - Token-aware formatting
    """

    def __init__(self, max_entries: int = 100):
        """Initialize the scratchpad.

        Args:
            max_entries: Maximum entries to retain
        """
        self.max_entries = max_entries
        self._entries: dict[str, ScratchpadEntry] = {}
        self._counter = 0
        self._session_start = time.time()

    def _generate_id(self) -> str:
        """Generate unique entry ID."""
        self._counter += 1
        return f"sp_{self._counter}_{int(time.time())}"

    def add(
        self,
        thought_type: ThoughtType,
        content: str,
        priority: int = 0,
        parent_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> ScratchpadEntry:
        """Add an entry to the scratchpad.

        Args:
            thought_type: Type of thought
            content: Thought content
            priority: Priority level
            parent_id: Parent entry ID for hierarchy
            tags: Tags for filtering
            metadata: Additional metadata

        Returns:
            Created entry
        """
        entry_id = self._generate_id()

        entry = ScratchpadEntry(
            id=entry_id,
            thought_type=thought_type,
            content=content,
            priority=priority,
            parent_id=parent_id,
            tags=tags or [],
            metadata=metadata or {},
        )

        self._entries[entry_id] = entry

        # Prune if needed
        if len(self._entries) > self.max_entries:
            self._prune()

        return entry

    def add_goal(self, goal: str, priority: int = 10) -> ScratchpadEntry:
        """Add a goal."""
        return self.add(ThoughtType.GOAL, goal, priority=priority, tags=["active"])

    def add_subgoal(self, subgoal: str, parent_goal_id: str) -> ScratchpadEntry:
        """Add a subgoal linked to parent goal."""
        return self.add(
            ThoughtType.SUBGOAL,
            subgoal,
            priority=8,
            parent_id=parent_goal_id,
            tags=["active"],
        )

    def add_observation(self, observation: str, tags: list[str] | None = None) -> ScratchpadEntry:
        """Add an observation."""
        return self.add(ThoughtType.OBSERVATION, observation, priority=3, tags=tags)

    def add_hypothesis(self, hypothesis: str, confidence: float = 0.5) -> ScratchpadEntry:
        """Add a hypothesis."""
        return self.add(
            ThoughtType.HYPOTHESIS,
            hypothesis,
            priority=5,
            metadata={"confidence": confidence},
        )

    def add_question(self, question: str, priority: int = 6) -> ScratchpadEntry:
        """Add an open question."""
        return self.add(ThoughtType.QUESTION, question, priority=priority, tags=["open"])

    def add_decision(self, decision: str, reasoning: str = "") -> ScratchpadEntry:
        """Add a decision with reasoning."""
        return self.add(
            ThoughtType.DECISION,
            decision,
            priority=7,
            metadata={"reasoning": reasoning},
        )

    def add_plan(self, plan: str, steps: list[str] | None = None) -> ScratchpadEntry:
        """Add a plan with optional steps."""
        entry = self.add(ThoughtType.PLAN, plan, priority=8, tags=["active"])
        if steps:
            for i, step in enumerate(steps):
                self.add(
                    ThoughtType.TODO,
                    step,
                    parent_id=entry.id,
                    metadata={"step_number": i + 1},
                )
        return entry

    def add_blocker(self, blocker: str, severity: str = "medium") -> ScratchpadEntry:
        """Add a blocking issue."""
        return self.add(
            ThoughtType.BLOCKER,
            blocker,
            priority=9,
            tags=["unresolved"],
            metadata={"severity": severity},
        )

    def add_progress(self, progress: str, percentage: float | None = None) -> ScratchpadEntry:
        """Add a progress update."""
        return self.add(
            ThoughtType.PROGRESS,
            progress,
            metadata={"percentage": percentage} if percentage else {},
        )

    def add_todo(self, todo: str, priority: int = 5) -> ScratchpadEntry:
        """Add a todo item."""
        return self.add(ThoughtType.TODO, todo, priority=priority, tags=["pending"])

    def complete_todo(self, entry_id: str, result: str = "") -> None:
        """Mark a todo as complete."""
        if entry_id in self._entries:
            entry = self._entries[entry_id]
            entry.resolved = True
            entry.thought_type = ThoughtType.DONE
            if "pending" in entry.tags:
                entry.tags.remove("pending")
            entry.tags.append("completed")
            if result:
                entry.metadata["result"] = result

    def resolve_question(self, entry_id: str, answer: str) -> None:
        """Resolve an open question."""
        if entry_id in self._entries:
            entry = self._entries[entry_id]
            entry.resolved = True
            if "open" in entry.tags:
                entry.tags.remove("open")
            entry.tags.append("answered")
            entry.metadata["answer"] = answer

    def resolve_blocker(self, entry_id: str, resolution: str) -> None:
        """Resolve a blocker."""
        if entry_id in self._entries:
            entry = self._entries[entry_id]
            entry.resolved = True
            if "unresolved" in entry.tags:
                entry.tags.remove("unresolved")
            entry.tags.append("resolved")
            entry.metadata["resolution"] = resolution

    def get(self, entry_id: str) -> ScratchpadEntry | None:
        """Get an entry by ID."""
        return self._entries.get(entry_id)

    def get_by_type(self, thought_type: ThoughtType) -> list[ScratchpadEntry]:
        """Get all entries of a specific type."""
        return [e for e in self._entries.values() if e.thought_type == thought_type]

    def get_by_tag(self, tag: str) -> list[ScratchpadEntry]:
        """Get all entries with a specific tag."""
        return [e for e in self._entries.values() if tag in e.tags]

    def get_unresolved(self) -> list[ScratchpadEntry]:
        """Get all unresolved entries (questions, blockers, todos)."""
        return [
            e for e in self._entries.values()
            if not e.resolved and e.thought_type in (
                ThoughtType.QUESTION,
                ThoughtType.BLOCKER,
                ThoughtType.TODO,
            )
        ]

    def get_active_goals(self) -> list[ScratchpadEntry]:
        """Get active goals and subgoals."""
        return [
            e for e in self._entries.values()
            if e.thought_type in (ThoughtType.GOAL, ThoughtType.SUBGOAL)
            and "active" in e.tags
        ]

    def get_children(self, parent_id: str) -> list[ScratchpadEntry]:
        """Get child entries of a parent."""
        return [e for e in self._entries.values() if e.parent_id == parent_id]

    def get_recent(self, count: int = 10) -> list[ScratchpadEntry]:
        """Get most recent entries."""
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: e.timestamp,
            reverse=True,
        )
        return sorted_entries[:count]

    def get_high_priority(self, min_priority: int = 7) -> list[ScratchpadEntry]:
        """Get high priority entries."""
        return [e for e in self._entries.values() if e.priority >= min_priority]

    def _prune(self) -> None:
        """Remove low-priority old entries to stay under limit."""
        # Score entries
        scored = []
        now = time.time()
        for entry in self._entries.values():
            age_hours = (now - entry.timestamp) / 3600
            score = entry.priority - (age_hours * 0.1)
            if not entry.resolved:
                score += 2  # Boost unresolved items
            if entry.thought_type in (ThoughtType.GOAL, ThoughtType.BLOCKER):
                score += 3  # Boost important types
            scored.append((entry.id, score))

        # Sort by score
        scored.sort(key=lambda x: x[1])

        # Remove bottom 20%
        to_remove = int(len(scored) * 0.2)
        for entry_id, _ in scored[:to_remove]:
            del self._entries[entry_id]

    def format_for_prompt(
        self,
        max_tokens: int = 1500,
        include_types: list[ThoughtType] | None = None,
        exclude_resolved: bool = False,
    ) -> str:
        """Format scratchpad for inclusion in prompt.

        Args:
            max_tokens: Approximate token limit
            include_types: Only include specific types
            exclude_resolved: Exclude resolved items

        Returns:
            Formatted scratchpad string
        """
        lines = ["<scratchpad>"]

        # Filter entries
        entries = list(self._entries.values())
        if include_types:
            entries = [e for e in entries if e.thought_type in include_types]
        if exclude_resolved:
            entries = [e for e in entries if not e.resolved]

        # Sort by priority and timestamp
        entries.sort(key=lambda e: (-e.priority, -e.timestamp))

        # Group by type
        by_type: dict[ThoughtType, list[ScratchpadEntry]] = {}
        for entry in entries:
            if entry.thought_type not in by_type:
                by_type[entry.thought_type] = []
            by_type[entry.thought_type].append(entry)

        # Format sections
        type_order = [
            ThoughtType.GOAL,
            ThoughtType.SUBGOAL,
            ThoughtType.BLOCKER,
            ThoughtType.QUESTION,
            ThoughtType.PLAN,
            ThoughtType.TODO,
            ThoughtType.DECISION,
            ThoughtType.HYPOTHESIS,
            ThoughtType.OBSERVATION,
            ThoughtType.PROGRESS,
            ThoughtType.INSIGHT,
        ]

        char_count = 0
        max_chars = max_tokens * 4  # Rough estimate

        for thought_type in type_order:
            if thought_type not in by_type:
                continue

            type_entries = by_type[thought_type][:5]  # Limit per type
            section = f"\n## {thought_type.value.upper()}"

            for entry in type_entries:
                status = "✓" if entry.resolved else "○"
                line = f"\n{status} [{entry.id}] {entry.content}"
                if entry.metadata:
                    meta = ", ".join(f"{k}={v}" for k, v in entry.metadata.items() if k not in ("step_number",))
                    if meta:
                        line += f" ({meta})"

                if char_count + len(section) + len(line) > max_chars:
                    break

                if section:
                    lines.append(section)
                    section = ""
                lines.append(line)
                char_count += len(line)

        lines.append("\n</scratchpad>")
        return "\n".join(lines)

    def get_summary(self) -> dict:
        """Get a summary of scratchpad state."""
        by_type = {}
        for entry in self._entries.values():
            type_name = entry.thought_type.value
            if type_name not in by_type:
                by_type[type_name] = {"total": 0, "resolved": 0}
            by_type[type_name]["total"] += 1
            if entry.resolved:
                by_type[type_name]["resolved"] += 1

        return {
            "total_entries": len(self._entries),
            "by_type": by_type,
            "unresolved_count": len(self.get_unresolved()),
            "active_goals": len(self.get_active_goals()),
            "session_duration_minutes": (time.time() - self._session_start) / 60,
        }

    def export(self) -> dict:
        """Export scratchpad for serialization."""
        return {
            "entries": [e.to_dict() for e in self._entries.values()],
            "counter": self._counter,
            "session_start": self._session_start,
        }

    def import_data(self, data: dict) -> None:
        """Import scratchpad from serialized data."""
        self._entries.clear()
        for entry_data in data.get("entries", []):
            entry = ScratchpadEntry.from_dict(entry_data)
            self._entries[entry.id] = entry
        self._counter = data.get("counter", len(self._entries))

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()
        self._counter = 0

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        summary = self.get_summary()
        return f"Scratchpad({summary['total_entries']} entries, {summary['unresolved_count']} unresolved)"
