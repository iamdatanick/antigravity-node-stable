"""Self-Debate System for validating approaches.

Enables the agent to debate with itself about the best approach,
identify potential issues, and validate reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .multi_llm import ModelTier, MultiLLMRouter


class DebateOutcome(Enum):
    """Possible outcomes of a debate."""

    CONSENSUS = "consensus"  # All positions agree
    SYNTHESIS = "synthesis"  # Combined best of all positions
    ORIGINAL_WINS = "original_wins"  # Original approach is best
    CHALLENGER_WINS = "challenger_wins"  # Alternative is better
    UNDECIDED = "undecided"  # No clear winner


@dataclass
class Position:
    """A position in the debate."""

    stance: str
    reasoning: str
    confidence: float = 0.5
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stance": self.stance,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "evidence": self.evidence,
        }


@dataclass
class DebateResult:
    """Result of a debate."""

    outcome: DebateOutcome
    winning_position: Position | None
    synthesized_approach: str | None
    key_insights: list[str]
    concerns_addressed: list[str]
    remaining_risks: list[str]
    confidence: float
    debate_transcript: list[dict]

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome.value,
            "winning_position": self.winning_position.to_dict() if self.winning_position else None,
            "synthesized_approach": self.synthesized_approach,
            "key_insights": self.key_insights,
            "concerns_addressed": self.concerns_addressed,
            "remaining_risks": self.remaining_risks,
            "confidence": self.confidence,
        }


class DebateSystem:
    """System for self-debate and approach validation.

    Features:
    - Multi-perspective analysis
    - Devil's advocate challenges
    - Synthesis of best ideas
    - Risk identification
    - Confidence calibration
    """

    def __init__(
        self,
        llm_router: MultiLLMRouter | None = None,
        max_rounds: int = 3,
    ):
        """Initialize the debate system.

        Args:
            llm_router: LLM router for generating positions
            max_rounds: Maximum debate rounds
        """
        self.llm_router = llm_router
        self.max_rounds = max_rounds

    async def debate_approach(
        self,
        task: str,
        proposed_approach: str,
        context: str | None = None,
        perspectives: list[str] | None = None,
    ) -> DebateResult:
        """Debate a proposed approach to a task.

        Args:
            task: The task to solve
            proposed_approach: The proposed solution approach
            context: Additional context
            perspectives: Specific perspectives to consider

        Returns:
            Debate result with outcome and insights
        """
        if not self.llm_router:
            # Synchronous fallback - just validate without LLM
            return self._simple_validation(task, proposed_approach)

        transcript = []

        # Round 1: Present the proposed approach
        proponent = await self._generate_position(
            role="proponent",
            task=task,
            approach=proposed_approach,
            context=context,
        )
        transcript.append({"role": "proponent", "position": proponent.to_dict()})

        # Round 2: Devil's advocate challenges
        challenger = await self._generate_challenge(
            task=task,
            approach=proposed_approach,
            proponent_position=proponent,
            context=context,
        )
        transcript.append({"role": "challenger", "position": challenger.to_dict()})

        # Round 3: Additional perspectives
        additional_positions = []
        default_perspectives = perspectives or ["practical", "risk-aware", "innovative"]

        for perspective in default_perspectives[:2]:  # Limit to 2 additional
            pos = await self._generate_perspective(
                task=task,
                approach=proposed_approach,
                perspective=perspective,
                existing_positions=[proponent, challenger],
                context=context,
            )
            additional_positions.append(pos)
            transcript.append({"role": perspective, "position": pos.to_dict()})

        # Synthesize and judge
        all_positions = [proponent, challenger] + additional_positions
        result = await self._synthesize_and_judge(
            task=task,
            original_approach=proposed_approach,
            positions=all_positions,
            transcript=transcript,
        )

        return result

    async def quick_validate(
        self,
        task: str,
        approach: str,
        criteria: list[str] | None = None,
    ) -> tuple[bool, list[str], list[str]]:
        """Quick validation of an approach.

        Args:
            task: The task
            approach: The approach
            criteria: Validation criteria

        Returns:
            Tuple of (is_valid, concerns, suggestions)
        """
        if not self.llm_router:
            return True, [], []

        prompt = f"""Quickly validate this approach for the given task.

TASK: {task}

APPROACH: {approach}

CRITERIA TO CHECK:
{chr(10).join(f"- {c}" for c in (criteria or ["correctness", "completeness", "efficiency", "safety"]))}

Respond in this format:
VALID: [yes/no]
CONCERNS:
- [concern 1]
- [concern 2]
SUGGESTIONS:
- [suggestion 1]
- [suggestion 2]
"""

        response = await self.llm_router.call(
            messages=[{"role": "user", "content": prompt}],
            force_tier=ModelTier.FAST,
        )

        # Parse response
        is_valid = "VALID: yes" in response.content.lower()
        concerns = []
        suggestions = []

        in_concerns = False
        in_suggestions = False

        for line in response.content.split("\n"):
            line = line.strip()
            if line.startswith("CONCERNS"):
                in_concerns = True
                in_suggestions = False
            elif line.startswith("SUGGESTIONS"):
                in_concerns = False
                in_suggestions = True
            elif line.startswith("- "):
                if in_concerns:
                    concerns.append(line[2:])
                elif in_suggestions:
                    suggestions.append(line[2:])

        return is_valid, concerns, suggestions

    async def identify_risks(
        self,
        task: str,
        approach: str,
        context: str | None = None,
    ) -> list[dict]:
        """Identify potential risks in an approach.

        Args:
            task: The task
            approach: The approach
            context: Additional context

        Returns:
            List of identified risks with severity and mitigation
        """
        if not self.llm_router:
            return []

        prompt = f"""Identify potential risks in this approach.

TASK: {task}

APPROACH: {approach}

{f"CONTEXT: {context}" if context else ""}

For each risk, provide:
1. Risk description
2. Severity (low/medium/high/critical)
3. Likelihood (unlikely/possible/likely/certain)
4. Potential mitigation

Format as:
RISK: [description]
SEVERITY: [level]
LIKELIHOOD: [level]
MITIGATION: [how to address]
---
"""

        response = await self.llm_router.call(
            messages=[{"role": "user", "content": prompt}],
            force_tier=ModelTier.BALANCED,
        )

        # Parse risks
        risks = []
        current_risk = {}

        for line in response.content.split("\n"):
            line = line.strip()
            if line.startswith("RISK:"):
                if current_risk:
                    risks.append(current_risk)
                current_risk = {"description": line[5:].strip()}
            elif line.startswith("SEVERITY:"):
                current_risk["severity"] = line[9:].strip().lower()
            elif line.startswith("LIKELIHOOD:"):
                current_risk["likelihood"] = line[11:].strip().lower()
            elif line.startswith("MITIGATION:"):
                current_risk["mitigation"] = line[11:].strip()
            elif line == "---" and current_risk:
                risks.append(current_risk)
                current_risk = {}

        if current_risk:
            risks.append(current_risk)

        return risks

    async def _generate_position(
        self,
        role: str,
        task: str,
        approach: str,
        context: str | None,
    ) -> Position:
        """Generate a position for the debate."""
        prompt = f"""You are the {role} in a debate about the best approach to a task.

TASK: {task}

PROPOSED APPROACH: {approach}

{f"CONTEXT: {context}" if context else ""}

Provide your position:
1. Your stance (support or oppose with nuance)
2. Key reasoning
3. Strengths of this approach
4. Weaknesses or concerns
5. Evidence or examples supporting your view
6. Your confidence level (0.0 to 1.0)

Be thorough but concise.
"""

        response = await self.llm_router.call(
            messages=[{"role": "user", "content": prompt}],
            force_tier=ModelTier.BALANCED,
        )

        # Parse into Position
        return self._parse_position(response.content, role)

    async def _generate_challenge(
        self,
        task: str,
        approach: str,
        proponent_position: Position,
        context: str | None,
    ) -> Position:
        """Generate a devil's advocate challenge."""
        prompt = f"""You are a devil's advocate challenging a proposed approach.

TASK: {task}

PROPOSED APPROACH: {approach}

PROPONENT'S ARGUMENT:
{proponent_position.reasoning}

Strengths claimed: {", ".join(proponent_position.strengths)}

{f"CONTEXT: {context}" if context else ""}

Your job is to:
1. Find flaws in the reasoning
2. Identify risks not mentioned
3. Propose alternative approaches
4. Challenge assumptions

Provide:
- Your counter-stance
- Detailed reasoning why the approach may fail or be suboptimal
- Specific weaknesses
- Alternative approaches to consider
- Your confidence in your critique (0.0 to 1.0)
"""

        response = await self.llm_router.call(
            messages=[{"role": "user", "content": prompt}],
            force_tier=ModelTier.BALANCED,
        )

        return self._parse_position(response.content, "challenger")

    async def _generate_perspective(
        self,
        task: str,
        approach: str,
        perspective: str,
        existing_positions: list[Position],
        context: str | None,
    ) -> Position:
        """Generate a perspective from a specific viewpoint."""
        positions_summary = "\n".join(
            f"- {p.stance}: {p.reasoning[:200]}..." for p in existing_positions
        )

        prompt = f"""Evaluate this approach from a {perspective} perspective.

TASK: {task}

PROPOSED APPROACH: {approach}

EXISTING POSITIONS:
{positions_summary}

{f"CONTEXT: {context}" if context else ""}

From a {perspective} perspective, evaluate:
1. What aspects of the approach align with {perspective} values?
2. What aspects conflict with {perspective} considerations?
3. What would a {perspective} modification look like?
4. Key insights from this perspective

Provide your stance, reasoning, strengths, weaknesses, and confidence.
"""

        response = await self.llm_router.call(
            messages=[{"role": "user", "content": prompt}],
            force_tier=ModelTier.FAST,
        )

        return self._parse_position(response.content, perspective)

    async def _synthesize_and_judge(
        self,
        task: str,
        original_approach: str,
        positions: list[Position],
        transcript: list[dict],
    ) -> DebateResult:
        """Synthesize positions and determine outcome."""
        positions_text = "\n\n".join(
            f"POSITION {i + 1}:\nStance: {p.stance}\nReasoning: {p.reasoning}\n"
            f"Confidence: {p.confidence}"
            for i, p in enumerate(positions)
        )

        prompt = f"""You are the judge synthesizing a debate about an approach.

TASK: {task}

ORIGINAL APPROACH: {original_approach}

POSITIONS IN DEBATE:
{positions_text}

Synthesize the debate:

1. OUTCOME: [consensus/synthesis/original_wins/challenger_wins/undecided]
2. BEST APPROACH: [The synthesized or winning approach]
3. KEY INSIGHTS: [Bullet points of main learnings]
4. CONCERNS ADDRESSED: [Which concerns from the debate are resolved]
5. REMAINING RISKS: [What risks still need attention]
6. CONFIDENCE: [Your confidence in this judgment, 0.0-1.0]

Be decisive but fair.
"""

        response = await self.llm_router.call(
            messages=[{"role": "user", "content": prompt}],
            force_tier=ModelTier.POWERFUL,  # Use powerful model for judgment
        )

        # Parse result
        return self._parse_debate_result(response.content, positions, transcript)

    def _parse_position(self, content: str, role: str) -> Position:
        """Parse LLM output into Position."""
        stance = f"{role}'s position"
        reasoning = content[:500]
        confidence = 0.5
        strengths = []
        weaknesses = []

        lines = content.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            lower = line.lower()

            if "stance" in lower or "position" in lower:
                stance = line.split(":", 1)[-1].strip() if ":" in line else stance
            elif "reasoning" in lower:
                current_section = "reasoning"
            elif "strength" in lower:
                current_section = "strengths"
            elif "weakness" in lower or "concern" in lower:
                current_section = "weaknesses"
            elif "confidence" in lower:
                try:
                    confidence = float("".join(c for c in line if c.isdigit() or c == "."))
                    if confidence > 1:
                        confidence = confidence / 100
                except ValueError:
                    pass
            elif line.startswith("- "):
                if current_section == "strengths":
                    strengths.append(line[2:])
                elif current_section == "weaknesses":
                    weaknesses.append(line[2:])

        return Position(
            stance=stance,
            reasoning=reasoning,
            confidence=confidence,
            strengths=strengths[:5],
            weaknesses=weaknesses[:5],
        )

    def _parse_debate_result(
        self,
        content: str,
        positions: list[Position],
        transcript: list[dict],
    ) -> DebateResult:
        """Parse synthesis output into DebateResult."""
        outcome = DebateOutcome.SYNTHESIS
        synthesized = None
        insights = []
        addressed = []
        risks = []
        confidence = 0.7

        lines = content.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            lower = line.lower()

            if "outcome" in lower:
                if "consensus" in lower:
                    outcome = DebateOutcome.CONSENSUS
                elif "original" in lower:
                    outcome = DebateOutcome.ORIGINAL_WINS
                elif "challenger" in lower:
                    outcome = DebateOutcome.CHALLENGER_WINS
                elif "undecided" in lower:
                    outcome = DebateOutcome.UNDECIDED
            elif "best approach" in lower or "synthesized" in lower:
                current_section = "synthesized"
                if ":" in line:
                    synthesized = line.split(":", 1)[-1].strip()
            elif "insight" in lower:
                current_section = "insights"
            elif "addressed" in lower:
                current_section = "addressed"
            elif "risk" in lower and "remain" in lower:
                current_section = "risks"
            elif "confidence" in lower:
                try:
                    confidence = float("".join(c for c in line if c.isdigit() or c == "."))
                    if confidence > 1:
                        confidence = confidence / 100
                except ValueError:
                    pass
            elif line.startswith("- "):
                item = line[2:]
                if current_section == "insights":
                    insights.append(item)
                elif current_section == "addressed":
                    addressed.append(item)
                elif current_section == "risks":
                    risks.append(item)
            elif current_section == "synthesized" and line and not line.startswith("-"):
                synthesized = (synthesized + " " + line) if synthesized else line

        # Determine winning position
        winning = None
        if outcome == DebateOutcome.ORIGINAL_WINS and positions:
            winning = positions[0]
        elif outcome == DebateOutcome.CHALLENGER_WINS and len(positions) > 1:
            winning = positions[1]

        return DebateResult(
            outcome=outcome,
            winning_position=winning,
            synthesized_approach=synthesized,
            key_insights=insights[:10],
            concerns_addressed=addressed[:10],
            remaining_risks=risks[:10],
            confidence=confidence,
            debate_transcript=transcript,
        )

    def _simple_validation(self, task: str, approach: str) -> DebateResult:
        """Simple validation without LLM."""
        return DebateResult(
            outcome=DebateOutcome.ORIGINAL_WINS,
            winning_position=Position(
                stance="Default approval",
                reasoning="No LLM available for deep validation",
                confidence=0.5,
            ),
            synthesized_approach=approach,
            key_insights=["Approach accepted without deep validation"],
            concerns_addressed=[],
            remaining_risks=["No validation performed"],
            confidence=0.5,
            debate_transcript=[],
        )
