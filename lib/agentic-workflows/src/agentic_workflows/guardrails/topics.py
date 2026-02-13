"""Topic Filtering for Agentic Workflows.

Provides filtering of off-topic or restricted content.

Usage:
    from agentic_workflows.guardrails.topics import TopicFilter

    filter = TopicFilter(blocked_topics=["violence", "illegal_activities"])
    result = filter.check(text)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class TopicCategory(Enum):
    """Content topic categories."""

    # Harmful content
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    SELF_HARM = "self_harm"

    # Illegal activities
    ILLEGAL_ACTIVITIES = "illegal_activities"
    DRUG_USE = "drug_use"
    WEAPONS = "weapons"
    HACKING = "hacking"

    # Adult content
    ADULT_CONTENT = "adult_content"
    EXPLICIT_LANGUAGE = "explicit_language"

    # Misinformation
    MISINFORMATION = "misinformation"
    CONSPIRACY = "conspiracy"

    # Financial
    FINANCIAL_ADVICE = "financial_advice"
    GAMBLING = "gambling"

    # Medical/Legal
    MEDICAL_ADVICE = "medical_advice"
    LEGAL_ADVICE = "legal_advice"

    # Political
    POLITICAL_CONTENT = "political_content"
    ELECTION = "election"

    # Personal
    PERSONAL_INFO_REQUEST = "personal_info_request"

    # Off-topic
    OFF_TOPIC = "off_topic"


@dataclass
class TopicMatch:
    """A detected topic match."""

    category: TopicCategory
    confidence: float
    triggers: List[str] = field(default_factory=list)
    context: str = ""


# Topic patterns (keyword-based)
TOPIC_PATTERNS: Dict[TopicCategory, List[str]] = {
    TopicCategory.VIOLENCE: [
        r"\b(kill|murder|assassinate|attack|bomb|explode|shoot|stab)\b",
        r"\b(violence|violent|assault|hurt|harm|destroy)\b",
    ],
    TopicCategory.HATE_SPEECH: [
        r"\b(hate|racist|sexist|bigot|discriminat)\b",
        r"\b(slur|derogatory|offensive)\b",
    ],
    TopicCategory.ILLEGAL_ACTIVITIES: [
        r"\b(illegal|crime|criminal|steal|theft|fraud|launder)\b",
        r"\b(counterfeit|forge|smuggle|traffick)\b",
    ],
    TopicCategory.DRUG_USE: [
        r"\b(cocaine|heroin|meth|fentanyl|opioid|drug deal)\b",
        r"\b(get high|smoke weed|do drugs)\b",
    ],
    TopicCategory.WEAPONS: [
        r"\b(gun|firearm|rifle|pistol|ammunition|explosive)\b",
        r"\b(weapon|bomb|grenade|missile)\b",
    ],
    TopicCategory.HACKING: [
        r"\b(hack|exploit|vulnerability|malware|virus|trojan)\b",
        r"\b(crack password|bypass security|sql injection)\b",
    ],
    TopicCategory.SELF_HARM: [
        r"\b(suicide|self-harm|cut myself|kill myself)\b",
        r"\b(end my life|want to die|hurt myself)\b",
    ],
    TopicCategory.FINANCIAL_ADVICE: [
        r"\b(invest in|buy stocks|sell stocks|trading advice)\b",
        r"\b(financial advice|investment advice)\b",
    ],
    TopicCategory.MEDICAL_ADVICE: [
        r"\b(diagnose|prescription|treatment for|cure for)\b",
        r"\b(medical advice|what medication|should I take)\b",
    ],
    TopicCategory.LEGAL_ADVICE: [
        r"\b(legal advice|sue|lawsuit|court case)\b",
        r"\b(should I hire a lawyer|is this legal)\b",
    ],
    TopicCategory.PERSONAL_INFO_REQUEST: [
        r"\b(your address|your phone|your ssn|your password)\b",
        r"\b(where do you live|what's your number)\b",
    ],
}


class TopicFilter:
    """Filters content by topic.

    Example:
        filter = TopicFilter(
            blocked_topics=[TopicCategory.VIOLENCE, TopicCategory.ILLEGAL_ACTIVITIES],
        )

        result = filter.check(text)
        if result.blocked:
            print(f"Content blocked: {result.reason}")
    """

    def __init__(
        self,
        blocked_topics: Optional[List[TopicCategory]] = None,
        allowed_topics: Optional[List[TopicCategory]] = None,
        custom_patterns: Optional[Dict[TopicCategory, List[str]]] = None,
        min_confidence: float = 0.5,
    ):
        """Initialize filter.

        Args:
            blocked_topics: Topics to block.
            allowed_topics: Only allow these topics (if set).
            custom_patterns: Additional patterns.
            min_confidence: Minimum confidence threshold.
        """
        self.blocked_topics = set(blocked_topics or [])
        self.allowed_topics = set(allowed_topics) if allowed_topics else None
        self.min_confidence = min_confidence

        # Compile patterns
        self.patterns: Dict[TopicCategory, List[re.Pattern]] = {}

        for category in TopicCategory:
            patterns = TOPIC_PATTERNS.get(category, [])
            if custom_patterns and category in custom_patterns:
                patterns = patterns + custom_patterns[category]

            self.patterns[category] = [
                re.compile(p, re.IGNORECASE)
                for p in patterns
            ]

    def detect_topics(self, text: str) -> List[TopicMatch]:
        """Detect topics in text.

        Args:
            text: Text to analyze.

        Returns:
            List of TopicMatch objects.
        """
        matches = []

        for category, patterns in self.patterns.items():
            triggers = []

            for pattern in patterns:
                for match in pattern.finditer(text):
                    triggers.append(match.group())

            if triggers:
                # Calculate confidence based on number of triggers
                confidence = min(1.0, len(triggers) * 0.2 + 0.4)

                matches.append(TopicMatch(
                    category=category,
                    confidence=confidence,
                    triggers=list(set(triggers)),
                ))

        # Filter by confidence
        matches = [m for m in matches if m.confidence >= self.min_confidence]

        return matches

    def check(self, text: str) -> "TopicFilterResult":
        """Check text against topic filter.

        Args:
            text: Text to check.

        Returns:
            TopicFilterResult.
        """
        detected = self.detect_topics(text)

        # Check blocked topics
        blocked_matches = [
            m for m in detected if m.category in self.blocked_topics
        ]

        if blocked_matches:
            return TopicFilterResult(
                passed=False,
                blocked=True,
                reason=f"Contains blocked topics: {[m.category.value for m in blocked_matches]}",
                detected_topics=detected,
                blocked_topics=blocked_matches,
            )

        # Check allowed topics if set
        if self.allowed_topics:
            detected_categories = {m.category for m in detected}
            if not detected_categories.intersection(self.allowed_topics):
                return TopicFilterResult(
                    passed=False,
                    blocked=True,
                    reason="Content does not match allowed topics",
                    detected_topics=detected,
                )

        return TopicFilterResult(
            passed=True,
            blocked=False,
            detected_topics=detected,
        )

    def is_safe(self, text: str) -> bool:
        """Quick check if text is safe.

        Args:
            text: Text to check.

        Returns:
            True if safe.
        """
        return self.check(text).passed


@dataclass
class TopicFilterResult:
    """Result of topic filtering."""

    passed: bool
    blocked: bool = False
    reason: str = ""
    detected_topics: List[TopicMatch] = field(default_factory=list)
    blocked_topics: List[TopicMatch] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


# Pre-built filters
def create_safe_filter() -> TopicFilter:
    """Create a filter blocking harmful content.

    Returns:
        TopicFilter with safe defaults.
    """
    return TopicFilter(blocked_topics=[
        TopicCategory.VIOLENCE,
        TopicCategory.HATE_SPEECH,
        TopicCategory.SELF_HARM,
        TopicCategory.ILLEGAL_ACTIVITIES,
        TopicCategory.HACKING,
    ])


__all__ = [
    "TopicCategory",
    "TopicMatch",
    "TOPIC_PATTERNS",
    "TopicFilter",
    "TopicFilterResult",
    "create_safe_filter",
]
