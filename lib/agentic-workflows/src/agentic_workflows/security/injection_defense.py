"""Prompt injection defense with multi-layer detection.

Provides defense against both direct and indirect injection attacks.

Direct injection: Malicious content in user input
Indirect injection: Malicious content in tool outputs or external data
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class ThreatLevel(Enum):
    """Threat level classification."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __ge__(self, other: ThreatLevel) -> bool:
        order = [ThreatLevel.NONE, ThreatLevel.LOW, ThreatLevel.MEDIUM,
                 ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        return order.index(self) >= order.index(other)

    def __gt__(self, other: ThreatLevel) -> bool:
        order = [ThreatLevel.NONE, ThreatLevel.LOW, ThreatLevel.MEDIUM,
                 ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        return order.index(self) > order.index(other)

    def __le__(self, other: ThreatLevel) -> bool:
        return not self.__gt__(other)

    def __lt__(self, other: ThreatLevel) -> bool:
        return not self.__ge__(other)


@dataclass
class ScanResult:
    """Result of injection scan."""

    threat_level: ThreatLevel
    confidence: float
    matches: list[str] = field(default_factory=list)
    details: dict[str, any] = field(default_factory=dict)

    @property
    def is_safe(self) -> bool:
        """Check if input is considered safe."""
        return self.threat_level in (ThreatLevel.NONE, ThreatLevel.LOW)


class PromptInjectionDefense:
    """Multi-layer prompt injection detection."""

    # Known injection patterns (direct attacks)
    INJECTION_PATTERNS = [
        # Direct instruction override
        (r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", 0.9),
        (r"disregard\s+(all\s+)?(previous|prior|above)", 0.85),
        (r"forget\s+(everything|all)\s+(you|i)\s+(told|said)", 0.85),

        # Role manipulation
        (r"you\s+are\s+now\s+(?:a\s+)?(?:new|different)", 0.8),
        (r"pretend\s+(?:you\s+are|to\s+be)\s+(?:a\s+)?(?!helpful)", 0.75),
        (r"act\s+as\s+(?:if\s+)?(?:you\s+are\s+)?(?:a\s+)?(?!assistant)", 0.7),
        (r"roleplay\s+as", 0.7),
        (r"switch\s+(?:to\s+)?(?:a\s+)?(?:new\s+)?(?:persona|personality|character)", 0.85),

        # System prompt extraction
        (r"(?:what|show|reveal|display|print|output)\s+(?:is\s+)?(?:your\s+)?(?:system\s+)?prompt", 0.9),
        (r"(?:repeat|echo|say)\s+(?:back\s+)?(?:your\s+)?(?:initial\s+)?instructions?", 0.85),
        (r"beginning\s+of\s+(?:the\s+)?conversation", 0.6),

        # Delimiter/encoding attacks
        (r"```\s*system", 0.8),
        (r"\[\s*SYSTEM\s*\]", 0.85),
        (r"<\s*(?:system|admin|root)\s*>", 0.85),
        (r"={3,}\s*(?:SYSTEM|ADMIN|ROOT)", 0.8),

        # Jailbreak attempts
        (r"(?:dan|dev|evil)\s*mode", 0.9),
        (r"jailbreak", 0.95),
        (r"bypass\s+(?:your\s+)?(?:safety|security|restrictions?|filters?)", 0.9),
        (r"disable\s+(?:your\s+)?(?:safety|security|restrictions?|filters?)", 0.9),

        # Encoded content
        (r"base64\s*:\s*[A-Za-z0-9+/=]{20,}", 0.7),
        (r"\\x[0-9a-fA-F]{2}(?:\\x[0-9a-fA-F]{2}){5,}", 0.75),
        (r"&#x?[0-9a-fA-F]+;(?:&#x?[0-9a-fA-F]+;){5,}", 0.7),
    ]

    # Indirect injection patterns (tool output attacks)
    INDIRECT_INJECTION_PATTERNS = [
        # Hidden instructions in tool output
        (r"\[INST\].*?\[/INST\]", 0.95),
        (r"<\|im_start\|>system", 0.95),
        (r"<\|system\|>", 0.95),
        (r"Human:.*?Assistant:", 0.85),
        (r"<\|user\|>.*?<\|assistant\|>", 0.9),

        # Markdown/HTML hiding techniques
        (r"<!--.*?ignore.*?-->", 0.8),
        (r"<!--.*?instruction.*?-->", 0.85),
        (r"\[//\]:\s*#\s*\(.*?instruction.*?\)", 0.85),
        (r"<script[^>]*>.*?</script>", 0.9),

        # Invisible characters and zero-width content
        (r"[\u200b\u200c\u200d\ufeff]{3,}", 0.7),  # Zero-width chars
        (r"[\u2060\u2061\u2062\u2063]+", 0.65),  # Word joiners

        # Prompt leakage markers
        (r"END\s+OF\s+(?:SYSTEM\s+)?(?:PROMPT|INSTRUCTIONS)", 0.9),
        (r"BEGIN\s+(?:NEW\s+)?(?:USER\s+)?(?:INPUT|MESSAGE)", 0.85),

        # Data exfiltration attempts
        (r"(?:send|post|transmit)\s+(?:to|data\s+to)\s+(?:http|https|ftp)", 0.8),
        (r"curl\s+.*?-d\s+", 0.75),
        (r"fetch\s*\(\s*['\"]https?://", 0.7),

        # Model-specific injection markers
        (r"<\|(?:end)?(?:of)?(?:text|turn|message)\|>", 0.85),
        (r"\[/?(?:SYS|INST|TOOL)\]", 0.9),
    ]

    # Heuristic indicators
    HEURISTIC_PATTERNS = [
        # Urgency/authority claims
        (r"(?:this\s+is\s+)?(?:very\s+)?(?:urgent|important|critical|emergency)", 0.3),
        (r"(?:i\s+am\s+)?(?:an?\s+)?(?:admin|administrator|developer|owner)", 0.4),
        (r"(?:special|elevated|admin)\s+(?:access|privileges?|permissions?)", 0.5),

        # Manipulation tactics
        (r"(?:don't|do\s+not)\s+(?:worry|think)\s+about\s+(?:safety|security|rules)", 0.6),
        (r"(?:it's|this\s+is)\s+(?:okay|fine|safe|allowed)\s+(?:to|for)", 0.4),
        (r"trust\s+me", 0.3),
        (r"(?:no\s+one|nobody)\s+will\s+(?:know|find\s+out|notice)", 0.5),

        # Context confusion
        (r"in\s+(?:this|the)\s+(?:new|updated|revised)\s+(?:context|scenario)", 0.4),
        (r"(?:hypothetically|theoretically)\s+(?:speaking)?", 0.3),
        (r"for\s+(?:educational|research|testing)\s+purposes?", 0.4),
    ]

    def __init__(
        self,
        sensitivity: float = 0.7,
        custom_patterns: list[tuple[str, float]] | None = None,
        semantic_checker: Callable[[str], float] | None = None,
    ):
        """Initialize defense system.

        Args:
            sensitivity: Detection sensitivity (0.0-1.0). Higher = more sensitive.
            custom_patterns: Additional (pattern, weight) tuples to check.
            semantic_checker: Optional callback for semantic analysis.
        """
        if not 0.0 <= sensitivity <= 1.0:
            raise ValueError("Sensitivity must be between 0.0 and 1.0")

        self.sensitivity = sensitivity
        self.semantic_checker = semantic_checker

        # Compile patterns for efficiency
        self._compiled_injection = [
            (re.compile(p, re.IGNORECASE), w)
            for p, w in self.INJECTION_PATTERNS
        ]
        self._compiled_heuristic = [
            (re.compile(p, re.IGNORECASE), w)
            for p, w in self.HEURISTIC_PATTERNS
        ]

        if custom_patterns:
            self._compiled_injection.extend([
                (re.compile(p, re.IGNORECASE), w)
                for p, w in custom_patterns
            ])

    def scan(self, text: str) -> ScanResult:
        """Scan text for injection attempts.

        Args:
            text: Text to scan.

        Returns:
            ScanResult with threat level and details.
        """
        if not text or not text.strip():
            return ScanResult(
                threat_level=ThreatLevel.NONE,
                confidence=1.0,
                details={"reason": "Empty input"}
            )

        matches: list[str] = []
        scores: list[float] = []
        details: dict[str, any] = {
            "pattern_matches": [],
            "heuristic_matches": [],
            "semantic_score": 0.0,
        }

        # Layer 1: Pattern matching
        for pattern, weight in self._compiled_injection:
            if match := pattern.search(text):
                matched_text = match.group(0)
                matches.append(matched_text)
                scores.append(weight)
                details["pattern_matches"].append({
                    "pattern": pattern.pattern,
                    "matched": matched_text,
                    "weight": weight,
                })

        # Layer 2: Heuristic analysis
        heuristic_score = 0.0
        for pattern, weight in self._compiled_heuristic:
            if match := pattern.search(text):
                matched_text = match.group(0)
                heuristic_score += weight
                details["heuristic_matches"].append({
                    "pattern": pattern.pattern,
                    "matched": matched_text,
                    "weight": weight,
                })

        # Cap heuristic contribution
        heuristic_score = min(heuristic_score, 0.5)
        if heuristic_score > 0:
            scores.append(heuristic_score)

        # Layer 3: Semantic analysis (if available)
        if self.semantic_checker:
            try:
                semantic_score = self.semantic_checker(text)
                details["semantic_score"] = semantic_score
                if semantic_score > 0.3:
                    scores.append(semantic_score)
            except Exception as e:
                details["semantic_error"] = str(e)

        # Calculate final threat score
        if not scores:
            return ScanResult(
                threat_level=ThreatLevel.NONE,
                confidence=0.9,
                matches=matches,
                details=details,
            )

        # Weighted combination with sensitivity
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        combined_score = (max_score * 0.7 + avg_score * 0.3) * self.sensitivity

        # Determine threat level
        if combined_score >= 0.8:
            threat_level = ThreatLevel.CRITICAL
        elif combined_score >= 0.6:
            threat_level = ThreatLevel.HIGH
        elif combined_score >= 0.4:
            threat_level = ThreatLevel.MEDIUM
        elif combined_score >= 0.2:
            threat_level = ThreatLevel.LOW
        else:
            threat_level = ThreatLevel.NONE

        return ScanResult(
            threat_level=threat_level,
            confidence=min(combined_score + 0.2, 1.0),
            matches=matches,
            details=details,
        )

    def sanitize(self, text: str, replacement: str = "[REDACTED]") -> str:
        """Remove detected injection patterns from text.

        Args:
            text: Text to sanitize.
            replacement: Replacement string for matched patterns.

        Returns:
            Sanitized text.
        """
        result = text
        for pattern, _ in self._compiled_injection:
            result = pattern.sub(replacement, result)
        return result

    def scan_tool_output(self, text: str, tool_name: str = "") -> ScanResult:
        """Scan tool output for indirect injection attempts.

        This method specifically looks for patterns that indicate
        an attacker is trying to inject instructions through tool
        outputs (e.g., web pages, file contents, API responses).

        Args:
            text: Tool output text to scan.
            tool_name: Name of the tool that produced the output.

        Returns:
            ScanResult with threat level and details.
        """
        if not text or not text.strip():
            return ScanResult(
                threat_level=ThreatLevel.NONE,
                confidence=1.0,
                details={"reason": "Empty output", "tool": tool_name}
            )

        matches: list[str] = []
        scores: list[float] = []
        details: dict[str, any] = {
            "tool": tool_name,
            "indirect_matches": [],
            "unicode_anomalies": [],
        }

        # Layer 1: Indirect injection pattern matching
        compiled_indirect = [
            (re.compile(p, re.IGNORECASE | re.DOTALL), w)
            for p, w in self.INDIRECT_INJECTION_PATTERNS
        ]

        for pattern, weight in compiled_indirect:
            if match := pattern.search(text):
                matched_text = match.group(0)
                matches.append(matched_text)
                scores.append(weight)
                details["indirect_matches"].append({
                    "pattern": pattern.pattern,
                    "matched": matched_text[:100],  # Truncate for safety
                    "weight": weight,
                })

        # Layer 2: Unicode anomaly detection
        unicode_score = self._detect_unicode_anomalies(text)
        if unicode_score > 0:
            scores.append(unicode_score)
            details["unicode_anomalies"].append({
                "score": unicode_score,
                "description": "Suspicious unicode characters detected",
            })

        # Layer 3: Content structure analysis
        structure_score = self._analyze_content_structure(text)
        if structure_score > 0.3:
            scores.append(structure_score)
            details["structure_score"] = structure_score

        # Calculate final threat score
        if not scores:
            return ScanResult(
                threat_level=ThreatLevel.NONE,
                confidence=0.9,
                matches=matches,
                details=details,
            )

        # Weighted combination - indirect injection is more concerning
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        combined_score = (max_score * 0.8 + avg_score * 0.2) * self.sensitivity

        # Determine threat level - slightly lower thresholds for indirect
        if combined_score >= 0.75:
            threat_level = ThreatLevel.CRITICAL
        elif combined_score >= 0.55:
            threat_level = ThreatLevel.HIGH
        elif combined_score >= 0.35:
            threat_level = ThreatLevel.MEDIUM
        elif combined_score >= 0.15:
            threat_level = ThreatLevel.LOW
        else:
            threat_level = ThreatLevel.NONE

        return ScanResult(
            threat_level=threat_level,
            confidence=min(combined_score + 0.15, 1.0),
            matches=matches,
            details=details,
        )

    def _detect_unicode_anomalies(self, text: str) -> float:
        """Detect suspicious unicode usage that might hide injection.

        Args:
            text: Text to analyze.

        Returns:
            Anomaly score (0.0 to 1.0).
        """
        anomaly_count = 0

        # Check for zero-width characters
        zero_width = ['\u200b', '\u200c', '\u200d', '\ufeff', '\u2060']
        for char in zero_width:
            count = text.count(char)
            if count > 2:
                anomaly_count += count * 0.1

        # Check for right-to-left override
        rtl_chars = ['\u202e', '\u202d', '\u202c']
        for char in rtl_chars:
            if char in text:
                anomaly_count += 0.3

        # Check for homoglyph-heavy content
        homoglyph_categories = ['Mn', 'Mc', 'Me']  # Mark categories
        mark_count = sum(1 for c in text if unicodedata.category(c) in homoglyph_categories)
        if mark_count > len(text) * 0.1:
            anomaly_count += 0.2

        # Check for unusual whitespace
        unusual_space = ['\u00a0', '\u2000', '\u2001', '\u2002', '\u2003',
                        '\u2004', '\u2005', '\u2006', '\u2007', '\u2008',
                        '\u2009', '\u200a', '\u202f', '\u205f', '\u3000']
        for char in unusual_space:
            if text.count(char) > 3:
                anomaly_count += 0.1

        return min(anomaly_count, 1.0)

    def _analyze_content_structure(self, text: str) -> float:
        """Analyze content structure for injection indicators.

        Args:
            text: Text to analyze.

        Returns:
            Suspicion score (0.0 to 1.0).
        """
        score = 0.0

        # Check for role-play setup patterns
        role_patterns = [
            r"from now on",
            r"for the rest of",
            r"starting now",
            r"new instructions",
            r"updated instructions",
            r"revised instructions",
        ]
        for pattern in role_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.15

        # Check for fake conversation structure
        conversation_markers = [
            (r"^User:", re.MULTILINE),
            (r"^Assistant:", re.MULTILINE),
            (r"^Human:", re.MULTILINE),
            (r"^AI:", re.MULTILINE),
        ]
        marker_count = sum(
            1 for pattern, flags in conversation_markers
            if re.search(pattern, text, flags)
        )
        if marker_count >= 2:
            score += 0.3

        # Check for instruction-like imperatives at start of lines
        imperative_pattern = r"^(?:You must|Always|Never|Do not|Please|Remember to)"
        imperatives = len(re.findall(imperative_pattern, text, re.MULTILINE | re.IGNORECASE))
        if imperatives >= 3:
            score += 0.2

        return min(score, 1.0)

    def get_explanation(self, result: ScanResult) -> str:
        """Generate human-readable explanation of scan result.

        Args:
            result: ScanResult to explain.

        Returns:
            Explanation string.
        """
        if result.threat_level == ThreatLevel.NONE:
            return "No injection patterns detected."

        explanations = []

        if result.details.get("pattern_matches"):
            patterns = result.details["pattern_matches"]
            explanations.append(
                f"Found {len(patterns)} injection pattern(s): "
                f"{', '.join(p['matched'][:50] for p in patterns[:3])}"
            )

        if result.details.get("indirect_matches"):
            indirect = result.details["indirect_matches"]
            explanations.append(
                f"Found {len(indirect)} indirect injection pattern(s)"
            )

        if result.details.get("heuristic_matches"):
            heuristics = result.details["heuristic_matches"]
            explanations.append(
                f"Found {len(heuristics)} suspicious indicator(s)"
            )

        if result.details.get("semantic_score", 0) > 0.3:
            explanations.append(
                f"Semantic analysis flagged content "
                f"(score: {result.details['semantic_score']:.2f})"
            )

        if result.details.get("unicode_anomalies"):
            explanations.append("Unicode anomalies detected")

        return (
            f"Threat Level: {result.threat_level.value.upper()} "
            f"(confidence: {result.confidence:.0%}). "
            + " ".join(explanations)
        )
