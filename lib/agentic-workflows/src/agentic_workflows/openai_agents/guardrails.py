"""Guardrails for agent input/output validation.

This module provides guardrail classes for validating agent inputs and outputs,
integrating with agentic_workflows security module.

Key classes:
- InputGuardrail: Validates user input before agent processing
- OutputGuardrail: Validates agent output before returning
- Built-in guardrails: ContentFilter, PIIDetector, InjectionDefense

Reference: https://github.com/openai/openai-agents-python
"""

from __future__ import annotations

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class GuardrailAction(Enum):
    """Action to take when guardrail fails."""

    BLOCK = "block"  # Block the request/response
    WARN = "warn"  # Log warning but continue
    SANITIZE = "sanitize"  # Remove/replace problematic content
    ESCALATE = "escalate"  # Escalate to human review


@dataclass
class GuardrailResult:
    """Result of guardrail validation.

    Attributes:
        passed: Whether validation passed.
        action: Recommended action if failed.
        message: Human-readable message.
        violations: List of specific violations.
        sanitized_content: Content after sanitization (if applicable).
        confidence: Confidence score (0-1).
    """

    passed: bool
    action: GuardrailAction = GuardrailAction.BLOCK
    message: str = ""
    violations: list[str] = field(default_factory=list)
    sanitized_content: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseGuardrail(ABC):
    """Base class for all guardrails.

    Guardrails validate content and return a GuardrailResult indicating
    whether the content is acceptable.
    """

    def __init__(
        self,
        name: str,
        action: GuardrailAction = GuardrailAction.BLOCK,
        enabled: bool = True,
    ):
        """Initialize guardrail.

        Args:
            name: Guardrail identifier.
            action: Default action on failure.
            enabled: Whether guardrail is active.
        """
        self.name = name
        self.action = action
        self.enabled = enabled

    @abstractmethod
    def validate(self, content: str) -> GuardrailResult:
        """Validate content.

        Args:
            content: Content to validate.

        Returns:
            Validation result.
        """
        pass

    def __call__(self, content: str) -> GuardrailResult:
        """Allow calling guardrail directly."""
        if not self.enabled:
            return GuardrailResult(passed=True)
        return self.validate(content)


class InputGuardrail(BaseGuardrail):
    """Guardrail for validating user input.

    Input guardrails run before the agent processes user messages.
    They can block malicious input, sanitize content, or log warnings.

    Example:
        guardrail = InputGuardrail(
            name="length_check",
            validator=lambda x: len(x) < 10000,
            message="Input too long",
        )
        result = guardrail.validate(user_input)
    """

    def __init__(
        self,
        name: str,
        validator: Callable[[str], bool | tuple[bool, str]] | None = None,
        action: GuardrailAction = GuardrailAction.BLOCK,
        message: str = "",
        enabled: bool = True,
    ):
        """Initialize input guardrail.

        Args:
            name: Guardrail name.
            validator: Validation function.
            action: Action on failure.
            message: Custom failure message.
            enabled: Whether active.
        """
        super().__init__(name, action, enabled)
        self.validator = validator
        self.message = message

    def validate(self, content: str) -> GuardrailResult:
        """Validate input content.

        Args:
            content: User input.

        Returns:
            Validation result.
        """
        if self.validator is None:
            return GuardrailResult(passed=True)

        try:
            result = self.validator(content)

            if isinstance(result, bool):
                if result:
                    return GuardrailResult(passed=True)
                else:
                    return GuardrailResult(
                        passed=False,
                        action=self.action,
                        message=self.message or f"Input guardrail '{self.name}' failed",
                        violations=[self.name],
                    )
            else:
                # Tuple of (passed, message)
                passed, msg = result
                return GuardrailResult(
                    passed=passed,
                    action=self.action if not passed else GuardrailAction.BLOCK,
                    message=msg if not passed else "",
                    violations=[self.name] if not passed else [],
                )

        except Exception as e:
            logger.error(f"Input guardrail '{self.name}' error: {e}")
            return GuardrailResult(
                passed=False,
                action=GuardrailAction.BLOCK,
                message=f"Guardrail error: {str(e)}",
                violations=[self.name],
            )


class OutputGuardrail(BaseGuardrail):
    """Guardrail for validating agent output.

    Output guardrails run before returning agent responses.
    They can filter sensitive information, block harmful content,
    or sanitize outputs.

    Example:
        guardrail = OutputGuardrail(
            name="no_code",
            validator=lambda x: "```" not in x,
            message="Code blocks not allowed",
        )
    """

    def __init__(
        self,
        name: str,
        validator: Callable[[str], bool | tuple[bool, str]] | None = None,
        action: GuardrailAction = GuardrailAction.BLOCK,
        message: str = "",
        sanitizer: Callable[[str], str] | None = None,
        enabled: bool = True,
    ):
        """Initialize output guardrail.

        Args:
            name: Guardrail name.
            validator: Validation function.
            action: Action on failure.
            message: Custom failure message.
            sanitizer: Optional function to sanitize content.
            enabled: Whether active.
        """
        super().__init__(name, action, enabled)
        self.validator = validator
        self.message = message
        self.sanitizer = sanitizer

    def validate(self, content: str) -> GuardrailResult:
        """Validate output content.

        Args:
            content: Agent output.

        Returns:
            Validation result.
        """
        if self.validator is None:
            return GuardrailResult(passed=True)

        try:
            result = self.validator(content)

            if isinstance(result, bool):
                passed = result
                msg = ""
            else:
                passed, msg = result

            if passed:
                return GuardrailResult(passed=True)

            # Try sanitization if available
            sanitized = None
            if self.sanitizer and self.action == GuardrailAction.SANITIZE:
                sanitized = self.sanitizer(content)

            return GuardrailResult(
                passed=False,
                action=self.action,
                message=msg or self.message or f"Output guardrail '{self.name}' failed",
                violations=[self.name],
                sanitized_content=sanitized,
            )

        except Exception as e:
            logger.error(f"Output guardrail '{self.name}' error: {e}")
            return GuardrailResult(
                passed=False,
                action=GuardrailAction.BLOCK,
                message=f"Guardrail error: {str(e)}",
                violations=[self.name],
            )


# ============================================================================
# Built-in Guardrails
# ============================================================================


class ContentFilterGuardrail(BaseGuardrail):
    """Filter content based on blocked patterns.

    Blocks content containing specified patterns or keywords.
    """

    # Default blocked patterns
    DEFAULT_PATTERNS = [
        r"\b(kill|murder|assassinate)\s+(yourself|himself|herself|themselves)\b",
        r"\b(how\s+to\s+)?(make|create|build)\s+(a\s+)?(bomb|explosive|weapon)\b",
        r"\b(hack|crack|break\s+into)\s+",
    ]

    def __init__(
        self,
        name: str = "content_filter",
        blocked_patterns: list[str] | None = None,
        blocked_keywords: list[str] | None = None,
        case_sensitive: bool = False,
        action: GuardrailAction = GuardrailAction.BLOCK,
        enabled: bool = True,
    ):
        """Initialize content filter.

        Args:
            name: Guardrail name.
            blocked_patterns: Regex patterns to block.
            blocked_keywords: Keywords to block.
            case_sensitive: Whether matching is case-sensitive.
            action: Action on match.
            enabled: Whether active.
        """
        super().__init__(name, action, enabled)
        self.case_sensitive = case_sensitive

        # Compile patterns
        flags = 0 if case_sensitive else re.IGNORECASE
        patterns = blocked_patterns or self.DEFAULT_PATTERNS
        self.patterns = [re.compile(p, flags) for p in patterns]

        # Keywords
        self.keywords = blocked_keywords or []
        if not case_sensitive:
            self.keywords = [k.lower() for k in self.keywords]

    def validate(self, content: str) -> GuardrailResult:
        """Check content against filters.

        Args:
            content: Content to check.

        Returns:
            Validation result.
        """
        violations = []
        check_content = content if self.case_sensitive else content.lower()

        # Check patterns
        for pattern in self.patterns:
            if match := pattern.search(content):
                violations.append(f"Pattern match: {match.group()[:50]}")

        # Check keywords
        for keyword in self.keywords:
            if keyword in check_content:
                violations.append(f"Keyword: {keyword}")

        if violations:
            return GuardrailResult(
                passed=False,
                action=self.action,
                message="Content contains blocked patterns or keywords",
                violations=violations,
            )

        return GuardrailResult(passed=True)


class PIIDetectorGuardrail(BaseGuardrail):
    """Detect and optionally redact personally identifiable information.

    Detects common PII patterns like emails, phone numbers, SSNs, etc.
    """

    # PII patterns
    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone_us": r"\b(?:\+1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
        "ssn": r"\b[0-9]{3}[-\s]?[0-9]{2}[-\s]?[0-9]{4}\b",
        "credit_card": r"\b(?:[0-9]{4}[-\s]?){3}[0-9]{4}\b",
        "ip_address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
        "date_of_birth": r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])[/-](?:19|20)?[0-9]{2}\b",
    }

    def __init__(
        self,
        name: str = "pii_detector",
        detect_types: list[str] | None = None,
        redact: bool = False,
        redaction_pattern: str = "[REDACTED:{type}]",
        action: GuardrailAction = GuardrailAction.WARN,
        enabled: bool = True,
    ):
        """Initialize PII detector.

        Args:
            name: Guardrail name.
            detect_types: PII types to detect (None = all).
            redact: Whether to redact detected PII.
            redaction_pattern: Pattern for redaction.
            action: Action on detection.
            enabled: Whether active.
        """
        super().__init__(name, action, enabled)
        self.redact = redact
        self.redaction_pattern = redaction_pattern

        # Compile patterns
        types_to_detect = detect_types or list(self.PII_PATTERNS.keys())
        self.patterns = {
            pii_type: re.compile(pattern)
            for pii_type, pattern in self.PII_PATTERNS.items()
            if pii_type in types_to_detect
        }

    def validate(self, content: str) -> GuardrailResult:
        """Detect PII in content.

        Args:
            content: Content to check.

        Returns:
            Validation result with detected PII types.
        """
        violations = []
        sanitized = content

        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(content)
            if matches:
                violations.append(f"{pii_type}: {len(matches)} found")

                if self.redact:
                    redaction = self.redaction_pattern.format(type=pii_type.upper())
                    sanitized = pattern.sub(redaction, sanitized)

        if violations:
            return GuardrailResult(
                passed=False,
                action=self.action,
                message="PII detected in content",
                violations=violations,
                sanitized_content=sanitized if self.redact else None,
                metadata={"pii_types": [v.split(":")[0] for v in violations]},
            )

        return GuardrailResult(passed=True)


class InjectionDefenseGuardrail(BaseGuardrail):
    """Defense against prompt injection attacks.

    Integrates with agentic_workflows injection_defense module.
    """

    def __init__(
        self,
        name: str = "injection_defense",
        sensitivity: float = 0.7,
        block_threshold: str = "medium",  # "low", "medium", "high", "critical"
        action: GuardrailAction = GuardrailAction.BLOCK,
        enabled: bool = True,
    ):
        """Initialize injection defense.

        Args:
            name: Guardrail name.
            sensitivity: Detection sensitivity (0-1).
            block_threshold: Minimum threat level to block.
            action: Action on detection.
            enabled: Whether active.
        """
        super().__init__(name, action, enabled)
        self.sensitivity = sensitivity
        self.block_threshold = block_threshold

        # Try to import from security module
        try:
            from agentic_workflows.security.injection_defense import (
                PromptInjectionDefense,
                ThreatLevel,
            )
            self._defense = PromptInjectionDefense(sensitivity=sensitivity)
            self._threat_levels = {
                "low": ThreatLevel.LOW,
                "medium": ThreatLevel.MEDIUM,
                "high": ThreatLevel.HIGH,
                "critical": ThreatLevel.CRITICAL,
            }
        except ImportError:
            logger.warning("injection_defense module not available, using fallback")
            self._defense = None
            self._threat_levels = {}

    def validate(self, content: str) -> GuardrailResult:
        """Check for injection attempts.

        Args:
            content: Content to check.

        Returns:
            Validation result.
        """
        if self._defense is None:
            # Fallback: basic pattern check
            return self._fallback_check(content)

        from agentic_workflows.security.injection_defense import ThreatLevel

        result = self._defense.scan(content)

        threshold = self._threat_levels.get(self.block_threshold, ThreatLevel.MEDIUM)

        if result.threat_level >= threshold:
            return GuardrailResult(
                passed=False,
                action=self.action,
                message=self._defense.get_explanation(result),
                violations=result.matches,
                confidence=result.confidence,
                metadata={
                    "threat_level": result.threat_level.value,
                    "details": result.details,
                },
            )

        return GuardrailResult(
            passed=True,
            confidence=1.0 - result.confidence,
            metadata={"threat_level": result.threat_level.value},
        )

    def _fallback_check(self, content: str) -> GuardrailResult:
        """Fallback injection check when security module unavailable.

        Args:
            content: Content to check.

        Returns:
            Validation result.
        """
        # Basic patterns to check
        dangerous_patterns = [
            r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?",
            r"disregard\s+(?:all\s+)?(?:previous|prior)",
            r"you\s+are\s+now\s+a?",
            r"pretend\s+(?:you\s+are|to\s+be)",
            r"(?:reveal|show)\s+(?:your\s+)?(?:system\s+)?prompt",
            r"jailbreak",
            r"bypass\s+(?:your\s+)?(?:safety|security)",
        ]

        violations = []
        for pattern in dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                violations.append(pattern)

        if violations:
            return GuardrailResult(
                passed=False,
                action=self.action,
                message="Potential injection attempt detected",
                violations=violations[:5],  # Limit violations shown
                confidence=0.7,
            )

        return GuardrailResult(passed=True)


class LengthGuardrail(BaseGuardrail):
    """Validate content length.

    Ensures content is within specified length bounds.
    """

    def __init__(
        self,
        name: str = "length_check",
        min_length: int = 0,
        max_length: int = 100000,
        action: GuardrailAction = GuardrailAction.BLOCK,
        enabled: bool = True,
    ):
        """Initialize length guardrail.

        Args:
            name: Guardrail name.
            min_length: Minimum allowed length.
            max_length: Maximum allowed length.
            action: Action on violation.
            enabled: Whether active.
        """
        super().__init__(name, action, enabled)
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, content: str) -> GuardrailResult:
        """Check content length.

        Args:
            content: Content to check.

        Returns:
            Validation result.
        """
        length = len(content)

        if length < self.min_length:
            return GuardrailResult(
                passed=False,
                action=self.action,
                message=f"Content too short: {length} < {self.min_length}",
                violations=["min_length"],
                metadata={"length": length},
            )

        if length > self.max_length:
            return GuardrailResult(
                passed=False,
                action=self.action,
                message=f"Content too long: {length} > {self.max_length}",
                violations=["max_length"],
                metadata={"length": length},
                sanitized_content=content[:self.max_length] if self.action == GuardrailAction.SANITIZE else None,
            )

        return GuardrailResult(passed=True, metadata={"length": length})


class ToxicityGuardrail(BaseGuardrail):
    """Detect toxic or harmful language.

    Uses pattern matching and optional ML-based detection.
    """

    # Toxic patterns (simplified - production would use ML model)
    TOXIC_PATTERNS = [
        r"\b(hate|despise)\s+(you|them|all)\b",
        r"\b(stupid|idiot|moron|dumb)\b",
        r"\b(shut\s+up|go\s+away|leave\s+me\s+alone)\b",
    ]

    def __init__(
        self,
        name: str = "toxicity",
        threshold: float = 0.5,
        patterns: list[str] | None = None,
        action: GuardrailAction = GuardrailAction.WARN,
        enabled: bool = True,
    ):
        """Initialize toxicity guardrail.

        Args:
            name: Guardrail name.
            threshold: Toxicity threshold (0-1).
            patterns: Custom toxic patterns.
            action: Action on detection.
            enabled: Whether active.
        """
        super().__init__(name, action, enabled)
        self.threshold = threshold

        patterns_to_use = patterns or self.TOXIC_PATTERNS
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns_to_use]

    def validate(self, content: str) -> GuardrailResult:
        """Check for toxicity.

        Args:
            content: Content to check.

        Returns:
            Validation result.
        """
        matches = []
        for pattern in self.patterns:
            if match := pattern.search(content):
                matches.append(match.group())

        # Simple scoring based on matches
        score = min(len(matches) * 0.2, 1.0)

        if score >= self.threshold:
            return GuardrailResult(
                passed=False,
                action=self.action,
                message="Potentially toxic content detected",
                violations=matches,
                confidence=score,
                metadata={"toxicity_score": score},
            )

        return GuardrailResult(
            passed=True,
            confidence=1.0 - score,
            metadata={"toxicity_score": score},
        )


# ============================================================================
# Guardrail Chain
# ============================================================================


class GuardrailChain:
    """Chain multiple guardrails together.

    Runs guardrails in sequence and aggregates results.

    Example:
        chain = GuardrailChain([
            LengthGuardrail(max_length=10000),
            ContentFilterGuardrail(),
            PIIDetectorGuardrail(redact=True),
        ])
        result = chain.validate(content)
    """

    def __init__(
        self,
        guardrails: list[BaseGuardrail],
        stop_on_failure: bool = True,
    ):
        """Initialize guardrail chain.

        Args:
            guardrails: List of guardrails to run.
            stop_on_failure: Stop on first failure.
        """
        self.guardrails = guardrails
        self.stop_on_failure = stop_on_failure

    def validate(self, content: str) -> GuardrailResult:
        """Run all guardrails on content.

        Args:
            content: Content to validate.

        Returns:
            Aggregated result.
        """
        all_violations = []
        all_metadata = {}
        sanitized = content
        worst_action = GuardrailAction.WARN
        passed = True

        for guardrail in self.guardrails:
            if not guardrail.enabled:
                continue

            result = guardrail.validate(sanitized)

            if not result.passed:
                passed = False
                all_violations.extend(result.violations)
                all_metadata[guardrail.name] = result.metadata

                # Track worst action
                if result.action == GuardrailAction.BLOCK:
                    worst_action = GuardrailAction.BLOCK
                elif result.action == GuardrailAction.ESCALATE and worst_action != GuardrailAction.BLOCK:
                    worst_action = GuardrailAction.ESCALATE

                # Apply sanitization if available
                if result.sanitized_content:
                    sanitized = result.sanitized_content

                if self.stop_on_failure and result.action == GuardrailAction.BLOCK:
                    return GuardrailResult(
                        passed=False,
                        action=GuardrailAction.BLOCK,
                        message=result.message,
                        violations=all_violations,
                        sanitized_content=sanitized if sanitized != content else None,
                        metadata=all_metadata,
                    )

        return GuardrailResult(
            passed=passed,
            action=worst_action if not passed else GuardrailAction.BLOCK,
            message="Guardrail chain completed" if passed else "One or more guardrails failed",
            violations=all_violations,
            sanitized_content=sanitized if sanitized != content else None,
            metadata=all_metadata,
        )

    def add(self, guardrail: BaseGuardrail) -> "GuardrailChain":
        """Add a guardrail to the chain.

        Args:
            guardrail: Guardrail to add.

        Returns:
            Self for chaining.
        """
        self.guardrails.append(guardrail)
        return self


# ============================================================================
# Decorator for custom guardrails
# ============================================================================


def guardrail(
    name: str | None = None,
    guardrail_type: str = "input",
    action: GuardrailAction = GuardrailAction.BLOCK,
) -> Callable[[Callable[[str], bool | tuple[bool, str]]], InputGuardrail | OutputGuardrail]:
    """Decorator to create a guardrail from a validation function.

    Args:
        name: Guardrail name (defaults to function name).
        guardrail_type: "input" or "output".
        action: Action on failure.

    Returns:
        Guardrail instance.

    Example:
        @guardrail(name="no_urls", guardrail_type="output")
        def check_no_urls(content: str) -> bool:
            return "http" not in content
    """

    def decorator(
        func: Callable[[str], bool | tuple[bool, str]]
    ) -> InputGuardrail | OutputGuardrail:
        guardrail_name = name or func.__name__

        if guardrail_type == "input":
            return InputGuardrail(
                name=guardrail_name,
                validator=func,
                action=action,
                message=func.__doc__ or "",
            )
        else:
            return OutputGuardrail(
                name=guardrail_name,
                validator=func,
                action=action,
                message=func.__doc__ or "",
            )

    return decorator
