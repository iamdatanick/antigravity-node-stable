"""Guardrails Module for Agentic Workflows.

Provides input/output validation, PII detection, topic filtering, and jailbreak prevention.

Usage:
    from agentic_workflows.guardrails import (
        GuardrailValidator, PIIDetector, TopicFilter, JailbreakDetector
    )

    validator = GuardrailValidator()
    result = await validator.validate(input_text)
"""

from agentic_workflows.guardrails.jailbreak import (
    JailbreakDetection,
    JailbreakDetector,
    JailbreakType,
)
from agentic_workflows.guardrails.pii import (
    PIIDetector,
    PIIMatch,
    PIIType,
)
from agentic_workflows.guardrails.topics import (
    TopicCategory,
    TopicFilter,
    TopicMatch,
)
from agentic_workflows.guardrails.validator import (
    GuardrailValidator,
    ValidationResult,
    ValidationRule,
)

__all__ = [
    # Validator
    "ValidationResult",
    "ValidationRule",
    "GuardrailValidator",
    # PII
    "PIIType",
    "PIIMatch",
    "PIIDetector",
    # Topics
    "TopicCategory",
    "TopicMatch",
    "TopicFilter",
    # Jailbreak
    "JailbreakType",
    "JailbreakDetection",
    "JailbreakDetector",
]
