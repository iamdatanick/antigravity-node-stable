"""Guardrails Module for Agentic Workflows.

Provides input/output validation, PII detection, topic filtering, and jailbreak prevention.

Usage:
    from agentic_workflows.guardrails import (
        GuardrailValidator, PIIDetector, TopicFilter, JailbreakDetector
    )

    validator = GuardrailValidator()
    result = await validator.validate(input_text)
"""

from agentic_workflows.guardrails.validator import (
    ValidationResult,
    ValidationRule,
    GuardrailValidator,
)
from agentic_workflows.guardrails.pii import (
    PIIType,
    PIIMatch,
    PIIDetector,
)
from agentic_workflows.guardrails.topics import (
    TopicCategory,
    TopicMatch,
    TopicFilter,
)
from agentic_workflows.guardrails.jailbreak import (
    JailbreakType,
    JailbreakDetection,
    JailbreakDetector,
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
