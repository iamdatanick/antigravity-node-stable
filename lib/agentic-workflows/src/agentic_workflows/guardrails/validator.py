"""Guardrail Validator for Agentic Workflows.

Provides a unified validation framework for input/output validation.

Usage:
    from agentic_workflows.guardrails.validator import GuardrailValidator

    validator = GuardrailValidator()
    validator.add_rule(max_length_rule)
    result = await validator.validate(text)
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ValidationAction(Enum):
    """Action to take on validation failure."""

    ALLOW = "allow"  # Allow with warning
    BLOCK = "block"  # Block entirely
    MODIFY = "modify"  # Modify and continue
    REVIEW = "review"  # Flag for human review


@dataclass
class ValidationResult:
    """Result of validation."""

    passed: bool
    action: ValidationAction = ValidationAction.ALLOW
    message: str = ""
    violations: List[str] = field(default_factory=list)
    modified_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed


class ValidationRule(ABC):
    """Abstract base class for validation rules."""

    name: str = "base_rule"
    action: ValidationAction = ValidationAction.BLOCK

    @abstractmethod
    def validate(self, text: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate text.

        Args:
            text: Text to validate.
            context: Optional context.

        Returns:
            ValidationResult.
        """
        pass


class MaxLengthRule(ValidationRule):
    """Validate maximum text length."""

    name = "max_length"

    def __init__(self, max_length: int = 100000, action: ValidationAction = ValidationAction.MODIFY):
        self.max_length = max_length
        self.action = action

    def validate(self, text: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        if len(text) <= self.max_length:
            return ValidationResult(passed=True)

        if self.action == ValidationAction.MODIFY:
            return ValidationResult(
                passed=True,
                action=self.action,
                message=f"Text truncated from {len(text)} to {self.max_length} characters",
                modified_text=text[:self.max_length],
            )

        return ValidationResult(
            passed=False,
            action=self.action,
            message=f"Text exceeds maximum length of {self.max_length}",
            violations=[f"length: {len(text)} > {self.max_length}"],
        )


class PatternBlockRule(ValidationRule):
    """Block text matching specific patterns."""

    name = "pattern_block"

    def __init__(
        self,
        patterns: List[str],
        case_sensitive: bool = False,
        action: ValidationAction = ValidationAction.BLOCK,
    ):
        self.patterns = [
            re.compile(p, 0 if case_sensitive else re.IGNORECASE)
            for p in patterns
        ]
        self.action = action

    def validate(self, text: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        violations = []
        for pattern in self.patterns:
            if pattern.search(text):
                violations.append(f"pattern: {pattern.pattern}")

        if violations:
            return ValidationResult(
                passed=False,
                action=self.action,
                message="Blocked pattern detected",
                violations=violations,
            )

        return ValidationResult(passed=True)


class AllowListRule(ValidationRule):
    """Allow only text matching allowed patterns."""

    name = "allow_list"

    def __init__(
        self,
        allowed_patterns: List[str],
        case_sensitive: bool = False,
    ):
        self.patterns = [
            re.compile(p, 0 if case_sensitive else re.IGNORECASE)
            for p in allowed_patterns
        ]

    def validate(self, text: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        for pattern in self.patterns:
            if pattern.search(text):
                return ValidationResult(passed=True)

        return ValidationResult(
            passed=False,
            action=ValidationAction.BLOCK,
            message="Text does not match any allowed patterns",
        )


class CustomRule(ValidationRule):
    """Custom validation rule using a function."""

    def __init__(
        self,
        name: str,
        check_fn: Callable[[str, Optional[Dict]], ValidationResult],
        action: ValidationAction = ValidationAction.BLOCK,
    ):
        self.name = name
        self.check_fn = check_fn
        self.action = action

    def validate(self, text: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        return self.check_fn(text, context)


class GuardrailValidator:
    """Unified guardrail validator.

    Example:
        validator = GuardrailValidator()
        validator.add_rule(MaxLengthRule(10000))
        validator.add_rule(PatternBlockRule(["password", "secret"]))

        result = await validator.validate(user_input)
        if not result.passed:
            print(f"Validation failed: {result.message}")
    """

    def __init__(self, fail_fast: bool = False):
        """Initialize validator.

        Args:
            fail_fast: Stop on first failure.
        """
        self.rules: List[ValidationRule] = []
        self.fail_fast = fail_fast

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule.

        Args:
            rule: Rule to add.
        """
        self.rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name.

        Args:
            name: Rule name.

        Returns:
            True if removed.
        """
        original_count = len(self.rules)
        self.rules = [r for r in self.rules if r.name != name]
        return len(self.rules) < original_count

    async def validate(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate text against all rules.

        Args:
            text: Text to validate.
            context: Optional context.

        Returns:
            Combined ValidationResult.
        """
        all_violations = []
        current_text = text
        final_action = ValidationAction.ALLOW

        for rule in self.rules:
            result = rule.validate(current_text, context)

            if not result.passed:
                all_violations.extend(result.violations)

                # Track most severe action
                if result.action == ValidationAction.BLOCK:
                    final_action = ValidationAction.BLOCK
                elif result.action == ValidationAction.REVIEW and final_action != ValidationAction.BLOCK:
                    final_action = ValidationAction.REVIEW

                if self.fail_fast:
                    return result

            # Apply modifications
            if result.modified_text:
                current_text = result.modified_text

        if all_violations:
            return ValidationResult(
                passed=final_action != ValidationAction.BLOCK,
                action=final_action,
                message=f"Found {len(all_violations)} violation(s)",
                violations=all_violations,
                modified_text=current_text if current_text != text else None,
            )

        return ValidationResult(
            passed=True,
            modified_text=current_text if current_text != text else None,
        )

    def validate_sync(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Synchronous validation.

        Args:
            text: Text to validate.
            context: Optional context.

        Returns:
            ValidationResult.
        """
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.validate(text, context)
        )


# Pre-built validators
def create_safe_validator() -> GuardrailValidator:
    """Create a validator with safe defaults.

    Returns:
        GuardrailValidator with default rules.
    """
    validator = GuardrailValidator()
    validator.add_rule(MaxLengthRule(100000))
    validator.add_rule(PatternBlockRule([
        r"rm\s+-rf\s+/",
        r"DROP\s+TABLE",
        r"DELETE\s+FROM.*WHERE\s+1=1",
    ]))
    return validator


__all__ = [
    "ValidationAction",
    "ValidationResult",
    "ValidationRule",
    "MaxLengthRule",
    "PatternBlockRule",
    "AllowListRule",
    "CustomRule",
    "GuardrailValidator",
    "create_safe_validator",
]
