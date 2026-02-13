"""PII Detection for Agentic Workflows.

Provides detection of personally identifiable information (PII).

Usage:
    from agentic_workflows.guardrails.pii import PIIDetector

    detector = PIIDetector()
    matches = detector.detect(text)
    redacted = detector.redact(text)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of PII."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    ADDRESS = "address"
    NAME = "name"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    BANK_ACCOUNT = "bank_account"
    MEDICAL_RECORD = "medical_record"
    API_KEY = "api_key"
    PASSWORD = "password"


@dataclass
class PIIMatch:
    """A detected PII match."""

    pii_type: PIIType
    value: str
    start: int
    end: int
    confidence: float = 1.0
    context: str = ""


# PII Patterns
PII_PATTERNS: Dict[PIIType, List[Tuple[str, float]]] = {
    PIIType.EMAIL: [
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", 0.95),
    ],
    PIIType.PHONE: [
        (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", 0.8),
        (r"\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b", 0.9),
        (r"\b\+1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", 0.9),
    ],
    PIIType.SSN: [
        (r"\b\d{3}-\d{2}-\d{4}\b", 0.95),
        (r"\b\d{9}\b", 0.3),  # Low confidence - could be other numbers
    ],
    PIIType.CREDIT_CARD: [
        (r"\b4[0-9]{12}(?:[0-9]{3})?\b", 0.9),  # Visa
        (r"\b5[1-5][0-9]{14}\b", 0.9),  # Mastercard
        (r"\b3[47][0-9]{13}\b", 0.9),  # Amex
        (r"\b6(?:011|5[0-9]{2})[0-9]{12}\b", 0.9),  # Discover
    ],
    PIIType.IP_ADDRESS: [
        (r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b", 0.85),
    ],
    PIIType.DATE_OF_BIRTH: [
        (r"\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12][0-9]|3[01])/(?:19|20)\d{2}\b", 0.7),
        (r"\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])\b", 0.7),
    ],
    PIIType.API_KEY: [
        (r"\b(sk-[a-zA-Z0-9]{32,})\b", 0.95),  # OpenAI
        (r"\b(sk-ant-[a-zA-Z0-9-]{90,})\b", 0.95),  # Anthropic
        (r"\b(ghp_[a-zA-Z0-9]{36})\b", 0.95),  # GitHub
        (r"\b(AKIA[0-9A-Z]{16})\b", 0.95),  # AWS
    ],
    PIIType.PASSWORD: [
        (r"(?i)password[\"']?\s*[:=]\s*[\"']?([^\"'\s,;]+)", 0.9),
        (r"(?i)pwd[\"']?\s*[:=]\s*[\"']?([^\"'\s,;]+)", 0.85),
        (r"(?i)secret[\"']?\s*[:=]\s*[\"']?([^\"'\s,;]+)", 0.8),
    ],
}


class PIIDetector:
    """Detects PII in text.

    Example:
        detector = PIIDetector()

        # Detect PII
        matches = detector.detect("Email: john@example.com, SSN: 123-45-6789")
        for match in matches:
            print(f"Found {match.pii_type}: {match.value}")

        # Redact PII
        redacted = detector.redact(text)
    """

    def __init__(
        self,
        pii_types: Optional[List[PIIType]] = None,
        min_confidence: float = 0.5,
        custom_patterns: Optional[Dict[PIIType, List[Tuple[str, float]]]] = None,
    ):
        """Initialize detector.

        Args:
            pii_types: Types to detect (all if None).
            min_confidence: Minimum confidence threshold.
            custom_patterns: Additional patterns.
        """
        self.pii_types = pii_types or list(PIIType)
        self.min_confidence = min_confidence

        # Compile patterns
        self.patterns: Dict[PIIType, List[Tuple[re.Pattern, float]]] = {}

        for pii_type in self.pii_types:
            patterns = PII_PATTERNS.get(pii_type, [])
            if custom_patterns and pii_type in custom_patterns:
                patterns = patterns + custom_patterns[pii_type]

            self.patterns[pii_type] = [
                (re.compile(p, re.IGNORECASE), conf)
                for p, conf in patterns
            ]

    def detect(self, text: str) -> List[PIIMatch]:
        """Detect PII in text.

        Args:
            text: Text to scan.

        Returns:
            List of PIIMatch objects.
        """
        matches = []

        for pii_type, patterns in self.patterns.items():
            for pattern, confidence in patterns:
                if confidence < self.min_confidence:
                    continue

                for match in pattern.finditer(text):
                    # Get context around match
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 20)
                    context = text[start:end]

                    matches.append(PIIMatch(
                        pii_type=pii_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                        context=context,
                    ))

        # Sort by position
        matches.sort(key=lambda m: m.start)

        return matches

    def redact(
        self,
        text: str,
        replacement: str = "[REDACTED]",
        pii_types: Optional[List[PIIType]] = None,
    ) -> str:
        """Redact PII from text.

        Args:
            text: Text to redact.
            replacement: Replacement string.
            pii_types: Types to redact (all detected if None).

        Returns:
            Redacted text.
        """
        matches = self.detect(text)

        if pii_types:
            matches = [m for m in matches if m.pii_type in pii_types]

        # Sort by position (reverse) for replacement
        matches.sort(key=lambda m: m.start, reverse=True)

        result = text
        for match in matches:
            result = result[:match.start] + replacement + result[match.end:]

        return result

    def mask(
        self,
        text: str,
        mask_char: str = "*",
        visible_chars: int = 4,
    ) -> str:
        """Mask PII in text, keeping some characters visible.

        Args:
            text: Text to mask.
            mask_char: Character to use for masking.
            visible_chars: Number of characters to keep visible.

        Returns:
            Masked text.
        """
        matches = self.detect(text)
        matches.sort(key=lambda m: m.start, reverse=True)

        result = text
        for match in matches:
            value = match.value
            if len(value) <= visible_chars:
                masked = mask_char * len(value)
            else:
                masked = value[:visible_chars] + mask_char * (len(value) - visible_chars)
            result = result[:match.start] + masked + result[match.end:]

        return result

    def has_pii(self, text: str) -> bool:
        """Check if text contains PII.

        Args:
            text: Text to check.

        Returns:
            True if PII found.
        """
        return len(self.detect(text)) > 0

    def get_pii_summary(self, text: str) -> Dict[str, int]:
        """Get summary of PII types found.

        Args:
            text: Text to analyze.

        Returns:
            Dict mapping PII type to count.
        """
        matches = self.detect(text)
        summary: Dict[str, int] = {}

        for match in matches:
            key = match.pii_type.value
            summary[key] = summary.get(key, 0) + 1

        return summary


__all__ = [
    "PIIType",
    "PIIMatch",
    "PII_PATTERNS",
    "PIIDetector",
]
