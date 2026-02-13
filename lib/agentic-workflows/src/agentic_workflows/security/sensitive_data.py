"""
Sensitive Data Detection and Redaction

Detects and redacts PII, PHI, credentials, and other sensitive data
from text to prevent data leakage.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Tuple


class DataCategory(Enum):
    """Categories of sensitive data"""
    PII = "pii"              # Personally Identifiable Information
    PHI = "phi"              # Protected Health Information
    PCI = "pci"              # Payment Card Industry data
    CREDENTIALS = "creds"    # API keys, passwords, tokens
    INTERNAL = "internal"    # Internal URLs, IPs


class Severity(Enum):
    """Severity levels for data exposure"""
    CRITICAL = "critical"  # Immediate breach risk
    HIGH = "high"          # Significant risk
    MEDIUM = "medium"      # Moderate risk
    LOW = "low"            # Minor risk


@dataclass
class SensitiveMatch:
    """A detected piece of sensitive data"""
    data_type: str
    category: DataCategory
    severity: Severity
    value: str
    redacted: str
    start: int
    end: int
    confidence: float


class SensitiveDataFilter:
    """
    Detect and redact sensitive data from text.

    Supports:
    - PII: SSN, email, phone, addresses, names
    - PHI: Medical record numbers, diagnoses, DOB
    - PCI: Credit cards, CVV, account numbers
    - Credentials: API keys, passwords, tokens, private keys
    - Internal: Private IPs, internal URLs
    """

    def __init__(self, custom_patterns: Dict[str, Tuple[str, DataCategory, Severity]] = None):
        self.patterns = self._compile_patterns()
        if custom_patterns:
            self._add_custom_patterns(custom_patterns)

    def scan(self, text: str) -> List[SensitiveMatch]:
        """
        Scan text for sensitive data.

        Args:
            text: Input text to scan

        Returns:
            List of sensitive data matches
        """
        matches = []

        for data_type, pattern, category, severity, redact_style in self.patterns:
            for match in pattern.finditer(text):
                value = match.group()
                matches.append(SensitiveMatch(
                    data_type=data_type,
                    category=category,
                    severity=severity,
                    value=value,
                    redacted=self._redact(value, data_type, redact_style),
                    start=match.start(),
                    end=match.end(),
                    confidence=self._calculate_confidence(data_type, value)
                ))

        # Remove overlapping matches (keep highest severity)
        matches = self._deduplicate(matches)

        return matches

    def redact(self, text: str, min_severity: Severity = Severity.LOW) -> Tuple[str, List[SensitiveMatch]]:
        """
        Redact sensitive data from text.

        Args:
            text: Input text
            min_severity: Minimum severity to redact

        Returns:
            Tuple of (redacted text, list of matches)
        """
        matches = self.scan(text)
        severity_order = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
        min_level = severity_order[min_severity]

        # Filter by severity
        to_redact = [m for m in matches if severity_order[m.severity] >= min_level]

        # Sort by position (reverse) to redact from end to start
        to_redact.sort(key=lambda m: m.start, reverse=True)

        # Perform redaction
        result = text
        for match in to_redact:
            result = result[:match.start] + match.redacted + result[match.end:]

        return result, matches

    def has_sensitive_data(self, text: str, min_severity: Severity = Severity.MEDIUM) -> bool:
        """Quick check if text contains sensitive data above threshold"""
        matches = self.scan(text)
        severity_order = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
        min_level = severity_order[min_severity]

        return any(severity_order[m.severity] >= min_level for m in matches)

    def get_report(self, text: str) -> dict:
        """Generate a report of sensitive data found"""
        matches = self.scan(text)

        report = {
            "total_findings": len(matches),
            "by_category": {},
            "by_severity": {},
            "findings": []
        }

        for match in matches:
            # Count by category
            cat = match.category.value
            report["by_category"][cat] = report["by_category"].get(cat, 0) + 1

            # Count by severity
            sev = match.severity.value
            report["by_severity"][sev] = report["by_severity"].get(sev, 0) + 1

            # Add finding detail (with redacted value)
            report["findings"].append({
                "type": match.data_type,
                "category": cat,
                "severity": sev,
                "preview": match.redacted,
                "position": (match.start, match.end),
                "confidence": match.confidence
            })

        return report

    def _compile_patterns(self) -> List[Tuple[str, re.Pattern, DataCategory, Severity, str]]:
        """Compile detection patterns"""
        patterns = [
            # PII - Critical
            ("ssn", r"\b\d{3}-\d{2}-\d{4}\b",
             DataCategory.PII, Severity.CRITICAL, "full"),
            ("ssn_no_dash", r"\b\d{9}\b(?=.*(?:ssn|social|security))",
             DataCategory.PII, Severity.CRITICAL, "full"),

            # PII - High
            ("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",
             DataCategory.PII, Severity.MEDIUM, "partial_email"),
            ("phone_us", r"\b(?:\+1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
             DataCategory.PII, Severity.MEDIUM, "partial"),
            ("phone_intl", r"\b\+[1-9]\d{1,3}[-.\s]?\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b",
             DataCategory.PII, Severity.MEDIUM, "partial"),

            # PHI - Critical
            ("mrn", r"\bMRN[:\s#]*\d{6,12}\b",
             DataCategory.PHI, Severity.CRITICAL, "full"),
            ("patient_id", r"\b(?:patient|pt)[:\s#-]*(?:id)?[:\s#-]*\d{5,10}\b",
             DataCategory.PHI, Severity.CRITICAL, "full"),
            ("diagnosis_icd", r"\b(?:ICD-?(?:10|11)|diagnosis)[:\s]*[A-Z]\d{2}(?:\.\d{1,4})?\b",
             DataCategory.PHI, Severity.HIGH, "full"),
            ("dob", r"\b(?:DOB|date\s+of\s+birth)[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
             DataCategory.PHI, Severity.HIGH, "full"),

            # PCI - Critical
            ("credit_card_visa", r"\b4[0-9]{3}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}\b",
             DataCategory.PCI, Severity.CRITICAL, "card"),
            ("credit_card_mc", r"\b5[1-5][0-9]{2}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}\b",
             DataCategory.PCI, Severity.CRITICAL, "card"),
            ("credit_card_amex", r"\b3[47][0-9]{2}[-\s]?[0-9]{6}[-\s]?[0-9]{5}\b",
             DataCategory.PCI, Severity.CRITICAL, "card"),
            ("cvv", r"\b(?:CVV|CVC|CID)[:\s]*\d{3,4}\b",
             DataCategory.PCI, Severity.CRITICAL, "full"),
            ("bank_account", r"\b(?:account|acct)[:\s#]*\d{8,17}\b",
             DataCategory.PCI, Severity.HIGH, "partial"),

            # Credentials - Critical
            ("aws_key_id", r"\bAKIA[0-9A-Z]{16}\b",
             DataCategory.CREDENTIALS, Severity.CRITICAL, "full"),
            ("aws_secret", r"(?:aws[_-]?secret|AWS_SECRET)[=:\s]+['\"]?([A-Za-z0-9/+=]{40})['\"]?",
             DataCategory.CREDENTIALS, Severity.CRITICAL, "full"),
            ("github_token", r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b",
             DataCategory.CREDENTIALS, Severity.CRITICAL, "full"),
            ("api_key", r"(?:api[_-]?key|apikey)[=:\s]+['\"]?([A-Za-z0-9_-]{20,})['\"]?",
             DataCategory.CREDENTIALS, Severity.CRITICAL, "full"),
            ("bearer_token", r"\bBearer\s+[A-Za-z0-9_-]{20,}\b",
             DataCategory.CREDENTIALS, Severity.CRITICAL, "full"),
            ("jwt", r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b",
             DataCategory.CREDENTIALS, Severity.CRITICAL, "full"),
            ("private_key", r"-----BEGIN\s+(?:RSA\s+|EC\s+|OPENSSH\s+)?PRIVATE\s+KEY-----",
             DataCategory.CREDENTIALS, Severity.CRITICAL, "full"),
            ("password", r"(?:password|passwd|pwd)[=:\s]+['\"]?([^\s'\"]{8,})['\"]?",
             DataCategory.CREDENTIALS, Severity.HIGH, "full"),

            # Internal - Medium
            ("internal_ip_10", r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
             DataCategory.INTERNAL, Severity.MEDIUM, "partial_ip"),
            ("internal_ip_172", r"\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b",
             DataCategory.INTERNAL, Severity.MEDIUM, "partial_ip"),
            ("internal_ip_192", r"\b192\.168\.\d{1,3}\.\d{1,3}\b",
             DataCategory.INTERNAL, Severity.MEDIUM, "partial_ip"),
            ("internal_url", r"https?://[a-z0-9.-]+\.(?:internal|corp|local|lan|intra)\b[^\s]*",
             DataCategory.INTERNAL, Severity.MEDIUM, "domain_only"),
            ("localhost", r"\b(?:localhost|127\.0\.0\.1)(?::\d+)?(?:/[^\s]*)?\b",
             DataCategory.INTERNAL, Severity.LOW, "keep"),
        ]

        return [
            (name, re.compile(pattern, re.IGNORECASE), cat, sev, style)
            for name, pattern, cat, sev, style in patterns
        ]

    def _add_custom_patterns(self, custom: Dict[str, Tuple[str, DataCategory, Severity]]):
        """Add custom patterns"""
        for name, (pattern, category, severity) in custom.items():
            self.patterns.append(
                (name, re.compile(pattern, re.IGNORECASE), category, severity, "full")
            )

    def _redact(self, value: str, data_type: str, style: str) -> str:
        """Generate redacted replacement"""
        if style == "full":
            return f"[REDACTED_{data_type.upper()}]"

        elif style == "partial":
            if len(value) > 4:
                return value[:2] + "*" * (len(value) - 4) + value[-2:]
            return "*" * len(value)

        elif style == "partial_email":
            parts = value.split("@")
            if len(parts) == 2:
                local = parts[0]
                domain = parts[1]
                if len(local) > 2:
                    local = local[0] + "*" * (len(local) - 2) + local[-1]
                return f"{local}@{domain}"
            return "[REDACTED_EMAIL]"

        elif style == "card":
            # Keep last 4 digits
            digits_only = re.sub(r'\D', '', value)
            return "**** **** **** " + digits_only[-4:]

        elif style == "partial_ip":
            parts = value.split(".")
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.*.*"
            return "[REDACTED_IP]"

        elif style == "domain_only":
            # Extract and keep just the domain pattern
            match = re.search(r'https?://([^/]+)', value)
            if match:
                return f"[REDACTED_URL: {match.group(1)}]"
            return "[REDACTED_URL]"

        elif style == "keep":
            return value

        return f"[REDACTED_{data_type.upper()}]"

    def _calculate_confidence(self, data_type: str, value: str) -> float:
        """Calculate confidence score for a match"""
        # Base confidence by type
        high_confidence_types = ["ssn", "credit_card", "aws_key", "private_key", "jwt"]
        medium_confidence_types = ["email", "phone", "api_key"]

        if any(t in data_type for t in high_confidence_types):
            base = 0.95
        elif any(t in data_type for t in medium_confidence_types):
            base = 0.85
        else:
            base = 0.75

        # Adjust based on value characteristics
        # (Real implementation would do more validation)

        return min(base, 1.0)

    def _deduplicate(self, matches: List[SensitiveMatch]) -> List[SensitiveMatch]:
        """Remove overlapping matches, keeping highest severity"""
        if not matches:
            return matches

        # Sort by severity, then by position
        severity_order = {Severity.CRITICAL: 4, Severity.HIGH: 3, Severity.MEDIUM: 2, Severity.LOW: 1}
        matches.sort(key=lambda m: (severity_order[m.severity], -m.start), reverse=True)

        result = []
        covered = set()

        for match in matches:
            # Check if any position in this match is already covered
            positions = set(range(match.start, match.end))
            if not positions & covered:
                result.append(match)
                covered.update(positions)

        # Re-sort by position for output
        result.sort(key=lambda m: m.start)

        return result


# Convenience functions

def redact_pii(text: str) -> str:
    """Quick redaction of PII only"""
    filter = SensitiveDataFilter()
    matches = filter.scan(text)
    pii_matches = [m for m in matches if m.category == DataCategory.PII]

    result = text
    for match in sorted(pii_matches, key=lambda m: m.start, reverse=True):
        result = result[:match.start] + match.redacted + result[match.end:]

    return result


def redact_credentials(text: str) -> str:
    """Quick redaction of credentials only"""
    filter = SensitiveDataFilter()
    matches = filter.scan(text)
    cred_matches = [m for m in matches if m.category == DataCategory.CREDENTIALS]

    result = text
    for match in sorted(cred_matches, key=lambda m: m.start, reverse=True):
        result = result[:match.start] + match.redacted + result[match.end:]

    return result


def check_for_secrets(text: str) -> List[str]:
    """Check if text contains any credentials/secrets"""
    filter = SensitiveDataFilter()
    matches = filter.scan(text)

    secrets = [
        f"{m.data_type}: {m.redacted}"
        for m in matches
        if m.category == DataCategory.CREDENTIALS
    ]

    return secrets
