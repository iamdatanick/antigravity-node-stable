"""Alerting system for workflow monitoring."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    def __ge__(self, other: AlertSeverity) -> bool:
        order = [
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.ERROR,
            AlertSeverity.CRITICAL,
        ]
        return order.index(self) >= order.index(other)


class AlertStatus(Enum):
    """Alert status."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class MetricType(Enum):
    """Types of metrics for alerting."""

    COST = "cost"
    TOKENS = "tokens"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    CUSTOM = "custom"


@dataclass
class AlertRule:
    """Alert rule definition."""

    name: str
    metric_type: MetricType
    threshold: float
    comparison: str  # "gt", "gte", "lt", "lte", "eq"
    severity: AlertSeverity = AlertSeverity.WARNING
    window_seconds: float = 60.0  # Evaluation window
    cooldown_seconds: float = 300.0  # Minimum time between alerts
    description: str = ""
    labels: dict[str, str] = field(default_factory=dict)

    # Runtime state
    last_triggered: float | None = None
    trigger_count: int = 0

    def evaluate(self, value: float) -> bool:
        """Evaluate if threshold is breached.

        Args:
            value: Current metric value.

        Returns:
            True if threshold breached.
        """
        if self.comparison == "gt":
            return value > self.threshold
        elif self.comparison == "gte":
            return value >= self.threshold
        elif self.comparison == "lt":
            return value < self.threshold
        elif self.comparison == "lte":
            return value <= self.threshold
        elif self.comparison == "eq":
            return value == self.threshold
        return False

    def can_trigger(self) -> bool:
        """Check if alert can trigger (cooldown elapsed)."""
        if self.last_triggered is None:
            return True
        return time.time() - self.last_triggered >= self.cooldown_seconds


@dataclass
class Alert:
    """An alert instance."""

    id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    status: AlertStatus = AlertStatus.ACTIVE
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    acknowledged_at: float | None = None
    resolved_at: float | None = None

    @property
    def duration_seconds(self) -> float:
        """Get alert duration."""
        end_time = self.resolved_at or time.time()
        return end_time - self.timestamp


class AlertManager:
    """Manages alerts and alert rules.

    Features:
    - Rule-based alerting
    - Multiple severity levels
    - Alert lifecycle (active -> acknowledged -> resolved)
    - Cooldown periods
    - Alert handlers/callbacks
    """

    def __init__(
        self,
        on_alert: Callable[[Alert], None] | None = None,
        default_cooldown: float = 300.0,
    ):
        """Initialize alert manager.

        Args:
            on_alert: Callback when alert triggers.
            default_cooldown: Default cooldown between alerts.
        """
        self.on_alert = on_alert
        self.default_cooldown = default_cooldown

        self._rules: dict[str, AlertRule] = {}
        self._alerts: dict[str, Alert] = {}
        self._metrics: dict[str, list[tuple[float, float]]] = {}  # metric -> [(timestamp, value)]
        self._lock = threading.Lock()
        self._alert_counter = 0

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.

        Args:
            rule: Alert rule to add.
        """
        with self._lock:
            self._rules[rule.name] = rule

    def remove_rule(self, name: str) -> bool:
        """Remove an alert rule.

        Args:
            name: Rule name.

        Returns:
            True if removed.
        """
        with self._lock:
            if name in self._rules:
                del self._rules[name]
                return True
            return False

    def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> list[Alert]:
        """Record a metric value and check rules.

        Args:
            metric_type: Type of metric.
            value: Metric value.
            labels: Optional labels for filtering.

        Returns:
            List of triggered alerts.
        """
        timestamp = time.time()

        with self._lock:
            # Store metric
            key = metric_type.value
            if key not in self._metrics:
                self._metrics[key] = []
            self._metrics[key].append((timestamp, value))

            # Clean old metrics
            self._cleanup_metrics(key)

        # Check rules
        triggered_alerts = []
        for rule in self._rules.values():
            if rule.metric_type != metric_type:
                continue

            # Check labels match
            if labels and rule.labels:
                if not all(labels.get(k) == v for k, v in rule.labels.items()):
                    continue

            # Evaluate
            if rule.evaluate(value) and rule.can_trigger():
                alert = self._create_alert(rule, value, labels)
                triggered_alerts.append(alert)

        return triggered_alerts

    def _cleanup_metrics(self, key: str, max_age: float = 3600.0) -> None:
        """Remove old metric data."""
        cutoff = time.time() - max_age
        self._metrics[key] = [(t, v) for t, v in self._metrics[key] if t > cutoff]

    def _create_alert(
        self,
        rule: AlertRule,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> Alert:
        """Create and store an alert."""
        with self._lock:
            self._alert_counter += 1
            alert_id = f"alert-{self._alert_counter:06d}"

        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            message=f"{rule.name}: {value} {rule.comparison} {rule.threshold}",
            value=value,
            threshold=rule.threshold,
            labels={**rule.labels, **(labels or {})},
            metadata={"description": rule.description},
        )

        with self._lock:
            self._alerts[alert_id] = alert
            rule.last_triggered = time.time()
            rule.trigger_count += 1

        # Callback
        if self.on_alert:
            self.on_alert(alert)

        return alert

    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: Alert to acknowledge.

        Returns:
            True if acknowledged.
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert and alert.status == AlertStatus.ACTIVE:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = time.time()
                return True
            return False

    def resolve(self, alert_id: str) -> bool:
        """Resolve an alert.

        Args:
            alert_id: Alert to resolve.

        Returns:
            True if resolved.
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert and alert.status != AlertStatus.RESOLVED:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = time.time()
                return True
            return False

    def get_alert(self, alert_id: str) -> Alert | None:
        """Get alert by ID."""
        return self._alerts.get(alert_id)

    def get_active_alerts(
        self,
        severity: AlertSeverity | None = None,
    ) -> list[Alert]:
        """Get active alerts.

        Args:
            severity: Filter by minimum severity.

        Returns:
            List of active alerts.
        """
        with self._lock:
            alerts = [a for a in self._alerts.values() if a.status == AlertStatus.ACTIVE]

            if severity:
                alerts = [a for a in alerts if a.severity >= severity]

            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def get_alerts(
        self,
        status: AlertStatus | None = None,
        severity: AlertSeverity | None = None,
        rule_name: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[Alert]:
        """Get alerts with filtering.

        Args:
            status: Filter by status.
            severity: Filter by minimum severity.
            rule_name: Filter by rule name.
            start_time: Filter by start timestamp.
            end_time: Filter by end timestamp.

        Returns:
            Matching alerts.
        """
        with self._lock:
            alerts = list(self._alerts.values())

        if status:
            alerts = [a for a in alerts if a.status == status]
        if severity:
            alerts = [a for a in alerts if a.severity >= severity]
        if rule_name:
            alerts = [a for a in alerts if a.rule_name == rule_name]
        if start_time:
            alerts = [a for a in alerts if a.timestamp >= start_time]
        if end_time:
            alerts = [a for a in alerts if a.timestamp <= end_time]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def get_metric_value(
        self,
        metric_type: MetricType,
        aggregation: str = "last",
        window_seconds: float = 60.0,
    ) -> float | None:
        """Get aggregated metric value.

        Args:
            metric_type: Metric type.
            aggregation: "last", "avg", "min", "max", "sum"
            window_seconds: Time window.

        Returns:
            Aggregated value or None.
        """
        with self._lock:
            key = metric_type.value
            if key not in self._metrics:
                return None

            cutoff = time.time() - window_seconds
            values = [v for t, v in self._metrics[key] if t > cutoff]

            if not values:
                return None

            if aggregation == "last":
                return values[-1]
            elif aggregation == "avg":
                return sum(values) / len(values)
            elif aggregation == "min":
                return min(values)
            elif aggregation == "max":
                return max(values)
            elif aggregation == "sum":
                return sum(values)

            return None

    def get_stats(self) -> dict[str, Any]:
        """Get alert manager statistics."""
        with self._lock:
            total_alerts = len(self._alerts)
            active = sum(1 for a in self._alerts.values() if a.status == AlertStatus.ACTIVE)
            acknowledged = sum(
                1 for a in self._alerts.values() if a.status == AlertStatus.ACKNOWLEDGED
            )
            resolved = sum(1 for a in self._alerts.values() if a.status == AlertStatus.RESOLVED)

            by_severity = {}
            for severity in AlertSeverity:
                by_severity[severity.value] = sum(
                    1
                    for a in self._alerts.values()
                    if a.severity == severity and a.status == AlertStatus.ACTIVE
                )

            return {
                "total_alerts": total_alerts,
                "active": active,
                "acknowledged": acknowledged,
                "resolved": resolved,
                "by_severity": by_severity,
                "rules_count": len(self._rules),
            }

    def format_alerts(self, alerts: list[Alert] | None = None) -> str:
        """Format alerts for display.

        Args:
            alerts: Alerts to format (or active alerts).

        Returns:
            Formatted string.
        """
        if alerts is None:
            alerts = self.get_active_alerts()

        if not alerts:
            return "No active alerts."

        lines = [f"=== Active Alerts ({len(alerts)}) ==="]

        for alert in alerts:
            severity_icon = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.ERROR: "❌",
                AlertSeverity.CRITICAL: "🚨",
            }.get(alert.severity, "")

            lines.append(f"\n{severity_icon} [{alert.severity.value.upper()}] {alert.rule_name}")
            lines.append(f"   ID: {alert.id}")
            lines.append(f"   Message: {alert.message}")
            lines.append(f"   Duration: {alert.duration_seconds:.1f}s")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all alerts and metrics."""
        with self._lock:
            self._alerts.clear()
            self._metrics.clear()
            for rule in self._rules.values():
                rule.last_triggered = None
                rule.trigger_count = 0


# Convenience function for creating common rules
def create_cost_alert(
    name: str,
    threshold_usd: float,
    severity: AlertSeverity = AlertSeverity.WARNING,
) -> AlertRule:
    """Create a cost alert rule."""
    return AlertRule(
        name=name,
        metric_type=MetricType.COST,
        threshold=threshold_usd,
        comparison="gte",
        severity=severity,
        description=f"Alert when cost reaches ${threshold_usd:.2f}",
    )


def create_error_rate_alert(
    name: str,
    threshold_percent: float,
    severity: AlertSeverity = AlertSeverity.ERROR,
) -> AlertRule:
    """Create an error rate alert rule."""
    return AlertRule(
        name=name,
        metric_type=MetricType.ERROR_RATE,
        threshold=threshold_percent,
        comparison="gte",
        severity=severity,
        description=f"Alert when error rate reaches {threshold_percent:.1f}%",
    )


def create_latency_alert(
    name: str,
    threshold_ms: float,
    severity: AlertSeverity = AlertSeverity.WARNING,
) -> AlertRule:
    """Create a latency alert rule."""
    return AlertRule(
        name=name,
        metric_type=MetricType.LATENCY,
        threshold=threshold_ms,
        comparison="gte",
        severity=severity,
        description=f"Alert when latency reaches {threshold_ms:.0f}ms",
    )
