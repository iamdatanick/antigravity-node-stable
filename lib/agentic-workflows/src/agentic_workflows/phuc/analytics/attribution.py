"""Marketing Attribution Engine for PHUC platform."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime, timedelta
import math


class AttributionModel(Enum):
    """Attribution model types."""
    FIRST_TOUCH = "first_touch"
    LAST_TOUCH = "last_touch"
    LINEAR = "linear"
    TIME_DECAY = "time_decay"
    POSITION_BASED = "position_based"
    DATA_DRIVEN = "data_driven"


@dataclass
class Touchpoint:
    """Marketing touchpoint."""
    id: str
    campaign_id: str
    channel: str
    npi: str
    timestamp: datetime
    engagement_score: float = 0.0
    
    @property
    def age_days(self) -> float:
        return (datetime.utcnow() - self.timestamp).total_seconds() / 86400


@dataclass
class Conversion:
    """Prescription conversion."""
    id: str
    npi: str
    ndc: str
    rx_type: str  # NRx or TRx
    timestamp: datetime
    quantity: int = 1


@dataclass
class AttributionResult:
    """Attribution calculation result."""
    campaign_id: str
    channel: str
    model: AttributionModel
    
    attributed_nrx: float = 0.0
    attributed_trx: float = 0.0
    
    touchpoint_count: int = 0
    conversion_count: int = 0
    
    cost: float = 0.0
    
    @property
    def cost_per_nrx(self) -> float:
        return self.cost / self.attributed_nrx if self.attributed_nrx > 0 else 0
    
    @property
    def roas(self) -> float:
        # Assume average Rx value of $150
        revenue = (self.attributed_nrx + self.attributed_trx) * 150
        return revenue / self.cost if self.cost > 0 else 0
    
    @property
    def confidence(self) -> float:
        # Higher confidence with more touchpoints and conversions
        base = min(self.touchpoint_count / 100, 1.0) * 0.5
        conversion_factor = min(self.conversion_count / 50, 1.0) * 0.5
        return base + conversion_factor


class AttributionEngine:
    """Multi-touch attribution engine."""
    
    def __init__(self, d1_client, model: AttributionModel = AttributionModel.DATA_DRIVEN):
        self.d1 = d1_client
        self.model = model
        
        # Model parameters
        self.time_decay_halflife = 7  # days
        self.position_weights = {"first": 0.4, "middle": 0.2, "last": 0.4}
    
    async def calculate(
        self,
        campaign_id: str,
        start_date: str,
        end_date: str
    ) -> AttributionResult:
        """Calculate attribution for campaign."""
        # Get touchpoints
        touchpoints = await self._get_touchpoints(campaign_id, start_date, end_date)
        
        # Get conversions
        conversions = await self._get_conversions(campaign_id, start_date, end_date)
        
        # Get campaign cost
        cost = await self._get_campaign_cost(campaign_id)
        
        # Apply model
        if self.model == AttributionModel.FIRST_TOUCH:
            nrx, trx = self._first_touch(touchpoints, conversions)
        elif self.model == AttributionModel.LAST_TOUCH:
            nrx, trx = self._last_touch(touchpoints, conversions)
        elif self.model == AttributionModel.LINEAR:
            nrx, trx = self._linear(touchpoints, conversions)
        elif self.model == AttributionModel.TIME_DECAY:
            nrx, trx = self._time_decay(touchpoints, conversions)
        elif self.model == AttributionModel.POSITION_BASED:
            nrx, trx = self._position_based(touchpoints, conversions)
        else:  # DATA_DRIVEN
            nrx, trx = self._data_driven(touchpoints, conversions)
        
        return AttributionResult(
            campaign_id=campaign_id,
            channel=self._get_primary_channel(touchpoints),
            model=self.model,
            attributed_nrx=nrx,
            attributed_trx=trx,
            touchpoint_count=len(touchpoints),
            conversion_count=len(conversions),
            cost=cost
        )
    
    async def _get_touchpoints(
        self,
        campaign_id: str,
        start_date: str,
        end_date: str
    ) -> list[Touchpoint]:
        """Get campaign touchpoints."""
        result = await self.d1.query("""
            SELECT id, campaign_id, channel, npi, timestamp, engagement_score
            FROM touchpoints
            WHERE campaign_id = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """, [campaign_id, start_date, end_date])
        
        return [Touchpoint(
            id=r["id"],
            campaign_id=r["campaign_id"],
            channel=r["channel"],
            npi=r["npi"],
            timestamp=datetime.fromisoformat(r["timestamp"]),
            engagement_score=r.get("engagement_score", 0)
        ) for r in result.results]
    
    async def _get_conversions(
        self,
        campaign_id: str,
        start_date: str,
        end_date: str
    ) -> list[Conversion]:
        """Get attributed conversions."""
        result = await self.d1.query("""
            SELECT c.id, c.npi, c.ndc, c.rx_type, c.timestamp, c.quantity
            FROM conversions c
            INNER JOIN touchpoints t ON c.npi = t.npi
            WHERE t.campaign_id = ? AND c.timestamp BETWEEN ? AND ?
        """, [campaign_id, start_date, end_date])
        
        return [Conversion(
            id=r["id"],
            npi=r["npi"],
            ndc=r["ndc"],
            rx_type=r["rx_type"],
            timestamp=datetime.fromisoformat(r["timestamp"]),
            quantity=r.get("quantity", 1)
        ) for r in result.results]
    
    async def _get_campaign_cost(self, campaign_id: str) -> float:
        """Get total campaign cost."""
        result = await self.d1.query(
            "SELECT SUM(cost) as total FROM campaign_costs WHERE campaign_id = ?",
            [campaign_id]
        )
        return result.first().get("total", 0) if result.first() else 0
    
    def _get_primary_channel(self, touchpoints: list[Touchpoint]) -> str:
        """Get most common channel."""
        if not touchpoints:
            return "unknown"
        channels = {}
        for t in touchpoints:
            channels[t.channel] = channels.get(t.channel, 0) + 1
        return max(channels, key=channels.get)
    
    def _first_touch(
        self,
        touchpoints: list[Touchpoint],
        conversions: list[Conversion]
    ) -> tuple[float, float]:
        """First-touch attribution."""
        nrx = sum(1 for c in conversions if c.rx_type == "NRx")
        trx = sum(c.quantity for c in conversions)
        return float(nrx), float(trx)
    
    def _last_touch(
        self,
        touchpoints: list[Touchpoint],
        conversions: list[Conversion]
    ) -> tuple[float, float]:
        """Last-touch attribution."""
        return self._first_touch(touchpoints, conversions)
    
    def _linear(
        self,
        touchpoints: list[Touchpoint],
        conversions: list[Conversion]
    ) -> tuple[float, float]:
        """Linear attribution - equal credit."""
        if not touchpoints:
            return 0.0, 0.0
        
        nrx = sum(1 for c in conversions if c.rx_type == "NRx")
        trx = sum(c.quantity for c in conversions)
        
        # Credit per touchpoint
        weight = 1.0 / len(touchpoints)
        return nrx * weight, trx * weight
    
    def _time_decay(
        self,
        touchpoints: list[Touchpoint],
        conversions: list[Conversion]
    ) -> tuple[float, float]:
        """Time-decay attribution."""
        if not touchpoints:
            return 0.0, 0.0
        
        nrx = sum(1 for c in conversions if c.rx_type == "NRx")
        trx = sum(c.quantity for c in conversions)
        
        # Calculate decay weights
        weights = []
        for t in touchpoints:
            decay = math.exp(-t.age_days / self.time_decay_halflife)
            weights.append(decay)
        
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0, 0.0
        
        # Most recent touchpoint gets highest attribution
        recent_weight = weights[-1] / total_weight
        return nrx * recent_weight, trx * recent_weight
    
    def _position_based(
        self,
        touchpoints: list[Touchpoint],
        conversions: list[Conversion]
    ) -> tuple[float, float]:
        """Position-based (U-shaped) attribution."""
        if not touchpoints:
            return 0.0, 0.0
        
        nrx = sum(1 for c in conversions if c.rx_type == "NRx")
        trx = sum(c.quantity for c in conversions)
        
        n = len(touchpoints)
        if n == 1:
            return float(nrx), float(trx)
        
        # First and last get 40%, middle splits 20%
        first = self.position_weights["first"]
        last = self.position_weights["last"]
        middle = self.position_weights["middle"] / max(n - 2, 1)
        
        # Average weight across positions
        avg_weight = (first + last + middle * (n - 2)) / n
        return nrx * avg_weight, trx * avg_weight
    
    def _data_driven(
        self,
        touchpoints: list[Touchpoint],
        conversions: list[Conversion]
    ) -> tuple[float, float]:
        """Data-driven attribution using Shapley values."""
        # Simplified version - would use ML model in production
        # Combine time decay with engagement scoring
        if not touchpoints:
            return 0.0, 0.0
        
        nrx = sum(1 for c in conversions if c.rx_type == "NRx")
        trx = sum(c.quantity for c in conversions)
        
        weights = []
        for t in touchpoints:
            decay = math.exp(-t.age_days / self.time_decay_halflife)
            engagement = t.engagement_score
            weight = decay * (0.5 + 0.5 * engagement)
            weights.append(weight)
        
        total = sum(weights)
        if total == 0:
            return 0.0, 0.0
        
        # Weighted average
        avg_weight = sum(w / total * w for w in weights)
        return nrx * avg_weight, trx * avg_weight
    
    async def compare_models(
        self,
        campaign_id: str,
        start_date: str,
        end_date: str
    ) -> dict[str, AttributionResult]:
        """Compare all attribution models."""
        results = {}
        original_model = self.model
        
        for model in AttributionModel:
            self.model = model
            results[model.value] = await self.calculate(campaign_id, start_date, end_date)
        
        self.model = original_model
        return results
