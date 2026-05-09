"""Shared types for the growth estimator subpackage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Confidence = Literal["high", "med", "low"]
Metric = Literal["revenue", "eps", "fcf", "earnings", "unspecified"]


@dataclass(frozen=True)
class GuidanceCandidate:
    """A single growth-rate hypothesis from one source.

    All `growth_rate` values are decimals (0.085 = 8.5%), not percentages.
    """

    growth_rate: float
    source: str
    confidence: Confidence = "med"
    metric: Metric = "unspecified"
    period: str = ""
    citation: str = ""
    snippet: str = ""


@dataclass(frozen=True)
class GrowthEstimate:
    """Blended output of multiple :class:`GuidanceCandidate` sources."""

    growth_rate: float
    candidates: tuple[GuidanceCandidate, ...] = field(default_factory=tuple)
    method: str = "weighted_median"
    fallback_used: bool = False
