"""Growth-rate estimator: 8-K regex + Yahoo consensus + trailing FCF CAGR."""

from valuation.growth.estimate import compute_growth_estimate
from valuation.growth.types import GuidanceCandidate, GrowthEstimate

__all__ = ["GuidanceCandidate", "GrowthEstimate", "compute_growth_estimate"]
