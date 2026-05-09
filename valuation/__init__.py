"""DCF Intrinsic Value Engine."""

from valuation.growth.estimate import compute_growth_estimate
from valuation.models.dcf import intrinsic_value_per_share
from valuation.models.eps_dcf import intrinsic_price_from_eps

__all__ = [
    "intrinsic_value_per_share",
    "intrinsic_price_from_eps",
    "compute_growth_estimate",
]
__version__ = "0.2.0"
