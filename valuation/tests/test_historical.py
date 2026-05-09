"""Tests for historical CAGR helper."""

import math

from valuation.growth.historical import fcf_cagr, historical_growth_candidate


def test_fcf_geometric_vs_yoy_fallback():
    s = [10.0, 12.1, 14.641]
    g = fcf_cagr(s)
    assert g is not None
    assert math.isclose(float(g), 0.210, rel_tol=1e-3)


def test_historical_clip_bounds():
    long_run = list(range(1, 8))  # gentle ramp
    c = historical_growth_candidate([float(v) * 500 for v in long_run])
    assert c is not None
    assert -0.10 <= c.growth_rate <= 0.30
