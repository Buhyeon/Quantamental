"""EPS per-share intrinsic model smoke + monotonic guards."""

from __future__ import annotations

import math
import warnings

import pytest

from valuation.models.eps_dcf import (
    intrinsic_price_from_eps,
    intrinsic_price_from_eps_explicit_decay,
)


def test_eps_zero_growth_perpetuity_shape():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = intrinsic_price_from_eps(
            base_eps=2.50,
            growth_rate=0.0,
            wacc=0.10,
            terminal_growth=0.0,
            projection_years=5,
        )
    analytic = sum(
        2.50 / (1.10) ** t for t in range(1, 6)
    ) + (1 / 1.10**5) * (2.50 / 0.10)
    assert math.isclose(r["intrinsic_price_per_share"], analytic, rel_tol=1e-9)


def test_higher_eps_growth_raises_price():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        lo = intrinsic_price_from_eps(
            2.0, 0.04, 0.10, 0.025, projection_years=5
        )["intrinsic_price_per_share"]
        hi = intrinsic_price_from_eps(
            2.0, 0.12, 0.10, 0.025, projection_years=5
        )["intrinsic_price_per_share"]
    assert hi > lo


@pytest.mark.parametrize("bad_eps", [-1.0, 0.0])
def test_nonpositive_eps_warns(bad_eps):
    with pytest.warns(UserWarning):
        intrinsic_price_from_eps(
            base_eps=bad_eps,
            growth_rate=0.01,
            wacc=0.11,
            terminal_growth=0.02,
        )


def test_eps_explicit_decay_pv_identity():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = intrinsic_price_from_eps_explicit_decay(
            base_eps=2.0,
            growth_rate=0.05,
            wacc=0.10,
            terminal_growth=0.02,
            projection_years=4,
        )
    assert len(r["explicit_growth_rates"]) == 4
    assert math.isclose(
        sum(r["pv_eps"])
        + sum(r["pv_terminal_period_eps"])
        + r["pv_terminal_value_per_share"],
        r["intrinsic_price_per_share"],
        rel_tol=1e-9,
    )
