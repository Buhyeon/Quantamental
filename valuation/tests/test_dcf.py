"""Tests for the DCF math.

Use synthetic inputs that have closed-form analytical answers so we can pin the
math down precisely (rather than asserting on a single magic number).
"""

from __future__ import annotations

import math

import pytest

from valuation.models.dcf import (
    explicit_yoy_rates_decay,
    intrinsic_value_per_share,
    intrinsic_value_per_share_explicit_decay,
)


def test_zero_growth_perpetuity_matches_closed_form():
    """When growth = 0 and terminal_growth = 0, EV must equal FCF / WACC.

    This is the textbook perpetuity identity: discounting a constant cash flow
    at rate r forever yields CF/r, regardless of how many years are explicitly
    projected before the terminal value kicks in.
    """
    result = intrinsic_value_per_share(
        base_fcf=100.0,
        growth_rate=0.0,
        wacc=0.10,
        terminal_growth=0.0,
        shares_outstanding=1.0,
        net_debt=0.0,
        projection_years=5,
    )
    assert math.isclose(result["enterprise_value"], 1000.0, rel_tol=1e-9)
    assert math.isclose(result["intrinsic_value_per_share"], 1000.0, rel_tol=1e-9)


def test_net_debt_reduces_equity_value():
    no_debt = intrinsic_value_per_share(
        base_fcf=100.0, growth_rate=0.05, wacc=0.10, terminal_growth=0.02,
        shares_outstanding=10.0, net_debt=0.0,
    )
    with_debt = intrinsic_value_per_share(
        base_fcf=100.0, growth_rate=0.05, wacc=0.10, terminal_growth=0.02,
        shares_outstanding=10.0, net_debt=200.0,
    )
    delta = no_debt["equity_value"] - with_debt["equity_value"]
    assert math.isclose(delta, 200.0, rel_tol=1e-9)
    # Per-share impact = net_debt / shares
    per_share_delta = no_debt["intrinsic_value_per_share"] - with_debt["intrinsic_value_per_share"]
    assert math.isclose(per_share_delta, 20.0, rel_tol=1e-9)


def test_doubling_shares_halves_per_share_value():
    a = intrinsic_value_per_share(
        base_fcf=100.0, growth_rate=0.05, wacc=0.10, terminal_growth=0.02,
        shares_outstanding=10.0, net_debt=0.0,
    )
    b = intrinsic_value_per_share(
        base_fcf=100.0, growth_rate=0.05, wacc=0.10, terminal_growth=0.02,
        shares_outstanding=20.0, net_debt=0.0,
    )
    assert math.isclose(b["intrinsic_value_per_share"], a["intrinsic_value_per_share"] / 2)


def test_higher_growth_increases_value_monotonically():
    """All else equal, higher projected growth -> higher intrinsic value."""
    values = []
    for g in [0.02, 0.05, 0.10, 0.15]:
        r = intrinsic_value_per_share(
            base_fcf=100.0, growth_rate=g, wacc=0.10, terminal_growth=0.02,
            shares_outstanding=1.0, net_debt=0.0,
        )
        values.append(r["intrinsic_value_per_share"])
    assert values == sorted(values)


def test_higher_wacc_decreases_value_monotonically():
    values = []
    for w in [0.07, 0.09, 0.11, 0.13]:
        r = intrinsic_value_per_share(
            base_fcf=100.0, growth_rate=0.05, wacc=w, terminal_growth=0.02,
            shares_outstanding=1.0, net_debt=0.0,
        )
        values.append(r["intrinsic_value_per_share"])
    assert values == sorted(values, reverse=True)


def test_wacc_must_exceed_terminal_growth():
    with pytest.raises(ValueError, match="terminal_growth"):
        intrinsic_value_per_share(
            base_fcf=100.0, growth_rate=0.05, wacc=0.02, terminal_growth=0.03,
            shares_outstanding=1.0, net_debt=0.0,
        )


def test_zero_or_negative_shares_rejected():
    with pytest.raises(ValueError, match="shares_outstanding"):
        intrinsic_value_per_share(
            base_fcf=100.0, growth_rate=0.05, wacc=0.10, terminal_growth=0.02,
            shares_outstanding=0.0, net_debt=0.0,
        )


def test_intermediate_values_exposed():
    """The result dict must expose every intermediate so users can audit the math."""
    r = intrinsic_value_per_share(
        base_fcf=100.0, growth_rate=0.10, wacc=0.10, terminal_growth=0.02,
        shares_outstanding=10.0, net_debt=50.0, projection_years=5,
    )
    assert len(r["projected_fcfs"]) == 5
    assert len(r["pv_fcfs"]) == 5
    # Year-1 projected FCF must equal base * (1+g)
    assert math.isclose(r["projected_fcfs"][0], 110.0, rel_tol=1e-9)
    # PV of FCFs sum + PV(TV) = enterprise value
    assert math.isclose(
        sum(r["pv_fcfs"]) + r["pv_terminal_value"], r["enterprise_value"], rel_tol=1e-9
    )
    # Equity = EV - net_debt
    assert math.isclose(r["equity_value"], r["enterprise_value"] - 50.0, rel_tol=1e-9)


def test_aggressive_growth_emits_warning():
    with pytest.warns(UserWarning, match="aggressive"):
        intrinsic_value_per_share(
            base_fcf=100.0, growth_rate=0.30, wacc=0.10, terminal_growth=0.02,
            shares_outstanding=1.0, net_debt=0.0,
        )


def test_explicit_yoy_rates_decay_shape():
    r = explicit_yoy_rates_decay(0.10, 0.02, projection_years=5, high_years=3)
    assert len(r) == 5
    assert r[:3] == [0.10, 0.10, 0.10]
    assert math.isclose(r[3], 0.06, rel_tol=1e-12)
    assert math.isclose(r[4], 0.02, rel_tol=1e-12)


def test_explicit_decay_matches_constant_when_terminal_equals_high():
    g = t = 0.05
    r_decay = intrinsic_value_per_share_explicit_decay(
        base_fcf=100.0,
        growth_rate=g,
        wacc=0.10,
        terminal_growth=t,
        shares_outstanding=1.0,
        net_debt=0.0,
        projection_years=5,
    )
    r_const = intrinsic_value_per_share(
        base_fcf=100.0,
        growth_rate=g,
        wacc=0.10,
        terminal_growth=t,
        shares_outstanding=1.0,
        net_debt=0.0,
        projection_years=5,
    )
    assert math.isclose(
        r_decay["intrinsic_value_per_share"],
        r_const["intrinsic_value_per_share"],
        rel_tol=1e-9,
    )


def test_explicit_decay_pv_plus_tv_equals_ev():
    r = intrinsic_value_per_share_explicit_decay(
        base_fcf=100.0,
        growth_rate=0.08,
        wacc=0.10,
        terminal_growth=0.02,
        shares_outstanding=10.0,
        net_debt=50.0,
        projection_years=5,
    )
    assert len(r["explicit_growth_rates"]) == 5
    assert math.isclose(
        sum(r["pv_fcfs"]) + r["pv_terminal_value"], r["enterprise_value"], rel_tol=1e-9
    )
