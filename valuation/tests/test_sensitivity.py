"""Tests for sensitivity grid and Monte Carlo helpers."""

from __future__ import annotations

import math
import random

from valuation.models.dcf import intrinsic_value_per_share
from valuation.models import sensitivity


def test_linspace_odd_steps_bounds():
    xs = sensitivity._linspace(0.0, 1.0, 3)
    assert xs == [0.0, 0.5, 1.0]


def test_linspace_single_step():
    xs = sensitivity._linspace(0.08, 0.08, 1)
    assert xs == [0.08]


def test_percentiles_linear_interpolation():
    vals = sorted([float(i) for i in range(1, 101)])
    # k = (n-1)*p; linear interpolation between sorted neighbors.
    assert math.isclose(sensitivity._percentile(vals, 0.50), 50.5, rel_tol=1e-9)
    assert math.isclose(sensitivity._percentile(vals, 0.05), 5.95, rel_tol=1e-9)


def test_sensitivity_center_cell_matches_direct_dcf():
    base_fcf = 100.0
    shares = 1.0
    net_debt = 0.0
    g, w, t = 0.08, 0.09, 0.025
    yrs = 5
    direct = intrinsic_value_per_share(
        base_fcf=base_fcf,
        growth_rate=g,
        wacc=w,
        terminal_growth=t,
        shares_outstanding=shares,
        net_debt=net_debt,
        projection_years=yrs,
    )["intrinsic_value_per_share"]
    grid = sensitivity.intrinsic_sensitivity_grid(
        base_fcf=base_fcf,
        shares_outstanding=shares,
        net_debt=net_debt,
        projection_years=yrs,
        center_growth=g,
        center_wacc=w,
        terminal_growth=t,
        growth_half_width=0.03,
        wacc_half_width=0.02,
        steps=3,
    )
    mid_iv = grid.intrinsic_per_share[1][1]
    assert math.isclose(mid_iv, direct, rel_tol=1e-9)


def test_monte_carlo_all_valid_with_stable_ranges():
    rng = random.Random(12345)
    mc = sensitivity.monte_carlo_intrinsic(
        base_fcf=100.0,
        shares_outstanding=10.0,
        net_debt=0.0,
        projection_years=5,
        center_growth=0.06,
        center_wacc=0.11,
        center_terminal=0.02,
        n_samples=200,
        growth_half_width=0.02,
        wacc_half_width=0.02,
        terminal_half_width=0.004,
        rng=rng,
    )
    assert mc.n_valid == 200
    assert math.isfinite(mc.mean)
    assert mc.p5 <= mc.p50 <= mc.p95


def test_format_sensitivity_grid_table_non_empty():
    grid = sensitivity.intrinsic_sensitivity_grid(
        base_fcf=100.0,
        shares_outstanding=1.0,
        net_debt=0.0,
        projection_years=5,
        center_growth=0.05,
        center_wacc=0.09,
        terminal_growth=0.02,
        growth_half_width=0.02,
        wacc_half_width=0.02,
        steps=2,
    )
    lines = sensitivity.format_sensitivity_grid_table(grid)
    assert len(lines) >= 3
    assert "terminal fixed" in lines[1]


def test_flatten_sensitivity_grid_matches_element_count():
    grid = sensitivity.intrinsic_sensitivity_grid(
        base_fcf=100.0,
        shares_outstanding=1.0,
        net_debt=0.0,
        projection_years=5,
        center_growth=0.05,
        center_wacc=0.09,
        terminal_growth=0.02,
        growth_half_width=0.02,
        wacc_half_width=0.02,
        steps=2,
    )
    pts = sensitivity.flatten_sensitivity_grid(grid)
    assert len(pts) == len(grid.growth_axis) * len(grid.wacc_axis)
