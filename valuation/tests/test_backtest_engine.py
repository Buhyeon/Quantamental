"""Backtest hit-detection helpers."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from valuation.backtest.engine import (
    HitResult,
    first_hit_details,
    first_hit_trading_days,
    hit_within_horizon,
)


def _make_price_series(vals: list[float], start: date = date(2020, 1, 2)) -> pd.Series:
    idx = [start + timedelta(days=i) for i in range(len(vals))]
    return pd.Series(vals, index=idx)


def test_first_hit_within_band():
    s = _make_price_series([90.0, 101.0])
    hr = first_hit_details(s, intrinsic=100.0, tolerance=0.02, anchor_date=date(2020, 1, 2))
    assert hr.hit
    assert hr.calendar_days_from_anchor == 1
    assert hr.trading_sessions_included == 2
    assert hr.bar_index_zero_based == 1


def test_first_hit_miss():
    s = _make_price_series([50.0, 55.0])
    hr = first_hit_details(s, intrinsic=100.0, tolerance=0.02, anchor_date=date(2020, 1, 2))
    assert not hr.hit
    assert hr.first_hit_date is None


def test_first_hit_trading_days_compat_anchor():
    s = _make_price_series([100.0])
    hit, idx = first_hit_trading_days(
        s, intrinsic=100.0, tolerance=0.02, anchor_date=date(2020, 1, 2)
    )
    assert hit
    assert idx == 0


def test_hit_within_horizon_calendar():
    hr = HitResult(
        hit=True,
        intrinsic=100.0,
        tolerance=0.02,
        first_hit_date=date(2020, 1, 10),
        calendar_days_from_anchor=8,
        bar_index_zero_based=8,
        trading_sessions_included=9,
    )
    assert hit_within_horizon(hr, max_trading_sessions=None, max_calendar_days=10)
    assert not hit_within_horizon(hr, max_trading_sessions=None, max_calendar_days=5)


def test_hit_within_horizon_false_when_miss():
    miss = HitResult(
        hit=False,
        intrinsic=100.0,
        tolerance=0.02,
        first_hit_date=None,
        calendar_days_from_anchor=None,
        bar_index_zero_based=None,
        trading_sessions_included=None,
    )
    assert not hit_within_horizon(miss, max_trading_sessions=100, max_calendar_days=500)


def test_hit_within_horizon_trading_sessions():
    hr = HitResult(
        hit=True,
        intrinsic=100.0,
        tolerance=0.02,
        first_hit_date=date(2020, 1, 5),
        calendar_days_from_anchor=3,
        bar_index_zero_based=3,
        trading_sessions_included=4,
    )
    assert hit_within_horizon(hr, max_trading_sessions=5, max_calendar_days=None)
    assert not hit_within_horizon(hr, max_trading_sessions=2, max_calendar_days=None)
