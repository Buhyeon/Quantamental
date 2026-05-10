"""Rolling IV revision dates from submissions."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from valuation.backtest.rolling_iv import (
    build_iv_revision_table,
    compute_segment_statistics,
    forward_fill_iv_daily,
    revision_anchor_dates,
)
from valuation.config import DCFAssumptions
from valuation.data.edgar import EdgarError


def test_revision_anchor_dates_filters_forms_and_range():
    sub = {
        "filings": {
            "form": ["10-K", "10-Q", "8-K", "10-K"],
            "filingDate": ["2020-03-15", "2020-05-01", "2020-06-01", "2019-11-01"],
        }
    }
    with patch("valuation.data.edgar.fetch_submissions", return_value=sub):
        out = revision_anchor_dates(1, date(2020, 1, 1), date(2020, 12, 31))
    assert out == [date(2020, 3, 15), date(2020, 5, 1)]


def test_revision_anchor_dates_reads_filings_recent_nesting():
    """SEC company submissions expose parallel arrays under filings["recent"]."""
    sub = {
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q", "8-K", "10-K"],
                "filingDate": ["2020-03-15", "2020-05-01", "2020-06-01", "2019-11-01"],
            }
        }
    }
    with patch("valuation.data.edgar.fetch_submissions", return_value=sub):
        out = revision_anchor_dates(1, date(2020, 1, 1), date(2020, 12, 31))
    assert out == [date(2020, 3, 15), date(2020, 5, 1)]


def test_revision_anchor_dates_includes_amended_forms_when_in_range():
    sub = {
        "filings": {
            "recent": {
                "form": ["10-K/A"],
                "filingDate": ["2021-04-02"],
            }
        }
    }
    with patch("valuation.data.edgar.fetch_submissions", return_value=sub):
        out = revision_anchor_dates(123, date(2021, 1, 1), date(2021, 12, 31))
    assert out == [date(2021, 4, 2)]


def test_build_iv_revision_table_collects_exceptions():
    anchors = [date(2020, 1, 2), date(2020, 6, 1)]
    assumptions = DCFAssumptions()
    assumptions.validate()

    bt_empty = SimpleNamespace(iv_fcf_per_share=-1.0, iv_eps_per_share=None)
    with patch(
        "valuation.backtest.rolling_iv.run_backtest_valuation",
        side_effect=[EdgarError("x"), bt_empty],
    ):
        df, skips = build_iv_revision_table(
            "TEST",
            anchors,
            assumptions,
            auto_growth=False,
            auto_growth_mode="blended",
            fcf_cagr_window=None,
            include_yahoo_consensus=False,
            fcf_avg_years=1,
            match_fcf_for_eps=True,
            manual_growth=0.08,
            eps_growth_override=None,
            model_mode="fcf",
            iv_mode="FCF IV",
        )
    assert df.empty and len(skips) == 2
    assert "EdgarError" in skips[0][1]
    assert "intrinsic" in skips[1][1].lower() or "usable" in skips[1][1].lower()


def test_forward_fill_iv_daily_merge_asof():
    ix = pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-08"])
    closes = pd.Series([10.0, 11.0, 12.0], index=ix)
    revisions = pd.DataFrame(
        {
            "revision_date": [date(2024, 1, 3), date(2024, 1, 7)],
            "iv": [100.0, 200.0],
        }
    )
    out = forward_fill_iv_daily(closes, revisions)
    assert list(out.columns) == ["session_date", "Close", "iv_step"]
    # Before first revision: NaN
    assert out.iloc[0]["iv_step"] != out.iloc[0]["iv_step"]  # NaN
    # 2024-01-03 revision not a session; Jan 4 uses 100
    assert float(out.iloc[1]["iv_step"]) == 100.0
    assert float(out.iloc[2]["iv_step"]) == 200.0


def test_compute_segment_statistics_includes_filing_day_bar():
    # Revision B on 2020-01-03; price touches new IV band on that same session.
    prices = pd.Series([50.0, 100.0], index=[date(2020, 1, 2), date(2020, 1, 3)])
    revisions = pd.DataFrame(
        {
            "revision_date": [date(2020, 1, 2), date(2020, 1, 3)],
            "iv": [200.0, 100.0],
        }
    )
    tolerance = 0.02  # band 98–102 includes 100
    segs = compute_segment_statistics(
        prices, revisions, tolerance=tolerance, chart_end=date(2025, 1, 1)
    )
    assert len(segs) == 1
    assert segs[0].hit_new
