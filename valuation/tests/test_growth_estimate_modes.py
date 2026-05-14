"""Auto-growth mode and FCF CAGR window wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from valuation.config import DEFAULT_GROWTH_RATE
from valuation.growth.estimate import compute_growth_estimate
from valuation.growth.types import GuidanceCandidate


def test_consensus_only_uses_yahoo_candidates_only():
    fake_eps = GuidanceCandidate(
        growth_rate=0.07,
        source="financial_consensus",
        confidence="med",
        metric="eps",
        period="test",
        citation="mock",
        snippet="",
    )
    with patch(
        "valuation.growth.estimate.yahoo_financial_consensus_candidates",
        return_value=[fake_eps],
    ) as yh:
        out = compute_growth_estimate(
            "TEST",
            cik=1,
            fcf_values=[100.0, 110.0, 120.0],
            auto_growth_mode="consensus_only",
        )
    yh.assert_called_once_with("TEST")
    assert len(out.candidates) == 1
    assert out.candidates[0].source == "financial_consensus"


def test_blended_passes_last_two_fcf_points_to_historical():
    captured: list[list[float]] = []

    def capture_hist(fv):
        captured.append(list(fv))
        return None

    with (
        patch(
            "valuation.growth.estimate.fetch_submissions",
            return_value={"filings": {"form": [], "filingDate": [], "accessionNumber": [], "primaryDocument": []}},
        ),
        patch(
            "valuation.growth.estimate.eight_k_guidance_candidates",
            return_value=[],
        ),
        patch(
            "valuation.growth.estimate.yahoo_financial_consensus_candidates",
            return_value=[],
        ),
        patch(
            "valuation.growth.estimate.historical_growth_candidate",
            side_effect=capture_hist,
        ),
    ):
        compute_growth_estimate(
            "X",
            cik=1,
            fcf_values=[10.0, 20.0, 40.0, 80.0],
            auto_growth_mode="blended",
            fcf_cagr_window=2,
        )

    assert captured == [[40.0, 80.0]]


def test_fcf_cagr_two_year_matches_yoy_when_positive():
    """Last-two-FY CAGR equals simple YoY when both FCF values are positive."""
    with (
        patch(
            "valuation.growth.estimate.fetch_submissions",
            return_value={"filings": {"form": [], "filingDate": [], "accessionNumber": [], "primaryDocument": []}},
        ),
        patch("valuation.growth.estimate.eight_k_guidance_candidates", return_value=[]),
        patch("valuation.growth.estimate.yahoo_financial_consensus_candidates", return_value=[]),
    ):
        out = compute_growth_estimate(
            "X",
            cik=1,
            fcf_values=[100.0, 121.0],
            auto_growth_mode="blended",
            fcf_cagr_window=2,
        )
    hist_cand = next(c for c in out.candidates if c.source == "fcf_cagr")
    assert abs(hist_cand.growth_rate - 0.21) < 1e-9


def test_blended_fcf_averages_sec_legs_no_yahoo():
    yh_mixed = MagicMock()
    with (
        patch("valuation.growth.estimate.fetch_company_facts") as fetch_cf,
        patch("valuation.growth.estimate.mean_fy_fcf_yoy_last_n", return_value=0.10),
        patch("valuation.growth.estimate.mean_ttm_fcf_yoy_growth_rates", return_value=0.30),
        patch(
            "valuation.growth.estimate.yahoo_one_year_mixed_growth_for_high_phase",
            yh_mixed,
        ),
    ):
        out = compute_growth_estimate(
            "TEST",
            cik=123,
            fcf_values=[100.0, 110.0, 121.0],
            auto_growth_mode="blended_fcf",
            company_facts={"facts": "stub"},
        )
    fetch_cf.assert_not_called()
    yh_mixed.assert_not_called()
    assert out.method == "sec_trailing_fcf_only"
    assert {c.source for c in out.candidates} == {"sec_fcf_fy_yoy", "sec_fcf_ttm_yoy"}
    assert abs(out.growth_rate - 0.20) < 1e-9
    assert not out.fallback_used


def test_blended_fcf_fallback_when_no_sec_legs():
    with (
        patch("valuation.growth.estimate.fetch_company_facts", return_value={}),
        patch("valuation.growth.estimate.mean_fy_fcf_yoy_last_n", return_value=None),
        patch("valuation.growth.estimate.mean_ttm_fcf_yoy_growth_rates", return_value=None),
    ):
        out = compute_growth_estimate(
            "TEST",
            cik=123,
            fcf_values=[],
            auto_growth_mode="blended_fcf",
        )
    assert out.growth_rate == DEFAULT_GROWTH_RATE
    assert out.method == "default_no_sec_trailing_fcf"
    assert out.fallback_used
    assert len(out.candidates) == 0
