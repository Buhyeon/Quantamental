"""Auto-growth mode and FCF CAGR window wiring."""

from __future__ import annotations

from unittest.mock import patch

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
