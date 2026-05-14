"""include_yahoo_consensus flag on compute_growth_estimate."""

from __future__ import annotations

from unittest.mock import patch

from valuation.growth.estimate import compute_growth_estimate
from valuation.growth.types import GuidanceCandidate


def test_include_yahoo_false_skips_financial_consensus():
    hist = GuidanceCandidate(
        growth_rate=0.05,
        source="fcf_cagr",
        confidence="med",
        metric="fcf",
        period="2y history",
        citation="x",
        snippet="",
    )
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
            return_value=[hist],
        ) as yh,
        patch(
            "valuation.growth.estimate.historical_growth_candidate",
            return_value=hist,
        ),
    ):
        out = compute_growth_estimate(
            "X",
            cik=1,
            fcf_values=[100.0, 110.0],
            auto_growth_mode="blended",
            include_yahoo_consensus=False,
        )
    yh.assert_not_called()
    assert all(c.source != "financial_consensus" for c in out.candidates)
