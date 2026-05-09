"""Orchestrates growth sources + blend into a single :class:`GrowthEstimate`."""

from __future__ import annotations

from valuation.data.edgar import fetch_submissions
from valuation.growth.blend import blend_candidates
from valuation.growth.consensus import yahoo_financial_consensus_candidates
from valuation.growth.eight_k import eight_k_guidance_candidates
from valuation.growth.historical import historical_growth_candidate
from valuation.growth.types import GrowthEstimate


def compute_growth_estimate(
    ticker: str,
    cik: int,
    fcf_values: list[float],
    *,
    filings_block: dict | None = None,
) -> GrowthEstimate:
    """Gather 8-K regex, Yahoo consensus fields, historical FCF CAGR; blend.

    If ``filings_block`` is omitted, pulls submissions once via
    :func:`valuation.data.edgar.fetch_submissions`.
    """
    if filings_block is None:
        filings_block = fetch_submissions(cik)["filings"]

    cands = []

    hist = historical_growth_candidate(fcf_values)
    if hist:
        cands.append(hist)

    cands.extend(eight_k_guidance_candidates(cik, filings_block))
    cands.extend(yahoo_financial_consensus_candidates(ticker))

    return blend_candidates(cands)
