"""Orchestrates growth sources + blend into a single :class:`GrowthEstimate`."""

from __future__ import annotations

from typing import Literal

from valuation.data.edgar import fetch_submissions
from valuation.growth.blend import blend_candidates
from valuation.growth.consensus import yahoo_financial_consensus_candidates
from valuation.growth.eight_k import eight_k_guidance_candidates
from valuation.growth.historical import historical_growth_candidate
from valuation.growth.types import GrowthEstimate

AutoGrowthMode = Literal["blended", "consensus_only"]


def _fcf_slice_for_cagr(fcf_values: list[float], window: int | None) -> list[float]:
    """Trailing FY FCF series for CAGR; ``window <= 0`` or ``None`` = full history."""
    if len(fcf_values) < 2:
        return fcf_values
    if window is None or window <= 0:
        return list(fcf_values)
    w = min(int(window), len(fcf_values))
    return list(fcf_values[-w:])


def compute_growth_estimate(
    ticker: str,
    cik: int,
    fcf_values: list[float],
    *,
    filings_block: dict | None = None,
    auto_growth_mode: AutoGrowthMode = "blended",
    fcf_cagr_window: int | None = 2,
) -> GrowthEstimate:
    """Gather growth candidates and blend into a :class:`GrowthEstimate`.

    - ``blended``: optional trailing FCF CAGR (window FYs), 8-K regex, Yahoo summary.
    - ``consensus_only``: Yahoo ``Ticker.info`` EPS / revenue hints only.

    If ``filings_block`` is omitted, pulls submissions once for blended mode via
    :func:`valuation.data.edgar.fetch_submissions`.

    ``fcf_cagr_window``: last N fiscal FCF points for CAGR (default ``2``).
    ``None`` or ``<= 0`` means full history (callers map CLI/UI ``0`` to ``None``).
    """
    if auto_growth_mode == "consensus_only":
        cands = list(yahoo_financial_consensus_candidates(ticker))
        return blend_candidates(cands)

    if filings_block is None:
        filings_block = fetch_submissions(cik)["filings"]

    cands = []
    slice_vals = _fcf_slice_for_cagr(fcf_values, fcf_cagr_window)
    hist = historical_growth_candidate(slice_vals)
    if hist:
        cands.append(hist)

    cands.extend(eight_k_guidance_candidates(cik, filings_block))
    cands.extend(yahoo_financial_consensus_candidates(ticker))

    return blend_candidates(cands)
