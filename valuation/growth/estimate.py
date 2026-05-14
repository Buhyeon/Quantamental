"""Orchestrates growth sources + blend into a single :class:`GrowthEstimate`."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from valuation.config import DEFAULT_GROWTH_RATE
from valuation.data.edgar import fetch_company_facts, fetch_submissions, mean_ttm_fcf_yoy_growth_rates
from valuation.growth.blend import blend_candidates
from valuation.growth.consensus import (
    yahoo_financial_consensus_candidates,
    yahoo_one_year_eps_growth_for_high_phase,
    yahoo_one_year_mixed_growth_for_high_phase,
)
from valuation.growth.eight_k import eight_k_guidance_candidates
from valuation.growth.historical import historical_growth_candidate
from valuation.growth.sec_fcf_hist import mean_fy_fcf_yoy_last_n
from valuation.growth.types import GuidanceCandidate, GrowthEstimate

AutoGrowthMode = Literal[
    "fcf_sec_yahoo_mixed",
    "eps_yahoo",
    "blended_fcf",
    "blended",
    "consensus_only",
]

_CLIP_AUTO_LO = -0.15
_CLIP_AUTO_HI = 1.00


def _clip_auto(g: float) -> float:
    return max(_CLIP_AUTO_LO, min(_CLIP_AUTO_HI, g))


def _fcf_slice_for_cagr(fcf_values: list[float], window: int | None) -> list[float]:
    """Trailing FY FCF series for CAGR; ``window <= 0`` or ``None`` = full history."""
    if len(fcf_values) < 2:
        return fcf_values
    if window is None or window <= 0:
        return list(fcf_values)
    w = min(int(window), len(fcf_values))
    return list(fcf_values[-w:])


def _sec_fcf_trailing_legs(
    facts: dict[str, Any], fcf_values: list[float]
) -> tuple[float | None, float | None]:
    """(FY YoY mean, TTM YoY mean) from 10-K and 10-Q bridges."""
    fy_g = mean_fy_fcf_yoy_last_n(fcf_values, n=3)
    ttm_g = mean_ttm_fcf_yoy_growth_rates(facts, max_rates=3)
    return fy_g, ttm_g


def _growth_fcf_sec_yahoo_mixed(
    ticker: str,
    cik: int,
    fcf_values: list[float],
    *,
    company_facts: dict[str, Any] | None,
    baseline_eps: float | None,
) -> GrowthEstimate:
    facts = company_facts if company_facts is not None else fetch_company_facts(cik)
    fy_g, ttm_g = _sec_fcf_trailing_legs(facts, fcf_values)
    sec_parts: list[float] = []
    cands: list[GuidanceCandidate] = []
    if fy_g is not None:
        cg = _clip_auto(fy_g)
        sec_parts.append(cg)
        cands.append(
            GuidanceCandidate(
                growth_rate=cg,
                source="sec_fcf_fy_yoy",
                confidence="med",
                metric="fcf",
                period="last 3 FY YoY mean (10-K)",
                citation="SEC EDGAR companyfacts",
                snippet=f"raw_mean_yoy={fy_g:+.2%}",
            )
        )
    if ttm_g is not None:
        cg = _clip_auto(ttm_g)
        sec_parts.append(cg)
        cands.append(
            GuidanceCandidate(
                growth_rate=cg,
                source="sec_fcf_ttm_yoy",
                confidence="med",
                metric="fcf",
                period="TTM FCF YoY mean (10-Q)",
                citation="SEC EDGAR 10-Q bridge",
                snippet=f"raw_mean_yoy={ttm_g:+.2%}",
            )
        )
    g_sec = sum(sec_parts) / len(sec_parts) if sec_parts else None

    g_yh: float | None = None
    yh_snip = ""
    try:
        g_yh, yh_snip = yahoo_one_year_mixed_growth_for_high_phase(ticker)
        g_yh = _clip_auto(g_yh)
        cands.append(
            GuidanceCandidate(
                growth_rate=g_yh,
                source="yahoo_analyst_blend",
                confidence="med",
                metric="fcf",
                period="Yahoo 1y EPS+revenue (held 3y high-growth)",
                citation="yfinance ticker.info",
                snippet=yh_snip,
            )
        )
    except (ValueError, OSError, TypeError, KeyError, ArithmeticError):
        g_yh = None

    fb = False
    if g_sec is not None and g_yh is not None:
        rate = _clip_auto((g_sec + g_yh) / 2.0)
        method = "simple_avg(sec,yahoo)"
    elif g_sec is not None:
        rate = _clip_auto(g_sec)
        method = "sec_only_yahoo_unavailable"
        fb = g_yh is None
    elif g_yh is not None:
        rate = _clip_auto(g_yh)
        method = "yahoo_only_sec_unavailable"
        fb = True
    else:
        rate = DEFAULT_GROWTH_RATE
        method = "default_no_sec_or_yahoo"
        fb = True

    return GrowthEstimate(
        growth_rate=rate,
        candidates=tuple(cands),
        method=method,
        fallback_used=fb,
    )


def _growth_fcf_sec_trailing_blend(
    ticker: str,
    cik: int,
    fcf_values: list[float],
    *,
    company_facts: dict[str, Any] | None,
    baseline_eps: float | None,
) -> GrowthEstimate:
    """SEC-only trailing FCF growth: FY YoY + TTM YoY (no Yahoo, no 8-K)."""
    del ticker, baseline_eps
    facts = company_facts if company_facts is not None else fetch_company_facts(cik)
    fy_g, ttm_g = _sec_fcf_trailing_legs(facts, fcf_values)
    sec_parts: list[float] = []
    cands: list[GuidanceCandidate] = []
    if fy_g is not None:
        cg = _clip_auto(fy_g)
        sec_parts.append(cg)
        cands.append(
            GuidanceCandidate(
                growth_rate=cg,
                source="sec_fcf_fy_yoy",
                confidence="med",
                metric="fcf",
                period="last 3 FY YoY mean (10-K)",
                citation="SEC EDGAR companyfacts",
                snippet=f"raw_mean_yoy={fy_g:+.2%}",
            )
        )
    if ttm_g is not None:
        cg = _clip_auto(ttm_g)
        sec_parts.append(cg)
        cands.append(
            GuidanceCandidate(
                growth_rate=cg,
                source="sec_fcf_ttm_yoy",
                confidence="med",
                metric="fcf",
                period="TTM FCF YoY mean (10-Q)",
                citation="SEC EDGAR 10-Q bridge",
                snippet=f"raw_mean_yoy={ttm_g:+.2%}",
            )
        )
    g_sec = sum(sec_parts) / len(sec_parts) if sec_parts else None
    if g_sec is not None:
        rate = _clip_auto(g_sec)
        return GrowthEstimate(
            growth_rate=rate,
            candidates=tuple(cands),
            method="sec_trailing_fcf_only",
            fallback_used=False,
        )
    return GrowthEstimate(
        growth_rate=DEFAULT_GROWTH_RATE,
        candidates=tuple(cands),
        method="default_no_sec_trailing_fcf",
        fallback_used=True,
    )


def _growth_eps_yahoo_only(
    ticker: str,
    *,
    baseline_eps: float | None,
) -> GrowthEstimate:
    del baseline_eps  # Yahoo path uses ticker.info only; kept for API compatibility.
    try:
        g, snip = yahoo_one_year_eps_growth_for_high_phase(ticker)
        g = _clip_auto(g)
        c = GuidanceCandidate(
            growth_rate=g,
            source="yahoo_eps_forward",
            confidence="med",
            metric="eps",
            period="Yahoo 1y forward vs trailing EPS (held 3y high-growth)",
            citation="yfinance ticker.info",
            snippet=snip,
        )
        return GrowthEstimate(
            growth_rate=g,
            candidates=(c,),
            method="yahoo_eps_only",
            fallback_used=False,
        )
    except (ValueError, OSError, TypeError, KeyError, ArithmeticError):
        return GrowthEstimate(
            growth_rate=DEFAULT_GROWTH_RATE,
            candidates=tuple(),
            method="default_yahoo_eps_failed",
            fallback_used=True,
        )


def compute_growth_estimate(
    ticker: str,
    cik: int,
    fcf_values: list[float],
    *,
    filings_block: dict | None = None,
    auto_growth_mode: AutoGrowthMode = "fcf_sec_yahoo_mixed",
    fcf_cagr_window: int | None = 2,
    include_yahoo_consensus: bool = True,
    as_of: date | None = None,
    company_facts: dict[str, Any] | None = None,
    baseline_eps: float | None = None,
) -> GrowthEstimate:
    """Gather growth candidates and blend into a :class:`GrowthEstimate`.

    - ``fcf_sec_yahoo_mixed`` (default): SEC FY YoY + TTM YoY (10-K/10-Q) averaged with
      Yahoo ``ticker.info`` one-year forward EPS+revenue growth (same rate for the 3y
      high-growth leg of the DCF).
    - ``eps_yahoo``: Yahoo one-year forward EPS growth only.
    - ``blended_fcf``: SEC trailing FCF only (mean FY YoY + mean TTM YoY from 10-K/10-Q).
    - ``blended``: trailing FCF CAGR (window FYs), 8-K regex, Yahoo summary.
    - ``consensus_only``: Yahoo ``Ticker.info`` EPS / revenue hints only (blended).

    ``company_facts``: optional SEC companyfacts JSON for TTM FCF YoY (avoids refetch).

    ``baseline_eps``: reserved for callers; Yahoo growth uses Yahoo fields only.
    """
    if auto_growth_mode == "fcf_sec_yahoo_mixed":
        return _growth_fcf_sec_yahoo_mixed(
            ticker, cik, fcf_values, company_facts=company_facts, baseline_eps=baseline_eps
        )
    if auto_growth_mode == "eps_yahoo":
        return _growth_eps_yahoo_only(ticker, baseline_eps=baseline_eps)

    if auto_growth_mode == "blended_fcf":
        return _growth_fcf_sec_trailing_blend(
            ticker, cik, fcf_values, company_facts=company_facts, baseline_eps=baseline_eps
        )

    if auto_growth_mode == "consensus_only":
        cands = (
            list(yahoo_financial_consensus_candidates(ticker))
            if include_yahoo_consensus
            else []
        )
        return blend_candidates(cands)

    if filings_block is None:
        filings_block = fetch_submissions(cik)["filings"]

    cands: list[GuidanceCandidate] = []
    slice_vals = _fcf_slice_for_cagr(fcf_values, fcf_cagr_window)
    hist = historical_growth_candidate(slice_vals)
    if hist:
        cands.append(hist)

    cands.extend(eight_k_guidance_candidates(cik, filings_block, as_of=as_of))
    if include_yahoo_consensus:
        cands.extend(yahoo_financial_consensus_candidates(ticker))

    return blend_candidates(cands)
