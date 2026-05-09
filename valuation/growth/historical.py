"""Historical FCF CAGR — the always-on safety net for the growth blender."""

from __future__ import annotations

from typing import Sequence

from valuation.growth.types import GuidanceCandidate

CAGR_LOWER_BOUND = -0.10
CAGR_UPPER_BOUND = 0.30


def _yoy_growth_mean(values: Sequence[float]) -> float | None:
    """Mean of year-over-year growth rates, ignoring zero/negative denominators."""
    rates: list[float] = []
    for prev, curr in zip(values, values[1:]):
        if prev <= 0:
            continue
        rates.append((curr - prev) / prev)
    if not rates:
        return None
    return sum(rates) / len(rates)


def fcf_cagr(values: Sequence[float]) -> float | None:
    """Compute compound annual growth rate of an FCF series.

    Returns ``None`` when the series can't support a meaningful CAGR (e.g.
    fewer than 2 points or non-positive endpoints). When endpoints are
    non-positive, falls back to the mean year-over-year growth of the
    remaining positive prev/curr pairs.
    """
    vals = list(values)
    if len(vals) < 2:
        return None
    start, end = vals[0], vals[-1]
    n = len(vals) - 1
    if start > 0 and end > 0:
        return (end / start) ** (1.0 / n) - 1.0
    return _yoy_growth_mean(vals)


def historical_growth_candidate(
    fcf_values: Sequence[float],
    *,
    lo: float = CAGR_LOWER_BOUND,
    hi: float = CAGR_UPPER_BOUND,
) -> GuidanceCandidate | None:
    """Return a :class:`GuidanceCandidate` for FCF CAGR, clipped to a sane band."""
    vals_len = len(tuple(fcf_values))
    raw = fcf_cagr(fcf_values)
    if raw is None:
        return None
    clipped = max(lo, min(hi, raw))
    n = vals_len
    return GuidanceCandidate(
        growth_rate=clipped,
        source="fcf_cagr",
        confidence="med",
        metric="fcf",
        period=f"{n}y history",
        citation="historical FCF series from EDGAR 10-K",
        snippet=(
            f"raw CAGR={raw:+.2%}"
            + (f" (clipped to {clipped:+.2%})" if clipped != raw else "")
        ),
    )
