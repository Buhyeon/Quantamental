"""Weighted-median blend of growth guidance candidates + IQR outlier pruning."""

from __future__ import annotations

import statistics
from typing import Sequence

from valuation.config import DEFAULT_GROWTH_RATE
from valuation.growth.types import GuidanceCandidate, GrowthEstimate


def candidate_weight(c: GuidanceCandidate) -> float:
    """Planner weights: consensus + regex + CAGR."""
    if c.source == "financial_consensus":
        return 0.55
    if c.source == "8-K regex":
        return 0.4
    if c.source == "fcf_cagr":
        return 0.7
    return 0.35


def _weighted_median(pairs: Sequence[tuple[float, float]]) -> float:
    """``pairs`` = (value, weight); returns weighted median."""
    if not pairs:
        return float("nan")
    seq = sorted(pairs, key=lambda x: x[0])
    tw = sum(w for _, w in seq)
    if tw <= 0:
        return float(statistics.median([v for v, _ in seq]))
    target = tw / 2.0
    acc = 0.0
    for v, w in seq:
        acc += w
        if acc >= target:
            return v
    return seq[-1][0]


def _iqr_bounds(values: Sequence[float]) -> tuple[float, float] | None:
    """Tukey fence [Q1 - 3IQR, Q3 + 3IQR], or None when not enough data."""
    vals = [float(x) for x in values]
    if len(vals) < 4:
        return None
    try:
        q1, _, q3 = statistics.quantiles(vals, n=4, method="inclusive")
    except statistics.StatisticsError:
        return None
    iqr = q3 - q1
    return q1 - 3 * iqr, q3 + 3 * iqr


def blend_candidates(
    candidates: Sequence[GuidanceCandidate],
    *,
    default_rate: float = DEFAULT_GROWTH_RATE,
) -> GrowthEstimate:
    """IQR-remove growth outliers, then take a weighted median of survivors.

    If the IQR fence removes every point, fall back to the full candidate set.
    If the result is unusable, prefer FCF CAGR-only, else ``default_rate``.
    """
    cand_list = tuple(candidates)
    if not cand_list:
        return GrowthEstimate(
            growth_rate=default_rate,
            candidates=tuple(),
            method="default_no_candidates",
            fallback_used=True,
        )

    bounds = _iqr_bounds([c.growth_rate for c in cand_list])
    wl: list[GuidanceCandidate]
    if bounds is None:
        wl = list(cand_list)
    else:
        lo, hi = bounds
        wl = [c for c in cand_list if lo <= c.growth_rate <= hi]
        if not wl:
            wl = list(cand_list)

    wm = _weighted_median([(c.growth_rate, candidate_weight(c)) for c in wl])
    if wm != wm:  # NaN
        wm = statistics.median([c.growth_rate for c in wl])

    fb = False
    if wm != wm:
        cagr = next((c.growth_rate for c in cand_list if c.source == "fcf_cagr"), None)
        if cagr is not None:
            wm = cagr
            fb = True
        else:
            wm = default_rate
            fb = True

    clipped = max(-0.20, min(0.65, wm))

    return GrowthEstimate(
        growth_rate=clipped,
        candidates=cand_list,
        method="weighted_median+IQR_outliers",
        fallback_used=fb,
    )
