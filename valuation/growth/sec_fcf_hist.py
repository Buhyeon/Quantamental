"""Trailing SEC FCF growth from 10-K FY series (mean of last YoY rates)."""

from __future__ import annotations

from typing import Sequence


def mean_fy_fcf_yoy_last_n(values: Sequence[float], *, n: int = 3) -> float | None:
    """Mean of up to ``n`` most recent fiscal YoY FCF growth rates (10-K).

    ``values`` are ascending fiscal-year FCF levels (oldest first).
    """
    vals = [float(x) for x in values]
    if len(vals) < 2:
        return None
    rates: list[float] = []
    for i in range(len(vals) - 1, 0, -1):
        prev, cur = vals[i - 1], vals[i]
        if prev <= 0:
            continue
        rates.append((cur - prev) / prev)
        if len(rates) >= n:
            break
    if not rates:
        return None
    return sum(rates) / len(rates)
