"""Phase 3: sensitivity grids and Monte Carlo around DCF assumptions.

No I/O. Builds on :func:`valuation.models.dcf.intrinsic_value_per_share`.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SensitivityGridPoint:
    growth_rate: float
    wacc: float
    terminal_growth: float
    intrinsic_value_per_share: float


@dataclass(frozen=True)
class IntrinsicSensitivityGrid:
    """Rectangular intrinsic value grid: rows = growth, cols = WACC."""

    terminal_growth: float
    growth_axis: tuple[float, ...]
    wacc_axis: tuple[float, ...]
    intrinsic_per_share: tuple[tuple[float, ...], ...]


def _linspace(lo: float, hi: float, steps: int) -> list[float]:
    if steps < 2:
        return [float(lo)] if steps == 1 else []
    delta = hi - lo
    return [lo + delta * i / (steps - 1) for i in range(steps)]


def intrinsic_sensitivity_grid(
    *,
    base_fcf: float,
    shares_outstanding: float,
    net_debt: float,
    projection_years: int,
    center_growth: float,
    center_wacc: float,
    terminal_growth: float,
    growth_half_width: float = 0.03,
    wacc_half_width: float = 0.02,
    steps: int = 3,
    growth_min: float = 0.0,
    wacc_min_above_terminal: float = 0.005,
) -> IntrinsicSensitivityGrid:
    """2D grid: vary growth × WACC; hold terminal growth fixed.

    ``growth_half_width`` / ``wacc_half_width`` expand from the center in each direction;
    Growth is clipped to ``>= growth_min``.
    Each WACC is clipped to ``> terminal_growth + wacc_min_above_terminal``.
    """
    from valuation.models.dcf import intrinsic_value_per_share

    g_lo = max(growth_min, center_growth - growth_half_width)
    g_hi = center_growth + growth_half_width
    w_lo = max(terminal_growth + wacc_min_above_terminal, center_wacc - wacc_half_width)
    w_hi = max(w_lo + 1e-6, center_wacc + wacc_half_width)

    growth_axis = _linspace(g_lo, g_hi, steps)
    wacc_axis = _linspace(w_lo, w_hi, steps)

    matrix: list[list[float]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for g in growth_axis:
            row: list[float] = []
            for w in wacc_axis:
                if w <= terminal_growth:
                    row.append(float("nan"))
                    continue
                r = intrinsic_value_per_share(
                    base_fcf=base_fcf,
                    growth_rate=g,
                    wacc=w,
                    terminal_growth=terminal_growth,
                    shares_outstanding=shares_outstanding,
                    net_debt=net_debt,
                    projection_years=projection_years,
                )
                row.append(r["intrinsic_value_per_share"])
            matrix.append(row)
    return IntrinsicSensitivityGrid(
        terminal_growth=terminal_growth,
        growth_axis=tuple(growth_axis),
        wacc_axis=tuple(wacc_axis),
        intrinsic_per_share=tuple(tuple(r) for r in matrix),
    )


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


@dataclass(frozen=True)
class MonteCarloSummary:
    n_samples: int
    n_valid: int
    mean: float
    std: float
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    values: tuple[float, ...]


def monte_carlo_intrinsic(
    *,
    base_fcf: float,
    shares_outstanding: float,
    net_debt: float,
    projection_years: int,
    center_growth: float,
    center_wacc: float,
    center_terminal: float,
    n_samples: int,
    growth_half_width: float = 0.03,
    wacc_half_width: float = 0.015,
    terminal_half_width: float = 0.0075,
    rng: random.Random | None = None,
    max_attempts_factor: int = 50,
) -> MonteCarloSummary:
    """Sample growth, WACC, and terminal growth uniformly around centers; collect IV/share.

    Each draw enforces ``wacc > terminal_growth`` (re-samples up to
    ``n_samples * max_attempts_factor`` attempts before giving up).
    """
    from valuation.models.dcf import intrinsic_value_per_share

    rng = rng or random.Random()
    values: list[float] = []
    attempts = 0
    max_attempts = max(n_samples * max_attempts_factor, n_samples * 10)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        while len(values) < n_samples and attempts < max_attempts:
            attempts += 1
            g = rng.uniform(
                max(0.0, center_growth - growth_half_width),
                center_growth + growth_half_width,
            )
            w = rng.uniform(
                max(center_terminal + 0.01, center_wacc - wacc_half_width),
                center_wacc + wacc_half_width,
            )
            t_lo = max(0.0, center_terminal - terminal_half_width)
            t_hi = min(center_terminal + terminal_half_width, w - 0.005)
            if t_hi < t_lo:
                continue
            term = rng.uniform(t_lo, t_hi)
            if w <= term:
                continue
            try:
                r = intrinsic_value_per_share(
                    base_fcf=base_fcf,
                    growth_rate=g,
                    wacc=w,
                    terminal_growth=term,
                    shares_outstanding=shares_outstanding,
                    net_debt=net_debt,
                    projection_years=projection_years,
                )
            except ValueError:
                continue
            values.append(r["intrinsic_value_per_share"])

    if not values:
        return MonteCarloSummary(
            n_samples=n_samples,
            n_valid=0,
            mean=float("nan"),
            std=float("nan"),
            p5=float("nan"),
            p25=float("nan"),
            p50=float("nan"),
            p75=float("nan"),
            p95=float("nan"),
            values=tuple(),
        )

    values.sort()
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    std = var**0.5

    return MonteCarloSummary(
        n_samples=n_samples,
        n_valid=n,
        mean=mean,
        std=std,
        p5=_percentile(values, 0.05),
        p25=_percentile(values, 0.25),
        p50=_percentile(values, 0.50),
        p75=_percentile(values, 0.75),
        p95=_percentile(values, 0.95),
        values=tuple(values),
    )


def format_sensitivity_grid_table(grid: IntrinsicSensitivityGrid) -> list[str]:
    """Build ASCII table rows for a rectangular growth × WACC intrinsic grid."""
    growth_axis = grid.growth_axis
    wacc_axis = grid.wacc_axis
    if not growth_axis or not wacc_axis:
        return ["(empty sensitivity grid)"]

    gw = max(10, max(len(f"{g:.2%}") for g in growth_axis))
    ww = max(8, max(len(f"{w:.2%}") for w in wacc_axis))

    header = " " * (gw + 2) + "".join(f"{w:>{ww}.2%}" for w in wacc_axis)
    rows = [
        header,
        f"(terminal fixed at {grid.terminal_growth:.2%})",
    ]
    for i, g in enumerate(growth_axis):
        line = f"{g:>{gw}.2%} |"
        for j, _w in enumerate(wacc_axis):
            v = grid.intrinsic_per_share[i][j]
            if v != v:  # NaN
                line += f"{'--':>{ww}}"
            else:
                line += f"{v:>{ww},.0f}"
        rows.append(line)
    return rows


def flatten_sensitivity_grid(grid: IntrinsicSensitivityGrid) -> list[SensitivityGridPoint]:
    """Expand a grid into individual :class:`SensitivityGridPoint` records."""
    out: list[SensitivityGridPoint] = []
    for i, g in enumerate(grid.growth_axis):
        for j, w in enumerate(grid.wacc_axis):
            iv = grid.intrinsic_per_share[i][j]
            if iv != iv:
                continue
            out.append(
                SensitivityGridPoint(
                    growth_rate=g,
                    wacc=w,
                    terminal_growth=grid.terminal_growth,
                    intrinsic_value_per_share=iv,
                )
            )
    return out
