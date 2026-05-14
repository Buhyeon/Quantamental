"""Discounted Cash Flow (DCF) model.

Pure math, no I/O. Given a base FCF and a set of assumptions, return the
intrinsic equity value per share.

The model:
  1. Project FCF for ``projection_years`` at ``growth_rate``.
  2. Discount each projected FCF back to today by ``wacc``.
  3. Compute terminal value at the end of the projection using Gordon Growth:
        TV = FCF_N * (1 + g_terminal) / (wacc - g_terminal)
     and discount it back too.
  4. Enterprise Value = sum of PV(FCFs) + PV(TV).
  5. Equity Value = EV - Net Debt.
  6. Intrinsic Value / share = Equity Value / Shares Outstanding.
"""

from __future__ import annotations

import warnings
from typing import TypedDict

from valuation.config import DEFAULT_PROJECTION_YEARS


class DCFResult(TypedDict):
    base_fcf: float
    growth_rate: float
    wacc: float
    terminal_growth: float
    projection_years: int
    terminal_period_years: int
    projected_fcfs: list[float]
    pv_fcfs: list[float]
    explicit_growth_rates: list[float]
    projected_terminal_fcfs: list[float]
    pv_terminal_period_fcfs: list[float]
    terminal_value: float
    pv_terminal_value: float
    enterprise_value: float
    net_debt: float
    equity_value: float
    shares_outstanding: float
    intrinsic_value_per_share: float


def explicit_yoy_rates_decay(
    high_g: float,
    terminal_g: float,
    projection_years: int,
    high_years: int = 3,
) -> list[float]:
    """YoY growth rates: ``high_g`` for the first ``high_years``, then linear fade to ``terminal_g``.

    The last explicit year uses growth rate ``terminal_g``, matching Gordon perpetuity input.
    """
    if projection_years < 1:
        raise ValueError("projection_years must be >= 1")
    span = min(high_years, projection_years)
    rates: list[float] = [high_g] * span
    rem = projection_years - span
    if rem <= 0:
        return rates
    for k in range(1, rem + 1):
        rates.append(high_g + (terminal_g - high_g) * (k / rem))
    return rates


def _terminal_phase_and_gordon(
    fcf_after_explicit: float,
    wacc: float,
    terminal_growth: float,
    terminal_period_years: int,
    explicit_years: int,
) -> tuple[list[float], list[float], float, float]:
    """After explicit horizon, optional years at ``terminal_growth`` then Gordon TV.

    First terminal-phase FCF is at calendar year ``explicit_years + 1``:
    ``fcf_after_explicit * (1 + terminal_growth)``. Gordon value uses FCF at
    ``explicit_years + terminal_period_years`` and is discounted to t=0 from
    that same year index.
    """
    if terminal_period_years < 0:
        raise ValueError("terminal_period_years must be >= 0")
    tg = terminal_growth
    proj_terminal_fcfs: list[float] = []
    pv_terminal: list[float] = []
    fcf_prev = fcf_after_explicit
    for j in range(1, terminal_period_years + 1):
        fcf_j = fcf_prev * (1.0 + tg)
        proj_terminal_fcfs.append(fcf_j)
        t_abs = explicit_years + j
        pv_terminal.append(fcf_j / (1.0 + wacc) ** t_abs)
        fcf_prev = fcf_j

    horizon = explicit_years + terminal_period_years
    fcf_for_tv = fcf_prev
    terminal_value = fcf_for_tv * (1.0 + tg) / (wacc - tg)
    pv_terminal_value = terminal_value / (1.0 + wacc) ** horizon
    return proj_terminal_fcfs, pv_terminal, terminal_value, pv_terminal_value


def intrinsic_value_per_share(
    base_fcf: float,
    growth_rate: float,
    wacc: float,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float,
    projection_years: int = DEFAULT_PROJECTION_YEARS,
    terminal_period_years: int = 0,
) -> DCFResult:
    """Compute intrinsic equity value per share via DCF.

    Args:
        base_fcf: Starting (year 0) Free Cash Flow in dollars.
        growth_rate: Annual FCF growth during the explicit projection (e.g. 0.10).
        wacc: Discount rate / required return (e.g. 0.09). Must exceed
            ``terminal_growth`` or the Gordon-Growth terminal value diverges.
        terminal_growth: Perpetual growth rate after the projection (e.g. 0.025).
        shares_outstanding: Diluted shares outstanding.
        net_debt: Total debt minus cash (negative = net cash). Subtracted from
            enterprise value to get equity value.
        projection_years: Number of explicit-projection years (default from config).

    Returns:
        A :class:`DCFResult` with every intermediate value exposed for auditing.
    """
    if wacc <= terminal_growth:
        raise ValueError(
            f"WACC ({wacc:.4f}) must exceed terminal_growth ({terminal_growth:.4f}); "
            "otherwise terminal value diverges to infinity."
        )
    if projection_years < 1:
        raise ValueError("projection_years must be >= 1")
    if terminal_period_years < 0:
        raise ValueError("terminal_period_years must be >= 0")
    if shares_outstanding <= 0:
        raise ValueError("shares_outstanding must be > 0")

    if growth_rate > 0.25:
        warnings.warn(
            f"Growth rate {growth_rate:.1%} is aggressive (>25%); DCF results "
            "are highly sensitive to optimistic growth assumptions.",
            stacklevel=2,
        )
    if wacc < 0.06:
        warnings.warn(
            f"WACC {wacc:.1%} is unusually low (<6%); intrinsic value will be inflated.",
            stacklevel=2,
        )

    projected_fcfs: list[float] = []
    pv_fcfs: list[float] = []
    for t in range(1, projection_years + 1):
        fcf_t = base_fcf * (1 + growth_rate) ** t
        pv_t = fcf_t / (1 + wacc) ** t
        projected_fcfs.append(fcf_t)
        pv_fcfs.append(pv_t)

    fcf_final = projected_fcfs[-1]
    proj_t, pv_t, terminal_value, pv_terminal_value = _terminal_phase_and_gordon(
        fcf_final,
        wacc,
        terminal_growth,
        terminal_period_years,
        projection_years,
    )

    enterprise_value = sum(pv_fcfs) + sum(pv_t) + pv_terminal_value
    equity_value = enterprise_value - net_debt
    intrinsic = equity_value / shares_outstanding
    explicit_rates = [growth_rate] * projection_years

    return DCFResult(
        base_fcf=base_fcf,
        growth_rate=growth_rate,
        wacc=wacc,
        terminal_growth=terminal_growth,
        projection_years=projection_years,
        terminal_period_years=terminal_period_years,
        projected_fcfs=projected_fcfs,
        pv_fcfs=pv_fcfs,
        explicit_growth_rates=explicit_rates,
        projected_terminal_fcfs=proj_t,
        pv_terminal_period_fcfs=pv_t,
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal_value,
        enterprise_value=enterprise_value,
        net_debt=net_debt,
        equity_value=equity_value,
        shares_outstanding=shares_outstanding,
        intrinsic_value_per_share=intrinsic,
    )


def intrinsic_value_per_share_explicit_decay(
    base_fcf: float,
    growth_rate: float,
    wacc: float,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float,
    projection_years: int = DEFAULT_PROJECTION_YEARS,
    high_growth_years: int = 3,
    terminal_period_years: int = 0,
) -> DCFResult:
    """DCF with explicit YoY rates: ``high_growth_years`` at ``growth_rate``, then linear decay to ``terminal_growth``."""
    if wacc <= terminal_growth:
        raise ValueError(
            f"WACC ({wacc:.4f}) must exceed terminal_growth ({terminal_growth:.4f}); "
            "otherwise terminal value diverges to infinity."
        )
    if projection_years < 1:
        raise ValueError("projection_years must be >= 1")
    if terminal_period_years < 0:
        raise ValueError("terminal_period_years must be >= 0")
    if shares_outstanding <= 0:
        raise ValueError("shares_outstanding must be > 0")

    rates = explicit_yoy_rates_decay(
        growth_rate, terminal_growth, projection_years, high_years=high_growth_years
    )
    if max(rates) > 0.25:
        warnings.warn(
            f"Peak explicit growth {max(rates):.1%} is aggressive (>25%); "
            "DCF results are highly sensitive.",
            stacklevel=2,
        )
    if wacc < 0.06:
        warnings.warn(
            f"WACC {wacc:.1%} is unusually low (<6%); intrinsic value will be inflated.",
            stacklevel=2,
        )

    projected_fcfs: list[float] = []
    pv_fcfs: list[float] = []
    fcf_prev = base_fcf
    for t in range(projection_years):
        g = rates[t]
        fcf_t = fcf_prev * (1 + g)
        projected_fcfs.append(fcf_t)
        pv_fcfs.append(fcf_t / (1 + wacc) ** (t + 1))
        fcf_prev = fcf_t

    fcf_final = projected_fcfs[-1]
    proj_t, pv_t, terminal_value, pv_terminal_value = _terminal_phase_and_gordon(
        fcf_final,
        wacc,
        terminal_growth,
        terminal_period_years,
        projection_years,
    )

    enterprise_value = sum(pv_fcfs) + sum(pv_t) + pv_terminal_value
    equity_value = enterprise_value - net_debt
    intrinsic = equity_value / shares_outstanding

    return DCFResult(
        base_fcf=base_fcf,
        growth_rate=growth_rate,
        wacc=wacc,
        terminal_growth=terminal_growth,
        projection_years=projection_years,
        terminal_period_years=terminal_period_years,
        projected_fcfs=projected_fcfs,
        pv_fcfs=pv_fcfs,
        explicit_growth_rates=rates,
        projected_terminal_fcfs=proj_t,
        pv_terminal_period_fcfs=pv_t,
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal_value,
        enterprise_value=enterprise_value,
        net_debt=net_debt,
        equity_value=equity_value,
        shares_outstanding=shares_outstanding,
        intrinsic_value_per_share=intrinsic,
    )
