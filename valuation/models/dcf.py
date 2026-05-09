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


class DCFResult(TypedDict):
    base_fcf: float
    growth_rate: float
    wacc: float
    terminal_growth: float
    projection_years: int
    projected_fcfs: list[float]
    pv_fcfs: list[float]
    terminal_value: float
    pv_terminal_value: float
    enterprise_value: float
    net_debt: float
    equity_value: float
    shares_outstanding: float
    intrinsic_value_per_share: float


def intrinsic_value_per_share(
    base_fcf: float,
    growth_rate: float,
    wacc: float,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float,
    projection_years: int = 5,
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
        projection_years: Number of explicit-projection years (default 5).

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
    terminal_value = fcf_final * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal_value = terminal_value / (1 + wacc) ** projection_years

    enterprise_value = sum(pv_fcfs) + pv_terminal_value
    equity_value = enterprise_value - net_debt
    intrinsic = equity_value / shares_outstanding

    return DCFResult(
        base_fcf=base_fcf,
        growth_rate=growth_rate,
        wacc=wacc,
        terminal_growth=terminal_growth,
        projection_years=projection_years,
        projected_fcfs=projected_fcfs,
        pv_fcfs=pv_fcfs,
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal_value,
        enterprise_value=enterprise_value,
        net_debt=net_debt,
        equity_value=equity_value,
        shares_outstanding=shares_outstanding,
        intrinsic_value_per_share=intrinsic,
    )
