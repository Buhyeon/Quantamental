"""Per-share EPS DCF (no enterprise-value bridge).

Discounts projected EPS flows directly into an intrinsic equity price::

    P0 = Σ PV(EPS_t) + PV(TV_EPS)
"""

from __future__ import annotations

import warnings
from typing import TypedDict

from valuation.config import DEFAULT_PROJECTION_YEARS
from valuation.models.dcf import explicit_yoy_rates_decay, _terminal_phase_and_gordon


class EPSDCFResult(TypedDict):
    base_eps: float
    growth_rate: float
    wacc: float
    terminal_growth: float
    projection_years: int
    terminal_period_years: int
    projected_eps: list[float]
    pv_eps: list[float]
    explicit_growth_rates: list[float]
    projected_terminal_eps: list[float]
    pv_terminal_period_eps: list[float]
    terminal_value_per_share: float
    pv_terminal_value_per_share: float
    intrinsic_price_per_share: float


def intrinsic_price_from_eps(
    base_eps: float,
    growth_rate: float,
    wacc: float,
    terminal_growth: float,
    projection_years: int = DEFAULT_PROJECTION_YEARS,
    terminal_period_years: int = 0,
) -> EPSDCFResult:
    """DCF on EPS only — intrinsic **price per share**.

    Mirrors :func:`valuation.models.dcf.intrinsic_value_per_share` maths but skips
    net-debt reconciliation (EPS is already a per-share dividendable claim).
    """
    if wacc <= terminal_growth:
        raise ValueError(
            f"Discount rate ({wacc:.4f}) must exceed terminal_growth ({terminal_growth:.4f})."
        )
    if projection_years < 1:
        raise ValueError("projection_years must be >= 1")
    if terminal_period_years < 0:
        raise ValueError("terminal_period_years must be >= 0")
    if base_eps <= 0:
        warnings.warn(
            "Base EPS <= 0; per-share intrinsic value models are unreliable for "
            "loss-making companies.",
            stacklevel=2,
        )

    if growth_rate > 0.25:
        warnings.warn(
            f"EPS growth assumption {growth_rate:.1%} is aggressive (>25%).",
            stacklevel=2,
        )
    if wacc < 0.06:
        warnings.warn(
            f"Discount rate {wacc:.1%} is unusually low (<6%); value is inflated.",
            stacklevel=2,
        )

    projected_eps: list[float] = []
    pv_eps: list[float] = []
    for t in range(1, projection_years + 1):
        eps_t = base_eps * (1 + growth_rate) ** t
        pv_t = eps_t / (1 + wacc) ** t
        projected_eps.append(eps_t)
        pv_eps.append(pv_t)

    eps_final = projected_eps[-1]
    proj_t, pv_t, tv, pv_tv = _terminal_phase_and_gordon(
        eps_final,
        wacc,
        terminal_growth,
        terminal_period_years,
        projection_years,
    )

    intrinsic = sum(pv_eps) + sum(pv_t) + pv_tv
    explicit_rates = [growth_rate] * projection_years

    return EPSDCFResult(
        base_eps=base_eps,
        growth_rate=growth_rate,
        wacc=wacc,
        terminal_growth=terminal_growth,
        projection_years=projection_years,
        terminal_period_years=terminal_period_years,
        projected_eps=projected_eps,
        pv_eps=pv_eps,
        explicit_growth_rates=explicit_rates,
        projected_terminal_eps=proj_t,
        pv_terminal_period_eps=pv_t,
        terminal_value_per_share=tv,
        pv_terminal_value_per_share=pv_tv,
        intrinsic_price_per_share=intrinsic,
    )


def intrinsic_price_from_eps_explicit_decay(
    base_eps: float,
    growth_rate: float,
    wacc: float,
    terminal_growth: float,
    projection_years: int = DEFAULT_PROJECTION_YEARS,
    high_growth_years: int = 3,
    terminal_period_years: int = 0,
) -> EPSDCFResult:
    """EPS DCF with explicit YoY rates fading to ``terminal_growth``."""
    if wacc <= terminal_growth:
        raise ValueError(
            f"Discount rate ({wacc:.4f}) must exceed terminal_growth ({terminal_growth:.4f})."
        )
    if projection_years < 1:
        raise ValueError("projection_years must be >= 1")
    if terminal_period_years < 0:
        raise ValueError("terminal_period_years must be >= 0")
    if base_eps <= 0:
        warnings.warn(
            "Base EPS <= 0; per-share intrinsic value models are unreliable for "
            "loss-making companies.",
            stacklevel=2,
        )

    rates = explicit_yoy_rates_decay(
        growth_rate, terminal_growth, projection_years, high_years=high_growth_years
    )
    if max(rates) > 0.25:
        warnings.warn(
            f"Peak explicit EPS growth {max(rates):.1%} is aggressive (>25%).",
            stacklevel=2,
        )
    if wacc < 0.06:
        warnings.warn(
            f"Discount rate {wacc:.1%} is unusually low (<6%); value is inflated.",
            stacklevel=2,
        )

    projected_eps: list[float] = []
    pv_eps: list[float] = []
    eps_prev = base_eps
    for t in range(projection_years):
        g = rates[t]
        eps_t = eps_prev * (1 + g)
        projected_eps.append(eps_t)
        pv_eps.append(eps_t / (1 + wacc) ** (t + 1))
        eps_prev = eps_t

    eps_final = projected_eps[-1]
    proj_t, pv_t, tv, pv_tv = _terminal_phase_and_gordon(
        eps_final,
        wacc,
        terminal_growth,
        terminal_period_years,
        projection_years,
    )

    intrinsic = sum(pv_eps) + sum(pv_t) + pv_tv

    return EPSDCFResult(
        base_eps=base_eps,
        growth_rate=growth_rate,
        wacc=wacc,
        terminal_growth=terminal_growth,
        projection_years=projection_years,
        terminal_period_years=terminal_period_years,
        projected_eps=projected_eps,
        pv_eps=pv_eps,
        explicit_growth_rates=rates,
        projected_terminal_eps=proj_t,
        pv_terminal_period_eps=pv_t,
        terminal_value_per_share=tv,
        pv_terminal_value_per_share=pv_tv,
        intrinsic_price_per_share=intrinsic,
    )
