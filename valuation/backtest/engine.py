"""Point-in-time intrinsic value for backtests (no live Yahoo consensus by default)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from valuation.config import DCFAssumptions, DEFAULT_GROWTH_RATE, average_fcf
from valuation.models.dcf import intrinsic_value_per_share
from valuation.models.eps_dcf import intrinsic_price_from_eps
from valuation.data.edgar import Fundamentals
from valuation.data.edgar_asof import get_fundamentals_as_of
from valuation.growth.estimate import AutoGrowthMode, compute_growth_estimate
from valuation.growth.types import GrowthEstimate
from valuation.models.dcf import intrinsic_value_per_share_explicit_decay
from valuation.models.eps_dcf import intrinsic_price_from_eps_explicit_decay

ModelMode = Literal["fcf", "eps", "both"]


@dataclass(frozen=True)
class BacktestValuation:
    """Intrinsic value snapshot at ``as_of`` with supporting inputs."""

    as_of: date
    ticker: str
    fundamentals: Fundamentals
    growth_estimate: GrowthEstimate | None
    growth_fcf: float
    growth_eps: float
    iv_fcf_per_share: float | None
    iv_eps_per_share: float | None
    base_fcf: float
    base_eps: float
    base_fcf_label: str
    base_eps_label: str


def _ensure_date(idx_val) -> date:
    """Coerce Timestamp / datetime to plain date for calendar math."""
    if isinstance(idx_val, date):
        return idx_val
    if hasattr(idx_val, "date") and callable(getattr(idx_val, "date", None)):
        return idx_val.date()
    return date(int(idx_val.year), int(idx_val.month), int(idx_val.day))


@dataclass(frozen=True)
class HitResult:
    """First touch of a tolerance band around ``intrinsic`` in chronologically ascending ``prices``."""

    hit: bool
    """Whether any bar satisfies the band."""
    intrinsic: float
    tolerance: float
    first_hit_date: date | None
    """Trading session calendar date where first band touch occurs."""
    calendar_days_from_anchor: int | None
    """`(first_hit_date - anchor_date).days` when anchoring attribution to ``anchor_date``."""
    bar_index_zero_based: int | None
    """Index of first hit row in ``prices``."""
    trading_sessions_included: int | None
    """Trading sessions counted from first row of ``prices`` through ``first_hit``, inclusive (= bar_index_zero_based + 1)."""


def first_hit_details(
    prices,
    intrinsic: float,
    *,
    tolerance: float,
    anchor_date: date,
) -> HitResult:
    """First session where ``lo <= close <= hi``.

    Band: ``intrinsic * (1 ± tolerance)``. ``calendar_days_from_anchor`` uses
    ``(first_hit_date - anchor_date).days``. ``anchor_date`` should match the
    economic reference (valuation date or revision date)—the price series can
    start on a later first trading day.
    """
    import pandas as pd

    if intrinsic <= 0 or pd.isna(intrinsic):
        return HitResult(
            hit=False,
            intrinsic=float(intrinsic),
            tolerance=tolerance,
            first_hit_date=None,
            calendar_days_from_anchor=None,
            bar_index_zero_based=None,
            trading_sessions_included=None,
        )
    lo = intrinsic * (1.0 - tolerance)
    hi = intrinsic * (1.0 + tolerance)
    for i, (idx_val, px) in enumerate(prices.items()):
        if pd.isna(px):
            continue
        fv = float(px)
        if lo <= fv <= hi:
            hit_dt = _ensure_date(idx_val)
            return HitResult(
                hit=True,
                intrinsic=float(intrinsic),
                tolerance=tolerance,
                first_hit_date=hit_dt,
                calendar_days_from_anchor=(hit_dt - anchor_date).days,
                bar_index_zero_based=i,
                trading_sessions_included=i + 1,
            )
    return HitResult(
        hit=False,
        intrinsic=float(intrinsic),
        tolerance=tolerance,
        first_hit_date=None,
        calendar_days_from_anchor=None,
        bar_index_zero_based=None,
        trading_sessions_included=None,
    )


def hit_within_horizon(
    hit: HitResult,
    max_trading_sessions: int | None,
    max_calendar_days: int | None,
) -> bool:
    """True if unrestricted, else ``hit`` must satisfy each supplied positive cap (AND).

    If only one cap is supplied, only that cap is checked (typical Basket UI pattern).
    """
    if not hit.hit:
        return False
    if max_trading_sessions is not None and max_trading_sessions > 0:
        if (
            hit.trading_sessions_included is None
            or hit.trading_sessions_included > max_trading_sessions
        ):
            return False
    if max_calendar_days is not None and max_calendar_days > 0:
        if (
            hit.calendar_days_from_anchor is None
            or hit.calendar_days_from_anchor > max_calendar_days
        ):
            return False
    return True


def _base_fcf_anchor(f: Fundamentals, avg_years: int) -> tuple[float, str]:
    if f.ttm_fcf is not None:
        return float(f.ttm_fcf), "TTM"
    val = average_fcf(f.fcf_values, years=avg_years)
    if avg_years <= 1:
        return val, "latest FY (10-K)"
    return val, f"{avg_years}yr FY avg"


def _base_eps_anchor(f: Fundamentals, avg_years: int) -> tuple[float, str]:
    if f.ttm_eps is not None:
        return float(f.ttm_eps), "TTM"
    vals = list(f.eps_values)
    if not vals:
        return 0.0, "n/a"
    yrs = max(1, min(avg_years, len(vals)))
    chunk = vals[-yrs:]
    val = sum(chunk) / len(chunk)
    if avg_years <= 1:
        return val, "latest FY (10-K)"
    return val, f"{avg_years}yr FY avg"


def run_backtest_valuation(
    ticker: str,
    as_of: date,
    assumptions: DCFAssumptions,
    *,
    model_mode: ModelMode = "fcf",
    auto_growth: bool = True,
    auto_growth_mode: AutoGrowthMode = "blended",
    fcf_cagr_window: int | None = 2,
    include_yahoo_consensus: bool = False,
    fcf_avg_years: int = 1,
    match_fcf_for_eps: bool = True,
    manual_growth: float = DEFAULT_GROWTH_RATE,
    eps_growth_override: float | None = None,
) -> BacktestValuation:
    """Fundamentals as-of ``as_of``; growth excludes Yahoo unless opted in."""
    f = get_fundamentals_as_of(ticker, as_of)
    fcf_window = None if fcf_cagr_window == 0 else fcf_cagr_window

    growth_est: GrowthEstimate | None = None
    if auto_growth:
        growth_est = compute_growth_estimate(
            ticker=ticker,
            cik=f.cik,
            fcf_values=f.fcf_values,
            auto_growth_mode=auto_growth_mode,
            fcf_cagr_window=fcf_window,
            include_yahoo_consensus=include_yahoo_consensus,
            as_of=as_of,
            company_facts=None,
            baseline_eps=None,
        )
        growth_fcf = growth_est.growth_rate
    else:
        growth_fcf = manual_growth

    if match_fcf_for_eps:
        growth_eps = growth_fcf
    else:
        growth_eps = (
            float(eps_growth_override)
            if eps_growth_override is not None
            else DEFAULT_GROWTH_RATE
        )

    base_fcf, fcf_lbl = _base_fcf_anchor(f, fcf_avg_years)
    base_eps, eps_lbl = _base_eps_anchor(f, fcf_avg_years)

    iv_fcf = iv_eps = None
    wacc = assumptions.wacc
    tg = assumptions.terminal_growth
    py = assumptions.projection_years
    tp = assumptions.terminal_period_years

    if model_mode in ("fcf", "both") and f.shares_outstanding > 0 and base_fcf != 0:
        if auto_growth:
            r = intrinsic_value_per_share_explicit_decay(
                base_fcf=base_fcf,
                growth_rate=growth_fcf,
                wacc=wacc,
                terminal_growth=tg,
                shares_outstanding=f.shares_outstanding,
                net_debt=f.net_debt,
                projection_years=py,
                terminal_period_years=tp,
            )
        else:
            r = intrinsic_value_per_share(
                base_fcf=base_fcf,
                growth_rate=growth_fcf,
                wacc=wacc,
                terminal_growth=tg,
                shares_outstanding=f.shares_outstanding,
                net_debt=f.net_debt,
                projection_years=py,
                terminal_period_years=tp,
            )
        iv_fcf = float(r["intrinsic_value_per_share"])

    if model_mode in ("eps", "both") and base_eps > 0:
        if auto_growth:
            r_e = intrinsic_price_from_eps_explicit_decay(
                base_eps=base_eps,
                growth_rate=growth_eps,
                wacc=wacc,
                terminal_growth=tg,
                projection_years=py,
                terminal_period_years=tp,
            )
        else:
            r_e = intrinsic_price_from_eps(
                base_eps=base_eps,
                growth_rate=growth_eps,
                wacc=wacc,
                terminal_growth=tg,
                projection_years=py,
                terminal_period_years=tp,
            )
        iv_eps = float(r_e["intrinsic_price_per_share"])

    return BacktestValuation(
        as_of=as_of,
        ticker=ticker.upper(),
        fundamentals=f,
        growth_estimate=growth_est,
        growth_fcf=growth_fcf,
        growth_eps=growth_eps,
        iv_fcf_per_share=iv_fcf,
        iv_eps_per_share=iv_eps,
        base_fcf=base_fcf,
        base_eps=base_eps,
        base_fcf_label=fcf_lbl,
        base_eps_label=eps_lbl,
    )


def first_hit_trading_days(
    prices,
    intrinsic: float,
    *,
    tolerance: float,
    anchor_date: date | None = None,
) -> tuple[bool, int | None]:
    """Backward-compatible `(hit, bar_index_zero_based)`.

    If ``anchor_date`` is omitted, uses the date of the **first row** when hit for
    ``HitResult`` internals is not requested—only index returned.
    """
    if getattr(prices, "empty", False):
        return False, None

    anchor = anchor_date if anchor_date is not None else _ensure_date(prices.index[0])
    hr = first_hit_details(prices, intrinsic, tolerance=tolerance, anchor_date=anchor)
    return hr.hit, hr.bar_index_zero_based
