"""CAPM-based WACC using book debt (EDGAR), market cap, Treasury yield, Yahoo beta."""

from __future__ import annotations

from dataclasses import dataclass

import yfinance as yf

from valuation.config import (
    DEFAULT_DEBT_SPREAD_OVER_RF,
    DEFAULT_RISK_FREE_FALLBACK,
    STATUTORY_US_CORP_TAX_RATE,
    WACC_CLIP_HI,
    WACC_CLIP_LO,
    equity_risk_premium,
)
from valuation.data.edgar import Fundamentals


@dataclass(frozen=True)
class WACCBreakdown:
    wacc: float
    risk_free: float
    beta: float
    equity_risk_premium: float
    cost_of_equity: float
    cost_of_debt_pretax: float
    marginal_tax_rate: float
    weight_equity: float
    weight_debt: float
    market_value_equity: float
    book_value_debt: float
    debt_cost_source: str  # "interest_over_debt" | "rf_spread"


def risk_free_10y() -> float:
    """10-year US Treasury yield as annual decimal (^TNX quote is usually in % pts)."""
    try:
        tn = yf.Ticker("^TNX")
        raw = tn.fast_info.get("last_price")
        if raw is None:
            h = tn.history(period="5d")
            if not h.empty:
                raw = float(h["Close"].iloc[-1])
        if raw is None:
            return DEFAULT_RISK_FREE_FALLBACK
        v = float(raw)
        rf = v / 100.0 if v > 1.0 else v
        return max(0.01, min(0.15, rf))
    except Exception:
        return DEFAULT_RISK_FREE_FALLBACK


def equity_beta(ticker: str) -> float:
    """Yahoo 5-year levered beta from ``ticker.info``; default 1.0 if missing."""
    sym = ticker.strip().upper()
    try:
        t = yf.Ticker(sym)
        info = getattr(t, "info", None) or {}
        b = info.get("beta")
        if b is None:
            return 1.0
        bf = float(b)
        if not (-0.5 <= bf <= 3.5):
            return 1.0
        return bf
    except Exception:
        return 1.0


def marginal_tax_rate(f: Fundamentals) -> float:
    """Effective marginal rate for shielding; fallback to US statutory."""
    pt = f.pretax_income
    tax = f.income_tax_expense
    if pt > 1e-6:
        eff = abs(tax / pt)
        return max(0.0, min(0.35, eff))
    return STATUTORY_US_CORP_TAX_RATE


def pretax_cost_of_debt(f: Fundamentals, risk_free: float) -> tuple[float, str]:
    """Book interest expense / book liabilities, else Rf + spread proxy."""
    td = f.total_debt
    if td > 1e-6 and f.interest_expense > 1e-6:
        r = f.interest_expense / td
        r = max(risk_free + 0.005, min(0.30, r))
        return r, "interest_over_debt"
    spread = max(risk_free + DEFAULT_DEBT_SPREAD_OVER_RF, risk_free + 0.005)
    return min(spread, 0.25), "rf_spread"


def estimate_wacc(ticker: str, f: Fundamentals, price_per_share: float) -> WACCBreakdown:
    """WACC = w_e r_e + w_d r_d (1-T) with CAPM for r_e and book debt as D proxy."""
    sh = f.shares_outstanding
    px = float(price_per_share)
    e = max(0.0, sh * px)
    d = max(0.0, f.total_debt)

    if e <= 0:
        raise ValueError("Market value of equity must be positive for WACC.")

    v = e + d
    we = e / v
    wd = d / v

    rf = risk_free_10y()
    erp = equity_risk_premium()
    b = equity_beta(ticker)
    re = rf + b * erp

    rd, rd_src = pretax_cost_of_debt(f, rf)
    t = marginal_tax_rate(f)

    raw_wacc = we * re + wd * rd * (1.0 - t)
    wacc = max(WACC_CLIP_LO, min(WACC_CLIP_HI, raw_wacc))

    return WACCBreakdown(
        wacc=wacc,
        risk_free=rf,
        beta=b,
        equity_risk_premium=erp,
        cost_of_equity=re,
        cost_of_debt_pretax=rd,
        marginal_tax_rate=t,
        weight_equity=we,
        weight_debt=wd,
        market_value_equity=e,
        book_value_debt=d,
        debt_cost_source=rd_src,
    )
