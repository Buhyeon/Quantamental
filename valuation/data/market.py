"""yfinance wrapper for live market data.

Used only for the *noise* side of the comparison: what the market currently quotes,
not what the business is worth. We pin to ``fast_info`` because ``.info`` triggers
a much heavier scrape that breaks frequently when Yahoo changes their HTML.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
import yfinance as yf


class MarketDataError(RuntimeError):
    pass


def get_current_price(ticker: str) -> float:
    """Return the latest available close/last price for ``ticker``."""
    t = yf.Ticker(ticker)
    try:
        price = t.fast_info.get("last_price")
    except Exception as exc:
        raise MarketDataError(f"yfinance fast_info failed for {ticker!r}: {exc}") from exc

    if price is None:
        # Fallback to the last bar of the daily history.
        hist = t.history(period="5d")
        if hist.empty:
            raise MarketDataError(f"No price data available for {ticker!r}.")
        price = float(hist["Close"].iloc[-1])
    return float(price)


def try_yfinance_shares_outstanding(ticker: str) -> float | None:
    """Best-effort diluted / common shares from Yahoo ``fast_info`` (lightweight).

    Used only when SEC EDGAR does not expose a usable share count. Avoids calling
    ``.info``, which triggers a heavy HTML scrape that breaks frequently.
    """
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info
    except Exception:
        return None

    raw_vals: list[object | None] = []
    getter = getattr(fi, "get", None)
    if callable(getter):
        for k in ("shares_outstanding", "shares", "impliedSharesOutstanding"):
            raw_vals.append(getter(k))
    for attr in ("shares_outstanding", "shares", "impliedSharesOutstanding"):
        try:
            raw_vals.append(getattr(fi, attr))
        except Exception:
            raw_vals.append(None)

    for raw in raw_vals:
        if raw is None:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if v > 0:
            return v
    return None


def fetch_yfinance_info(ticker: str) -> dict[str, Any]:
    """Return Yahoo Finance ``Ticker.info`` summary dict, or ``{}`` on failure.

    This triggers yfinance's heavier HTML-backed scrape (same tradeoff as
    consensus/WACC paths). Prefer caching at the call site when appropriate.
    """
    sym = ticker.strip().upper()
    if not sym:
        return {}
    try:
        t = yf.Ticker(sym)
        info = getattr(t, "info", None) or {}
    except Exception:
        return {}
    return info if isinstance(info, dict) else {}


def format_compact_usd(x: float | None) -> str:
    """Format USD with ``t`` / ``b`` / ``m`` / ``k`` suffixes (e.g. ``$4.39t``)."""
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    sign = "-" if v < 0 else ""
    ax = abs(v)
    if ax >= 1e12:
        return f"{sign}${ax / 1e12:.2f}t"
    if ax >= 1e9:
        return f"{sign}${ax / 1e9:.2f}b"
    if ax >= 1e6:
        return f"{sign}${ax / 1e6:.2f}m"
    if ax >= 1e3:
        return f"{sign}${ax / 1e3:.2f}k"
    return f"{sign}${ax:.2f}"


def format_pct_from_decimal(x: float | None, *, digits: int = 2) -> str:
    """Format Yahoo-style ratio stored as decimal (e.g. ``0.25`` -> ``25.00%``)."""
    if x is None:
        return "—"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return "—"
    return f"{f * 100:.{digits}f}%"


def format_pe_triple(
    trailing_pe: float | None,
    forward_pe: float | None,
    price: float | None,
    forward_eps: float | None,
    *,
    diff_threshold: float = 0.5,
) -> str:
    """``TTM | Fwd | alt`` for display; third slot is EPS-implied P/E when it differs from Yahoo forward P/E."""

    def _one(v: float | None) -> str:
        if v is None:
            return "—"
        try:
            f = float(v)
        except (TypeError, ValueError):
            return "—"
        if not (0 < f < 1e4):
            return "—"
        return f"{f:.2f}"

    seg1 = _one(trailing_pe)
    seg2 = _one(forward_pe)
    seg3 = "—"
    if price is not None and forward_eps is not None:
        try:
            px = float(price)
            eps = float(forward_eps)
        except (TypeError, ValueError):
            px, eps = 0.0, 0.0
        if px > 0 and eps > 0:
            calc = px / eps
            if 0 < calc < 1e4:
                if forward_pe is None:
                    seg3 = f"{calc:.2f}"
                else:
                    try:
                        fp = float(forward_pe)
                    except (TypeError, ValueError):
                        seg3 = f"{calc:.2f}"
                    else:
                        seg3 = f"{calc:.2f}" if abs(calc - fp) >= diff_threshold else "—"
    return f"{seg1} | {seg2} | {seg3}"


def get_price_history(ticker: str, period: str = "5y"):
    """Return a DataFrame of daily OHLCV bars for the requested period."""
    t = yf.Ticker(ticker)
    hist = t.history(period=period)
    if hist.empty:
        raise MarketDataError(f"No history available for {ticker!r} over {period}.")
    return hist


def get_price_history_daterange(ticker: str, start: date, end: date) -> pd.Series:
    """Daily close indexed by ``datetime.date``, ``start`` through ``end`` inclusive."""
    if end < start:
        raise MarketDataError("end must be >= start")
    t = yf.Ticker(ticker)
    hist = t.history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat())
    if hist.empty:
        raise MarketDataError(
            f"No history for {ticker!r} from {start} to {end}."
        )
    s = hist["Close"].copy()
    s.index = pd.Index([pd.Timestamp(ts).date() for ts in s.index])
    return s
