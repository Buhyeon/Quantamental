"""yfinance wrapper for live market data.

Used only for the *noise* side of the comparison: what the market currently quotes,
not what the business is worth. We pin to ``fast_info`` because ``.info`` triggers
a much heavier scrape that breaks frequently when Yahoo changes their HTML.
"""

from __future__ import annotations

from datetime import date, timedelta

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
