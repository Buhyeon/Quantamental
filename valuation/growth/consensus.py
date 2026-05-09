"""Yahoo Finance summary fields as consensus-style growth inputs.

Uses ``Ticker.info`` (same heavy fetch as elsewhere when needed). Parsed values
feed ``--auto-growth`` beside FCF CAGR and 8-K regex — not audited SEC data."""

from __future__ import annotations

from typing import Any

import yfinance as yf

from valuation.growth.types import GuidanceCandidate

# Keep raw Yahoo decimals reasonable before blending (final clip remains in blend).
_CONS_CLIP_LO = -0.15
_CONS_CLIP_HI = 0.85


def _first_positive(info: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        v = info.get(k)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > 0:
            return f
    return None


def _clip_frac(x: float) -> float:
    return max(_CONS_CLIP_LO, min(_CONS_CLIP_HI, x))


def _maybe_revenue_growth(info: dict[str, Any]) -> float | None:
    v = info.get("revenueGrowth")
    if v is None:
        return None
    try:
        g = float(v)
    except (TypeError, ValueError):
        return None
    # Yahoo reports YoY revenue growth as a decimal (e.g. 0.25 = +25%).
    if not (-0.5 <= g <= 2.0):
        return None
    return _clip_frac(g)


def yahoo_financial_consensus_candidates(ticker: str) -> list[GuidanceCandidate]:
    """Derived growth hints from yfinance summary: EPS forward vs trailing, revenue YoY."""
    sym = ticker.strip().upper()
    if not sym:
        return []

    try:
        t = yf.Ticker(sym)
        info = getattr(t, "info", None) or {}
    except Exception:
        return []

    if not isinstance(info, dict):
        return []

    out: list[GuidanceCandidate] = []

    fwd = _first_positive(info, ("forwardEps", "epsForward"))
    trail = _first_positive(
        info, ("trailingEps", "epsTrailingTwelveMonths")
    )
    if fwd is not None and trail is not None and trail > 0:
        eps_g = (fwd - trail) / trail
        eps_g = _clip_frac(eps_g)
        out.append(
            GuidanceCandidate(
                growth_rate=eps_g,
                source="financial_consensus",
                confidence="med",
                metric="eps",
                period="yahoo forward vs trailing EPS",
                citation="yfinance ticker.info EPS",
                snippet=f"fwd={fwd:g} trail={trail:g}",
            )
        )

    rev_g = _maybe_revenue_growth(info)
    if rev_g is not None:
        out.append(
            GuidanceCandidate(
                growth_rate=rev_g,
                source="financial_consensus",
                confidence="med",
                metric="revenue",
                period="yahoo revenueGrowth",
                citation="yfinance ticker.info",
                snippet=f"revenueGrowth={rev_g:.2%}",
            )
        )

    return out
