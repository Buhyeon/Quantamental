"""
Streamlit: Dashboard — EDGAR historical series plus a Yahoo Finance summary panel.
"""

from __future__ import annotations

import html
from datetime import date, datetime, timezone
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from valuation.data.edgar import EdgarError, Fundamentals, get_fundamentals
from valuation.data.market import (
    MarketDataError,
    fetch_yfinance_info,
    format_compact_usd,
    format_pe_triple,
    format_pct_from_decimal,
    get_current_price,
)


def _fcf_history_to_df(f: Fundamentals, metric_name: str) -> pd.DataFrame:
    hist = f.fcf_history
    if hist:
        return pd.DataFrame(
            [{"Period": str(y), metric_name: float(v)} for y, v in hist]
        )
    vals = f.fcf_values
    if vals:
        return pd.DataFrame(
            [
                {
                    "Period": f"T-{len(vals) - i - 1}" if i < len(vals) - 1 else "Latest FY",
                    metric_name: float(v),
                }
                for i, v in enumerate(vals)
            ]
        )
    return pd.DataFrame()


def _eps_history_to_df(f: Fundamentals, metric_name: str) -> pd.DataFrame:
    hist = f.eps_history
    if hist:
        return pd.DataFrame(
            [{"Period": str(y), metric_name: float(v)} for y, v in hist]
        )
    vals = f.eps_values
    if vals:
        return pd.DataFrame(
            [
                {
                    "Period": f"T-{len(vals) - i - 1}" if i < len(vals) - 1 else "Latest FY",
                    metric_name: float(v),
                }
                for i, v in enumerate(vals)
            ]
        )
    return pd.DataFrame()


def _info_float(info: dict[str, Any], *keys: str) -> float | None:
    for k in keys:
        v = info.get(k)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f:  # not NaN
            return f
    return None


def _forward_eps(info: dict[str, Any]) -> float | None:
    return _info_float(info, "forwardEps", "epsForward")


def _format_ratio(v: float | None, *, digits: int = 2) -> str:
    if v is None:
        return "—"
    if not (0 < abs(v) < 1e6):
        return "—"
    return f"{v:.{digits}f}"


def _format_dividend_date(val: Any) -> str:
    if val is None:
        return "—"
    if isinstance(val, str) and val.strip():
        s = val.strip()
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            try:
                d = date.fromisoformat(s[:10])
                return d.strftime("%d %b %Y")
            except ValueError:
                return s
        return s
    try:
        ts = float(val)
    except (TypeError, ValueError):
        return "—"
    if ts > 1e12:
        ts /= 1000.0
    if ts > 1e9:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%d %b %Y")
    return "—"


def _dash_row(label: str, value: str, sub: str | None = None) -> str:
    sub_html = (
        f'<div class="dash-sub">{html.escape(sub)}</div>'
        if sub
        else ""
    )
    return (
        f'<div class="dash-row">'
        f'<div class="dash-lab">{html.escape(label)}{sub_html}</div>'
        f'<div class="dash-val">{html.escape(value)}</div>'
        f"</div>"
    )


def _dash_card(title: str, inner: str) -> str:
    return (
        f'<div class="dash-card">'
        f'<div class="dash-card-title">{html.escape(title)}</div>'
        f"{inner}"
        f"</div>"
    )


def build_summary_html(
    f: Fundamentals,
    ticker: str,
    price: float | None,
    info: dict[str, Any],
) -> str:
    """Build escaped HTML for the five summary cards (Yahoo + EDGAR mix)."""
    mc = _info_float(info, "marketCap")
    trail_pe = _info_float(info, "trailingPE")
    fwd_pe = _info_float(info, "forwardPE")
    fwd_eps = _forward_eps(info)
    pe_line = format_pe_triple(trail_pe, fwd_pe, price, fwd_eps)
    ps = _info_float(info, "priceToSalesTrailingTwelveMonths", "priceToSales")
    ev_ebitda = _info_float(info, "enterpriseToEbitda")
    pb = _info_float(info, "priceToBook")

    val_inner = (
        _dash_row("Market Cap", format_compact_usd(mc))
        + _dash_row("PE (TTM | Fwd | EPS-implied)", pe_line)
        + _dash_row("Price To Sales", _format_ratio(ps))
        + _dash_row("EV To EBITDA", _format_ratio(ev_ebitda))
        + _dash_row("Price to Book", _format_ratio(pb))
    )

    shares = float(f.shares_outstanding) if f.shares_outstanding and f.shares_outstanding > 0 else 0.0
    ttm_fcf = f.ttm_fcf
    yahoo_fcf = _info_float(info, "freeCashflow")
    yahoo_mc = _info_float(info, "marketCap")
    sbc = _info_float(info, "stockBasedCompensation")

    fcf_yield_pct: float | None = None
    fcf_ps: float | None = None
    fcf_sub: str | None = None
    source_note = ""

    if ttm_fcf is not None and shares > 0 and price and price > 0:
        fcf_ps = float(ttm_fcf) / shares
        fcf_yield_pct = (fcf_ps / price) * 100
        fcf_sub = f"FCF Per Share / Price (${fcf_ps:.2f} / ${price:,.2f})"
    elif yahoo_fcf is not None and yahoo_mc and yahoo_mc > 0:
        fcf_yield_pct = (yahoo_fcf / yahoo_mc) * 100
        fcf_sub = "Free cash flow / market cap (Yahoo TTM proxy)"

    adj_yield_pct: float | None = None
    adj_ps: float | None = None
    adj_sub: str | None = None
    sbc_impact: float | None = None

    raw_total_fcf: float | None = None
    if ttm_fcf is not None:
        raw_total_fcf = float(ttm_fcf)
    elif yahoo_fcf is not None:
        raw_total_fcf = float(yahoo_fcf)

    if raw_total_fcf is not None and sbc is not None and shares > 0 and price and price > 0:
        adj_total = raw_total_fcf - float(sbc)
        adj_ps = adj_total / shares
        adj_yield_pct = (adj_ps / price) * 100
        adj_sub = f"Adj. FCF Per Share / Price (${adj_ps:.2f} / ${price:,.2f})"
        if raw_total_fcf != 0:
            sbc_impact = ((adj_total - raw_total_fcf) / abs(raw_total_fcf)) * 100

    cf_inner = ""
    if fcf_yield_pct is not None:
        cf_inner += _dash_row(
            "Free Cash Flow Yield",
            f"{fcf_yield_pct:.2f}%",
            fcf_sub,
        )
    else:
        cf_inner += _dash_row("Free Cash Flow Yield", "—", None)

    if adj_yield_pct is not None:
        cf_inner += _dash_row(
            "SBC Adj. Free Cash Flow Yield",
            f"{adj_yield_pct:.2f}%",
            adj_sub,
        )
    else:
        cf_inner += _dash_row("SBC Adj. Free Cash Flow Yield", "—", None)

    if sbc_impact is not None:
        cf_inner += _dash_row("SBC Impact", f"{sbc_impact:.2f}%")
    else:
        cf_inner += _dash_row("SBC Impact", "—")

    pm = _info_float(info, "profitMargins")
    om = _info_float(info, "operatingMargins")
    eg = _info_float(info, "earningsQuarterlyGrowth")
    rg = _info_float(info, "revenueGrowth")

    mg_inner = (
        _dash_row("Profit Margin", format_pct_from_decimal(pm))
        + _dash_row("Operating Margin", format_pct_from_decimal(om))
        + _dash_row(
            "Quarterly Earnings (YoY)",
            format_pct_from_decimal(eg),
        )
        + _dash_row(
            "Quarterly Revenue (YoY)",
            format_pct_from_decimal(rg),
        )
    )

    cash_v = float(f.cash)
    debt_v = float(f.total_debt)
    net_v = float(f.net_debt)
    bal_inner = (
        _dash_row("Cash", format_compact_usd(cash_v))
        + _dash_row("Debt", format_compact_usd(debt_v))
        + _dash_row("Net", format_compact_usd(net_v))
    )

    dy = _info_float(info, "dividendYield")
    pr = _info_float(info, "payoutRatio")
    exd = info.get("exDividendDate")
    if exd is None:
        exd = info.get("dividendDate")
    dy_s = format_pct_from_decimal(dy) if dy is not None and dy <= 1 else "—"
    if dy is not None and dy > 1:
        try:
            dy_s = f"{float(dy):.2f}%"
        except (TypeError, ValueError):
            dy_s = "—"
    pr_s = format_pct_from_decimal(pr) if pr is not None else "—"

    div_inner = (
        _dash_row("Dividend Yield", dy_s)
        + _dash_row("Payout Ratio", pr_s)
        + _dash_row("Payout Date", _format_dividend_date(exd))
    )

    top = (
        '<div class="dash-grid-top">'
        f"{_dash_card('Valuation', val_inner)}"
        f"{_dash_card('Cash Flow', cf_inner)}"
        f"{_dash_card('Margins & Growth', mg_inner)}"
        "</div>"
    )
    bot = (
        '<div class="dash-grid-bot">'
        f"{_dash_card('Balance', bal_inner)}"
        f"{_dash_card('Dividend', div_inner)}"
        "</div>"
    )
    return f'<div class="dash-summary-wrap">{top}{bot}</div>'


@st.cache_data(ttl=900)
def _cached_yahoo_info(ticker: str) -> dict[str, Any]:
    return fetch_yfinance_info(ticker)


st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.5rem; }
    .dash-summary-wrap { margin: 0 0 1.25rem 0; }
    .dash-grid-top {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .dash-grid-bot {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
        max-width: calc(66.666% + 0.5rem);
    }
    @media (max-width: 1100px) {
        .dash-grid-top { grid-template-columns: 1fr; }
        .dash-grid-bot { grid-template-columns: 1fr; max-width: 100%; }
    }
    .dash-card {
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 8px;
        padding: 0.75rem 1rem 0.5rem 1rem;
    }
    .dash-card-title {
        font-weight: 700;
        font-size: 1rem;
        margin: 0 0 0.5rem 0;
        padding-bottom: 0.35rem;
        border-bottom: 1px dotted rgba(148, 163, 184, 0.55);
    }
    .dash-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.35rem 0;
        border-bottom: 1px dotted rgba(148, 163, 184, 0.35);
        font-size: 0.92rem;
    }
    .dash-row:last-child { border-bottom: none; }
    .dash-lab { color: inherit; text-align: left; flex: 1; min-width: 0; }
    .dash-val { font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }
    .dash-sub { font-size: 0.78rem; opacity: 0.82; margin-top: 0.2rem; line-height: 1.25; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">Dashboard</div>', unsafe_allow_html=True)
st.caption(
    "Historical **FCF** and **EPS** charts from SEC EDGAR (10-K / 10-Q). "
    "Summary cards use **Yahoo Finance** live summary fields where available (valuation, margins, dividend); "
    "**Balance** cash/debt is from the latest EDGAR balance sheet extraction."
)

with st.sidebar:
    st.title("Dashboard Settings")
    ticker = st.text_input("Ticker Symbol", value="AAPL").strip().upper() or "AAPL"
    run = st.button("Load History", type="primary", use_container_width=True)

if not run:
    st.info("Enter a ticker and click Load History.")
    st.stop()

try:
    with st.spinner(f"Loading {ticker} (EDGAR + market data)..."):
        f = get_fundamentals(ticker)
        info = _cached_yahoo_info(ticker)
        try:
            px = get_current_price(ticker)
        except MarketDataError:
            px = None
except (EdgarError, RuntimeError) as e:
    st.error(f"Data Retrieval Error: {e}")
    st.stop()

if not info:
    st.warning(
        "Yahoo Finance summary (`info`) was empty or unavailable — valuation, margins, "
        "and dividend rows may show dashes. EDGAR charts and balance sheet figures still load."
    )

st.markdown(build_summary_html(f, ticker, px, info), unsafe_allow_html=True)

df_fcf = _fcf_history_to_df(f, "Free Cash Flow")
df_eps = _eps_history_to_df(f, "EPS")

st.subheader(f"{f.company_name} ({ticker})")

st.divider()

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("#### Free Cash Flow History")
    if not df_fcf.empty:
        df_fcf["FCF (Billions)"] = df_fcf["Free Cash Flow"] / 1e9

        fcf_chart = alt.Chart(df_fcf).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Period:O", sort=None, title="Fiscal Period"),
            y=alt.Y("FCF (Billions):Q", title="USD (Billions)"),
            color=alt.condition(
                alt.datum["FCF (Billions)"] > 0,
                alt.value("#27AE60"),
                alt.value("#E74C3C"),
            ),
            tooltip=["Period", "Free Cash Flow"],
        ).properties(height=350)
        st.altair_chart(fcf_chart, use_container_width=True)
    else:
        st.warning("No FCF history available.")

with col_chart2:
    st.markdown("#### Earnings Per Share (Diluted)")
    if not df_eps.empty:
        eps_chart = alt.Chart(df_eps).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("Period:O", sort=None, title="Fiscal Period"),
            y=alt.Y("EPS:Q", title="USD per Share"),
            color=alt.value("#2E86C1"),
            tooltip=["Period", "EPS"],
        ).properties(height=350)
        st.altair_chart(eps_chart, use_container_width=True)
    else:
        st.warning("No EPS history available.")
