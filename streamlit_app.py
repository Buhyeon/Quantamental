"""Streamlit UI for intrinsic value vs market price (DCF from EDGAR + Yahoo).

Run from repo root:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import altair as alt
import pandas as pd
import streamlit as st

from valuation.config import (
    DCFAssumptions,
    DEFAULT_FCF_AVG_YEARS,
    DEFAULT_GROWTH_RATE,
    DEFAULT_PROJECTION_YEARS,
    DEFAULT_TERMINAL_GROWTH,
    DEFAULT_WACC,
    average_fcf,
)
from valuation.data.edgar import EdgarError, Fundamentals, get_fundamentals
from valuation.data.market import MarketDataError, get_current_price
from valuation.growth.estimate import compute_growth_estimate
from valuation.models.dcf import intrinsic_value_per_share
from valuation.models.eps_dcf import intrinsic_price_from_eps
from valuation.models.sensitivity import intrinsic_sensitivity_grid
from valuation.models.wacc_estimate import estimate_wacc

# Ensure config / dotenv loads on import paths used above.


def _base_fcf(f: Fundamentals, avg_years: int) -> tuple[float, str]:
    if f.ttm_fcf is not None:
        return float(f.ttm_fcf), "TTM"
    val = average_fcf(f.fcf_values, years=avg_years)
    if avg_years <= 1:
        return val, "latest FY (10-K)"
    return val, f"{avg_years}yr FY avg"


def _base_eps(f: Fundamentals, avg_years: int) -> tuple[float, str]:
    if f.ttm_eps is not None:
        return float(f.ttm_eps), "TTM"
    vals = list(f.eps_values)
    yrs = max(1, min(avg_years, len(vals)))
    chunk = vals[-yrs:]
    val = sum(chunk) / len(chunk)
    if avg_years <= 1:
        return val, "latest FY (10-K)"
    return val, f"{avg_years}yr FY avg"


def _growth_sort_weight(source: str) -> float:
    return {
        "fcf_cagr": 2.5,
        "financial_consensus": 2.2,
        "8-K regex": 1.8,
    }.get(source, 1.0)


st.set_page_config(page_title="Quantamental DCF", layout="wide")


st.title("DCF intrinsic value vs market")
st.caption(
    "SEC EDGAR fundamentals + Yahoo Finance. Set `SEC_USER_AGENT` in `.env` "
    "(see README)."
)

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker", value="AAPL").strip().upper() or "AAPL"
    model_mode = st.selectbox("Model", ("fcf", "eps", "both"), index=0)
    auto_growth = st.checkbox("--auto-growth", value=False)
    auto_wacc = st.checkbox("--auto-wacc", value=False)

    terminal = st.slider(
        "Terminal growth", min_value=0.0, max_value=0.05, value=float(DEFAULT_TERMINAL_GROWTH), step=0.0025, format="%.4f")
    projection_years = st.number_input("Explicit years", min_value=1, max_value=15, value=DEFAULT_PROJECTION_YEARS)
    fcf_avg_years = st.number_input("--fcf-avg-years (fallback anchors)", min_value=1, max_value=10, value=DEFAULT_FCF_AVG_YEARS)

    if auto_growth:
        growth_manual = DEFAULT_GROWTH_RATE
        st.info("Growth from 8‑K + Yahoo consensus + FCF CAGR blend.")
    else:
        growth_manual = st.slider("Growth rate (manual)", min_value=-0.2, max_value=0.5, value=float(DEFAULT_GROWTH_RATE), step=0.005, format="%.3f")

    eps_growth = st.slider(
        "--eps-growth override (EPS model only)",
        min_value=-0.20,
        max_value=0.50,
        value=float(DEFAULT_GROWTH_RATE),
        step=0.005,
        format="%.3f",
        help="Only applied when unchecked below.",
    )
    match_fcf_for_eps = st.checkbox("EPS growth = FCF growth", value=True)

    if auto_wacc:
        wacc_val = DEFAULT_WACC
        st.info("WACC from CAPM + book leverage (needs live Yahoo price).")
    else:
        wacc_default = terminal + 0.07
        if wacc_default > 0.5:
            wacc_default = DEFAULT_WACC
        wacc_val = st.slider(
            "WACC (manual)",
            min_value=float(terminal + 0.002),
            max_value=0.30,
            value=float(round(max(wacc_default, terminal + 0.02), 4)),
            step=0.0025,
            format="%.4f",
        )

    show_grid = st.checkbox("Show growth × WACC sensitivity heatmap", value=False)
    grid_steps = st.number_input("Grid steps", min_value=3, max_value=9, value=5)
    grid_g_half = st.slider("Growth half‑width", 0.01, 0.08, 0.03, 0.005)
    grid_w_half = st.slider("WACC half‑width", 0.01, 0.06, 0.02, 0.005)

run = st.button("Run valuation", type="primary")

if not run:
    st.markdown("Configure the sidebar and click **Run valuation**.")
    st.stop()

need_fcf = model_mode in ("fcf", "both")
need_eps = model_mode in ("eps", "both")
growth_eps = growth_manual if match_fcf_for_eps else float(eps_growth)

errors: list[str] = []

try:
    f = get_fundamentals(ticker)
except EdgarError as e:
    st.error(str(e))
    st.stop()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

if auto_growth and not f.fcf_history:
    st.error("--auto-growth requires trailing FCF on EDGAR for this ticker.")
    st.stop()
if need_fcf and not f.fcf_history:
    st.error("No FCF history for this ticker.")
    st.stop()
if need_eps and not f.eps_history:
    st.error("No diluted/basic EPS XBRL for this ticker.")
    st.stop()
if need_fcf or model_mode != "eps":
    if f.shares_outstanding <= 0:
        st.error("Could not determine shares outstanding.")
        st.stop()

growth_est = None

assumptions = DCFAssumptions(
    growth_rate=growth_manual,
    wacc=wacc_val if not auto_wacc else DEFAULT_WACC,
    terminal_growth=float(terminal),
    projection_years=int(projection_years),
)
try:
    assumptions.validate()
except ValueError as e:
    st.error(str(e))
    st.stop()

if auto_growth:
    with st.spinner("Blending growth…"):
        growth_est = compute_growth_estimate(ticker=ticker, cik=f.cik, fcf_values=f.fcf_values)
    growth_fcf = growth_est.growth_rate
    assumptions = replace(assumptions, growth_rate=growth_fcf)
else:
    growth_fcf = growth_manual
if not match_fcf_for_eps:
    growth_eps = float(eps_growth)
else:
    growth_eps = growth_fcf

market_price = None
try:
    market_price = get_current_price(ticker)
except MarketDataError as e:
    errors.append(str(e))

wacc_breakdown = None
if auto_wacc:
    if market_price is None:
        st.error("Auto WACC needs a Yahoo price.")
        st.stop()
    try:
        wacc_breakdown = estimate_wacc(ticker, f, market_price)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    assumptions = replace(assumptions, wacc=wacc_breakdown.wacc)

st.subheader(f"{f.company_name} ({ticker})")

cols = st.columns(3)
cols[0].metric("Shares (used)", f"{f.shares_outstanding:,.0f}")
cols[1].metric("Total debt (book)", f"${f.total_debt/1e9:,.2f}B")
cols[2].metric("Net debt", f"${f.net_debt/1e9:,.2f}B")


if growth_est is not None:
    st.markdown("##### Expected growth blend")
    g = growth_est
    st.caption(f"{g.growth_rate:.2%} — {g.method} ({len(g.candidates)} candidates)")
    cand_rows = []
    for c in sorted(g.candidates, key=lambda x: (-_growth_sort_weight(x.source), x.source)):
        cand_rows.append(
            {
                "Source": c.source,
                "Metric": c.metric,
                "Growth": f"{c.growth_rate:.2%}",
                "Citation": (c.citation or "")[:60],
                "Snippet": (c.snippet or "")[:120],
            }
        )
    if cand_rows:
        st.dataframe(pd.DataFrame(cand_rows), use_container_width=True, hide_index=True)

if wacc_breakdown is not None:
    wb = wacc_breakdown
    st.markdown("##### Auto WACC (CAPM)")
    wcols = st.columns(4)
    wcols[0].metric("WACC (used)", f"{wb.wacc:.2%}")
    wcols[1].metric("Re (CAPM)", f"{wb.cost_of_equity:.2%}")
    wcols[2].metric("Rd pretax", f"{wb.cost_of_debt_pretax:.2%}")
    wcols[3].metric("We / Wd", f"{wb.weight_equity:.0%} / {wb.weight_debt:.0%}")
    with st.expander("WACC detail"):
        st.write(
            {
                "Rf (^TNX)": f"{wb.risk_free:.2%}",
                "Beta": wb.beta,
                "ERP": f"{wb.equity_risk_premium:.2%}",
                "Marginal T": f"{wb.marginal_tax_rate:.1%}",
                "Debt proxy": wb.debt_cost_source,
                "E mkt": wb.market_value_equity,
                "D book": wb.book_value_debt,
            }
        )

fcf_label = eps_label = ""
base_fcf = base_eps = 0.0
if need_fcf:
    base_fcf, fcf_label = _base_fcf(f, int(fcf_avg_years))
    st.markdown(f"**Base FCF** ({fcf_label}): `${base_fcf/1e9:,.3f}B`")
if need_eps:
    base_eps, eps_label = _base_eps(f, int(fcf_avg_years))
    st.markdown(f"**Base EPS** ({eps_label}): `${base_eps:,.3f}`/shr")

dcf_cols = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    if need_fcf:
        r = intrinsic_value_per_share(
            base_fcf=base_fcf,
            growth_rate=growth_fcf,
            wacc=float(assumptions.wacc),
            terminal_growth=float(terminal),
            shares_outstanding=f.shares_outstanding,
            net_debt=f.net_debt,
            projection_years=int(projection_years),
        )
        iv_fcf = float(r["intrinsic_value_per_share"])
        dcf_cols.append(("FCF DCF intrinsic / sh", iv_fcf))
        st.metric("FCF intrinsic / share", f"${iv_fcf:,.2f}")
    else:
        iv_fcf = None
    if need_eps:
        r_eps = intrinsic_price_from_eps(
            base_eps=base_eps,
            growth_rate=float(growth_eps),
            wacc=float(assumptions.wacc),
            terminal_growth=float(terminal),
            projection_years=int(projection_years),
        )
        iv_eps = float(r_eps["intrinsic_price_per_share"])
        dcf_cols.append(("EPS DCF intrinsic / sh", iv_eps))
        st.metric("EPS intrinsic / share", f"${iv_eps:,.2f}")
    else:
        iv_eps = None

consensus_iv = (
    ((iv_fcf or 0) + (iv_eps or 0)) / 2
    if iv_fcf is not None and iv_eps is not None
    else (iv_fcf or iv_eps)
)

if consensus_iv is not None and iv_fcf is not None and iv_eps is not None:
    st.metric("Consensus IV (mean FCF & EPS)", f"${consensus_iv:,.2f}")

if errors:
    for e in errors:
        st.warning(e)

active_iv = consensus_iv
if active_iv is not None and market_price is not None:
    mos = (active_iv - market_price) / market_price
    st.metric("Market price (Yahoo)", f"${market_price:,.2f}", delta=f"MoS vs active IV {mos:+.1%}")

    chart_rows = [{"Label": k, "$ / share": v} for k, v in dcf_cols]
    if iv_fcf is not None and iv_eps is not None:
        chart_rows.append({"Label": "Consensus IV (mean)", "$ / share": float(active_iv)})
    chart_rows.append({"Label": "Market price", "$ / share": float(market_price)})
    df_bar = pd.DataFrame(chart_rows)
    chart = (
        alt.Chart(df_bar)
        .mark_bar()
        .encode(
            x=alt.X("$ / share", title="$ / share"),
            y=alt.Y("Label", sort=None, title=""),
            tooltip=["Label", "$ / share"],
        )
        .properties(height=min(280, 56 * len(df_bar)))
    )
    st.altair_chart(chart, use_container_width=True)

if show_grid and need_fcf:
    st.markdown("##### FCF sensitivity (intrinsic / share)")
    grid = intrinsic_sensitivity_grid(
        base_fcf=base_fcf,
        shares_outstanding=f.shares_outstanding,
        net_debt=f.net_debt,
        projection_years=int(projection_years),
        center_growth=float(growth_fcf),
        center_wacc=float(assumptions.wacc),
        terminal_growth=float(terminal),
        growth_half_width=float(grid_g_half),
        wacc_half_width=float(grid_w_half),
        steps=int(grid_steps),
    )
    hm_data = []
    for i, g in enumerate(grid.growth_axis):
        for j, w in enumerate(grid.wacc_axis):
            v = grid.intrinsic_per_share[i][j]
            if v == v:  # not NaN
                hm_data.append({"Growth": f"{g:.1%}", "WACC": f"{w:.1%}", "IV / sh": v})
    df_h = pd.DataFrame(hm_data)
    if not df_h.empty:
        w_lab = [f"{w:.1%}" for w in grid.wacc_axis]
        g_lab = [f"{g:.1%}" for g in grid.growth_axis]
        heat = (
            alt.Chart(df_h)
            .mark_rect()
            .encode(
                x=alt.X("WACC:N", sort=w_lab, title="WACC"),
                y=alt.Y("Growth:N", sort=g_lab, title="Growth"),
                color=alt.Color("IV / sh", scale=alt.Scale(scheme="viridis")),
                tooltip=["Growth", "WACC", "IV / sh"],
            )
            .properties(width=560, height=360)
        )
        st.altair_chart(heat, use_container_width=True)

st.caption(
    "DCF assumptions: "
    + f"wacc={float(assumptions.wacc):.2%}, terminal={float(terminal):.2%}, "
    + f"growth FCF={float(growth_fcf):.2%}, growth EPS={float(growth_eps):.2%}."
)
