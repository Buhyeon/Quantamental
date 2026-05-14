"""
Streamlit UI for intrinsic value vs market price (DCF from EDGAR + Yahoo; optional FMP for WACC MRP).
Ticker and factor inputs live in the main area; app navigation lists tools in the sidebar.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Literal

import altair as alt
import pandas as pd
import streamlit as st

from valuation.config import (
    DCFAssumptions,
    DEFAULT_FCF_AVG_YEARS,
    DEFAULT_GROWTH_RATE,
    DEFAULT_TERMINAL_GROWTH,
    DEFAULT_TERMINAL_PERIOD_YEARS,
    DEFAULT_WACC,
    average_fcf,
)
from valuation.data.edgar import EdgarError, Fundamentals, fetch_company_facts, get_fundamentals
from valuation.data.market import MarketDataError, get_current_price, get_price_history
from valuation.growth.estimate import AutoGrowthMode, compute_growth_estimate
from valuation.models.dcf import intrinsic_value_per_share, intrinsic_value_per_share_explicit_decay
from valuation.models.eps_dcf import intrinsic_price_from_eps, intrinsic_price_from_eps_explicit_decay
from valuation.models.sensitivity import intrinsic_sensitivity_grid
from valuation.models.wacc_estimate import estimate_wacc

EXPLICIT_HORIZON_YEARS = 8
HIGH_GROWTH_YEARS = 3


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
        "yahoo_analyst_blend": 2.4,
        "yahoo_eps_forward": 2.4,
        "sec_fcf_fy_yoy": 2.3,
        "sec_fcf_ttm_yoy": 2.3,
    }.get(source, 1.0)


_UI_AUTO_GROWTH_LABEL_TO_MODE: dict[str, AutoGrowthMode] = {
    "Mixed (default)": "fcf_sec_yahoo_mixed",
    "Blended FCF": "blended_fcf",
    "EPS": "consensus_only",
}


def _fcf_mode_from_ui(label: str) -> AutoGrowthMode:
    return _UI_AUTO_GROWTH_LABEL_TO_MODE[label]


def _dcf_path_from_fcf_mode(mode: AutoGrowthMode) -> Literal["fcf", "eps"]:
    return "eps" if mode == "consensus_only" else "fcf"


st.set_page_config(
    page_title="Quantamental DCF",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
    .main-header { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: 600; }
    .dcf-compact-section { font-size: 0.95rem; font-weight: 600; margin: 0 0 0.25rem 0; color: inherit; }
    .block-container { padding-top: 1rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.caption("Use the **navigation** control above to open other valuation tools.")

st.title("DCF intrinsic value vs market")

ticker = st.text_input("Ticker", value="AAPL").strip().upper() or "AAPL"

col_growth, col_wacc = st.columns(2, gap="medium")

with col_growth:
    st.markdown('<p class="dcf-compact-section">Growth</p>', unsafe_allow_html=True)
    auto_growth = st.checkbox("Auto-growth", value=True)
    fcf_growth_source = st.radio(
        "Auto-growth option",
        tuple(_UI_AUTO_GROWTH_LABEL_TO_MODE.keys()),
        index=0,
        disabled=not auto_growth,
        horizontal=True,
    )
    fcf_mode = _fcf_mode_from_ui(fcf_growth_source)
    dcf_path = _dcf_path_from_fcf_mode(fcf_mode)
    if auto_growth:
        growth_manual = DEFAULT_GROWTH_RATE
    else:
        growth_manual = st.slider("Manual growth rate", -0.05, 1.0, float(DEFAULT_GROWTH_RATE), 0.005, "%.3f")

    if not auto_growth and dcf_path == "eps":
        eps_growth = st.slider("EPS growth override", -0.05, 1.0, float(DEFAULT_GROWTH_RATE), 0.005, "%.3f")
    else:
        eps_growth = growth_manual

with col_wacc:
    st.markdown('<p class="dcf-compact-section">WACC & terminal</p>', unsafe_allow_html=True)
    auto_wacc = st.checkbox("Auto-WACC", value=True)
    if auto_wacc:
        wacc_val = DEFAULT_WACC
    else:
        wacc_val = st.slider("Manual WACC", 0.02, 0.30, 0.08, 0.0025, "%.4f")

    terminal = st.slider(
        "Terminal growth",
        0.0,
        0.06,
        float(DEFAULT_TERMINAL_GROWTH),
        0.001,
        "%.4f",
        help="Perpetuity + explicit-period decay target.",
    )
    auto_terminal_period = st.checkbox(
        f"Auto terminal period ({DEFAULT_TERMINAL_PERIOD_YEARS}y)", value=True
    )
    if auto_terminal_period:
        terminal_period_years = DEFAULT_TERMINAL_PERIOD_YEARS
    else:
        terminal_period_years = st.number_input(
            "Terminal period (y)",
            0,
            15,
            DEFAULT_TERMINAL_PERIOD_YEARS,
            help="Years at terminal g after explicit phase.",
        )
    st.caption(
        f"Explicit: {EXPLICIT_HORIZON_YEARS}y ({HIGH_GROWTH_YEARS}y high g → fade to terminal)."
    )

with st.expander("Sensitivity", expanded=False):
    sensitivity_enabled = st.checkbox("Enable sensitivity analysis", value=False)
    if sensitivity_enabled:
        show_grid = st.checkbox("Show heatmap", value=False)
        grid_steps = st.number_input("Grid steps", 3, 9, 5)
        grid_g_half = st.slider("Growth half-width", 0.01, 0.08, 0.03, 0.005)
        grid_w_half = st.slider("WACC half-width", 0.01, 0.06, 0.02, 0.005)
    else:
        show_grid = False
        grid_steps = 5
        grid_g_half = 0.03
        grid_w_half = 0.02

run = st.button("Run valuation", type="primary", use_container_width=True)

if not run:
    st.info("Set parameters above, then click **Run valuation**.")
    st.stop()

try:
    with st.spinner("Fetching fundamentals..."):
        f = get_fundamentals(ticker)
except (EdgarError, RuntimeError) as e:
    st.error(f"Error: {e}")
    st.stop()

try:
    facts_json = fetch_company_facts(f.cik)
except EdgarError:
    facts_json = None

assumptions = DCFAssumptions(
    growth_rate=growth_manual,
    wacc=wacc_val,
    terminal_growth=float(terminal),
    projection_years=EXPLICIT_HORIZON_YEARS,
    terminal_period_years=int(terminal_period_years),
)

base_eps_anchor, _ = _base_eps(f, DEFAULT_FCF_AVG_YEARS) if f.eps_values else (None, "")

growth_est_fcf = None
growth_est_eps = None
if auto_growth:
    if dcf_path == "fcf":
        growth_est_fcf = compute_growth_estimate(
            ticker=ticker,
            cik=f.cik,
            fcf_values=f.fcf_values,
            auto_growth_mode=fcf_mode,
            company_facts=facts_json,
            baseline_eps=base_eps_anchor,
        )
        assumptions = replace(assumptions, growth_rate=growth_est_fcf.growth_rate)
        growth_fcf = growth_est_fcf.growth_rate
        growth_eps = growth_manual
    else:
        growth_est_eps = compute_growth_estimate(
            ticker=ticker,
            cik=f.cik,
            fcf_values=f.fcf_values,
            auto_growth_mode=fcf_mode,
            company_facts=facts_json,
            baseline_eps=base_eps_anchor,
        )
        growth_eps = growth_est_eps.growth_rate
        assumptions = replace(assumptions, growth_rate=growth_eps)
        growth_fcf = growth_manual
else:
    growth_fcf = growth_manual
    growth_eps = float(eps_growth) if dcf_path == "eps" else growth_manual
    assumptions = replace(
        assumptions, growth_rate=growth_eps if dcf_path == "eps" else growth_fcf
    )

market_price = None
try:
    market_price = get_current_price(ticker)
except MarketDataError:
    pass

wacc_breakdown = None
if auto_wacc and market_price:
    wacc_breakdown = estimate_wacc(ticker, f, market_price)
    assumptions = replace(assumptions, wacc=wacc_breakdown.wacc)

tab1, tab2, tab3 = st.tabs(["Summary", "Analysis details", "Sensitivity"])

with tab1:
    st.subheader(f"{f.company_name} ({ticker})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shares used", f"{f.shares_outstanding:,.0f}")
    c2.metric("Total debt", f"${f.total_debt/1e9:,.2f}B")
    c3.metric("Net debt", f"${f.net_debt/1e9:,.2f}B")
    c4.metric("Market price", f"${market_price:,.2f}" if market_price else "N/A")

    st.divider()

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.markdown("### Valuation results")
        base_fcf, fcf_label = _base_fcf(f, DEFAULT_FCF_AVG_YEARS)
        base_eps, eps_label = _base_eps(f, DEFAULT_FCF_AVG_YEARS)

        dcf_cols = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if dcf_path == "fcf":
                if auto_growth:
                    r_fcf = intrinsic_value_per_share_explicit_decay(
                        base_fcf=base_fcf,
                        growth_rate=growth_fcf,
                        wacc=float(assumptions.wacc),
                        terminal_growth=float(terminal),
                        shares_outstanding=f.shares_outstanding,
                        net_debt=f.net_debt,
                        projection_years=EXPLICIT_HORIZON_YEARS,
                        high_growth_years=HIGH_GROWTH_YEARS,
                        terminal_period_years=int(terminal_period_years),
                    )
                else:
                    r_fcf = intrinsic_value_per_share(
                        base_fcf=base_fcf,
                        growth_rate=growth_fcf,
                        wacc=float(assumptions.wacc),
                        terminal_growth=float(terminal),
                        shares_outstanding=f.shares_outstanding,
                        net_debt=f.net_debt,
                        projection_years=EXPLICIT_HORIZON_YEARS,
                        terminal_period_years=int(terminal_period_years),
                    )
                iv_fcf = float(r_fcf["intrinsic_value_per_share"])
                st.metric("FCF intrinsic / share", f"${iv_fcf:,.2f}")
                dcf_cols.append(("FCF model", iv_fcf))
                iv_eps = None
            else:
                iv_fcf = None
                if auto_growth:
                    r_eps = intrinsic_price_from_eps_explicit_decay(
                        base_eps=base_eps,
                        growth_rate=float(growth_eps),
                        wacc=float(assumptions.wacc),
                        terminal_growth=float(terminal),
                        projection_years=EXPLICIT_HORIZON_YEARS,
                        high_growth_years=HIGH_GROWTH_YEARS,
                        terminal_period_years=int(terminal_period_years),
                    )
                else:
                    r_eps = intrinsic_price_from_eps(
                        base_eps=base_eps,
                        growth_rate=float(growth_eps),
                        wacc=float(assumptions.wacc),
                        terminal_growth=float(terminal),
                        projection_years=EXPLICIT_HORIZON_YEARS,
                        terminal_period_years=int(terminal_period_years),
                    )
                iv_eps = float(r_eps["intrinsic_price_per_share"])
                st.metric("EPS intrinsic / share", f"${iv_eps:,.2f}")
                dcf_cols.append(("EPS model", iv_eps))

        active_iv = iv_fcf if iv_fcf is not None else iv_eps

    with res_col2:
        if active_iv and market_price:
            mos = (active_iv - market_price) / market_price
            st.metric("Margin of safety", f"{mos:+.1%}")

            chart_data = [{"Label": k, "$": v} for k, v in dcf_cols]
            chart_data.append({"Label": "Market price", "$": float(market_price)})
            df_chart = pd.DataFrame(chart_data)

            chart = (
                alt.Chart(df_chart)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("$:Q", title="Value per share"),
                    y=alt.Y("Label:N", sort="-x", title=None),
                    color=alt.condition(
                        alt.datum.Label == "Market price",
                        alt.value("#E74C3C"),
                        alt.value("#2E86C1"),
                    ),
                )
                .properties(height=200)
            )
            st.altair_chart(chart, use_container_width=True)

    if active_iv is not None:
        st.markdown("### Market price vs intrinsic (1 year)")
        try:
            hist = get_price_history(ticker, period="1y")
        except MarketDataError as exc:
            st.warning(f"Could not load price history: {exc}")
        else:
            df_px = hist[["Close"]].reset_index()
            df_px = df_px.rename(columns={df_px.columns[0]: "Date"})
            mkt = pd.DataFrame(
                {
                    "Date": pd.to_datetime(df_px["Date"]),
                    "Price": df_px["Close"].astype(float),
                    "Series": "Market price",
                }
            )
            d0, d1 = mkt["Date"].min(), mkt["Date"].max()
            iv_line = pd.DataFrame(
                {
                    "Date": [d0, d1],
                    "Price": [float(active_iv), float(active_iv)],
                    "Series": "Intrinsic value (this run)",
                }
            )
            plot_df = pd.concat([mkt, iv_line], ignore_index=True)
            price_chart = (
                alt.Chart(plot_df)
                .mark_line()
                .encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("Price:Q", title="USD / share"),
                    color=alt.Color(
                        "Series:N",
                        title=None,
                        scale=alt.Scale(
                            domain=["Market price", "Intrinsic value (this run)"],
                            range=["#E74C3C", "#2E86C1"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("Date:T", title="Date"),
                        alt.Tooltip("Series:N", title="Series"),
                        alt.Tooltip("Price:Q", title="$/sh", format=",.2f"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(price_chart, use_container_width=True)
            st.caption(
                "Intrinsic value is the model output for your current inputs (flat line); "
                "it is not a point-in-time historical fair value series."
            )

with tab2:
    det_col1, det_col2 = st.columns(2)
    with det_col1:
        st.markdown("#### Growth drivers")
        ge = growth_est_fcf if growth_est_fcf is not None else growth_est_eps
        if ge is not None:
            path_label = "FCF path" if growth_est_fcf is not None else "EPS path"
            st.caption(f"{path_label}: {ge.method}")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Source": c.source,
                            "Rate": f"{c.growth_rate:.2%}",
                            "Snippet": c.snippet,
                        }
                        for c in sorted(
                            ge.candidates,
                            key=lambda x: (-_growth_sort_weight(x.source), x.source),
                        )
                    ]
                ),
                hide_index=True,
            )
    with det_col2:
        st.markdown("#### WACC breakdown")
        if wacc_breakdown:
            wb = wacc_breakdown
            st.write(f"**Cost of equity (CAPM):** {wb.cost_of_equity:.2%}")
            st.write(f"**Cost of debt (pre-tax):** {wb.cost_of_debt_pretax:.2%}")
            st.write(
                f"**Weights (E / D):** {wb.weight_equity:.0%} / {wb.weight_debt:.0%}  "
                f"(D market ${wb.market_value_debt/1e9:,.2f}B)"
            )
            st.write(
                f"**MRP:** {wb.equity_risk_premium:.2%} ({wb.equity_risk_premium_source})  "
                f"**Beta:** {wb.beta}"
            )
            st.caption(f"Debt proxy: {wb.debt_market_proxy_source}")

with tab3:
    if not sensitivity_enabled:
        st.caption("Open the **Sensitivity** section above and enable analysis to configure the heatmap.")
    elif not show_grid:
        st.caption("Turn on **Show heatmap** to view the FCF sensitivity grid.")
    elif show_grid and dcf_path == "fcf":
        st.markdown("#### FCF sensitivity grid")
        grid = intrinsic_sensitivity_grid(
            base_fcf=base_fcf,
            shares_outstanding=f.shares_outstanding,
            net_debt=f.net_debt,
            projection_years=EXPLICIT_HORIZON_YEARS,
            center_growth=float(growth_fcf),
            center_wacc=float(assumptions.wacc),
            terminal_growth=float(terminal),
            growth_half_width=float(grid_g_half),
            wacc_half_width=float(grid_w_half),
            steps=int(grid_steps),
            use_explicit_decay=bool(auto_growth),
            high_growth_years=HIGH_GROWTH_YEARS,
            terminal_period_years=int(terminal_period_years),
        )
        hm_data = [
            {"Growth": f"{g:.1%}", "WACC": f"{w:.1%}", "IV": v}
            for i, g in enumerate(grid.growth_axis)
            for j, w in enumerate(grid.wacc_axis)
            if (v := grid.intrinsic_per_share[i][j]) == v
        ]

        heat = (
            alt.Chart(pd.DataFrame(hm_data))
            .mark_rect()
            .encode(
                x=alt.X("WACC:N", sort=[f"{w:.1%}" for w in grid.wacc_axis]),
                y=alt.Y("Growth:N", sort="descending"),
                color=alt.Color("IV:Q", scale=alt.Scale(scheme="viridis")),
                tooltip=["Growth", "WACC", "IV"],
            )
            .properties(height=450)
        )
        st.altair_chart(heat, use_container_width=True)
    elif show_grid and dcf_path == "eps":
        st.info("Sensitivity heatmap is for the FCF model. Switch Auto-growth to Mixed or Blended FCF to view it.")

st.divider()
_path = (
    f"explicit decay {EXPLICIT_HORIZON_YEARS}y ({HIGH_GROWTH_YEARS}y high → terminal g)"
    if auto_growth
    else "constant explicit growth"
)
_growth_note = (
    f"growth EPS={growth_eps:.2%}"
    if dcf_path == "eps"
    else f"growth FCF={growth_fcf:.2%}"
)
st.caption(
    f"Config: WACC={assumptions.wacc:.2%}, terminal g={terminal:.2%}, "
    f"terminal period={int(terminal_period_years)}y, {_growth_note}, "
    f"path={_path}; data: SEC EDGAR, Yahoo Finance; optional FMP for auto-WACC MRP."
)
