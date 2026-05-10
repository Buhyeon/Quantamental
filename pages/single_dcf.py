"""
Streamlit UI for intrinsic value vs market price (DCF from EDGAR + Yahoo).
Aesthetic version with restored OG logo and no emojis.
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
from valuation.growth.estimate import AutoGrowthMode, compute_growth_estimate
from valuation.models.dcf import intrinsic_value_per_share, intrinsic_value_per_share_explicit_decay
from valuation.models.eps_dcf import intrinsic_price_from_eps, intrinsic_price_from_eps_explicit_decay
from valuation.models.sensitivity import intrinsic_sensitivity_grid
from valuation.models.wacc_estimate import estimate_wacc

# --- Helper Functions ---
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

# --- Page Config ---
st.set_page_config(
    page_title="Quantamental DCF", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for OG logo styling and clean typography
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
    .main-header { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("Quantamental DCF")
    st.caption("SEC EDGAR fundamentals + Yahoo Finance")
    
    ticker = st.text_input("Ticker", value="AAPL").strip().upper() or "AAPL"
    model_mode = st.selectbox("Model", ("fcf", "eps", "both"), index=0)
    
    with st.expander("Growth", expanded=True):
        auto_growth = st.checkbox("Auto-growth", value=True)
        auto_mode_label = st.radio(
            "Auto-growth source",
            ("Blended (FCF CAGR + 8-K + Yahoo)", "Consensus only (Yahoo)"),
            index=0,
            disabled=not auto_growth,
        )
        auto_growth_mode: AutoGrowthMode = (
            "consensus_only"
            if auto_mode_label.startswith("Consensus")
            else "blended"
        )
        fcf_cagr_fy_window = st.number_input(
            "FCF CAGR FY window (blended)",
            min_value=0,
            max_value=30,
            value=2,
            disabled=not auto_growth or auto_growth_mode == "consensus_only",
            help="Number of trailing fiscal years for FCF CAGR. 0 = full EDGAR history.",
        )
        if auto_growth:
            growth_manual = DEFAULT_GROWTH_RATE
        else:
            growth_manual = st.slider("Manual Rate", -0.2, 0.5, float(DEFAULT_GROWTH_RATE), 0.005, "%.3f")
        
        match_fcf_for_eps = st.checkbox("Match EPS to FCF", value=True)
        if not match_fcf_for_eps:
            eps_growth = st.slider("EPS Growth Override", -0.2, 0.5, float(DEFAULT_GROWTH_RATE), 0.005, "%.3f")
        else:
            eps_growth = growth_manual

    with st.expander("WACC & Terminal", expanded=True):
        auto_wacc = st.checkbox("Auto-WACC", value=True)
        if auto_wacc:
            wacc_val = DEFAULT_WACC
        else:
            wacc_val = st.slider("Manual WACC", 0.02, 0.30, 0.08, 0.0025, "%.4f")
        
        terminal = st.slider(
            "Terminal growth (perpetuity + decay target)",
            0.0,
            0.06,
            float(DEFAULT_TERMINAL_GROWTH),
            0.001,
            "%.4f",
        )
        projection_years = st.number_input("Projection Years", 1, 15, DEFAULT_PROJECTION_YEARS)

    with st.expander("Sensitivity Settings"):
        show_grid = st.checkbox("Show Heatmap", value=True)
        grid_steps = st.number_input("Grid Steps", 3, 9, 5)
        grid_g_half = st.slider("Growth Half-width", 0.01, 0.08, 0.03, 0.005)
        grid_w_half = st.slider("WACC Half-width", 0.01, 0.06, 0.02, 0.005)

    fcf_avg_years = st.number_input("FCF Avg Years", 1, 10, DEFAULT_FCF_AVG_YEARS)
    run = st.button("Run Valuation", type="primary", use_container_width=True)

# --- Main App ---
st.title("DCF intrinsic value vs market")

if not run:
    st.info("Set parameters in the sidebar and click Run Valuation.")
    st.stop()

# --- Data Processing ---
try:
    with st.spinner("Fetching fundamentals..."):
        f = get_fundamentals(ticker)
except (EdgarError, RuntimeError) as e:
    st.error(f"Error: {e}")
    st.stop()

assumptions = DCFAssumptions(
    growth_rate=growth_manual,
    wacc=wacc_val,
    terminal_growth=float(terminal),
    projection_years=int(projection_years),
)

growth_est = None
fcf_cagr_window = None if int(fcf_cagr_fy_window) == 0 else int(fcf_cagr_fy_window)
if auto_growth:
    growth_est = compute_growth_estimate(
        ticker=ticker,
        cik=f.cik,
        fcf_values=f.fcf_values,
        auto_growth_mode=auto_growth_mode,
        fcf_cagr_window=fcf_cagr_window,
    )
    growth_fcf = growth_est.growth_rate
    assumptions = replace(assumptions, growth_rate=growth_fcf)
else:
    growth_fcf = growth_manual

growth_eps = growth_fcf if match_fcf_for_eps else float(eps_growth)

market_price = None
try:
    market_price = get_current_price(ticker)
except MarketDataError:
    pass

wacc_breakdown = None
if auto_wacc and market_price:
    wacc_breakdown = estimate_wacc(ticker, f, market_price)
    assumptions = replace(assumptions, wacc=wacc_breakdown.wacc)

# --- Layout ---
tab1, tab2, tab3 = st.tabs(["Summary", "Analysis Details", "Sensitivity"])

with tab1:
    st.subheader(f"{f.company_name} ({ticker})")
    
    # Financial Snapshot
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shares Used", f"{f.shares_outstanding:,.0f}")
    c2.metric("Total Debt", f"${f.total_debt/1e9:,.2f}B")
    c3.metric("Net Debt", f"${f.net_debt/1e9:,.2f}B")
    c4.metric("Market Price", f"${market_price:,.2f}" if market_price else "N/A")

    st.divider()

    # Results
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.markdown("### Valuation Results")
        base_fcf, fcf_label = _base_fcf(f, int(fcf_avg_years))
        base_eps, eps_label = _base_eps(f, int(fcf_avg_years))

        dcf_cols = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if model_mode in ("fcf", "both"):
                if auto_growth:
                    r_fcf = intrinsic_value_per_share_explicit_decay(
                        base_fcf=base_fcf,
                        growth_rate=growth_fcf,
                        wacc=float(assumptions.wacc),
                        terminal_growth=float(terminal),
                        shares_outstanding=f.shares_outstanding,
                        net_debt=f.net_debt,
                        projection_years=int(projection_years),
                    )
                else:
                    r_fcf = intrinsic_value_per_share(
                        base_fcf=base_fcf, growth_rate=growth_fcf, wacc=float(assumptions.wacc),
                        terminal_growth=float(terminal), shares_outstanding=f.shares_outstanding,
                        net_debt=f.net_debt, projection_years=int(projection_years)
                    )
                iv_fcf = float(r_fcf["intrinsic_value_per_share"])
                st.metric("FCF Intrinsic / Share", f"${iv_fcf:,.2f}")
                dcf_cols.append(("FCF Model", iv_fcf))
            else: iv_fcf = None

            if model_mode in ("eps", "both"):
                if auto_growth:
                    r_eps = intrinsic_price_from_eps_explicit_decay(
                        base_eps=base_eps,
                        growth_rate=float(growth_eps),
                        wacc=float(assumptions.wacc),
                        terminal_growth=float(terminal),
                        projection_years=int(projection_years),
                    )
                else:
                    r_eps = intrinsic_price_from_eps(
                        base_eps=base_eps, growth_rate=float(growth_eps), wacc=float(assumptions.wacc),
                        terminal_growth=float(terminal), projection_years=int(projection_years)
                    )
                iv_eps = float(r_eps["intrinsic_price_per_share"])
                st.metric("EPS Intrinsic / Share", f"${iv_eps:,.2f}")
                dcf_cols.append(("EPS Model", iv_eps))
            else: iv_eps = None

        active_iv = ((iv_fcf or 0) + (iv_eps or 0)) / ((1 if iv_fcf else 0) + (1 if iv_eps else 0))
        if iv_fcf and iv_eps:
            st.metric("Consensus IV", f"${active_iv:,.2f}")

    with res_col2:
        if active_iv and market_price:
            mos = (active_iv - market_price) / market_price
            st.metric("Margin of Safety", f"{mos:+.1%}")
            
            # Comparison Chart
            chart_data = [{"Label": k, "$": v} for k, v in dcf_cols]
            chart_data.append({"Label": "Market Price", "$": float(market_price)})
            df_chart = pd.DataFrame(chart_data)
            
            chart = alt.Chart(df_chart).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("$:Q", title="Value per Share"),
                y=alt.Y("Label:N", sort="-x", title=None),
                color=alt.condition(
                    alt.datum.Label == "Market Price",
                    alt.value("#E74C3C"), alt.value("#2E86C1")
                )
            ).properties(height=200)
            st.altair_chart(chart, use_container_width=True)

with tab2:
    det_col1, det_col2 = st.columns(2)
    with det_col1:
        st.markdown("#### Growth Drivers")
        if growth_est:
            st.caption(f"Blended via {growth_est.method}")
            st.dataframe(pd.DataFrame([{
                "Source": c.source, 
                "Rate": f"{c.growth_rate:.2%}", 
                "Snippet": c.snippet
            } for c in growth_est.candidates]), hide_index=True)
            
    with det_col2:
        st.markdown("#### WACC Breakdown")
        if wacc_breakdown:
            wb = wacc_breakdown
            st.write(f"**Cost of Equity (CAPM):** {wb.cost_of_equity:.2%}")
            st.write(f"**Cost of Debt:** {wb.cost_of_debt_pretax:.2%}")
            st.write(f"**Weights (E/D):** {wb.weight_equity:.0%} / {wb.weight_debt:.0%}")
            st.write(f"**Beta:** {wb.beta}")

with tab3:
    if show_grid and model_mode in ("fcf", "both"):
        st.markdown("#### FCF Sensitivity Grid")
        grid = intrinsic_sensitivity_grid(
            base_fcf=base_fcf, shares_outstanding=f.shares_outstanding,
            net_debt=f.net_debt, projection_years=int(projection_years),
            center_growth=float(growth_fcf), center_wacc=float(assumptions.wacc),
            terminal_growth=float(terminal), growth_half_width=float(grid_g_half),
            wacc_half_width=float(grid_w_half), steps=int(grid_steps),
            use_explicit_decay=bool(auto_growth),
        )
        hm_data = [{"Growth": f"{g:.1%}", "WACC": f"{w:.1%}", "IV": v}
                   for i, g in enumerate(grid.growth_axis)
                   for j, w in enumerate(grid.wacc_axis)
                   if (v := grid.intrinsic_per_share[i][j]) == v]
        
        heat = alt.Chart(pd.DataFrame(hm_data)).mark_rect().encode(
            x=alt.X("WACC:N", sort=[f"{w:.1%}" for w in grid.wacc_axis]),
            y=alt.Y("Growth:N", sort="descending"),
            color=alt.Color("IV:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=["Growth", "WACC", "IV"]
        ).properties(height=450)
        st.altair_chart(heat, use_container_width=True)

st.divider()
_path = "explicit decay (3y high → terminal)" if auto_growth else "constant explicit growth"
st.caption(
    f"Config: WACC={assumptions.wacc:.2%}, Terminal={terminal:.2%}, "
    f"Growth FCF={growth_fcf:.2%}, explicit path={_path}"
)