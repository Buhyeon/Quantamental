"""
Streamlit: multi-ticker DCF screener (FCF or EPS path, same growth/WACC model as Single Stock DCF).
"""

from __future__ import annotations

import time
from typing import Literal

import pandas as pd
import streamlit as st

from valuation.config import (
    DEFAULT_FCF_AVG_YEARS,
    DEFAULT_GROWTH_RATE,
    DEFAULT_PROJECTION_YEARS,
    DEFAULT_TERMINAL_GROWTH,
    DEFAULT_TERMINAL_PERIOD_YEARS,
    DEFAULT_WACC,
    average_fcf,
)
from valuation.data.edgar import EdgarError, Fundamentals, fetch_company_facts, get_fundamentals
from valuation.data.market import MarketDataError, get_current_price
from valuation.growth.estimate import AutoGrowthMode, compute_growth_estimate
from valuation.models.dcf import intrinsic_value_per_share, intrinsic_value_per_share_explicit_decay
from valuation.models.eps_dcf import intrinsic_price_from_eps, intrinsic_price_from_eps_explicit_decay
from valuation.models.wacc_estimate import estimate_wacc

HIGH_GROWTH_YEARS = 3

_UI_AUTO_GROWTH_LABEL_TO_MODE: dict[str, AutoGrowthMode] = {
    "Mixed (default)": "fcf_sec_yahoo_mixed",
    "Blended FCF": "blended_fcf",
    "EPS": "consensus_only",
}


def _fcf_mode_from_ui(label: str) -> AutoGrowthMode:
    return _UI_AUTO_GROWTH_LABEL_TO_MODE[label]


def _dcf_path_from_fcf_mode(mode: AutoGrowthMode) -> Literal["fcf", "eps"]:
    return "eps" if mode == "consensus_only" else "fcf"


def _base_fcf(f: Fundamentals, avg_years: int) -> float:
    if f.ttm_fcf is not None:
        return float(f.ttm_fcf)
    return average_fcf(f.fcf_values, years=avg_years)


def _base_eps(f: Fundamentals, avg_years: int) -> float:
    if f.ttm_eps is not None:
        return float(f.ttm_eps)
    vals = list(f.eps_values)
    if not vals:
        raise ValueError("No EPS history for EPS DCF.")
    yrs = max(1, min(avg_years, len(vals)))
    chunk = vals[-yrs:]
    return sum(chunk) / len(chunk)


st.set_page_config(page_title="Quantamental Screener", layout="wide")

st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.5rem; }
    .dcf-compact-section { font-size: 0.95rem; font-weight: 600; margin: 0 0 0.25rem 0; color: inherit; }
    .block-container { padding-top: 1rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.caption("Use the **navigation** control above to open other valuation tools.")

st.markdown('<div class="main-header">Screener</div>', unsafe_allow_html=True)
st.caption("FCF or EPS intrinsic value vs market for multiple tickers (same DCF setup as Single Stock DCF).")

ticker_input = st.text_area(
    "Tickers (comma-separated)",
    value="AAPL, MSFT, GOOGL, META, AMZN, TSLA, NVDA",
    height=120,
    help="Paste tickers separated by commas. Large lists may take several minutes.",
)

col_growth, col_wacc = st.columns(2, gap="medium")

with col_growth:
    st.markdown('<p class="dcf-compact-section">Growth</p>', unsafe_allow_html=True)
    auto_growth = st.checkbox("Auto-growth", value=True, key="scr_auto_growth")
    fcf_growth_source = st.radio(
        "Auto-growth option",
        tuple(_UI_AUTO_GROWTH_LABEL_TO_MODE.keys()),
        index=0,
        disabled=not auto_growth,
        horizontal=True,
        key="scr_growth_src",
    )
    fcf_mode = _fcf_mode_from_ui(fcf_growth_source)
    dcf_path = _dcf_path_from_fcf_mode(fcf_mode)
    if auto_growth:
        growth_manual = DEFAULT_GROWTH_RATE
    else:
        growth_manual = st.slider(
            "Manual growth rate",
            -0.05,
            1.0,
            float(DEFAULT_GROWTH_RATE),
            0.005,
            "%.3f",
            key="scr_growth_manual",
        )
    if not auto_growth and dcf_path == "eps":
        eps_growth = st.slider(
            "EPS growth override",
            -0.05,
            1.0,
            float(DEFAULT_GROWTH_RATE),
            0.005,
            "%.3f",
            key="scr_eps_manual",
        )
    else:
        eps_growth = growth_manual

with col_wacc:
    st.markdown('<p class="dcf-compact-section">WACC & terminal</p>', unsafe_allow_html=True)
    auto_wacc = st.checkbox("Auto-WACC", value=True, key="scr_auto_wacc")
    if auto_wacc:
        wacc_val = DEFAULT_WACC
    else:
        wacc_val = st.slider(
            "Manual WACC",
            0.02,
            0.30,
            float(DEFAULT_WACC),
            0.0025,
            "%.4f",
            key="scr_wacc_manual",
        )
    terminal = st.slider(
        "Terminal growth",
        0.0,
        0.06,
        float(DEFAULT_TERMINAL_GROWTH),
        0.001,
        "%.4f",
        help="Perpetuity + explicit-period decay target.",
        key="scr_terminal",
    )
    auto_terminal_period = st.checkbox(
        f"Auto terminal period ({DEFAULT_TERMINAL_PERIOD_YEARS}y)",
        value=True,
        key="scr_auto_term",
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
            key="scr_term_years",
        )
    projection_years = st.number_input(
        "Explicit horizon (years)",
        1,
        15,
        int(DEFAULT_PROJECTION_YEARS),
        help="Length of explicit forecast (high-growth + fade), same role as Single Stock DCF.",
        key="scr_proj_years",
    )
    st.caption(f"High-growth phase: {HIGH_GROWTH_YEARS}y (then linear fade to terminal g).")

run_screener = st.button("Run screener", type="primary", use_container_width=True)

if not run_screener:
    st.info("Enter tickers and parameters, then click **Run screener**.")
    st.stop()

tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
if not tickers:
    st.warning("Please enter at least one ticker.")
    st.stop()

results: list[dict] = []
my_bar = st.progress(0, text="Scanning…")
total = len(tickers)

for i, ticker in enumerate(tickers):
    my_bar.progress((i + 1) / total, text=f"Processing {ticker} ({i + 1}/{total})")
    try:
        f = get_fundamentals(ticker)
        try:
            facts_json = fetch_company_facts(f.cik)
        except EdgarError:
            facts_json = None

        market_price: float | None = None
        try:
            market_price = get_current_price(ticker)
        except MarketDataError:
            market_price = None

        wacc_use = float(wacc_val)
        if auto_wacc and market_price is not None:
            try:
                wbd = estimate_wacc(ticker, f, market_price)
                wacc_use = float(wbd.wacc)
            except Exception:
                pass

        base_eps_anchor: float | None = None
        if f.eps_values or f.ttm_eps is not None:
            try:
                base_eps_anchor = _base_eps(f, DEFAULT_FCF_AVG_YEARS)
            except ValueError:
                base_eps_anchor = None

        growth_fcf = growth_manual
        growth_eps = float(eps_growth)
        if auto_growth:
            if dcf_path == "fcf":
                ge = compute_growth_estimate(
                    ticker=ticker,
                    cik=f.cik,
                    fcf_values=f.fcf_values,
                    auto_growth_mode=fcf_mode,
                    company_facts=facts_json,
                    baseline_eps=base_eps_anchor,
                )
                growth_fcf = ge.growth_rate
            else:
                ge = compute_growth_estimate(
                    ticker=ticker,
                    cik=f.cik,
                    fcf_values=f.fcf_values,
                    auto_growth_mode=fcf_mode,
                    company_facts=facts_json,
                    baseline_eps=base_eps_anchor,
                )
                growth_eps = ge.growth_rate

        terminal_f = float(terminal)
        term_y = int(terminal_period_years)
        proj_y = int(projection_years)

        if dcf_path == "fcf":
            base_fcf = _base_fcf(f, DEFAULT_FCF_AVG_YEARS)
            if auto_growth:
                r = intrinsic_value_per_share_explicit_decay(
                    base_fcf=base_fcf,
                    growth_rate=growth_fcf,
                    wacc=wacc_use,
                    terminal_growth=terminal_f,
                    shares_outstanding=f.shares_outstanding,
                    net_debt=f.net_debt,
                    projection_years=proj_y,
                    high_growth_years=HIGH_GROWTH_YEARS,
                    terminal_period_years=term_y,
                )
            else:
                r = intrinsic_value_per_share(
                    base_fcf=base_fcf,
                    growth_rate=growth_fcf,
                    wacc=wacc_use,
                    terminal_growth=terminal_f,
                    shares_outstanding=f.shares_outstanding,
                    net_debt=f.net_debt,
                    projection_years=proj_y,
                    terminal_period_years=term_y,
                )
            iv = float(r["intrinsic_value_per_share"])
            model_lbl = "FCF"
        else:
            base_eps = _base_eps(f, DEFAULT_FCF_AVG_YEARS)
            if auto_growth:
                r = intrinsic_price_from_eps_explicit_decay(
                    base_eps=base_eps,
                    growth_rate=float(growth_eps),
                    wacc=wacc_use,
                    terminal_growth=terminal_f,
                    projection_years=proj_y,
                    high_growth_years=HIGH_GROWTH_YEARS,
                    terminal_period_years=term_y,
                )
            else:
                r = intrinsic_price_from_eps(
                    base_eps=base_eps,
                    growth_rate=float(growth_eps),
                    wacc=wacc_use,
                    terminal_growth=terminal_f,
                    projection_years=proj_y,
                    terminal_period_years=term_y,
                )
            iv = float(r["intrinsic_price_per_share"])
            model_lbl = "EPS"

        if f.shares_outstanding <= 0:
            raise ValueError("Invalid shares outstanding.")

        if market_price and market_price > 0:
            mos = (iv - market_price) / market_price
        else:
            mos = None

        results.append(
            {
                "Ticker": ticker,
                "Company Name": f.company_name,
                "Model": model_lbl,
                "Intrinsic / share": iv,
                "Market Price": market_price,
                "Margin of Safety": mos,
                "Status": "Success",
            }
        )
    except Exception as e:
        results.append(
            {
                "Ticker": ticker,
                "Company Name": "N/A",
                "Model": "",
                "Intrinsic / share": None,
                "Market Price": None,
                "Margin of Safety": None,
                "Status": f"Failed: {e}",
            }
        )
    time.sleep(0.05)

my_bar.empty()

if results:
    df = pd.DataFrame(results)
    df_success = df[df["Status"] == "Success"].drop(columns=["Status"])
    df_failed = df[df["Status"] != "Success"]

    if not df_success.empty:
        st.subheader("Results")

        def color_mos(val):
            if pd.isna(val):
                return ""
            color = "#27AE60" if val > 0 else "#E74C3C"
            return f"color: {color}; font-weight: bold;"

        format_dict = {
            "Market Price": "${:.2f}",
            "Intrinsic / share": "${:.2f}",
            "Margin of Safety": "{:.2%}",
        }
        styled_df = df_success.style.format(format_dict).map(color_mos, subset=["Margin of Safety"])
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=min(35 * len(df_success) + 38, 600),
            hide_index=True,
        )

    if not df_failed.empty:
        with st.expander(f"Failed ({len(df_failed)})"):
            st.dataframe(df_failed[["Ticker", "Status"]], use_container_width=True, hide_index=True)
