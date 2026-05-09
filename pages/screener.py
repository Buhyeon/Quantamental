"""
Streamlit Multi-page: Batch Screener
Runs DCF valuation across multiple tickers to find Margin of Safety.
"""

from __future__ import annotations

import time
import pandas as pd
import streamlit as st

# Assuming these are available from your valuation package
from valuation.config import (
    DEFAULT_FCF_AVG_YEARS,
    DEFAULT_PROJECTION_YEARS,
    DEFAULT_TERMINAL_GROWTH,
    DEFAULT_WACC,
    average_fcf,
)
from valuation.data.edgar import EdgarError, Fundamentals, get_fundamentals
from valuation.data.market import MarketDataError, get_current_price
from valuation.models.dcf import intrinsic_value_per_share

# --- Helper Function ---
def _base_fcf(f: Fundamentals, avg_years: int) -> float:
    if f.ttm_fcf is not None:
        return float(f.ttm_fcf)
    return average_fcf(f.fcf_values, years=avg_years)

# --- Page Config ---
st.set_page_config(page_title="Quantamental Screener", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">Batch DCF Screener</div>', unsafe_allow_html=True)
st.caption("Run FCF-based Intrinsic Value calculations across multiple tickers.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.title("Screener Settings")
    
    ticker_input = st.text_area(
        "Tickers (comma separated)", 
        value="AAPL, MSFT, GOOGL, META, AMZN, TSLA, NVDA",
        height=150,
        help="Paste up to 500 tickers here. Warning: Large batches may take several minutes."
    )
    
    with st.expander("Global Assumptions", expanded=True):
        growth_rate = st.slider("Flat Growth Rate", 0.0, 0.50, 0.15, 0.01)
        wacc = st.slider("Discount Rate (WACC)", 0.05, 0.20, float(DEFAULT_WACC), 0.01)
        terminal = st.slider("Terminal Growth", 0.0, 0.05, float(DEFAULT_TERMINAL_GROWTH), 0.005)
        fcf_avg_years = st.number_input("FCF Avg Years", 1, 10, DEFAULT_FCF_AVG_YEARS)

    run_screener = st.button("Run Screener", type="primary", use_container_width=True)

# --- Processing Logic ---
if run_screener:
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    
    if not tickers:
        st.warning("Please enter at least one ticker.")
        st.stop()

    results = []
    
    # UI Elements for progress
    progress_text = "Scanning market data. Please wait."
    my_bar = st.progress(0, text=progress_text)
    status_container = st.empty()
    
    total = len(tickers)
    
    for i, ticker in enumerate(tickers):
        # Update progress
        progress_pct = (i + 1) / total
        my_bar.progress(progress_pct, text=f"Processing {ticker} ({i+1}/{total})")
        
        try:
            # 1. Fetch Fundamentals
            f = get_fundamentals(ticker)
            
            if not f.fcf_history:
                raise ValueError("No FCF history found.")
            if f.shares_outstanding <= 0:
                raise ValueError("Invalid shares outstanding.")
                
            base_fcf = _base_fcf(f, int(fcf_avg_years))
            
            # 2. Calculate FCF Intrinsic Value
            iv_res = intrinsic_value_per_share(
                base_fcf=base_fcf,
                growth_rate=growth_rate,
                wacc=wacc,
                terminal_growth=terminal,
                shares_outstanding=f.shares_outstanding,
                net_debt=f.net_debt,
                projection_years=DEFAULT_PROJECTION_YEARS
            )
            iv = float(iv_res["intrinsic_value_per_share"])
            
            # 3. Fetch Market Price
            market_price = get_current_price(ticker)
            
            # 4. Calculate Margin of Safety
            if market_price and market_price > 0:
                mos = (iv - market_price) / market_price
            else:
                mos = None
                
            results.append({
                "Ticker": ticker,
                "Company Name": f.company_name,
                "Market Price": market_price,
                "FCF Intrinsic Value": iv,
                "Margin of Safety": mos,
                "Status": "Success"
            })
            
        except Exception as e:
            # Catch EDGAR errors, Yahoo errors, or missing data gracefully
            results.append({
                "Ticker": ticker,
                "Company Name": "N/A",
                "Market Price": None,
                "FCF Intrinsic Value": None,
                "Margin of Safety": None,
                "Status": f"Failed: {str(e)}"
            })
            
        # Slight sleep to respect API limits if you scale up to 500
        time.sleep(0.1) 

    my_bar.empty() # Clear progress bar when done
    
    # --- Display Results ---
    if results:
        df = pd.DataFrame(results)
        
        # Filter out failed runs for the main view, but keep them for debugging
        df_success = df[df["Status"] == "Success"].drop(columns=["Status"])
        df_failed = df[df["Status"] != "Success"]
        
        if not df_success.empty:
            st.subheader("Screener Results")
            
            # Style the dataframe to highlight Margin of Safety
            def color_mos(val):
                if pd.isna(val):
                    return ''
                color = '#27AE60' if val > 0 else '#E74C3C' # Green if undervalued, Red if overvalued
                return f'color: {color}; font-weight: bold;'
            
            # Formatting options
            format_dict = {
                "Market Price": "${:.2f}",
                "FCF Intrinsic Value": "${:.2f}",
                "Margin of Safety": "{:.2%}"
            }
            
            styled_df = df_success.style.format(format_dict).map(color_mos, subset=['Margin of Safety'])
            
            # Use st.dataframe with column config for a clean UI
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=min(35 * len(df_success) + 38, 600), # Dynamic height
                hide_index=True
            )
            
        if not df_failed.empty:
            with st.expander(f"Failed Data Pulls ({len(df_failed)})"):
                st.dataframe(df_failed[["Ticker", "Status"]], use_container_width=True, hide_index=True)