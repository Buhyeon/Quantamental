"""
Streamlit Multi-page: Historical Fundamentals Dashboard
Visualizes past FCF, EPS, and Balance Sheet trends from EDGAR.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from valuation.data.edgar import EdgarError, get_fundamentals

st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">Historical Fundamentals</div>', unsafe_allow_html=True)
st.caption("10-K Data extracted directly from SEC EDGAR XBRL filings.")

# --- Sidebar ---
with st.sidebar:
    st.title("Dashboard Settings")
    ticker = st.text_input("Ticker Symbol", value="AAPL").strip().upper() or "AAPL"
    run = st.button("Load History", type="primary", use_container_width=True)

if not run:
    st.info("Enter a ticker and click Load History.")
    st.stop()

# --- Data Fetching ---
try:
    with st.spinner(f"Pulling SEC filings for {ticker}..."):
        f = get_fundamentals(ticker)
except (EdgarError, RuntimeError) as e:
    st.error(f"Data Retrieval Error: {e}")
    st.stop()

# --- Helper to parse history ---
# We don't know the exact dictionary structure of your f.fcf_history, 
# so this safely falls back to the f.fcf_values array if needed.
def format_history(history_obj, values_list, metric_name):
    if isinstance(history_obj, dict) and history_obj:
        # Assuming dict keys are years/dates
        df = pd.DataFrame([{"Period": str(k), metric_name: float(v)} for k, v in history_obj.items()])
    elif values_list:
        # Fallback if it's just a raw list (oldest to newest)
        vals = list(values_list)
        df = pd.DataFrame([
            {"Period": f"T-{len(vals)-i-1}" if i < len(vals)-1 else "Latest FY", metric_name: float(v)} 
            for i, v in enumerate(vals)
        ])
    else:
        return pd.DataFrame()
    return df

# Process the data
df_fcf = format_history(getattr(f, 'fcf_history', None), f.fcf_values, "Free Cash Flow")
df_eps = format_history(getattr(f, 'eps_history', None), f.eps_values, "EPS")

# --- Layout ---
st.subheader(f"{f.company_name} ({ticker})")

# Top Level Current Stats
c1, c2, c3, c4 = st.columns(4)
c1.metric("Shares Outstanding", f"{f.shares_outstanding/1e6:,.1f}M")
c2.metric("Total Debt", f"${f.total_debt/1e9:,.2f}B")
c3.metric("Net Debt", f"${f.net_debt/1e9:,.2f}B")

# Calculate cash balance (Net Debt = Total Debt - Cash, so Cash = Total Debt - Net Debt)
cash_equiv = f.total_debt - f.net_debt
c4.metric("Cash & Equivalents", f"${cash_equiv/1e9:,.2f}B")

st.divider()

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("#### Free Cash Flow History")
    if not df_fcf.empty:
        # Convert to Billions for cleaner Y-axis if the numbers are huge
        df_fcf["FCF (Billions)"] = df_fcf["Free Cash Flow"] / 1e9
        
        fcf_chart = alt.Chart(df_fcf).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Period:O", sort=None, title="Fiscal Period"),
            y=alt.Y("FCF (Billions):Q", title="USD (Billions)"),
            color=alt.condition(
                alt.datum["FCF (Billions)"] > 0,
                alt.value("#27AE60"),  # Green for positive FCF
                alt.value("#E74C3C")   # Red for negative FCF
            ),
            tooltip=["Period", "Free Cash Flow"]
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
            tooltip=["Period", "EPS"]
        ).properties(height=350)
        st.altair_chart(eps_chart, use_container_width=True)
    else:
        st.warning("No EPS history available.")