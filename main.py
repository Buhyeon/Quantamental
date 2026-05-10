import streamlit as st

st.set_page_config(page_title="Quantamental", layout="wide")

# Define the pages
dcf_page = st.Page("pages/single_dcf.py", title="Single Stock DCF", default=True)
backtest_page = st.Page("pages/backtest.py", title="Backtesting")
dashboard_page = st.Page("pages/dashboard.py", title="Historical Dashboard")
screener_page = st.Page("pages/screener.py", title="Batch Screener")

# Set up the navigation router
pg = st.navigation(
    {
        "Valuation Tools": [dcf_page, screener_page, backtest_page],
        "Fundamental Analysis": [dashboard_page]
    }
)

pg.run()