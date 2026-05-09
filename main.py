import streamlit as st

# Define the pages and their clean display titles
dcf_page = st.Page("pages/single_dcf.py", title="Single Stock DCF", default=True)
screener_page = st.Page("pages/screener.py", title="Batch Screener")

# Set up the navigation router
pg = st.navigation(
    {
        "Valuation Tools": [dcf_page, screener_page]
    }
)

# Run the selected page
pg.run()