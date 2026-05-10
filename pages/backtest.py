"""
Backtesting: intrinsic value at a historical date vs subsequent price path.

Uses EDGAR facts filed on or before the valuation date; Yahoo consensus is off
by default (not point-in-time). WACC is manual (live CAPM is current-data).
"""

from __future__ import annotations

import warnings
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

from valuation.backtest.engine import (
    first_hit_details,
    hit_within_horizon,
    run_backtest_valuation,
)
from valuation.backtest.rolling_iv import (
    revision_anchor_dates,
    run_rolling_iv_analysis,
    select_iv,
)
from valuation.config import (
    DCFAssumptions,
    DEFAULT_FCF_AVG_YEARS,
    DEFAULT_GROWTH_RATE,
    DEFAULT_PROJECTION_YEARS,
    DEFAULT_TERMINAL_GROWTH,
    DEFAULT_WACC,
)
from valuation.data.edgar import EdgarError, lookup_cik
from valuation.data.market import MarketDataError, get_price_history_daterange
from valuation.growth.estimate import AutoGrowthMode

st.set_page_config(page_title="Backtesting", layout="wide")


@st.cache_data(ttl=1800)
def _cached_revision_filing_dates(
    cik: int,
    start_iso: str,
    end_iso: str,
) -> tuple[str, ...]:
    d0 = date.fromisoformat(start_iso)
    d1 = date.fromisoformat(end_iso)
    return tuple(str(d) for d in revision_anchor_dates(cik, d0, d1))


st.title("Backtesting")
st.caption(
    "Intrinsic value uses SEC filings known by the valuation date. "
    "Default growth excludes Yahoo (not historical). Use manual WACC."
)

tab_single, tab_basket, tab_rolling = st.tabs(["Single ticker", "Basket", "Rolling IV"])

with st.sidebar:
    st.header("DCF assumptions (backtest)")
    wacc_val = st.slider("WACC", 0.02, 0.30, float(DEFAULT_WACC), 0.0025, "%.4f")
    terminal = st.slider(
        "Terminal growth",
        0.0,
        0.06,
        float(DEFAULT_TERMINAL_GROWTH),
        0.001,
        "%.4f",
    )
    projection_years = st.number_input(
        "Projection years", 1, 15, DEFAULT_PROJECTION_YEARS
    )
    auto_growth = st.checkbox("Auto-growth", value=True)
    auto_mode = st.radio(
        "Growth mode",
        ("Blended (FCF CAGR + 8-K, no Yahoo)", "Consensus only (Yahoo—live data)"),
        index=0,
        disabled=not auto_growth,
    )
    auto_growth_mode: AutoGrowthMode = (
        "consensus_only" if "Consensus" in auto_mode else "blended"
    )
    include_yahoo = auto_growth_mode == "consensus_only"
    fcf_cagr_win = st.number_input(
        "FCF CAGR FY window (blended)",
        0,
        30,
        2,
        disabled=not auto_growth or auto_growth_mode == "consensus_only",
    )
    manual_growth = st.slider(
        "Manual growth (if auto off)",
        -0.2,
        0.5,
        float(DEFAULT_GROWTH_RATE),
        0.005,
        "%.3f",
        disabled=auto_growth,
    )
    match_eps = st.checkbox("Match EPS growth to FCF", value=True)
    eps_override = st.slider(
        "EPS growth override",
        -0.2,
        0.5,
        float(DEFAULT_GROWTH_RATE),
        0.005,
        "%.3f",
        disabled=auto_growth or match_eps,
    )
    fcf_avg_years = st.number_input(
        "FCF / EPS anchor years",
        1,
        10,
        DEFAULT_FCF_AVG_YEARS,
    )

assumptions = DCFAssumptions(
    growth_rate=DEFAULT_GROWTH_RATE,
    wacc=wacc_val,
    terminal_growth=float(terminal),
    projection_years=int(projection_years),
)

fcf_win = None if int(fcf_cagr_win) == 0 else int(fcf_cagr_win)
eps_ov = None if match_eps else float(eps_override)


def _single_iv_mode_label(model_mode: str) -> str:
    return (
        "FCF IV"
        if model_mode == "fcf"
        else "EPS IV" if model_mode == "eps" else "Average (FCF + EPS)"
    )


def _months_approx(calendar_days: int) -> float:
    return round(calendar_days / 30.44, 1)


def _show_skipped_revision_table(
    rows: list[tuple[date, str]],
    *,
    title: str,
    expanded: bool = False,
) -> None:
    with st.expander(title, expanded=expanded):
        st.dataframe(
            pd.DataFrame(
                [{"Anchor": d.isoformat(), "Reason": msg} for d, msg in rows]
            ),
            hide_index=True,
            use_container_width=True,
        )


def _convergence_summary_table(seg_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Median, mean, min, max of time-to-hit, using only rows where that band hit."""
    empty_cols = (
        "Statistic",
        "Cal days→new IV band",
        "Sessions→new IV band",
        "Cal days→prior IV band",
        "Sessions→prior IV band",
    )
    if seg_df.empty or "Hit new IV band" not in seg_df.columns:
        z = pd.DataFrame(columns=list(empty_cols))
        return z, {"Filing segments": 0, "Hits→new IV": 0, "Hits→prior IV": 0}

    def _four_stats(hit_col: str, val_col: str) -> dict[str, float | None]:
        raw = pd.to_numeric(seg_df.loc[seg_df[hit_col].astype(bool), val_col], errors="coerce").dropna()
        if raw.empty:
            return {"Median": None, "Mean": None, "Min": None, "Max": None}
        v = raw.astype(float)
        return {
            "Median": float(v.median()),
            "Mean": float(v.mean()),
            "Min": float(v.min()),
            "Max": float(v.max()),
        }

    n_new = _four_stats("Hit new IV band", "Cal days→new")
    s_new = _four_stats("Hit new IV band", "Sessions→new")
    n_prev = _four_stats("Hit prior IV band", "Cal days→prior")
    s_prev = _four_stats("Hit prior IV band", "Sessions→prior")

    order = ("Median", "Mean", "Min", "Max")
    rows = [
        {
            "Statistic": label,
            "Cal days→new IV band": n_new[label],
            "Sessions→new IV band": s_new[label],
            "Cal days→prior IV band": n_prev[label],
            "Sessions→prior IV band": s_prev[label],
        }
        for label in order
    ]
    meta = {
        "Filing segments": int(len(seg_df)),
        "Hits→new IV": int(seg_df["Hit new IV band"].astype(bool).sum()),
        "Hits→prior IV": int(seg_df["Hit prior IV band"].astype(bool).sum()),
    }
    return pd.DataFrame(rows), meta


def _ribbon_rows_price_vs_fv(daily: pd.DataFrame) -> pd.DataFrame:
    """Contiguous shaded bands between Close and stepped fair value."""
    cols = ("session_date", "Close", "iv_step")
    if daily.empty or any(c not in daily.columns for c in cols):
        return pd.DataFrame()
    work = daily.loc[:, cols].dropna(subset=("iv_step", "Close")).sort_values("session_date")
    work = work.reset_index(drop=True)
    if work.empty:
        return work
    work["t"] = pd.to_datetime(work["session_date"], utc=False, errors="coerce")
    work = work.dropna(subset=("t",))
    work["_over"] = work["Close"] >= work["iv_step"]
    work["_grp"] = work["_over"].ne(work["_over"].shift()).cumsum()
    strips: list[pd.DataFrame] = []
    for _, grp in work.groupby("_grp", sort=False):
        g = grp.copy()
        g["regime"] = "Overvalued" if bool(grp["_over"].iloc[0]) else "Undervalued"
        g["y_hi"] = g[["Close", "iv_step"]].max(axis=1)
        g["y_lo"] = g[["Close", "iv_step"]].min(axis=1)
        # Unique id per contiguous run so Altair does not connect separate orange/blue spans.
        g["segment_id"] = int(g["_grp"].iloc[0])
        strips.append(
            g[["t", "Close", "iv_step", "regime", "y_hi", "y_lo", "segment_id"]]
        )
    return pd.concat(strips, ignore_index=True)


def _rolling_iv_morningstar_chart(
    daily: pd.DataFrame,
    *,
    ticker: str,
) -> tuple[alt.Chart, pd.Series | None]:
    """Morningstar-like: shaded over/under, gray price path, stepped black fair value."""
    src_all = daily.copy()
    src_all["t"] = pd.to_datetime(src_all["session_date"], utc=False, errors="coerce")
    fv_src = (
        src_all.dropna(subset=("iv_step", "Close"))
        .sort_values("t")
        .dropna(subset=("t",))
        .drop_duplicates(subset=("t",), keep="last")
        .reset_index(drop=True)
    )
    ribbons = _ribbon_rows_price_vs_fv(daily)
    if fv_src.empty:
        return alt.Chart(fv_src).mark_point(), None

    y_title = "$ / share"
    # Saturated ramps read well on Streamlit dark theme; ribbons stay legible vs near-black Vega bg.
    color_scale = alt.Scale(
        domain=["Overvalued", "Undervalued"],
        range=["#ff922b", "#4dabf7"],
    )

    if not ribbons.empty:
        shading = alt.Chart(ribbons).mark_area(opacity=0.52, interpolate="linear").encode(
            x=alt.X("t:T", title="Date"),
            y=alt.Y("y_hi:Q", title=y_title),
            y2=alt.Y2("y_lo:Q"),
            color=alt.Color(
                "regime:N",
                scale=color_scale,
                legend=alt.Legend(orient="top", columns=2, title=None),
            ),
            detail=alt.Detail("segment_id:N"),
            order=alt.Order("t:T"),
            tooltip=[
                alt.Tooltip("t:T", title="Date"),
                alt.Tooltip("Close:Q", title="Market price", format=".2f"),
                alt.Tooltip("iv_step:Q", title="Fair value", format=".2f"),
                alt.Tooltip("regime:N", title=None),
            ],
        )
    else:
        shading = None
    dots = alt.Chart(fv_src).mark_circle(size=11, opacity=0.42, color="#cbd5e1").encode(
        x=alt.X("t:T", title="Date"),
        y=alt.Y("Close:Q", title=y_title),
        tooltip=[
            alt.Tooltip("t:T", title="Date"),
            alt.Tooltip("Close:Q", title="Market price", format=".2f"),
        ],
    )
    px_line = alt.Chart(fv_src).mark_line(
        interpolate="linear",
        stroke="#f8fafc",
        strokeWidth=1.5,
        opacity=0.95,
    ).encode(
        x="t:T",
        y="Close:Q",
        tooltip=[
            alt.Tooltip("t:T", title="Date"),
            alt.Tooltip("Close:Q", title="Market price", format=".2f"),
            alt.Tooltip("iv_step:Q", title="Fair value", format=".2f"),
        ],
    )
    fv_line = alt.Chart(fv_src).mark_line(
        interpolate="step-after",
        color="#fff3bf",
        strokeWidth=3.0,
    ).encode(
        x="t:T",
        y=alt.Y("iv_step:Q", title=y_title),
        tooltip=[
            alt.Tooltip("t:T", title="Date"),
            alt.Tooltip("iv_step:Q", title="Fair value", format=".2f"),
        ],
    )

    combo = (shading + dots + px_line + fv_line) if shading is not None else (dots + px_line + fv_line)
    last = fv_src.iloc[-1]

    labeled = (
        combo.properties(
            height=440,
            title=alt.TitleParams(
                text=f"{ticker.upper()} — price vs model fair value",
                subtitle=(
                    "Orange: price above modeled fair value. Blue: below. Cream step line: fair "
                    "(10-K / 10-Q filings)."
                ),
                color="#f8fafc",
                subtitleColor="#cbd5e1",
                fontSize=18,
                subtitleFontSize=13,
            ),
        )
        .configure_axis(
            gridColor="#475569",
            domainColor="#94a3b8",
            tickColor="#94a3b8",
            labelColor="#e2e8f0",
            titleColor="#f1f5f9",
        )
        .configure_view(strokeWidth=0)
        .configure_legend(labelColor="#e2e8f0", titleColor="#f1f5f9")
    )
    return labeled, last


COMMON_BT_KWARGS = dict(
    auto_growth=auto_growth,
    auto_growth_mode=auto_growth_mode,
    fcf_cagr_window=fcf_win,
    include_yahoo_consensus=include_yahoo,
    fcf_avg_years=int(fcf_avg_years),
    match_fcf_for_eps=match_eps,
    manual_growth=float(manual_growth),
    eps_growth_override=eps_ov,
)


with tab_single:
    c1, c2 = st.columns(2)
    with c1:
        ticker_s = st.text_input("Ticker", value="AAPL", key="bt_ticker").strip().upper() or "AAPL"
    with c2:
        model_mode = st.selectbox("Model", ("fcf", "eps", "both"), index=0, key="bt_model")

    d1, d2 = st.columns(2)
    with d1:
        val_date = st.date_input(
            "Valuation date (as-of)",
            value=date(2020, 1, 2),
            max_value=date.today(),
            key="bt_val_date",
        )
    with d2:
        end_chart = st.date_input(
            "Chart end date",
            value=date.today(),
            max_value=date.today(),
            key="bt_end",
        )

    tol_single = st.slider(
        "Band tolerance (hit test)",
        0.0,
        0.10,
        0.02,
        0.005,
        "%.3f",
        help="Hit when close is within this fraction of intrinsic value.",
        key="bt_tol_s",
    )

    run_s = st.button("Run single-ticker backtest", type="primary")

    if run_s:
        if end_chart < val_date:
            st.error("Chart end must be on or after valuation date.")
        else:
            try:
                assumptions.validate()
            except ValueError as e:
                st.error(str(e))
                st.stop()
            try:
                with st.spinner("Loading fundamentals (as-of)..."):
                    bt = run_backtest_valuation(
                        ticker_s,
                        val_date,
                        assumptions,
                        model_mode=model_mode,
                        **COMMON_BT_KWARGS,
                    )
            except EdgarError as e:
                st.error(f"EDGAR: {e}")
                st.stop()

            iv_line = select_iv(bt, _single_iv_mode_label(model_mode))

            st.subheader("Intrinsic snapshot")
            mcols = st.columns(4)
            if bt.iv_fcf_per_share is not None:
                mcols[0].metric("IV / share (FCF)", f"${bt.iv_fcf_per_share:,.2f}")
            if bt.iv_eps_per_share is not None:
                mcols[1].metric("IV / share (EPS)", f"${bt.iv_eps_per_share:,.2f}")
            mcols[2].metric("Growth (FCF path)", f"{bt.growth_fcf:.2%}")
            mcols[3].metric("Base FCF label", bt.base_fcf_label)

            if iv_line is None or iv_line <= 0:
                st.warning("Could not compute intrinsic value for this mode / inputs.")
            else:
                try:
                    with st.spinner("Loading prices..."):
                        px = get_price_history_daterange(ticker_s, val_date, end_chart)
                except MarketDataError as e:
                    st.error(str(e))
                    st.stop()

                hd = first_hit_details(
                    px, iv_line, tolerance=float(tol_single), anchor_date=val_date
                )
                st.write(
                    f"**Within-band touch:** {'Yes' if hd.hit else 'No'}  "
                )
                if hd.hit and hd.calendar_days_from_anchor is not None:
                    mo = _months_approx(hd.calendar_days_from_anchor)
                    st.info(
                        f"From valuation date (**{val_date.isoformat()}**): "
                        f"**{hd.trading_sessions_included}** trading sessions (bar #{hd.bar_index_zero_based} zero-based index), "
                        f"**{hd.calendar_days_from_anchor}** calendar days to first touch on **{hd.first_hit_date}** "
                        f"(~**{mo}** months)."
                    )

                df_px = px.reset_index()
                df_px.columns = ["Date", "Close"]
                chart = (
                    alt.Chart(df_px)
                    .mark_line(color="#e74c3c")
                    .encode(x="Date:T", y=alt.Y("Close:Q", title="Price / share"))
                )
                rule_df = pd.DataFrame({"iv": [float(iv_line)]})
                iv_rule = (
                    alt.Chart(rule_df)
                    .mark_rule(color="#2980b9", strokeDash=[4, 4])
                    .encode(y="iv:Q")
                )
                st.altair_chart(
                    (chart + iv_rule).properties(height=400),
                    use_container_width=True,
                )

            if bt.growth_estimate:
                with st.expander("Growth candidates"):
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Source": c.source,
                                    "Rate": f"{c.growth_rate:.2%}",
                                    "Snippet": c.snippet[:120],
                                }
                                for c in bt.growth_estimate.candidates
                            ]
                        ),
                        hide_index=True,
                    )


with tab_basket:
    st.markdown(
        "Enter tickers (comma or newline separated). Same assumptions as sidebar."
    )
    basket_raw = st.text_area("Tickers", value="AAPL\nMSFT", height=120)
    val_date_b = st.date_input(
        "Valuation date (shared)",
        value=date(2020, 1, 2),
        max_value=date.today(),
        key="bt_b_val",
    )
    end_chart_b = st.date_input(
        "Forward end date",
        value=date.today(),
        max_value=date.today(),
        key="bt_b_end",
    )
    model_b = st.selectbox(
        "IV for hit test",
        ("FCF IV", "EPS IV", "Average (FCF + EPS)"),
        index=0,
    )
    tol_b = st.slider(
        "Band tolerance",
        0.0,
        0.10,
        0.02,
        0.005,
        "%.3f",
        key="bt_tol_b",
    )
    st.subheader("Optional hit horizon")
    use_horizon = st.checkbox(
        "Require hit within a maximum horizon",
        value=False,
        help="Adds 'hit within horizon' column and conditional hit rate; 0 disables that cap.",
    )
    hz_mode = st.radio(
        "Horizon type",
        ("Calendar days", "Trading sessions"),
        horizontal=True,
        disabled=not use_horizon,
    )
    hz_val = st.number_input(
        "Max horizon",
        min_value=0,
        max_value=5000,
        value=252,
        key="bt_hz_val",
        disabled=not use_horizon,
        help="0 or leave unused = no horizon cap.",
    )

    run_b = st.button("Run basket")

    if run_b:
        max_cal = hz_val if use_horizon and hz_mode.startswith("Calendar") else None
        max_tr = hz_val if use_horizon and hz_mode.startswith("Trading") else None
        if hz_val <= 0 or not use_horizon:
            max_cal = None
            max_tr = None

        tickers = [
            t.strip().upper()
            for t in basket_raw.replace(",", "\n").splitlines()
            if t.strip()
        ]
        if not tickers:
            st.warning("Add at least one ticker.")
        elif end_chart_b < val_date_b:
            st.error("End date must be on or after valuation date.")
        else:
            try:
                assumptions.validate()
            except ValueError as e:
                st.error(str(e))
                st.stop()

            rows = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for tk in tickers:
                    row_base = {
                        "Ticker": tk,
                        "IV target": None,
                        "Hit anytime": False,
                        "Hit within horizon": False,
                        "Trading sessions": None,
                        "Calendar days": None,
                        "Months approx": None,
                        "Note": "",
                    }
                    try:
                        bt = run_backtest_valuation(
                            tk,
                            val_date_b,
                            assumptions,
                            model_mode="both",
                            **COMMON_BT_KWARGS,
                        )
                    except EdgarError as exc:
                        row_base["Note"] = str(exc)[:80]
                        rows.append(row_base)
                        continue

                    iv_t = select_iv(bt, model_b)
                    row_base["IV target"] = iv_t
                    if iv_t is None or iv_t <= 0:
                        row_base["Note"] = "No IV"
                        rows.append(row_base)
                        continue

                    try:
                        px_b = get_price_history_daterange(tk, val_date_b, end_chart_b)
                    except MarketDataError as exc:
                        row_base["IV target"] = iv_t
                        row_base["Note"] = str(exc)[:80]
                        rows.append(row_base)
                        continue

                    hd = first_hit_details(
                        px_b, iv_t, tolerance=float(tol_b), anchor_date=val_date_b
                    )
                    row_base["Hit anytime"] = hd.hit
                    row_base["Hit within horizon"] = hit_within_horizon(
                        hd, max_trading_sessions=max_tr, max_calendar_days=max_cal
                    )
                    if hd.hit:
                        row_base["Trading sessions"] = hd.trading_sessions_included
                        row_base["Calendar days"] = hd.calendar_days_from_anchor
                        cd = hd.calendar_days_from_anchor or 0
                        row_base["Months approx"] = _months_approx(cd)
                    rows.append(row_base)

            df_r = pd.DataFrame(rows)
            valid = df_r[df_r["IV target"].notna() & (df_r["IV target"] > 0)]
            hits_any = int(valid["Hit anytime"].sum())
            denom = len(valid)
            rate_any = hits_any / denom if denom else 0.0
            c_metric1, c_metric2 = st.columns(2)
            c_metric1.metric(
                "Hit rate (anytime)",
                f"{rate_any:.1%}",
                help=f"{hits_any} / {denom} tickers with valid IV",
            )
            if max_cal or max_tr:
                hits_hz = int(valid["Hit within horizon"].sum())
                rate_hz = hits_hz / denom if denom else 0.0
                c_metric2.metric(
                    "Hit rate within horizon",
                    f"{rate_hz:.1%}",
                    help=f"{hits_hz} / {denom} with active horizon cap",
                )
            else:
                c_metric2.metric(
                    "Hit rate within horizon",
                    "—",
                    help="Enable horizon filter above to score timed hits.",
                )
            st.dataframe(df_r, hide_index=True, use_container_width=True)


with tab_rolling:
    st.markdown(
        "Recompute intrinsic value each time a **10-K**, **10-Q**, or amended **10-K/A** / **10-Q/A** "
        "filing arrives in-window; "
        "then compare how quickly price enters the tolerance band vs **new** IV and **prior** IV."
    )
    st.caption(
        "Filing anchors load the primary EDGAR submissions feed plus up to **20** archived submission "
        "JSON files—needed for liquid names whose “recent” list is crowded by other forms (~5y of quarterlies)."
    )
    rk1, rk2 = st.columns(2)
    with rk1:
        roll_ticker_s = (
            st.text_input("Ticker (single)", value="AAPL", key="rb_tsingle")
            .strip()
            .upper()
            or "AAPL"
        )
    with rk2:
        roll_mode_iv = st.selectbox(
            "IV metric",
            ("FCF IV", "EPS IV", "Average (FCF + EPS)"),
            index=0,
            key="rb_ivm",
        )
    roll_model = st.selectbox("DCF inputs", ("fcf", "eps", "both"), index=2, key="rb_mdl")

    rr1, rr2 = st.columns(2)
    with rr1:
        roll_start = st.date_input("Start date", value=date(2020, 1, 2), key="rb_rs")
    with rr2:
        roll_end = st.date_input(
            "End date",
            value=date.today(),
            max_value=date.today(),
            key="rb_re",
        )
    tol_r = st.slider(
        "Band tolerance",
        0.0,
        0.10,
        0.02,
        0.005,
        "%.3f",
        key="rb_tol",
    )

    rb_run_single = st.button("Run rolling IV (single)")
    rb_basket_text = st.text_area(
        "Basket tickers (optional; runs summary only)",
        "",
        height=70,
        key="rb_tbasket",
        help="Leave empty to skip basket batch.",
    )
    rb_run_basket = st.button("Run rolling IV basket summary")

    if rb_run_single:
        if roll_end < roll_start:
            st.error("End >= start.")
        else:
            try:
                assumptions.validate()
            except ValueError as e:
                st.error(str(e))
                st.stop()

            sym = roll_ticker_s
            try:
                cik, _name = lookup_cik(sym)
            except EdgarError as e:
                st.error(str(e))
                st.stop()

            fd_iso = _cached_revision_filing_dates(
                cik, roll_start.isoformat(), roll_end.isoformat()
            )
            fds = [date.fromisoformat(x) for x in fd_iso]

            with st.spinner("Rolling IV computation…"):
                try:
                    ana = run_rolling_iv_analysis(
                        sym,
                        roll_start,
                        roll_end,
                        assumptions,
                        tolerance=float(tol_r),
                        iv_mode=roll_mode_iv,
                        model_mode=roll_model,
                        filing_dates_override=fds,
                        **COMMON_BT_KWARGS,
                    )
                except (EdgarError, MarketDataError) as e:
                    st.error(str(e))
                    st.stop()

            if ana.revisions.empty:
                st.warning("No valid IV snapshots in window.")
                if ana.skipped_revision_attempts:
                    _show_skipped_revision_table(
                        ana.skipped_revision_attempts,
                        title="Skipped anchor attempts (why no IV)",
                        expanded=True,
                    )
                st.stop()

            if ana.skipped_revision_attempts:
                st.warning(
                    f"{len(ana.skipped_revision_attempts)} revision date(s) did not yield a modeled IV "
                    "(errors or unsupported inputs). Showing successful anchors only.",
                )
                _show_skipped_revision_table(
                    ana.skipped_revision_attempts,
                    title="Skipped revision anchors",
                )

            st.subheader("Revision anchors")
            st.dataframe(
                ana.revisions,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "revision_date": st.column_config.DateColumn(
                        "Revision date", format="YYYY-MM-DD", width="medium"
                    ),
                    "iv": st.column_config.NumberColumn(
                        "IV ($/sh)", format="%.4f", width="small"
                    ),
                },
            )

            seg_df = pd.DataFrame(
                [
                    {
                        "Revision": s.revision_date,
                        "Hit new IV band": s.hit_new,
                        "Cal days→new": s.calendar_days_to_new_band,
                        "Sessions→new": s.trading_sessions_to_new_band,
                        "Hit prior IV band": s.hit_prev,
                        "Cal days→prior": s.calendar_days_to_prev_band,
                        "Sessions→prior": s.trading_sessions_to_prev_band,
                    }
                    for s in ana.segments
                ]
            )
            st.subheader("Per-filing convergence (sessions after filing)")
            st.dataframe(seg_df, hide_index=True, use_container_width=True)

            if seg_df.empty:
                st.info(
                    "No per-filing segments yet (needs at least two IV revisions after the window start)."
                )
            else:
                st.subheader("Convergence summary (median · mean · min · max)")
                roll_summ, meta = _convergence_summary_table(seg_df)
                m1, m2, m3 = st.columns(3)
                m1.metric("Filing segments", meta["Filing segments"])
                m2.metric("Hits→new IV band", meta["Hits→new IV"])
                m3.metric("Hits→prior IV band", meta["Hits→prior IV"])
                df_show = roll_summ.copy()
                for c in df_show.columns:
                    if c == "Statistic":
                        continue
                    df_show[c] = df_show[c].apply(
                        lambda v: round(v, 2) if v is not None and pd.notna(v) else None
                    )
                st.dataframe(df_show, hide_index=True, use_container_width=True)
                st.caption(
                    "Each row summarizes time-to-hit across filing segments that hit that band "
                    "(new IV columns use rows with “Hit new IV band”; prior columns use “Hit prior IV band”). "
                    "Blank cells mean no hit for that band."
                )

            dn = ana.daily
            if dn.empty:
                st.warning("No overlapping price series.")
            else:
                st.markdown("##### Price vs fair value")
                cc, kk = st.columns([4.1, 1.0])
                with cc:
                    mv_chart, last_row = _rolling_iv_morningstar_chart(dn, ticker=sym)
                    st.altair_chart(mv_chart, use_container_width=True)
                with kk:
                    st.caption("As of chart end session")
                    if last_row is not None:
                        ts = pd.Timestamp(last_row["t"])
                        fv = float(last_row["iv_step"])
                        px_last = float(last_row["Close"])
                        st.markdown("**Modeled fair value**")
                        st.markdown(f"${fv:,.2f}")
                        st.caption(ts.strftime("%d %b %Y"))
                        st.divider()
                        st.markdown("**Last close**")
                        st.markdown(f"${px_last:,.2f}")


    if rb_run_basket:
        if roll_end < roll_start:
            st.error("End >= start.")
        else:
            tks = [
                x.strip().upper()
                for x in rb_basket_text.replace(",", "\n").splitlines()
                if x.strip()
            ]
            if not tks:
                st.warning("Enter at least one ticker for basket mode.")
            else:
                summary_rows = []
                for sym in tks:
                    try:
                        cik, _ = lookup_cik(sym)
                    except EdgarError:
                        summary_rows.append(
                            {"Ticker": sym, "Revisions": None, "Note": "CIK"}
                        )
                        continue
                    fd_iso = _cached_revision_filing_dates(
                        cik, roll_start.isoformat(), roll_end.isoformat()
                    )
                    fds = [date.fromisoformat(x) for x in fd_iso]
                    try:
                        ana_b = run_rolling_iv_analysis(
                            sym,
                            roll_start,
                            roll_end,
                            assumptions,
                            tolerance=float(tol_r),
                            iv_mode=roll_mode_iv,
                            model_mode=roll_model,
                            filing_dates_override=fds,
                            **COMMON_BT_KWARGS,
                        )
                    except (EdgarError, MarketDataError) as exc:
                        summary_rows.append(
                            {"Ticker": sym, "Revisions": 0, "Note": str(exc)[:60]}
                        )
                        continue

                    n_skipped = len(ana_b.skipped_revision_attempts)
                    skip_note = (
                        f"{n_skipped} revision anchor(s) skipped — run Rolling IV single for details"
                        if n_skipped
                        else ""
                    )
                    nh = sum(1 for s in ana_b.segments if s.hit_new)
                    hp = sum(1 for s in ana_b.segments if s.hit_prev)
                    summary_rows.append(
                        {
                            "Ticker": sym,
                            "Anchors IV": len(ana_b.revisions),
                            "Segments": len(ana_b.segments),
                            "Hits→new IV": nh,
                            "Hits→prior IV": hp,
                            "Median cal days→new": pd.Series(
                                [s.calendar_days_to_new_band for s in ana_b.segments if s.hit_new]
                            ).median(),
                            "Median cal days→prior": pd.Series(
                                [s.calendar_days_to_prev_band for s in ana_b.segments if s.hit_prev]
                            ).median(),
                            "Note": skip_note,
                        }
                    )
                st.subheader("Basket rolling summary")
                st.dataframe(pd.DataFrame(summary_rows), hide_index=True)
