"""Recompute intrinsic value at SEC periodic filing anchors; track price convergence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from valuation.backtest.engine import (
    BacktestValuation,
    first_hit_details,
    run_backtest_valuation,
)
from valuation.config import DCFAssumptions
from valuation.data.edgar import (
    DEFAULT_SUBMISSION_SIDECARS_FOR_ROLLING,
    lookup_cik,
    submission_filings_form_dates_merged,
)
from valuation.data.market import get_price_history_daterange
from valuation.growth.estimate import AutoGrowthMode


ModelMode = str

DEFAULT_PERIODIC_FORMS = frozenset({"10-K", "10-Q", "10-K/A", "10-Q/A"})


def revision_anchor_dates(
    cik: int,
    start: date,
    end: date,
    *,
    forms: frozenset[str] = DEFAULT_PERIODIC_FORMS,
    max_submission_sidecars: int = DEFAULT_SUBMISSION_SIDECARS_FOR_ROLLING,
) -> list[date]:
    """Sorted unique filing dates in ``[start, end]`` for SEC forms.

    Merges the primary submissions ``recent`` block with supplemental
    ``filings.files`` JSON (see :func:`~valuation.data.edgar.submission_filings_form_dates_merged`)
    so older 10‑K/10‑Q anchors remain available for busy filers.
    """
    fh, fdates = submission_filings_form_dates_merged(
        cik, max_sidecars=max_submission_sidecars
    )
    seen: set[date] = set()
    for form, fds in zip(fh, fdates):
        if form not in forms:
            continue
        try:
            d = datetime.strptime(str(fds)[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        if start <= d <= end:
            seen.add(d)
    return sorted(seen)


def valuation_anchor_dates(start: date, filing_dates_in_range: list[date]) -> list[date]:
    """User window start plus all filing anchors (dedupe sorted)."""
    return sorted({start}.union(set(filing_dates_in_range)))


def _to_date(x) -> date:
    return pd.Timestamp(x).date()


def select_iv(backtest_row: BacktestValuation, mode: str) -> float | None:
    if mode == "FCF IV":
        return backtest_row.iv_fcf_per_share
    if mode == "EPS IV":
        return backtest_row.iv_eps_per_share
    if mode == "Average (FCF + EPS)":
        if (
            backtest_row.iv_fcf_per_share is not None
            and backtest_row.iv_eps_per_share is not None
        ):
            return (
                backtest_row.iv_fcf_per_share + backtest_row.iv_eps_per_share
            ) / 2.0
    return None


@dataclass
class RollingSegmentStat:
    """After a filing at ``revision_date``, time to trade into bands around IV_new vs IV_prev."""

    revision_date: date
    iv_previous: float
    iv_new: float
    hit_new: bool
    hit_prev: bool
    calendar_days_to_new_band: int | None
    trading_sessions_to_new_band: int | None
    calendar_days_to_prev_band: int | None
    trading_sessions_to_prev_band: int | None


def _skip_reason(exc: BaseException, *, max_chars: int = 200) -> str:
    msg = f"{type(exc).__name__}: {exc}"
    return msg if len(msg) <= max_chars else msg[: max_chars - 3] + "..."


@dataclass
class RollingIVAnalysis:
    ticker: str
    start: date
    end: date
    revisions: pd.DataFrame
    daily: pd.DataFrame
    segments: list[RollingSegmentStat] = field(default_factory=list)
    skipped_revision_attempts: list[tuple[date, str]] = field(default_factory=list)


def build_iv_revision_table(
    ticker: str,
    anchor_dates: list[date],
    assumptions: DCFAssumptions,
    *,
    auto_growth: bool,
    auto_growth_mode: AutoGrowthMode,
    fcf_cagr_window: int | None,
    include_yahoo_consensus: bool,
    fcf_avg_years: int,
    match_fcf_for_eps: bool,
    manual_growth: float,
    eps_growth_override: float | None,
    model_mode: ModelMode,
    iv_mode: str,
) -> tuple[pd.DataFrame, list[tuple[date, str]]]:
    """DCF at each revision date using live ``DCFAssumptions``.

    Returns the successful revision rows plus ``(anchor_date, reason)`` skips.
    """
    dcf_model_mode: ModelMode = (
        "both" if iv_mode == "Average (FCF + EPS)" else model_mode
    )

    rows: list[dict] = []
    skipped: list[tuple[date, str]] = []
    for d in anchor_dates:
        try:
            bt = run_backtest_valuation(
                ticker,
                d,
                assumptions,
                model_mode=dcf_model_mode,  # type: ignore[arg-type]
                auto_growth=auto_growth,
                auto_growth_mode=auto_growth_mode,
                fcf_cagr_window=fcf_cagr_window,
                include_yahoo_consensus=include_yahoo_consensus,
                fcf_avg_years=fcf_avg_years,
                match_fcf_for_eps=match_fcf_for_eps,
                manual_growth=manual_growth,
                eps_growth_override=eps_growth_override,
            )
        except BaseException as exc:
            if type(exc) in (KeyboardInterrupt, SystemExit):
                raise
            skipped.append((d, _skip_reason(exc)))
            continue
        iv = select_iv(bt, iv_mode)
        if iv is None or iv <= 0:
            skipped.append((d, "No usable intrinsic for selected IV metric"))
            continue
        rows.append({"revision_date": d, "iv": float(iv)})
    return pd.DataFrame(rows), skipped


def forward_fill_iv_daily(closing_prices: pd.Series, revisions: pd.DataFrame) -> pd.DataFrame:
    """One row per session with Close and stepped ``iv_step``."""
    s = closing_prices.copy()
    s.index = pd.Index([_to_date(x) for x in s.index])

    df = s.reset_index()
    if df.shape[1] < 2:
        return pd.DataFrame()
    df = df.rename(columns={df.columns[0]: "session_date", df.columns[1]: "Close"})
    if revisions.empty:
        df["iv_step"] = float("nan")
        return df
    lhs = df.copy()
    lhs["_ord"] = range(len(lhs))
    lhs["session_dt"] = pd.to_datetime(lhs["session_date"])

    rv = revisions.sort_values("revision_date")[["revision_date", "iv"]].copy()
    rv = rv.drop_duplicates(subset=["revision_date"], keep="last")
    rv["rev_dt"] = pd.to_datetime(rv["revision_date"])

    merged = pd.merge_asof(
        lhs.sort_values("session_dt"),
        rv.sort_values("rev_dt"),
        left_on="session_dt",
        right_on="rev_dt",
        direction="backward",
    )
    merged = merged.sort_values("_ord")
    merged["iv_step"] = merged["iv"]
    return merged[["session_date", "Close", "iv_step"]]


def compute_segment_statistics(
    prices: pd.Series,
    revisions: pd.DataFrame,
    *,
    tolerance: float,
    chart_end: date,
) -> list[RollingSegmentStat]:
    """After each revision following the first, measure hits for IV_new vs IV_prior.

    Scans closes on sessions with ``session_date >= revision_date`` through
    ``chart_end``, aligned with stepped IV where ``revision_date <= session``.
    """
    if revisions.shape[0] < 2:
        return []

    px = prices.copy()
    px.index = pd.Index([_to_date(x) for x in px.index])

    segments: list[RollingSegmentStat] = []
    dd = revisions["revision_date"].tolist()
    ivs = revisions["iv"].tolist()
    for i in range(1, len(dd)):
        t_i = dd[i]
        iv_new = float(ivs[i])
        iv_prev = float(ivs[i - 1])
        sub = px[(px.index >= t_i) & (px.index <= chart_end)].sort_index()
        if sub.empty:
            segments.append(
                RollingSegmentStat(
                    revision_date=t_i,
                    iv_previous=iv_prev,
                    iv_new=iv_new,
                    hit_new=False,
                    hit_prev=False,
                    calendar_days_to_new_band=None,
                    trading_sessions_to_new_band=None,
                    calendar_days_to_prev_band=None,
                    trading_sessions_to_prev_band=None,
                )
            )
            continue

        hn = first_hit_details(sub, iv_new, tolerance=tolerance, anchor_date=t_i)
        hp = first_hit_details(sub, iv_prev, tolerance=tolerance, anchor_date=t_i)
        segments.append(
            RollingSegmentStat(
                revision_date=t_i,
                iv_previous=iv_prev,
                iv_new=iv_new,
                hit_new=hn.hit,
                hit_prev=hp.hit,
                calendar_days_to_new_band=hn.calendar_days_from_anchor,
                trading_sessions_to_new_band=hn.trading_sessions_included,
                calendar_days_to_prev_band=hp.calendar_days_from_anchor,
                trading_sessions_to_prev_band=hp.trading_sessions_included,
            )
        )
    return segments


def run_rolling_iv_analysis(
    ticker: str,
    start: date,
    chart_end: date,
    assumptions: DCFAssumptions,
    *,
    tolerance: float,
    iv_mode: str,
    model_mode: ModelMode,
    auto_growth: bool,
    auto_growth_mode: AutoGrowthMode,
    fcf_cagr_window: int | None,
    include_yahoo_consensus: bool,
    fcf_avg_years: int,
    match_fcf_for_eps: bool,
    manual_growth: float,
    eps_growth_override: float | None,
    forms: frozenset[str] = DEFAULT_PERIODIC_FORMS,
    filing_dates_override: list[date] | None = None,
    max_submission_sidecars: int = DEFAULT_SUBMISSION_SIDECARS_FOR_ROLLING,
) -> RollingIVAnalysis:
    sym = ticker.strip().upper()
    cik, _ = lookup_cik(sym)
    filing_dates = (
        filing_dates_override
        if filing_dates_override is not None
        else revision_anchor_dates(
            cik,
            start,
            chart_end,
            forms=forms,
            max_submission_sidecars=max_submission_sidecars,
        )
    )
    anchors = valuation_anchor_dates(start, filing_dates)
    anchors = [d for d in anchors if start <= d <= chart_end]

    revisions, revision_skipped = build_iv_revision_table(
        sym,
        anchors,
        assumptions,
        auto_growth=auto_growth,
        auto_growth_mode=auto_growth_mode,
        fcf_cagr_window=fcf_cagr_window,
        include_yahoo_consensus=include_yahoo_consensus,
        fcf_avg_years=fcf_avg_years,
        match_fcf_for_eps=match_fcf_for_eps,
        manual_growth=manual_growth,
        eps_growth_override=eps_growth_override,
        model_mode=model_mode,
        iv_mode=iv_mode,
    )
    prices = get_price_history_daterange(sym, start, chart_end)
    daily = forward_fill_iv_daily(prices, revisions)
    segments = compute_segment_statistics(
        prices, revisions, tolerance=tolerance, chart_end=chart_end
    )
    return RollingIVAnalysis(
        ticker=sym,
        start=start,
        end=chart_end,
        revisions=revisions,
        daily=daily,
        segments=segments,
        skipped_revision_attempts=revision_skipped,
    )
