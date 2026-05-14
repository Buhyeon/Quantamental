"""SEC EDGAR XBRL client.

Pulls audited annual fundamentals (10-K filings) for a given ticker. We resolve
ticker -> CIK via the public mapping file, then hit the companyfacts XBRL endpoint
which returns every reported concept across every filing the company has ever made.

EDGAR docs: https://www.sec.gov/edgar/sec-api-documentation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import requests

from valuation.config import (
    EDGAR_COMPANYFACTS_URL,
    EDGAR_SUBMISSIONS_SIDECAR_URL,
    EDGAR_SUBMISSIONS_URL,
    EDGAR_TICKERS_URL,
    edgar_headers,
)

# Per-process cache so we hit the SEC ticker-mapping endpoint only once.
_TICKER_CACHE: dict[str, dict] | None = None


# Concept fallback chains. Different filers tag the same line item under
# slightly different us-gaap concepts; we try them in order and use the first
# chain that yields data.
CFO_CONCEPTS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByOperatingActivities",
]
CAPEX_CONCEPTS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
]
SHARES_CONCEPTS = [
    "CommonStockSharesOutstanding",
    "EntityCommonStockSharesOutstanding",
]
LONG_TERM_DEBT_CONCEPTS = [
    "LongTermDebtNoncurrent",
    "LongTermDebt",
]
SHORT_TERM_DEBT_CONCEPTS = [
    "LongTermDebtCurrent",
    "DebtCurrent",
]
CASH_CONCEPTS = [
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    "Cash",
]
EPS_CONCEPTS = [
    "EarningsPerShareDiluted",
    "EarningsPerShareBasic",
]
INTEREST_EXPENSE_CONCEPTS = [
    "InterestExpense",
    "InterestExpenseDebt",
]
PRETAX_INCOME_CONCEPTS = [
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
    "IncomeBeforeIncomeTaxes",
]
INCOME_TAX_EXPENSE_CONCEPTS = [
    "IncomeTaxExpense",
    "IncomeTaxExpenseBenefit",
]


@dataclass
class Fundamentals:
    """Cleaned annual fundamentals for a company."""

    ticker: str
    cik: int
    company_name: str
    fcf_history: list[tuple[int, float]] = field(default_factory=list)
    """List of (fiscal_year, free_cash_flow) sorted ascending."""
    eps_history: list[tuple[int, float]] = field(default_factory=list)
    """Diluted/basic GAAP EPS per fiscal year from 10-K (USD/share)."""
    ttm_fcf: float | None = None
    """Most recent trailing-12-month FCF inferred from 10-Q YTD bridges."""
    ttm_eps: float | None = None
    """Most recent trailing-12-month EPS inferred from 10-Q YTD bridges."""
    shares_outstanding: float = 0.0
    long_term_debt: float = 0.0
    short_term_debt: float = 0.0
    cash: float = 0.0
    interest_expense: float = 0.0
    """Latest FY interest expense from 10-K (USD), for pre-tax cost of debt proxy."""
    pretax_income: float = 0.0
    """Latest FY pretax income from 10-K (USD)."""
    income_tax_expense: float = 0.0
    """Latest FY income tax expense from 10-K (USD)."""

    @property
    def total_debt(self) -> float:
        return self.long_term_debt + self.short_term_debt

    @property
    def net_debt(self) -> float:
        """Debt minus cash. Negative means net cash position."""
        return self.total_debt - self.cash

    @property
    def fcf_values(self) -> list[float]:
        return [v for _, v in self.fcf_history]

    @property
    def eps_values(self) -> list[float]:
        return [v for _, v in self.eps_history]

    @property
    def latest_eps(self) -> float:
        if not self.eps_history:
            return 0.0
        return float(self.eps_history[-1][1])

    def avg_eps(self, years: int = 3) -> float:
        vals = self.eps_values
        if not vals:
            raise ValueError("No EPS history.")
        window = vals[-years:] if len(vals) >= years else vals
        return sum(window) / len(window)


class EdgarError(RuntimeError):
    pass


def _get(url: str, host_override: str | None = None) -> dict:
    headers = edgar_headers()
    if host_override:
        headers["Host"] = host_override
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise EdgarError(f"GET {url} -> HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()


def _load_ticker_map() -> dict[str, dict]:
    """Return a uppercase-ticker -> {cik_str, ticker, title} map, cached."""
    global _TICKER_CACHE
    if _TICKER_CACHE is not None:
        return _TICKER_CACHE
    raw = _get(EDGAR_TICKERS_URL, host_override="www.sec.gov")
    mapping: dict[str, dict] = {}
    for entry in raw.values():
        mapping[entry["ticker"].upper()] = entry
    _TICKER_CACHE = mapping
    return mapping


def lookup_cik(ticker: str) -> tuple[int, str]:
    """Resolve a ticker symbol to (CIK, company_name)."""
    tmap = _load_ticker_map()
    entry = tmap.get(ticker.upper())
    if not entry:
        raise EdgarError(f"Ticker {ticker!r} not found in SEC ticker mapping.")
    return int(entry["cik_str"]), entry["title"]


def fetch_company_facts(cik: int) -> dict:
    return _get(EDGAR_COMPANYFACTS_URL.format(cik=cik))


def fetch_submissions(cik: int) -> dict:
    """SEC company submissions JSON used for filings lists + filings metadata."""
    return _get(EDGAR_SUBMISSIONS_URL.format(cik=cik))


def filings_recent_parallel_block(filings: dict) -> dict:
    """Return the dict holding parallel filing arrays (`form`, `filingDate`, …).

    Live submissions nest these under ``filings[\"recent\"]``. Some fixtures omit
    that wrapper and place the arrays on ``filings`` itself.
    """
    recent = filings.get("recent")
    if isinstance(recent, dict):
        form = recent.get("form")
        fds = recent.get("filingDate")
        if isinstance(form, list) and isinstance(fds, list):
            return recent
    return filings


def submission_parallel_list(obj: dict, key: str) -> list:
    """Normalize SEC parallel-array fields to a Python list."""
    v = obj.get(key)
    if v is None:
        return []
    return list(v) if isinstance(v, list) else [v]


def form_filing_date_pairs_from_filings_blob(filings_obj: dict | None) -> list[tuple[str, str]]:
    """``(form, filingDate_raw)`` rows from ``filings`` subtree (handles ``recent`` nesting)."""
    if not isinstance(filings_obj, dict):
        return []
    block = filings_recent_parallel_block(filings_obj)
    fh = submission_parallel_list(block, "form")
    fds = submission_parallel_list(block, "filingDate")
    n = min(len(fh), len(fds))
    return [(str(fh[i]), str(fds[i])) for i in range(n)]


# Rolling IV requests enough supplemental shards to approximate ~5y of periodic filings
# when the primary payload is crowded by 8‑K etc. Tune via ``max_sidecars``.
DEFAULT_SUBMISSION_SIDECARS_FOR_ROLLING = 20


def submission_filings_form_dates_merged(
    cik: int,
    *,
    max_sidecars: int = DEFAULT_SUBMISSION_SIDECARS_FOR_ROLLING,
) -> tuple[list[str], list[str]]:
    """Combine primary submissions ``recent`` rows with historical ``filings.files`` shards.

    The SEC keeps roughly the latest thousand filing rows in ``recent``. Active issuers
    can exhaust that window with marginal forms, hiding older ``10‑Q``/``10‑K`` anchors.
    Supplemental JSON files reuse the same ``recent`` parallel-array schema.
    """
    root = fetch_submissions(cik)
    filings_obj = root["filings"] if isinstance(root.get("filings"), dict) else {}
    merged_pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def ingest(blob: dict | None) -> None:
        for form_s, fds in form_filing_date_pairs_from_filings_blob(blob):
            key = (form_s, (fds[:10] if fds else ""))
            if key in seen:
                continue
            seen.add(key)
            merged_pairs.append((form_s, fds))

    ingest(filings_obj)

    files_meta = filings_obj.get("files")
    if isinstance(files_meta, list) and max_sidecars > 0:
        lim = max(0, min(int(max_sidecars), len(files_meta)))
        for finfo in files_meta[:lim]:
            if not isinstance(finfo, dict):
                continue
            name = finfo.get("name")
            if not isinstance(name, str) or ".json" not in name.lower():
                continue
            side = _get(EDGAR_SUBMISSIONS_SIDECAR_URL.format(basename=name))
            blob = (
                side["filings"]
                if isinstance(side.get("filings"), dict)
                else (side if isinstance(side, dict) else None)
            )
            ingest(blob)

    if not merged_pairs:
        return [], []
    fh_o, fds_o = zip(*merged_pairs)
    return list(fh_o), list(fds_o)


def _annual_rows(facts: dict, concept: str, unit: str = "USD") -> list[dict]:
    """Return 10-K (annual, full-year) rows for a single us-gaap concept.

    Filters to ``form == "10-K"`` and ``fp == "FY"`` to ensure we only see
    audited, fiscal-year totals (not quarterly or amended values). Dedupes by
    fiscal year, keeping the latest filing for that year.
    """
    try:
        units = facts["facts"]["us-gaap"][concept]["units"]
    except KeyError:
        return []
    rows = units.get(unit) or units.get("USD/shares") or []
    annual = [r for r in rows if r.get("form") == "10-K" and r.get("fp") == "FY"]
    by_year: dict[int, dict] = {}
    for r in annual:
        fy = r.get("fy")
        if fy is None:
            continue
        existing = by_year.get(fy)
        # Keep the most recently filed value for each fiscal year (later 10-Ks
        # often restate prior years).
        if existing is None or r.get("filed", "") > existing.get("filed", ""):
            by_year[fy] = r
    return [by_year[fy] for fy in sorted(by_year)]


def _quarterly_rows(facts: dict, concept: str, unit: str = "USD") -> list[dict]:
    """Return deduped 10-Q YTD rows for Q1/Q2/Q3 keyed by (fy, fp)."""
    try:
        units = facts["facts"]["us-gaap"][concept]["units"]
    except KeyError:
        return []
    rows = units.get(unit) or units.get("USD/shares") or []
    quarterly = [
        r
        for r in rows
        if r.get("form") == "10-Q" and r.get("fp") in {"Q1", "Q2", "Q3"}
    ]
    by_key: dict[tuple[int, str], dict] = {}
    for r in quarterly:
        fy = r.get("fy")
        fp = r.get("fp")
        if fy is None or fp is None:
            continue
        k = (int(fy), str(fp))
        existing = by_key.get(k)
        if existing is None or r.get("filed", "") > existing.get("filed", ""):
            by_key[k] = r
    return sorted(by_key.values(), key=lambda r: (int(r["fy"]), str(r["fp"])))


def _first_concept_with_data(
    facts: dict, concepts: Iterable[str], unit: str = "USD"
) -> list[dict]:
    for c in concepts:
        rows = _annual_rows(facts, c, unit=unit)
        if rows:
            return rows
    return []


def _latest_value(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return float(rows[-1]["val"])


def _series(rows: list[dict]) -> dict[int, float]:
    return {int(r["fy"]): float(r["val"]) for r in rows}


def _annual_val_by_fy(annual_rows: list[dict]) -> dict[int, float]:
    return {int(r["fy"]): float(r["val"]) for r in annual_rows}


def _ttm_component_at_quarter(
    annual_rows: list[dict], quarterly_rows: list[dict], q_row: dict
) -> float | None:
    """TTM for one flow (CFO or CapEx) at quarter ``q_row``: FY[q_fy-1] + YTD_q − YTD_prior."""
    if not annual_rows or not quarterly_rows:
        return None
    ann = _annual_val_by_fy(annual_rows)
    q_fy = int(q_row["fy"])
    q_fp = str(q_row["fp"])
    base = ann.get(q_fy - 1)
    if base is None:
        return None
    prior = None
    for row in quarterly_rows:
        if int(row["fy"]) == q_fy - 1 and str(row["fp"]) == q_fp:
            prior = row
            break
    if prior is None:
        return None
    return base + float(q_row["val"]) - float(prior["val"])


def _ttm_fcf_by_quarter_key(facts: dict) -> dict[tuple[int, str], float]:
    """Map ``(fiscal_year, Qn)`` -> TTM FCF for each 10-Q anchor (CFO TTM − CapEx TTM)."""
    out: dict[tuple[int, str], float] = {}
    for c_cfo in CFO_CONCEPTS:
        cfo_a = _annual_rows(facts, c_cfo, unit="USD")
        cfo_q = _quarterly_rows(facts, c_cfo, unit="USD")
        if not cfo_a or not cfo_q:
            continue
        for c_cx in CAPEX_CONCEPTS:
            cx_a = _annual_rows(facts, c_cx, unit="USD")
            cx_q = _quarterly_rows(facts, c_cx, unit="USD")
            if not cx_a or not cx_q:
                continue
            for q_row in cfo_q:
                ttm_cfo = _ttm_component_at_quarter(cfo_a, cfo_q, q_row)
                if ttm_cfo is None:
                    continue
                k = (int(q_row["fy"]), str(q_row["fp"]))
                q_cx = None
                for r in cx_q:
                    if int(r["fy"]) == k[0] and str(r["fp"]) == k[1]:
                        q_cx = r
                        break
                if q_cx is None:
                    continue
                ttm_cx = _ttm_component_at_quarter(cx_a, cx_q, q_cx)
                if ttm_cx is None:
                    continue
                out[k] = ttm_cfo - ttm_cx
            if out:
                return out
    return out


def mean_ttm_fcf_yoy_growth_rates(facts: dict, *, max_rates: int = 3) -> float | None:
    """Mean of up to ``max_rates`` YoY TTM FCF changes at the same fiscal quarter (10-Q).

    Uses the most recent comparable YoY pairs first.
    """
    by_q = _ttm_fcf_by_quarter_key(facts)
    if len(by_q) < 2:
        return None
    fp_order = {"Q1": 1, "Q2": 2, "Q3": 3}
    keys = sorted(by_q.keys(), key=lambda x: (x[0], fp_order.get(x[1], 0)))
    rates: list[float] = []
    for fy, fp in reversed(keys):
        prev_k = (fy - 1, fp)
        if prev_k not in by_q:
            continue
        cur_v, prev_v = by_q[(fy, fp)], by_q[prev_k]
        if prev_v <= 0:
            continue
        rates.append((cur_v - prev_v) / prev_v)
        if len(rates) >= max_rates:
            break
    if not rates:
        return None
    return sum(rates) / len(rates)


def _ttm_from_10q_bridge(
    annual_rows: list[dict], quarterly_rows: list[dict]
) -> float | None:
    """Compute TTM from latest 10-K and two comparable 10-Q YTD rows.

    Formula: ``TTM = FY_last_10K + YTD_current_10Q - YTD_prior_year_same_quarter``.
    """
    if not annual_rows or not quarterly_rows:
        return None
    annual_latest = annual_rows[-1]
    annual_fy = int(annual_latest["fy"])
    q_latest = quarterly_rows[-1]
    q_fy = int(q_latest["fy"])
    q_fp = str(q_latest["fp"])
    if q_fy <= annual_fy:
        return None
    prior = None
    for row in quarterly_rows:
        if int(row["fy"]) == q_fy - 1 and str(row["fp"]) == q_fp:
            prior = row
    if prior is None:
        return None
    return (
        float(annual_latest["val"])
        + float(q_latest["val"])
        - float(prior["val"])
    )


def _concept_ttm_from_10q_bridge(
    facts: dict, concepts: Iterable[str], unit: str = "USD"
) -> float | None:
    for c in concepts:
        annual = _annual_rows(facts, c, unit=unit)
        quarterly = _quarterly_rows(facts, c, unit=unit)
        ttm = _ttm_from_10q_bridge(annual, quarterly)
        if ttm is not None:
            return ttm
    return None


def extract_eps_history(facts: dict) -> list[tuple[int, float]]:
    rows = _first_concept_with_data(facts, EPS_CONCEPTS, unit="USD/shares")
    if not rows:
        return []
    srs = [(int(r["fy"]), float(r["val"])) for r in rows]
    return srs


def extract_fcf_history(facts: dict) -> list[tuple[int, float]]:
    """Compute FCF = CFO - CapEx for each fiscal year present in both series."""
    cfo_rows = _first_concept_with_data(facts, CFO_CONCEPTS)
    capex_rows = _first_concept_with_data(facts, CAPEX_CONCEPTS)
    if not cfo_rows or not capex_rows:
        raise EdgarError(
            "Could not find both CFO and CapEx concepts in EDGAR companyfacts. "
            "The filer may use non-standard XBRL tags."
        )
    cfo = _series(cfo_rows)
    capex = _series(capex_rows)
    common_years = sorted(set(cfo) & set(capex))
    return [(y, cfo[y] - capex[y]) for y in common_years]


def extract_ttm_eps(facts: dict) -> float | None:
    """Return latest TTM EPS from 10-Q YTD bridge if available."""
    return _concept_ttm_from_10q_bridge(facts, EPS_CONCEPTS, unit="USD/shares")


def extract_ttm_fcf(facts: dict) -> float | None:
    """Return latest TTM FCF from 10-Q YTD bridges of CFO/CapEx concepts."""
    cfo_ttm = _concept_ttm_from_10q_bridge(facts, CFO_CONCEPTS, unit="USD")
    capex_ttm = _concept_ttm_from_10q_bridge(facts, CAPEX_CONCEPTS, unit="USD")
    if cfo_ttm is None or capex_ttm is None:
        return None
    return cfo_ttm - capex_ttm


def _latest_quarterly_shares(facts: dict) -> float:
    """Prefer most recent 10-Q point-in-time shares (FY + Q ordering)."""
    for c in SHARES_CONCEPTS:
        rows = _quarterly_rows(facts, c, unit="shares")
        if rows:
            return float(rows[-1]["val"])
    return 0.0


def extract_shares_outstanding(
    facts: dict, cik: int | None = None, *, submissions_fallback: bool = True
) -> float:
    """Return shares from EDGAR: latest 10-Q, else 10-K XBRL, else optional submissions JSON."""
    tenq = _latest_quarterly_shares(facts)
    if tenq > 0:
        return tenq

    rows = _first_concept_with_data(facts, SHARES_CONCEPTS, unit="shares")
    if rows:
        return _latest_value(rows)
    # Fallback: submissions endpoint exposes EntityCommonStockSharesOutstanding.
    if cik is not None and submissions_fallback:
        try:
            sub = _get(EDGAR_SUBMISSIONS_URL.format(cik=cik))
            v = sub.get("entityCommonStockSharesOutstanding") or 0
            fv = float(v)
            if fv > 0:
                return fv
        except EdgarError:
            pass
    return 0.0


def extract_balance_sheet(facts: dict) -> tuple[float, float, float]:
    """Return (long_term_debt, short_term_debt, cash) from latest 10-K."""
    lt = _latest_value(_first_concept_with_data(facts, LONG_TERM_DEBT_CONCEPTS))
    st = _latest_value(_first_concept_with_data(facts, SHORT_TERM_DEBT_CONCEPTS))
    cash = _latest_value(_first_concept_with_data(facts, CASH_CONCEPTS))
    return lt, st, cash


def extract_interest_and_tax_line_items(facts: dict) -> tuple[float, float, float]:
    """Latest FY interest expense, pretax income, income tax from 10-K XBRL."""
    interest = _latest_value(_first_concept_with_data(facts, INTEREST_EXPENSE_CONCEPTS))
    pretax = _latest_value(_first_concept_with_data(facts, PRETAX_INCOME_CONCEPTS))
    tax = _latest_value(_first_concept_with_data(facts, INCOME_TAX_EXPENSE_CONCEPTS))
    return interest, pretax, tax


def get_fundamentals(ticker: str) -> Fundamentals:
    """High-level entry point: ticker -> cleaned :class:`Fundamentals`."""
    cik, name = lookup_cik(ticker)
    facts = fetch_company_facts(cik)
    fcf = extract_fcf_history(facts)
    eps = extract_eps_history(facts)
    ttm_fcf = extract_ttm_fcf(facts)
    ttm_eps = extract_ttm_eps(facts)
    shares = extract_shares_outstanding(facts, cik=cik)
    if shares <= 0:
        # yfinance fills gaps when XBRL lacks a standard share tag (e.g. some filers).
        from valuation.data.market import try_yfinance_shares_outstanding

        ys = try_yfinance_shares_outstanding(ticker)
        if ys is not None and ys > 0:
            shares = ys
    lt, st, cash = extract_balance_sheet(facts)
    interest, pretax, tax_exp = extract_interest_and_tax_line_items(facts)
    return Fundamentals(
        ticker=ticker.upper(),
        cik=cik,
        company_name=name,
        fcf_history=fcf,
        eps_history=eps,
        ttm_fcf=ttm_fcf,
        ttm_eps=ttm_eps,
        shares_outstanding=shares,
        long_term_debt=lt,
        short_term_debt=st,
        cash=cash,
        interest_expense=interest,
        pretax_income=pretax,
        income_tax_expense=tax_exp,
    )
