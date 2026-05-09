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


def extract_shares_outstanding(facts: dict, cik: int | None = None) -> float:
    """Return shares from EDGAR: latest 10-Q, else 10-K XBRL, else submissions JSON."""
    tenq = _latest_quarterly_shares(facts)
    if tenq > 0:
        return tenq

    rows = _first_concept_with_data(facts, SHARES_CONCEPTS, unit="shares")
    if rows:
        return _latest_value(rows)
    # Fallback: submissions endpoint exposes EntityCommonStockSharesOutstanding.
    if cik is not None:
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
