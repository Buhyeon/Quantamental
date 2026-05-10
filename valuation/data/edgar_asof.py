"""Point-in-time EDGAR companyfacts: filter XBRL rows by filing date."""

from __future__ import annotations

import copy
from datetime import date

from valuation.data.edgar import (
    Fundamentals,
    extract_balance_sheet,
    extract_eps_history,
    extract_fcf_history,
    extract_interest_and_tax_line_items,
    extract_shares_outstanding,
    extract_ttm_eps,
    extract_ttm_fcf,
    fetch_company_facts,
    lookup_cik,
)


def _filed_on_or_before(row: dict, as_of: date) -> bool:
    fd = row.get("filed")
    if not fd or not isinstance(fd, str):
        return False
    return fd[:10] <= as_of.isoformat()


def filter_companyfacts_as_of(facts: dict, as_of: date) -> dict:
    """Return a deep copy of ``facts`` keeping only rows with ``filed`` on or before ``as_of``.

    Rows without a ``filed`` field are dropped (conservative for backtesting).
    """
    out = copy.deepcopy(facts)
    try:
        gaap = out["facts"]["us-gaap"]
    except KeyError:
        return out
    for _concept, payload in gaap.items():
        units = payload.get("units") or {}
        for unit_key, rows in list(units.items()):
            if not isinstance(rows, list):
                continue
            units[unit_key] = [r for r in rows if _filed_on_or_before(r, as_of)]
    return out


def get_fundamentals_as_of(ticker: str, as_of: date) -> Fundamentals:
    """Build :class:`Fundamentals` using only facts known by ``as_of`` (SEC ``filed`` date).

    Does not use Yahoo or submissions JSON fallbacks for shares (EDGAR XBRL only).
    """
    cik, name = lookup_cik(ticker)
    raw = fetch_company_facts(cik)
    facts = filter_companyfacts_as_of(raw, as_of)
    fcf = extract_fcf_history(facts)
    eps = extract_eps_history(facts)
    ttm_fcf = extract_ttm_fcf(facts)
    ttm_eps = extract_ttm_eps(facts)
    shares = extract_shares_outstanding(facts, cik=cik, submissions_fallback=False)
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
