"""Tests for the EDGAR XBRL parser.

We mock ``requests.get`` to return canned JSON resembling EDGAR responses, then
assert that our extractors filter to 10-K rows, dedupe by fiscal year, and apply
the expected fallback chains.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

# Set a dummy SEC_USER_AGENT BEFORE importing edgar so dotenv-loaded config
# doesn't reject us at import time.
os.environ.setdefault("SEC_USER_AGENT", "Test Runner test@example.com")

from valuation.data import edgar  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_ticker_cache():
    edgar._TICKER_CACHE = None
    yield
    edgar._TICKER_CACHE = None


def _row(fy: int, val: float, form: str = "10-K", fp: str = "FY", filed: str = "2024-01-01") -> dict:
    return {"fy": fy, "val": val, "form": form, "fp": fp, "filed": filed}


def _facts(concepts: dict[str, list[dict]], unit: str = "USD") -> dict:
    """Build a minimal companyfacts payload from a {concept: rows} mapping."""
    return {
        "facts": {
            "us-gaap": {
                concept: {"units": {unit: rows}}
                for concept, rows in concepts.items()
            }
        }
    }


def test_annual_rows_filters_non_10k_and_quarterly():
    facts = _facts({
        "NetCashProvidedByUsedInOperatingActivities": [
            _row(2021, 100, form="10-K", fp="FY"),
            _row(2022, 110, form="10-K", fp="FY"),
            _row(2022, 30,  form="10-Q", fp="Q1"),  # must be ignored
            _row(2023, 120, form="8-K",  fp="FY"),  # must be ignored
            _row(2023, 125, form="10-K", fp="FY"),
        ]
    })
    rows = edgar._annual_rows(facts, "NetCashProvidedByUsedInOperatingActivities")
    years = [r["fy"] for r in rows]
    vals = [r["val"] for r in rows]
    assert years == [2021, 2022, 2023]
    assert vals == [100, 110, 125]


def test_annual_rows_dedupes_by_fy_keeping_latest_filing():
    """When the same FY is reported in multiple 10-Ks (later restating the
    earlier one), we keep the most recently filed value."""
    facts = _facts({
        "NetCashProvidedByUsedInOperatingActivities": [
            _row(2022, 100, filed="2023-02-01"),
            _row(2022, 105, filed="2024-02-01"),  # restated, should win
        ]
    })
    rows = edgar._annual_rows(facts, "NetCashProvidedByUsedInOperatingActivities")
    assert [r["val"] for r in rows] == [105]


def test_concept_fallback_chain():
    """If the primary concept is missing, we fall back to alternates."""
    facts = _facts({
        # Primary CFO concept missing entirely.
        "NetCashProvidedByOperatingActivities": [_row(2023, 200)],
    })
    rows = edgar._first_concept_with_data(facts, edgar.CFO_CONCEPTS)
    assert [r["val"] for r in rows] == [200]


def test_extract_fcf_history_subtracts_capex():
    facts = _facts({
        "NetCashProvidedByUsedInOperatingActivities": [
            _row(2021, 100), _row(2022, 200), _row(2023, 300),
        ],
        "PaymentsToAcquirePropertyPlantAndEquipment": [
            _row(2021, 30), _row(2022, 60), _row(2023, 90),
        ],
    })
    fcf = edgar.extract_fcf_history(facts)
    assert fcf == [(2021, 70.0), (2022, 140.0), (2023, 210.0)]


def test_extract_fcf_history_intersects_years():
    """Years where one of CFO/CapEx is missing must be dropped from the history."""
    facts = _facts({
        "NetCashProvidedByUsedInOperatingActivities": [
            _row(2020, 50), _row(2021, 100), _row(2022, 200),
        ],
        "PaymentsToAcquirePropertyPlantAndEquipment": [
            _row(2021, 30), _row(2022, 60), _row(2023, 90),
        ],
    })
    fcf = edgar.extract_fcf_history(facts)
    assert [y for y, _ in fcf] == [2021, 2022]


def test_extract_fcf_history_raises_when_missing():
    facts = _facts({
        "NetCashProvidedByUsedInOperatingActivities": [_row(2023, 100)],
        # No CapEx concept at all.
    })
    with pytest.raises(edgar.EdgarError, match="CFO and CapEx"):
        edgar.extract_fcf_history(facts)


def test_extract_ttm_eps_uses_10q_bridge():
    facts = _facts(
        {
            "EarningsPerShareDiluted": [
                _row(2023, 8.0, form="10-K", fp="FY"),
                _row(2024, 2.0, form="10-Q", fp="Q1"),
                _row(2023, 1.6, form="10-Q", fp="Q1"),
            ]
        },
        unit="USD/shares",
    )
    assert edgar.extract_ttm_eps(facts) == pytest.approx(8.4)


def test_extract_ttm_fcf_uses_10q_bridge():
    facts = _facts(
        {
            "NetCashProvidedByUsedInOperatingActivities": [
                _row(2023, 400, form="10-K", fp="FY"),
                _row(2024, 120, form="10-Q", fp="Q1"),
                _row(2023, 90, form="10-Q", fp="Q1"),
            ],
            "PaymentsToAcquirePropertyPlantAndEquipment": [
                _row(2023, 100, form="10-K", fp="FY"),
                _row(2024, 35, form="10-Q", fp="Q1"),
                _row(2023, 20, form="10-Q", fp="Q1"),
            ],
        }
    )
    # CFO TTM = 400 + 120 - 90 = 430 ; CapEx TTM = 100 + 35 - 20 = 115 ; FCF = 315
    assert edgar.extract_ttm_fcf(facts) == pytest.approx(315.0)


def test_extract_ttm_eps_none_when_no_newer_10q():
    facts = _facts(
        {
            "EarningsPerShareDiluted": [
                _row(2023, 8.0, form="10-K", fp="FY"),
                _row(2023, 2.0, form="10-Q", fp="Q1"),
            ]
        },
        unit="USD/shares",
    )
    assert edgar.extract_ttm_eps(facts) is None


def test_extract_balance_sheet_uses_latest_value():
    facts = {
        "facts": {
            "us-gaap": {
                "LongTermDebtNoncurrent": {"units": {"USD": [
                    _row(2022, 1000), _row(2023, 1200),
                ]}},
                "LongTermDebtCurrent": {"units": {"USD": [
                    _row(2022, 100), _row(2023, 150),
                ]}},
                "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [
                    _row(2022, 500), _row(2023, 800),
                ]}},
            }
        }
    }
    lt, st, cash = edgar.extract_balance_sheet(facts)
    assert (lt, st, cash) == (1200.0, 150.0, 800.0)


def test_extract_shares_outstanding_uses_shares_unit():
    facts = {
        "facts": {
            "us-gaap": {
                "CommonStockSharesOutstanding": {"units": {"shares": [
                    _row(2022, 1_000_000_000), _row(2023, 1_050_000_000),
                ]}}
            }
        }
    }
    assert edgar.extract_shares_outstanding(facts) == 1_050_000_000.0


def test_extract_shares_outstanding_prefers_most_recent_10q():
    """Latest 10-Q share point should override older fiscal-year-only tags."""
    facts = {
        "facts": {
            "us-gaap": {
                "CommonStockSharesOutstanding": {"units": {"shares": [
                    _row(2023, 1_000_000_000),
                    _row(2025, 2_575_000_000, form="10-Q", fp="Q3", filed="2025-10-02"),
                    _row(2025, 2_550_000_000, form="10-Q", fp="Q2", filed="2025-07-01"),
                ]}}
            }
        }
    }
    assert edgar.extract_shares_outstanding(facts, cik=None) == 2_575_000_000.0


def test_get_fundamentals_yfinance_when_edgar_has_no_share_tag():
    """If EDGAR has no usable share count, fall back to yfinance."""
    tickers_payload = {"0": {"cik_str": 1, "ticker": "XX", "title": "Dummy Co"}}
    facts_payload = {
        "facts": {
            "us-gaap": {
                "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
                    _row(2022, 200), _row(2023, 300),
                ]}},
                "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
                    _row(2022, 60), _row(2023, 90),
                ]}},
                "EarningsPerShareDiluted": {"units": {"USD/shares": [_row(2023, 5.0)]}},
                "LongTermDebtNoncurrent": {"units": {"USD": [_row(2023, 10_000)]}},
                "LongTermDebtCurrent": {"units": {"USD": [_row(2023, 1_000)]}},
                "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [
                    _row(2023, 5_000),
                ]}},
            }
        }
    }

    def _fake_get(url, headers=None, timeout=None):
        class R:
            status_code = 200
            text = ""

            def __init__(self, payload): self._p = payload

            def json(self): return self._p

        if "company_tickers" in url:
            return R(tickers_payload)
        if "companyfacts" in url:
            return R(facts_payload)
        if "submissions" in url.lower():
            return R({"filings": {}})
        raise AssertionError(f"Unexpected URL: {url}")

    with (
        patch("valuation.data.edgar.requests.get", side_effect=_fake_get),
        patch(
            "valuation.data.market.try_yfinance_shares_outstanding",
            return_value=9.99e8,
        ),
    ):
        f = edgar.get_fundamentals("XX")

    assert f.shares_outstanding == 9.99e8


def test_lookup_cik_resolves_ticker():
    canned_tickers = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
    }

    class _Resp:
        status_code = 200
        text = ""
        def json(self): return canned_tickers

    with patch("valuation.data.edgar.requests.get", return_value=_Resp()):
        cik, name = edgar.lookup_cik("aapl")
    assert cik == 320193
    assert name == "Apple Inc."


def test_lookup_cik_unknown_ticker_raises():
    canned_tickers = {"0": {"cik_str": 1, "ticker": "AAPL", "title": "Apple Inc."}}

    class _Resp:
        status_code = 200
        text = ""
        def json(self): return canned_tickers

    with patch("valuation.data.edgar.requests.get", return_value=_Resp()):
        with pytest.raises(edgar.EdgarError, match="not found"):
            edgar.lookup_cik("ZZZZ")


def test_get_fundamentals_end_to_end_mocked():
    """Exercise the full ticker -> Fundamentals path with both endpoints mocked."""
    tickers_payload = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    facts_payload = {
        "facts": {
            "us-gaap": {
                "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
                    _row(2021, 100), _row(2022, 200), _row(2023, 300),
                    _row(2024, 130, form="10-Q", fp="Q1", filed="2024-05-01"),
                    _row(2023, 100, form="10-Q", fp="Q1", filed="2023-05-01"),
                ]}},
                "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
                    _row(2021, 30), _row(2022, 60), _row(2023, 90),
                    _row(2024, 25, form="10-Q", fp="Q1", filed="2024-05-01"),
                    _row(2023, 20, form="10-Q", fp="Q1", filed="2023-05-01"),
                ]}},
                "EarningsPerShareDiluted": {"units": {"USD/shares": [
                    _row(2023, 7.8),
                    _row(2024, 2.1, form="10-Q", fp="Q1", filed="2024-05-01"),
                    _row(2023, 1.7, form="10-Q", fp="Q1", filed="2023-05-01"),
                ]}},
                "CommonStockSharesOutstanding": {"units": {"shares": [
                    _row(2023, 16_000_000_000),
                ]}},
                "LongTermDebtNoncurrent": {"units": {"USD": [_row(2023, 100_000)]}},
                "LongTermDebtCurrent":    {"units": {"USD": [_row(2023, 10_000)]}},
                "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [
                    _row(2023, 50_000),
                ]}},
            }
        }
    }

    def _fake_get(url, headers=None, timeout=None):
        class R:
            status_code = 200
            text = ""
            def __init__(self, payload): self._p = payload
            def json(self): return self._p
        if "company_tickers" in url:
            return R(tickers_payload)
        if "companyfacts" in url:
            return R(facts_payload)
        raise AssertionError(f"Unexpected URL fetched: {url}")

    with patch("valuation.data.edgar.requests.get", side_effect=_fake_get):
        f = edgar.get_fundamentals("AAPL")

    assert f.ticker == "AAPL"
    assert f.cik == 320193
    assert f.company_name == "Apple Inc."
    assert f.fcf_values == [70.0, 140.0, 210.0]
    assert f.ttm_fcf == pytest.approx(235.0)  # (300+130-100) - (90+25-20)
    assert f.ttm_eps == pytest.approx(8.2)    # 7.8 + 2.1 - 1.7
    assert f.shares_outstanding == 16_000_000_000.0
    assert f.long_term_debt == 100_000.0
    assert f.short_term_debt == 10_000.0
    assert f.cash == 50_000.0
    assert f.net_debt == 60_000.0  # 100k + 10k - 50k


def test_submission_filings_form_dates_merged_pulls_sidecar_json():
    primary = {
        "filings": {
            "recent": {"form": ["10-K"], "filingDate": ["2023-01-01"]},
            "files": [{"name": "CIK0000000007-submissions-001.json"}],
        }
    }
    archival = {
        "filings": {
            "recent": {
                "form": ["10-Q", "10-K"],
                "filingDate": ["2022-06-01", "2024-06-01"],
            }
        }
    }

    def fake_get(url, host_override=None):
        if "submissions-001.json" in url:
            return archival
        raise AssertionError(f"Unexpected URL: {url}")

    with patch.object(edgar, "fetch_submissions", return_value=primary):
        with patch.object(edgar, "_get", side_effect=fake_get):
            fh, fds = edgar.submission_filings_form_dates_merged(7, max_sidecars=20)

    assert len(fh) == 3
    assert set(zip(fh, fds)) == {
        ("10-K", "2023-01-01"),
        ("10-Q", "2022-06-01"),
        ("10-K", "2024-06-01"),
    }
