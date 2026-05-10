"""Point-in-time companyfacts filtering."""

from __future__ import annotations

from datetime import date

from valuation.data.edgar_asof import filter_companyfacts_as_of


def test_filter_companyfacts_as_of_drops_future_filings():
    facts = {
        "facts": {
            "us-gaap": {
                "Foo": {
                    "units": {
                        "USD": [
                            {"filed": "2019-03-01", "val": 1.0, "fy": 2018},
                            {"filed": "2024-03-01", "val": 2.0, "fy": 2023},
                        ]
                    }
                }
            }
        }
    }
    out = filter_companyfacts_as_of(facts, date(2020, 6, 1))
    rows = out["facts"]["us-gaap"]["Foo"]["units"]["USD"]
    assert len(rows) == 1
    assert rows[0]["val"] == 1.0


def test_filter_drops_rows_without_filed():
    facts = {
        "facts": {
            "us-gaap": {
                "Foo": {
                    "units": {
                        "USD": [
                            {"val": 1.0},
                            {"filed": "2019-01-01", "val": 2.0},
                        ]
                    }
                }
            }
        }
    }
    out = filter_companyfacts_as_of(facts, date(2020, 1, 1))
    rows = out["facts"]["us-gaap"]["Foo"]["units"]["USD"]
    assert len(rows) == 1
    assert rows[0]["val"] == 2.0


def test_filter_handles_missing_us_gaap():
    facts = {"facts": {}}
    assert filter_companyfacts_as_of(facts, date(2020, 1, 1)) == {"facts": {}}
