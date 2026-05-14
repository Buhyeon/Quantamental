"""Tests for yfinance wrappers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from valuation.data import market


@patch("valuation.data.market.yf.Ticker")
def test_try_yfinance_shares_outstanding_from_mapping(mock_tick):
    fake_fi = {"shares_outstanding": 2.5e9}
    mock_inst = MagicMock()
    mock_inst.fast_info = fake_fi
    mock_tick.return_value = mock_inst
    assert market.try_yfinance_shares_outstanding("META") == pytest.approx(2.5e9)


@patch("valuation.data.market.yf.Ticker")
def test_try_yfinance_shares_returns_none_when_empty(mock_tick):
    mock_inst = MagicMock()
    mock_inst.fast_info = {}
    mock_tick.return_value = mock_inst
    assert market.try_yfinance_shares_outstanding("ZZINVALID") is None


def test_format_compact_usd_trillion_billion():
    assert market.format_compact_usd(4.39e12) == "$4.39t"
    assert market.format_compact_usd(68.51e9) == "$68.51b"
    assert market.format_compact_usd(1.2e6) == "$1.20m"
    assert market.format_compact_usd(-5.9e9) == "-$5.90b"
    assert market.format_compact_usd(None) == "—"


def test_format_pct_from_decimal():
    assert market.format_pct_from_decimal(0.2715) == "27.15%"
    assert market.format_pct_from_decimal(None) == "—"


def test_format_pe_triple():
    assert market.format_pe_triple(50.0, 40.0, 100.0, 2.5) == "50.00 | 40.00 | —"
    assert (
        market.format_pe_triple(36.08, 33.11, 298.89, 9.55, diff_threshold=0.5)
        == "36.08 | 33.11 | 31.30"
    )
    assert (
        market.format_pe_triple(10.0, None, 100.0, 5.0, diff_threshold=0.5)
        == "10.00 | — | 20.00"
    )
    assert market.format_pe_triple(None, None, None, None) == "— | — | —"


@patch("valuation.data.market.yf.Ticker")
def test_fetch_yfinance_info_returns_dict(mock_tick):
    mock_inst = MagicMock()
    mock_inst.info = {"marketCap": 1e12}
    mock_tick.return_value = mock_inst
    assert market.fetch_yfinance_info("AAPL") == {"marketCap": 1e12}


@patch("valuation.data.market.yf.Ticker")
def test_fetch_yfinance_info_empty_on_bad_ticker(mock_tick):
    mock_inst = MagicMock()
    mock_inst.info = None
    mock_tick.return_value = mock_inst
    assert market.fetch_yfinance_info("X") == {}
