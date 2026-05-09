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
