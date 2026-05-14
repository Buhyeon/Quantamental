"""FMP market risk premium (mocked HTTP)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from valuation.data.fmp import FMPError, latest_market_risk_premium


def test_latest_market_risk_premium_parses_percent_points():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {"country": "United States", "totalEquityRiskPremium": "5.5"},
    ]
    with patch.dict("os.environ", {"FMP_API_KEY": "x"}):
        with patch("valuation.data.fmp.requests.get", return_value=mock_resp):
            v = latest_market_risk_premium()
    assert abs(v - 0.055) < 1e-9


def test_latest_market_risk_premium_prefers_us_row():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {"country": "Canada", "totalEquityRiskPremium": 3.0},
        {"country": "United States", "totalEquityRiskPremium": 6.0},
    ]
    with patch.dict("os.environ", {"FMP_API_KEY": "x"}):
        with patch("valuation.data.fmp.requests.get", return_value=mock_resp):
            v = latest_market_risk_premium()
    assert abs(v - 0.06) < 1e-9


def test_fmp_api_key_missing():
    import valuation.data.fmp as fmp_mod

    with patch.dict("os.environ", {"FMP_API_KEY": ""}):
        with pytest.raises(FMPError, match="FMP_API_KEY"):
            fmp_mod.fmp_api_key()
