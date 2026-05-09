"""Yahoo-derived consensus candidates (mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import valuation.growth.consensus as cons


@patch.object(cons.yf, "Ticker")
def test_consensus_eps_forward_vs_trailing_and_revenue(mock_tick):
    ti = MagicMock()
    ti.info = {
        "forwardEps": 20.0,
        "trailingEps": 16.0,
        "revenueGrowth": 0.125,
    }
    mock_tick.return_value = ti

    cands = cons.yahoo_financial_consensus_candidates("ABCD")

    eps = next(x for x in cands if x.metric == "eps")
    rev = next(x for x in cands if x.metric == "revenue")

    assert eps.source == "financial_consensus"
    assert abs(eps.growth_rate - (20.0 / 16.0 - 1.0)) < 1e-9
    assert rev.growth_rate == 0.125


@patch.object(cons.yf, "Ticker")
def test_consensus_empty_on_fetch_error(mock_tick):
    mock_tick.side_effect = OSError("network")
    assert cons.yahoo_financial_consensus_candidates("X") == []
