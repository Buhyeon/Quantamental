"""WACC estimator (mocked Yahoo / Rf)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from valuation.data.edgar import Fundamentals
from valuation.models import wacc_estimate as w


def _sample_f() -> Fundamentals:
    return Fundamentals(
        ticker="TST",
        cik=1,
        company_name="Test Co",
        shares_outstanding=1_000_000_000.0,
        long_term_debt=10e9,
        short_term_debt=2e9,
        cash=3e9,
        interest_expense=600e6,
        pretax_income=40e9,
        income_tax_expense=8e9,
    )


@pytest.fixture
def sample_f() -> Fundamentals:
    return _sample_f()


def test_marginal_tax_from_edgar(sample_f: Fundamentals):
    t = w.marginal_tax_rate(sample_f)
    assert abs(t - 0.2) < 0.01


def test_estimate_wacc_structure(sample_f: Fundamentals):
    with (
        patch.object(w, "risk_free_10y", return_value=0.04),
        patch.object(w, "equity_beta", return_value=1.2),
        patch("valuation.config.equity_risk_premium", return_value=0.05),
    ):
        bd = w.estimate_wacc("TST", sample_f, 200.0)
    assert 0.04 < bd.wacc < 0.30
    assert bd.weight_equity + bd.weight_debt == pytest.approx(1.0)


def test_estimate_wacc_equity_only_when_no_debt():
    f = _sample_f()
    f = Fundamentals(
        ticker=f.ticker,
        cik=f.cik,
        company_name=f.company_name,
        shares_outstanding=f.shares_outstanding,
        long_term_debt=0,
        short_term_debt=0,
        cash=f.cash,
        interest_expense=0,
        pretax_income=f.pretax_income,
        income_tax_expense=f.income_tax_expense,
    )
    with (
        patch.object(w, "risk_free_10y", return_value=0.04),
        patch.object(w, "equity_beta", return_value=1.0),
        patch("valuation.config.equity_risk_premium", return_value=0.055),
    ):
        bd = w.estimate_wacc("TST", f, 100.0)
    assert bd.weight_debt == 0.0
    assert bd.wacc == pytest.approx(bd.cost_of_equity)
