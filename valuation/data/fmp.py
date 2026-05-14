"""Financial Modeling Prep (FMP) HTTP client — market risk premium only (no analyst estimates)."""

from __future__ import annotations

import os
from typing import Any

import requests

FMP_V4_BASE = "https://financialmodelingprep.com/api/v4"
DEFAULT_TIMEOUT_S = 25.0


class FMPError(RuntimeError):
    pass


def fmp_api_key() -> str:
    key = os.getenv("FMP_API_KEY", "").strip()
    if not key:
        raise FMPError("FMP_API_KEY is not set in the environment.")
    return key


def _get_json(url: str, params: dict[str, Any]) -> Any:
    p = {**params, "apikey": fmp_api_key()}
    resp = requests.get(url, params=p, timeout=DEFAULT_TIMEOUT_S)
    if resp.status_code != 200:
        raise FMPError(f"GET {url} -> HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()


def latest_market_risk_premium() -> float:
    """Latest total equity market risk premium (decimal), US row when available.

    Endpoint: ``/api/v4/market_risk_premium``. Values are often in *percent points*
    (e.g. ``5.2`` meaning 5.2%); we normalize to decimal.
    """
    url = f"{FMP_V4_BASE}/market_risk_premium"
    data = _get_json(url, {})
    if not isinstance(data, list) or not data:
        raise FMPError("Empty market_risk_premium response.")
    row = None
    for r in data:
        if isinstance(r, dict):
            c = str(r.get("country", "")).lower()
            if c in ("united states", "usa", "us"):
                row = r
                break
    if row is None:
        row = data[0]
    if not isinstance(row, dict):
        raise FMPError("Unexpected market_risk_premium row shape.")
    raw = row.get("totalEquityRiskPremium")
    if raw is None:
        raw = row.get("equityRiskPremium")
    if raw is None:
        raise FMPError("No totalEquityRiskPremium in market_risk_premium row.")
    v = float(raw)
    if v > 1.0:
        v = v / 100.0
    return max(0.02, min(0.15, v))
