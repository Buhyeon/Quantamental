"""Default assumptions and EDGAR HTTP configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DOTENV_PROJECT_PATH = _PROJECT_ROOT / ".env"


def _strip_bom_prefixed_env_keys() -> int:
    """Rename env vars that start with a UTF-8 BOM (common when `.env` was saved with BOM)."""
    n = 0
    for k in list(os.environ.keys()):
        if k.startswith("\ufeff"):
            nk = k.lstrip("\ufeff")
            if nk:
                os.environ[nk] = os.environ[k]
                n += 1
            del os.environ[k]
    return n


# Load bundled project `.env` first so API keys resolve even when cwd is elsewhere;
# optional second pass overlays a cwd-local `.env` so developers can override.
# `utf-8-sig` strips a leading BOM from the file itself when present.
load_dotenv(_DOTENV_PROJECT_PATH, encoding="utf-8-sig")
load_dotenv(encoding="utf-8-sig")
_strip_bom_prefixed_env_keys()


# DCF defaults. Overrideable via CLI flags or by passing kwargs to the model.
DEFAULT_GROWTH_RATE: float = 0.08
DEFAULT_WACC: float = 0.09
DEFAULT_TERMINAL_GROWTH: float = 0.034
DEFAULT_PROJECTION_YEARS: int = 10
# When 10-Q TTM bridging is unavailable, `--fcf-avg-years` averages trailing FY
# 10-K anchors (default 1 = latest FY only; use 3+ for smoothed averages).
DEFAULT_FCF_AVG_YEARS: int = 1

# CAPM / WACC estimation (--auto-wacc). ERP can be overridden via EQUITY_RISK_PREMIUM in ``.env``.
DEFAULT_EQUITY_RISK_PREMIUM: float = 0.055
DEFAULT_RISK_FREE_FALLBACK: float = 0.045
DEFAULT_DEBT_SPREAD_OVER_RF: float = 0.025
STATUTORY_US_CORP_TAX_RATE: float = 0.21
WACC_CLIP_LO: float = 0.04
WACC_CLIP_HI: float = 0.30


def equity_risk_premium() -> float:
    """ERP as decimal (e.g. 0.055 = 5.5%). Overridable via ``EQUITY_RISK_PREMIUM``."""
    raw = os.getenv("EQUITY_RISK_PREMIUM", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_EQUITY_RISK_PREMIUM


# SEC EDGAR HTTP configuration.
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
# Supplemental filings JSON when ``recent`` exceeds ~1000 rows (basename from ``filings.files``).
EDGAR_SUBMISSIONS_SIDECAR_URL = "https://data.sec.gov/submissions/{basename}"
EDGAR_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{filename}"


def sec_user_agent() -> str:
    """Return the User-Agent header value for SEC EDGAR requests.

    SEC requires identifying contact info or they rate-limit / block requests.
    """
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if not ua or "@" not in ua:
        raise RuntimeError(
            "SEC_USER_AGENT not configured. Copy .env.example to .env and set "
            'SEC_USER_AGENT="Your Name your.email@example.com".'
        )
    return ua


def edgar_headers() -> dict[str, str]:
    return {
        "User-Agent": sec_user_agent(),
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }


@dataclass(frozen=True)
class DCFAssumptions:
    """Bundle of DCF inputs the user can tune."""

    growth_rate: float = DEFAULT_GROWTH_RATE
    wacc: float = DEFAULT_WACC
    terminal_growth: float = DEFAULT_TERMINAL_GROWTH
    projection_years: int = DEFAULT_PROJECTION_YEARS

    def validate(self) -> None:
        if self.wacc <= self.terminal_growth:
            raise ValueError(
                f"WACC ({self.wacc:.2%}) must exceed terminal growth "
                f"({self.terminal_growth:.2%}); otherwise terminal value diverges."
            )
        if self.projection_years < 1:
            raise ValueError("projection_years must be >= 1")


def average_fcf(history: Iterable[float], years: int = DEFAULT_FCF_AVG_YEARS) -> float:
    """Return trailing-N FY average FCF when TTM isn't available.

    With ``years == 1`` this is latest fiscal year only. Larger N smooths
    one-off swings. Fewer FY points available than requested uses all points.
    """
    values = list(history)
    if not values:
        raise ValueError("Cannot average an empty FCF history.")
    window = values[-years:] if len(values) >= years else values
    return sum(window) / len(window)
