"""8-K filing guidance extraction via SEC EDGAR.

Scans recent 8-Ks (last 12 months), fetches the primary HTML document and any
EX-99.x presentation exhibits, strips HTML to plain text, and applies lightweight
regex patterns for numeric growth guidance.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
import requests

from valuation.config import EDGAR_ARCHIVES_URL, sec_user_agent
from valuation.growth.types import GuidanceCandidate


class _HTMLToText(HTMLParser):
    """Minimal HTML -> text stripper (no external deps)."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        self._chunks.append(data)

    def get_text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._chunks)).strip()


def _strip_html(html: str) -> str:
    p = _HTMLToText()
    try:
        p.feed(html)
        p.close()
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)
    return p.get_text()


def _edgar_get_text(url: str) -> str:
    headers = {
        "User-Agent": sec_user_agent(),
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }
    resp = requests.get(url, headers=headers, timeout=45)
    resp.raise_for_status()
    return _strip_html(resp.text)


def _accession_to_no_dashes(accession: str) -> str:
    return accession.replace("-", "")


def _parse_list_field(obj: dict, key: str) -> list:
    v = obj.get(key)
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return list(v)


def recent_8k_filings(filings: dict, *, within_days: int = 365) -> list[dict]:
    """Filter the ``filings`` block of a submissions JSON to recent 8-K rows."""
    forms = _parse_list_field(filings, "form")
    dates = _parse_list_field(filings, "filingDate")
    accessions = _parse_list_field(filings, "accessionNumber")
    primary_docs = _parse_list_field(filings, "primaryDocument")

    cutoff = datetime.now(timezone.utc).date() - timedelta(days=within_days)
    out: list[dict] = []
    for form, fdate, acc, doc in zip(forms, dates, accessions, primary_docs):
        if form != "8-K":
            continue
        try:
            fd = datetime.strptime(fdate, "%Y-%m-%d").date()
        except ValueError:
            continue
        if fd < cutoff:
            continue
        out.append(
            {
                "filing_date": fdate,
                "accessionNumber": acc,
                "primaryDocument": doc,
            }
        )
    return out


_GUIDANCE_PATTERNS = [
    re.compile(
        r"(?:grow|growth|increase|expand|rise|accelerat(?:e|ing|ed)?)"
        r"[^\d%\n]{0,200}?\b(\d{1,2}(?:\.\d{1,4})?)\s*(?:%|percent(?:age)?(?:\s+points?)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:guidance|outlook|expect(?:s|ation|ed)?|anticipate(?:s|d)?)"
        r"[^\d%\n]{0,200}?\b(\d{1,2}(?:\.\d{1,4})?)\s*%",
        re.IGNORECASE,
    ),
]


def _extract_regex_candidates(
    text: str, *, citation: str, source: str = "8-K regex"
) -> list[GuidanceCandidate]:
    out: list[GuidanceCandidate] = []
    for pat in _GUIDANCE_PATTERNS:
        for m in pat.finditer(text):
            try:
                pct = float(m.group(1))
            except (ValueError, IndexError):
                continue
            if not (0.0 <= pct <= 65.0):
                continue
            dec = pct / 100.0
            lo, hi = max(0, m.start() - 40), min(len(text), m.end() + 40)
            snippet = text[lo:hi].strip()
            out.append(
                GuidanceCandidate(
                    growth_rate=dec,
                    source=source,
                    confidence="low",
                    metric="unspecified",
                    period="",
                    citation=citation,
                    snippet=snippet,
                )
            )
    return out


_EX99_RE = re.compile(r"^(EX-99\.\d+)", re.IGNORECASE)


def _exhibit_urls_from_primary_index(
    cik: int, accession_nd: str, primary_html: str, primary_url: str
) -> list[str]:
    """Find EX-99.x hrefs inside the filing index embedded in ``primary_html``."""

    urls: list[str] = []

    hrefs = re.findall(
        r'href="([^"]+)"',
        primary_html,
        flags=re.IGNORECASE,
    )
    for href in hrefs:
        basename = href.rsplit("/", 1)[-1].split("?", 1)[0]
        if _EX99_RE.match(basename or ""):
            if href.startswith("/"):
                full = "https://www.sec.gov" + href
            elif href.startswith("http"):
                full = href
            else:
                base = primary_url.rsplit("/", 1)[0]
                full = base + "/" + href
            urls.append(full)
    dedup = []
    seen: set[str] = set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup


def eight_k_guidance_candidates(
    cik: int, submissions_filings_block: dict, *, limit: int = 8
) -> list[GuidanceCandidate]:
    """Return regex guidance candidates from recent 8-K filing HTML + EX-99 exhibits."""
    recent = recent_8k_filings(submissions_filings_block)[:limit]
    out: list[GuidanceCandidate] = []

    for row in recent:
        acc = row["accessionNumber"]
        doc = row["primaryDocument"]
        fdate = row["filing_date"]
        acc_nd = _accession_to_no_dashes(acc)
        primary_url = EDGAR_ARCHIVES_URL.format(
            cik=cik, accession_no_dashes=acc_nd, filename=doc
        )
        try:
            primary_html = _edgar_get_text(primary_url)
        except Exception:
            continue

        cite = f"8-K {fdate} primary"
        out.extend(_extract_regex_candidates(primary_html, citation=cite))

        for ex_url in _exhibit_urls_from_primary_index(
            cik, acc_nd, primary_html, primary_url
        ):
            try:
                ex_text = _edgar_get_text(ex_url)
            except Exception:
                continue
            ex_name = ex_url.rsplit("/", 1)[-1]
            cite_ex = f"8-K {fdate} {ex_name}"
            out.extend(_extract_regex_candidates(ex_text, citation=cite_ex))

    return out


def eight_k_excerpts_for_llm(
    cik: int, submissions_filings_block: dict, *, limit: int = 3, max_chars: int = 12000
) -> str:
    """Concatenate truncated 8-K + EX-99 text for LLM consumption."""
    recent = recent_8k_filings(submissions_filings_block)[:limit]
    chunks: list[str] = []

    for row in recent:
        acc = row["accessionNumber"]
        doc = row["primaryDocument"]
        fdate = row["filing_date"]
        acc_nd = _accession_to_no_dashes(acc)
        primary_url = EDGAR_ARCHIVES_URL.format(
            cik=cik, accession_no_dashes=acc_nd, filename=doc
        )
        try:
            primary_html = _edgar_get_text(primary_url)
        except Exception:
            continue
        header = f"=== 8-K filed {fdate} primary document ===\n"
        chunks.append(header + primary_html[: max_chars // (limit + 1)])

        for ex_url in _exhibit_urls_from_primary_index(
            cik, acc_nd, primary_html, primary_url
        )[:2]:
            try:
                ex_text = _edgar_get_text(ex_url)
            except Exception:
                continue
            ex_name = ex_url.rsplit("/", 1)[-1]
            chunks.append(
                f"=== EXHIBIT {ex_name} from 8-K {fdate} ===\n"
                + ex_text[: max_chars // (limit * 2 + 1)]
            )

    text = "\n\n".join(chunks)
    return text[:max_chars]
