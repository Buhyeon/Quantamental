# DCF Intrinsic Value Engine

A Python toolkit that combines **audited SEC EDGAR fundamentals** (via the official
bulk APIs) with **yfinance quotes**, then runs one or two **discounted-cash-flow**

- **FCF DCF** — classic enterprise value bridge (FCF → EV → equity / share)
- **EPS DCF** — per-share Gordon-style flow (EPS → discounted stream + terminal EPS)
- **Most-recent anchors** — base FCF/EPS use **TTM** (10-Q YTD bridge when
  available); otherwise **latest FY** from 10-K (use `--fcf-avg-years N` for
  multi-year FY smoothing)
- **`--auto-growth`** — blends **8-K / EX‑99 regex** signals, **Yahoo Finance
  summary consensus** (`forwardEps` vs trailing EPS implied growth and
  `revenueGrowth`), and **historical FCF CAGR** via a weighted‑median /
  IQR‑cleaned blender (see `valuation/growth/`).

## Why

The tape shows *price*. Your model shows *economic* *value*. The spread is the margin
of safety (or your red flag).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
copy .env.example .env
```

### Streamlit dashboard (optional)

Interactive charts for intrinsic vs market price, growth blend, optional WACC
breakdown, and an FCF sensitivity heatmap.

```bash
streamlit run streamlit_app.py
```

Serve from the **repository root** so `valuation` imports resolve. **`SEC_USER_AGENT`**
still required in `.env` for EDGAR.

### Environment variables (`/.env`)

| Key | Purpose |
| --- | ------- |
| `SEC_USER_AGENT` | **Mandatory** EDGAR contact string (`Full Name email@fqdn`) |
| `EQUITY_RISK_PREMIUM` | Optional decimal ERP for `--auto-wacc` CAPM (default **0.055**) |

## Usage

### Core DCF (manual growth assumption)

```bash
python -m valuation AAPL
python -m valuation MSFT --growth 0.08 --wacc 0.09 --terminal 0.025
```

### Auto growth (uses Yahoo `.info`, no transcript or LLM)

Blends 8‑K regex signals, Yahoo summary consensus fields, and FCF CAGR. **`--growth` is ignored.**

```bash
python -m valuation META --auto-growth --model both
python -m valuation AAPL MSFT GOOGL --auto-growth
```

### Auto WACC (CAPM + book capital structure)

Uses **10Y Treasury (^TNX)** as **R<sub>f</sub>**, **Yahoo beta** from `ticker.info`, and
**ERP** (default 5.5%, overridable via **`EQUITY_RISK_PREMIUM`**). **Cost of debt**
prefers **interest expense ÷ total debt** from the latest FY **10-K**; otherwise
**R<sub>f</sub> + spread**. Weights: **E = price × shares**, **D = book liabilities**
(EDGAR). **`--wacc` is ignored when `--auto-wacc` is set.**

```bash
python -m valuation META --auto-wacc --auto-growth
```

### Model selection

| `--model` | Meaning |
|-----------|---------|
| `fcf` (default) | FCF equity bridge only |
| `eps` | Per-share EPS DCF (no EV / net debt step) |
| `both` | Prints both + **simple arithmetic mean consensus** |

Override EPS projection growth while keeping FCF on auto/manual:

```bash
python -m valuation MSFT --model both --eps-growth 0.06
python -m valuation MSFT --model both --auto-growth --eps-growth 0.06
```

### Phase 3 (still supported)

Batch tickers + optional **FCF-only** Monte Carlo / sensitivity grids:

```bash
python -m valuation AAPL MSFT GOOGL
python -m valuation AAPL --sensitivity --sens-steps 5 --monte-carlo 2000 --mc-seed 42
```

> `--sensitivity` / `--monte-carlo` operate on the **FCF engine** (`--model fcf`
> or `--model both`). EPS-only prints a warning stub.

### Example output excerpt

```
MegaCorp Inc (MEGA)
  Base FCF (TTM):       $4.8B
  Base EPS (TTM):       $6.10/shr
  Shares outstanding:       1.20B
  Net debt:                 $8.9B

  Expected growth: 10.8% (weighted_median+IQR_outliers, 7 raw candidates)
    - financial_consensus: +10.6% metric=eps | yahoo forward vs trailing EPS | \"fwd=… trail=…\"
    - financial_consensus: +18.7% metric=revenue | yahoo revenueGrowth | \"revenueGrowth=…\"
    - fcf_cagr: +9.4% | historical FCF series from EDGAR 10-K | raw CAGR=+9.38%
    - 8-K regex: +8.0% | 8-K 2025-08-01 ex99-1.htm | \"operating margin improvement of 8%\"

  DCF inputs: growth(FCF/EPS-shared)=10.79% ...
  FCF intrinsic/share:       $187.54
  EPS intrinsic/share (direct): $173.92
  Consensus IV (simple mean): $180.73
```

## Architecture

```
valuation/
├── data/edgar.py        # XBRL fundamentals + EPS + filings JSON helpers
├── data/market.py       # Yahoo quotes + share-count fallback (``fast_info``)
├── growth/
│   ├── consensus.py     # Yahoo ``.info`` (EPS implied growth, revenueGrowth)
│   ├── eight_k.py       # EDGAR 8‑K / exhibit regex guidance
│   └── (estimate, blend, historical, types)
├── models/
│   ├── dcf.py           # FCF → EV equity bridge
│   ├── eps_dcf.py       # EPS per-share perpetual style DCF
│   ├── wacc_estimate.py # CAPM / book-structure WACC
│   └── sensitivity.py   # Scenario grids / Monte Carlo
├── config.py            # Defaults + `.env` readers
└── cli.py               # Typer-ready argparse façade
```

## Defaults

| Knob | Default | Sanity band |
| ---- | ------- | ----------- |
| Manual `--growth` | 8 % | Tune per company risk |
| `--wacc` | 9 % | Manual only; use **`--auto-wacc`** for CAPM-based WACC |
| Terminal `g` | 2.5 % | Must stay `< wacc` |
| Horizon `--years` | 5 yrs | Matches original spec |
| Base FCF / EPS | **TTM** from `10-Q` when possible; else trailing **FY** average via `--fcf-avg-years` (default **1** = latest FY) | Shared by FCF **and** EPS |

## Running tests

```bash
pytest valuation/tests -q
```

## Limitations / caveats

- **WACC** is manual (no CAPM helper yet — same as milestone 1 scope).
- **`--auto-growth` Yahoo fields** (`ticker.info`) are scraped summaries, not EDGAR;
  they often lag and can omit tickers Yahoo doesn’t summarize well.
- **`--auto-wacc`**: debt is **book** value (not market value of bonds); **beta** is
  Yahoo’s packaged estimate; **^TNX** and **interest expense** tags can misfire for
  some filers — treat as a screening discount rate, not a funding memo.
- **Shares** for FCF / `both`: latest **10-Q** from EDGAR, then **10-K** / submissions
  JSON, then **yfinance** `fast_info` if still missing (Yahoo can disagree with SEC).
- 8-K **PDF exhibits** aren't parsed yet (regex only hits HTML-ish text blobs).
- **Forward guidance extraction is probabilistic**. Always read the attribution
  lines before trusting the blender output.
