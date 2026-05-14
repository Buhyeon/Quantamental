"""Command-line entry point: ``python -m valuation TICKER [TICKER ...] [flags]``."""

from __future__ import annotations

import argparse
import random
import sys
import warnings
from dataclasses import replace

from valuation.config import (
    DCFAssumptions,
    DEFAULT_FCF_AVG_YEARS,
    DEFAULT_GROWTH_RATE,
    DEFAULT_PROJECTION_YEARS,
    DEFAULT_TERMINAL_GROWTH,
    DEFAULT_TERMINAL_PERIOD_YEARS,
    DEFAULT_WACC,
    average_fcf,
)
from valuation.data.edgar import EdgarError, fetch_company_facts, get_fundamentals
from valuation.data.market import MarketDataError, get_current_price
from valuation.growth.estimate import AutoGrowthMode, compute_growth_estimate
from valuation.growth.types import GuidanceCandidate, GrowthEstimate
from valuation.models.dcf import intrinsic_value_per_share, intrinsic_value_per_share_explicit_decay
from valuation.models.eps_dcf import intrinsic_price_from_eps, intrinsic_price_from_eps_explicit_decay
from valuation.models.wacc_estimate import WACCBreakdown, estimate_wacc
from valuation.models.sensitivity import (
    format_sensitivity_grid_table,
    intrinsic_sensitivity_grid,
    monte_carlo_intrinsic,
)


def _fmt_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    a = abs(x)
    if a >= 1e9:
        return f"{sign}${a/1e9:,.1f}B"
    if a >= 1e6:
        return f"{sign}${a/1e6:,.1f}M"
    if a >= 1e3:
        return f"{sign}${a/1e3:,.1f}K"
    return f"{sign}${a:,.2f}"


def _fmt_shares(x: float) -> str:
    if x >= 1e9:
        return f"{x/1e9:,.2f}B"
    if x >= 1e6:
        return f"{x/1e6:,.2f}M"
    return f"{x:,.0f}"


def _group_key(c: GuidanceCandidate) -> str:
    lab = (c.metric or "generic").strip() or "generic"
    return f"{c.source}:{lab}:{c.snippet[:22]}"


def candidate_sort_weight(c: GuidanceCandidate) -> float:
    return {
        "fcf_cagr": 2.5,
        "financial_consensus": 2.2,
        "8-K regex": 1.8,
        "yahoo_analyst_blend": 2.4,
        "yahoo_eps_forward": 2.4,
        "sec_fcf_fy_yoy": 2.3,
        "sec_fcf_ttm_yoy": 2.3,
    }.get(c.source, 1.0)


def _print_auto_wacc(wb: WACCBreakdown) -> None:
    print("  WACC (auto CAPM, market-style debt proxy):")
    print(
        f"    Rf (10Y ^TNX): {wb.risk_free:.2%}  Beta: {wb.beta:.2f}  "
        f"MRP: {wb.equity_risk_premium:.2%} ({wb.equity_risk_premium_source})  "
        f"-> Re (CAPM): {wb.cost_of_equity:.2%}"
    )
    print(
        f"    Rd pretax: {wb.cost_of_debt_pretax:.2%} "
        f"({wb.debt_cost_source})  Marginal T: {wb.marginal_tax_rate:.1%}"
    )
    print(
        f"    We: {wb.weight_equity:.1%}  Wd: {wb.weight_debt:.1%}  "
        f"(E mkt {_fmt_money(wb.market_value_equity)}  "
        f"D mkt {_fmt_money(wb.market_value_debt)} [{wb.debt_market_proxy_source}]  "
        f"D book {_fmt_money(wb.book_value_debt)})"
    )
    print(f"    WACC used (clipped): {wb.wacc:.2%}")


def _print_auto_growth(ge: GrowthEstimate) -> None:
    fb_note = " (includes default/CAGR fallback)" if ge.fallback_used else ""
    print(
        f"  Expected growth: {ge.growth_rate:.1%} ({ge.method}, "
        f"{len(ge.candidates)} raw candidates){fb_note}"
    )
    if not ge.candidates:
        print(
            "    (no scraped candidates — defaulted to CAGR / global default anchor)"
        )
        return
    seen: set[str] = set()
    rows = sorted(
        ge.candidates, key=lambda x: (-candidate_sort_weight(x), x.source, x.metric)
    )
    for c in rows:
        k = _group_key(c)
        if k in seen:
            continue
        seen.add(k)
        pct = f"{c.growth_rate:+.2%}"
        tail = ""
        if c.source == "financial_consensus":
            tail = f" metric={c.metric}"
            if c.period:
                tail += f" | {c.period}"
        cite = (c.citation or "")[:72].replace("\n", " ")
        snip = (c.snippet or "")[:92].replace("\n", " ")
        line = f"    - {c.source}: {pct}{tail}"
        if cite:
            line += f" | {cite}"
        if snip:
            line += f' | "{snip}"'
        print(line)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="valuation",
        description=(
            "DCF intrinsic equity value from SEC EDGAR + yfinance. "
        "`--auto-growth` estimates growth (SEC+FMP, legacy blended, or Yahoo-only). "
            "`--model fcf|eps|both` swaps the numerator (FCF vs EPS)."
        ),
    )
    p.add_argument(
        "ticker",
        nargs="+",
        help="One or more ticker symbols (e.g. AAPL MSFT GOOGL)",
    )
    p.add_argument(
        "--auto-growth",
        action="store_true",
        help=(
            "Estimated growth (`--growth` ignored): blended sources or Yahoo-only via "
            "`--auto-growth-mode`. Explicit forecast uses decay after year 3."
        ),
    )
    p.add_argument(
        "--auto-growth-mode",
        choices=("fcf_sec_yahoo", "blended_fcf", "eps_yahoo", "blended", "consensus"),
        default="fcf_sec_yahoo",
        help=(
            "fcf_sec_yahoo: SEC FY+TTM FCF growth blended with Yahoo 1y forward EPS+revenue (default). "
            "blended_fcf: SEC trailing FCF only (FY YoY + TTM YoY), no Yahoo. "
            "eps_yahoo: Yahoo 1y EPS forward growth for the EPS DCF arm; FCF arm still uses fcf_sec_yahoo. "
            "blended: legacy FY CAGR + 8-K + Yahoo. consensus: Yahoo ticker.info only."
        ),
    )
    p.add_argument(
        "--fcf-cagr-years",
        type=int,
        default=2,
        metavar="N",
        help=(
            "FY points for FCF CAGR in blended mode (default 2). "
            "Use 0 for full EDGAR FY history."
        ),
    )
    p.add_argument(
        "--model",
        choices=("fcf", "eps", "both"),
        default="fcf",
        help="Which DCF numerator to discount (FCF EV bridge vs EPS/share vs both).",
    )
    p.add_argument(
        "--growth",
        type=float,
        default=DEFAULT_GROWTH_RATE,
        help=(
            "Manual FCF-driven growth assumption (ignored under --auto-growth). "
            f"Default {DEFAULT_GROWTH_RATE}."
        ),
    )
    p.add_argument(
        "--eps-growth",
        type=float,
        default=None,
        metavar="RATE",
        help="Separate growth override for EPS DCF only (otherwise matches FCF path).",
    )
    p.add_argument(
        "--wacc",
        type=float,
        default=DEFAULT_WACC,
        help=(
            "Unified discount rate when not using --auto-wacc "
            f"(default {DEFAULT_WACC}). Ignored if --auto-wacc is set."
        ),
    )
    p.add_argument(
        "--auto-wacc",
        action="store_true",
        help=(
            "Compute WACC from CAPM (Rf=^TNX, beta from Yahoo; MRP from FMP when configured) "
            "with market-cap equity and debt proxy from enterprise value minus market cap "
            "(falls back to EDGAR book debt)."
        ),
    )
    p.add_argument(
        "--terminal",
        type=float,
        default=DEFAULT_TERMINAL_GROWTH,
        help=(
            "Terminal/perpetuity growth and explicit-period decay target "
            f"(default {DEFAULT_TERMINAL_GROWTH})."
        ),
    )
    p.add_argument(
        "--terminal-period-years",
        type=int,
        default=DEFAULT_TERMINAL_PERIOD_YEARS,
        metavar="N",
        help=(
            "Years after the explicit forecast growing at terminal rate before Gordon "
            f"(default {DEFAULT_TERMINAL_PERIOD_YEARS})."
        ),
    )
    p.add_argument(
        "--match-eps-growth-to-fcf",
        action="store_true",
        help="Under --auto-growth, force EPS growth to match the FCF growth estimate.",
    )
    p.add_argument(
        "--years",
        type=int,
        default=DEFAULT_PROJECTION_YEARS,
        help=f"Explicit forecast horizon length (default {DEFAULT_PROJECTION_YEARS}).",
    )
    p.add_argument(
        "--fcf-avg-years",
        type=int,
        default=DEFAULT_FCF_AVG_YEARS,
        help=(
            "When TTM (10-Q) is missing: trailing FY 10-K averaging window for "
            "base FCF and base EPS (default "
            f"{DEFAULT_FCF_AVG_YEARS}, i.e. latest FY only)."
        ),
    )
    p.add_argument(
        "--sensitivity",
        action="store_true",
        help="Growth x WACC grid for **FCF** intrinsic value only.",
    )
    p.add_argument(
        "--sens-steps",
        type=int,
        default=3,
        metavar="N",
        help="Grid resolution per sensitivity axis.",
    )
    p.add_argument(
        "--sens-growth-width",
        type=float,
        default=0.03,
        metavar="RATE",
    )
    p.add_argument(
        "--sens-wacc-width",
        type=float,
        default=0.02,
        metavar="RATE",
    )
    p.add_argument(
        "--monte-carlo",
        type=int,
        default=None,
        metavar="N",
        help="FCF-centric Monte Carlo on growth/WACC/terminal.",
    )
    p.add_argument(
        "--mc-seed",
        type=int,
        default=None,
        metavar="INT",
    )
    p.add_argument(
        "--mc-growth-width",
        type=float,
        default=0.03,
        metavar="RATE",
    )
    p.add_argument(
        "--mc-wacc-width",
        type=float,
        default=0.015,
        metavar="RATE",
    )
    p.add_argument(
        "--mc-terminal-width",
        type=float,
        default=0.0075,
        metavar="RATE",
    )
    return p


def _avg_eps_anchor(f_eps, avg_years: int) -> float:
    vals = list(f_eps.eps_values)
    if not vals:
        raise ValueError("empty eps")
    yrs = max(1, min(avg_years, len(vals)))
    chunk = vals[-yrs:]
    return sum(chunk) / len(chunk)


def _base_fcf_anchor(f, avg_years: int) -> tuple[float, str]:
    if f.ttm_fcf is not None:
        return float(f.ttm_fcf), "TTM"
    val = average_fcf(f.fcf_values, years=avg_years)
    if avg_years <= 1:
        return val, "latest FY (10-K)"
    return val, f"{avg_years}yr FY avg"


def _base_eps_anchor(f, avg_years: int) -> tuple[float, str]:
    if f.ttm_eps is not None:
        return float(f.ttm_eps), "TTM"
    val = _avg_eps_anchor(f, avg_years)
    if avg_years <= 1:
        return val, "latest FY (10-K)"
    return val, f"{avg_years}yr FY avg"


def _run_one_ticker(
    ticker: str,
    base_assumptions: DCFAssumptions,
    *,
    auto_growth: bool,
    auto_growth_mode: str,
    fcf_cagr_window: int | None,
    manual_growth: float,
    eps_growth_override: float | None,
    fcf_avg_years: int,
    model_mode: str,
    sensitivity: bool,
    sens_steps: int,
    sens_growth_width: float,
    sens_wacc_width: float,
    monte_carlo_n: int | None,
    mc_seed: int | None,
    mc_growth_width: float,
    mc_wacc_width: float,
    mc_terminal_width: float,
    auto_wacc: bool,
    terminal_period_years: int,
    match_eps_to_fcf: bool,
) -> int:
    try:
        f = get_fundamentals(ticker)
    except EdgarError as exc:
        print(f"EDGAR error ({ticker}): {exc}", file=sys.stderr)
        return 1

    need_fcf = model_mode in ("fcf", "both")
    need_eps = model_mode in ("eps", "both")

    growth_estimate_fcf: GrowthEstimate | None = None
    growth_estimate_eps: GrowthEstimate | None = None

    def _fcf_mode_from_cli(mode: str) -> AutoGrowthMode:
        if mode == "blended":
            return "blended"
        if mode == "blended_fcf":
            return "blended_fcf"
        if mode == "consensus":
            return "consensus_only"
        return "fcf_sec_yahoo_mixed"

    def _eps_mode_from_cli(mode: str, fcf_m: AutoGrowthMode) -> AutoGrowthMode:
        if mode == "eps_yahoo":
            return "eps_yahoo"
        if mode == "consensus":
            return "consensus_only"
        if mode == "blended":
            return "blended"
        return "eps_yahoo"

    base_eps_anchor: float | None = None
    if need_eps:
        try:
            base_eps_anchor, _ = _base_eps_anchor(f, fcf_avg_years)
        except ValueError:
            base_eps_anchor = None

    facts_json = None
    if auto_growth:
        try:
            facts_json = fetch_company_facts(f.cik)
        except EdgarError:
            facts_json = None

    fcf_mode = _fcf_mode_from_cli(auto_growth_mode)
    eps_mode = _eps_mode_from_cli(auto_growth_mode, fcf_mode)

    if auto_growth:
        if need_fcf:
            growth_estimate_fcf = compute_growth_estimate(
                ticker=ticker,
                cik=f.cik,
                fcf_values=f.fcf_values,
                auto_growth_mode=fcf_mode,
                fcf_cagr_window=fcf_cagr_window,
                company_facts=facts_json,
                baseline_eps=base_eps_anchor,
            )
            growth_fcf = growth_estimate_fcf.growth_rate
        else:
            growth_fcf = manual_growth

        if need_eps:
            if match_eps_to_fcf:
                growth_eps = growth_fcf
            elif eps_growth_override is not None:
                growth_eps = float(eps_growth_override)
            else:
                growth_estimate_eps = compute_growth_estimate(
                    ticker=ticker,
                    cik=f.cik,
                    fcf_values=f.fcf_values,
                    auto_growth_mode=eps_mode,
                    fcf_cagr_window=fcf_cagr_window,
                    company_facts=facts_json,
                    baseline_eps=base_eps_anchor,
                )
                growth_eps = growth_estimate_eps.growth_rate
        else:
            growth_eps = manual_growth
    else:
        growth_fcf = manual_growth
        growth_eps = (
            float(eps_growth_override)
            if eps_growth_override is not None
            else manual_growth
        )

    if need_fcf and not f.fcf_history:
        print(f"No FCF history found for {ticker}.", file=sys.stderr)
        return 1
    if need_eps and not f.eps_history:
        print(f"No diluted/basic EPS XBRL tagging for {ticker}.", file=sys.stderr)
        return 1
    if need_fcf or model_mode != "eps":
        if f.shares_outstanding <= 0:
            print(
                f"Could not determine diluted shares outstanding for {ticker}.",
                file=sys.stderr,
            )
            return 1

    assumptions = replace(base_assumptions, growth_rate=growth_fcf)

    market_price: float | None = None
    wacc_breakdown: WACCBreakdown | None = None
    try:
        market_price = get_current_price(ticker)
    except MarketDataError as exc:
        market_price = None
        if auto_wacc:
            print(
                f"Auto WACC ({ticker}) requires Yahoo price — {exc}",
                file=sys.stderr,
            )
            return 1

    if auto_wacc:
        if market_price is None:
            print(f"Auto WACC ({ticker}): no price available.", file=sys.stderr)
            return 1
        try:
            wacc_breakdown = estimate_wacc(ticker, f, market_price)
        except ValueError as exc:
            print(f"WACC estimation ({ticker}): {exc}", file=sys.stderr)
            return 1
        assumptions = replace(assumptions, wacc=wacc_breakdown.wacc)

    base_fcf, base_fcf_label = _base_fcf_anchor(f, fcf_avg_years) if need_fcf else (0.0, "")
    base_eps, base_eps_label = _base_eps_anchor(f, fcf_avg_years) if need_eps else (0.0, "")

    print(f"{f.company_name} ({f.ticker})")

    if need_fcf:
        print(f"  Base FCF ({base_fcf_label}):       {_fmt_money(base_fcf)}")
    if need_eps:
        print(f"  Base EPS ({base_eps_label}):       ${base_eps:,.2f}/shr")

    if need_fcf or model_mode != "eps":
        print(f"  Shares outstanding:       {_fmt_shares(f.shares_outstanding)}")
        debt_lbl = _fmt_money(f.net_debt) + (" (net cash)" if f.net_debt < 0 else "")
        print(f"  Net debt:                 {debt_lbl}")
    print()

    if growth_estimate_fcf:
        _print_auto_growth(growth_estimate_fcf)
        print()
    if growth_estimate_eps and growth_estimate_eps is not growth_estimate_fcf:
        _print_auto_growth(growth_estimate_eps)
        print()

    if wacc_breakdown is not None:
        _print_auto_wacc(wacc_breakdown)
        print()

    decay_note = " (explicit decay → terminal)" if auto_growth else ""
    print(
        f"  DCF inputs: "
        + (f"growth(FCF/EPS-shared)={growth_fcf:.2%}  "
           if growth_eps == growth_fcf
           else f"growth(FCF)={growth_fcf:.2%}  growth(EPS)={growth_eps:.2%}  ")
        + f"wacc={assumptions.wacc:.2%}  terminal={assumptions.terminal_growth:.2%} "
        f"explicit_years={assumptions.projection_years}  "
        f"terminal_period={assumptions.terminal_period_years}; model={model_mode.upper()}"
        f"{decay_note}"
    )

    iv_fcf = iv_eps = consensus = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if need_fcf:
            if auto_growth:
                r_fcf = intrinsic_value_per_share_explicit_decay(
                    base_fcf=base_fcf,
                    growth_rate=growth_fcf,
                    wacc=assumptions.wacc,
                    terminal_growth=assumptions.terminal_growth,
                    shares_outstanding=f.shares_outstanding,
                    net_debt=f.net_debt,
                    projection_years=assumptions.projection_years,
                    terminal_period_years=assumptions.terminal_period_years,
                )
            else:
                r_fcf = intrinsic_value_per_share(
                    base_fcf=base_fcf,
                    growth_rate=growth_fcf,
                    wacc=assumptions.wacc,
                    terminal_growth=assumptions.terminal_growth,
                    shares_outstanding=f.shares_outstanding,
                    net_debt=f.net_debt,
                    projection_years=assumptions.projection_years,
                    terminal_period_years=assumptions.terminal_period_years,
                )
            iv_fcf = float(r_fcf["intrinsic_value_per_share"])
            print(f"  FCF intrinsic/share:       ${iv_fcf:,.2f}")
            print(f"    EV:                      {_fmt_money(r_fcf['enterprise_value'])}")
            print(f"    Equity:                  {_fmt_money(r_fcf['equity_value'])}")
        if need_eps:
            if auto_growth:
                r_eps = intrinsic_price_from_eps_explicit_decay(
                    base_eps=base_eps,
                    growth_rate=growth_eps,
                    wacc=assumptions.wacc,
                    terminal_growth=assumptions.terminal_growth,
                    projection_years=assumptions.projection_years,
                    terminal_period_years=assumptions.terminal_period_years,
                )
            else:
                r_eps = intrinsic_price_from_eps(
                    base_eps=base_eps,
                    growth_rate=growth_eps,
                    wacc=assumptions.wacc,
                    terminal_growth=assumptions.terminal_growth,
                    projection_years=assumptions.projection_years,
                    terminal_period_years=assumptions.terminal_period_years,
                )
            iv_eps = float(r_eps["intrinsic_price_per_share"])
            print(f"  EPS intrinsic/share (direct): ${iv_eps:,.2f}")

    if iv_fcf is not None and iv_eps is not None:
        consensus = (iv_fcf + iv_eps) / 2.0
        print(f"  Consensus IV (simple mean): ${consensus:,.2f}")

    active_iv = consensus if consensus is not None else (iv_fcf or iv_eps)

    price: float | None = market_price
    if price is None:
        try:
            price = get_current_price(ticker)
        except MarketDataError as exc:
            print(f"Warning ({ticker}): market price unavailable — {exc}", file=sys.stderr)

    label = (
        "consensus"
        if consensus is not None
        else ("FCF_IV" if model_mode != "eps" else "EPS_IV")
    )
    if active_iv is not None and price is not None:
        mos = (active_iv - price) / price
        tag = "UNDERVALUED" if mos > 0 else "OVERVALUED"
        print(f"  Market price (yfinance fast_info): ${price:,.2f}")
        print(f"  MoS [{label}] vs px:               {mos:+.1%} ({tag})")

    if sensitivity and need_fcf:
        grid = intrinsic_sensitivity_grid(
            base_fcf=base_fcf,
            shares_outstanding=f.shares_outstanding,
            net_debt=f.net_debt,
            projection_years=assumptions.projection_years,
            center_growth=growth_fcf,
            center_wacc=assumptions.wacc,
            terminal_growth=assumptions.terminal_growth,
            growth_half_width=sens_growth_width,
            wacc_half_width=sens_wacc_width,
            steps=max(1, sens_steps),
            use_explicit_decay=auto_growth,
            terminal_period_years=assumptions.terminal_period_years,
        )
        print()
        print("  FCF sensitivity (USD/sh intrinsic, growth-rows x WACC-cols)")
        for line in format_sensitivity_grid_table(grid):
            print(f"    {line}")
    elif sensitivity and model_mode == "eps":
        print(
            "`--sensitivity` is disabled for EPS-only runs (needs FCF inputs).",
            file=sys.stderr,
        )

    if monte_carlo_n and need_fcf:
        rng = random.Random(mc_seed) if mc_seed is not None else random.Random()
        mc = monte_carlo_intrinsic(
            base_fcf=base_fcf,
            shares_outstanding=f.shares_outstanding,
            net_debt=f.net_debt,
            projection_years=assumptions.projection_years,
            center_growth=growth_fcf,
            center_wacc=assumptions.wacc,
            center_terminal=assumptions.terminal_growth,
            n_samples=int(monte_carlo_n),
            growth_half_width=mc_growth_width,
            wacc_half_width=mc_wacc_width,
            terminal_half_width=mc_terminal_width,
            rng=rng,
            use_explicit_decay=auto_growth,
            terminal_period_years=assumptions.terminal_period_years,
        )
        print()
        sfx = f", seed={mc_seed}" if mc_seed is not None else ""
        print(f"  FCF MonteCarlo ({mc.n_valid}/{mc.n_samples} usable draws{sfx})")
        if mc.n_valid > 0:
            print(
                f"    mean=${mc.mean:,.2f} sigma=${mc.std:,.2f} "
                f"p5=${mc.p5:,.2f} p50=${mc.p50:,.2f} p95=${mc.p95:,.2f}"
            )
    elif monte_carlo_n and model_mode == "eps":
        print(
            "`--monte-carlo` applies to the FCF valuation path only "
            "(re-run with `--model fcf` or `both`).",
            file=sys.stderr,
        )

    return 0


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    assumptions = DCFAssumptions(
        growth_rate=0.0,
        wacc=DEFAULT_WACC if args.auto_wacc else args.wacc,
        terminal_growth=args.terminal,
        projection_years=args.years,
        terminal_period_years=args.terminal_period_years,
    )
    try:
        assumptions.validate()
    except ValueError as exc:
        print(f"Invalid CLI assumptions: {exc}", file=sys.stderr)
        return 2

    if args.sens_steps < 1:
        print("`--sens-steps` >= 1", file=sys.stderr)
        return 2

    if args.terminal_period_years < 0:
        print("`--terminal-period-years` must be >= 0", file=sys.stderr)
        return 2

        print("`--fcf-cagr-years` must be >= 0", file=sys.stderr)
        return 2

    fcf_window: int | None = (
        None if args.fcf_cagr_years == 0 else int(args.fcf_cagr_years)
    )

    ec = 0
    tickers = [t.upper() for t in args.ticker]

    for i, tk in enumerate(tickers):
        if len(tickers) > 1 and i > 0:
            print()

        ec |= _run_one_ticker(
            tk,
            assumptions,
            auto_growth=args.auto_growth,
            auto_growth_mode=args.auto_growth_mode,
            fcf_cagr_window=fcf_window,
            manual_growth=args.growth,
            eps_growth_override=args.eps_growth,
            fcf_avg_years=args.fcf_avg_years,
            model_mode=args.model,
            sensitivity=args.sensitivity,
            sens_steps=args.sens_steps,
            sens_growth_width=args.sens_growth_width,
            sens_wacc_width=args.sens_wacc_width,
            monte_carlo_n=args.monte_carlo,
            mc_seed=args.mc_seed,
            mc_growth_width=args.mc_growth_width,
            mc_wacc_width=args.mc_wacc_width,
            mc_terminal_width=args.mc_terminal_width,
            auto_wacc=args.auto_wacc,
            terminal_period_years=args.terminal_period_years,
            match_eps_to_fcf=args.match_eps_growth_to_fcf,
        )

    return ec


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
