"""Regex guidance extraction helpers on canned 8-K-style prose."""

from __future__ import annotations

from valuation.growth.eight_k import _extract_regex_candidates


def test_regex_finds_explicit_percent_near_growth():
    text = (
        "We expect revenue growth of 11% compounded annually "
        "over the forecast horizon excluding FX."
    )
    cand = _extract_regex_candidates(text, citation="fixture", source="fixture")
    assert cand
    assert abs(cand[0].growth_rate - 0.11) < 1e-9


def test_regex_guidance_sentence():
    text = (
        "Management reiterated guidance projecting EPS acceleration of roughly 9% "
        "for fiscal 2026 relative to FY2025."
    )
    cand = _extract_regex_candidates(text, citation="fixture", source="fixture")
    rates = {round(c.growth_rate, 3) for c in cand}
    assert 0.09 in rates
