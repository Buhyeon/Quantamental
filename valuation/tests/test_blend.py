"""Blender deterministic behaviour."""

from __future__ import annotations

import math

from valuation.config import DEFAULT_GROWTH_RATE
from valuation.growth.blend import blend_candidates
from valuation.growth.types import GuidanceCandidate


def test_blend_fallback_when_empty_candidates():
    out = blend_candidates(tuple())
    assert out.growth_rate == DEFAULT_GROWTH_RATE


def test_weighted_median_with_consensus_matches_rate():
    cands = (
        GuidanceCandidate(0.10, source="fcf_cagr", confidence="med", metric="fcf"),
        GuidanceCandidate(
            0.10,
            source="financial_consensus",
            confidence="med",
            metric="revenue",
            snippet="yahoo",
        ),
    )
    out = blend_candidates(cands)
    assert math.isclose(out.growth_rate, 0.10, abs_tol=1e-9)
