"""JSON array coercion helper tests."""

from __future__ import annotations

from valuation.growth import json_parse


def test_parse_embedded_json_array():
    messy = """Here we go ```json [{"metric":"eps","growth_pct":8.25,
    "period":"FY2028","confidence":"high","quote":"guided"}] ```
    Done."""
    objs = json_parse.parse_json_array(messy)
    assert len(objs) == 1
    assert objs[0]["growth_pct"] == 8.25
