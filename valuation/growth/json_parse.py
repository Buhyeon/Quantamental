"""Tiny JSON array extraction helpers (tests / legacy parsers)."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_array(raw: str) -> list[dict[str, Any]]:
    """Parse a JSON array from model output or embedded ```json fences."""
    raw = raw.strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[[\s\S]*\]", raw)
        if not m:
            return []
        parsed = json.loads(m.group(0))
    if not isinstance(parsed, list):
        return []
    return [x for x in parsed if isinstance(x, dict)]
