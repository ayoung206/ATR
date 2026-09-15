"""JSON object extraction for structured LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(text: str, *, greedy: bool = False) -> dict[str, Any]:
    """Parse a JSON object, optionally extracting it from surrounding prose.

    ``greedy`` spans the first opening brace through the last closing brace,
    as required by the decomposer's nested response format. Other callers use
    the shortest brace-delimited block. Invalid or non-object responses return
    an empty dictionary.
    """
    if not isinstance(text, str):
        return {}
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    pattern = r"\{.*\}" if greedy else r"\{.*?\}"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}
