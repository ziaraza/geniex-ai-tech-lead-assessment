"""Response parsing for structured triage output.

CONFIDENCE PARSING
------------------
Two new fields are extracted: CONFIDENCE and CONFIDENCE_REASON.

CONFIDENCE defaults to 'low' (not a neutral default) if absent or unparseable,
because a missing confidence field means the model produced unexpected output —
the conservative choice is to flag the ticket for human review.
"""

import re

DEFAULT_VALUES = {
    "category": "other",
    "priority": "medium",
    "assigned_team": "L1",
    "summary": "Unable to parse model response",
    "confidence": "low",
    "confidence_reason": "Confidence field absent — parse failure.",
}

REQUIRED_FIELDS = ["category", "priority", "assigned_team", "summary"]

VALID_CONFIDENCE_VALUES = {"high", "medium", "low"}


def parse_triage_response(raw_text):
    """Parse the model's structured response into a dict.

    Returns a dict with all classification fields plus:
      - confidence: 'high' | 'medium' | 'low'
      - confidence_reason: str
      - _parse_used_defaults: list[str]  — internal signal for agent.py
    """
    result = {}
    used_defaults = []

    for field in ["CATEGORY", "PRIORITY", "ASSIGNED_TEAM", "SUMMARY",
                  "CONFIDENCE", "CONFIDENCE_REASON"]:
        match = re.search(rf"^{field}:\s*(.+)", raw_text, re.MULTILINE)
        key = field.lower()
        if match:
            result[key] = match.group(1).strip()
        else:
            result[key] = DEFAULT_VALUES[key]
            if field in [f.upper() for f in REQUIRED_FIELDS]:
                used_defaults.append(key)

    # Normalise and validate confidence value.
    raw_conf = result["confidence"].lower().split()[0]  # e.g. "high." → "high"
    if raw_conf not in VALID_CONFIDENCE_VALUES:
        result["confidence"] = "low"
        result["confidence_reason"] = (
            f"Unparseable confidence value '{result['confidence']}' — defaulting to low."
        )
        used_defaults.append("confidence")

    result["_parse_used_defaults"] = used_defaults
    return result
