"""Exact physical-object identity, independent of retrieval ranking."""
import re
from typing import Any, Mapping

def _prioritize_visual_candidates(
    selected: list[Mapping[str, Any]],
    evidence: list[dict[str, Any]],
) -> list[tuple[Mapping[str, Any], dict[str, Any]]]:
    """Spend bounded vision calls on parser-incomplete regions first."""

    candidates = list(zip(selected, evidence))
    candidates.sort(
        key=lambda pair: (
            not bool(
                pair[1].get("provenance", {}).get("needs_visual_verification")
            ),
            -float(
                pair[1].get("provenance", {}).get("retrieval_score") or 0.0
            ),
        )
    )
    return candidates


def _explicit_layout_label(text: str, object_type: str | None) -> tuple[str, str] | None:
    """Return a normalized explicitly named table/figure label, if present."""

    value = str(text or "")
    if object_type in {None, "table"}:
        match = re.search(
            r"(?:\btable\b|表)\s*((?:[IVXLCDM]+|\d+|[A-Z]\d+)(?:[.-]\d+)?)\b",
            value,
            re.IGNORECASE,
        )
        if match:
            return "table", match.group(1).strip(". ").upper()
    if object_type in {None, "figure"}:
        match = re.search(
            r"(?:\bfig(?:ure)?\b|图)\s*((?:[IVXLCDM]+|\d+|[A-Z]\d+)(?:[.-]\d+)?)\b",
            value,
            re.IGNORECASE,
        )
        if match:
            return "figure", match.group(1).strip(". ").upper()
    return None


def _matches_explicit_layout_label(
    item: Mapping[str, Any],
    target: tuple[str, str],
) -> bool:
    # A cross-reference in body/nearby text is not this object's identity.
    # Prefer the parser-associated caption; only an anchored own-text heading
    # may substitute for a missing caption. Never turn TABREF35 into Table 35.
    text = str(item.get("caption") or item.get("text") or "").strip()
    if not re.match(r"^(?:table\b|表|fig(?:ure)?\b|图)", text, re.I):
        return False
    kind = item.get("object_type")
    return (kind is None or kind == target[0]) and _explicit_layout_label(text, target[0]) == target


