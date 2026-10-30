"""Actionable reference errors, without repairing scientific claims for a model."""

from difflib import get_close_matches
import json

from .research_contracts import FINISH_DECISION_VERSION, FINISH_DECISIONS

REFERENCE_PROJECTION_VERSION = "registered-observation-projection-v1"


def resolve_finish_decision(value):
    """Translate a registered analyst stop code without guessing new decisions."""
    if not isinstance(value, str) or value not in FINISH_DECISIONS:
        raise ValueError(
            "finish.decision must be a registered decision: " + ", ".join(FINISH_DECISIONS)
        )
    canonical = FINISH_DECISIONS[value]
    return canonical, {
        "version": FINISH_DECISION_VERSION,
        "submitted_decision": value,
        "canonical_decision": canonical,
    }


def project_observed_refs(values, numeric_catalog, pattern_catalog, *, label, nonempty=True):
    """Resolve registered aliases only; never guess/repair an unknown reference."""
    if not isinstance(values, list) or len(values) > 256 or (nonempty and not values):
        raise ValueError(label + " requires a bounded list of registered observation IDs")
    resolved, selections = [], []
    for value in values:
        if not isinstance(value, str):
            require_observed_refs([value], numeric_catalog, label=label)
        targets = pattern_catalog[value]["source_refs"] if value in pattern_catalog else [value]
        require_observed_refs(targets, numeric_catalog, label=label)
        resolved.extend(targets)
        selections.append({"selected_id": value, "numeric_refs": list(targets)})
    return list(dict.fromkeys(resolved)), selections


def require_observed_refs(values, catalog, *, label="fact_refs", nonempty=True):
    if not isinstance(values, list) or (nonempty and not values):
        raise ValueError(label + " requires a nonempty list of observed fact IDs")
    invalid = [value for value in values if not isinstance(value, str) or value not in catalog]
    if not invalid:
        return
    details = []
    for value in invalid[:6]:
        name = value if isinstance(value, str) else repr(value)
        parts = name.split("/")
        prefix = "/".join(parts[:2]) + "/"
        matches = sorted(
            key for key in catalog if key.startswith(prefix) and key.rsplit("/", 1)[-1] == parts[-1]
        )[:3]
        if not matches:
            matches = get_close_matches(name, sorted(catalog), n=3, cutoff=0.4)
        details.append({"unknown_id": name[:160], "existing_observed_ids": matches})
    raise ValueError(
        label
        + " contains unknown observed fact IDs: "
        + json.dumps(details)
        + ". Choose exact existing paths; do not invent a delta or metric. Claims were not repaired."
    )
