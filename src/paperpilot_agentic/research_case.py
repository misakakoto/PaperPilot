"""Host registration of a historical evidence chain, never execution authority."""

import json

from .reference_methods import encoded, sha
from .research_assets import immutable, owned_folder
from .research_diagnostics import load_bundle


def register_case(runtime, value):
    required = {
        "version",
        "historical_run_id",
        "historical_candidate_id",
        "experiment_bundle_id",
        "sources",
        "proposal",
        "approved_candidate",
        "approval",
        "historical_decisions",
        "historical_source_identity",
        "imported_verification",
        "assurance",
    }
    if set(value) != required or value["version"] != "research-case-v1":
        raise ValueError("Expected a host research-case-v1 manifest")
    bundle_id = value["experiment_bundle_id"]
    bundle = load_bundle(runtime, {"identity": bundle_id, "bundle_sha256": bundle_id})
    if value["imported_verification"] != bundle["verification"]:
        raise ValueError("Case cannot upgrade imported training provenance")
    card, approval = value["approved_candidate"], value["approval"]
    if (
        approval.get("status") != "APPROVED"
        or approval.get("request", {}).get("context") != card
        or approval["request"].get("arguments") != {"binding_id": value["historical_candidate_id"]}
        or card.get("binding_id") != value["historical_candidate_id"]
        or card.get("hypothesis") != value["proposal"]
    ):
        raise ValueError("Historical approval, candidate and proposal must match")
    ids = []
    for item in value["sources"]:
        if sha(encoded(item["snapshot"])) != item["sha256"]:
            raise ValueError("Historical source snapshot changed")
        ids.append(item["snapshot"]["evidence_id"])
    if (
        not ids
        or len(ids) != len(set(ids))
        or not set(value["proposal"]["evidence_ids"]) <= set(ids)
    ):
        raise ValueError("Proposal needs distinct registered source snapshots")
    raw = encoded(value)
    identity = sha(raw)
    immutable(owned_folder(runtime, "research-cases", identity), "case.json", raw)
    return {
        "case_id": identity,
        "experiment_bundle_id": bundle_id,
        "source_ids": ids,
        "execution_authority": "NONE; historical evidence linkage only",
    }


def load_case(runtime, case_id, bundle_id):
    raw = (owned_folder(runtime, "research-cases", case_id) / "case.json").read_bytes()
    if sha(raw) != case_id:
        raise ValueError("Research case identity changed")
    value = json.loads(raw)
    if bundle_id is not None and value["experiment_bundle_id"] != bundle_id:
        raise ValueError("Research case belongs to a different experiment bundle")
    return value


def case_navigation(value, case_id):
    return {
        "case_id": case_id,
        "historical_run_id": value["historical_run_id"],
        "historical_candidate_id": value["historical_candidate_id"],
        "source_ids": [s["snapshot"]["evidence_id"] for s in value["sources"]],
        "historical_hypothesis": value["proposal"],
        "approval_status": value["approval"]["status"],
        "assurance": "HISTORICAL_LINK_ONLY; hypothesis not established, no new training",
    }
