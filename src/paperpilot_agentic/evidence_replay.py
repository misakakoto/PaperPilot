"""Zero-network replay of persisted, crop-bound paid evidence.

Replay is deliberately separate from a live Agent run. It never calls a model,
does not mutate the original runtime database and cannot be reported as a new
online answer. Its purpose is to verify that current evidence gates, source
scope, context selection and deterministic synthesis can consume already-paid
artifacts after a bug fix.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .claim_graph import derive_claim_evidence_state
from .intelligence import DeterministicAnswerSynthesizer, _select_synthesis_evidence
from .qwen_vision import reaudit_qwen_result

_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)%?(?![\w.])")


def _read_json(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {source.name}")
    return value, hashlib.sha256(raw).hexdigest()


def _claim_source_sha(claim: Mapping[str, Any]) -> str:
    metadata = dict(claim.get("metadata") or {})
    values = list(metadata.get("paper_sha256s") or [])
    if len(values) != 1 or not re.fullmatch(r"[a-f0-9]{64}", str(values[0])):
        raise ValueError("Replay requires exactly one SHA-bound source for each imported claim")
    return str(values[0])


def _persisted_result_from_snapshot_evidence(item: Mapping[str, Any]) -> dict[str, Any]:
    provenance = dict(item.get("provenance") or {})
    metadata = dict(provenance.get("vision_metadata") or {})
    if not metadata:
        raise ValueError("Snapshot evidence has no persisted vision metadata")
    return {
        "description": str(item.get("content") or ""),
        "confidence": float(item.get("confidence") or 0.0),
        "verified": bool(provenance.get("vision_verified")),
        "metadata": metadata,
    }


def _import_table(
    spec: Mapping[str, Any],
    *,
    claim_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_claim_id = str(spec.get("target_claim_id") or "")
    if target_claim_id not in claim_by_id:
        raise ValueError(f"Unknown target claim_id: {target_claim_id}")
    expected_sha = _claim_source_sha(claim_by_id[target_claim_id])
    kind = str(spec.get("kind") or "")
    source_record_sha = ""
    if kind == "snapshot_qwen_evidence":
        snapshot, source_record_sha = _read_json(str(spec.get("snapshot") or ""))
        evidence_id = str(spec.get("evidence_id") or "")
        matches = [
            item
            for item in list(snapshot.get("evidence") or [])
            if str(item.get("evidence_id") or "") == evidence_id
        ]
        if len(matches) != 1:
            raise ValueError("Snapshot evidence_id must resolve exactly once")
        source_item = dict(matches[0])
        provenance = dict(source_item.get("provenance") or {})
        source_sha = str(
            provenance.get("document_sha256")
            or provenance.get("paper_sha256")
            or ""
        )
        image_path = str(spec.get("image") or source_item.get("artifact_path") or "")
        original_result = _persisted_result_from_snapshot_evidence(source_item)
        page = int(spec.get("page") or source_item.get("page") or 0) or None
        source_uri = str(
            spec.get("source_uri")
            or source_item.get("source_uri")
            or f"paper://{source_sha}/layout/replayed-table"
        )
    elif kind == "qwen_record":
        record, source_record_sha = _read_json(str(spec.get("record") or ""))
        original_result = dict(record.get("result") or {})
        source_sha = str(spec.get("paper_sha256") or "")
        image_path = str(spec.get("image") or "")
        page = int(spec.get("page") or 0) or None
        source_uri = str(
            spec.get("source_uri")
            or f"paper://{source_sha}/layout/replayed-table"
        )
    else:
        raise ValueError(f"Unsupported replay import kind: {kind}")
    if source_sha != expected_sha:
        raise ValueError("Imported table SHA does not match the target Claim source scope")

    audited = reaudit_qwen_result(
        original_result,
        image_path,
        prompt=str(spec.get("prompt") or "Read the exact visible table."),
        context={"caption": str(spec.get("caption") or "")},
    )
    if not bool(audited.get("verified")):
        reason = (audited.get("metadata") or {}).get("table_structure_reason")
        raise ValueError(f"Imported Qwen table failed current offline audit: {reason}")
    content = str(audited.get("description") or "")
    digest = hashlib.sha256(
        f"{target_claim_id}\x1f{source_sha}\x1f{page}\x1f{content}".encode("utf-8")
    ).hexdigest()
    evidence_id = f"evidence_replay_{digest[:24]}"
    metadata = dict(audited.get("metadata") or {})
    evidence = {
        "evidence_id": evidence_id,
        "claim_id": target_claim_id,
        "claim_ids": [target_claim_id],
        "polarity": "SUPPORTS",
        "claim_polarities": {target_claim_id: "SUPPORTS"},
        "evidence_type": "TABLE",
        "content": content,
        "source_uri": source_uri,
        "artifact_path": None,
        "page": page,
        "confidence": float(audited.get("confidence") or 0.0),
        "provenance": {
            "paper_sha256": source_sha,
            "document_sha256": source_sha,
            "gap": "numeric_results",
            "vision_verified": True,
            "vision_metadata": metadata,
            "offline_replay": True,
            "verification": {
                "verifier": "runtime-verified-visual-table-v1",
                "degraded": False,
                "claim_coverage": 1.0,
            },
        },
    }
    audit = {
        "kind": kind,
        "target_claim_id": target_claim_id,
        "paper_sha256": source_sha,
        "page": page,
        "source_record_sha256": source_record_sha,
        "image_sha256": metadata.get("image_sha256"),
        "verified": True,
        "table_structure_reason": metadata.get("table_structure_reason"),
        "relevance_mode": metadata.get("relevance_mode"),
        "numeric_token_count": len(_NUMBER_RE.findall(content)),
        "evidence_id": evidence_id,
        "supersedes_evidence_id": (
            str(spec.get("evidence_id") or "")
            if kind == "snapshot_qwen_evidence"
            else None
        ),
    }
    return evidence, audit


def replay_paid_evidence(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a private replay result without mutating any source artifact."""

    base, base_sha = _read_json(str(manifest.get("base_snapshot") or ""))
    claims = copy.deepcopy(list((base.get("claim_graph") or {}).get("claims") or []))
    claim_by_id = {str(item.get("claim_id") or ""): item for item in claims}
    if not claims or "run" not in base:
        raise ValueError("Base snapshot is not a PaperPilot Agent snapshot")
    evidence = [
        copy.deepcopy(item)
        for item in list(base.get("evidence") or [])
        if str(item.get("polarity") or "NEUTRAL") != "NEUTRAL"
    ]
    imported: list[dict[str, Any]] = []
    import_audits: list[dict[str, Any]] = []
    superseded_evidence_ids: set[str] = set()
    for raw_spec in list(manifest.get("imports") or []):
        if not isinstance(raw_spec, Mapping):
            raise TypeError("Each replay import must be an object")
        item, audit = _import_table(raw_spec, claim_by_id=claim_by_id)
        imported.append(item)
        import_audits.append(audit)
        if audit.get("supersedes_evidence_id"):
            superseded_evidence_ids.add(str(audit["supersedes_evidence_id"]))
        claim = claim_by_id[str(item["claim_id"])]
        claim["gaps"] = [gap for gap in list(claim.get("gaps") or []) if gap != "numeric_results"]
        metadata = dict(claim.get("metadata") or {})
        metadata["offline_replay_table_evidence_id"] = item["evidence_id"]
        metadata["offline_replay_only"] = True
        claim["metadata"] = metadata
    # A current-gate re-audit replaces the exact persisted visual evidence only
    # inside this immutable replay. Keeping both the known-bad transcription and
    # its corrected transcription would contaminate final synthesis.
    if superseded_evidence_ids:
        evidence = [
            item
            for item in evidence
            if str(item.get("evidence_id") or "") not in superseded_evidence_ids
        ]
    evidence.extend(imported)

    # Re-adjudicate every Claim under the current evidence policy. This is what
    # makes replay useful after a verifier/aggregation bug fix: old terminal
    # labels are not blindly trusted, while the original database stays intact.
    for claim in claims:
        claim_id = str(claim.get("claim_id") or "")
        scoped_rows: list[dict[str, Any]] = []
        for item in evidence:
            polarities = dict(item.get("claim_polarities") or {})
            linked = list(item.get("claim_ids") or [])
            if item.get("claim_id") and item.get("claim_id") not in linked:
                linked.append(item.get("claim_id"))
            if claim_id not in {str(value) for value in linked}:
                continue
            row = copy.deepcopy(item)
            row["link_polarity"] = str(
                polarities.get(claim_id) or row.get("polarity") or "NEUTRAL"
            )
            scoped_rows.append(row)
        state = derive_claim_evidence_state(claim, scoped_rows)
        claim["status"] = state.pop("status").value
        claim["confidence"] = state.pop("confidence")
        claim["gaps"] = state.pop("gaps")
        metadata = dict(claim.get("metadata") or {})
        metadata.update(state)
        metadata["offline_replay_recomputed"] = True
        claim["metadata"] = metadata
        if claim["status"] in {"SUPPORTED", "CONTRADICTED"}:
            metadata.pop("partial_reason", None)
            metadata.pop("unresolved_reason", None)

    selected, context_stats = _select_synthesis_evidence(claims, evidence)
    answer = dict(
        DeterministicAnswerSynthesizer().synthesize(
            question=str(base["run"].get("question") or ""),
            claims=claims,
            evidence=selected,
        )
    )
    answer["synthesis_context"] = context_stats
    resolved = sum(
        str(item.get("status")) in {"SUPPORTED", "CONTRADICTED"}
        for item in claims
    )
    partial = sum(str(item.get("status")) == "PARTIAL" for item in claims)
    outcome = "ANSWERED" if resolved == len(claims) else ("PARTIAL" if resolved or partial else "UNRESOLVED")
    scope_violations = 0
    for item in selected:
        source_sha = str((item.get("provenance") or {}).get("paper_sha256") or "")
        for claim_id in list(item.get("claim_ids") or ([item.get("claim_id")] if item.get("claim_id") else [])):
            claim = claim_by_id.get(str(claim_id))
            if claim is not None and source_sha and source_sha != _claim_source_sha(claim):
                scope_violations += 1
    return {
        "schema_version": 1,
        "kind": "zero_network_paid_evidence_replay",
        "source_snapshot_sha256": base_sha,
        "source_run_id": base["run"].get("run_id"),
        "source_runtime_status": base["run"].get("status"),
        "source_research_outcome": base["run"].get("research_outcome"),
        "question": base["run"].get("question"),
        "claims": claims,
        "evidence": evidence,
        "import_audits": import_audits,
        "answer": answer,
        "replay_research_outcome": outcome,
        "source_scope_violations": scope_violations,
        "network_calls": 0,
        "paid_requests": 0,
        "evidence_boundary": (
            "Current code replay of already-paid, crop-bound evidence; not a new live LLM run."
        ),
    }


def redacted_replay_report(result: Mapping[str, Any]) -> dict[str, Any]:
    """Strip private excerpts and paths while retaining reproducible assertions."""

    claims = list(result.get("claims") or [])
    answer = dict(result.get("answer") or {})
    return {
        "schema_version": 1,
        "evaluation_kind": result.get("kind"),
        "source_snapshot_sha256": result.get("source_snapshot_sha256"),
        "source_runtime_status": result.get("source_runtime_status"),
        "source_research_outcome": result.get("source_research_outcome"),
        "replay_research_outcome": result.get("replay_research_outcome"),
        "claims": [
            {
                "claim_id": item.get("claim_id"),
                "status": item.get("status"),
                "remaining_gap_count": len(item.get("gaps") or []),
            }
            for item in claims
        ],
        "imports": list(result.get("import_audits") or []),
        "answer_mode": answer.get("mode"),
        "citation_count": len(answer.get("citation_ids") or []),
        "synthesis_context": answer.get("synthesis_context") or {},
        "source_scope_violations": result.get("source_scope_violations"),
        "network_calls": 0,
        "paid_requests": 0,
        "private_evidence_persisted": False,
        "evidence_boundary": result.get("evidence_boundary"),
    }
