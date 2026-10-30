from __future__ import annotations

from typing import Any, Mapping

from .db import RuntimeDatabase
from .models import (
    EvidencePolarity,
    EvidenceRecord,
    EvidenceType,
    new_id,
)


class EvidenceLedger:
    """Append-only, provenance-bearing evidence ledger.

    Evidence is deduplicated by a stable content/provenance hash. Claims link to
    evidence through a separate relation, so one source object can support several
    claims without copying or weakening provenance.
    """

    def __init__(self, db: RuntimeDatabase):
        self.db = db

    def add(
        self,
        *,
        run_id: str,
        evidence_type: EvidenceType | str,
        content: str,
        claim_id: str | None = None,
        polarity: EvidencePolarity | str = EvidencePolarity.SUPPORTS,
        source_uri: str | None = None,
        page: int | None = None,
        bbox: list[float] | None = None,
        object_type: str | None = None,
        artifact_path: str | None = None,
        confidence: float = 0.8,
        provenance: Mapping[str, Any] | None = None,
    ) -> tuple[str, bool]:
        record = EvidenceRecord(
            evidence_id=new_id("evidence"),
            run_id=run_id,
            claim_id=claim_id,
            evidence_type=evidence_type if isinstance(evidence_type, EvidenceType) else EvidenceType(str(evidence_type)),
            content=content,
            polarity=polarity if isinstance(polarity, EvidencePolarity) else EvidencePolarity(str(polarity)),
            source_uri=source_uri,
            page=page,
            bbox=bbox,
            object_type=object_type,
            artifact_path=artifact_path,
            confidence=max(0.0, min(1.0, float(confidence))),
            provenance=dict(provenance or {}),
        )
        return self.db.add_evidence(record)

    def add_from_tool(
        self,
        *,
        run_id: str,
        claim_id: str | None,
        values: list[Mapping[str, Any]],
        tool_name: str,
        tool_call_id: str,
    ) -> list[str]:
        evidence_ids: list[str] = []
        for raw in values:
            provenance = dict(raw.get("provenance") or {})
            provenance.setdefault("tool_name", tool_name)
            provenance.setdefault("tool_call_id", tool_call_id)
            evidence_id, _ = self.add(
                run_id=run_id,
                claim_id=str(raw.get("claim_id") or claim_id) if (raw.get("claim_id") or claim_id) else None,
                evidence_type=raw.get("evidence_type", EvidenceType.TEXT.value),
                content=str(raw.get("content", "")),
                polarity=raw.get("polarity", EvidencePolarity.SUPPORTS.value),
                source_uri=raw.get("source_uri"),
                page=int(raw["page"]) if raw.get("page") is not None else None,
                bbox=list(raw["bbox"]) if raw.get("bbox") is not None else None,
                object_type=raw.get("object_type"),
                artifact_path=raw.get("artifact_path"),
                confidence=float(raw.get("confidence", 0.8)),
                provenance=provenance,
            )
            evidence_ids.append(evidence_id)
        return evidence_ids
