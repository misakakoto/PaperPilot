from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .db import RuntimeDatabase
from .models import EvidencePolarity


def source_attribution_audit(
    db: RuntimeDatabase,
    run_id: str,
    *,
    claim_scope_resolver: Callable[[str, set[str]], set[str]] | None = None,
) -> dict[str, Any]:
    """Audit non-neutral Claim-Evidence links against deterministic paper scopes."""

    checked_links = 0
    narrowed_claims = 0
    violations: list[dict[str, Any]] = []
    for claim in db.list_claims(run_id):
        allowed = {
            str(value)
            for value in claim.metadata.get("paper_sha256s", [])
            if str(value)
        }
        if not allowed:
            continue
        if claim_scope_resolver is not None:
            resolved = set(claim_scope_resolver(claim.statement, allowed))
            if resolved and resolved.issubset(allowed):
                if resolved != allowed:
                    narrowed_claims += 1
                allowed = resolved
        for evidence in db.list_evidence(run_id, claim.claim_id):
            claim_polarities = dict(evidence.get("claim_polarities") or {})
            polarity = str(
                claim_polarities.get(claim.claim_id)
                or evidence.get("link_polarity")
                or evidence.get("polarity")
                or ""
            )
            if polarity not in {
                EvidencePolarity.SUPPORTS.value,
                EvidencePolarity.CONTRADICTS.value,
            }:
                continue
            checked_links += 1
            provenance = evidence.get("provenance") or {}
            observed = str(
                provenance.get("paper_sha256")
                or provenance.get("document_sha256")
                or ""
            )
            if observed in allowed:
                continue
            violations.append(
                {
                    "claim_id": claim.claim_id,
                    "evidence_id": evidence.get("evidence_id"),
                    "polarity": polarity,
                    "allowed_paper_sha256s": sorted(allowed),
                    "observed_paper_sha256": observed or None,
                    "reason": "non-neutral evidence link is outside the claim paper scope",
                }
            )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "checked_non_neutral_links": checked_links,
        "claims_narrowed_from_statement": narrowed_claims,
        "violation_count": len(violations),
        "violations": violations,
        "status": "PASS" if not violations else "FAIL",
    }
