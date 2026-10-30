from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from .context_engine import build_claim_coverage
from .db import RuntimeDatabase
from .models import (
    ActionCandidate,
    Budget,
    ClaimRecord,
    ClaimStatus,
    EvidencePolarity,
    EvidenceType,
    RiskLevel,
    new_id,
)
from .verification import trusted_semantic_adjudication_confidence

_GAP_TO_EVIDENCE: dict[str, EvidenceType] = {
    "primary_source": EvidenceType.TEXT,
    "text": EvidenceType.TEXT,
    "definition": EvidenceType.TEXT,
    "counter_evidence": EvidenceType.TEXT,
    "numeric_results": EvidenceType.TABLE,
    "table": EvidenceType.TABLE,
    "qualitative_result": EvidenceType.FIGURE,
    "figure": EvidenceType.FIGURE,
    "formula": EvidenceType.EQUATION,
    "equation": EvidenceType.EQUATION,
    "implementation": EvidenceType.CODE,
    "code": EvidenceType.CODE,
    "independent_experiment": EvidenceType.EXPERIMENT,
    "reproduction": EvidenceType.EXPERIMENT,
    "experiment_evaluation": EvidenceType.EXPERIMENT,
    "paper_experiment": EvidenceType.TEXT,
    "executed_experiment_result": EvidenceType.EXPERIMENT,
}

_EVIDENCE_WEIGHT: dict[EvidenceType, float] = {
    EvidenceType.TEXT: 0.45,
    EvidenceType.TABLE: 0.85,
    EvidenceType.FIGURE: 0.45,
    EvidenceType.EQUATION: 0.55,
    EvidenceType.CODE: 0.55,
    EvidenceType.EXPERIMENT: 0.9,
    EvidenceType.METADATA: 0.2,
}

_NUMERIC_VALUE_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?")
_DISCOVERY_PHRASE_RE = re.compile(
    r"\b(?:they\s+are|include(?:s|d)?|including|consist(?:s|ed)?\s+of|comprise(?:s|d)?|"
    r"we\s+propose|we\s+introduce|proposes?|introduces?)\s+([^.;]{12,320})",
    re.IGNORECASE,
)
_DISCOVERY_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9-]{1,9}\b")
_DISCOVERY_ACRONYM_STOP = {
    "AL",
    "BDA",
    "BCD",
    "CD",
    "CHEN",
    "CNN",
    "IEEE",
    "MAMBA",
    "REMOTE",
    "RNN",
    "SCD",
    "SENSING",
    "TABLE",
}


def _evidence_satisfies_gap_contract(
    gap: str,
    evidence_type: EvidenceType,
    row: Mapping[str, Any],
) -> bool:
    expected_type = _GAP_TO_EVIDENCE.get(gap)
    if expected_type is not None and evidence_type != expected_type:
        return False
    if gap == "numeric_results":
        provenance = dict(row.get("provenance") or {})
        if provenance.get("needs_visual_verification") and not provenance.get(
            "vision_verified"
        ):
            return False
        # A caption such as "TABLE XII" is a table object, but it does not
        # contain the requested ablation values and cannot close this gap.
        if not _NUMERIC_VALUE_RE.search(str(row.get("content") or "")):
            return False
    return True


def _claim_value(claim: ClaimRecord | Mapping[str, Any], name: str, default: Any) -> Any:
    if isinstance(claim, Mapping):
        return claim.get(name, default)
    return getattr(claim, name, default)


def _is_closure_grade_nonsemantic_evidence(
    claim: ClaimRecord | Mapping[str, Any],
    row: Mapping[str, Any],
    evidence_type: EvidenceType,
) -> bool:
    """Recognize deterministic evidence gates that are stronger than retrieval."""

    provenance = dict(row.get("provenance") or {})
    verification = dict(provenance.get("verification") or {})
    if bool(verification.get("degraded")):
        return False
    allowed = {
        str(value)
        for value in dict(_claim_value(claim, "metadata", {}) or {}).get(
            "paper_sha256s", []
        )
        if str(value)
    }
    source_sha = str(
        provenance.get("paper_sha256")
        or provenance.get("document_sha256")
        or ""
    )
    if allowed and source_sha not in allowed:
        return False
    if evidence_type == EvidenceType.EXPERIMENT and provenance.get("protocol_hash"):
        return True
    return bool(
        evidence_type == EvidenceType.TABLE
        and str(verification.get("verifier") or "")
        == "runtime-verified-visual-table-v1"
        and provenance.get("vision_verified")
        and dict(provenance.get("vision_metadata") or {}).get("table_structure_valid")
    )


def derive_claim_evidence_state(
    claim: ClaimRecord | Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    *,
    blockers: Iterable[str] = (),
) -> dict[str, Any]:
    """Pure evidence aggregation shared by live runs and zero-network replay.

    Strict, SHA-scoped paper Claims require at least one closure-grade item. A
    grounded semantic entailment is weighted by its adjudicated confidence rather
    than by Chinese/English lexical overlap; lexical or degraded fallbacks remain
    auditable but cannot resolve the gap on their own.
    """

    support_not = 1.0
    contradict_not = 1.0
    observed_types: set[EvidenceType] = set()
    resolved_gaps: set[str] = set()
    gap_support_not: dict[str, float] = {}
    closure_grade_ids: list[str] = []
    semantic_ids: list[str] = []
    metadata = dict(_claim_value(claim, "metadata", {}) or {})
    strict_source_scope = bool(metadata.get("paper_sha256s"))

    for row in rows:
        evidence_type = EvidenceType(str(row["evidence_type"]))
        confidence = max(0.0, min(1.0, float(row["confidence"])))
        polarity_value = str(row.get("link_polarity") or row.get("polarity"))
        if polarity_value not in {
            EvidencePolarity.SUPPORTS.value,
            EvidencePolarity.CONTRADICTS.value,
        }:
            continue
        polarity = EvidencePolarity(polarity_value)
        semantic_confidence = trusted_semantic_adjudication_confidence(
            claim, row, polarity
        )
        nonsemantic_grade = _is_closure_grade_nonsemantic_evidence(
            claim, row, evidence_type
        )
        closure_grade = semantic_confidence is not None or nonsemantic_grade
        base = _EVIDENCE_WEIGHT[evidence_type]
        contribution = max(0.0, min(0.98, base * confidence))
        if semantic_confidence is not None:
            # A directly quoted, source-bound semantic adjudication is stronger
            # than generic TEXT evidence, while remaining below certainty.
            contribution = max(contribution, min(0.95, semantic_confidence))
            semantic_ids.append(str(row.get("evidence_id") or ""))
        if closure_grade:
            closure_grade_ids.append(str(row.get("evidence_id") or ""))

        if polarity == EvidencePolarity.CONTRADICTS:
            contradict_not *= 1.0 - contribution
        else:
            support_not *= 1.0 - contribution
        observed_types.add(evidence_type)
        gap = (row.get("provenance") or {}).get("gap")
        if gap:
            gap = str(gap)
            if _evidence_satisfies_gap_contract(gap, evidence_type, row) and (
                not strict_source_scope or closure_grade
            ):
                # A relevant item alone is not a resolved information gap.
                # Previously two weak TEXT items could clear the gap while the
                # aggregate claim remained PARTIAL, leaving no action to schedule.
                gap_support_not[gap] = gap_support_not.get(gap, 1.0) * (1.0 - contribution)

    resolved_gaps.update(gap for gap, remaining in gap_support_not.items() if 1.0 - remaining >= 0.65)

    support = 1.0 - support_not
    contradict = 1.0 - contradict_not
    net = support - contradict
    gaps = [
        str(gap)
        for gap in list(_claim_value(claim, "gaps", []) or [])
        if str(gap) not in resolved_gaps
    ]
    required = [
        item if isinstance(item, EvidenceType) else EvidenceType(str(item))
        for item in list(_claim_value(claim, "required_evidence_types", []) or [])
    ]
    missing_required = [item for item in required if item not in observed_types]
    blocker_ids = sorted(set(str(value) for value in blockers if str(value)))
    closure_gate_satisfied = not strict_source_scope or bool(closure_grade_ids)
    conflict = support >= 0.65 and contradict >= 0.65 and abs(net) <= 0.12
    if (
        contradict >= 0.65
        and contradict > support + 0.12
        and closure_gate_satisfied
    ):
        status = ClaimStatus.CONTRADICTED
    elif (
        support >= 0.65
        and support > contradict + 0.12
        and not gaps
        and not missing_required
        and not blocker_ids
        and closure_gate_satisfied
    ):
        status = ClaimStatus.SUPPORTED
    elif observed_types:
        status = ClaimStatus.PARTIAL
    else:
        status = ClaimStatus.UNKNOWN
    return {
        "status": status,
        "confidence": max(0.0, min(1.0, max(support, contradict))),
        "gaps": gaps,
        "support_score": round(support, 6),
        "contradict_score": round(contradict, 6),
        "observed_evidence_types": sorted(item.value for item in observed_types),
        "missing_required_types": [item.value for item in missing_required],
        "evidence_conflict": conflict,
        "blocked_by_claim_ids": blocker_ids,
        "closure_gate_required": strict_source_scope,
        "closure_gate_satisfied": closure_gate_satisfied,
        "closure_grade_evidence_ids": sorted(set(filter(None, closure_grade_ids))),
        "semantic_adjudication_evidence_ids": sorted(set(filter(None, semantic_ids))),
        "semantic_adjudication_policy": "source-bound-grounded-quote-v1",
    }


class ClaimGraph:
    def __init__(self, db: RuntimeDatabase):
        self.db = db

    def add_claim(
        self,
        *,
        run_id: str,
        statement: str,
        importance: float = 1.0,
        confidence: float = 0.0,
        gaps: Iterable[str] | None = None,
        required_evidence_types: Iterable[EvidenceType | str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        claim_id: str | None = None,
    ) -> ClaimRecord:
        normalized_gaps = list(
            dict.fromkeys(
                str(x).strip()
                for x in (gaps or ["primary_source"])
                if str(x).strip()
            )
        )
        normalized_metadata = dict(metadata or {})
        # ``gaps`` shrinks as evidence arrives.  Persisting the original set is
        # what lets the scheduler measure real Claim coverage instead of merely
        # counting retrieved chunks.
        normalized_metadata.setdefault("initial_gaps", list(normalized_gaps))
        record = ClaimRecord(
            claim_id=claim_id or new_id("claim"),
            run_id=run_id,
            statement=statement.strip(),
            status=ClaimStatus.UNKNOWN,
            importance=max(0.0, min(1.0, float(importance))),
            confidence=max(0.0, min(1.0, float(confidence))),
            gaps=normalized_gaps,
            required_evidence_types=[
                item if isinstance(item, EvidenceType) else EvidenceType(str(item))
                for item in (required_evidence_types or [])
            ],
            metadata=normalized_metadata,
        )
        self.db.add_claim(record)
        return record

    def add_edge(self, run_id: str, source: str, target: str, relation: str, metadata: Mapping[str, Any] | None = None) -> str:
        allowed = {"supports", "depends_on", "contradicts", "qualifies", "derived_from"}
        if relation not in allowed:
            raise ValueError(f"Unsupported claim relation {relation!r}; expected one of {sorted(allowed)}")
        return self.db.add_claim_edge(run_id, source, target, relation, metadata)

    def mark_attempt(self, claim_id: str, gap: str, *, success: bool) -> ClaimRecord:
        claim = self.db.get_claim(claim_id)
        attempts = dict(claim.metadata.get("gap_attempts") or {})
        item = dict(attempts.get(gap) or {})
        item["count"] = int(item.get("count", 0)) + 1
        item["successes"] = int(item.get("successes", 0)) + (1 if success else 0)
        attempts[gap] = item
        claim.metadata["gap_attempts"] = attempts
        self.db.update_claim(claim)
        return claim

    def update_from_evidence(self, claim_id: str) -> ClaimRecord:
        claim = self.db.get_claim(claim_id)
        if self.is_terminal(claim):
            # A finalized claim is a deliberate bounded decision. Recomputing
            # must not reopen it and loop on claim.finalize.
            return claim
        blockers = self.blocking_dependencies(claim)
        state = derive_claim_evidence_state(
            claim,
            self.db.list_evidence(claim.run_id, claim_id),
            blockers=blockers,
        )
        claim.status = state.pop("status")
        claim.confidence = float(state.pop("confidence"))
        claim.gaps = list(state.pop("gaps"))
        claim.metadata.update(state)
        self.db.update_claim(claim)
        return claim

    def blocking_dependencies(self, claim: ClaimRecord) -> list[str]:
        """Return unresolved targets for edges expressed as ``source depends_on target``."""
        blockers: list[str] = []
        for edge in self.db.list_claim_edges(claim.run_id):
            if edge["relation"] != "depends_on" or edge["source_claim_id"] != claim.claim_id:
                continue
            dependency = self.db.get_claim(str(edge["target_claim_id"]))
            if dependency.status != ClaimStatus.SUPPORTED:
                blockers.append(dependency.claim_id)
        return sorted(set(blockers))

    def finalize_unresolved(self, claim_id: str, reason: str) -> ClaimRecord:
        claim = self.update_from_evidence(claim_id)
        if claim.status not in {ClaimStatus.SUPPORTED, ClaimStatus.CONTRADICTED}:
            if claim.status == ClaimStatus.PARTIAL:
                claim.metadata["partial_reason"] = reason
            else:
                claim.status = ClaimStatus.UNRESOLVED
                claim.metadata["unresolved_reason"] = reason
            claim.metadata["finalized"] = True
            claim.metadata["finalization_reason"] = reason
            self.db.update_claim(claim)
        return claim

    @staticmethod
    def is_terminal(claim: ClaimRecord) -> bool:
        return claim.status in {
            ClaimStatus.SUPPORTED,
            ClaimStatus.CONTRADICTED,
            ClaimStatus.UNRESOLVED,
        } or bool(claim.metadata.get("finalized"))

    def summary(self, run_id: str) -> dict[str, Any]:
        claims = self.db.list_claims(run_id)
        return {
            "claims": [claim.to_dict() for claim in claims],
            "edges": self.db.list_claim_edges(run_id),
            "counts": {
                status.value: sum(1 for claim in claims if claim.status == status)
                for status in ClaimStatus
            },
        }


class EvidenceGapScheduler:
    """Selects the next action by expected evidence gain per cost/risk unit."""

    def __init__(self, db: RuntimeDatabase, graph: ClaimGraph):
        self.db = db
        self.graph = graph

    def select_next(self, run_id: str, *, budget: Budget) -> ActionCandidate | None:
        run = self.db.get_run(run_id)
        if int(run["used_steps"]) >= budget.max_steps or int(run["used_tool_calls"]) >= budget.max_tool_calls:
            return None
        candidates: list[ActionCandidate] = []
        for claim in self.db.list_claims(run_id):
            if self.graph.is_terminal(claim):
                continue
            if self.graph.blocking_dependencies(claim):
                continue
            candidates.extend(self._for_claim(claim, budget))
        if not candidates:
            return None
        allowed = [candidate for candidate in candidates if self._risk(candidate).rank <= budget.max_risk.rank]
        if not allowed:
            return None
        # Once a protocol exists, finish its approval/execution/evaluation phase
        # before spending the remaining model budget on another search loop.
        # Dependency checks and risk filtering above still apply; experiment.run
        # still goes through PersistentToolRuntime's approval gate.
        continuations = [item for item in allowed
                         if item.tool_name in {'experiment.run', 'experiment.evaluate'}]
        if continuations:
            chosen = max(continuations, key=lambda item: (
                item.tool_name == 'experiment.evaluate', item.score, item.claim_id or ''))
            chosen.reason = 'Continue preregistered experiment phase: ' + chosen.reason
            return chosen
        return max(allowed, key=lambda item: (item.score, item.expected_gain, item.claim_id or ""))

    def _risk(self, action: ActionCandidate) -> RiskLevel:
        if action.tool_name == "experiment.run":
            return RiskLevel.HIGH
        if action.tool_name == "code.inspect":
            return RiskLevel.READ_ONLY
        return RiskLevel.READ_ONLY

    def _for_claim(self, claim: ClaimRecord, budget: Budget) -> list[ActionCandidate]:
        attempts = dict(claim.metadata.get("gap_attempts") or {})
        uncertainty = max(0.05, 1.0 - claim.confidence)
        coverage = build_claim_coverage(
            claim,
            self.db.list_evidence(claim.run_id, claim.claim_id),
        )
        coverage_need = max(0.0, 1.0 - float(coverage["coverage_ratio"]))
        paper_coverage = coverage.get("paper_coverage_ratio")
        result: list[ActionCandidate] = []
        gaps = claim.gaps or self._missing_required_gaps(claim)
        if not gaps:
            return [
                ActionCandidate(
                    tool_name="claim.finalize",
                    arguments={"claim_id": claim.claim_id},
                    claim_id=claim.claim_id,
                    reason="No unresolved evidence gaps remain",
                    expected_gain=0.2 * claim.importance,
                    token_cost=0.05,
                    time_cost=0.05,
                    risk_cost=0.0,
                )
            ]
        for gap in gaps:
            attempt_count = int((attempts.get(gap) or {}).get("count", 0))
            # Evaluation is deterministic for a fixed protocol/result pair; a
            # second pass cannot add information and would only duplicate the
            # same evidence record.
            max_attempts = 1 if gap in {"experiment_evaluation", "executed_experiment_result"} else int(
                claim.metadata.get("max_gap_attempts", 2)
            )
            if attempt_count >= max_attempts:
                continue
            action = self._action_for_gap(
                claim,
                gap,
                attempt_index=attempt_count,
            )
            if action is None:
                continue
            decay = 1.0 / (1.0 + attempt_count)
            action.arguments.setdefault("attempt_index", attempt_count + 1)
            action.expected_gain *= claim.importance * uncertainty * decay
            # Coverage rewards closing an explicit evidence contract.  Paper
            # diversity rewards reaching missing scoped papers, while the
            # uncertainty term favors Claims whose current conclusion is least
            # stable.  All components are persisted in ActionCandidate traces.
            action.coverage_gain = (
                0.20
                + 0.40 * coverage_need
                + (0.10 if coverage.get("missing_evidence_types") else 0.0)
            ) * decay
            action.diversity_gain = (
                0.20 * max(0.0, 1.0 - float(paper_coverage)) * decay
                if paper_coverage is not None
                else 0.0
            )
            action.uncertainty_reduction = min(0.25, uncertainty * 0.20) * decay
            action.utility_version = "claim-coverage-cost-v1"
            result.append(action)
        if not result and gaps:
            result.append(
                ActionCandidate(
                    tool_name="claim.finalize",
                    arguments={"claim_id": claim.claim_id, "reason": "Evidence gaps exhausted within retry budget"},
                    claim_id=claim.claim_id,
                    reason="No safe action remains for unresolved gaps",
                    expected_gain=0.1 * claim.importance,
                    token_cost=0.05,
                    time_cost=0.05,
                    risk_cost=0.0,
                )
            )
        return result

    def _missing_required_gaps(self, claim: ClaimRecord) -> list[str]:
        observed = {EvidenceType(x) for x in claim.metadata.get("observed_evidence_types", [])}
        inverse = {value: key for key, value in _GAP_TO_EVIDENCE.items()}
        return [inverse.get(item, item.value.lower()) for item in claim.required_evidence_types if item not in observed]

    def _action_for_gap(
        self,
        claim: ClaimRecord,
        gap: str,
        *,
        attempt_index: int = 0,
    ) -> ActionCandidate | None:
        metadata = claim.metadata
        common = {"claim_id": claim.claim_id, "gap": gap}
        paper_sha256s = metadata.get("paper_sha256s")
        if isinstance(paper_sha256s, (list, tuple, set)) and paper_sha256s:
            common["paper_sha256s"] = sorted({str(value) for value in paper_sha256s})
        if gap in {"experiment_evaluation", "executed_experiment_result"}:
            result = metadata.get("experiment_result")
            protocol = metadata.get("experiment")
            if not isinstance(result, Mapping) or not isinstance(protocol, Mapping):
                # Legacy planners used ``experiment_evaluation`` both for
                # experiments reported in papers and for an Agent-executed
                # protocol. Preserve resumability by treating the former as a
                # literature search; the explicit new gap remains execution-only.
                if gap == "experiment_evaluation":
                    pass
                else:
                    return None
            else:
                return ActionCandidate(
                    tool_name="experiment.evaluate",
                    arguments={"result": dict(result), "protocol": dict(protocol), **common},
                    claim_id=claim.claim_id,
                    gap=gap,
                    reason="Evaluate sandbox metrics against the preregistered protocol",
                    expected_gain=0.95,
                    token_cost=0.05,
                    time_cost=0.1,
                    risk_cost=0.0,
                )
        if gap in {"independent_experiment", "reproduction"}:
            experiment = metadata.get("experiment")
            if not isinstance(experiment, Mapping):
                experiment_spec = metadata.get("experiment_spec")
                if not isinstance(experiment_spec, Mapping):
                    return None
                return ActionCandidate(
                    tool_name="experiment.plan",
                    arguments={
                        "spec": dict(experiment_spec),
                        "source_dir": metadata.get("experiment_source_dir"),
                        "command": metadata.get("experiment_command"),
                        "output_dir": metadata.get("experiment_output_dir"),
                        "dataset": metadata.get("experiment_dataset"),
                        "docker_image": metadata.get("experiment_docker_image"),
                        **common,
                    },
                    claim_id=claim.claim_id,
                    gap=gap,
                    reason="Compile a model proposal into a validated preregistered protocol",
                    expected_gain=0.8,
                    token_cost=0.05,
                    time_cost=0.1,
                    risk_cost=0.0,
                )
            return ActionCandidate(
                tool_name="experiment.run",
                arguments={**dict(experiment), **common},
                claim_id=claim.claim_id,
                gap=gap,
                reason=f"Run a preregistered isolated experiment to close {gap}",
                expected_gain=1.0,
                token_cost=0.1,
                time_cost=float(metadata.get("experiment_time_cost", 1.0)),
                risk_cost=1.5,
            )
        if gap in {"implementation", "code"}:
            path = metadata.get("repository_path") or metadata.get("code_path")
            if not path:
                return None
            return ActionCandidate(
                tool_name="code.inspect",
                arguments={"path": str(path), "query": claim.statement, **common},
                claim_id=claim.claim_id,
                gap=gap,
                reason="Inspect implementation without executing repository code",
                expected_gain=0.7,
                token_cost=0.2,
                time_cost=0.2,
                risk_cost=0.05,
            )
        if gap in {"numeric_results", "table", "qualitative_result", "figure", "formula", "equation"}:
            pdf_path = metadata.get("pdf_path")
            pdf_paths = metadata.get("pdf_paths")
            if not pdf_path and isinstance(pdf_paths, (list, tuple)) and pdf_paths:
                pdf_path = pdf_paths[attempt_index % len(pdf_paths)]
            # When the question explicitly names a paper and the resolver has
            # already bound its exact local PDF/SHA, inspect the table first.
            # This avoids spending one semantic-verifier call merely to learn
            # that an already-known paper contains an ablation table.  Broad or
            # weakly scoped numeric questions retain the text-discovery-first
            # path because they still need paper/component routing.
            strict_named_pdf = bool(
                pdf_path
                and str(metadata.get("source_scope_mode") or "")
                == "strict_explicit_title"
            )
            layout_attempt = (
                0 if gap == "numeric_results" and strict_named_pdf else
                1 if gap == "numeric_results" else
                0
            )
            if pdf_path and attempt_index == layout_attempt:
                object_type = {
                    "numeric_results": "table",
                    "table": "table",
                    "qualitative_result": "figure",
                    "figure": "figure",
                    "formula": "equation",
                    "equation": "equation",
                }[gap]
                layout_queries = [
                    str(item).strip()
                    for item in metadata.get("candidate_queries", [])
                    if str(item).strip()
                ]
                discovery_hints = self._discovery_hints(claim, gap)
                # Never let sibling-Claim discovery hints erase an explicit
                # object reference from the active Claim.  In a multi-paper
                # run this previously turned "Table XII" into a generic
                # ablation query, allowing another table to consume the
                # bounded vision budget before the named object was read.
                query_parts = [
                    claim.statement,
                    *layout_queries[:3],
                    "ablation quantitative results table",
                    *discovery_hints,
                ]
                layout_query = " ".join(dict.fromkeys(query_parts))
                return ActionCandidate(
                    tool_name="paper.inspect_layout",
                    arguments={"pdf_path": str(pdf_path), "object_type": object_type, "query": layout_query, **common},
                    claim_id=claim.claim_id,
                    gap=gap,
                    reason=f"Inspect layout-aware {object_type} evidence",
                    # A concrete object named by the user is bounded and highly
                    # informative.  Schedule it before broad method searches so
                    # semantic verification cannot starve the requested table
                    # or figure under a tight provider-attempt budget.
                    expected_gain=0.95 if strict_named_pdf else 0.8,
                    token_cost=0.25,
                    time_cost=0.3,
                    risk_cost=0.02,
                )
        queries = [
            str(item).strip()
            for item in metadata.get("candidate_queries", [])
            if str(item).strip()
        ]
        query = queries[min(attempt_index, len(queries) - 1)] if queries else claim.statement
        if gap == "counter_evidence":
            query = f"counter evidence limitations failure {claim.statement}"
        external_candidates = metadata.get("external_candidates") or []
        if attempt_index >= 3 and isinstance(external_candidates, list) and external_candidates:
            candidate = next(
                (
                    item
                    for item in external_candidates
                    if isinstance(item, Mapping) and item.get("full_text_url")
                ),
                None,
            )
            if candidate is not None:
                return ActionCandidate(
                    tool_name="scholar.fetch",
                    arguments={
                        "url": str(candidate["full_text_url"]),
                        "source_id": candidate.get("source_id"),
                        "title": candidate.get("title"),
                        **common,
                    },
                    claim_id=claim.claim_id,
                    gap=gap,
                    reason="Fetch allowlisted scholarly full text discovered by metadata search",
                    expected_gain=0.65,
                    token_cost=0.1,
                    time_cost=0.35,
                    risk_cost=0.02,
                )
        if attempt_index >= 2 and bool(metadata.get("allow_external_search", False)):
            expected_titles = [
                str(value).strip()
                for value in metadata.get("expected_paper_titles", [])
                if str(value).strip()
            ][:8]
            return ActionCandidate(
                tool_name="scholar.search",
                arguments={
                    "query": query,
                    "top_k": min(8, int(metadata.get("top_k", 8))),
                    "expected_titles": expected_titles,
                    **common,
                },
                claim_id=claim.claim_id,
                gap=gap,
                reason="Local evidence remains insufficient; query traceable scholarly APIs",
                expected_gain=0.5,
                token_cost=0.1,
                time_cost=0.3,
                risk_cost=0.01,
            )
        return ActionCandidate(
            tool_name="paper.search",
            arguments={
                "query": query,
                "queries": queries[:3] if attempt_index == 0 and queries else [query],
                "top_k": int(metadata.get("top_k", 8)),
                **common,
            },
            claim_id=claim.claim_id,
            gap=gap,
            reason=f"Retrieve primary evidence for gap {gap}",
            expected_gain=(
                0.5 if gap == "numeric_results" and attempt_index == 0 else 0.55
            ),
            token_cost=0.3,
            time_cost=0.2,
            risk_cost=0.01,
        )

    def _discovery_hints(self, claim: ClaimRecord, gap: str) -> list[str]:
        hints: list[str] = []
        acronym_counts: dict[str, int] = {}
        allowed_papers = {
            str(value)
            for value in claim.metadata.get("paper_sha256s", [])
            if str(value)
        }
        for row in self.db.list_evidence(claim.run_id):
            provenance = dict(row.get("provenance") or {})
            row_paper = str(
                provenance.get("paper_sha256")
                or provenance.get("document_sha256")
                or ""
            )
            if allowed_papers and row_paper not in allowed_papers:
                continue
            same_claim = claim.claim_id in {
                str(value) for value in row.get("claim_ids", []) if str(value)
            }
            if row.get("claim_id"):
                same_claim = same_claim or str(row["claim_id"]) == claim.claim_id
            # Use the numeric claim's own discovery pass plus already verified
            # method evidence from a sibling claim scoped to the same paper.
            if same_claim and str(provenance.get("gap") or "") != gap:
                continue
            verification = dict(provenance.get("verification") or {})
            quote = str(verification.get("grounded_quote") or "").strip()
            if quote:
                hints.append(quote[:320])
            content = re.sub(r"\s+", " ", str(row.get("content") or ""))
            acronym_source = f"{quote} {content}"
            for acronym in _DISCOVERY_ACRONYM_RE.findall(acronym_source):
                if (
                    acronym in _DISCOVERY_ACRONYM_STOP
                    or re.fullmatch(r"F[A-Z]?\d+|S\d+", acronym)
                ):
                    continue
                acronym_counts[acronym] = acronym_counts.get(acronym, 0) + 1
            for match in _DISCOVERY_PHRASE_RE.finditer(content):
                phrase = match.group(1).strip()
                if phrase:
                    hints.append(phrase[:320])
                if len(hints) >= 6:
                    break
            if len(hints) >= 6:
                break
        unique = list(dict.fromkeys(hints))

        def information_score(value: str) -> int:
            folded = value.casefold()
            list_markers = value.count(",") + value.count("、") + folded.count(" and ")
            experiment_terms = sum(
                marker in folded
                for marker in (
                    "sequential",
                    "cross",
                    "parallel",
                    "baseline",
                    "without",
                    "offset",
                    "variant",
                    "消融",
                    "基线",
                )
            )
            return list_markers * 2 + experiment_terms

        ranked = sorted(
            unique,
            key=lambda value: (information_score(value), -len(value)),
            reverse=True,
        )
        if ranked and information_score(ranked[0]) > 0:
            best = information_score(ranked[0])
            ranked = [value for value in ranked if information_score(value) == best]
        # One highly specific enumeration plus repeated module acronyms is
        # usually a better table-routing query than several long prose
        # sentences full of generic model terms.
        selected = ranked[:1]
        acronyms = sorted(
            acronym_counts,
            key=lambda value: (acronym_counts[value], len(value), value),
            reverse=True,
        )[:4]
        if acronyms:
            selected.append(" ".join(acronyms))
        return selected
