"""Claim-aware, recoverable context engineering for PaperPilot.

The runtime keeps raw evidence, tool results and conversation records outside the
model window.  This module builds a small *working set* for one decision, records
exact lineage back to durable state, and applies deterministic compaction before
anything is sent to an LLM.  A summary is navigation state only; it is never
accepted as scientific evidence.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .db import RuntimeDatabase
from .models import ClaimRecord, ClaimStatus, stable_hash


_GAP_TO_TYPE = {
    "primary_source": "TEXT",
    "text": "TEXT",
    "definition": "TEXT",
    "counter_evidence": "TEXT",
    "numeric_results": "TABLE",
    "table": "TABLE",
    "qualitative_result": "FIGURE",
    "figure": "FIGURE",
    "formula": "EQUATION",
    "equation": "EQUATION",
    "implementation": "CODE",
    "code": "CODE",
    "independent_experiment": "EXPERIMENT",
    "reproduction": "EXPERIMENT",
    "experiment_evaluation": "EXPERIMENT",
    "executed_experiment_result": "EXPERIMENT",
}

_MECHANISM_QUESTION_RE = re.compile(
    r"(?:机制|模块|如何(?:组织|工作|处理)|作用于什么|mechanism|how\s+.*(?:work|operate))",
    re.IGNORECASE,
)
_MECHANISM_ANCHOR_PATTERNS = (
    re.compile(
        r"\b([A-Z][A-Za-z0-9_-]{1,30})\s+(?:(?i:modeling)\s+)?(?i:mechanism)\b",
    ),
    re.compile(
        r"\b([A-Z][A-Za-z0-9_-]{1,30})(?:\s*的)?(?:核心)?(?:机制|模块)\b",
    ),
)
_MECHANISM_ANCHOR_STOPWORDS = {
    "changeMamba".casefold(),
    "flowMamba".casefold(),
    "paper".casefold(),
    "table".casefold(),
    "model".casefold(),
}
_QUESTION_IDENTIFIER_STOPWORDS = _MECHANISM_ANCHOR_STOPWORDS | {
    value.casefold()
    for value in (
        "Remote",
        "Sensing",
        "Change",
        "Detection",
        "With",
        "Spatiotemporal",
        "State",
        "Space",
        "Building",
        "Damage",
        "Assessment",
        "Optics",
        "Flow",
        "Based",
        "Sequential",
        "Cross",
        "Parallel",
    )
}
# The three common organization labels above are meaningful anchors despite also
# being ordinary English words. Keep them out of the prose-stopword set.
_QUESTION_IDENTIFIER_STOPWORDS -= {"sequential", "cross", "parallel"}
_OPERATION_CONCEPTS: tuple[tuple[str, re.Pattern[str], re.Pattern[str]], ...] = (
    (
        "temporal_order",
        re.compile(r"temporal\s+order|time\s+order|时间顺序|时序(?:展开|排列)", re.I),
        re.compile(r"temporal\s+order|time\s+order|时间顺序|时序(?:展开|排列)|依次(?:连接|排列)|先.{0,20}(?:再|后)", re.I),
    ),
    (
        "cross_ordering",
        re.compile(r"cross[-\s]?order|interleav|alternat(?:e|ing)|交叉排序|交错|交替", re.I),
        re.compile(r"cross[-\s]?order|interleav|alternat(?:e|ing)|交叉排序|交错|交替", re.I),
    ),
    (
        "channel_dimension",
        re.compile(r"channel\s+dimension|通道维", re.I),
        re.compile(r"channel\s+dimension|通道维", re.I),
    ),
    (
        "concatenation",
        re.compile(r"concatenat|\bconcat[a-z]*\b|拼接|串接", re.I),
        re.compile(r"concatenat|\bconcat[a-z]*\b|拼接|串接", re.I),
    ),
    (
        "joint_modeling",
        re.compile(r"joint(?:ly)?\s+(?:model|modeling)|performing\s+joint|联合建模|共同建模", re.I),
        re.compile(r"joint(?:ly)?\s+(?:model|modeling)|performing\s+joint|联合建模|共同建模", re.I),
    ),
    (
        "optical_flow",
        re.compile(r"optical\s+flow|flow\s+field|光流", re.I),
        re.compile(r"optical\s+flow|flow\s+field|光流", re.I),
    ),
    (
        "pixel_displacement",
        re.compile(r"pixel[-\s]?wise\s+displacement|spatial\s+(?:misalignment|offset)|像素(?:级)?位移|空间错位|空间偏移", re.I),
        re.compile(r"pixel[-\s]?wise\s+displacement|spatial\s+(?:misalignment|offset)|像素(?:级)?位移|空间错位|空间偏移", re.I),
    ),
    (
        "warping",
        re.compile(r"\bwarp(?:ing|ed|s)?\b|remap|扭曲|形变|重采样", re.I),
        re.compile(r"\bwarp(?:ing|ed|s)?\b|remap|扭曲|形变|重采样", re.I),
    ),
)
_UNSUPPORTED_MECHANISM_OPERATIONS: tuple[
    tuple[str, re.Pattern[str], re.Pattern[str]], ...
] = (
    (
        "independent_processing",
        re.compile(r"\bindependen\w*\b|独立(?:地)?(?:处理|建模|分支)?", re.I),
        re.compile(r"\bindependen\w*\b|独立(?:地)?(?:处理|建模|分支)?", re.I),
    ),
    (
        "branch_processing",
        re.compile(r"\bbranch(?:es)?\b|分支", re.I),
        re.compile(r"\bbranch(?:es)?\b|分支", re.I),
    ),
    (
        "attention",
        re.compile(r"\battention\b|注意力", re.I),
        re.compile(r"\battention\b|注意力", re.I),
    ),
    (
        "fusion",
        re.compile(r"\bfus(?:e|es|ed|ing|ion)\b|融合", re.I),
        re.compile(r"\bfus(?:e|es|ed|ing|ion)\b|融合", re.I),
    ),
    (
        "linear_processing",
        re.compile(r"\blinear(?:ly)?\b|线性", re.I),
        re.compile(r"\blinear(?:ly)?\b|线性", re.I),
    ),
)
_CHINESE_OPERATION_NEGATION_RE = re.compile(
    r"(?:并非|并不(?:是)?|不是|未(?:采用|进行|使用|引入|涉及)?|"
    r"没有|无需|无须|而非|而不是|避免|"
    r"不(?:把|会|是|采用|进行|通过|需要|应当|应该|再)?)"
    r"[^，,。；;！？!?：:\n]{0,12}$",
    re.I,
)
_ENGLISH_OPERATION_NEGATION_RE = re.compile(
    r"(?:\bnot\b|\bnever\b|\bwithout\b|\brather\s+than\b|\binstead\s+of\b)"
    r"(?:[\s\W]+[A-Za-z0-9_-]+){0,4}[\s\W]*$",
    re.I,
)
_SPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|px)?(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def requested_answer_sections(question: str) -> list[str]:
    """Extract explicit answer-structure requirements from a user question."""

    text = str(question or "")
    separation_requested = any(
        marker in text for marker in ("明确分成", "分别标注", "分开回答", "区分")
    )
    labels: list[str] = []
    if separation_requested and "方法机制" in text and "实验结论" in text:
        labels.extend(["方法机制", "实验结论"])
    if separation_requested and "论文事实" in text and "比较性推断" in text:
        labels.extend(["论文事实", "比较性推断"])
    if "结论边界" in text and (separation_requested or labels):
        labels.append("结论边界")
    return list(dict.fromkeys(labels))


def _canonical_numbers(value: str) -> set[str]:
    """Compare numeric values rather than harmless formatting differences.

    Table cells commonly omit a percent or pixel suffix that is stated once in
    the column header.  The structured synthesizer already accepts that form;
    the final audit must apply the same rule or it can reject its own validated
    output.  Numeric values that are genuinely absent remain violations.
    """

    result: set[str] = set()
    for raw in _NUMBER_RE.findall(str(value or "")):
        number = re.sub(r"(?:%|px)$", "", raw, flags=re.IGNORECASE)
        try:
            normalized = format(float(number), ".15g")
        except ValueError:
            normalized = number.casefold()
        result.add(normalized)
    return result


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    """Soft/hard working-set limits, independent from provider hard limits."""

    controller_soft_tokens: int = 6_000
    controller_hard_tokens: int = 8_000
    synthesis_soft_tokens: int = 24_000
    synthesis_hard_tokens: int = 32_000
    max_recent_actions: int = 6
    max_controller_evidence: int = 8
    controller_excerpt_chars: int = 1_000
    text_synthesis_excerpt_chars: int = 1_400
    structured_synthesis_excerpt_chars: int = 4_000
    policy_version: str = "claim-working-set-v1"

    def __post_init__(self) -> None:
        if not 512 <= self.controller_soft_tokens <= self.controller_hard_tokens:
            raise ValueError("invalid controller context limits")
        if not 1_024 <= self.synthesis_soft_tokens <= self.synthesis_hard_tokens:
            raise ValueError("invalid synthesis context limits")


@dataclass(frozen=True, slots=True)
class ContextAssembly:
    snapshot_id: str
    stage: str
    payload: dict[str, Any]
    estimated_tokens_before: int
    estimated_tokens_after: int
    compressed: bool
    compression_reason: str | None
    lineage: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def estimate_context_tokens(value: Any) -> int:
    """Conservative language-neutral estimate for context selection.

    Provider admission still uses the stricter UTF-8 byte upper bound in
    ``PersistentLLMRuntime``.  This estimate is only for choosing a useful
    working set: roughly four ASCII characters or three UTF-8 bytes per token,
    whichever is larger, plus a small JSON-structure allowance.
    """

    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    by_chars = math.ceil(len(text) / 4)
    by_bytes = math.ceil(len(text.encode("utf-8")) / 3)
    return max(1, by_chars, by_bytes) + 16


def _claim_value(claim: ClaimRecord | Mapping[str, Any], key: str, default: Any) -> Any:
    if isinstance(claim, Mapping):
        return claim.get(key, default)
    return getattr(claim, key, default)


def build_claim_coverage(
    claim: ClaimRecord | Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a compact coverage row used by both scheduling and context."""

    metadata = dict(_claim_value(claim, "metadata", {}) or {})
    initial_gaps = list(
        dict.fromkeys(
            str(item)
            for item in (metadata.get("initial_gaps") or _claim_value(claim, "gaps", []) or [])
            if str(item)
        )
    )
    open_gaps = list(
        dict.fromkeys(str(item) for item in (_claim_value(claim, "gaps", []) or []) if str(item))
    )
    if not initial_gaps:
        initial_gaps = list(open_gaps)
    resolved_gaps = [item for item in initial_gaps if item not in set(open_gaps)]
    observed_types: set[str] = set()
    strong_ids: list[str] = []
    support_ids: list[str] = []
    contradict_ids: list[str] = []
    neutral_ids: list[str] = []
    papers: set[str] = set()
    degraded_count = 0
    for item in evidence:
        evidence_id = str(item.get("evidence_id") or "")
        evidence_type = str(item.get("evidence_type") or "TEXT")
        polarity = str(item.get("link_polarity") or item.get("polarity") or "NEUTRAL")
        provenance = dict(item.get("provenance") or {})
        verification = dict(provenance.get("verification") or {})
        paper = str(
            provenance.get("paper_sha256")
            or provenance.get("document_sha256")
            or provenance.get("source_id")
            or ""
        )
        if paper:
            papers.add(paper)
        observed_types.add(evidence_type)
        if polarity == "SUPPORTS":
            support_ids.append(evidence_id)
        elif polarity == "CONTRADICTS":
            contradict_ids.append(evidence_id)
        else:
            neutral_ids.append(evidence_id)
        degraded = bool(verification.get("degraded"))
        degraded_count += int(degraded)
        relevant = bool(verification.get("relevant", polarity != "NEUTRAL"))
        strict_verifier = str(verification.get("verifier") or "")
        if (
            evidence_id
            and polarity != "NEUTRAL"
            and relevant
            and not degraded
            and float(item.get("confidence") or 0.0) >= 0.65
            and strict_verifier not in {"lexical-fallback-v1", "runtime-verifier-budget-v2"}
        ):
            strong_ids.append(evidence_id)
    total = max(1, len(initial_gaps))
    coverage_ratio = 1.0 - min(1.0, len(open_gaps) / total)
    allowed_papers = {
        str(item) for item in metadata.get("paper_sha256s", []) if str(item)
    }
    missing_types = sorted(
        {
            _GAP_TO_TYPE[gap]
            for gap in open_gaps
            if gap in _GAP_TO_TYPE and _GAP_TO_TYPE[gap] not in observed_types
        }
    )
    return {
        "claim_id": str(_claim_value(claim, "claim_id", "")),
        "statement": str(_claim_value(claim, "statement", ""))[:1_200],
        "status": str(_claim_value(claim, "status", ClaimStatus.UNKNOWN)),
        "importance": float(_claim_value(claim, "importance", 1.0)),
        "confidence": float(_claim_value(claim, "confidence", 0.0)),
        "initial_gaps": initial_gaps,
        "open_gaps": open_gaps,
        "resolved_gaps": resolved_gaps,
        "coverage_ratio": round(coverage_ratio, 6),
        "observed_evidence_types": sorted(observed_types),
        "missing_evidence_types": missing_types,
        "strong_evidence_ids": sorted(set(filter(None, strong_ids))),
        "support_count": len(set(filter(None, support_ids))),
        "contradict_count": len(set(filter(None, contradict_ids))),
        "neutral_count": len(set(filter(None, neutral_ids))),
        "degraded_evidence_count": degraded_count,
        "paper_count": len(papers),
        "expected_paper_count": len(allowed_papers),
        "paper_coverage_ratio": (
            round(min(1.0, len(papers & allowed_papers) / len(allowed_papers)), 6)
            if allowed_papers
            else None
        ),
        "evidence_conflict": bool(metadata.get("evidence_conflict")),
    }


def evidence_excerpt(item: Mapping[str, Any], *, max_chars: int) -> str:
    """Prefer grounded quotes and structured numeric rows over flat prefixes."""

    content = str(item.get("content") or "").strip()
    provenance = dict(item.get("provenance") or {})
    verification = dict(provenance.get("verification") or {})
    grounded = str(verification.get("grounded_quote") or "").strip()
    evidence_type = str(item.get("evidence_type") or "TEXT")
    pieces: list[str] = []
    if grounded:
        pieces.append(grounded)
    if evidence_type in {"TABLE", "EXPERIMENT"}:
        numeric_lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and (_NUMBER_RE.search(line) or "table" in line.casefold())
        ]
        pieces.extend(numeric_lines)
    if not pieces:
        pieces.append(content)
    normalized = _SPACE_RE.sub(" ", " | ".join(dict.fromkeys(pieces))).strip()
    if len(normalized) <= max_chars:
        return normalized
    # Preserve both the claim-bearing beginning and numeric/qualifying tail.
    head = max(1, int(max_chars * 0.65))
    tail = max(1, max_chars - head - 3)
    return f"{normalized[:head]}...{normalized[-tail:]}"


class ResearchContextEngine:
    """Build and persist a Claim-centred model working set."""

    def __init__(self, db: RuntimeDatabase, policy: ContextPolicy | None = None) -> None:
        self.db = db
        self.policy = policy or ContextPolicy()

    def build_controller_context(
        self,
        run_id: str,
        *,
        claim_id: str,
        gap: str,
        budget: Mapping[str, Any],
    ) -> ContextAssembly:
        run = self.db.get_run(run_id)
        claims = self.db.list_claims(run_id)
        target = self.db.get_claim(claim_id)
        all_evidence = self.db.list_evidence(run_id)
        target_evidence = self.db.list_evidence(run_id, claim_id)
        plan = dict(run["metadata"].get("research_plan") or {})
        coverage = [
            build_claim_coverage(
                claim,
                [
                    item
                    for item in all_evidence
                    if claim.claim_id in set(item.get("claim_ids") or [])
                    or item.get("claim_id") == claim.claim_id
                ],
            )
            for claim in claims
        ]
        recent_actions = self._recent_actions(run_id)
        working_evidence = self._select_working_evidence(
            target_evidence,
            gap=gap,
            maximum=self.policy.max_controller_evidence,
            excerpt_chars=self.policy.controller_excerpt_chars,
        )
        artifacts = [
            {
                "artifact_id": item["artifact_id"],
                "kind": item["kind"],
                "sha256": item.get("sha256"),
                "path_hint": str(item.get("path") or "").replace("\\", "/").rsplit("/", 1)[-1],
            }
            for item in self.db.list_artifacts(run_id)[-8:]
        ]
        payload: dict[str, Any] = {
            "policy": self.policy.policy_version,
            "pinned": {
                "question": str(run["question"])[:3_000],
                "question_hash": stable_hash(str(run["question"])),
                "objective": str(plan.get("objective") or run["question"])[:2_000],
                "constraints": [str(item)[:600] for item in plan.get("constraints", [])[:12]],
                "output_requirements": [
                    str(item)[:600] for item in plan.get("output_requirements", [])[:12]
                ],
                "evidence_is_authoritative": True,
                "summary_is_not_evidence": True,
            },
            "target": {
                "claim_id": target.claim_id,
                "statement": target.statement[:1_500],
                "gap": gap,
                "required_evidence_type": _GAP_TO_TYPE.get(gap),
            },
            "coverage_matrix": coverage,
            "recent_actions": recent_actions,
            "evidence_working_set": working_evidence,
            "artifact_refs": artifacts,
            "budget": dict(budget),
        }
        before = estimate_context_tokens(payload)
        payload, reason = self._compact_controller_payload(payload)
        after = estimate_context_tokens(payload)
        lineage = {
            "evidence_ids": [str(item["evidence_id"]) for item in working_evidence],
            "tool_call_ids": [str(item["tool_call_id"]) for item in recent_actions],
            "artifact_ids": [str(item["artifact_id"]) for item in artifacts],
            "event_ids": [
                str(item["event_id"]) for item in self.db.list_events(run_id, limit=100)[-20:]
            ],
        }
        snapshot_id = self.db.add_context_snapshot(
            run_id=run_id,
            stage="controller",
            policy_version=self.policy.policy_version,
            payload=payload,
            lineage=lineage,
            estimated_tokens_before=before,
            estimated_tokens_after=after,
            compressed=reason is not None,
            compression_reason=reason,
        )
        return ContextAssembly(
            snapshot_id=snapshot_id,
            stage="controller",
            payload=payload,
            estimated_tokens_before=before,
            estimated_tokens_after=after,
            compressed=reason is not None,
            compression_reason=reason,
            lineage=lineage,
        )

    def prepare_synthesis_context(
        self,
        run_id: str,
        *,
        question: str,
        claims: Sequence[Mapping[str, Any]],
        evidence: Sequence[Mapping[str, Any]],
        selection_stats: Mapping[str, Any],
    ) -> tuple[dict[str, str], ContextAssembly]:
        excerpts = {
            str(item["evidence_id"]): evidence_excerpt(
                item,
                max_chars=(
                    self.policy.structured_synthesis_excerpt_chars
                    if str(item.get("evidence_type")) in {"TABLE", "EXPERIMENT", "EQUATION"}
                    else self.policy.text_synthesis_excerpt_chars
                ),
            )
            for item in evidence
        }
        payload: dict[str, Any] = {
            "policy": self.policy.policy_version,
            "question": question[:3_000],
            "claims": [
                {
                    "claim_id": item.get("claim_id"),
                    "statement": str(item.get("statement") or "")[:1_500],
                    "status": item.get("status"),
                    "gaps": list(item.get("gaps") or []),
                }
                for item in claims
            ],
            "evidence": [
                {
                    "evidence_id": item.get("evidence_id"),
                    "claim_ids": item.get("claim_ids")
                    or ([item.get("claim_id")] if item.get("claim_id") else []),
                    "polarity": item.get("polarity"),
                    "type": item.get("evidence_type"),
                    "page": item.get("page"),
                    "content": excerpts[str(item["evidence_id"])],
                }
                for item in evidence
            ],
            "selection": dict(selection_stats),
        }
        before = estimate_context_tokens(payload)
        reason: str | None = None
        if before > self.policy.synthesis_soft_tokens:
            reason = "synthesis_soft_limit"
            # Shrink TEXT first; structured numeric evidence keeps the larger
            # allowance because a partial table can silently change meaning.
            for item in payload["evidence"]:
                if item["type"] == "TEXT":
                    item["content"] = evidence_excerpt(
                        next(row for row in evidence if row["evidence_id"] == item["evidence_id"]),
                        max_chars=700,
                    )
            if estimate_context_tokens(payload) > self.policy.synthesis_hard_tokens:
                for item in payload["evidence"]:
                    limit = 1_600 if item["type"] in {"TABLE", "EXPERIMENT", "EQUATION"} else 360
                    source = next(row for row in evidence if row["evidence_id"] == item["evidence_id"])
                    item["content"] = evidence_excerpt(source, max_chars=limit)
        after = estimate_context_tokens(payload)
        if after > self.policy.synthesis_hard_tokens:
            raise ValueError(
                "Claim-fair synthesis context cannot fit the configured hard limit without "
                "dropping an entire Claim; refuse instead of silently losing evidence"
            )
        excerpts = {
            str(item["evidence_id"]): str(item["content"])
            for item in payload["evidence"]
        }
        lineage = {
            "evidence_ids": sorted(excerpts),
            "tool_call_ids": [],
            "artifact_ids": [],
            "event_ids": [],
        }
        snapshot_id = self.db.add_context_snapshot(
            run_id=run_id,
            stage="synthesizer",
            policy_version=self.policy.policy_version,
            payload=payload,
            lineage=lineage,
            estimated_tokens_before=before,
            estimated_tokens_after=after,
            compressed=reason is not None,
            compression_reason=reason,
        )
        return excerpts, ContextAssembly(
            snapshot_id=snapshot_id,
            stage="synthesizer",
            payload=payload,
            estimated_tokens_before=before,
            estimated_tokens_after=after,
            compressed=reason is not None,
            compression_reason=reason,
            lineage=lineage,
        )

    def restore(self, snapshot_id: str) -> dict[str, Any]:
        """Restore the exact compact model view and its source lineage."""

        return self.db.get_context_snapshot(snapshot_id)

    def _recent_actions(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.db.list_tool_calls(run_id)[-self.policy.max_recent_actions :]
        result: list[dict[str, Any]] = []
        for row in rows:
            args = dict(row.get("args") or row.get("arguments") or {})
            output = dict(row.get("result") or {})
            tool_output = dict(output.get("output") or {})
            result.append(
                {
                    "tool_call_id": row["tool_call_id"],
                    "tool": row["tool_name"],
                    "status": row["status"],
                    "claim_id": args.get("claim_id"),
                    "gap": args.get("gap"),
                    "queries": [
                        str(item)[:300]
                        for item in (args.get("queries") or [args.get("query")])
                        if str(item or "").strip()
                    ][:4],
                    "result_summary": {
                        key: tool_output[key]
                        for key in (
                            "query",
                            "result_count",
                            "candidate_count",
                            "backend",
                            "reason",
                        )
                        if key in tool_output
                    },
                    "error": str(row.get("error") or "")[:500] or None,
                }
            )
        return result

    def _select_working_evidence(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        gap: str,
        maximum: int,
        excerpt_chars: int,
    ) -> list[dict[str, Any]]:
        expected_type = _GAP_TO_TYPE.get(gap)
        remaining = list(rows)
        selected: list[dict[str, Any]] = []
        seen_papers: set[str] = set()
        seen_polarities: set[str] = set()
        while remaining and len(selected) < maximum:
            def utility(item: Mapping[str, Any]) -> tuple[float, float, str]:
                provenance = dict(item.get("provenance") or {})
                verification = dict(provenance.get("verification") or {})
                paper = str(
                    provenance.get("paper_sha256")
                    or provenance.get("document_sha256")
                    or provenance.get("source_id")
                    or ""
                )
                polarity = str(item.get("link_polarity") or item.get("polarity") or "NEUTRAL")
                strict = float(
                    bool(verification)
                    and not verification.get("degraded")
                    and verification.get("relevant", polarity != "NEUTRAL")
                )
                score = (
                    strict * 3.0
                    + float(str(item.get("evidence_type")) == expected_type) * 2.0
                    + float(polarity != "NEUTRAL")
                    + float(bool(paper and paper not in seen_papers)) * 0.8
                    + float(polarity not in seen_polarities) * 0.5
                    + float(item.get("confidence") or 0.0)
                )
                return score, float(item.get("confidence") or 0.0), str(item.get("evidence_id") or "")

            best = max(remaining, key=utility)
            remaining.remove(best)
            provenance = dict(best.get("provenance") or {})
            verification = dict(provenance.get("verification") or {})
            paper = str(
                provenance.get("paper_sha256")
                or provenance.get("document_sha256")
                or provenance.get("source_id")
                or ""
            )
            polarity = str(best.get("link_polarity") or best.get("polarity") or "NEUTRAL")
            if paper:
                seen_papers.add(paper)
            seen_polarities.add(polarity)
            selected.append(
                {
                    "evidence_id": best.get("evidence_id"),
                    "type": best.get("evidence_type"),
                    "polarity": polarity,
                    "confidence": best.get("confidence"),
                    "paper_sha256": paper or None,
                    "page": best.get("page"),
                    "verification": {
                        "verifier": verification.get("verifier"),
                        "relevant": verification.get("relevant"),
                        "degraded": verification.get("degraded", False),
                        "reason": str(verification.get("reason") or "")[:500],
                    },
                    "excerpt": evidence_excerpt(best, max_chars=excerpt_chars),
                }
            )
        return selected

    def _compact_controller_payload(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None]:
        if estimate_context_tokens(payload) <= self.policy.controller_soft_tokens:
            return payload, None
        compacted = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
        reason = "controller_soft_limit"
        actions = compacted["recent_actions"]
        evidence = compacted["evidence_working_set"]
        compacted["compaction"] = {
            "omitted_action_count": max(0, len(actions) - 3),
            "omitted_evidence_count": 0,
            "raw_state_remains_durable": True,
            "summary_is_not_evidence": True,
        }
        compacted["recent_actions"] = actions[-3:]
        for item in evidence:
            item["excerpt"] = str(item.get("excerpt") or "")[:480]
        compacted["artifact_refs"] = compacted["artifact_refs"][-4:]
        if estimate_context_tokens(compacted) > self.policy.controller_hard_tokens:
            reason = "controller_hard_limit"
            compacted["recent_actions"] = compacted["recent_actions"][-2:]
            target_id = str(compacted["target"]["claim_id"])
            compacted["coverage_matrix"] = [
                item
                for item in compacted["coverage_matrix"]
                if item["claim_id"] == target_id or item["open_gaps"]
            ]
            kept = evidence[:6]
            compacted["compaction"]["omitted_evidence_count"] = len(evidence) - len(kept)
            compacted["evidence_working_set"] = kept
            for item in compacted["evidence_working_set"]:
                item["excerpt"] = str(item.get("excerpt") or "")[:220]
            compacted["artifact_refs"] = []
        if estimate_context_tokens(compacted) > self.policy.controller_hard_tokens:
            # Never remove pinned constraints or the target Claim.  Raw evidence
            # can be reloaded by ID, so retain identifiers and validation state.
            kept = compacted["evidence_working_set"][:3]
            compacted["compaction"]["omitted_evidence_count"] += max(
                0, len(compacted["evidence_working_set"]) - len(kept)
            )
            compacted["evidence_working_set"] = kept
            for item in kept:
                item["excerpt"] = str(item.get("excerpt") or "")[:120]
        if estimate_context_tokens(compacted) > self.policy.controller_hard_tokens:
            raise ValueError("Pinned research state exceeds the controller hard context limit")
        return compacted, reason


def _mechanism_anchors(value: str) -> list[str]:
    anchors: list[str] = []
    for pattern in _MECHANISM_ANCHOR_PATTERNS:
        for match in pattern.finditer(str(value or "")):
            anchor = str(match.group(1) or "").strip()
            if (
                anchor
                and anchor.casefold() not in _MECHANISM_ANCHOR_STOPWORDS
                and anchor not in anchors
            ):
                anchors.append(anchor)
    return anchors


def _question_mechanism_anchors(question: str, statement: str) -> list[str]:
    """Recover named variants when one Claim groups a user-supplied list.

    A planner may produce one Claim such as "Sequential, Cross and Parallel token
    organization" rather than three ``X mechanism`` Claims.  Candidate identifiers
    must occur in both the question and Claim, which avoids treating unrelated paper
    title words or evidence vocabulary as answer requirements.
    """

    statement_folded = str(statement or "").casefold()
    anchors: list[str] = []
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9_-]{2,30}\b", str(question or "")):
        value = match.group(0)
        folded = value.casefold()
        if (
            folded in _QUESTION_IDENTIFIER_STOPWORDS
            or re.fullmatch(r"[IVXLCDM]+", value)
            or value.endswith("Mamba")
            or folded not in statement_folded
        ):
            continue
        if value not in anchors:
            anchors.append(value)
    return anchors


def mechanism_anchors_for_claim(question: str, statement: str) -> list[str]:
    """Return stable named mechanism anchors shared by planning and auditing.

    Planner output often mixes forms such as ``Sequential mechanism`` and a
    grouped ``Sequential, Cross, and Parallel`` list.  Keeping this extraction
    in one place prevents planning guards and final-answer gates from assigning
    different names to the same user-requested mechanism.
    """

    return list(
        dict.fromkeys(
            [
                *_mechanism_anchors(statement),
                *_question_mechanism_anchors(question, statement),
            ]
        )
    )


def _best_mechanism_segment(
    content: str,
    *,
    anchor: str,
    all_anchors: Sequence[str],
) -> str:
    """Select the most operation-rich span for one named mechanism.

    A retrieved paragraph can define several variants.  Taking the whole paragraph
    would incorrectly require every variant's operations in every answer section.
    Candidate spans therefore start at the target name and stop at the next named
    mechanism before selecting the span with the most concrete operation cues.
    """

    text = str(content or "")
    if not text or not anchor:
        return ""
    target = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(anchor)}(?![A-Za-z0-9_])",
        re.I,
    )
    other = [
        value for value in all_anchors if value.casefold() != anchor.casefold()
    ]
    boundary = (
        re.compile(
            r"(?<![A-Za-z0-9_])(?:"
            + "|".join(re.escape(value) for value in other)
            + r")(?![A-Za-z0-9_])",
            re.I,
        )
        if other
        else None
    )
    candidates: list[str] = []
    for match in target.finditer(text):
        end = min(len(text), match.start() + 1_200)
        if boundary is not None:
            next_boundary = boundary.search(text, match.end())
            if next_boundary is not None:
                end = min(end, next_boundary.start())
        candidates.append(text[match.start() : end])
    if not candidates:
        return ""

    def score(candidate: str) -> tuple[int, int]:
        concept_count = sum(
            bool(evidence_pattern.search(candidate))
            for _, evidence_pattern, _ in _OPERATION_CONCEPTS
        )
        return concept_count, min(len(candidate), 1_200)

    return max(candidates, key=score)


def _concise_mechanism_excerpt(segment: str, concept_end: int) -> str:
    """Keep the defining operation while dropping the following paper section."""

    text = re.sub(r"\s+", " ", segment).strip()
    if not text:
        return ""
    boundaries = [
        match.start() + (1 if match.group(0).startswith(".") else 0)
        for match in re.finditer(
            r",\s*i\.e\.,?|\.\s+(?=(?:Finally|Through|The|Based|In|We)\b)",
            text,
            re.I,
        )
        if match.start() >= concept_end
    ]
    end = min(boundaries) if boundaries else len(text)
    return text[: min(end, 600)].rstrip(" ,;")


def mechanism_operation_requirements(
    *,
    claims: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    question: str,
    evidence_ids_by_claim: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, list[str]]]:
    """Extract evidence-grounded operation concepts required per mechanism.

    This helper does not invent answer content or judge general semantics.  It
    only exposes narrow, bilingual operation concepts already present in the
    evidence selected for a particular Claim.  The same requirements are sent
    to the synthesizer and enforced by :func:`mechanism_fidelity_issues`.
    """

    details = mechanism_evidence_spans(
        claims=claims,
        evidence=evidence,
        question=question,
        evidence_ids_by_claim=evidence_ids_by_claim,
    )
    return {
        claim_id: {
            anchor: list(detail["concepts"])
            for anchor, detail in per_anchor.items()
        }
        for claim_id, per_anchor in details.items()
    }


def mechanism_evidence_spans(
    *,
    claims: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    question: str,
    evidence_ids_by_claim: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Select one operation-rich source span for each named mechanism."""

    if not _MECHANISM_QUESTION_RE.search(str(question or "")):
        return {}
    claim_anchors = {
        str(claim.get("claim_id") or ""): mechanism_anchors_for_claim(
            question,
            str(claim.get("statement") or ""),
        )
        for claim in claims
    }
    all_anchors = list(
        dict.fromkeys(
            anchor
            for anchors in claim_anchors.values()
            for anchor in anchors
        )
    )
    if not all_anchors:
        return {}
    evidence_by_id = {
        str(item.get("evidence_id") or ""): item for item in evidence
    }
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for claim in claims:
        claim_id = str(claim.get("claim_id") or "")
        anchors = claim_anchors.get(claim_id) or []
        if not anchors:
            continue
        per_anchor: dict[str, dict[str, Any]] = {}
        for anchor in anchors:
            best: dict[str, Any] | None = None
            for evidence_id in evidence_ids_by_claim.get(claim_id, []):
                item = evidence_by_id.get(str(evidence_id))
                if item is None:
                    continue
                segment = _best_mechanism_segment(
                    str(item.get("content") or ""),
                    anchor=anchor,
                    all_anchors=all_anchors,
                )
                concepts: list[str] = []
                concept_end = 0
                for name, evidence_pattern, _ in _OPERATION_CONCEPTS:
                    match = evidence_pattern.search(segment)
                    if match is None:
                        continue
                    concepts.append(name)
                    concept_end = max(concept_end, match.end())
                candidate = {
                    "evidence_id": str(evidence_id),
                    "excerpt": _concise_mechanism_excerpt(segment, concept_end),
                    "concepts": concepts,
                }
                if concepts and (
                    best is None
                    or (len(concepts), len(candidate["excerpt"]))
                    > (len(best["concepts"]), len(best["excerpt"]))
                ):
                    best = candidate
            if best is not None:
                per_anchor[anchor] = best
        if per_anchor:
            result[claim_id] = per_anchor
    return result


def mechanism_fidelity_issues(
    final_answer: Mapping[str, Any],
    *,
    claims: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    question: str | None = None,
) -> dict[str, list[str]]:
    """Find omitted or unsupported operations in named mechanism answers.

    This is intentionally a narrow deterministic guard, not a general LLM judge.
    It activates only for an explicit mechanism/how question, only for named
    mechanism Claims, and only for concepts visibly present in the cited span.
    Bilingual answer patterns allow a Chinese synthesis to preserve English-paper
    operations without requiring literal translation strings.
    """

    if not _MECHANISM_QUESTION_RE.search(str(question or "")):
        return {}
    section_map = {
        str(key): {str(item) for item in values}
        for key, values in dict(final_answer.get("section_evidence_map") or {}).items()
    }
    section_text_map = {
        str(key): str(value)
        for key, value in dict(final_answer.get("section_text_map") or {}).items()
    }
    evidence_by_id = {
        str(item.get("evidence_id") or ""): item for item in evidence
    }
    claim_anchors: dict[str, list[str]] = {}
    for claim in claims:
        claim_id = str(claim.get("claim_id") or "")
        statement = str(claim.get("statement") or "")
        claim_anchors[claim_id] = mechanism_anchors_for_claim(
            str(question or ""),
            statement,
        )
    all_anchors = list(
        dict.fromkeys(
            anchor
            for anchors in claim_anchors.values()
            for anchor in anchors
        )
    )
    if not all_anchors:
        return {}

    issues: dict[str, list[str]] = {}
    for claim in claims:
        claim_id = str(claim.get("claim_id") or "")
        if str(claim.get("status") or "") not in {"SUPPORTED", "PARTIAL", "CONTRADICTED"}:
            continue
        anchors = claim_anchors.get(claim_id) or []
        section_text = section_text_map.get(claim_id, "")
        cited_ids = section_map.get(claim_id, set())
        if not anchors or not section_text or not cited_ids:
            continue
        missing: list[str] = []
        for anchor in anchors:
            required: set[str] = set()
            evidence_segments: list[str] = []
            for evidence_id in cited_ids:
                item = evidence_by_id.get(evidence_id)
                if item is None:
                    continue
                segment = _best_mechanism_segment(
                    str(item.get("content") or ""),
                    anchor=anchor,
                    all_anchors=all_anchors,
                )
                evidence_segments.append(segment)
                for name, evidence_pattern, _ in _OPERATION_CONCEPTS:
                    if evidence_pattern.search(segment):
                        required.add(name)
            answer_segment = _best_mechanism_segment(
                section_text,
                anchor=anchor,
                all_anchors=all_anchors,
            )
            for name, _, answer_pattern in _OPERATION_CONCEPTS:
                if name not in required or answer_pattern.search(answer_segment):
                    continue
                missing.append(f"{anchor}:{name}" if len(anchors) > 1 else name)
            for name, evidence_pattern, answer_pattern in _UNSUPPORTED_MECHANISM_OPERATIONS:
                if not _has_unnegated_operation(answer_pattern, answer_segment) or any(
                    evidence_pattern.search(segment) for segment in evidence_segments
                ):
                    continue
                label = f"unsupported:{name}"
                missing.append(f"{anchor}:{label}" if len(anchors) > 1 else label)
        if missing:
            issues[claim_id] = missing
    return issues


def _has_unnegated_operation(pattern: re.Pattern[str], text: str) -> bool:
    """Return whether an operation is asserted instead of locally negated.

    The fidelity guard is deliberately lexical, but a bare regex treated
    ``并非独立处理`` as an assertion of independent processing.  Inspecting the
    short clause immediately before every match keeps the deterministic guard
    explainable while allowing corrective phrases in Chinese and English.  If
    any occurrence is not negated, the operation remains an assertion.
    """

    for match in pattern.finditer(str(text or "")):
        prefix = str(text or "")[max(0, match.start() - 64) : match.start()]
        clause = re.split(r"[。；;！？!?：:\n]", prefix)[-1]
        if _CHINESE_OPERATION_NEGATION_RE.search(clause):
            continue
        if _ENGLISH_OPERATION_NEGATION_RE.search(clause):
            continue
        return True
    return False


def audit_final_answer_quality(
    final_answer: Mapping[str, Any],
    *,
    claims: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    question: str | None = None,
) -> dict[str, Any]:
    """Deterministic post-synthesis contract; no LLM judge is involved."""

    known = {str(item.get("evidence_id") or "") for item in evidence}
    cited = {str(item) for item in final_answer.get("citation_ids", []) if str(item)}
    invalid = sorted(cited - known)
    claim_ids = {str(item.get("claim_id") or "") for item in claims}
    covered = {
        str(item) for item in final_answer.get("covered_claim_ids", []) if str(item)
    }
    if not covered and len(claim_ids) == 1 and str(final_answer.get("answer") or "").strip():
        covered = set(claim_ids)
    section_map = {
        str(key): {str(item) for item in values}
        for key, values in dict(final_answer.get("section_evidence_map") or {}).items()
    }
    section_text_map = {
        str(key): str(value)
        for key, value in dict(final_answer.get("section_text_map") or {}).items()
    }
    synthesis_section_map = {
        str(key): {str(item) for item in values}
        for key, values in dict(
            final_answer.get("synthesis_section_evidence_map") or {}
        ).items()
    }
    synthesis_section_text_map = {
        str(key): str(value)
        for key, value in dict(
            final_answer.get("synthesis_section_text_map") or {}
        ).items()
    }
    evidence_by_id = {
        str(item.get("evidence_id") or ""): item for item in evidence
    }
    link_map: dict[str, set[str]] = {claim_id: set() for claim_id in claim_ids}
    for item in evidence:
        evidence_id = str(item.get("evidence_id") or "")
        linked = list(item.get("claim_ids") or [])
        if item.get("claim_id") and not linked:
            linked = [item.get("claim_id")]
        claim_polarities = dict(item.get("claim_polarities") or {})
        for claim_id in linked:
            claim_id = str(claim_id)
            polarity = str(
                claim_polarities.get(claim_id)
                or (
                    item.get("link_polarity")
                    if len(linked) == 1
                    else item.get("polarity")
                )
                or item.get("polarity")
                or "NEUTRAL"
            )
            if polarity != "NEUTRAL":
                link_map.setdefault(claim_id, set()).add(evidence_id)
    source_violations = {
        claim_id: sorted(ids - link_map.get(claim_id, set()))
        for claim_id, ids in section_map.items()
        if ids - link_map.get(claim_id, set())
    }
    missing_claims = sorted(claim_ids - covered)
    missing_resolved_citations = sorted(
        claim_id
        for claim_id in claim_ids
        if str(next(item for item in claims if str(item.get("claim_id")) == claim_id).get("status"))
        in {"SUPPORTED", "PARTIAL", "CONTRADICTED"}
        and link_map.get(claim_id)
        and not (section_map.get(claim_id, set()) or (cited & link_map[claim_id]))
    )
    numeric_violations: dict[str, list[str]] = {}
    claim_echo_violations: list[str] = []
    thin_answer_violations: list[str] = []
    missing_result_values: list[str] = []
    answer_text = str(final_answer.get("answer") or "")
    question_cjk = len(re.findall(r"[\u4e00-\u9fff]", str(question or "")))
    answer_cjk = len(re.findall(r"[\u4e00-\u9fff]", answer_text))
    language_mismatch_claim_ids: list[str] = []
    if question_cjk >= 4 and section_text_map:
        for claim_id, text in section_text_map.items():
            # A deterministic Chinese conclusion-boundary suffix must not hide
            # that the model-generated substantive section is still English.
            substantive = re.split(r"结论边界[:：]", text, maxsplit=1)[0]
            cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", substantive))
            latin_chars = len(re.findall(r"[A-Za-z]", substantive))
            if latin_chars >= 20 and cjk_chars < 4:
                language_mismatch_claim_ids.append(claim_id)
    response_language_mismatch = bool(
        question_cjk >= 4 and (answer_cjk < 4 or language_mismatch_claim_ids)
    )
    question_text = str(question or "")
    required_section_labels = requested_answer_sections(question_text)
    missing_requested_section_labels = [
        label for label in required_section_labels if label not in answer_text
    ]
    mechanism_fidelity = mechanism_fidelity_issues(
        final_answer,
        claims=claims,
        evidence=evidence,
        question=question,
    )
    mechanism_unsupported = {
        claim_id: [item for item in values if "unsupported:" in item]
        for claim_id, values in mechanism_fidelity.items()
        if any("unsupported:" in item for item in values)
    }
    mechanism_omissions = {
        claim_id: [item for item in values if "unsupported:" not in item]
        for claim_id, values in mechanism_fidelity.items()
        if any("unsupported:" not in item for item in values)
    }
    claim_by_id = {
        str(item.get("claim_id") or ""): item for item in claims
    }
    for claim_id, text in section_text_map.items():
        section_ids = section_map.get(claim_id, set())
        if not section_ids:
            continue
        grounded_parts: list[str] = []
        for evidence_id in section_ids:
            source = evidence_by_id.get(evidence_id)
            if source is None:
                continue
            grounded_parts.append(str(source.get("content") or ""))
            if source.get("page") is not None:
                grounded_parts.append(str(source["page"]))
        introduced = sorted(
            _canonical_numbers(text)
            - _canonical_numbers("\n".join(grounded_parts))
        )
        if introduced:
            numeric_violations[claim_id] = introduced
        claim = claim_by_id.get(claim_id, {})
        status = str(claim.get("status") or "UNKNOWN")
        if status not in {"SUPPORTED", "PARTIAL", "CONTRADICTED"}:
            continue
        statement = str(claim.get("statement") or "")
        normalized_text = _normalize_answer_prose(text)
        normalized_statement = _normalize_answer_prose(statement)
        if (
            normalized_statement
            and normalized_statement in normalized_text
            and len(normalized_text) <= len(normalized_statement) + 24
        ):
            claim_echo_violations.append(claim_id)
        if section_ids and len(normalized_text) < 24:
            thin_answer_violations.append(claim_id)
        gaps = {str(item).casefold() for item in claim.get("gaps") or []}
        linked_sources = [
            evidence_by_id[evidence_id]
            for evidence_id in section_ids
            if evidence_id in evidence_by_id
        ]
        asks_for_results = bool(gaps & {"numeric_results", "table"})
        has_numeric_evidence = any(
            _NUMBER_RE.search(str(source.get("content") or ""))
            for source in linked_sources
        )
        if asks_for_results and has_numeric_evidence and not _NUMBER_RE.search(text):
            missing_result_values.append(claim_id)
    for section_name, text in synthesis_section_text_map.items():
        section_ids = synthesis_section_map.get(section_name, set())
        grounded_parts = [
            str(evidence_by_id[evidence_id].get("content") or "")
            for evidence_id in section_ids
            if evidence_id in evidence_by_id
        ]
        introduced = sorted(
            _canonical_numbers(text)
            - _canonical_numbers("\n".join(grounded_parts))
        )
        if introduced:
            numeric_violations[f"synthesis:{section_name}"] = introduced
    issues: list[str] = []
    if invalid:
        issues.append("invalid_citations")
    if missing_claims:
        issues.append("missing_claim_sections")
    if source_violations:
        issues.append("cross_claim_citation")
    if missing_resolved_citations:
        issues.append("resolved_claim_without_citation")
    if numeric_violations:
        issues.append("ungrounded_numeric_literals")
    if claim_echo_violations:
        issues.append("claim_echo_without_synthesis")
    if thin_answer_violations:
        issues.append("answer_section_too_thin")
    if missing_result_values:
        issues.append("grounded_result_values_omitted")
    if response_language_mismatch:
        issues.append("response_language_mismatch")
    if missing_requested_section_labels:
        issues.append("requested_answer_sections_missing")
    if mechanism_omissions:
        issues.append("mechanism_operation_omitted")
    if mechanism_unsupported:
        issues.append("mechanism_operation_unsupported")
    return {
        "policy": "claim-answer-quality-v2",
        "passed": not issues,
        "issues": issues,
        "claim_count": len(claim_ids),
        "covered_claim_count": len(claim_ids & covered),
        "claim_coverage": round(len(claim_ids & covered) / max(1, len(claim_ids)), 6),
        "citation_count": len(cited),
        "citation_validity": round(len(cited & known) / max(1, len(cited)), 6),
        "invalid_citation_ids": invalid,
        "missing_claim_ids": missing_claims,
        "missing_resolved_citation_claim_ids": missing_resolved_citations,
        "source_violations": source_violations,
        "numeric_violations": numeric_violations,
        "claim_echo_violations": sorted(claim_echo_violations),
        "thin_answer_violations": sorted(thin_answer_violations),
        "missing_result_value_claim_ids": sorted(missing_result_values),
        "response_language_mismatch": response_language_mismatch,
        "language_mismatch_claim_ids": sorted(language_mismatch_claim_ids),
        "required_section_labels": required_section_labels,
        "missing_requested_section_labels": missing_requested_section_labels,
        "mechanism_operation_omissions": mechanism_omissions,
        "mechanism_operation_unsupported": mechanism_unsupported,
        "question_cjk_chars": question_cjk,
        "answer_cjk_chars": answer_cjk,
        "numeric_grounding_gate": final_answer.get("numeric_grounding_gate"),
    }


def _normalize_answer_prose(value: str) -> str:
    """Normalize prose only for conservative echo/empty-answer detection."""

    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").casefold())
