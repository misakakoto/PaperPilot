"""One-attempt synthesis-only validation for already-verified replay evidence.

This module does not turn a replay into a live Agent run. It exists to validate
the final grounded-synthesis contract with a hard, process-local upstream-attempt
cap after retrieval and verification have already completed.
"""

from __future__ import annotations

from typing import Any, Mapping

from .context_engine import audit_final_answer_quality
from .intelligence import LLMAnswerSynthesizer, _select_synthesis_evidence


class SingleUpstreamAttemptLLM:
    """Wrap an LLM and reject any second high-level call or provider attempt."""

    def __init__(
        self,
        inner: Any,
        *,
        force_thinking_enabled: bool | None = None,
    ) -> None:
        self.inner = inner
        self.model = str(getattr(inner, "model", "unknown"))
        self.force_thinking_enabled = force_thinking_enabled
        self.high_level_calls = 0
        self.upstream_attempts = 0
        self.last_response: Any = None

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        thinking_enabled: bool | None = None,
        before_attempt: Any = None,
    ) -> Any:
        if self.high_level_calls >= 1:
            raise RuntimeError("Synthesis-only validation permits one high-level LLM call")
        self.high_level_calls += 1

        def enforce_attempt_cap() -> None:
            if self.upstream_attempts >= 1:
                raise RuntimeError(
                    "Synthesis-only validation permits one upstream provider attempt"
                )
            self.upstream_attempts += 1
            if before_attempt is not None:
                before_attempt()

        effective_thinking = (
            self.force_thinking_enabled
            if self.force_thinking_enabled is not None
            else thinking_enabled
        )
        response = self.inner.complete(
            messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
            thinking_enabled=effective_thinking,
            before_attempt=enforce_attempt_cap,
        )
        self.last_response = response
        return response


def synthesize_verified_replay_once(
    replay: Mapping[str, Any],
    model: SingleUpstreamAttemptLLM,
) -> dict[str, Any]:
    """Synthesize an ANSWERED replay once after fail-closed preflight checks."""

    if str(replay.get("kind") or "") != "zero_network_paid_evidence_replay":
        raise ValueError("Input is not a zero-network paid-evidence replay")
    if str(replay.get("replay_research_outcome") or "") != "ANSWERED":
        raise ValueError("Synthesis-only validation requires an ANSWERED replay")
    if int(replay.get("source_scope_violations") or 0) != 0:
        raise ValueError("Replay has source-scope violations")
    if int(replay.get("network_calls") or 0) != 0 or int(
        replay.get("paid_requests") or 0
    ) != 0:
        raise ValueError("Replay input must itself be a zero-network artifact")

    claims = list(replay.get("claims") or [])
    evidence = list(replay.get("evidence") or [])
    if not claims or any(
        str(item.get("status") or "") not in {"SUPPORTED", "CONTRADICTED"}
        for item in claims
    ):
        raise ValueError("Every replay Claim must be resolved before paid synthesis")
    selected, context_stats = _select_synthesis_evidence(claims, evidence)
    if int(context_stats.get("represented_claim_count") or 0) != len(claims):
        raise ValueError("Selected synthesis context does not represent every Claim")

    result = dict(
        LLMAnswerSynthesizer(model).synthesize(
            question=str(replay.get("question") or ""),
            claims=claims,
            evidence=selected,
        )
    )
    allowed = {str(item.get("evidence_id") or "") for item in selected}
    citations = {str(item) for item in list(result.get("citation_ids") or [])}
    if str(result.get("mode") or "") != "llm-grounded-synthesis-v3":
        raise ValueError("Paid synthesis did not satisfy the structured v3 contract")
    if not citations or not citations.issubset(allowed):
        raise ValueError("Paid synthesis failed the final evidence allowlist audit")
    quality = audit_final_answer_quality(
        result,
        claims=claims,
        evidence=selected,
        question=str(replay.get("question") or ""),
    )
    if not quality["passed"]:
        raise ValueError(
            "Paid synthesis failed the final answer quality contract: "
            + ", ".join(quality["issues"])
        )
    return {
        "answer": result,
        "claim_count": len(claims),
        "selected_evidence_count": len(selected),
        "allowed_evidence_ids": sorted(allowed),
        "citation_ids": sorted(citations),
        "synthesis_context": context_stats,
        "answer_quality": quality,
    }
