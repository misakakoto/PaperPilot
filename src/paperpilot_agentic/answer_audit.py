"""Compatibility interface to the default answer-first pipeline."""
from typing import Any
from .answer_contracts import (
    QUANTITY_POLICY as QUANTITY_POLICY,
    RELATION_POLICY as RELATION_POLICY,
    PAPER_SCOPE_POLICY as PAPER_SCOPE_POLICY,
    QUESTION_SCOPE_POLICY as QUESTION_SCOPE_POLICY,
    MAX_PROVIDER_ATTEMPTS as MAX_PROVIDER_ATTEMPTS,
    evidence_spans as evidence_spans,
    audit_source_view as audit_source_view,
    audit_facts_view as audit_facts_view,
    requires_chinese_answer as requires_chinese_answer,
    validate_draft as validate_draft,
    normalize_duplicate_limitations as normalize_duplicate_limitations,
    normalize_inference_dependencies as normalize_inference_dependencies,
    unrequested_structural_formula as unrequested_structural_formula,
    draft_repair_diagnostics as draft_repair_diagnostics,
    validate_audit as validate_audit,
    conservative_audit_contract as conservative_audit_contract,
    audited_presentation as audited_presentation,
)

def answer_then_audit(question: str, evidence: list[dict], model: Any, *, on_progress=None,
                     quantity_policy=False, revision_feedback=None, original_question=None,
                     frozen_draft=None, audit_cited_only=False, allow_repair=True,
                     numeric_scope_mode='enforce', request_history=None, relation_policy=False,
                     question_scope_policy=False, draft_contract='flat', requested_tables=None,
                     source_inventory=None, evidence_read_policy='full', evidence_read_max_bytes=24000,
                     shared_read_packet=None) -> dict:
    from .answer_pipeline import AnswerOptions, execute_answer_flow
    options = AnswerOptions(
        quantity_policy=quantity_policy,
        revision_feedback=revision_feedback,
        original_question=original_question,
        frozen_draft=frozen_draft,
        audit_cited_only=audit_cited_only,
        allow_repair=allow_repair,
        numeric_scope_mode=numeric_scope_mode,
        request_history=request_history,
        relation_policy=relation_policy,
        question_scope_policy=question_scope_policy,
        draft_contract=draft_contract,
        requested_tables=requested_tables,
        source_inventory=source_inventory,
        evidence_read_policy=evidence_read_policy,
        evidence_read_max_bytes=evidence_read_max_bytes,
        shared_read_packet=shared_read_packet,
    )
    return execute_answer_flow(question, evidence, model, options, on_progress)
