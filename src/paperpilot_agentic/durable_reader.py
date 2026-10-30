"""Bridge experimental research components to shared production LLM accounting."""
import hashlib
import json
import re
from datetime import datetime, timezone

from .models import LLMCallPolicy, RunStatus, ResearchOutcome, EvidenceRecord, EvidenceType, EvidencePolarity


class ReaderAlreadyStarted(RuntimeError):
    """A live or interrupted reader must not be blindly replayed."""


def _reader_checkpoint(db, run_id, fingerprint, *, report=None, key='answer_first_checkpoint'):
    # An IMMEDIATE transaction serializes competing workers, including processes.
    # Do not hold a database transaction across any provider request.
    with db.transaction() as conn:
        row = conn.execute('SELECT metadata_json FROM runs WHERE run_id=?', (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        metadata = json.loads(row['metadata_json'])
        checkpoint = metadata.get(key)
        if checkpoint:
            if checkpoint['fingerprint'] != fingerprint:
                raise ValueError('Reader input or protocol changed; use a new run')
            if report is None:
                if checkpoint['state'] == 'FINISHED':
                    return checkpoint['report']
                raise ReaderAlreadyStarted('Reader is active or interrupted; reconcile before retrying')
        elif report is not None:
            raise ValueError('Reader checkpoint was not acquired')
        metadata[key] = {
            'fingerprint': fingerprint, 'state': 'FINISHED' if report is not None else 'STARTED',
            **({'report': report} if report is not None else {}),
        }
        conn.execute('UPDATE runs SET metadata_json=? WHERE run_id=?',
                     (json.dumps(metadata, ensure_ascii=False), run_id))
    return None


def run_durable_reader(db, run_id, evidence, model, *, protocol_id, on_progress=None, question=None,
                       finalize=True, checkpoint_key='answer_first_checkpoint', revision_feedback=None,
                       shared_read_packet=None):
    """Run source-bound answer/audit once; replay saved results without billing.

    This is the post-retrieval stage, not an autonomous full research workflow.
    Caller must identify model/config in protocol_id. A STARTED checkpoint cannot
    distinguish a live worker from a crash or lost provider response: never steal
    it on a timeout. Completed failures are also replayed, not silently retried.
    """
    from .answer_audit import answer_then_audit
    from .direct_answer import prepare_evidence

    if not protocol_id or not isinstance(protocol_id, str):
        raise ValueError('An explicit model/config protocol_id is required')
    run = db.get_run(run_id)
    question = run['question'] if question is None else question
    request_history = run['metadata'].get('answer_request_history') or []
    configured_read_policy = run['metadata'].get('reader_evidence_read_policy', 'full')
    # A bounded source recovery already supplies its own reviewed source window.
    # Do not select that window again or discard the known missing passages.
    read_policy = 'full' if revision_feedback is not None else configured_read_policy
    read_max_bytes = run['metadata'].get('reader_evidence_read_max_bytes', 24000)
    prepare_evidence(evidence)  # Reject malformed sources before acquiring the slot.
    if shared_read_packet is not None:
        from .read_packet import validate_shared_packet
        validate_shared_packet(shared_read_packet, run['question'], evidence, read_max_bytes)
    fingerprint = hashlib.sha256(json.dumps({
        'version': 'durable-answer-audit-v60-causal-scope-and-span-selection', 'question': question, 'original_question': run['question'],
        'evidence': evidence, 'protocol_id': protocol_id, 'revision_feedback': revision_feedback,
        'requested_tables': run['metadata'].get('requested_table_targets'),
        'source_inventory': run['metadata'].get('reader_source_inventory'),
        **({'read_packet_sha256': shared_read_packet.identity} if shared_read_packet is not None else {}),
        **({'request_history': request_history} if request_history else {}),
        **({'configured_evidence_read_policy': configured_read_policy, 'effective_evidence_read_policy': read_policy,
            'evidence_read_max_bytes': read_max_bytes,
            'initial_read_protocol': 'initial-whole-source-read-v1'} if configured_read_policy != 'full' else {}),
    }, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
    report = _reader_checkpoint(db, run_id, fingerprint, key=checkpoint_key)
    if report is None:
        db.update_run_status(run_id, RunStatus.RUNNING)
        sources = persist_reader_evidence(db, run_id, evidence)

        def progress(value):
            db.update_run_metadata(run_id, {'reader_progress': value})
            if on_progress:
                on_progress(value)

        report = answer_then_audit(question, sources, model, on_progress=progress,
                                   revision_feedback=revision_feedback, original_question=run['question'],
                                    request_history=request_history, relation_policy=True, question_scope_policy=True,
                                    requested_tables=run['metadata'].get('requested_table_targets'),
                                    source_inventory=run['metadata'].get('reader_source_inventory'),
                                    evidence_read_policy=read_policy, evidence_read_max_bytes=read_max_bytes,
                                    shared_read_packet=shared_read_packet)
        # Save before publication. A crash after this point can safely republish.
        _reader_checkpoint(db, run_id, fingerprint, report=report, key=checkpoint_key)
    if not finalize:
        return report
    if db.get_run(run_id)['status'] == 'CANCELLED':
        return report
    publish_reader_answer(db, run_id, report)
    status, outcome = reader_completion_state(report)
    db.update_run_outcome(run_id, outcome)
    db.update_run_status(run_id, status, error=report.get('error_type'))
    return report


def persist_reader_evidence(db, run_id, evidence):
    """Register neutral source material before generation, using run-scoped IDs."""
    result = []
    for item in evidence:
        content = str(item['content'])
        uri = item.get('source_uri')
        digest = hashlib.sha256(f'{run_id}\0{uri}\0{item.get("page")}\0{content}'.encode('utf-8')).hexdigest()
        kind = EvidenceType(item.get('evidence_type', EvidenceType.TEXT))
        provenance = dict(item.get('provenance') or {})
        stable_provenance = {key: provenance[key] for key in (
            'document_sha256', 'layout_object_id', 'paper_sha256', 'stable_id',
            'protocol_hash', 'source_hash_after', 'parser_cache_identity',
            'structured_extraction', 'reviewed_source') if provenance.get(key) is not None}
        if kind != EvidenceType.TEXT or item.get('bbox') is not None or item.get('object_type') or stable_provenance:
            # Identical transcriptions from different regions are not the same
            # visual observation. Preserve existing plain-text identity scheme.
            digest = hashlib.sha256(json.dumps({
                'text_identity': digest, 'kind': kind.value, 'bbox': item.get('bbox'),
                'object_type': item.get('object_type'),
                'stable_provenance': stable_provenance,
            }, sort_keys=True).encode('utf-8')).hexdigest()
        record = EvidenceRecord(evidence_id='evidence_' + digest[:32], run_id=run_id,
            evidence_type=kind, content=content, source_uri=uri, page=item.get('page'),
            bbox=item.get('bbox'), object_type=item.get('object_type'), artifact_path=item.get('artifact_path'),
            polarity=EvidencePolarity.NEUTRAL, confidence=0.0,
            provenance={**provenance, 'origin': 'answer-first-source-material',
                        'upstream_origin': provenance.get('origin'),
                        'retrieval_evidence_id': item['evidence_id'],
                        'page_numbering': item.get('page_numbering', 'unspecified')})
        eid, _ = db.add_evidence(record)
        result.append({**item, 'evidence_id': eid})
    return result


def publish_reader_answer(db, run_id, report):
    answer = str(report.get('answer') or '')
    citations = set(re.findall(r'\[(evidence_[A-Za-z0-9_-]+)\]', answer))
    owned = {e['evidence_id'] for e in db.list_evidence(run_id)}
    if citations - owned:
        raise ValueError('Answer cites evidence outside its run')
    sections = report.get('answer_sections')
    if sections is not None:
        if (not isinstance(sections, list) or len(sections) > 2 or
                any(not isinstance(s, dict) or set(s) != {'kind', 'text'} or
                    s['kind'] not in {'direct', 'context'} or not isinstance(s['text'], str)
                    or not s['text'].strip() for s in sections) or
                len({s['kind'] for s in sections}) != len(sections) or
                '\n\n'.join(s['text'] for s in sections) != answer):
            raise ValueError('Answer sections must preserve the complete audited answer exactly')
    result = {'answer': answer, 'citation_ids': sorted(citations),
              'mode': 'answer-first-source-audit-v3', 'review_required': True,
              'research_outcome': reader_completion_state(report)[1].value,
              'limitations': report.get('limitations', []),
              'limitations_in_answer': report.get('limitations_in_answer', False),
              'presentation_contract': report.get('presentation_contract'),
              'review_notes': [{'text': text, 'status': 'UNVERIFIED_MODEL_NOTE',
                                'origin': 'withheld_limitation'}
                               for text in dict.fromkeys(report.get('unverified_limitations', []))
                               if isinstance(text, str) and text.strip()],
              'answer_scope': report.get('answer_scope'),
              'audit': report.get('audit'), 'direct_answer_supported': report.get('direct_answer_supported', False),
              'numeric_scope': report.get('numeric_scope'),
              'answer_boundaries': report.get('answer_boundaries')}
    if sections is not None:
        result['answer_sections'] = sections
    db.update_run_metadata(run_id, {'final_answer': result, 'reader_report': report})
    return result


def reader_completion_state(report):
    if report.get('error_type') == 'LLMBudgetExhausted':
        return RunStatus.BUDGET_EXHAUSTED, ResearchOutcome.UNRESOLVED
    if report['status'] in {'FAILED', 'ROUTING_FAILED'}:
        return RunStatus.FAILED, ResearchOutcome.DEGRADED
    if report['status'] == 'NO_EVIDENCE' or not report.get('answer'):
        return RunStatus.COMPLETED, ResearchOutcome.UNRESOLVED
    if report.get('direct_answer_supported') is False:
        return RunStatus.COMPLETED, ResearchOutcome.UNRESOLVED
    # ANSWERED describes completed requested work, never factual acceptance.
    # The acceptance CLI still requires a separate five-dimension source review.
    if (report.get('answer_scope') == 'MODEL_REVIEWED'
            and report.get('completion_assurance') == 'MODEL_REVIEWED_NOT_FACTUAL_ACCEPTANCE'
            and report.get('direct_answer_supported') is True
            and report.get('audit', {}).get('coverage') == 'COMPLETE'):
        return RunStatus.COMPLETED, ResearchOutcome.ANSWERED
    return RunStatus.COMPLETED, ResearchOutcome.PARTIAL


class DurableResearchModel:
    def __init__(self, runtime, run_id, *, stage, reserve_class='general'):
        self.runtime = runtime
        self.run_id = run_id
        self.stage = stage
        self.reserve_class = reserve_class

    def for_repair(self):
        """Spend the reserved repair allowance, not the synthesis allowance."""
        return DurableResearchModel(self.runtime, self.run_id, stage='json_repair', reserve_class='repair')

    def for_content_patch(self):
        """One source-bound content revision, still inside the shared run budget."""
        return DurableResearchModel(self.runtime, self.run_id, stage='answer_content_patch', reserve_class='synthesis')

    def ensure_capacity(self, requests):
        return self.runtime.ensure_capacity(self.run_id, requests, reserve_class=self.reserve_class)

    def complete(self, messages, *, max_tokens, json_mode=False,
                 thinking_enabled=False, before_attempt=None, correlation_id=None):
        def guard():
            from .llm_runtime import LLMBudgetExhausted
            from .models import Budget
            run = self.runtime.db.get_run(self.run_id)
            if run['status'] == 'CANCELLED':
                raise InterruptedError('Run cancelled')
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(
                run['created_at'].replace('Z', '+00:00'))).total_seconds()
            if (run['metadata'].get('engine') == 'research_task' or
                    run['metadata'].get('engine') == 'research_prototype_v1' and
                    run['metadata'].get('prototype_clock_policy') == 'active-v1'):
                from .research_task import elapsed_seconds
                elapsed = elapsed_seconds(run)
            if elapsed >= Budget.from_dict(run['budget']).max_seconds:
                raise LLMBudgetExhausted('Time budget exhausted')
            if before_attempt:
                before_attempt()
        return self.runtime.complete(
            self.run_id, messages,
            policy=LLMCallPolicy(stage=self.stage, max_output_tokens=max_tokens,
                                 thinking_enabled=thinking_enabled, json_mode=json_mode,
                                 reserve_class=self.reserve_class),
            attempt_guard=guard, correlation_id=correlation_id,
        )
