"""Cross-page source reload and spending invariants, not scientific accuracy."""
from copy import deepcopy
import json

import pytest

from paperpilot_agentic.workset_review import enqueue_review, resume_review
from paperpilot_agentic.workset_synthesis import synthesize_worksets
from paperpilot_agentic.models import Budget, RunStatus
from test_workset_review import prepared


def setup(tmp_path, calls=30):
    runtime, provider, _, root, _ = prepared(tmp_path)
    rid = enqueue_review(runtime, root, budget=Budget(max_llm_calls=calls, max_tokens=1000000,
        synthesis_reserve_calls=0, repair_reserve_calls=0))['run']['run_id']
    return runtime, provider, rid


def test_cross_page_synthesis_reloads_originals_and_replays_without_billing(tmp_path):
    runtime, provider, rid = setup(tmp_path)
    manifest = runtime.db.get_run(rid)['metadata']['workset_manifest']
    pages = len(manifest['pages'])
    assert pages > 1
    resume_review(runtime, rid, max_batches=1)
    # A page's prose is deliberately poisoned: only original evidence may enter
    # the later Writer/Auditor, not this saved intermediate answer.
    checkpoint = deepcopy(runtime.db.get_run(rid)['metadata']['workset_page_0'])
    checkpoint['report']['answer'] = 'POISONED_INTERMEDIATE_ANSWER'
    runtime.db.update_run_metadata(rid, {'workset_page_0': checkpoint})
    messages = []
    original = provider.complete
    def capture(value, **kwargs):
        messages.append(deepcopy(value))
        return original(value, **kwargs)
    provider.complete = capture
    result = runtime.resume(rid)
    assert result['run']['status'] == 'COMPLETED', result['run'].get('error')
    assert result['run']['metadata']['workset_synthesis']['status'] == 'REVIEWED'
    assert provider.calls == 2 * pages + 2
    assert 'POISONED_INTERMEDIATE_ANSWER' not in json.dumps(messages, ensure_ascii=False)
    assert 'A source fact.' in json.dumps(messages, ensure_ascii=False)
    assert 'POISONED_INTERMEDIATE_ANSWER' not in result['final_answer']['answer']
    assert result['run']['research_outcome'] == 'PARTIAL'
    runtime.resume(rid)
    assert provider.calls == 2 * pages + 2
    # Simulate losing only the final status write after synthesis was saved.
    runtime.db.update_run_status(rid, RunStatus.RUNNING)
    assert runtime.resume(rid)['run']['status'] == 'COMPLETED'
    assert provider.calls == 2 * pages + 2
    checkpoint = deepcopy(runtime.db.get_run(rid)['metadata']['workset_synthesis_checkpoint'])
    checkpoint['state'] = 'STARTED'
    checkpoint.pop('report')
    runtime.db.update_run_metadata(rid, {'workset_synthesis_checkpoint': checkpoint})
    runtime.db.update_run_status(rid, RunStatus.RUNNING)
    assert runtime.resume(rid)['run']['status'] == 'RUNNING'
    assert provider.calls == 2 * pages + 2  # Unknown billing outcome is not retried.


def test_no_synthesis_budget_preserves_page_answers_and_discloses_skip(tmp_path):
    runtime, provider, rid = setup(tmp_path)
    pages = len(runtime.db.get_run(rid)['metadata']['workset_manifest']['pages'])
    with runtime.db.transaction() as conn:
        limits = runtime.db.get_run(rid)['budget']
        limits['max_llm_calls'] = 2 * pages
        conn.execute('UPDATE runs SET budget_json=? WHERE run_id=?', (json.dumps(limits), rid))
    result = runtime.resume(rid)
    assert result['run']['status'] == 'COMPLETED'
    assert result['run']['metadata']['workset_synthesis']['status'] == 'SKIPPED_CALL_BUDGET'
    assert '只读审阅' in result['final_answer']['answer']
    assert 'SKIPPED_CALL_BUDGET' in result['final_answer']['limitations'][-1]
    assert provider.calls == 2 * pages


def test_oversized_requirements_never_silently_truncated(tmp_path):
    runtime, provider, rid = setup(tmp_path)
    manifest = deepcopy(runtime.db.get_run(rid)['metadata']['workset_manifest'])
    for page in manifest['pages']:
        page['conditions']['no_train']['text'] = 'restriction' * 2000
    result = synthesize_worksets(runtime, rid, manifest, [{}] * len(manifest['pages']), guard=lambda: None)
    assert result['status'] == 'SKIPPED_REQUIREMENTS_WINDOW'
    assert provider.calls == 0


def test_missing_original_cannot_be_replaced_by_saved_answer(tmp_path):
    runtime, provider, rid = setup(tmp_path)
    manifest = runtime.db.get_run(rid)['metadata']['workset_manifest']
    reports = [{'answer': 'plausible answer', 'workset_context': {'selected_evidence_ids': ['foreign']}}
               for _ in manifest['pages']]
    with pytest.raises(ValueError, match='outside the run'):
        synthesize_worksets(runtime, rid, manifest, reports, guard=lambda: None)
    assert provider.calls == 0


def test_evidence_overflow_is_explicit_and_spends_no_calls(tmp_path):
    from paperpilot_agentic.durable_reader import persist_reader_evidence
    runtime, provider, rid = setup(tmp_path)
    manifest = runtime.db.get_run(rid)['metadata']['workset_manifest']
    rows = persist_reader_evidence(runtime.db, rid, [{'evidence_id': 'large',
        'source_uri': 'synthetic://long', 'content': '原文' * 5000}])
    reports = [{'workset_context': {'selected_evidence_ids': [rows[0]['evidence_id']]}}
               for _ in manifest['pages']]
    result = synthesize_worksets(runtime, rid, manifest, reports, guard=lambda: None)
    assert result['status'] == 'SKIPPED_EVIDENCE_WINDOW'
    assert provider.calls == 0
