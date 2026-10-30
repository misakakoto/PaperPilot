"""Cross-group protocol tests; not a factual quality benchmark."""
import json
from copy import deepcopy

import pytest

from paperpilot_agentic.durable_reader import ReaderAlreadyStarted, persist_reader_evidence
from paperpilot_agentic.workset_synthesis import synthesize_worksets
from test_workset_synthesis import setup


def fixture(tmp_path, calls=30, count=3, length=10000):
    runtime, provider, rid = setup(tmp_path, calls=calls)
    manifest = runtime.db.get_run(rid)['metadata']['workset_manifest']
    sources = persist_reader_evidence(runtime.db, rid, [
        {'evidence_id': f'raw{i}', 'source_uri': f'synthetic://paper{i}',
         'content': 'A source fact. ' + str(i) * length} for i in range(count)])
    reports = [{'answer': 'POISON_PRIOR_ANSWER', 'workset_context': {
        'selected_evidence_ids': [row['evidence_id'] for row in sources]}}
        for _ in manifest['pages']]
    return runtime, provider, rid, manifest, reports


def test_pairs_reload_originals_and_replay_without_extra_requests(tmp_path):
    runtime, provider, rid, manifest, reports = fixture(tmp_path)
    messages = []
    complete = provider.complete
    def capture(value, **kwargs):
        messages.append(deepcopy(value))
        return complete(value, **kwargs)
    provider.complete = capture
    result = synthesize_worksets(runtime, rid, manifest, reports, guard=lambda: None)
    cross = result['cross_group_review']
    assert cross['total_pairs'] == cross['reviewed_pairs'] == 2
    assert cross['unreviewed_pairs'] == 0
    assert result['cross_group_verified'] is False
    assert result['report']['answer_scope'] == 'PARTIAL'
    assert '跨组原文对照' in result['report']['answer']
    assert 'POISON_PRIOR_ANSWER' not in json.dumps(messages)
    assert provider.calls == 8  # Two groups and two pairs, Writer + Auditor each.
    assert synthesize_worksets(runtime, rid, manifest, reports, guard=lambda: None) == result
    assert provider.calls == 8

    checkpoint = deepcopy(runtime.db.get_run(rid)['metadata']['workset_cross_group_0'])
    checkpoint['state'] = 'STARTED'
    checkpoint.pop('report')
    runtime.db.update_run_metadata(rid, {'workset_cross_group_0': checkpoint})
    with pytest.raises(ReaderAlreadyStarted):
        synthesize_worksets(runtime, rid, manifest, reports, guard=lambda: None)
    assert provider.calls == 8  # Unknown provider outcome is never blindly retried.


def test_pair_budget_retains_group_answers_and_reports_missing_comparisons(tmp_path):
    runtime, provider, rid, manifest, reports = fixture(tmp_path, calls=6)
    result = synthesize_worksets(runtime, rid, manifest, reports, guard=lambda: None)
    cross = result['cross_group_review']
    assert cross['reviewed_pairs'] == 1 and cross['unreviewed_pairs'] == 1
    assert cross['stop_reason'] == 'CALL_BUDGET'
    assert result['status'] == 'REVIEWED_GROUPS'
    assert provider.calls == 6


def test_pairs_too_large_are_not_clipped_or_called(tmp_path):
    runtime, provider, rid, manifest, reports = fixture(tmp_path, length=13000)
    result = synthesize_worksets(runtime, rid, manifest, reports, guard=lambda: None)
    cross = result['cross_group_review']
    assert cross['reviewed_pairs'] == 0 and cross['unreviewed_pairs'] == 3
    assert len(cross['oversized_pairs']) == 3
    assert provider.calls == 6  # Only three original groups were reviewed.


def test_pair_safety_ceiling_preserves_true_denominator(tmp_path):
    runtime, provider, rid, manifest, reports = fixture(tmp_path, calls=100, count=9)
    result = synthesize_worksets(runtime, rid, manifest, reports, guard=lambda: None)
    cross = result['cross_group_review']
    assert cross['total_pairs'] == 32  # 36 pairs minus four within-group pairs.
    assert cross['reviewed_pairs'] == 12
    assert cross['unreviewed_pairs'] == 20
    assert cross['stop_reason'] == 'PAIR_LIMIT'
    assert provider.calls == 34  # Five groups plus twelve pairs.


def test_runtime_publishes_pair_findings_without_claiming_verified_completion(tmp_path):
    runtime, provider, rid = setup(tmp_path, calls=60)
    seen = []
    def search(*args, **kwargs):
        page = len(seen)
        seen.append(page)
        return [{'stable_id': f'chunk{page}-{i}', 'paper_sha256': 'sha',
                 'source_uri': f'paper://sha/chunk{page}-{i}',
                 'content': 'A source fact. ' + str(i) * 8000} for i in range(2)]
    runtime.search_adapter.search = search
    result = runtime.resume(rid)
    assert result['run']['status'] == 'COMPLETED', result['run'].get('error')
    cross = result['run']['metadata']['workset_synthesis']['cross_group_review']
    assert cross['reviewed_pairs'] > 0
    assert '跨组原文对照' in result['final_answer']['answer']
    assert any('非独立事实验证' in text for text in result['final_answer']['limitations'])
    assert result['run']['research_outcome'] == 'PARTIAL'
    calls = provider.calls
    runtime.resume(rid)
    assert provider.calls == calls


def test_supported_pair_answer_is_not_discarded_when_local_groups_abstain(tmp_path, monkeypatch):
    from paperpilot_agentic import workset_synthesis as module
    runtime, _, rid, manifest, reports = fixture(tmp_path)
    original = module.run_durable_reader
    def reader(*args, **kwargs):
        report = original(*args, **kwargs)
        if kwargs['checkpoint_key'].startswith('workset_synthesis_group_'):
            return {**report, 'answer': '', 'direct_answer_supported': False}
        return report
    monkeypatch.setattr(module, 'run_durable_reader', reader)
    result = synthesize_worksets(runtime, rid, manifest, reports, guard=lambda: None)
    assert result['status'] == 'REVIEWED_GROUPS'
    assert result['report']['direct_answer_supported'] is True
    assert '跨组原文对照' in result['report']['answer']
    assert result['cross_group_verified'] is False
