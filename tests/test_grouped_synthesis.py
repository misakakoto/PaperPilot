import json
from copy import deepcopy

from paperpilot_agentic.durable_reader import persist_reader_evidence
from paperpilot_agentic.workset_synthesis import evidence_groups, synthesize_worksets
from test_workset_synthesis import setup


def large_sources(runtime, rid):
    return persist_reader_evidence(runtime.db, rid, [
        {'evidence_id': f'raw{i}', 'source_uri': f'synthetic://paper{i}',
         'content': 'A source fact. ' + f'Paper {i} background. ' * 600} for i in range(3)])


def fixture(tmp_path):
    runtime, provider, rid = setup(tmp_path)
    manifest = runtime.db.get_run(rid)['metadata']['workset_manifest']
    manifest.pop('cross_group_review', None)  # Exercise version-3 compatibility.
    sources = large_sources(runtime, rid)
    reports = [{'answer': 'POISON_NOT_EVIDENCE', 'workset_context': {
        'selected_evidence_ids': [row['evidence_id'] for row in sources]}} for _ in manifest['pages']]
    return runtime, provider, rid, manifest, reports, sources


def test_groups_preserve_complete_records_and_disclose_indivisible_overflow():
    rows = [{'evidence_id': str(i), 'content': '段落' * 30} for i in range(100)]
    before = deepcopy(rows)
    groups = evidence_groups(rows, 800)
    assert [row for group in groups for row in group] == rows
    assert all(len(json.dumps(g, ensure_ascii=False).encode()) <= 800 for g in groups)
    assert evidence_groups(rows + [{'content': 'x' * 1000}], 800) is None
    assert before == rows


def test_grouped_review_is_replayable_and_never_claims_global_coverage(tmp_path):
    runtime, provider, rid, manifest, reports, sources = fixture(tmp_path)
    messages = []
    original = provider.complete
    def capture(value, **kwargs):
        messages.append(deepcopy(value))
        return original(value, **kwargs)
    provider.complete = capture
    result = synthesize_worksets(runtime, rid, manifest, reports, guard=lambda: None)
    assert result['status'] == 'REVIEWED_GROUPS', result
    assert result['group_count'] > 1
    assert result['cross_group_verified'] is False
    assert result['unreviewed_evidence_ids'] == []
    assert result['report']['answer_scope'] == 'PARTIAL'
    assert 'POISON_NOT_EVIDENCE' not in json.dumps(messages)
    assert result['evidence_ids'] == [row['evidence_id'] for row in sources]
    calls = provider.calls
    again = synthesize_worksets(runtime, rid, manifest, reports, guard=lambda: None)
    assert again == result and provider.calls == calls


def test_group_budget_preserves_partial_results_and_missing_original_ids(tmp_path):
    runtime, provider, rid, manifest, reports, _ = fixture(tmp_path)
    with runtime.db.transaction() as conn:
        limits = runtime.db.get_run(rid)['budget']
        limits['max_llm_calls'] = 2
        conn.execute('UPDATE runs SET budget_json=? WHERE run_id=?', (json.dumps(limits), rid))
    result = synthesize_worksets(runtime, rid, manifest, reports, guard=lambda: None)
    assert result['status'] == 'REVIEWED_GROUPS'
    assert result['stop_reason'] == 'CALL_BUDGET'
    assert result['unreviewed_evidence_ids']
    assert provider.calls == 2


def test_old_manifest_retains_previous_no_spend_overflow_behavior(tmp_path):
    runtime, provider, rid, manifest, reports, _ = fixture(tmp_path)
    manifest.pop('group_large_evidence')
    assert synthesize_worksets(runtime, rid, manifest, reports,
        guard=lambda: None)['status'] == 'SKIPPED_EVIDENCE_WINDOW'
    assert provider.calls == 0


def test_runtime_publishes_grouped_answer_with_explicit_cross_group_limit(tmp_path):
    runtime, provider, rid = setup(tmp_path)
    counter = []
    def search(*args, **kwargs):
        index = len(counter)
        counter.append(index)
        return [{'stable_id': f'chunk{index}', 'paper_sha256': 'sha',
                 'source_uri': f'paper://sha/chunk{index}',
                 'content': 'A source fact. ' + 'Detailed background. ' * 650}]
    runtime.search_adapter.search = search
    result = runtime.resume(rid)
    assert result['run']['status'] == 'COMPLETED', result['run'].get('error')
    synthesis = result['run']['metadata']['workset_synthesis']
    assert synthesis['status'] == 'REVIEWED_GROUPS'
    assert '原文组' in result['final_answer']['answer']
    assert any('不同组之间的冲突' in text for text in result['final_answer']['limitations'])
    assert result['run']['research_outcome'] == 'PARTIAL'
    calls = provider.calls
    runtime.resume(rid)
    assert provider.calls == calls
