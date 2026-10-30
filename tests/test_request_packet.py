from copy import deepcopy
import json

from paperpilot_agentic.request_packet import compact_payload, restore_value_references


def payload(*, audit=False):
    cells = {'Gate': '✓', 'Dataset': 'Validation-A', 'Score': '3.25', 'Offset': '8px'}
    content = 'Table III. Conditions and precision: ' + json.dumps(cells, ensure_ascii=False)
    value = {'question': 'Only compare mechanisms; use Table III for a numeric result.',
        'original_question': 'Only compare mechanisms; use Table III for a numeric result.',
        'request_contract': {'original_question': 'Only compare mechanisms; use Table III for a numeric result.',
                             'table_requirements': [{'paper_sha256': 'paper-A', 'label': 'Table III'}]},
        'request_history': [{'role': 'initial_request', 'user_request': 'Keep the 8GB limit.'},
                            {'role': 'current_request', 'user_request': 'Only compare mechanisms; use Table III for a numeric result.'}],
        'evidence': [{'evidence_id': 'e1', 'source_uri': 'paper://paper-A/region', 'page': 3, 'content': content}],
        'table_rows': [{'evidence_id': 'e1', 'row_id': 'host-owned-row', 'checked_columns': ['Gate'], 'cells': cells}],
        'source_inventory_navigation_only': [
            {'paper_sha256': 'paper-A', 'label': 'Table III', 'status': 'PRESENT', 'page': 3},
            {'paper_sha256': 'paper-A', 'label': 'Table IX', 'status': 'PRESENT', 'page': 8},
            {'paper_sha256': 'excluded-paper', 'label': 'Table III', 'status': 'PRESENT', 'page': 3}]}
    if audit:
        value['sources'] = [{'evidence_id': 'e1', 'spans': {'s0': content}}]
        value.pop('evidence')
    return value


def test_exact_sources_conditions_and_history_survive_value_deduplication():
    for audit in (False, True):
        original = payload(audit=audit)
        protected = deepcopy(original)
        compact = compact_payload(original, paper_scope={'paper-A'})
        restored = restore_value_references(compact)
        assert original == protected
        assert restored['table_rows'] == original['table_rows']
        assert restored['request_history'] == original['request_history']
        assert restored['request_contract'] == original['request_contract']
        field = 'sources' if audit else 'evidence'
        assert compact[field] == original[field]
        assert len(compact['source_inventory_navigation_only']) == 1
        assert compact['source_inventory_scope']['absence_is_not_established']
        assert 'cells_source_ref' in compact['table_rows'][0]
        assert compact['request_history'][0]['user_request'] == 'Keep the 8GB limit.'


def test_nonidentical_or_ambiguous_cell_text_keeps_full_catalog():
    for content in ('Only Score 3.25 is visible.', payload()['evidence'][0]['content'] * 2):
        original = payload()
        original['evidence'][0]['content'] = content
        compact = compact_payload(original)
        assert compact['table_rows'] == original['table_rows']


def test_mechanism_only_does_not_carry_old_ablation_navigation():
    original = payload()
    original['request_contract']['mechanism_only'] = True
    compact = compact_payload(original)
    assert compact['source_inventory_navigation_only'] == []
    assert compact['evidence'] == original['evidence']
    assert compact['table_rows'][0]['checked_columns'] == ['Gate']


def test_missing_table_location_is_retained_without_paper_absence_upgrade():
    original = payload()
    original['source_inventory_navigation_only'][0]['status'] = 'NOT_LOCATED'
    compact = compact_payload(original)
    assert compact['source_inventory_navigation_only'][0]['status'] == 'NOT_LOCATED'


def test_resolved_question_cannot_replace_actual_question_or_revoked_requirement():
    original = payload()
    original['question'] = 'Retrieval rewrite'
    original['request_history'].append({'role': 'current_request', 'user_request': 'Cancel the 8GB limit.'})
    compact = compact_payload(original)
    assert compact['original_question'] == original['original_question']
    assert restore_value_references(compact)['request_history'] == original['request_history']


def test_duplicate_audit_passages_share_text_without_merging_source_ownership():
    original = payload(audit=True)
    text = 'A measured value, with units and qualification. ' * 80
    original['sources'] = [
        {'evidence_id': 'e1', 'spans': {'s0': text, 's1': 'Same text with a DIFFERENT condition.'}},
        {'evidence_id': 'e2', 'spans': {'s2': text, 's3': 'Same text with a different condition.'}}]
    allowed = ['s0', 's1']
    original['facts'] = [
        {'id': 'f1', 'evidence_ids': ['e1'], 'allowed_span_ids': allowed},
        {'id': 'f2', 'evidence_ids': ['e1'], 'allowed_span_ids': allowed},
        {'id': 'f3', 'evidence_ids': ['e2'], 'allowed_span_ids': ['s2', 's3']}]
    compact = compact_payload(original)
    assert len(json.dumps(compact)) < len(json.dumps(original)) - len(text)
    restored = restore_value_references(compact)
    assert restored['sources'] == original['sources']
    assert restored['facts'] == original['facts']
    assert restored['request_history'] == original['request_history']
    assert restored['request_contract'] == original['request_contract']
    assert len(compact['sources']) == 2
    assert compact['sources'][0]['spans']['s1'] != compact['sources'][1]['spans']['s3']
    assert 's2' not in restored['facts'][0]['allowed_span_ids']


def test_cell_offsets_resolve_after_duplicate_span_text_is_restored():
    original = payload(audit=True)
    original['sources'].append({'evidence_id': 'e2', 'spans': {'s1': original['sources'][0]['spans']['s0']}})
    compact = compact_payload(original)
    assert isinstance(compact['sources'][0]['spans']['s0'], dict)
    restored = restore_value_references(compact)
    assert restored['table_rows'] == original['table_rows']
    assert restored['sources'] == original['sources']


def test_large_shared_allowlists_preserve_order_and_exact_membership():
    original = payload(audit=True)
    ids = [f's{i}' for i in range(40)]
    original['facts'] = [{'id': f'f{i}', 'evidence_ids': ['e1'], 'allowed_span_ids': ids}
                         for i in range(3)]
    original['facts'].append({'id': 'other', 'evidence_ids': ['e2'], 'allowed_span_ids': ['s99']})
    compact = compact_payload(original)
    assert len(compact['allowed_span_sets']) == 1
    assert 'allowed_span_ids' not in compact['facts'][0]
    assert compact['facts'][-1]['allowed_span_ids'] == ['s99']
    assert restore_value_references(compact)['facts'] == original['facts']


def test_full_audit_pipeline_keeps_uncited_sources_and_rejects_cross_owner_citation():
    from paperpilot_agentic.answer_audit import answer_then_audit
    from paperpilot_agentic.providers import LLMResponse
    text = 'A local encoder processes both temporal images with tied weights and retains spatial resolution.'
    sources = [{'evidence_id': 'e1', 'content': text}, {'evidence_id': 'e2', 'content': text}]
    draft = {'facts': [{'id': 'f1', 'text': text, 'kind': 'SOURCE_FACT', 'evidence_ids': ['e1']}],
             'direct_answer_id': 'f1', 'limitations': []}
    for sid, valid in [('s0', True), ('s1', False)]:
        seen = []
        class Model:
            def complete(self, messages, **kwargs):
                kwargs['before_attempt']()
                seen.append(messages)
                audit = {'assessments': [{'fact_id': 'f1', 'verdict': 'SUPPORTED',
                          'span_ids': [sid], 'relevance': 'DIRECT'}],
                         'coverage': 'COMPLETE', 'missing_aspects': []}
                return LLMResponse(json.dumps(draft if len(seen) == 1 else audit), finish_reason='stop')
        report = answer_then_audit('Describe the encoder.', sources, Model(), allow_repair=False)
        actual = json.loads(seen[1][-1]['content'])
        assert len(actual['sources']) == 2
        assert 'source_texts' not in actual, 'Shared small packets use direct original spans, not text indirection'
        restored = restore_value_references(actual)
        assert restored['facts'][0]['allowed_span_ids'] == ['s0']
        assert restored['sources'][1]['spans']['s1'] == text
        assert (report['audit']['assessments'][0]['verdict'] == 'SUPPORTED') == valid
