import json

import pytest

from paperpilot_agentic import request_scope
from paperpilot_agentic.numeric_scope import check_bindings
from test_experimental_scope import assessment


def test_mechanism_followup_cannot_acquire_tables_from_a_model_rewrite():
    original = 'Compare only the mechanisms of the two previously selected papers.'
    rewritten = 'Compare Alpha and Beta mechanisms using Table I and Table II.'
    query = request_scope.table_target_query(original, rewritten)
    assert query == original
    assert request_scope.automatic_table_targets(query, {'a', 'b'}, lambda text: [])[1] == []


@pytest.mark.parametrize('original', ['Explain the latter table.', '说明上一张表的数值。', 'Compare Alpha Table I.'])
def test_actual_table_followups_keep_resolved_ownership(original):
    rewritten = 'Explain Alpha Table I.'
    assert request_scope.table_target_query(original, rewritten) == rewritten


def test_explicit_table_number_cannot_be_changed_by_search_rewrite():
    original = 'Read Alpha Table IV.'
    assert request_scope.table_target_query(original, 'Read Alpha Table II.') == original


def test_reader_does_not_publish_wrong_table_clarification_from_a_saved_rewrite(tmp_path):
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from test_durable_reader import AnswerProvider

    class Search:
        def search(self, query, top_k=8, *, paper_sha256s=None):
            return [{'stable_id': 'chunk', 'content': 'A source fact.',
                     'paper_sha256': 'a', 'source_uri': 'paper://a/chunk', 'page': 1}]

    provider = AnswerProvider()
    runtime = ResearchAgentRuntime(db_path=tmp_path/'runtime.sqlite3', workspace=tmp_path,
        search_adapter=Search(), llm=provider, text_reader_protocol='test')
    meta = {'engine': 'answer_first_text', 'visual_policy': 'auto', 'paper_sha256s': ['a', 'b']}
    parent = runtime.enqueue(question='What fact?', metadata=meta)
    runtime.resume(parent['run']['run_id'])
    child = runtime.enqueue_follow_up(parent_run_id=parent['run']['run_id'],
        question='Compare only the mechanisms.', metadata={**meta, 'entry_resolution': {
            'question': 'Compare Alpha and Beta using Table I and Table II.',
            'use_prior_sources': True, 'needs_clarification': False, 'reason': 'Saved rewrite'}})
    result = runtime.resume(child['run']['run_id'])
    assert result['run']['metadata']['requested_table_targets'] == []
    assert result['run']['metadata']['rewrite_only_table_request_ignored'] is True
    assert result['final_answer']['answer'].startswith('A source fact.')
    assert result['run']['status'] == 'COMPLETED'


@pytest.mark.parametrize('text,rejected', [
    ('组合在表内已测配置中优于单机制，表明具有协同增益。', True),
    ('The tested combination demonstrates a synergistic effect.', True),
    ('组合性能最高，说明多机制协同优于单一机制。', True),
    ('组合数值更高，表明联合显著增强模型性能。', True),
    ('与基线相比，三种单机制均显著提升性能。', True),
    ('All tested single modules significantly improve performance.', True),
    ('与基线相比，三种单机制均提升了该表的 F1 数值。', False),
    ('表内数值更高，但不能证明显著提升。', False),
    ('组合在该表已测配置中优于单机制，但不能证明协同增益。', False),
    ('组合在该表数值更高，但不能证明统计显著性或协同。', False),
    ('组合在该表已测配置中 F1 最高，比基线高 0.5 个百分点。', False),
])
def test_ablation_ranking_does_not_prove_interaction(text, rejected):
    audit, diagnostics = assessment(text)
    assert bool(diagnostics['changed_fact_ids']) is rejected
    assert audit['assessments'][0]['verdict'] == ('UNCERTAIN' if rejected else 'SUPPORTED')


def table():
    data = {'column_headers': ['MFA', 'CDA', 'F1overall(%)', 'F1loc(%)'], 'rows': [
        {'row_label': 'baseline', 'checked_columns': [], 'cells': {'F1overall(%)': '70.0', 'F1loc(%)': '90.0'}},
        {'row_label': 'mfa', 'checked_columns': ['MFA'], 'cells': {'F1overall(%)': '71.0', 'F1loc(%)': '91.0'}},
        {'row_label': 'joint', 'checked_columns': ['MFA', 'CDA'], 'cells': {'F1overall(%)': '72.0', 'F1loc(%)': '92.0'}}]}
    return [{'evidence_id': 'table', 'content': 'Table I\nNamed-cell transcription:\n'+json.dumps(data)}]


@pytest.mark.parametrize('text,rejected', [
    ('Table I baseline overall F1 is 70.0%.', False),
    ('Table I overall F1 is 90.0%.', True),
    ('Table I F1 is 70.0%.', True),
    ('Table I 启用 MFA 不启用 CDA，overall F1 为 71.0%。', False),
    ('Table I 同时启用 MFA 和 CDA，overall F1 为 71.0%。', True),
    ('Table I 加入 MFA 后，overall F1 从 70.0% 变为 71.0%。', False),
    ('Table I 加入 MFA 后，overall F1 从 70.0% 变为 72.0%。', True),
])
def test_overall_f1_and_negated_components_preserve_wrong_column_and_row_rejection(text, rejected):
    fact = {'id': 'f', 'kind': 'SOURCE_FACT', 'text': text, 'evidence_ids': ['table']}
    checks = check_bindings(fact, table())['checks']
    errors = [c for c in checks if c['blocking'] and c['status'] != 'MATCH']
    assert bool(errors) is rejected, checks
