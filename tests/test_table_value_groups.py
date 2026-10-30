import copy
import pytest

from paperpilot_agentic.table_projection import project_table_facts_locally
from test_table_fact_projection import rows


def group_draft():
    return {'facts': [
        {'id': 'measurements', 'kind': 'TABLE_FACT', 'operation': 'value',
         'row_refs': [{'evidence_id': 't', 'row_id': row['row_id'], 'metrics': ['F1 (%)']}
                      for row in rows()]},
        {'id': 'comparison', 'kind': 'INFERENCE', 'text': 'Beta has the highest tested F1.',
         'evidence_ids': ['t'], 'depends_on': ['measurements']}],
        'direct_answer_id': 'comparison', 'limitations': []}


def test_initial_group_expands_into_independent_configuration_facts_and_premises():
    from paperpilot_agentic.table_projection import expand_table_value_groups
    draft = group_draft()
    before = copy.deepcopy(draft)
    normalized, trace = expand_table_value_groups(draft, rows())
    projected, _, errors = project_table_facts_locally(normalized, rows())
    assert not errors
    source_facts = [f for f in projected['facts'] if f['kind'] == 'SOURCE_FACT']
    assert len(source_facts) == 3
    assert all(len(f['row_refs']) == 1 for f in source_facts)
    assert [f['row_refs'][0]['row_id'] for f in source_facts] == ['r0', 'r1', 'r2']
    assert projected['facts'][-1]['depends_on'] == [f['id'] for f in source_facts]
    assert projected['direct_answer_id'] == 'comparison'
    assert trace[0]['raw_selection'] == before['facts'][0]
    assert draft == before


@pytest.mark.parametrize('defect', ['unknown', 'repeated', 'prose', 'collision', 'overflow'])
def test_invalid_group_is_not_partially_accepted_or_silently_truncated(defect):
    from paperpilot_agentic.table_projection import expand_table_value_groups
    draft = group_draft()
    if defect == 'unknown':
        draft['facts'][0]['row_refs'][1]['row_id'] = 'invented'
    elif defect == 'repeated':
        draft['facts'][0]['row_refs'][1] = copy.deepcopy(draft['facts'][0]['row_refs'][0])
    elif defect == 'prose':
        draft['facts'][0]['text'] = 'F1 is 999.9'
    elif defect == 'collision':
        draft['facts'][1]['id'] = 'measurements__value_1'
    else:
        draft['facts'].extend({'id': 'body'+str(i), 'kind': 'SOURCE_FACT',
                              'text': 'Body statement.', 'evidence_ids': ['body']}
                             for i in range(9))
    normalized, trace = expand_table_value_groups(draft, rows())
    assert normalized == draft
    assert trace[0]['status'] == 'REJECTED'
    _, _, errors = project_table_facts_locally(normalized, rows())
    assert 'measurements' in errors


def test_local_patch_projection_still_requires_exactly_one_configuration():
    from paperpilot_agentic.table_projection import project_table_facts
    with pytest.raises(ValueError, match='One value fact'):
        project_table_facts({'facts': [group_draft()['facts'][0]]}, rows())


def test_malformed_dependency_is_left_for_validation_without_losing_valid_measurements():
    from paperpilot_agentic.table_projection import expand_table_value_groups
    value = group_draft()
    value['facts'][1]['depends_on'] = [{'id': 'measurements'}]
    result, trace = expand_table_value_groups(value, rows())
    assert trace[0]['status'] == 'EXPANDED'
    assert result['facts'][-1]['depends_on'] == [{'id': 'measurements'}]
    assert len(result['facts']) == 4


def test_formal_pipeline_audits_each_configuration_and_expanded_comparison_premises():
    import json
    from paperpilot_agentic.answer_audit import answer_then_audit
    from paperpilot_agentic.providers import LLMResponse
    from test_reviewed_sources import artifact
    item = artifact()
    item.table_rows = [{'row_label': '', 'checked_columns': modules,
                        'cells': {'F1 (%)': value}}
                      for modules, value in [([], '61.20'), (['Alpha'], '62.30'), (['Beta'], '63.40')]]
    evidence = {'evidence_id': 't', 'source_uri': 'paper://p/layout/o', 'page': 2,
                'content': 'Table I. Baseline F1 (%) 61.20; Alpha 62.30; Beta 63.40.',
                'provenance': {'structured_extraction': item.model_dump(mode='json')}}
    class Model:
        calls = 0
        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            self.calls += 1
            payload = json.loads(messages[-1]['content'])
            if self.calls == 1:
                refs = [{'evidence_id': row['evidence_id'], 'row_id': row['row_id'],
                         'metrics': ['F1 (%)']} for row in payload['table_rows']]
                value = group_draft()
                value['facts'][0]['row_refs'] = refs
                value['facts'][1]['text'] = 'Beta 的 F1 在本表所测试配置中最高。'
            else:
                facts = payload['facts']
                assert len(facts) == 4
                assert len(facts[-1]['depends_on']) == 3
                assert all(f['kind'] == 'SOURCE_FACT' for f in facts[:3])
                value = {'assessments': [{'fact_id': f['id'], 'verdict': 'SUPPORTED',
                         'relevance': 'DIRECT', 'span_ids': f['allowed_span_ids']} for f in facts],
                         'coverage': 'COMPLETE', 'missing_aspects': []}
            return LLMResponse(json.dumps(value), finish_reason='stop')
    model = Model()
    report = answer_then_audit('给出 Table I 全部配置 F1，并比较最佳配置。', [evidence], model,
                              allow_repair=False)
    assert model.calls == 2
    assert all(value in report['answer'] for value in ('61.20', '62.30', '63.40'))
    assert '最高' in report['answer']
    assert len(report['raw_draft']['facts']) == 2
    assert report['table_value_group_expansion'][0]['status'] == 'EXPANDED'
    assert report['semantic_correctness'] == 'MODEL_AUDIT_ONLY_NOT_INDEPENDENT_GROUND_TRUTH'
