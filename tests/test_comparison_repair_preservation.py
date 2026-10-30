"""Protect comparison work and keep audit diagnostics out of user prose."""
import copy
import json

import pytest

from paperpilot_agentic.answer_contracts import audited_presentation
from paperpilot_agentic.answer_pipeline import (
    AnswerOptions, initialize, repair_isolated_facts,
)
from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.table_projection import project_table_facts
from test_table_fact_projection import rows


def test_local_repair_does_not_accept_value_in_place_of_rejected_table_inference():
    catalog = rows()
    selection = {'id': 'measured', 'kind': 'TABLE_FACT', 'operation': 'value',
                 'row_refs': [{'evidence_id': 't', 'row_id': 'r2', 'metrics': ['F1 (%)']}]}
    projected, _ = project_table_facts({'facts': [selection]}, catalog, chinese=True)
    original = {'facts': projected['facts'] + [
        {'id': 'comparison', 'kind': 'INFERENCE', 'text': 'Beta is best with F1 63.40%.',
         'evidence_ids': ['t'], 'depends_on': ['measured']}],
        'direct_answer_id': 'measured', 'limitations': []}
    source = {'evidence_id': 't', 'content': 'Table IV reports baseline, Alpha and Beta on Delta.',
              'source_uri': 'paper://p/table', 'page': 1}
    seen = []
    patch = {'facts': [{**selection, 'id': 'comparison'}], 'limitations': []}
    class Model:
        def ensure_capacity(self, requests):
            pass
        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            seen.append(messages)
            assert len(seen) == 1, 'Degraded patch must be rejected before another audit'
            return LLMResponse(json.dumps(patch), finish_reason='stop')
    model = Model()
    context = initialize('Compare Table IV F1 numeric results for baseline, Alpha and Beta.', [source], model,
                         AnswerOptions(), None)
    isolation = {'quarantined_fact_ids': ['comparison'],
                 'errors': [{'fact_id': 'comparison',
                            'error': 'Catalogued table numbers require TABLE_FACT projection'}]}
    context.report.update(table_rows=catalog, pre_isolation_draft=original,
                          draft_isolation=isolation, answer='Already audited measured fact.',
                          answer_scope='PARTIAL')
    before = copy.deepcopy(original)
    repair_isolated_facts(context, {'facts': projected['facts']},
                          lambda *args, **kwargs: pytest.fail('Unexpected audit'))
    assert isolation['repair_status'] == 'FAILED_VALIDATION'
    assert 'comparison' in isolation['repair_error']
    assert len(seen) == 1  # Rejected before wasting a follow-up audit call.
    assert context.report['answer'] == 'Already audited measured fact.'
    assert original == before


def test_extrema_display_names_the_actual_comparison_set_and_metric():
    selection = {'id': 'best', 'kind': 'TABLE_FACT', 'operation': 'max',
                 'row_refs': [{'evidence_id': 't', 'row_id': r['row_id'],
                               'metrics': ['F1 (%)']} for r in rows()]}
    projected, _ = project_table_facts({'facts': [selection]}, rows(), chinese=True)
    text = projected['facts'][0]['text']
    assert 'Alpha' in text and 'Beta' in text and '无勾选模块' in text
    assert 'F1 (%)' in text and '63.40' in text and '最高' in text


def test_internal_missing_aspect_is_not_user_answer_but_gap_stays_partial():
    draft = {'facts': [{'id': 'f', 'kind': 'SOURCE_FACT', 'text': 'Alpha 按时间组织特征。',
                       'evidence_ids': ['e']}], 'direct_answer_id': 'f', 'limitations': []}
    raw = 'The candidate answer fails to compare the best configuration. Missing inference.'
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT'}],
             'coverage': 'PARTIAL', 'missing_aspects': [raw],
             'requirement_assessments': [{'requirement': 'Compare configurations',
                                         'fact_ids': [], 'gap': raw}]}
    before = copy.deepcopy(audit)
    p = audited_presentation(draft, audit, question='说明机制，并比较最佳配置。')
    assert raw not in p['answer_units'] and all(raw not in x for x in p['limitations'])
    assert p['answer_scope'] == 'PARTIAL'
    assert p['limitations'] and all('未完成' in x for x in p['limitations'])
    assert raw in p['internal_gap_diagnostics']['missing_aspects']
    assert audit == before


def test_source_bound_request_quote_can_name_user_gap_without_rewriting_science():
    draft = {'facts': [], 'limitations': [], 'direct_answer_id': None}
    audit = {'assessments': [], 'coverage': 'PARTIAL', 'missing_aspects': ['Missing comparison'],
             'requirement_assessments': [{'requirement': 'Compare configurations', 'fact_ids': [],
                                         'gap': 'Missing comparison', 'request_quote': '比较最佳配置'}]}
    p = audited_presentation(draft, audit, question='说明机制，并比较最佳配置。')
    assert p['limitations'] == ['本轮未完成：比较最佳配置。']
    assert p['answer_scope'] == 'NO_SUPPORTED_ANSWER'


def test_request_quote_outside_question_is_never_displayed_as_a_user_requirement():
    draft = {'facts': [], 'limitations': [], 'direct_answer_id': None}
    audit = {'assessments': [], 'coverage': 'PARTIAL', 'missing_aspects': ['Missing experiment'],
             'requirement_assessments': [{'requirement': 'Train', 'fact_ids': [],
                                         'gap': 'Missing experiment', 'request_quote': '启动训练'}]}
    p = audited_presentation(draft, audit, question='只说明机制。')
    assert all('启动训练' not in x for x in p['limitations'])
    assert p['limitations']


@pytest.mark.parametrize('operation', ['max', 'min'])
def test_single_row_cannot_disguise_a_value_as_a_ranking(operation):
    selection = {'id': 'comparison', 'kind': 'TABLE_FACT', 'operation': operation,
                 'row_refs': [{'evidence_id': 't', 'row_id': 'r2', 'metrics': ['F1 (%)']}]}
    with pytest.raises(ValueError, match='at least two'):
        project_table_facts({'facts': [selection]}, rows(), chinese=True)


@pytest.mark.parametrize('kind', ['plain_inference', 'host_ranking'])
def test_legitimate_comparison_patch_is_still_admissible_and_not_auto_approved(kind):
    from paperpilot_agentic.answer_isolation import validate_patch_operations
    original = {'facts': [{'id': 'comparison', 'kind': 'INFERENCE',
                          'text': 'Beta has the best F1 63.40%.', 'evidence_ids': ['t']}]}
    isolation = {'errors': [{'fact_id': 'comparison',
                            'error': 'Catalogued table numbers require TABLE_FACT projection'}]}
    if kind == 'plain_inference':
        fact = {'id': 'comparison', 'kind': 'INFERENCE', 'text': 'Beta has the best tested F1.',
                'evidence_ids': ['t'], 'depends_on': ['measured']}
    else:
        fact = {'id': 'comparison', 'kind': 'TABLE_FACT', 'operation': 'max',
                'row_refs': [{'evidence_id': 't', 'row_id': r['row_id'], 'metrics': ['F1 (%)']}
                             for r in rows()]}
    patch = {'facts': [fact], 'limitations': []}
    before = copy.deepcopy(patch)
    assert validate_patch_operations(original, patch, isolation) is None
    assert patch == before  # This is a structural guard, not an audit verdict.


def test_scope_boundary_survives_while_internal_criticism_is_separated():
    draft = {'facts': [{'id': 'f', 'kind': 'SOURCE_FACT', 'text': 'Alpha 按时间组织特征。',
                       'evidence_ids': ['e']}], 'direct_answer_id': 'f',
             'limitations': ['当前证据无法确定其他数据集上的表现。']}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT'}],
             'coverage': 'COMPLETE', 'missing_aspects': [],
             'requirement_assessments': [{'requirement': '说明机制', 'fact_ids': ['f'], 'gap': '',
                                         'request_quote': '说明机制'}],
             'limitation_assessments': [{'limitation_index': 0, 'role': 'SCOPE_BOUNDARY',
                                         'reason': 'Limits untested extrapolation.'}]}
    p = audited_presentation(draft, audit, question='说明机制。')
    assert p['limitations'] == ['当前证据无法确定其他数据集上的表现。']
    assert p['answer_scope'] == 'MODEL_REVIEWED'


def test_untranslated_boundary_is_disclosed_and_does_not_become_complete():
    draft = {'facts': [{'id': 'f', 'kind': 'SOURCE_FACT', 'text': 'Alpha 按时间组织特征。',
                       'evidence_ids': ['e']}], 'direct_answer_id': 'f',
             'limitations': ['Cannot determine from the supplied passages whether other datasets behave the same.']}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT'}],
             'coverage': 'COMPLETE', 'missing_aspects': [],
             'requirement_assessments': [{'requirement': '说明机制', 'fact_ids': ['f'], 'gap': ''}],
             'limitation_assessments': [{'limitation_index': 0, 'role': 'SCOPE_BOUNDARY',
                                         'reason': 'Limits untested extrapolation.'}]}
    p = audited_presentation(draft, audit, question='说明机制。')
    assert p['answer_scope'] == 'PARTIAL'
    assert all('Cannot determine' not in value for value in p['limitations'])
    assert p['internal_gap_diagnostics']['unrendered_scope_boundaries']
    assert '边界' in p['limitations'][0]


@pytest.mark.parametrize('quote', ['', '启动训练', ['说明机制'], '说明机制' * 100])
def test_supplied_request_quote_must_belong_to_user_request(quote):
    from paperpilot_agentic.answer_gap_display import validate_request_quotes
    audit = {'requirement_assessments': [{'request_quote': quote}]}
    with pytest.raises(ValueError, match='original user request'):
        validate_request_quotes(audit, '说明机制。')


def test_formal_pipeline_displays_gap_notice_and_preserves_raw_audit():
    from paperpilot_agentic.answer_audit import answer_then_audit
    raw = 'The candidate answer fails to compare. Missing inference.'
    class Model:
        calls = 0
        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            self.calls += 1
            if self.calls == 1:
                value = {'facts': [{'id': 'f', 'kind': 'SOURCE_FACT',
                          'text': 'Alpha 按时间组织特征。', 'evidence_ids': ['e']}],
                         'direct_answer_id': 'f', 'limitations': []}
            else:
                value = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED',
                          'relevance': 'DIRECT', 'span_ids': ['s0']}], 'coverage': 'PARTIAL',
                         'missing_aspects': [raw], 'requirement_assessments': [
                         {'requirement': '比较配置', 'fact_ids': [], 'gap': raw,
                          'request_quote': '比较配置'}]}
            return LLMResponse(json.dumps(value), finish_reason='stop')
    model = Model()
    report = answer_then_audit('说明机制，并比较配置。', [{'evidence_id': 'e',
        'content': 'Alpha organizes features in temporal order.'}], model, allow_repair=False)
    assert model.calls == 2
    assert report['answer_scope'] == 'PARTIAL'
    assert report['limitations'] == ['本轮未完成：比较配置。']
    assert report['audit']['missing_aspects'] == [raw]
    assert report['internal_gap_diagnostics']['missing_aspects'] == [raw]
    assert raw not in '\n'.join(report['answer_units'])


@pytest.mark.parametrize('numeric_conditions', [False, True])
def test_host_ranking_patch_reaches_reaudit_and_final_comparison_without_free_values(numeric_conditions):
    from paperpilot_agentic.answer_audit import answer_then_audit
    from test_reviewed_sources import artifact
    item = artifact()
    item.table_rows = [{'row_label': '', 'checked_columns': modules,
                        'cells': {'F1 (%)': value}}
                      for modules, value in [([], '61.20'), (['Alpha'], '62.30'), (['Beta'], '63.40')]]
    if numeric_conditions:
        for row in item.table_rows:
            row['cells'].update({'batch_size': '4', 'crop_size': '256'})
    evidence = {'evidence_id': 't', 'source_uri': 'paper://p/layout/o', 'page': 2,
                'content': 'Table I. Baseline F1 (%) is 61.20. Alpha F1 (%) is 62.30. Beta F1 (%) is 63.40.',
                'provenance': {'structured_extraction': item.model_dump(mode='json')}}
    if numeric_conditions:
        evidence['content'] += ' All rows use batch_size=4 and crop_size=256.'
    class Model:
        calls = 0
        def ensure_capacity(self, requests):
            pass
        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            self.calls += 1
            payload = json.loads(messages[-1]['content'])
            if self.calls == 1:
                refs = [{'evidence_id': row['evidence_id'], 'row_id': row['row_id'],
                         'metrics': ['F1 (%)']} for row in payload['table_rows']]
                self.refs = refs
                value = {'facts': [
                    {'id': 'baseline', 'kind': 'TABLE_FACT', 'operation': 'value', 'row_refs': [refs[0]]},
                    {'id': 'candidate', 'kind': 'TABLE_FACT', 'operation': 'value', 'row_refs': [refs[2]]},
                    {'id': 'best', 'kind': 'INFERENCE', 'text': 'Beta 的 F1 63.40% 最高。',
                     'evidence_ids': ['t'], 'depends_on': ['candidate']}],
                    'direct_answer_id': 'baseline', 'limitations': []}
            elif self.calls == 3:
                assert payload['repair_fact_ids'] == ['best']
                value = {'facts': [{'id': 'best', 'kind': 'TABLE_FACT', 'operation': 'max',
                                   'row_refs': self.refs}], 'limitations': []}
            else:
                value = {'assessments': [{'fact_id': f['id'], 'verdict': 'SUPPORTED',
                         'relevance': 'DIRECT', 'span_ids': f['allowed_span_ids']}
                         for f in payload['facts']],
                         'coverage': 'PARTIAL' if self.calls == 2 else 'COMPLETE',
                         'missing_aspects': ['Missing comparison'] if self.calls == 2 else []}
            return LLMResponse(json.dumps(value), finish_reason='stop')
    model = Model()
    report = answer_then_audit('仅依据 Table I 比较配置的 F1 最大值。', [evidence], model)
    assert model.calls == 4
    assert report['draft_isolation']['repair_status'] == 'PATCH_AUDITED'
    assert report['raw_local_repair']['facts'][0]['operation'] == 'max'
    assert 'text' not in report['raw_local_repair']['facts'][0]
    assert 'Beta' in report['answer'] and 'Alpha' in report['answer'] and '最高' in report['answer']
    assert '63.40' in report['answer']
    assert report['answer_scope'] == 'MODEL_REVIEWED'
    assert report['semantic_correctness'] == 'MODEL_AUDIT_ONLY_NOT_INDEPENDENT_GROUND_TRUTH'
    if numeric_conditions:
        assert 'batch_size=4' in report['answer'] and 'crop_size=256' in report['answer']
