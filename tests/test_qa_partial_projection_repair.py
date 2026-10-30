from types import SimpleNamespace
import pytest

from paperpilot_agentic.answer_pipeline import AnswerOptions, initialize, generate_draft
from paperpilot_agentic.answer_contracts import conservative_audit_contract
from paperpilot_agentic.read_packet import build_read_packet
from paperpilot_agentic.answer_layout import host_requested_layout
from test_reviewed_sources import artifact


def test_real_writer_seam_keeps_valid_projection_when_another_fact_writes_numbers():
    item = artifact()
    evidence = {'evidence_id': 'table', 'source_uri': 'paper://p/layout/o', 'page': 2,
                'content': 'Table I F1 (%) = 62.30', 'object_type': 'table',
                'provenance': {'structured_extraction': item.model_dump(mode='json')}}
    ctx = initialize('Give Table I F1.', [evidence], SimpleNamespace(),
                     AnswerOptions(allow_repair=False), None)
    row = ctx.report['table_rows'][0]
    good = {'id': 'measured', 'kind': 'TABLE_FACT', 'operation': 'value',
            'row_refs': [{'evidence_id': row['evidence_id'], 'row_id': row['row_id'], 'metrics': ['F1 (%)']}]}
    wrong = {'id': 'explanation', 'kind': 'INFERENCE', 'text': 'The F1 is 99.90%.',
             'evidence_ids': [row['evidence_id']], 'depends_on': ['measured']}
    raw = {'facts': [good, wrong], 'direct_answer_id': 'measured', 'limitations': []}
    result = generate_draft(ctx, lambda *args, **kwargs: raw)
    assert [f['id'] for f in result['facts']] == ['measured']
    assert '62.30' in result['facts'][0]['text']
    assert '99.90' not in result['facts'][0]['text']
    assert ctx.report['draft_isolation']['quarantined_fact_ids'] == ['explanation']
    assert ctx.report['raw_draft'] == raw


def test_failed_audit_repair_wrong_navigation_id_does_not_erase_valid_audited_fact():
    draft = {'facts': [{'id': 'f1', 'kind': 'SOURCE_FACT', 'text': 'Alpha uses attention.',
                       'evidence_ids': ['method']}], 'direct_answer_id': 'f1', 'limitations': []}
    spans = [{'span_id': 's0', 'evidence_id': 'method', 'text': 'Alpha uses attention.'}]
    raw = {'assessments': [{'fact_id': 'f1', 'span_ids': ['s0'], 'verdict': 'SUPPORTED',
                           'relevance': 'DIRECT'}], 'coverage': 'COMPLETE',
           'missing_aspects': [], 'omitted_answer_spans': ['row_not_a_span']}
    result = conservative_audit_contract(raw, draft, spans, question='Explain Alpha.')
    assert result['assessments'][0]['verdict'] == 'SUPPORTED'
    assert result['omitted_answer_spans'] == []
    assert result['coverage'] == 'PARTIAL' and result['missing_aspects']
    assert raw['omitted_answer_spans'] == ['row_not_a_span']


def test_explicit_operation_definition_outranks_same_terms_in_figure_caption():
    def row(identifier, content):
        return {'evidence_id': identifier, 'source_uri': 'paper://p/' + identifier,
                'page': 3, 'content': content, 'provenance': {'section_path': ['Methods']}}
    caption = row('caption', 'Alpha, Beta and Gamma modeling mechanisms for tokens are shown in Fig. 4.' * 3)
    definition = row('definition', 'Alpha modeling is unfolding tokens. Beta modeling is cross-ordering tokens. '
                     'Gamma modeling is done by concatenating tokens in the channel dimension.' * 3)
    packet = build_read_packet('Explain Alpha, Beta and Gamma token organization.',
                               [caption, definition], max_bytes=630)
    assert [r['evidence_id'] for r in packet.rows] == ['definition']


def test_host_separates_mechanism_from_projected_experiment_without_rewriting():
    draft = {'facts': [
        {'id': 'm', 'kind': 'SOURCE_FACT', 'text': 'Alpha orders tokens.'},
        {'id': 't', 'kind': 'SOURCE_FACT', 'text': 'F1=62.30', 'row_refs': [{}]}]}
    audit = {'assessments': [{'fact_id': fid, 'verdict': 'SUPPORTED', 'relevance': 'DIRECT'} for fid in ('m', 't')]}
    p = {'answer_units': ['Alpha orders tokens. [e1]', 'F1=62.30 [t]'],
         'answer_scope': 'COMPLETE', 'limitations': []}
    result = host_requested_layout('请将方法机制和实验结论分开回答。', p, draft, audit)
    assert result['presentation_contract']['status'] == 'SATISFIED'
    assert result['answer'] == '### 方法机制\n\nAlpha orders tokens. [e1]\n\n### 实验结论\n\nF1=62.30 [t]'


def test_procedural_module_definition_is_not_lost_to_experiment_mentions():
    from paperpilot_agentic.read_packet import operation_definition_priority
    question = 'How does Omega align features?'
    assert operation_definition_priority(question,
        'We propose the Omega module. Features are first processed, fed into a predictor '
        'and aligned through warping.') > operation_definition_priority(question,
        'Omega improves results. We compare its warping ablations in this experiment.')


def test_actual_packet_handoff_does_not_rerank_a_two_source_subset():
    sources = [{'evidence_id': identifier, 'source_uri': 'paper://p/' + identifier,
                'page': 3, 'content': content, 'provenance': {'section_path': ['broken PDF heading']}}
               for identifier, content in [('method', 'Alpha modeling is unfolding tokens. '
                   'Beta modeling is cross-ordering tokens. Gamma modeling is concatenating tokens.'),
                   ('table', 'TABLE I\nAlpha Beta Gamma score 62.30')]]
    question = 'Explain Alpha Beta Gamma and Table I.'
    packet = build_read_packet(question, sources)
    assert {r['evidence_id'] for r in packet.rows} == {'method', 'table'}
    ctx = initialize(question, packet.rows, SimpleNamespace(),
                     AnswerOptions(shared_read_packet=packet), None)
    assert {r['evidence_id'] for r in ctx.rows} == {'method', 'table'}
    assert ctx.report['read_packet']['packet_sha256'] == packet.identity
    with pytest.raises(ValueError, match='identity'):
        initialize(question, [dict(r, content='modified') for r in packet.rows], SimpleNamespace(),
                   AnswerOptions(shared_read_packet=packet), None)


def test_empty_requested_inference_group_is_not_satisfied():
    draft = {'facts': [{'id': 'm', 'kind': 'SOURCE_FACT', 'text': 'Alpha orders tokens.'}]}
    audit = {'assessments': [{'fact_id': 'm', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT'}]}
    p = {'answer_units': ['Alpha orders tokens. [e1]', 'Current evidence is insufficient.'],
         'answer_scope': 'COMPLETE', 'limitations': ['Current evidence is insufficient.']}
    result = host_requested_layout('请分成“论文事实”“比较性推断”“结论边界”。', p, draft, audit)
    assert result['presentation_contract']['status'] == 'UNFULFILLED'
    assert result['answer_scope'] == 'PARTIAL'


def test_missing_json_instruction_is_rejected_before_dispatch():
    from paperpilot_agentic.answer_pipeline import prepared_messages
    with pytest.raises(ValueError, match='JSON instruction'):
        prepared_messages(SimpleNamespace(), [{'role': 'system', 'content': 'Return a facts object.'}])
