from paperpilot_agentic.answer_coverage import request_contract, apply_final_coverage, source_presence
from paperpilot_agentic.table_claims import row_catalog
from test_table_claim_rows import table, ref


def test_removed_numeric_result_cannot_keep_complete_request_coverage():
    source = table()
    rows = row_catalog([source])
    contract = request_contract('引用 Table IV 一个消融结果。', rows)
    draft = {'facts': [{'id': 'f', 'text': 'Alpha F1 为 62.3%。',
                      'evidence_ids': ['t'], 'row_refs': [ref(rows[1])]}]}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'UNCERTAIN', 'relevance': 'DIRECT'}],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    actual, report = apply_final_coverage(draft, audit, [source], rows, contract)
    assert actual['coverage'] == 'PARTIAL'
    assert report['requirements'][0]['retained_fact_ids'] == []
    assert report['requirements'][0]['status'] == 'MISSING'


def test_any_legal_ablation_row_satisfies_table_result_shape_not_ground_truth():
    source = table()
    rows = row_catalog([source])
    draft = {'facts': [{'id': 'f', 'text': 'Alpha F1 为 62.3%。',
                      'evidence_ids': ['t'], 'row_refs': [ref(rows[1])]}]}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT'}],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    actual, report = apply_final_coverage(draft, audit, [source], rows,
                                          request_contract('引用 Table IV 一个消融结果。', rows))
    assert actual['coverage'] == 'COMPLETE'
    assert report['requirements'][0]['retained_fact_ids'] == ['f']
    assert report['independent_source_review'] == 'PENDING'


def test_semantic_subpart_is_recomputed_against_retained_fact_ids():
    draft = {'facts': [{'id': 'bad', 'text': 'Unverified result', 'evidence_ids': ['t']},
                       {'id': 'good', 'text': 'Mechanism', 'evidence_ids': ['t']}]}
    audit = {'assessments': [{'fact_id': 'bad', 'verdict': 'UNSUPPORTED'},
                             {'fact_id': 'good', 'verdict': 'SUPPORTED'}],
             'coverage': 'COMPLETE', 'missing_aspects': [],
             'requirement_assessments': [{'requirement': '报告实验结果', 'fact_ids': ['bad'], 'gap': ''}]}
    actual, report = apply_final_coverage(draft, audit, [], [], request_contract('Explain and report.', []))
    assert actual['coverage'] == 'PARTIAL'
    assert report['requirements'][0]['status'] == 'MISSING'


def test_current_mechanism_only_request_does_not_inherit_old_numeric_work():
    contract = request_contract('基于上一问，只作机制层面的比较。', row_catalog([table()]))
    assert contract['mechanism_only'] and not contract['table_requirements']


def test_indexed_heading_proves_presence_but_a_mention_or_miss_does_not_prove_absence():
    rows = [{'source_uri': 'paper://p/c1', 'content': 'TABLE IV\nAblation results', 'page': 9},
            {'source_uri': 'paper://p/c2', 'content': 'See Table V.', 'page': 4}]
    result = source_presence(rows, [{'paper_sha256': 'p', 'label': 'Table IV'},
                                    {'paper_sha256': 'p', 'label': 'Table V'}])
    assert [r['status'] for r in result] == ['PRESENT', 'NOT_LOCATED']
    assert result[0]['page'] == 9


def test_inventory_locates_table_in_index_even_when_it_is_not_in_current_workset():
    from types import SimpleNamespace
    from paperpilot_agentic.answer_coverage import indexed_table_inventory

    class Store:
        def get_chunks(self, ids):
            assert ids == ['c1', 'c2']
            return [SimpleNamespace(stable_id='c1', paper_sha256='p', page_start=1, text='Method section'),
                    SimpleNamespace(stable_id='c2', paper_sha256='p', page_start=9, text='TABLE IV\nResults')]

    actual = indexed_table_inventory(Store(), [{'chunk_ids': ['c1', 'c2']}], 'p',
        [{'paper_sha256': 'p', 'label': 'Table V'}])
    assert actual['rows'][0]['status'] == 'PRESENT' and actual['rows'][0]['page'] == 9
    assert actual['absence_is_not_established']
    assert actual['rows'][1]['status'] == 'NOT_LOCATED'


def test_unrequested_failed_numeric_fact_does_not_expand_mechanism_followup():
    import json
    from paperpilot_agentic.answer_audit import answer_then_audit
    from paperpilot_agentic.providers import LLMResponse

    class Audit:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            return LLMResponse(json.dumps({'assessments': [
                {'fact_id': fid, 'verdict': 'SUPPORTED', 'relevance': 'DIRECT', 'span_ids': ['s0']}
                for fid in ['m', 'n']], 'coverage': 'COMPLETE', 'missing_aspects': [],
                'requirement_assessments': [{'requirement': '说明融合机制', 'fact_ids': ['m'], 'gap': ''}]}),
                finish_reason='stop')

    draft = {'facts': [{'id': 'm', 'text': 'Alpha 模块融合两路特征。', 'evidence_ids': ['t']},
                       {'id': 'n', 'text': 'Table IV 的 F1 为 999%。', 'evidence_ids': ['t']}],
             'direct_answer_id': 'm', 'limitations': []}
    report = answer_then_audit('延续上一问，只作机制层面的比较。',
        [{'evidence_id': 't', 'content': 'Alpha 模块融合两路特征。Table IV 报告 F1=62.3%。'}],
        Audit(), frozen_draft=draft)
    assert '融合两路特征' in report['answer'] and '999' not in report['answer']
    assert report['final_request_coverage']['coverage'] == 'COMPLETE'
    assert report['current_request_scope']['unrequested_result_fact_ids'] == ['n']
