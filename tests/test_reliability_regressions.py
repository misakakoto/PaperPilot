"""Frozen synthetic counterparts of the September 12 HTTP failures."""
import pytest

from paperpilot_agentic.numeric_scope import check_bindings, assertions


@pytest.mark.parametrize('symbol', ['FT1_i', 'FT2_i', 'Fseq_i', 'Fcrs_i', 'Fpra_i'])
def test_formula_tokens_are_not_methods(symbol):
    fact = {'id': 'f', 'text': f'Sequential modeling arranges {symbol} tokens.',
            'evidence_ids': ['e']}
    source = [{'evidence_id': 'e', 'content':
               'ExampleMamba Sequential modeling arranges ' + symbol.replace('_', '\n') + ' tokens.'}]
    assert not any(c['kind'] == 'method' and c['blocking']
                   for c in check_bindings(fact, source)['checks'])


@pytest.mark.parametrize('name', ['xBD', 'BRIGHT', 'Alpha-2', 'Beta_2025', 'FieldSet'])
@pytest.mark.parametrize('template', ['在 {name} 测试集上准确率为 87.25%',
                                     '87.25% 对应 {name} 数据集的总体准确率'])
def test_chinese_dataset_forms(name, template):
    parsed = assertions(template.format(name=name))
    assert len(parsed) == 1
    assert parsed[0]['dataset'] == name


def test_chinese_without_spaces_does_not_hide_percentages():
    result = assertions('FlowMamba在xBD测试集上的总体准确率96.68%、mIoU 68.41%')
    assert [item['value'] for item in result] == ['96.68', '68.41']


def test_leading_percentage_copula_binds_later_chinese_dataset():
    parsed = assertions('96.68% 是 FlowMamba 模型在 xBD 数据集上取得的整体准确率。')
    assert [(item['dataset'], item['value']) for item in parsed] == [('xBD', '96.68')]


def test_deterministic_numeric_attribution_resolves_respectively_and_caption():
    from paperpilot_agentic.numeric_scope import deterministic_numeric_attribution

    source = ('TABLE IV\nPERFORMANCES ON THE BRIGHT TEST SET.\n'
              'ChangeMamba attains an overall accuracy of 96.22% and a mIoU of 67.63%. '
              'The proposed FlowMamba further improves these results to 96.68% and 68.41%, respectively.')
    result = deterministic_numeric_attribution(
        '96.68% 对应哪个数据集、哪项指标、哪张表？',
        [{'evidence_id': 'e', 'content': source}])
    assert result['dataset'] == 'BRIGHT'
    assert result['metric'] == 'overall_accuracy'
    assert result['table'] == 'IV'
    assert '总体准确率' in result['answer']


@pytest.mark.parametrize('name', ['FakeMamba', 'GhostNet', 'WrongFormer', 'OtherMamba', 'AbsentNet'])
def test_missing_non_numeric_method_is_uncertainty_not_proven_conflict(name):
    fact = {'id': 'f', 'text': f'{name} organizes temporal features.', 'evidence_ids': ['e']}
    checks = check_bindings(fact, [{'evidence_id': 'e', 'content':
                                  'RealMamba organizes temporal features.'}])['checks']
    assert all(c['status'] != 'MISMATCH' for c in checks if c['kind'] == 'method')


def test_excluded_source_is_removed():
    from paperpilot_agentic.request_scope import resolve_request_scope
    def resolve(text):
        return {name for name in ('AlphaMamba', 'BetaMamba') if name in text}
    result = resolve_request_scope('仅依据 AlphaMamba，不要混入 BetaMamba。', resolve)
    assert result['included'] == ['AlphaMamba']
    assert result['excluded'] == ['BetaMamba']


def test_table_targets_are_bound_to_their_own_paper():
    from paperpilot_agentic.request_scope import automatic_table_targets
    def resolve(text):
        return {name for name in ('AlphaMamba', 'BetaMamba') if name in text}
    targets, gaps = automatic_table_targets(
        '分别总结 AlphaMamba Table XII 与 BetaMamba Table VII 的最佳结果',
        {'AlphaMamba', 'BetaMamba'}, resolve)
    assert not gaps
    assert targets == [{'paper_sha256': 'AlphaMamba', 'label': 'Table XII'},
                       {'paper_sha256': 'BetaMamba', 'label': 'Table VII'}]


def test_unread_table_is_not_proof_of_absence():
    from paperpilot_agentic.answer_boundaries import enforce_answer_boundaries
    draft = {'facts': [{'id': 'f', 'text': '论文 Table XII 未提供具体数值。',
                        'evidence_ids': ['e']}], 'limitations': []}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED'}],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    effective, _ = enforce_answer_boundaries(draft, audit, [{'evidence_id': 'e',
        'content': 'Table XII compares three modeling strategies.'}])
    assert effective['assessments'][0]['verdict'] == 'UNCERTAIN'


def test_inference_cannot_survive_its_rejected_premise():
    from paperpilot_agentic.answer_boundaries import enforce_answer_boundaries
    draft = {'facts': [{'id': 'f', 'text': 'A score', 'evidence_ids': ['e']},
                      {'id': 'g', 'kind': 'INFERENCE', 'text': 'A is better',
                       'depends_on': ['f'], 'evidence_ids': ['e']}], 'limitations': []}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'UNSUPPORTED'},
                            {'fact_id': 'g', 'verdict': 'SUPPORTED'}],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    effective, _ = enforce_answer_boundaries(draft, audit, [])
    assert effective['assessments'][1]['verdict'] == 'UNCERTAIN'


def test_paper_claim_cannot_bypass_audit_in_limitations():
    from paperpilot_agentic.answer_boundaries import safe_limitations
    visible, withheld = safe_limitations(['Table XII未提供各机制组合下的最佳性能数值。'])
    assert withheld
    assert not any('未提供' in v for v in visible)


@pytest.mark.parametrize('reason', ['，因为论文未报告任何失败。', '因为系统保证绝对安全。',
                                  '；该方案已优于所有基线。'])
def test_nominal_uncertainty_keeps_target_not_unverified_reason(reason):
    from paperpilot_agentic.answer_boundaries import safe_limitations
    value = '当前证据无法确定模型的运行时、硬件或内存需求' + reason
    visible, withheld = safe_limitations([value])
    assert visible == ['当前证据无法确定模型的运行时、硬件或内存需求。']
    assert withheld == [value]


def test_pure_nominal_uncertainty_needs_no_withheld_claim():
    from paperpilot_agentic.answer_boundaries import safe_limitations
    value = '当前证据无法确定方法的计算开销。'
    assert safe_limitations([value]) == ([value], [])


@pytest.mark.parametrize('value,topic,hidden', [
    ('The passages do not explicitly state whether all samples are sterile; the method guarantees safety.',
     'all samples are sterile', 'guarantees safety'),
    ('The supplied evidence does not establish whether the effect persists. The paper never reports failures.',
     'the effect persists', 'never reports failures'),
    ('当前证据无法确定是否全部样本无菌；该方法保证安全。', '是否全部样本无菌', '保证安全'),
])
def test_specific_uncertainty_does_not_publish_trailing_assertion(value, topic, hidden):
    from paperpilot_agentic.answer_boundaries import safe_limitations
    visible, withheld = safe_limitations([value])
    assert topic in ' '.join(visible)
    assert hidden not in ' '.join(visible)
    assert withheld == [value]


@pytest.mark.parametrize('suffix', ['；论文从未报告失败。', '; the treatment guarantees safety.'])
def test_chinese_subject_before_whether_keeps_only_uncertainty_clause(suffix):
    from paperpilot_agentic.answer_boundaries import safe_limitations
    value = '当前证据无法确定模型的部署延迟是否有依据' + suffix
    visible, withheld = safe_limitations([value])
    assert visible == ['当前证据无法确定模型的部署延迟是否有依据。']
    assert withheld == [value]


@pytest.mark.parametrize('value', [
    '论文从未报告失败，因此当前证据无法确定模型是否稳定。',
    '材料没有实验数据，当前证据无法确定两者是否有延迟依据，系统保证绝对安全。',
])
def test_embedded_uncertainty_does_not_promote_surrounding_claims(value):
    from paperpilot_agentic.answer_boundaries import safe_limitations
    visible, withheld = safe_limitations([value])
    assert len(visible) == 1
    assert visible[0].startswith('当前证据无法确定')
    assert '从未' not in visible[0] and '绝对安全' not in visible[0]
    assert withheld == [value]


def test_complete_audit_does_not_erase_specific_gap_because_other_claim_was_withheld():
    from paperpilot_agentic.answer_audit import audited_presentation
    gap = 'The passages do not explicitly state whether all samples are sterile; the method guarantees safety.'
    draft = {'facts': [{'id': 'f', 'text': 'A source fact.', 'evidence_ids': ['e']}],
             'direct_answer_id': 'f', 'limitations': [gap]}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT'}],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    result = audited_presentation(draft, audit)
    assert 'all samples are sterile' in ' '.join(result['limitations'])
    assert result['answer_scope'] == 'PARTIAL'
    assert result['unverified_limitations'] == [gap]


@pytest.mark.parametrize('claim', [
    'ChangeMamba does not compare directly with FlowMamba.',
    "ChangeMamba's sensitivity is not quantified in the same way.",
    'The paper does not explicitly describe the implementation. No further details are given.',
])
def test_english_paper_wide_negative_requires_exact_evidence(claim):
    from paperpilot_agentic.answer_boundaries import enforce_answer_boundaries
    draft = {'facts': [{'id': 'f', 'text': claim, 'evidence_ids': ['e']}],
             'direct_answer_id': 'f', 'limitations': []}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED'}],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    effective, _ = enforce_answer_boundaries(
        draft, audit, [{'evidence_id': 'e', 'content': 'The available passage describes the method.'}])
    assert effective['assessments'][0]['verdict'] == 'UNCERTAIN'


def test_unverified_limitation_does_not_make_complete_answer_partial():
    from paperpilot_agentic.answer_audit import audited_presentation
    draft = {'facts': [{'id': 'f', 'text': '已经回答。', 'evidence_ids': ['e']}],
             'direct_answer_id': 'f', 'limitations': ['论文未提供其他细节。']}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT'}],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    result = audited_presentation(draft, audit)
    assert result['answer_scope'] == 'MODEL_REVIEWED'
    assert result['limitations'] == []
    assert result['unverified_limitations'] == ['论文未提供其他细节。']


def test_flowmamba_complete_ablation_row_is_not_confused_with_offset_or_table_number():
    from paperpilot_agentic.numeric_scope import apply_scope_guard
    source = ('Table VII\nNamed-cell transcription (each row is independent):\n'
              '{"column_headers":["MFA","CDA","F1overall"],"rows":['
              '{"row_label":"baseline","checked_columns":[],"cells":{"F1overall":"80.14"}},'
              '{"row_label":"joint","checked_columns":["MFA","CDA"],'
              '"cells":{"F1overall":"80.53"}}]}')
    text = ('FlowMamba Table VII tested offsets 0, 5, 10, 15, 20 px. The best result is '
            "achieved with both MFA and CDA included (row labeled 'w/o' with checked_columns "
            "['MFA', 'CDA']), yielding F1overall=80.53%.")
    draft = {'facts': [{'id': 'f', 'text': text, 'evidence_ids': ['e']}],
             'direct_answer_id': 'f', 'limitations': []}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED'}],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    effective, report = apply_scope_guard(draft, audit, [{'evidence_id': 'e', 'content': source}])
    assert effective['assessments'][0]['verdict'] == 'SUPPORTED'
    row_checks = [c for c in report['facts'][0]['binding_checks'] if c['kind'] == 'table_row']
    assert row_checks == [{
        'kind': 'table_row', 'claim': "['cda', 'mfa'] f1overall=80.53", 'status': 'MATCH',
        'blocking': True, 'metric': 'f1overall', 'value': '80.53',
        'expected_checked_columns': ['cda', 'mfa']}]


def test_chinese_ablation_enumeration_preserves_each_configuration_row():
    from paperpilot_agentic.numeric_scope import apply_scope_guard
    source = ('Table VII\nNamed-cell transcription (each row is independent):\n'
              '{"column_headers":["MFA","CDA","F1overall"],"rows":['
              '{"checked_columns":[],"cells":{"F1overall":"80.14"}},'
              '{"checked_columns":["MFA"],"cells":{"F1overall":"80.26"}},'
              '{"checked_columns":["CDA"],"cells":{"F1overall":"80.43"}},'
              '{"checked_columns":["MFA","CDA"],"cells":{"F1overall":"80.53"}}]}')
    text = ('FlowMamba Table VII 的消融设置包括：(1) 基线，F1overall=80.14%；'
            '(2) 单独引入 MFA，F1overall=80.26%；(3) 单独引入 CDA，'
            'F1overall=80.43%；(4) 同时引入 MFA 和 CDA，F1overall=80.53%。')
    draft = {'facts': [{'id': 'f', 'text': text, 'evidence_ids': ['e']}],
             'direct_answer_id': 'f', 'limitations': []}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED'}],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    effective, report = apply_scope_guard(draft, audit, [{'evidence_id': 'e', 'content': source}])
    assert effective['assessments'][0]['verdict'] == 'SUPPORTED'
    assert [(c['expected_checked_columns'], c['value'], c['status'])
            for c in report['facts'][0]['binding_checks'] if c['kind'] == 'table_row'] == [
        ([], '80.14', 'MATCH'), (['mfa'], '80.26', 'MATCH'),
        (['cda'], '80.43', 'MATCH'), (['cda', 'mfa'], '80.53', 'MATCH')]


def test_cross_table_inference_relies_on_checked_premises_not_one_table_scope():
    from paperpilot_agentic.numeric_scope import check_bindings
    fact = {'id': 'i', 'kind': 'INFERENCE',
            'text': 'Table XII OA=92.35 and Table VII F1overall=80.53 are not directly comparable.',
            'evidence_ids': ['a', 'b']}
    evidence = [
        {'evidence_id': 'a', 'content': 'Table XII\nNamed-cell transcription\n'
         '{"column_headers":["OA"],"rows":[{"cells":{"OA":"92.35"}}]}'},
        {'evidence_id': 'b', 'content': 'Table VII\nNamed-cell transcription\n'
         '{"column_headers":["F1overall"],"rows":[{"cells":{"F1overall":"80.53"}}]}'},
    ]
    kinds = [item['kind'] for item in check_bindings(fact, evidence)['checks']]
    assert 'table_cell' not in kinds
    assert 'table_row' not in kinds


def test_w_o_labels_inside_best_row_annotation_do_not_create_baseline_claims():
    from paperpilot_agentic.numeric_scope import apply_scope_guard
    source = ('Table VII\nNamed-cell transcription (each row is independent):\n'
              '{"column_headers":["MFA","CDA","F1overall"],"rows":['
              '{"checked_columns":[],"cells":{"F1overall":"80.14"}},'
              '{"checked_columns":["MFA","CDA"],"cells":{"F1overall":"80.53"}}]}')
    text = ('当 MFA 与 CDA 均启用时（即“w/o”行中勾选 MFA 与 CDA），'
            '获得最佳整体性能，F1overall=80.53%。')
    draft = {'facts': [{'id': 'f', 'text': text, 'evidence_ids': ['e']}],
             'direct_answer_id': 'f', 'limitations': []}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED'}],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    effective, report = apply_scope_guard(draft, audit, [{'evidence_id': 'e', 'content': source}])
    assert effective['assessments'][0]['verdict'] == 'SUPPORTED'
    checks = [c for c in report['facts'][0]['binding_checks'] if c['kind'] == 'table_row']
    assert [(c['expected_checked_columns'], c['value']) for c in checks] == [
        (['cda', 'mfa'], '80.53')]


def test_scope_difference_inference_is_not_treated_as_paper_absence():
    from paperpilot_agentic.answer_boundaries import enforce_answer_boundaries
    draft = {'facts': [
        {'id': 'a', 'kind': 'SOURCE_FACT', 'text': 'A 在 SYSU 上评测。', 'evidence_ids': ['e']},
        {'id': 'b', 'kind': 'SOURCE_FACT', 'text': 'B 在 xBD 上评测。', 'evidence_ids': ['e']},
        {'id': 'i', 'kind': 'INFERENCE', 'text': '数据集不同，缺乏统一比较基础，不可直接比较。',
         'depends_on': ['a', 'b'], 'evidence_ids': ['e']}],
        'direct_answer_id': 'i', 'limitations': []}
    audit = {'assessments': [{'fact_id': fid, 'verdict': 'SUPPORTED'} for fid in ('a', 'b', 'i')],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    effective, _ = enforce_answer_boundaries(draft, audit, [{'evidence_id': 'e', 'content': 'scopes'}])
    assert effective['assessments'][2]['verdict'] == 'SUPPORTED'


def test_table_scoped_comparison_boundary_remains_user_visible():
    from paperpilot_agentic.answer_boundaries import safe_limitations
    value = 'Table XII 与 Table VII 的数据集和评价指标不同，这两张表的数值不能直接比较整体优劣。'
    visible, withheld = safe_limitations([value, '论文未提供其他实验。'])
    assert visible == [value]
    assert withheld == ['论文未提供其他实验。']


def test_possessive_experiment_name_does_not_expand_paper_scope():
    from paperpilot_agentic.request_scope import resolve_request_scope
    mapping = {'FlowMamba': {'flow'}, 'BRIGHT': {'bright-paper'}}
    def resolve(text):
        return {sha for name, values in mapping.items() if name in text for sha in values}
    scope = resolve_request_scope('FlowMamba 的 BRIGHT 实验中 96.68% 是什么指标？', resolve)
    assert scope['included'] == ['flow']


def test_numeric_lookup_query_removes_asserted_dataset_but_keeps_values():
    from paperpilot_agentic.request_scope import numeric_lookup_query
    assert numeric_lookup_query('FlowMamba 是否在 xBD 上取得 96.68% 总体准确率？') == '96.68%'
    assert numeric_lookup_query('no number') is None


def test_deterministic_full_combination_difference_uses_named_rows():
    from paperpilot_agentic.reader_visual import deterministic_table_difference
    content = ('Table XII\nNamed-cell transcription (each row is independent):\n'
               '{"column_headers":["Sequential","Cross","Parallel","F1(%)"],"rows":['
               '{"checked_columns":[],"cells":{"F1(%)":"81.11"}},'
               '{"checked_columns":["Sequential"],"cells":{"F1(%)":"82.20"}},'
               '{"checked_columns":["Sequential","Cross","Parallel"],'
               '"cells":{"F1(%)":"82.83"}}]}')
    result = deterministic_table_difference(
        'Table XII 的完整组合比基线 F1 高多少个百分点？',
        [{'evidence_id': 'e', 'content': content}])
    assert result['difference'] == '1.72'
    assert '82.83 − 81.11 = 1.72' in result['answer']


def test_deterministic_two_table_ablation_summary_keeps_comparison_boundary():
    from paperpilot_agentic.reader_visual import deterministic_ablation_summary
    left = ('Table XII\nNamed-cell transcription (each row is independent):\n'
            '{"rows":[{"checked_columns":[],"cells":{"F1":"81.11"}},'
            '{"checked_columns":["Sequential","Cross","Parallel"],'
            '"cells":{"F1":"82.83"}}],'
            '"nearby_pdf_caption_text":{"text":"ABLATION ON THE SYSU DATASET"}}')
    right = ('Table VII\nNamed-cell transcription (each row is independent):\n'
             '{"rows":[{"checked_columns":[],"cells":{"offset":"w/o",'
             '"F1overall(%)":"80.14"}},{"checked_columns":["MFA"],'
             '"cells":{"offset":"5px","F1overall(%)":"80.02"}},'
             '{"checked_columns":["MFA","CDA"],"cells":{"offset":"w/o",'
             '"F1overall(%)":"80.53"}}],'
             '"nearby_pdf_caption_text":{"text":"ABLATION ON XBD TEST SET"}}')
    result = deterministic_ablation_summary(
        '总结 ChangeMamba Table XII 与 FlowMamba Table VII 的消融设置和最佳结果；能否证明整体更优？',
        [{'evidence_id': 'left', 'content': left}, {'evidence_id': 'right', 'content': right}])
    assert 'F1=82.83' in result['answer']
    assert 'F1overall(%)=80.53' in result['answer']
    assert '5px' in result['answer']
    assert 'SYSU' in result['answer']
    assert 'XBD' in result['answer']
    assert '不能据此证明' in result['answer']


def test_invalid_repaired_audit_spans_are_conservatively_downgraded():
    from paperpilot_agentic.answer_audit import conservative_audit_contract
    draft = {'facts': [{'id': 'f', 'text': 'Claim', 'evidence_ids': ['e']},
                       {'id': 'g', 'text': 'Grounded', 'evidence_ids': ['e']}],
             'direct_answer_id': 'f', 'limitations': []}
    spans = [{'span_id': 'ok', 'evidence_id': 'e', 'text': 'Claim', 'start': 0, 'end': 5},
             {'span_id': 'grounded', 'evidence_id': 'e', 'text': 'Grounded', 'start': 6, 'end': 14},
             {'span_id': 'bad', 'evidence_id': 'other', 'text': 'Other', 'start': 0, 'end': 5}]
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                              'reason': 'Both', 'span_ids': ['ok', 'bad']},
                             {'fact_id': 'g', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                              'reason': 'Grounded', 'span_ids': ['grounded']}],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    value = conservative_audit_contract(audit, draft, spans, question='中文问题')
    assert value['assessments'][0]['verdict'] == 'UNCERTAIN'
    assert value['assessments'][0]['span_ids'] == []
    assert value['coverage'] == 'PARTIAL'


def test_generated_inference_requires_declared_fact_dependencies():
    from paperpilot_agentic.answer_audit import validate_draft
    evidence = [{'evidence_id': 'e'}]
    draft = {'facts': [
        {'id': 'f1', 'kind': 'SOURCE_FACT', 'text': 'A.', 'evidence_ids': ['e']},
        {'id': 'f2', 'kind': 'INFERENCE', 'text': 'Therefore B.', 'evidence_ids': ['e']}],
        'direct_answer_id': 'f2', 'limitations': []}
    with pytest.raises(ValueError, match='dependencies'):
        validate_draft(draft, evidence)


def test_chinese_question_rejects_all_english_generated_answer():
    from paperpilot_agentic.answer_audit import validate_draft
    draft = {'facts': [{'id': 'f1', 'text': 'The result is 80.53%.', 'evidence_ids': ['e']}],
             'direct_answer_id': 'f1', 'limitations': []}
    with pytest.raises(ValueError, match='language'):
        validate_draft(draft, [{'evidence_id': 'e'}], question='请总结表格结果')


def test_requested_caption_is_prioritized_over_background():
    from paperpilot_agentic.request_scope import prioritize_requested_evidence
    evidence = [{'evidence_id': 'bg', 'source_uri': 'paper://p/1', 'content': 'Background'},
                {'evidence_id': 'table', 'source_uri': 'paper://p/2', 'content': 'TABLE XII\nON DATA'}]
    ordered = prioritize_requested_evidence(evidence, [{'paper_sha256': 'p', 'label': 'Table XII'}])
    assert ordered[0]['evidence_id'] == 'table'


def test_exact_table_extrema_preserve_row_identity():
    from paperpilot_agentic.reader_visual import table_calculations
    result = table_calculations([{'cells': {'F1(%)': '81.11'}}, {'cells': {'F1(%)': '82.83'}}])
    assert result[0]['max_minus_min'] == '1.72'
    assert result[0]['max_row_indices'] == [1]
    assert result[0]['difference_unit'] == 'percentage_points'


@pytest.mark.parametrize('policy,allow,expected', [('auto', False, 'off'), ('auto', True, 'auto'),
                                                 ('off', True, 'off'), ('explicit', True, 'explicit')])
def test_visual_off_precedes_automatic_policy(policy, allow, expected):
    from paperpilot_agentic.request_scope import visual_policy
    assert visual_policy({'visual_policy': policy, 'allow_visual': allow}) == expected


def test_recovery_validation_preserves_raw_response():
    from paperpilot_agentic.reader_recovery import plan_evidence_reload
    from test_reader_recovery import Model, CANDIDATES
    seen = []
    with pytest.raises(ValueError):
        plan_evidence_reload('How?', {}, CANDIDATES, {'a'},
                             Model({'action': 'RELOAD', 'evidence_ids': ['invented'], 'reason': 'gap'}),
                             on_progress=lambda value: seen.append(value))
    assert seen[-1]['validation_error']
    assert 'invented' in seen[-1]['attempts'][0]['response']['text']


def test_automatic_table_reaches_formal_runtime(tmp_path, monkeypatch):
    from test_reader_pipeline import factory, AnswerProvider
    import paperpilot_agentic.reader_visual as visual_module
    captured = []
    def collect(runtime, rid, question, targets, scope, tool):
        captured.extend(targets)
        return [{'evidence_id': 'visual', 'evidence_type': 'TABLE', 'page': 1,
                 'source_uri': 'paper://sha/table', 'content': 'A source fact.'}]
    monkeypatch.setattr(visual_module, 'collect_visual_evidence', collect)
    runtime = factory(tmp_path, AnswerProvider())
    runtime.search_adapter.resolve_explicit_title_scope = lambda queries, **kwargs: {'sha'}
    rid = runtime.enqueue(question='Results in ExampleMamba Table XII', metadata={
        'engine': 'answer_first_text', 'visual_policy': 'auto'})['run']['run_id']
    result = runtime.resume(rid)
    assert captured == [{'paper_sha256': 'sha', 'label': 'Table XII'}]
    assert result['run']['status'] == 'COMPLETED'
    before = result['run']['used_llm_calls']
    assert runtime.resume(rid)['run']['used_llm_calls'] == before
    assert len(captured) == 1


def test_automatic_visual_failure_falls_back_to_scoped_text(tmp_path, monkeypatch):
    from test_reader_pipeline import factory, AnswerProvider
    import paperpilot_agentic.reader_visual as visual_module
    monkeypatch.setattr(visual_module, 'collect_visual_evidence', lambda *args: [])
    provider = AnswerProvider()
    runtime = factory(tmp_path, provider)
    runtime.search_adapter.resolve_explicit_title_scope = lambda queries, **kwargs: {'sha'}
    rid = runtime.enqueue(question='Results in ExampleMamba Table XII', metadata={
        'engine': 'answer_first_text', 'visual_policy': 'auto'})['run']['run_id']
    result = runtime.resume(rid)
    assert result['run']['status'] == 'COMPLETED'
    assert result['final_answer']['answer'].startswith('A source fact.')
    assert result['run']['metadata']['reader_visual_degraded_to_text'] is True


def test_visual_off_blocks_tools_even_with_explicit_targets(tmp_path, monkeypatch):
    from test_reader_pipeline import factory, AnswerProvider
    import paperpilot_agentic.reader_visual as visual_module
    monkeypatch.setattr(visual_module, 'collect_visual_evidence', lambda *args: pytest.fail('visual disabled'))
    runtime = factory(tmp_path, AnswerProvider())
    rid = runtime.enqueue(question='Table XII', metadata={'engine': 'answer_first_text',
        'paper_sha256s': ['sha'], 'visual_policy': 'auto', 'allow_visual': False,
        'reader_visual_targets': [{'paper_sha256': 'sha', 'label': 'Table XII'}]})['run']['run_id']
    assert runtime.resume(rid)['run']['status'] == 'COMPLETED'


def test_recovery_repairs_once_through_shared_allowance():
    from paperpilot_agentic.reader_recovery import plan_evidence_reload
    from test_reader_recovery import Model, CANDIDATES
    class Repairable(Model):
        def for_repair(self):
            return Model({'action': 'RELOAD', 'evidence_ids': ['b'], 'reason': 'gap'})
    seen = []
    value = plan_evidence_reload('How?', {}, CANDIDATES, {'a'},
        Repairable({'action': 'RELOAD', 'evidence_ids': ['bad'], 'reason': 'gap'}),
        on_progress=lambda v: seen.append(v))
    assert value['evidence_ids'] == ['b']
    assert len(seen[-1]['attempts']) == 2


def test_transient_provider_retry_does_not_consume_the_auditor_slot():
    import json

    from paperpilot_agentic.answer_audit import answer_then_audit
    from paperpilot_agentic.providers import LLMResponse

    evidence = [{'evidence_id': 'e', 'content': 'Sequential 按时间顺序排列双时相 token。'}]
    english = {'facts': [{'id': 'f1', 'kind': 'SOURCE_FACT',
                          'text': 'Sequential orders the two temporal token sequences.',
                          'evidence_ids': ['e'], 'depends_on': []}],
               'direct_answer_id': 'f1', 'limitations': []}
    chinese = {'facts': [{'id': 'f1', 'kind': 'SOURCE_FACT',
                          'text': 'Sequential 按时间顺序排列双时相 token。',
                          'evidence_ids': ['e'], 'depends_on': []}],
               'direct_answer_id': 'f1', 'limitations': []}
    audit = {'assessments': [{'fact_id': 'f1', 'verdict': 'SUPPORTED',
                              'relevance': 'DIRECT', 'reason': '原文直接说明。',
                              'span_ids': ['s0']}],
             'coverage': 'COMPLETE', 'missing_aspects': []}

    class RetryOnceModel:
        def __init__(self):
            self.outputs = iter([english, chinese, audit])
            self.calls = 0

        def complete(self, messages, **kwargs):
            self.calls += 1
            kwargs['before_attempt']()
            if self.calls == 1:
                kwargs['before_attempt']()  # one transient provider retry before success
            return LLMResponse(text=json.dumps(next(self.outputs), ensure_ascii=False),
                               finish_reason='stop')

    result = answer_then_audit('请解释 Sequential。', evidence, RetryOnceModel())
    assert result['status'] == 'REVIEW_REQUIRED'
    assert result['answer'].startswith('Sequential 按时间顺序')
    assert result['provider_attempts'] == 4


def test_too_many_valid_repaired_spans_downgrades_only_affected_fact():
    from paperpilot_agentic.answer_audit import conservative_audit_contract

    spans = [{'span_id': f's{i}', 'evidence_id': 'e', 'text': str(i),
              'start': i, 'end': i + 1} for i in range(13)]
    draft = {'facts': [
        {'id': 'f1', 'kind': 'SOURCE_FACT', 'text': 'Direct.', 'evidence_ids': ['e']},
        {'id': 'f2', 'kind': 'SOURCE_FACT', 'text': 'Context.', 'evidence_ids': ['e']}],
        'direct_answer_id': 'f1', 'limitations': []}
    audit = {'assessments': [
        {'fact_id': 'f1', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
         'reason': 'Too broad.', 'span_ids': [f's{i}' for i in range(13)]},
        {'fact_id': 'f2', 'verdict': 'SUPPORTED', 'relevance': 'NECESSARY_CONTEXT',
         'reason': 'Minimal.', 'span_ids': ['s0']}],
        'coverage': 'COMPLETE', 'missing_aspects': []}

    normalized = conservative_audit_contract(audit, draft, spans, question='中文问题')
    assert normalized['assessments'][0]['verdict'] == 'UNCERTAIN'
    assert normalized['assessments'][1]['verdict'] == 'SUPPORTED'
    assert normalized['coverage'] == 'PARTIAL'


def test_repaired_comparison_gets_structural_dependencies_from_shared_sources():
    import json

    from paperpilot_agentic.answer_audit import answer_then_audit
    from paperpilot_agentic.providers import LLMResponse

    evidence = [{'evidence_id': 'left', 'content': 'Table A uses dataset Alpha.'},
                {'evidence_id': 'right', 'content': 'Table B uses dataset Beta.'}]
    invalid = {'facts': [
        {'id': 'f1', 'kind': 'SOURCE_FACT', 'text': 'Table A 使用 Alpha。',
         'evidence_ids': ['left']},
        {'id': 'f2', 'kind': 'SOURCE_FACT', 'text': 'Table B 使用 Beta。',
         'evidence_ids': ['right']},
        {'id': 'f3', 'kind': 'LIMITATION', 'text': '两表不能直接比较。',
         'evidence_ids': ['left', 'right']}],
        'direct_answer_id': 'f3', 'limitations': []}
    repaired = {'facts': [
        {'id': 'f1', 'kind': 'SOURCE_FACT', 'text': 'Table A 使用 Alpha。',
         'evidence_ids': ['left']},
        {'id': 'f2', 'kind': 'SOURCE_FACT', 'text': 'Table B 使用 Beta。',
         'evidence_ids': ['right']},
        {'id': 'f3', 'kind': 'INFERENCE', 'text': '两张表使用不同数据集，不能直接比较。',
         'evidence_ids': ['left', 'right'], 'depends_on': []}],
        'direct_answer_id': 'f3', 'limitations': []}
    audit = {'assessments': [
        {'fact_id': 'f1', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
         'reason': 'A.', 'span_ids': ['s0']},
        {'fact_id': 'f2', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
         'reason': 'B.', 'span_ids': ['s1']},
        {'fact_id': 'f3', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
         'reason': 'Comparison follows.', 'span_ids': ['s0', 's1']}],
        'coverage': 'COMPLETE', 'missing_aspects': []}

    class Model:
        def __init__(self):
            # The eligible SOURCE_FACTs are now audited before a local patch.
            # Keep the original dependency and comparison assertions below.
            early_audit = {**audit, 'assessments': audit['assessments'][:2]}
            patch = {'facts': repaired['facts'][2:], 'limitations': []}
            self.outputs = iter([invalid, early_audit, patch, audit])

        def ensure_capacity(self, requests):
            pass  # Scripted provider, no network or billable requests.

        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            return LLMResponse(text=json.dumps(next(self.outputs), ensure_ascii=False),
                               finish_reason='stop')

    result = answer_then_audit('比较两张表。', evidence, Model())
    assert result['status'] == 'REVIEW_REQUIRED'
    assert result['draft']['facts'][-1]['depends_on'] == ['f1', 'f2']
    assert '不能直接比较' in result['answer']
    assert result['draft_isolation']['repair_status'] == 'PATCH_AUDITED'
    assert result['provider_attempts'] == 4


def test_draft_repair_reports_kind_and_language_failures_together():
    from paperpilot_agentic.answer_audit import draft_repair_diagnostics

    raw = {'facts': [{'id': 'f1', 'kind': 'LIMITATION',
                      'text': 'The tables cannot be directly compared.',
                      'evidence_ids': ['e']}],
           'direct_answer_id': 'f1', 'limitations': []}
    diagnostic = draft_repair_diagnostics(
        raw, question='请比较两张表。', primary=ValueError('Fact kind must distinguish source fact from inference'))
    assert 'Fact kind' in diagnostic
    assert 'Chinese original question' in diagnostic
