"""Normal and adversarial request wording beyond the three repair questions."""
import pytest
from paperpilot_agentic.answer_gap_display import validate_request_quotes, request_quote_units
from paperpilot_agentic.request_obligations import mechanism_only_request
from paperpilot_agentic.answer_contracts import audit_repair_diagnostics


@pytest.mark.parametrize('quote', [
    '排除 Beta 论文及其 Table IX 证据。',
    '请用中文解释实验结果。',
    '不要输出任何消融数值。',
    '只依据 Alpha 论文的 Table IV 及方法原文。',
])
def test_mechanism_followup_keeps_scope_style_and_negative_constraints(quote):
    current = '基于上一问，只作机制层面的比较。'
    history = [{'user_request': quote}]
    audit = {'requirement_assessments': [{'request_quote': quote, 'fact_ids': [], 'gap': ''}]}
    validate_request_quotes(audit, current, history)
    assert quote in [row['text'] for row in request_quote_units(current, history)]


@pytest.mark.parametrize('question', [
    '不只作机制层面的比较，还要给出实验结果。',
    '并非只作机制层面的比较。',
    'Do not only compare mechanisms; report the ablation results too.',
])
def test_negated_mechanism_only_is_not_a_scope_reset(question):
    assert not mechanism_only_request(question)


def test_diagnostic_collection_handles_bad_gap_type_without_crashing():
    draft = {'facts': [{'id': 'f', 'evidence_ids': ['e']}]}
    value = {'assessments': [], 'requirement_assessments': [{
        'request_quote': '证据不足时明确说明。', 'fact_ids': [], 'gap': None}]}
    diagnostic = audit_repair_diagnostics(value, draft, [],
        question='说明机制；证据不足时明确说明。', primary=ValueError('Bad gap type'))
    assert 'requirement_assessments[0]' in diagnostic


def test_previous_v51_checkpoint_is_not_reused_by_changed_answer_rules(tmp_path):
    import hashlib, json
    from paperpilot_agentic.db import RuntimeDatabase
    from paperpilot_agentic.models import Budget
    from paperpilot_agentic.durable_reader import run_durable_reader
    db = RuntimeDatabase(tmp_path/'old.sqlite3')
    question = 'Explain the method.'
    rid = db.create_run(question, Budget())
    evidence = [{'evidence_id': 'e', 'content': 'Alpha rearranges inputs.'}]
    # A stored legacy record, not a recomputation through the new version.
    legacy = {'version': 'durable-answer-audit-v51-active-conflict-and-source-parts',
              'question': question, 'original_question': question, 'evidence': evidence,
              'protocol_id': 'fixture', 'revision_feedback': None,
              'requested_tables': None, 'source_inventory': None}
    old = {'fingerprint': hashlib.sha256(json.dumps(legacy, sort_keys=True,
                ensure_ascii=False).encode('utf-8')).hexdigest(), 'state': 'FINISHED',
           'report': {'answer': 'Legacy reviewed answer; leave unchanged.'}}
    db.update_run_metadata(rid, {'answer_first_checkpoint': old})
    class NoCalls:
        def complete(self, *args, **kwargs):
            pytest.fail('Protocol change must stop before provider setup')
    with pytest.raises(ValueError, match='input or protocol changed'):
        run_durable_reader(db, rid, evidence, NoCalls(), protocol_id='fixture', finalize=False)
    assert db.get_run(rid)['metadata']['answer_first_checkpoint'] == old


def test_post_audit_source_repair_keeps_independent_unsupported_inference_out_of_required_patch():
    from copy import deepcopy
    from paperpilot_agentic.answer_isolation import extend_isolation_after_audit, repair_workset
    original = {'facts': [
        {'id': 'good', 'kind': 'SOURCE_FACT', 'text': 'Alpha inputs.', 'evidence_ids': ['e']},
        {'id': 'old', 'kind': 'SOURCE_FACT', 'text': 'Old malformed process.', 'evidence_ids': ['e']},
        {'id': 'beta', 'kind': 'SOURCE_FACT', 'text': 'Beta process and benefit.', 'evidence_ids': ['e']},
        {'id': 'dependent', 'kind': 'INFERENCE', 'text': 'Beta comparison.',
         'evidence_ids': ['e'], 'depends_on': ['beta']},
        {'id': 'guess', 'kind': 'INFERENCE', 'text': 'An unsupported benefit.',
         'evidence_ids': ['e'], 'depends_on': ['good']}], 'limitations': []}
    isolation = {'quarantined_fact_ids': ['old'], 'eligible_fact_ids': ['good','beta','dependent','guess'],
                 'errors': [{'fact_id': 'old', 'error': 'Malformed'}]}
    audit = {'assessments': [
        {'fact_id': 'good', 'verdict': 'SUPPORTED'},
        {'fact_id': 'beta', 'verdict': 'UNCERTAIN'},
        {'fact_id': 'dependent', 'verdict': 'UNSUPPORTED'},
        {'fact_id': 'guess', 'verdict': 'UNSUPPORTED'}]}
    before = deepcopy((original, isolation, audit))
    actual = extend_isolation_after_audit(original, isolation, audit)
    assert set(actual['quarantined_fact_ids']) == {'old','beta','dependent'}
    targets, premises, _, _ = repair_workset(original, actual, [{'evidence_id':'e','content':'Fixture'}])
    assert {f['id'] for f in targets} == {'old','beta','dependent'}
    assert [f['id'] for f in premises] == ['good']
    assert (original,isolation,audit) == before
