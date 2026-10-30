"""A disclosure policy must not invent scientific work or change source verdicts."""
from copy import deepcopy
from dataclasses import dataclass
import json
import pytest

from paperpilot_agentic.answer_audit import answer_then_audit


@dataclass(frozen=True)
class Response:
    text: str
    finish_reason: str = 'stop'
    prompt_tokens: int = 100
    completion_tokens: int = 100


def fixture():
    question = 'Explain how Alpha works; if evidence is insufficient, state so.'
    evidence = [{'evidence_id': 'e', 'content': 'Alpha reshapes the input tokens.'}]
    draft = {'facts': [{'id': 'a', 'kind': 'SOURCE_FACT',
                       'text': 'Alpha reshapes the input tokens.', 'evidence_ids': ['e']}],
             'direct_answer_id': 'a',
             'limitations': ['The supplied evidence does not report results on other datasets.']}
    audit = {'assessments': [{'fact_id': 'a', 'verdict': 'SUPPORTED', 'span_ids': ['s0'],
                             'relevance': 'DIRECT', 'reason': 'Exact original statement.'}],
             'coverage': 'COMPLETE', 'missing_aspects': [], 'omitted_answer_spans': [],
             'requirement_assessments': [
                 {'requirement': 'Explain how Alpha works', 'request_quote': 'Explain how Alpha works',
                  'fact_ids': ['a'], 'gap': ''},
                 {'requirement': 'Disclosure policy', 'request_quote': 'if evidence is insufficient, state so.',
                  'fact_ids': [], 'gap': 'The optional cross-dataset limitation is disclosed.',
                  'limitation_indices': [0]}],
             'limitation_assessments': [{'limitation_index': 0, 'role': 'REQUEST_GAP',
                 'requirement': 'Disclosure policy', 'reason': 'Disclosure was requested.'}]}
    return question, evidence, draft, audit


def test_formal_pipeline_does_not_repair_optional_policy_metadata_or_lose_complete_content():
    q, evidence, draft, audit = fixture()
    original = deepcopy((draft, audit))
    responses = iter((draft, audit))
    calls = []

    class Model:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            calls.append(deepcopy(messages))
            return Response(json.dumps(next(responses)))

    report = answer_then_audit(q, evidence, Model())
    assert len(calls) == 2, report.get('audit_repair')
    assert report['answer_scope'] == 'MODEL_REVIEWED'
    assert report['audit']['coverage'] == 'COMPLETE'
    assert report['audit']['assessments'][0]['verdict'] == 'SUPPORTED'
    assert report['limitations'] == []
    assert (draft, audit) == original


@pytest.mark.parametrize('problem', ['unsupported', 'real_gap', 'partial', 'missing', 'omitted'])
def test_policy_isolation_preserves_real_reasons_to_withhold_completion(problem):
    from paperpilot_agentic.answer_completion import isolate_disclosure_policy_metadata
    from paperpilot_agentic.answer_contracts import validate_audit, evidence_spans, audited_presentation
    from paperpilot_agentic.answer_coverage import apply_final_coverage, request_contract
    q, evidence, draft, raw = fixture()
    if problem == 'unsupported':
        raw['assessments'][0].update(verdict='UNSUPPORTED', span_ids=[])
    elif problem == 'real_gap':
        raw['requirement_assessments'][0].update(fact_ids=[], gap='The mechanism is not established.')
    elif problem == 'partial':
        raw['coverage'] = 'PARTIAL'
    elif problem == 'missing':
        raw['missing_aspects'] = ['A requested mechanism remains uncertain.']
    else:
        raw['omitted_answer_spans'] = ['s0']
    value = isolate_disclosure_policy_metadata(raw, draft, question=q)
    audit = validate_audit(value, draft, evidence, spans=evidence_spans(evidence),
                           require_relevance=True, question=q)
    effective, _ = apply_final_coverage(draft, audit, evidence, [], request_contract(q, []))
    assert audited_presentation(draft, effective, question=q)['answer_scope'] != 'MODEL_REVIEWED'
    assert effective['assessments'] == raw['assessments']


def test_a_policy_cannot_be_borrowed_from_another_request():
    from paperpilot_agentic.answer_completion import isolate_disclosure_policy_metadata
    _, _, draft, raw = fixture()
    with pytest.raises(ValueError, match='original user request'):
        isolate_disclosure_policy_metadata(raw, draft, question='Explain Alpha only.')


def test_shared_required_boundary_is_not_withheld_as_optional():
    from paperpilot_agentic.answer_completion import isolate_disclosure_policy_metadata
    q, _, draft, raw = fixture()
    raw['requirement_assessments'][0].update(gap='An explicitly requested boundary is unresolved.',
                                            limitation_indices=[0])
    actual = isolate_disclosure_policy_metadata(raw, draft, question=q)
    assert actual['withheld_limitation_indices'] == []
    assert actual['requirement_assessments'][0]['gap']


def test_model_cannot_grant_itself_a_limitation_waiver():
    from paperpilot_agentic.answer_completion import (
        isolate_disclosure_policy_metadata, validate_limitation_assessments)
    q, _, draft, raw = fixture()
    raw['requirement_assessments'] = raw['requirement_assessments'][:1]
    raw['limitation_assessments'][0]['requirement'] = 'Explain how Alpha works'
    raw['withheld_limitation_indices'] = [0]
    raw['disclosure_policy_metadata'] = {'withheld_optional_limitation_indices': [0]}
    actual = isolate_disclosure_policy_metadata(raw, draft, question=q)
    assert 'withheld_limitation_indices' not in actual
    with pytest.raises(ValueError, match='unresolved requested requirement'):
        validate_limitation_assessments(actual, draft)


def test_saved_v52_complete_facts_survive_policy_metadata_without_new_model_reasoning():
    from pathlib import Path
    from paperpilot_agentic.answer_completion import isolate_disclosure_policy_metadata
    from paperpilot_agentic.answer_contracts import validate_audit, audited_presentation
    path = Path(__file__).resolve().parents[1] / (
        'outputs/qa-timeout-frozen-three-20261014/revisions/0001/single_changemamba_detail-result.json')
    if not path.exists():
        pytest.skip('Local saved v52 result is not checked into Git')
    result = json.loads(path.read_text(encoding='utf-8'))
    p = result['run']['metadata']['answer_first_checkpoint']['report']
    # The final re-audit is SDK receipt 26; repaired_audit in this historical
    # checkpoint belongs to the earlier, eight-fact pre-patch candidate.
    receipt = path.parents[2] / 'receipts/response-026.json'
    if not receipt.exists():
        pytest.skip('Local final-audit SDK receipt is not checked into Git')
    raw = json.loads(json.loads(receipt.read_text(encoding='utf-8'))['choices'][0]['message']['content'])
    before = deepcopy(raw)
    draft = p['draft']
    selected = set(p['read_packet']['selected_evidence_ids'])
    evidence = [e for e in result['evidence'] if e['evidence_id'] in selected]
    question = result['run']['question']
    normalized = isolate_disclosure_policy_metadata(raw, draft, question=question)
    validated = validate_audit(normalized, draft, evidence, spans=p['source_spans'],
                               require_relevance=True, question=question)
    assert validated['coverage'] == 'COMPLETE'
    assert validated['assessments'] == raw['assessments']
    assert validated['withheld_limitation_indices'] == list(range(8))
    assert audited_presentation(draft, validated, question=question)['answer_scope'] == 'MODEL_REVIEWED'
    assert raw == before
