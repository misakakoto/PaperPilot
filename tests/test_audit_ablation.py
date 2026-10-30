"""Synthetic fixtures: these tests verify experiments, NOT model accuracy."""
from copy import deepcopy
import json

import pytest

from paperpilot_agentic.audit_ablation import build_plan, digest, run_plan, score
from paperpilot_agentic.direct_answer import prepare_evidence
from paperpilot_agentic.providers import LLMResponse


def fixture():
    question = 'Compare the measured results of Method A.'
    evidence = [{'evidence_id': 'e1', 'content':
                 'Dataset Red, Table IV. Method A achieves accuracy 96.68% on Red.'},
                {'evidence_id': 'e2', 'content':
                 'Dataset Blue, Table V. Method A achieves accuracy 91.20% on Blue.'}]
    wrong = 'Method A achieves accuracy 96.68% on Blue.'
    right = 'Method A achieves accuracy 96.68% on Red.'
    compound = right + ' ' + wrong
    draft = {'facts': [{'id': 'f1', 'text': compound, 'evidence_ids': ['e1', 'e2']},
                       {'id': 'f2', 'text': right, 'evidence_ids': ['e1']}],
             'direct_answer_id': 'f1', 'limitations': []}
    annotations = {'input_sha256': digest({'question': question,
                   'evidence': prepare_evidence(evidence), 'draft': draft}),
                   'facts': {'f1': {'expected': 'UNSUPPORTED', 'atoms': [
                       {'id': 'a1', 'start': 0, 'end': len(right),
                        'text': right, 'expected': 'SUPPORTED'},
                       {'id': 'a2', 'start': len(right) + 1, 'end': len(compound),
                        'text': wrong, 'expected': 'UNSUPPORTED'}]},
                             'f2': {'expected': 'SUPPORTED'}}}
    return question, evidence, draft, annotations


def test_pairing_preserves_every_source_and_original_fact():
    plan = build_plan(*fixture())
    assert plan['planned_max_requests'] == 5
    assert [j['arm'] for j in plan['jobs']] == ['whole', 'individual', 'individual', 'atomic', 'atomic']
    assert plan['jobs'][1]['draft']['facts'][0] == plan['draft']['facts'][0]
    assert plan['jobs'][-1]['draft']['facts'][0]['text'].endswith('on Blue.')
    assert plan['jobs'][-1]['original_fact'] == plan['draft']['facts'][0]


@pytest.mark.parametrize('change', ['drift', 'rewrite', 'missing', 'bad_label', 'bad_offset'])
def test_invalid_annotation_fails_before_model(change):
    q, rows, draft, annotations = fixture()
    if change == 'drift':
        rows[0]['content'] += ' Added source.'
    if change == 'rewrite':
        annotations['facts']['f1']['atoms'][1]['text'] = 'Corrected result.'
    if change == 'missing':
        del annotations['facts']['f2']
    if change == 'bad_label':
        annotations['facts']['f2']['expected'] = 'PASS'
    if change == 'bad_offset':
        annotations['facts']['f1']['atoms'][0]['start'] = -1
    with pytest.raises(ValueError):
        build_plan(q, rows, draft, annotations)


class Scripted:
    def __init__(self, broken=False):
        self.payloads = []
        self.broken = broken

    def complete(self, messages, **kwargs):
        kwargs['before_attempt']()
        payload = json.loads(messages[-1]['content'])
        self.payloads.append(payload)
        if self.broken:
            return LLMResponse(text='{bad', finish_reason='stop', prompt_tokens=10)
        # Deliberately rubber-stamps wrong facts to test false-positive scoring.
        audit = {'assessments': [{'fact_id': f['id'], 'verdict': 'SUPPORTED',
                  'relevance': 'DIRECT', 'span_ids': [f['allowed_span_ids'][0]]}
                 for f in payload['facts']], 'coverage': 'COMPLETE', 'missing_aspects': []}
        return LLMResponse(text=json.dumps(audit), finish_reason='stop', prompt_tokens=10,
                           completion_tokens=2)


def test_labels_hidden_context_fixed_and_false_approvals_counted():
    model = Scripted()
    plan = build_plan(*fixture())
    before = deepcopy(plan)
    snapshots = []
    report = run_plan(plan, model, max_requests=5, on_progress=snapshots.append)
    assert plan == before
    assert report['provider_attempts'] == 5
    assert report['known_tokens'] == 60
    assert all(p['sources'] == model.payloads[0]['sources'] for p in model.payloads)
    assert all('expected' not in json.dumps(p) and 'input_sha256' not in p for p in model.payloads)
    assert report['metrics']['whole']['false_approvals'] == 1
    assert report['metrics']['atomic']['false_approval_rate'] == 1
    assert snapshots[1]['calls'][0]['status'] == 'STARTED'
    assert report['semantic_correctness'] == 'NOT_INDEPENDENTLY_REVIEWED'


def test_failures_not_repaired_not_counted_as_correct_rejection():
    report = run_plan(build_plan(*fixture()), Scripted(broken=True), max_requests=5)
    assert report['provider_attempts'] == 5
    assert report['metrics']['whole']['failed'] == 2
    assert report['metrics']['whole']['correct'] == 0
    assert report['metrics']['whole']['supported_failures'] == 1
    assert report['metrics']['whole']['false_approval_rate'] is None
    assert report['metrics']['whole']['false_rejection_rate'] is None
    assert report['status'] == 'FAILED'


def test_underbudget_is_rejected_before_partial_pair():
    model = Scripted()
    with pytest.raises(ValueError, match='complete paired plan'):
        run_plan(build_plan(*fixture()), model, max_requests=4)
    assert model.payloads == []


def test_false_rejection_measured_separately():
    metric = score([{'arm': 'whole', 'expected': {'f1': 'SUPPORTED'}, 'report': {
        'status': 'REVIEW_REQUIRED', 'audit': {'assessments': [
            {'fact_id': 'f1', 'verdict': 'UNCERTAIN'}]}}}])['whole']
    assert metric['false_rejection_rate'] == 1
    assert metric['false_approval_rate'] is None


def test_target_subset_does_not_remove_other_facts_from_whole_audit():
    q, rows, draft, annotations = fixture()
    annotations['target_fact_ids'] = ['f1']
    del annotations['facts']['f2']
    plan = build_plan(q, rows, draft, annotations)
    assert plan['jobs'][0]['draft'] == draft
    assert plan['jobs'][0]['expected'] == {'f1': 'UNSUPPORTED'}
    assert len([j for j in plan['jobs'] if j['arm'] == 'individual']) == 1


def test_edited_positive_control_is_explicit_and_cannot_replace_original():
    q, rows, draft, annotations = fixture()
    control = {'origin': 'explicitly_edited_control', 'change_description': 'Use Red instead of Blue',
               'expected': 'SUPPORTED', 'draft': {'facts': [
                   {'id': 'control1', 'text': 'Method A achieves accuracy 96.68% on Red.',
                    'evidence_ids': ['e1']}], 'direct_answer_id': 'control1'}}
    annotations['controls'] = [control]
    plan = build_plan(q, rows, draft, annotations)
    assert plan['draft'] == draft
    assert plan['jobs'][-1]['arm'] == 'edited_control'
    del control['change_description']
    with pytest.raises(ValueError, match='edit description'):
        build_plan(q, rows, draft, annotations)
