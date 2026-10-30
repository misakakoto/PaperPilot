"""Diagnostic bookkeeping tests, not model-quality measurements."""
import json
from copy import deepcopy

import pytest

from evals.reliability.relation_policy_probe import run, synthetic_cases, RELATION_POLICY
from paperpilot_agentic.providers import LLMResponse


class ApprovesEverything:
    def __init__(self, fail=False):
        self.messages, self.fail = [], fail

    def complete(self, messages, **kwargs):
        kwargs['before_attempt']()
        self.messages.append(messages)
        if self.fail:
            raise RuntimeError('SECRET_TOKEN')
        payload = json.loads(messages[-1]['content'])
        value = {'assessments': [{'fact_id': fact['id'], 'verdict': 'SUPPORTED',
                 'span_ids': [fact['allowed_span_ids'][0]], 'relevance': 'DIRECT'}
                 for fact in payload['facts']], 'coverage': 'COMPLETE', 'missing_aspects': []}
        return LLMResponse(text=json.dumps(value), finish_reason='stop', prompt_tokens=5, completion_tokens=5)


def plan():
    return {'cases': synthetic_cases(), 'policy': RELATION_POLICY, 'planned_requests': 8,
            'scoring': 'synthetic labels', 'scope': 'audit-only'}


def test_pairs_keep_inputs_and_hide_labels_and_never_overwrite(tmp_path):
    frozen = plan()
    original = deepcopy(frozen)
    model = ApprovesEverything()
    summary = run(frozen, model, tmp_path/'experiment')
    assert frozen == original and summary['attempts'] == 8 and summary['known_tokens'] == 80
    assert summary['metrics']['baseline']['false_approvals'] == 4
    assert summary['metrics']['relation']['false_approvals'] == 4
    for index in range(0, 8, 2):
        first, second = model.messages[index:index+2]
        assert first[-1] == second[-1]
        baseline, treatment = sorted((first[0]['content'], second[0]['content']), key=len)
        assert treatment == baseline + '\n' + RELATION_POLICY
    assert all('"expected"' not in json.dumps(messages) for messages in model.messages)
    with pytest.raises(FileExistsError):
        run(frozen, model, tmp_path/'experiment')
    assert len(model.messages) == 8


def test_failed_requests_are_not_counted_as_correct_rejections_and_errors_are_redacted(tmp_path):
    summary = run(plan(), ApprovesEverything(fail=True), tmp_path/'failed')
    assert summary['attempts'] == 8
    assert summary['metrics']['baseline']['correct'] == 0
    assert summary['metrics']['baseline']['failed'] == 8
    assert summary['metrics']['baseline']['false_approval_rate'] is None
    assert all('SECRET_TOKEN' not in file.read_text(encoding='utf-8') for file in (tmp_path/'failed').glob('*.json'))


def test_scope_pairs_swap_only_question_and_do_not_count_failures_as_correct(tmp_path):
    from evals.reliability.answer_role_probe import freeze
    frozen = freeze()
    for absent, direct in zip(frozen['cases'][::2], frozen['cases'][1::2]):
        assert absent['draft'] == direct['draft'] and absent['evidence'] == direct['evidence']
        assert absent['question'] != direct['question']
        assert absent['expected_direct'] == {'f1': False}
        assert direct['expected_direct'] == {'f1': True}
    model = ApprovesEverything()
    summary = run(frozen, model, tmp_path/'scope')
    assert summary['attempts'] == 16
    for metrics in summary['direct_answer_metrics'].values():
        assert metrics == dict(total=8, correct=4, failed=0, false_direct=4, missed_direct=0,
                               expected_direct=4, expected_not_direct=4)
    assert all('source-only entailment' in messages[0]['content'] for messages in model.messages)
    assert all('expected_direct' not in json.dumps(messages) for messages in model.messages)
    failed = run(frozen, ApprovesEverything(fail=True), tmp_path/'failed-scope')
    assert all(m['failed'] == 8 and m['correct'] == 0 for m in failed['direct_answer_metrics'].values())
