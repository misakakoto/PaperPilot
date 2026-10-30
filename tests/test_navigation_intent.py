import json

import pytest

from paperpilot_agentic.navigation_intent import navigation_intent
from paperpilot_agentic.reader_recovery import plan_evidence_reload
from paperpilot_agentic.section_router import select_sections
from paperpilot_agentic.providers import LLMResponse


def test_intent_copies_full_compiled_state_without_promoting_it_to_evidence():
    text = json.dumps({'active_task': 'A', 'current_requirement': 'Use validation',
                       'historical_aliases': ['test'], 'paused_tasks': ['B']})
    history = [{'role': 'derived_request_state', 'user_request': text}]
    value = navigation_intent('Query', original_question='Continue on validation.', request_history=history)
    assert value == {'question': 'Query', 'original_question': 'Continue on validation.',
                     'request_history': history}
    history[0]['user_request'] = 'Changed by caller'
    assert value['request_history'][0]['user_request'] == text
    assert 'evidence' not in value


@pytest.mark.parametrize('history', ['not a list', [None], [{'user_request': 5}]])
def test_invalid_request_context_fails_before_model_billing(history):
    with pytest.raises(ValueError, match='owned user requests'):
        plan_evidence_reload('Question', {'limitations': ['Gap']}, [], set(), None,
                             request_history=history)
    result = select_sections('Question', [{'section_id': 'x', 'title': 'Setup', 'chars': 5}],
                             None, request_history=history)
    assert result['status'] == 'FAILED' and result['provider_attempts'] == 0


def test_recovery_format_repair_preserves_the_original_request_and_history():
    payloads = []
    class Model:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            payloads.append(json.loads(messages[1]['content']))
            value = {'action': 'RELOAD', 'evidence_ids': ['invented'], 'reason': 'Invalid ID'}
            if len(payloads) == 2:
                value = {'action': 'STOP', 'reason': 'Keep source scope fixed'}
            return LLMResponse(text=json.dumps(value), finish_reason='stop')

        def for_repair(self):
            return self

    history = [{'role': 'initial_request', 'user_request': 'Compare precision and speed.'}]
    result = plan_evidence_reload('Compare methods', {'limitations': ['Missing validation result']},
        [], set(), Model(), original_question='Use validation instead of test.', request_history=history)
    assert result['action'] == 'STOP'
    assert len(payloads) == 2 and payloads[0] == payloads[1]
    assert payloads[1]['original_question'] == 'Use validation instead of test.'
    assert payloads[1]['request_history'] == history


@pytest.mark.parametrize('stage', ['section', 'recovery'])
def test_navigation_request_growth_is_checked_before_billing(tmp_path, stage):
    from paperpilot_agentic.db import RuntimeDatabase
    from paperpilot_agentic.durable_reader import DurableResearchModel
    from paperpilot_agentic.llm_runtime import PersistentLLMRuntime, LLMBudgetExhausted
    from paperpilot_agentic.models import Budget
    class NeverCalled:
        model = 'fixture'
        def complete(self, *args, **kwargs):
            pytest.fail('Oversized intent must not reach provider')
    db = RuntimeDatabase(tmp_path / 'runtime.sqlite3')
    rid = db.create_run('Question', Budget(max_tokens=10000))
    model = DurableResearchModel(PersistentLLMRuntime(db, NeverCalled()), rid, stage=stage)
    history = [{'role': 'initial_request', 'user_request': 'Historical requirement. ' * 10000}]
    if stage == 'section':
        result = select_sections('Question', [{'section_id': 'x', 'title': 'Setup', 'chars': 5}],
                                 model, request_history=history)
        assert result['status'] == 'FAILED' and result['error_type'] == 'LLMBudgetExhausted'
    else:
        with pytest.raises(LLMBudgetExhausted):
            plan_evidence_reload('Question', {'limitations': ['Missing data']}, [], set(), model,
                                 request_history=history)
    assert db.get_run(rid)['used_llm_calls'] == 0
