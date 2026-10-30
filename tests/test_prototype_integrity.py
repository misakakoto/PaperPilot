import copy

import pytest

from paperpilot_agentic.episode_runner import ResearchStopped
from paperpilot_agentic.models import ToolCallStatus
from paperpilot_agentic.prototype_workspace import plan, complete
from test_prototype_workspace import waiting, tasks


def no_execution(*args):
    pytest.fail('Candidate execution must not start')


def test_planning_must_be_followed_by_work_and_repetition_stops(tmp_path, monkeypatch):
    import json
    from test_prototype_workspace import Reader
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from paperpilot_agentic.prototype_harness import PrototypeHarness
    from paperpilot_agentic.providers import LLMResponse

    class Planner(Reader):
        calls = 0

        def complete(self, messages, **kwargs):
            self.calls += 1
            if self.calls > 1:
                self.last_available = json.loads(messages[-1]['content']).get('available_actions', [])
            return LLMResponse(text=json.dumps({'action': 'plan', 'reason': 'repeat',
                                               'arguments': {'tasks': tasks()}}),
                               model=self.model, finish_reason='stop', prompt_tokens=20, completion_tokens=20)

    reader = Planner()
    runtime = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path/'runtime.db', llm=reader)
    monkeypatch.setattr(runtime.db, 'list_evidence', lambda _: [{'evidence_id': 'source', 'content': 'source'}])
    h = PrototypeHarness(runtime)
    rid = h.start('build', 'repeated-planning-fixture', 'parent', ['source'])['run']['run_id']
    h.advance(rid)
    assert reader.calls <= 4
    assert h._load(rid)['phase'] == 'PARTIAL'
    assert 'progress' in h._load(rid)['stop_reason']


def test_invalid_generated_source_returns_actionable_validation_error():
    from paperpilot_agentic.prototype_workspace import check_code
    with pytest.raises(ValueError, match='line 2'):
        check_code('def build_model(config):\n    return (\n')


def test_syntax_rejection_can_be_repaired_in_same_episode(tmp_path, monkeypatch):
    from test_prototype_workspace import stub_environment
    stub_environment(monkeypatch)
    import json
    from test_prototype_workspace import Reader
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from paperpilot_agentic.prototype_harness import PrototypeHarness
    from paperpilot_agentic.providers import LLMResponse

    class SyntaxReader(Reader):
        rejected = False
        saw_feedback = False

        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]['content'])
            response = super().complete(messages, **kwargs)
            action = json.loads(response.text)
            if action.get('arguments', {}).get('task_id') == 'code':
                if not self.rejected:
                    action['arguments']['content']['source'] = 'def build_model(config):\n return (\n'
                    self.rejected = True
                else:
                    self.saw_feedback = 'line 2' in payload['implementation_feedback']['error']
            return LLMResponse(text=json.dumps(action), model=self.model, finish_reason='stop',
                               prompt_tokens=20, completion_tokens=20)

    reader = SyntaxReader()
    runtime = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path/'runtime.db', llm=reader)
    monkeypatch.setattr(runtime.db, 'list_evidence', lambda _: [{'evidence_id': 'source', 'content': 'source'}])
    harness = PrototypeHarness(runtime)
    rid = harness.start('build', 'syntax-repair-fixture', 'parent', ['source'])['run']['run_id']
    harness.advance(rid)
    state = harness._load(rid)
    assert state['phase'] == 'WAITING_APPROVAL'
    assert reader.saw_feedback and state['work']['repairs'] == 2


def test_failed_execution_cannot_be_automatically_retried(tmp_path, monkeypatch):
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    state = harness._load(rid)
    runtime.approve(state['pending']['approval'])
    runtime.db.update_tool_call(state['pending']['call'], ToolCallStatus.FAILED)
    monkeypatch.setattr('paperpilot_agentic.prototype_harness.SandboxExperimentRunner.run', no_execution)
    with pytest.raises(ResearchStopped):
        harness._training(rid, state)


def test_approval_must_match_execution_binding(tmp_path, monkeypatch):
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    state = harness._load(rid)
    runtime.approve(state['pending']['approval'])
    approval = copy.deepcopy(runtime.db.get_approval(state['pending']['approval']))
    approval['request']['arguments']['code'] = 'different candidate'
    monkeypatch.setattr(runtime.db, 'get_approval', lambda _: approval)
    monkeypatch.setattr('paperpilot_agentic.prototype_harness.SandboxExperimentRunner.run', no_execution)
    with pytest.raises(ValueError, match='approval binding'):
        harness._training(rid, state)


def test_completion_requires_validation_of_every_current_candidate(tmp_path, monkeypatch):
    _, harness, rid = waiting(tmp_path, monkeypatch)
    state = harness._load(rid)
    work = state['work']
    rows = tasks() + [
        {'id': 'review', 'goal': 'review', 'capability': 'review',
         'depends_on': ['design', 'code', 'check'], 'acceptance': 'consistent'},
        {'id': 'unvalidated', 'goal': 'second candidate', 'capability': 'implement',
         'depends_on': ['design'], 'acceptance': 'must validate'}]
    plan(work, rows)
    complete(work, 'check', 'validate', {'passed': True})
    complete(work, 'review', 'review', {'implementation_consistent': True, 'issues': [], 'limitations': []})
    complete(work, 'unvalidated', 'implement', {'source': 'def build_model(c): return None', 'config': {}})
    harness._apply(rid, state, {'action': 'finish', 'reason': 'done', 'arguments': {'summary': 'done'}}, 7)
    assert state['completed'] is False


def test_repair_context_retains_exact_code_and_error_after_replan(tmp_path, monkeypatch):
    import json
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    state = harness._load(rid)
    code_id = state['work']['tasks']['code']['output']
    runtime.approve(state['pending']['approval'])
    runtime.db.update_tool_call(state['pending']['call'], ToolCallStatus.COMPLETED,
                               result={'ok': True, 'output': {'passed': False, 'error': 'shape mismatch'}})
    harness._training(rid, state)
    revised = tasks()
    revised[2]['goal'] = 'fix shape mismatch'
    harness._apply(rid, state, {'action': 'plan', 'reason': 'repair', 'arguments': {'tasks': revised}}, 7)
    state['last_observation'] = {'status': 'OK', 'action': 'plan'}
    _, messages = harness._context(rid, state)
    payload = json.loads(messages[-1]['content'])
    assert payload['repair_contexts']['code']['previous_artifact_id'] == code_id
    assert payload['validation_feedback']['result']['error'] == 'shape mismatch'


def test_validation_cannot_cover_two_candidates_at_once(tmp_path, monkeypatch):
    _, harness, rid = waiting(tmp_path, monkeypatch)
    state = harness._load(rid)
    rows = tasks() + [{'id': 'other', 'goal': 'other', 'capability': 'implement',
                       'depends_on': ['design'], 'acceptance': 'checked'}]
    rows[3]['depends_on'] = ['code', 'other']
    plan(state['work'], rows)
    complete(state['work'], 'other', 'implement', {'source': 'def build_model(c): return None', 'config': {}})
    with pytest.raises(ValueError, match='exactly one candidate'):
        harness._apply(rid, state, {'action': 'work', 'reason': 'validate',
                                  'arguments': {'task_id': 'check', 'content': None}}, 7)
