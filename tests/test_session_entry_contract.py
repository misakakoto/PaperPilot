"""Ordinary questions and research requests cross the same workbench interface."""
import json

import pytest

from paperpilot_agentic.providers import LLMResponse
from test_research_task import Library, ResearchProvider, workbench


class EntryLibrary(Library):
    def paper_outline(self, sha):
        return []

    def get_chunks(self, ids):
        return []


def entry_workbench(tmp_path, provider):
    wb = workbench(tmp_path, provider, comparison=False)
    wb.runtime.search_adapter.store = EntryLibrary()
    return wb


class EntryProvider(ResearchProvider):
    def __init__(self, route):
        super().__init__()
        self.route = route

    def complete(self, messages, *, before_attempt, **kwargs):
        payload = json.loads(messages[-1]['content'])
        if 'capabilities' not in payload:
            return super().complete(messages, before_attempt=before_attempt, **kwargs)
        before_attempt()
        self.calls += 1
        return LLMResponse(text=json.dumps(self.route), prompt_tokens=20, completion_tokens=10,
                           finish_reason='stop', model=self.model)


@pytest.mark.parametrize('optional_goal', [{}, {'goal': None}, {'goal': '  '}])
def test_ordinary_read_survives_omitted_display_goal(tmp_path, optional_goal):
    model = EntryProvider({'action': 'read', 'steps': ['Read the source'], **optional_goal})
    wb = entry_workbench(tmp_path, model)
    created = wb.submit({'request_id': 'read-missing-display-goal', 'action': 'auto',
        'question': 'What fact does this paper report?', 'paper_ids': [f'{1:064x}']})
    wb.runtime.resume(created['run_id'])
    view = wb.turn(created['run_id'])
    assert view['status'] == 'COMPLETED', view['error']
    assert 'A source fact.' in view['message']['answer']
    assert view['research_task'] is None
    assert not view['approvals']
    paid_calls = model.calls
    reopened = entry_workbench(tmp_path, model)
    reopened.runtime.resume(created['run_id'])
    assert model.calls == paid_calls
    assert reopened.turn(created['run_id'])['message'] == view['message']


@pytest.mark.parametrize('action', ['research', 'notebook'])
@pytest.mark.parametrize('paper_count', [1, 2])
def test_bounded_question_does_not_require_budget_for_unsubstantiated_research_route(tmp_path, action, paper_count):
    wb = entry_workbench(tmp_path, EntryProvider({'action': action, 'goal': 'List the facts',
        'dimensions': ['Facts']}))
    root = wb.submit({'request_id': 'bounded-question-route', 'action': 'auto',
        'question': 'List the reported source facts.',
        'paper_ids': [f'{i:064x}' for i in range(1, paper_count + 1)]})
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert view['status'] == 'COMPLETED', view['error']
    assert 'A source fact.' in view['message']['answer']
    assert view['research_task'] is None
    decision = wb.runtime.status(root['run_id'])['run']['metadata']['research_route_decision']
    assert decision['proposed_action'] == action
    assert decision['effective_action'] == 'read'


@pytest.mark.parametrize('action,kind,paper_count,question', [
    ('research', 'paperwise_review', 2, 'Compare every selected paper in a separate coverage table.'),
    ('research', 'research_program', 1, 'Build a research program with hypotheses and staged experiments.'),
    ('notebook', 'persistent_notebook', 1, 'Maintain a research notebook for this paper over future sessions.'),
])
def test_explicit_sustained_deliverable_still_requires_budget(tmp_path, action, kind, paper_count, question):
    model = EntryProvider({'action': action, 'research_intent': {'kind': kind, 'request_span': question}})
    wb = entry_workbench(tmp_path, model)
    root = wb.submit({'request_id': 'sustained-request-route', 'action': 'auto', 'question': question,
                     'paper_ids': [f'{i:064x}' for i in range(1, paper_count + 1)]})
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert view['status'] == 'WAITING_APPROVAL'
    assert view['research_task']['state'] == 'AWAITING_BUDGET'
    assert len(view['research_task']['coverage']) == paper_count
    assert model.calls == 1  # No per-paper reading until the user confirms the budget.
    assert not view['approvals']  # Research confirmation never creates training authority.


def test_flattened_model_intent_still_requires_grounded_budget_confirmation(tmp_path):
    question = 'Maintain a research notebook for this paper over future sessions.'
    model = EntryProvider({'action': 'notebook', 'research_intent': 'persistent_notebook',
                           'request_span': question})
    wb = entry_workbench(tmp_path, model)
    root = wb.submit({'request_id': 'flat-intent-provider', 'action': 'auto',
                     'question': question, 'paper_ids': [f'{1:064x}']})
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert view['status'] == 'WAITING_APPROVAL', view['error']
    assert view['research_task']['state'] == 'AWAITING_BUDGET'
    assert model.calls == 1
    assert not view['approvals']


def test_flattened_intent_cannot_invent_a_user_research_request(tmp_path):
    model = EntryProvider({'action': 'notebook', 'research_intent': 'persistent_notebook',
                           'request_span': 'Maintain a notebook: this is not the user request.'})
    wb = entry_workbench(tmp_path, model)
    root = wb.submit({'request_id': 'flat-intent-not-grounded', 'action': 'auto',
                     'question': 'What source fact is reported?', 'paper_ids': [f'{1:064x}']})
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert view['status'] == 'COMPLETED'
    assert view['message']['answer'] and view['research_task'] is None
    assert not view['approvals']


@pytest.mark.parametrize('route', [
    {'action': 'execute', 'research_intent': 'persistent_notebook', 'request_span': 'Run it.'},
    {'action': 'notebook', 'research_intent': {'kind': 'persistent_notebook', 'request_span': 'Run it.'},
     'request_span': 'A conflicting flat field.'},
])
def test_intent_shape_normalization_does_not_accept_execution_or_conflicting_fields(tmp_path, route):
    model = EntryProvider(route)
    wb = entry_workbench(tmp_path, model)
    root = wb.submit({'request_id': 'flat-intent-authority', 'action': 'auto',
                     'question': 'Run it.', 'paper_ids': [f'{1:064x}']})
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert view['status'] == 'FAILED'
    assert view['research_task'] is None and not view['approvals']
    assert not wb.runtime.status(root['run_id'])['tool_calls']


@pytest.mark.parametrize('kind,span', [
    ('research_program', 'Invented instruction not in the actual user question'),
    ('paperwise_review', 'List the facts in this paper.'),
])
def test_unsupported_intent_or_single_paper_inventory_remains_reading(tmp_path, kind, span):
    wb = entry_workbench(tmp_path, EntryProvider({'action': 'research',
        'research_intent': {'kind': kind, 'request_span': span}}))
    root = wb.submit({'request_id': 'invented-intent-route', 'action': 'auto',
        'question': 'List the facts in this paper.', 'paper_ids': [f'{1:064x}']})
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert view['status'] == 'COMPLETED', view['error']
    assert view['message']['answer'] and view['research_task'] is None


def test_research_without_selected_scope_requests_selection_not_implicit_library_coverage(tmp_path):
    question = 'Maintain a research notebook about all relevant papers.'
    model = EntryProvider({'action': 'notebook', 'research_intent':
        {'kind': 'persistent_notebook', 'request_span': question}})
    wb = entry_workbench(tmp_path, model)
    root = wb.submit({'request_id': 'missing-scope-route', 'action': 'auto', 'question': question})
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert view['research_task'] is None
    assert '请先在论文库选择要整理的论文。' in view['message']['limitations']
    assert model.calls == 1
    assert not wb.runtime.status(root['run_id'])['tool_calls']


@pytest.mark.parametrize('action', ['execute', 'design'])
def test_missing_display_goal_does_not_grant_execution_or_design_authority(tmp_path, action):
    model = EntryProvider({'action': action})
    wb = entry_workbench(tmp_path, model)
    root = wb.submit({'request_id': 'unauthorized-route', 'action': 'auto',
                     'question': 'Run a new experiment.', 'paper_ids': [f'{1:064x}']})
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert view['status'] == 'FAILED'
    assert not wb.runtime.status(root['run_id'])['tool_calls']
    assert not view['message']['answer'] and not view['approvals']
