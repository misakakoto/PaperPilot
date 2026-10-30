"""Research tasks are exercised through the conversation-facing interface."""
import json

import pytest

from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.workbench import ResearchWorkbench
from paperpilot_agentic.providers import LLMResponse
from test_workbench import Background


class Library:
    def paper_metadata(self, hashes):
        return [{'sha256': sha, 'title': 'Paper ' + sha[-3:], 'file_name': sha}
                for sha in sorted(hashes)]


class Search:
    store = Library()

    def search(self, question, top_k=8, *, paper_sha256s=None):
        return [{'stable_id': sha, 'paper_sha256': sha, 'page': 1,
                 'source_uri': f'paper://{sha}/original', 'content': 'A source fact.'}
                for sha in sorted(paper_sha256s)]


class ResearchProvider:
    model = 'fixture-research'

    def __init__(self):
        self.calls = 0

    def complete(self, messages, *, before_attempt, **kwargs):
        before_attempt()
        self.calls += 1
        payload = json.loads(messages[-1]['content'])
        if 'capabilities' in payload:
            value = {'action': 'research', 'goal': 'Compare originals',
                     'research_intent': {'kind': 'persistent_notebook', 'request_span': payload['request']},
                     'steps': ['Read papers', 'Compare facts'], 'gaps': [], 'dimensions': ['Reported facts'],
                     'reader_id': None, 'source_ids': [], 'validation_id': None}
        elif 'evidence' in payload:
            facts = [{'id': f'f{i}', 'text': 'A source fact.', 'evidence_ids': [e['evidence_id']]}
                     for i, e in enumerate(payload['evidence'])]
            value = {'facts': facts, 'direct_answer_id': 'f0', 'limitations': []}
            if 'comparison_fields' in payload:
                for fact in facts:
                    fact.update(comparison_field='module', comparison_value='A source fact.',
                                text=payload['comparison_fields']['module'] + '：A source fact.')
                value['comparison_records'] = [{'record_id': f'r{i}', 'fields': {'module': fact['id']},
                    'missing_fields': [k for k in payload['comparison_fields'] if k != 'module']}
                    for i, fact in enumerate(facts)]
        else:
            value = {'assessments': [{'fact_id': f['id'], 'verdict': 'SUPPORTED',
                'relevance': 'DIRECT', 'span_ids': [f['allowed_span_ids'][0]]}
                for f in payload['facts']], 'coverage': 'COMPLETE', 'missing_aspects': []}
            if 'comparison_records' in payload:
                value['record_assessments'] = [{'record_id': row['record_id'], 'verdict': 'SUPPORTED',
                                               'reason': 'Fixture joint attribution'}
                                              for row in payload['comparison_records']]
        return LLMResponse(text=json.dumps(value), prompt_tokens=20, completion_tokens=10,
                           finish_reason='stop', model=self.model)


class NoProvider:
    def complete(self, *args, **kwargs):
        raise AssertionError('Budget confirmation must precede bulk model calls')


def test_route_can_omit_display_only_steps_and_gaps_without_gaining_authority():
    from paperpilot_agentic.session_router import validate_route
    catalog = {'readers': [], 'validations': []}
    value = validate_route({'action': 'research', 'goal': 'List datasets',
                            'dimensions': ['Datasets'], 'source_ids': ['untrusted-model-id']}, catalog)
    assert value['steps'] == value['gaps'] == []
    with pytest.raises(ValueError, match='owned'):
        validate_route({'action': 'design', 'goal': 'Design a module'}, catalog)
    with pytest.raises(ValueError, match='authorize'):
        validate_route({'action': 'execute', 'goal': 'Train'}, catalog)


def workbench(tmp_path, provider=None, *, comparison=True):
    runtime = ResearchAgentRuntime(db_path=tmp_path/'runtime.db', workspace=tmp_path,
        search_adapter=Search(), llm=provider or NoProvider(),
        text_reader_protocol='research-task-fixture')
    class ComparisonWorkbench(ResearchWorkbench):
        def submit(self, payload):
            # These fixtures explicitly exercise the evidence-table artifact,
            # not a replacement for every ordinary research request.
            if payload.get('action') in {'auto', 'notebook'} and payload.get('paper_ids'):
                payload = {**payload, 'action': 'evidence_comparison'}
            return super().submit(payload)
    return (ComparisonWorkbench if comparison else ResearchWorkbench)(runtime, Background())


def test_ordinary_research_keeps_section_reports_and_comparison_is_explicit(tmp_path):
    fixture = workbench(tmp_path)
    wb = ResearchWorkbench(fixture.runtime, Background())
    root = wb.submit({'request_id': 'ordinary-section-plan', 'action': 'auto',
        'question': 'Summarize all selected papers.', 'paper_ids': [f'{i:064x}' for i in range(20)]})
    assert wb.turn(root['run_id'])['research_task']['report_format'] == 'section_report_v1'
    compare = wb.submit({'request_id': 'explicit-comparison-plan', 'action': 'evidence_comparison',
        'question': 'Compare experimental conditions.', 'paper_ids': [f'{1:064x}']})
    assert compare['start'] is False
    assert wb.turn(compare['run_id'])['research_task']['report_format'] == 'evidence_comparison_v1'


def test_twenty_selected_papers_offer_budget_before_retrieval(tmp_path):
    wb = workbench(tmp_path)
    result = wb.submit({'request_id': 'twenty-paper-request', 'action': 'auto',
        'question': '请逐篇比较这20篇论文的方法和实验条件',
        'paper_ids': [f'{i:064x}' for i in range(1, 21)]})
    view = wb.turn(result['run_id'])
    assert view['status'] == 'WAITING_APPROVAL'
    assert view['research_task']['state'] == 'AWAITING_BUDGET'
    assert len(view['research_task']['coverage']) == 20
    assert view['budget']['used_llm_calls'] == 0
    assert not wb.background.submitted


def test_budget_confirmation_is_owned_versioned_and_idempotent(tmp_path):
    wb = workbench(tmp_path)
    created = wb.submit({'request_id': 'budget-plan-request', 'action': 'auto',
        'question': 'Compare all methods', 'paper_ids': [f'{i:064x}' for i in range(20)]})
    task = wb.turn(created['run_id'])['research_task']
    payload = {'request_id': 'approve-budget-request', 'action': 'approve_research',
        'conversation_id': created['conversation_id'], 'parent_run_id': created['run_id'],
        'plan_revision': task['plan_revision'], 'budget': task['suggested_budget']}
    with pytest.raises(ValueError, match='another conversation'):
        wb.submit({**payload, 'request_id': 'foreign-budget-request', 'conversation_id': 'foreign'})
    with pytest.raises(ValueError, match='changed'):
        wb.submit({**payload, 'request_id': 'stale-budget-request', 'plan_revision': 'stale'})
    approved = wb.submit(payload)
    assert approved['run_id'] == created['run_id']
    assert wb.turn(approved['run_id'])['research_task']['state'] == 'READY'
    assert wb.submit(payload)['reused'] is True
    assert wb.background.submitted == [created['run_id']]
    assert len(wb.runtime.list_runs()) == 1


def test_approved_twenty_paper_task_completes_and_reuses_saved_report(tmp_path):
    provider = ResearchProvider()
    wb = workbench(tmp_path, provider)
    root = wb.submit({'request_id': 'execute-twenty-request', 'action': 'auto',
        'question': 'Compare the reported facts for every selected paper.',
        'paper_ids': [f'{i:064x}' for i in range(1, 21)]})
    plan = wb.turn(root['run_id'])['research_task']
    wb.submit({'request_id': 'execute-twenty-approval', 'action': 'approve_research',
        'conversation_id': root['conversation_id'], 'parent_run_id': root['run_id'],
        'plan_revision': plan['plan_revision'], 'budget': plan['suggested_budget']})
    result = wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert result['run']['status'] == 'COMPLETED'
    assert all(row['state'] == 'READ' for row in view['research_task']['coverage'])
    assert view['message']['answer']
    assert len(view['message']['citations']) == 20
    before = provider.calls
    reopened = workbench(tmp_path, provider)
    reopened.runtime.resume(root['run_id'])
    assert provider.calls == before
    assert reopened.turn(root['run_id'])['message']['answer'] == view['message']['answer']


def test_open_research_request_without_old_keywords_reaches_structured_route(tmp_path):
    from test_research_workbench_state import RouterModel
    wb = workbench(tmp_path, RouterModel())
    result = wb.submit({'request_id': 'semantic-route-request', 'action': 'auto',
                        'question': '请思考一种新的变化检测方法，并检验是否有效。'})
    wb.runtime.resume(result['run_id'])
    view = wb.turn(result['run_id'])
    assert view['budget']['used_llm_calls'] == 1
    assert view['message']['limitations'] == ['Need a reviewed protocol']
    assert not view['approvals']


def test_budget_pause_continues_same_root_without_repeating_completed_reads(tmp_path):
    wb = workbench(tmp_path, ResearchProvider())
    root = wb.submit({'request_id': 'pause-plan-request', 'action': 'auto',
        'question': 'Compare all facts', 'paper_ids': [f'{i:064x}' for i in range(20)]})
    plan = wb.turn(root['run_id'])['research_task']
    wb.submit({'request_id': 'small-budget-approval', 'action': 'approve_research',
        'conversation_id': root['conversation_id'], 'parent_run_id': root['run_id'],
        'plan_revision': plan['plan_revision'], 'budget': {**plan['suggested_budget'], 'max_tool_calls': 3}})
    wb.runtime.resume(root['run_id'])
    paused = wb.turn(root['run_id'])['research_task']
    assert paused['state'] == 'PAUSED_BUDGET'
    assert paused['usage']['used_tool_calls'] == 3
    wb.background.submitted.clear()
    wb.submit({'request_id': 'extend-budget-approval', 'action': 'continue_research',
        'conversation_id': root['conversation_id'], 'parent_run_id': root['run_id'],
        'plan_revision': paused['plan_revision'], 'budget': plan['suggested_budget']})
    wb.runtime.resume(root['run_id'])
    final = wb.turn(root['run_id'])
    assert final['status'] == 'COMPLETED'
    assert final['research_task']['usage']['used_tool_calls'] == 20
    assert len(wb.runtime.list_runs()) == 1


def test_unknown_paid_request_is_never_replayed(tmp_path):
    class LostResponse(ResearchProvider):
        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            self.calls += 1
            raise TimeoutError('response lost')
    provider = LostResponse()
    wb = workbench(tmp_path, provider)
    root = wb.submit({'request_id': 'unknown-plan-request', 'action': 'auto',
        'question': 'Compare all facts', 'paper_ids': [f'{i:064x}' for i in range(20)]})
    plan = wb.turn(root['run_id'])['research_task']
    wb.submit({'request_id': 'unknown-budget-approval', 'action': 'approve_research',
        'conversation_id': root['conversation_id'], 'parent_run_id': root['run_id'],
        'plan_revision': plan['plan_revision'], 'budget': plan['suggested_budget']})
    wb.runtime.resume(root['run_id'])
    assert wb.turn(root['run_id'])['research_task']['state'] == 'UNKNOWN'
    workbench(tmp_path, provider).runtime.resume(root['run_id'])
    assert provider.calls == 1


@pytest.mark.parametrize('count', [2, 20])
def test_paid_writer_is_reused_when_budget_pauses_before_audit(tmp_path, count):
    provider = ResearchProvider()
    wb = workbench(tmp_path, provider)
    root = wb.submit({'request_id': 'writer-cache-request', 'action': 'auto',
        'question': 'Compare all facts', 'paper_ids': [f'{i:064x}' for i in range(count)]})
    plan = wb.turn(root['run_id'])['research_task']
    payload = {'conversation_id': root['conversation_id'], 'parent_run_id': root['run_id']}
    wb.submit({**payload, 'request_id': 'writer-small-budget', 'action': 'approve_research',
        'plan_revision': plan['plan_revision'], 'budget': {**plan['suggested_budget'], 'max_llm_calls': 2}})
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert view['research_task']['state'] == 'PAUSED_BUDGET'
    assert provider.calls == 1  # reserved repair slot remains; Auditor not sent
    assert not view['message']['answer']  # no unreviewed Writer draft published
    reopened = workbench(tmp_path, provider)
    reopened.submit({**payload, 'request_id': 'writer-extend-budget', 'action': 'continue_research',
        'plan_revision': view['research_task']['plan_revision'], 'budget': plan['suggested_budget']})
    reopened.runtime.resume(root['run_id'])
    assert reopened.turn(root['run_id'])['status'] == 'COMPLETED'
    assert provider.calls == count * 2  # one per-paper pair; the saved Writer is not repeated


def test_original_source_exclusion_is_applied_before_budget_quote(tmp_path):
    wb = workbench(tmp_path)
    wb.runtime.search_adapter.resolve_explicit_title_scope = lambda queries, **kwargs: (
        {f'{1:064x}'} if any('Alpha' in q for q in queries) else set())
    root = wb.submit({'request_id': 'scope-exclusion-request', 'action': 'auto',
        'question': 'Compare all selected papers, excluding Alpha',
        'paper_ids': [f'{i:064x}' for i in range(1, 21)]})
    assert len(wb.turn(root['run_id'])['research_task']['coverage']) == 19


def test_notebook_entry_requires_the_same_total_budget(tmp_path):
    wb = workbench(tmp_path)
    root = wb.submit({'request_id': 'inventory-budget-request', 'action': 'notebook',
        'question': 'List the datasets for each selected paper.', 'paper_ids': [f'{1:064x}']})
    assert root['start'] is False
    assert wb.turn(root['run_id'])['research_task']['state'] == 'AWAITING_BUDGET'


def test_cancel_before_worker_start_can_continue_without_resetting_usage(tmp_path):
    wb = workbench(tmp_path, ResearchProvider())
    root = wb.submit({'request_id': 'cancel-plan-request', 'action': 'auto',
        'question': 'Compare all facts', 'paper_ids': [f'{i:064x}' for i in range(20)]})
    plan = wb.turn(root['run_id'])['research_task']
    payload = {'action': 'approve_research', 'conversation_id': root['conversation_id'],
        'parent_run_id': root['run_id'], 'plan_revision': plan['plan_revision'], 'budget': plan['suggested_budget']}
    wb.submit({**payload, 'request_id': 'cancel-budget-request'})
    wb.runtime.cancel(root['run_id'])
    paused = wb.turn(root['run_id'])['research_task']
    assert paused['state'] == 'PAUSED_CANCELLED'
    wb.background.submitted.clear()
    wb.submit({**payload, 'request_id': 'cancel-continue-request', 'action': 'continue_research',
               'plan_revision': paused['plan_revision']})
    wb.runtime.resume(root['run_id'])
    assert wb.turn(root['run_id'])['status'] == 'COMPLETED'


def test_changed_protocol_cannot_resume_an_approved_task(tmp_path):
    provider = ResearchProvider()
    wb = workbench(tmp_path, provider)
    root = wb.submit({'request_id': 'protocol-plan-request', 'action': 'auto',
        'question': 'Compare all facts', 'paper_ids': [f'{i:064x}' for i in range(20)]})
    plan = wb.turn(root['run_id'])['research_task']
    wb.submit({'request_id': 'protocol-budget-approval', 'action': 'approve_research',
        'conversation_id': root['conversation_id'], 'parent_run_id': root['run_id'],
        'plan_revision': plan['plan_revision'], 'budget': plan['suggested_budget']})
    wb.runtime.text_reader_protocol = 'changed-model-or-code'
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert view['status'] == 'FAILED'
    assert provider.calls == 0
    assert 'protocol' in view['research_task']['pause_reason'].lower()


def test_router_accepts_only_the_known_json_transport_marker():
    from paperpilot_agentic.session_router import validate_route
    route = {'action': 'research', 'goal': 'List datasets', 'steps': [], 'gaps': [],
             'reader_id': None, 'source_ids': [], 'validation_id': None, 'type': 'json_object'}
    assert validate_route(route, {})['action'] == 'research'
    with pytest.raises(ValueError):
        validate_route({**route, 'type': 'execute'}, {})
    with pytest.raises(ValueError):
        validate_route({**route, 'authorize_training': True}, {})


def test_read_route_does_not_require_irrelevant_experiment_identifiers():
    from paperpilot_agentic.session_router import validate_route
    # Independently reproduced with ChangeMamba, not a held-out QASPER answer.
    route = {'action': 'read', 'goal': 'Identify datasets', 'steps': ['Read originals'],
             'gaps': [], 'dimensions': [], 'reader_id': None, 'source_ids': []}
    assert validate_route(route, {})['validation_id'] is None
    with pytest.raises(ValueError):
        validate_route({**route, 'action': 'design'}, {'readers': [], 'validations': []})
    with pytest.raises(ValueError):
        validate_route({**route, 'action': 'execute'}, {})


def test_explicit_billing_rejection_is_not_mislabeled_as_lost_response(tmp_path):
    class BillingError(RuntimeError):
        status_code = 402
    class EmptyBalance(ResearchProvider):
        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            self.calls += 1
            raise BillingError('No balance')
    provider = EmptyBalance()
    wb = workbench(tmp_path, provider)
    root = wb.submit({'request_id': 'billing-plan-request', 'action': 'auto',
        'question': 'Compare all facts', 'paper_ids': [f'{i:064x}' for i in range(20)]})
    plan = wb.turn(root['run_id'])['research_task']
    wb.submit({'request_id': 'billing-budget-approval', 'action': 'approve_research',
        'conversation_id': root['conversation_id'], 'parent_run_id': root['run_id'],
        'plan_revision': plan['plan_revision'], 'budget': plan['suggested_budget']})
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert view['status'] == 'FAILED'
    assert '402' in view['research_task']['pause_reason']
    assert view['budget']['used_llm_calls'] == 1
    workbench(tmp_path, provider).runtime.resume(root['run_id'])
    assert provider.calls == 1


@pytest.mark.parametrize('count', [1, 5, 50])
def test_scale_keeps_all_selected_sources_in_bounded_reports(tmp_path, count):
    provider = ResearchProvider()
    wb = workbench(tmp_path, provider)
    root = wb.submit({'request_id': f'scale-{count}-request', 'action': 'auto',
        'question': 'Maintain a research notebook comparing all reported facts.',
        'paper_ids': [f'{i:064x}' for i in range(count)]})
    if root['start']:
        wb.runtime.resume(root['run_id'])
        wb.background.submitted.clear()
    plan = wb.turn(root['run_id'])['research_task']
    assert len(plan['coverage']) == count
    wb.submit({'request_id': f'scale-{count}-approve', 'action': 'approve_research',
        'conversation_id': root['conversation_id'], 'parent_run_id': root['run_id'],
        'plan_revision': plan['plan_revision'], 'budget': plan['suggested_budget']})
    wb.runtime.resume(root['run_id'])
    final = wb.turn(root['run_id'])
    assert final['status'] == 'COMPLETED'
    assert len(final['message']['citations']) == count
