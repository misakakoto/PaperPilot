"""Known omitted original spans trigger bounded revision, not another search."""
import json
from dataclasses import replace

from paperpilot_agentic.providers import LLMResponse
from test_session_entry_contract import EntryLibrary, entry_workbench


class InventorySearch:
    store = EntryLibrary()

    def search(self, question, top_k=8, *, paper_sha256s=None):
        sha = next(iter(paper_sha256s))
        return [{'stable_id': f'inventory-{name}', 'paper_sha256': sha, 'page': i + 1,
                 'source_uri': f'paper://{sha}/page/{i + 1}',
                 'content': f'OrionModel is evaluated on {name}.'}
                for i, name in enumerate(['DeltaSet', 'BetaSet'])]


class OmissionModel:
    model = 'fixture-omitted-original'

    def __init__(self, *, correct_on_revision=True):
        self.correct_on_revision = correct_on_revision

    def complete(self, messages, *, before_attempt, **kwargs):
        before_attempt()
        payload = json.loads(messages[-1]['content'])
        if 'capabilities' in payload:
            value = {'action': 'read'}
        elif 'evidence' in payload:
            rows = (payload['evidence'] if 'prior_review_untrusted' in payload and self.correct_on_revision
                    else payload['evidence'][:1])
            facts = [{'id': f'f{i}', 'text': source['content'], 'evidence_ids': [source['evidence_id']]}
                     for i, source in enumerate(rows)]
            value = {'facts': facts, 'direct_answer_id': 'f0', 'limitations': []}
        elif 'facts' in payload:
            facts = payload['facts']
            value = {'assessments': [{'fact_id': fact['id'], 'verdict': 'SUPPORTED',
                     'relevance': 'DIRECT', 'span_ids': fact['allowed_span_ids'][:1]} for fact in facts],
                     'coverage': 'COMPLETE', 'missing_aspects': [],
                     'omitted_answer_spans': ['s1'] if len(facts) == 1 else []}
        else:
            value = {'action': 'STOP', 'evidence_ids': [], 'reason': 'No unread evidence'}
        return LLMResponse(text=json.dumps(value), finish_reason='stop', model=self.model)


def test_known_omitted_source_is_revised_and_reaudited_before_publication(tmp_path):
    wb = entry_workbench(tmp_path, OmissionModel())
    wb.runtime.search_adapter = InventorySearch()
    root = wb.submit({'request_id': 'repair-known-omission', 'action': 'auto',
        'question': 'List the datasets OrionModel evaluates on.', 'paper_ids': [f'{1:064x}']})
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert view['status'] == 'COMPLETED', view['error']
    assert 'DeltaSet' in view['message']['answer']
    assert 'BetaSet' in view['message']['answer']
    assert len(view['message']['citations']) == 2


def test_revision_cannot_silently_replace_an_inventory_with_only_the_new_item(tmp_path):
    class ForgetfulRevision(OmissionModel):
        def complete(self, messages, *, before_attempt, **kwargs):
            payload = json.loads(messages[-1]['content'])
            if 'evidence' in payload and 'prior_review_untrusted' in payload:
                before_attempt()
                source = payload['evidence'][1]
                return LLMResponse(text=json.dumps({'facts': [{'id': 'new',
                    'text': source['content'], 'evidence_ids': [source['evidence_id']]}],
                    'direct_answer_id': 'new', 'limitations': []}), finish_reason='stop')
            if 'facts' in payload and payload['facts'][0]['id'] == 'new':
                before_attempt()
                return LLMResponse(text=json.dumps({'assessments': [{'fact_id': 'new',
                    'verdict': 'SUPPORTED', 'relevance': 'DIRECT', 'span_ids': ['s1']}],
                    'coverage': 'COMPLETE', 'missing_aspects': []}), finish_reason='stop')
            return super().complete(messages, before_attempt=before_attempt, **kwargs)

    wb = entry_workbench(tmp_path, ForgetfulRevision())
    wb.runtime.search_adapter = InventorySearch()
    root = wb.submit({'request_id': 'forgot-inventory', 'action': 'auto',
        'question': 'List the datasets OrionModel evaluates on.', 'paper_ids': [f'{1:064x}']})
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    # A supported BetaSet statement does not establish a complete inventory.
    assert not (view['message']['answer_scope'] == 'MODEL_REVIEWED'
                and 'DeltaSet' not in view['message']['answer'])
    assert view['message']['limitations']


def test_conflicting_inventory_summary_is_not_published_as_supported(tmp_path):
    class ConflictingInventory(OmissionModel):
        def complete(self, messages, *, before_attempt, **kwargs):
            payload = json.loads(messages[-1]['content'])
            if 'evidence' in payload:
                before_attempt()
                return LLMResponse(text=json.dumps({'facts': [
                    {'id': 'total', 'text': 'OrionModel uses only one dataset: DeltaSet.',
                     'evidence_ids': [payload['evidence'][0]['evidence_id']]},
                    {'id': 'additional', 'text': 'OrionModel also evaluates on BetaSet.',
                     'evidence_ids': [payload['evidence'][1]['evidence_id']]}],
                    'direct_answer_id': 'total', 'limitations': []}), finish_reason='stop')
            if 'facts' in payload:
                before_attempt()
                return LLMResponse(text=json.dumps({'assessments': [
                    {'fact_id': f['id'], 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                     'span_ids': f['allowed_span_ids']} for f in payload['facts']],
                    'coverage': 'COMPLETE', 'missing_aspects': [],
                    'answer_conflicts': [{'fact_ids': ['total', 'additional'],
                        'reason': 'An exclusive one-dataset total conflicts with an additional dataset in the same scope.'}]
                    }), finish_reason='stop')
            return super().complete(messages, before_attempt=before_attempt, **kwargs)

    wb = entry_workbench(tmp_path, ConflictingInventory())
    wb.runtime.search_adapter = InventorySearch()
    root = wb.submit({'request_id': 'conflicting-inventory', 'action': 'auto',
        'question': 'List the datasets OrionModel evaluates on.', 'paper_ids': [f'{1:064x}']})
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert 'only one dataset' not in view['message']['answer']
    assert view['message']['answer_scope'] != 'MODEL_REVIEWED'
    assert view['message']['limitations']
    assert view['budget']['used_llm_calls'] <= 5  # One bounded re-audit, never a new search loop.


def test_revision_that_cannot_afford_its_audit_discloses_the_old_answer_fallback(tmp_path):
    class CostlyRevision(OmissionModel):
        def complete(self, messages, **kwargs):
            response = super().complete(messages, **kwargs)
            payload = json.loads(messages[-1]['content'])
            if 'evidence' in payload and 'prior_review_untrusted' in payload:
                # External usage receipt exhausts the real runtime's remaining
                # allowance. The next preflight, not a mocked reader, must stop.
                return replace(response, prompt_tokens=92_000, completion_tokens=100)
            return response

    wb = entry_workbench(tmp_path, CostlyRevision())
    wb.runtime.search_adapter = InventorySearch()
    root = wb.submit({'request_id': 'revision-without-audit-budget', 'action': 'auto',
        'question': 'List the datasets OrionModel evaluates on.', 'paper_ids': [f'{1:064x}'],
        'budget': {'max_tokens': 100_000}})  # Freeze the original exhaustion allowance.
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert 'DeltaSet' in view['message']['answer']
    assert 'BetaSet' not in view['message']['answer']  # New Writer text was never audited.
    assert view['budget']['used_llm_calls'] == 4  # No failed preflight billed as a request.
    assert any('budget' in text.lower() for text in view['message']['limitations'])
    before = view['budget']['used_llm_calls']
    wb.runtime.resume(root['run_id'])
    assert wb.turn(root['run_id'])['budget']['used_llm_calls'] == before


def test_budget_approved_research_section_repairs_its_known_omission(tmp_path):
    wb = entry_workbench(tmp_path, OmissionModel())
    wb.runtime.search_adapter = InventorySearch()
    root = wb.submit({'request_id': 'research-known-omission', 'action': 'notebook',
        'question': 'Maintain an inventory of the datasets OrionModel evaluates on.',
        'paper_ids': [f'{1:064x}']})
    plan = wb.turn(root['run_id'])['research_task']
    wb.submit({'request_id': 'approve-omission-research', 'action': 'approve_research',
        'conversation_id': root['conversation_id'], 'parent_run_id': root['run_id'],
        'plan_revision': plan['plan_revision'], 'budget': plan['suggested_budget']})
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert view['status'] == 'COMPLETED', view['error']
    assert 'DeltaSet' in view['message']['answer']
    assert 'BetaSet' in view['message']['answer']


def test_omission_revision_is_bounded_when_the_model_keeps_omitting(tmp_path):
    wb = entry_workbench(tmp_path, OmissionModel(correct_on_revision=False))
    wb.runtime.search_adapter = InventorySearch()
    root = wb.submit({'request_id': 'persistent-omission', 'action': 'auto',
        'question': 'List the datasets OrionModel evaluates on.', 'paper_ids': [f'{1:064x}']})
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert view['status'] == 'COMPLETED'
    assert view['message']['answer_scope'] == 'PARTIAL'
    assert view['message']['limitations']
    assert view['budget']['used_llm_calls'] == 5  # Router, two Writer/Auditor pairs; no third revision.


def test_research_revision_resumes_without_regenerating_completed_paid_responses(tmp_path):
    model = OmissionModel()
    wb = entry_workbench(tmp_path, model)
    wb.runtime.search_adapter = InventorySearch()
    root = wb.submit({'request_id': 'research-revision-pause', 'action': 'notebook',
        'question': 'Maintain an inventory of the datasets OrionModel evaluates on.',
        'paper_ids': [f'{1:064x}']})
    plan = wb.turn(root['run_id'])['research_task']
    identity = {'conversation_id': root['conversation_id'], 'parent_run_id': root['run_id'],
                'plan_revision': plan['plan_revision']}
    wb.submit({**identity, 'request_id': 'approve-revision-small', 'action': 'approve_research',
               'budget': {**plan['suggested_budget'], 'max_llm_calls': 3}})
    wb.runtime.resume(root['run_id'])
    paused = wb.turn(root['run_id'])
    assert paused['research_task']['state'] == 'PAUSED_BUDGET'
    wb.background.submitted.clear()
    identity['plan_revision'] = paused['research_task']['plan_revision']
    wb.submit({**identity, 'request_id': 'continue-revision-budget', 'action': 'continue_research',
               'budget': plan['suggested_budget']})
    wb.runtime.resume(root['run_id'])
    final = wb.turn(root['run_id'])
    assert final['status'] == 'COMPLETED'
    assert 'BetaSet' in final['message']['answer']
    assert final['budget']['used_llm_calls'] == 4
