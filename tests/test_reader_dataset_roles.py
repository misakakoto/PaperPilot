"""A citation mentioning a dataset is not evidence of its training role."""
import json

import pytest

from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.dataset_use import apply_dataset_use_guard
from test_session_entry_contract import EntryLibrary, entry_workbench


class RoleSearch:
    store = EntryLibrary()

    def __init__(self, content='OrionModel experiments on the DeltaSet dataset demonstrate robustness.'):
        self.content = content

    def search(self, question, top_k=8, *, paper_sha256s=None):
        sha = next(iter(paper_sha256s))
        return [{'stable_id': 'dataset-role-original', 'paper_sha256': sha, 'page': 1,
                 'source_uri': f'paper://{sha}/original',
                 'content': self.content}]


class OverconfidentRoleModel:
    model = 'fixture-overconfident-reviewer'

    def __init__(self, fact='OrionModel 在 DeltaSet 数据集上进行了训练。'):
        self.fact = fact

    def complete(self, messages, *, before_attempt, **kwargs):
        before_attempt()
        payload = json.loads(messages[-1]['content'])
        if 'capabilities' in payload:
            value = {'action': 'read'}
        elif 'evidence' in payload:
            value = {'facts': [{'id': 'f1', 'text': self.fact,
                               'evidence_ids': [payload['evidence'][0]['evidence_id']]}],
                     'direct_answer_id': 'f1', 'limitations': []}
        elif 'facts' in payload:
            value = {'assessments': [{'fact_id': fact['id'], 'verdict': 'SUPPORTED',
                        'relevance': 'DIRECT', 'span_ids': [fact['allowed_span_ids'][0]]}
                       for fact in payload['facts']], 'coverage': 'COMPLETE', 'missing_aspects': []}
        else:
            value = {'action': 'STOP', 'evidence_ids': [], 'reason': 'No additional source in this fixture'}
        return LLMResponse(text=json.dumps(value), finish_reason='stop', model=self.model)


def test_experiment_mention_cannot_be_published_as_training_evidence(tmp_path):
    wb = entry_workbench(tmp_path, OverconfidentRoleModel())
    wb.runtime.search_adapter = RoleSearch()
    root = wb.submit({'request_id': 'dataset-training-role', 'action': 'auto',
                     'question': 'OrionModel 用什么数据训练？', 'paper_ids': [f'{1:064x}']})
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert view['status'] == 'COMPLETED', view['error']
    assert '在 DeltaSet 数据集上进行了训练' not in view['message']['answer']
    assert view['message']['limitations']


@pytest.mark.parametrize('source,fact,supported', [
    ('OrionModel was trained on the DeltaSet dataset.', 'OrionModel 在 DeltaSet 数据集上进行了训练。', True),
    ('OrionModel was trained on AlphaSet and tested on DeltaSet.', 'OrionModel 在 DeltaSet 数据集上进行了训练。', False),
    ('OrionModel was not trained on DeltaSet.', 'OrionModel 在 DeltaSet 数据集上进行了训练。', False),
    ('OrionModel 使用 DeltaSet 数据集进行训练。', 'OrionModel was trained on DeltaSet.', True),
    ('OrionModel training data is DeltaSet.', 'OrionModel 在 DeltaSet 数据集上进行了训练。', True),
    ('OrionModel was trained on AlphaSet. It was evaluated on DeltaSet.',
     'OrionModel was trained on AlphaSet and evaluated on DeltaSet.', True),
    ('OrionModel was trained on AlphaSet. Experiments on DeltaSet are reported.',
     'OrionModel 在 AlphaSet 和 DeltaSet 数据集上进行了训练。', False),
])
def test_training_support_keeps_dataset_roles_and_negation_distinct(tmp_path, source, fact, supported):
    wb = entry_workbench(tmp_path, OverconfidentRoleModel(fact))
    wb.runtime.search_adapter = RoleSearch(source)
    root = wb.submit({'request_id': 'dataset-role-polarity', 'action': 'auto',
        'question': 'What data was OrionModel trained on?', 'paper_ids': [f'{1:064x}']})
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert view['status'] == 'COMPLETED', view['error']
    assert (fact in view['message']['answer']) is supported


class MisplacedCitationSearch(RoleSearch):
    def search(self, question, top_k=8, *, paper_sha256s=None):
        rows = super().search(question, top_k, paper_sha256s=paper_sha256s)
        return rows + [{**rows[0], 'stable_id': 'actual-training-source', 'page': 2,
                        'source_uri': rows[0]['source_uri'] + '/training',
                        'content': 'OrionModel was trained on DeltaSet before evaluation.'}]


class ReselectingRoleModel(OverconfidentRoleModel):
    def complete(self, messages, *, before_attempt, **kwargs):
        payload = json.loads(messages[-1]['content'])
        hints = (payload.get('prior_review_untrusted') or {}).get('dataset_roles', {}).get('facts', [])
        candidates = [span for row in hints for span in row.get('candidate_source_spans', [])]
        if 'evidence' in payload and candidates:
            before_attempt()
            value = {'facts': [{'id': 'f1', 'text': self.fact,
                               'evidence_ids': [candidates[0]['evidence_id']]}],
                     'direct_answer_id': 'f1', 'limitations': []}
            return LLMResponse(text=json.dumps(value), finish_reason='stop', model=self.model)
        return super().complete(messages, before_attempt=before_attempt, **kwargs)


def test_wrong_citation_is_reselected_and_reaudited_not_silently_upgraded(tmp_path):
    wb = entry_workbench(tmp_path, ReselectingRoleModel())
    wb.runtime.search_adapter = MisplacedCitationSearch()
    root = wb.submit({'request_id': 'reselect-training-source', 'action': 'auto',
        'question': 'OrionModel 用什么数据训练？', 'paper_ids': [f'{1:064x}']})
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert '在 DeltaSet 数据集上进行了训练' in view['message']['answer']
    assert view['budget']['used_llm_calls'] == 5  # One Writer/Auditor revision, no extra planner.
    reports = wb.runtime.status(root['run_id'])['run']['metadata']['reader_rounds']
    assert reports[0]['dataset_roles']['changed_fact_ids'] == ['f1']
    assert reports[1]['dataset_roles']['changed_fact_ids'] == []
    assert reports[0]['draft']['facts'][0]['evidence_ids'] != reports[1]['draft']['facts'][0]['evidence_ids']


class MisboundCitationModel(OverconfidentRoleModel):
    def complete(self, messages, *, before_attempt, **kwargs):
        # The first audit locates a real supporting passage outside the Writer's
        # citation. A format-only repair can be coerced into a wrong allowed ID.
        before_attempt()
        payload = next(json.loads(m['content']) for m in reversed(messages)
                       if m['role'] == 'user' and m['content'].startswith('{'))
        if 'facts' in payload:
            repair = any(m['role'] == 'assistant' for m in messages)
            value = {'assessments': [{'fact_id': 'f1', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                      'span_ids': payload['facts'][0]['allowed_span_ids'][:1] if repair else ['s1']}],
                     'coverage': 'COMPLETE', 'missing_aspects': []}
        elif 'evidence' in payload:
            navigation = (payload.get('prior_review_untrusted') or {}).get('source_navigation', [])
            eid = navigation[0]['evidence_id'] if navigation else payload['evidence'][0]['evidence_id']
            value = {'facts': [{'id': 'f1', 'text': self.fact, 'evidence_ids': [eid]}],
                     'direct_answer_id': 'f1', 'limitations': []}
        else:
            value = {'action': 'read'}
        return LLMResponse(text=json.dumps(value), finish_reason='stop', model=self.model)


def test_audit_finding_another_original_repairs_writer_binding_not_the_quote(tmp_path):
    wb = entry_workbench(tmp_path, MisboundCitationModel())
    wb.runtime.search_adapter = MisplacedCitationSearch()
    root = wb.submit({'request_id': 'wrong-writer-binding', 'action': 'auto',
        'question': 'OrionModel 用什么数据训练？', 'paper_ids': [f'{1:064x}']})
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert '在 DeltaSet 数据集上进行了训练' in view['message']['answer']
    assert view['budget']['used_llm_calls'] == 5
    first, revised = wb.runtime.status(root['run_id'])['run']['metadata']['reader_rounds']
    assert first['raw_audit']['assessments'][0]['verdict'] == 'SUPPORTED'
    assert first['audit']['assessments'][0]['verdict'] == 'UNCERTAIN'
    assert first['citation_bindings']['changed_fact_ids'] == ['f1']
    assert not first.get('audit_repair')
    assert first['draft']['facts'][0]['evidence_ids'] != revised['draft']['facts'][0]['evidence_ids']


class DatasetOwnershipSearch(RoleSearch):
    def search(self, question, top_k=8, *, paper_sha256s=None):
        rows = super().search(question, top_k, paper_sha256s=paper_sha256s)
        rows[0]['content'] = ('Related work: DeltaSet was established by the earlier Delta Challenge '
                              'as a building damage assessment dataset.')
        return rows + [{**rows[0], 'stable_id': 'authors-own-results', 'page': 2,
                        'source_uri': rows[0]['source_uri'] + '/results',
                        'content': 'Our method reports building damage assessment results on the test set of DeltaSet.'}]


class DatasetOwnershipModel(OverconfidentRoleModel):
    def __init__(self):
        super().__init__('OrionModel 论文使用了 DeltaSet 数据集进行建筑物损伤评估。')

    def complete(self, messages, *, before_attempt, **kwargs):
        payload = json.loads(messages[-1]['content'])
        if 'evidence' in payload and payload.get('prior_review_untrusted'):
            before_attempt()
            navigation = payload['prior_review_untrusted']['source_navigation']
            return LLMResponse(text=json.dumps({'facts': [{'id': 'f1', 'text': self.fact,
                'evidence_ids': [navigation[0]['evidence_id']]}], 'direct_answer_id': 'f1',
                'limitations': []}), finish_reason='stop', model=self.model)
        if 'facts' in payload:
            before_attempt()
            return LLMResponse(text=json.dumps({'assessments': [{'fact_id': 'f1',
                'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                'span_ids': payload['facts'][0]['allowed_span_ids'][:1]}],
                'coverage': 'COMPLETE', 'missing_aspects': [], 'answer_conflicts': []}),
                finish_reason='stop', model=self.model)
        return super().complete(messages, before_attempt=before_attempt, **kwargs)


def test_background_dataset_description_cannot_prove_this_paper_used_it(tmp_path):
    wb = entry_workbench(tmp_path, DatasetOwnershipModel())
    wb.runtime.search_adapter = DatasetOwnershipSearch()
    root = wb.submit({'request_id': 'own-dataset-source', 'action': 'auto',
        'question': 'OrionModel 论文使用了什么数据集？', 'paper_ids': [f'{1:064x}']})
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert view['status'] == 'COMPLETED', view['error']
    assert 'DeltaSet' in view['message']['answer']
    first, revised = wb.runtime.status(root['run_id'])['run']['metadata']['reader_rounds']
    assert first['audit']['assessments'][0]['verdict'] == 'UNCERTAIN'
    assert first['draft']['facts'][0]['evidence_ids'] != revised['draft']['facts'][0]['evidence_ids']
    assert revised['audit']['assessments'][0]['verdict'] == 'SUPPORTED'


@pytest.mark.parametrize('source,supported', [
    ('Related work: DeltaSet was established by an earlier challenge for building damage assessment.', False),
    ('Our method reports building damage assessment results on the test set of DeltaSet.', True),
    ('IV. E XPERIMENTS AND DISCUSSION\nA. Datasets\n1) DeltaSet is a damage assessment dataset.', True),
    ('Our method did not use DeltaSet for building damage assessment.', False),
    ('A previous method reports results on DeltaSet for building damage assessment.', False),
    ('Our method reports results on AlphaSet. Previous work introduced DeltaSet for damage assessment.', False),
])
def test_named_dataset_own_use_requires_paper_experiment_source(tmp_path, source, supported):
    class SourceSelectingModel(OverconfidentRoleModel):
        def complete(self, messages, *, before_attempt, **kwargs):
            payload = json.loads(messages[-1]['content'])
            if 'facts' in payload:
                before_attempt()
                sid = next(sid for group in payload['sources'] for sid, text in group['spans'].items()
                           if 'DeltaSet' in text)
                return LLMResponse(text=json.dumps({'assessments': [{'fact_id': 'f1',
                    'verdict': 'SUPPORTED', 'relevance': 'DIRECT', 'span_ids': [sid]}],
                    'coverage': 'COMPLETE', 'missing_aspects': []}),
                    finish_reason='stop', model=self.model)
            return super().complete(messages, before_attempt=before_attempt, **kwargs)

    wb = entry_workbench(tmp_path, SourceSelectingModel(
        'OrionModel 论文使用了 DeltaSet 数据集进行建筑物损伤评估。'))
    wb.runtime.search_adapter = RoleSearch(source)
    root = wb.submit({'request_id': 'own-use-cases', 'action': 'auto',
        'question': 'OrionModel 论文使用了什么数据集？', 'paper_ids': [f'{1:064x}']})
    wb.runtime.resume(root['run_id'])
    view = wb.turn(root['run_id'])
    assert ('DeltaSet' in view['message']['answer']) is supported


def test_own_result_on_another_dataset_in_same_span_does_not_prove_use():
    source = ('Our method reports results on AlphaSet. Previous work introduced DeltaSet '
              'for damage assessment.')
    draft = {'facts': [{'id': 'f1', 'text': '本文使用了 DeltaSet 数据集。',
                        'evidence_ids': ['e1']}]}
    audit = {'assessments': [{'fact_id': 'f1', 'verdict': 'SUPPORTED',
                              'span_ids': ['s1']}], 'coverage': 'COMPLETE', 'missing_aspects': []}
    spans = [{'span_id': 's1', 'evidence_id': 'e1', 'text': source}]
    guarded, report = apply_dataset_use_guard(draft, audit, spans)
    assert guarded['assessments'][0]['verdict'] == 'UNCERTAIN'
    assert report['facts'][0]['status'] == 'OWN_USE_NOT_ESTABLISHED'
