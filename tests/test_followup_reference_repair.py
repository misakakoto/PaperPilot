import json
import pytest

from paperpilot_agentic.follow_up_reader import resolve_follow_up
from paperpilot_agentic.providers import LLMResponse


class Resolver:
    def __init__(self, questions):
        self.questions = iter(questions)
        self.inputs = []

    def complete(self, messages, **kwargs):
        kwargs['before_attempt']()
        self.inputs.append(messages)
        return LLMResponse(text=json.dumps({
            'question': next(self.questions), 'use_prior_sources': True,
            'needs_clarification': False, 'reason': 'Same comparison',
        }), finish_reason='stop')


def test_remaining_pair_reference_is_repaired_before_search():
    model = Resolver(['比较这两个方法的F1分数', '比较 RadarNet 与 OpticNet 的F1分数'])
    result = resolve_follow_up('比较这两个方法的F1分数', {
        'task_state': {'label': '比较 RadarNet 与 OpticNet'},
    }, model)
    assert result['question'] == '比较 RadarNet 与 OpticNet 的F1分数'
    assert result['provider_attempts'] == 2
    assert len(result['calls']) == 2
    assert not result['needs_clarification']


def test_repeated_unresolved_reference_does_not_reach_search():
    model = Resolver(['比较这两个方法', '比较这两个方法'])
    result = resolve_follow_up('比较这两个方法', {}, model)
    assert result['needs_clarification']
    assert result['clarification_code'] == 'unresolved_subject_reference'
    assert len(model.inputs) == 2


def test_standalone_question_does_not_pay_for_repair():
    model = Resolver(['What is RadarNet F1?'])
    result = resolve_follow_up('What is RadarNet F1?', {}, model)
    assert result['provider_attempts'] == 1
    assert not result['needs_clarification']


def test_repair_obeys_callers_budget():
    class Budgeted(Resolver):
        def complete(self, messages, **kwargs):
            if self.inputs:
                raise RuntimeError('Shared budget exhausted')
            return super().complete(messages, **kwargs)
    model = Budgeted(['Compare these two methods'])
    with pytest.raises(RuntimeError, match='Shared budget exhausted'):
        resolve_follow_up('Compare these two methods', {}, model)
    assert len(model.inputs) == 1


def test_unknown_history_still_requires_clarification_after_repair():
    model = Resolver(['比较这两个方法', '比较 RadarNet 与 OpticNet'])
    result = resolve_follow_up('比较这两个方法', {
        'task_history': {'requests': [], 'requirements_complete': False},
    }, model)
    assert result['needs_clarification']
    assert result['clarification_code'] == 'incomplete_requirement_memory'


@pytest.mark.parametrize('resolved', [True, False])
def test_normal_entry_searches_only_resolved_subject_and_replays(tmp_path, resolved):
    from test_reader_pipeline import factory, Search
    from test_durable_reader import AnswerProvider

    class Provider(AnswerProvider):
        def __init__(self):
            super().__init__()
            self.resolver = Resolver(['Compare these two methods',
                'Compare RadarNet and OpticNet' if resolved else 'Compare these two methods'])

        def complete(self, messages, **kwargs):
            if messages[0]['content'].startswith('Rewrite a follow-up'):
                self.calls += 1
                return self.resolver.complete(messages, **kwargs)
            return super().complete(messages, **kwargs)

    class RecordingSearch(Search):
        def __init__(self):
            self.queries = []

        def search(self, query, top_k=8, *, paper_sha256s=None):
            self.queries.append(query)
            return super().search(query, top_k, paper_sha256s=paper_sha256s)

    provider = Provider()
    runtime = factory(tmp_path, provider)
    search = runtime.search_adapter = RecordingSearch()
    parent = runtime.enqueue(question='Compare RadarNet and OpticNet',
        metadata={'engine': 'answer_first_text'})['run']['run_id']
    runtime.resume(parent)
    search.queries.clear()
    child = runtime.enqueue_follow_up(parent_run_id=parent,
        question='Compare these two methods')['run']['run_id']
    result = runtime.resume(child)
    assert search.queries == (['Compare RadarNet and OpticNet'] if resolved else [])
    assert result['run']['used_llm_calls'] == (4 if resolved else 2)
    count = provider.calls
    replay = runtime.resume(child)
    assert provider.calls == count
    assert replay['final_answer'] == result['final_answer']
