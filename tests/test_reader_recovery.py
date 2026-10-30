import json
import pytest
from paperpilot_agentic.reader_recovery import plan_evidence_reload
from paperpilot_agentic.providers import LLMResponse


class Model:
    def __init__(self, value):
        self.value = value
        self.calls = 0
    def complete(self, messages, **kwargs):
        kwargs['before_attempt']()
        self.calls += 1
        return LLMResponse(text=json.dumps(self.value), finish_reason='stop')


CANDIDATES = [{'evidence_id': 'a', 'content': 'Already read.', 'source_uri': 'paper://p/a'},
              {'evidence_id': 'b', 'content': 'Unread mechanism.', 'source_uri': 'paper://p/b'}]


@pytest.mark.parametrize('span_id,expected', [('s0', 'REVISE'), ('invented', 'STOP')])
def test_revision_selects_host_owned_span_and_gap_ids(span_id, expected):
    class Selecting(Model):
        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]['content'])
            assert '"quote":' not in messages[0]['content'], 'Do not teach the generator a conflicting legacy example'
            assert payload['revision_aspect_catalog'] == [{'aspect_id': 'g0', 'text': 'Missing operation'}]
            assert payload['revision_span_catalog'][0]['text'] == 'Already read.'
            assert payload['host_rejection_targets'][0]['fact_id'] == 'f'
            return super().complete(messages, **kwargs)
    model = Selecting({'action': 'REVISE', 'reason': 'Recheck the source, not the old approval',
                       'revision_basis': [{'aspect_id': 'g0', 'span_ids': [span_id]}]})
    report = {'limitations': ['Missing operation'], 'audit': {'assessments': [
        {'fact_id': 'f', 'verdict': 'UNCERTAIN', 'boundary_reason': 'paper_absence_not_established'}]},
        'answer_boundaries': {'changed_fact_ids': ['f']}}
    result = plan_evidence_reload('Explain mechanism', report, CANDIDATES, {'a'}, model)
    assert result['action'] == expected
    assert model.calls == 1
    if expected == 'REVISE':
        assert result['revision_basis'][0]['quote'] == 'Already read.'
        assert result['revision_validation'] == 'SOURCE_BOUND_NOT_FACT_VERIFIED'


def test_reload_only_accepts_saved_unread_ids():
    model = Model({'action': 'RELOAD', 'evidence_ids': ['b'], 'reason': 'Mechanism gap'})
    value = plan_evidence_reload('How?', {}, CANDIDATES, {'a'}, model)
    assert value['evidence_ids'] == ['b']
    assert model.calls == 1
    assert value['response']['finish_reason'] == 'stop'


@pytest.mark.parametrize('basis', [None, [], [{'aspect': 'Missing mechanism',
    'evidence_id': 'a', 'quote': 'A detail absent from the source.'}],
    [{'aspect': 'Another gap', 'evidence_id': 'a', 'quote': 'Already read.'}]])
def test_revision_cannot_claim_missing_aspects_resolved_without_original_quotes(basis):
    model = Model({'action': 'REVISE', 'reason': 'The existing evidence is sufficient',
                   'revision_basis': basis})
    result = plan_evidence_reload('Explain mechanism', {'limitations': ['Missing mechanism']},
                                  CANDIDATES, {'a'}, model, allow_search=True)
    assert result['action'] == 'STOP'
    assert result['proposed_action'] == 'REVISE'
    assert result['revision_validation'] == 'UNRESOLVED_BASIS'
    assert model.calls == 1


def test_grounded_revision_preserves_the_normal_rewrite_path():
    source = {'evidence_id': 'a', 'source_uri': 'paper://p/a', 'content': 'Grouping alternates positions.'}
    model = Model({'action': 'REVISE', 'reason': 'Original answers the omitted detail',
        'revision_basis': [{'aspect': 'Missing grouping detail', 'evidence_id': 'a',
                            'quote': 'Grouping alternates positions.'}]})
    result = plan_evidence_reload('How does grouping work?', {'limitations': ['Missing grouping detail']},
                                  [source], {'a'}, model)
    assert result['action'] == 'REVISE'
    assert result['revision_validation'] == 'SOURCE_BOUND_NOT_FACT_VERIFIED'


def test_known_span_omission_does_not_hide_an_additional_writer_gap():
    report = {'audit': {'omitted_answer_spans': ['s1']},
        'source_spans': [{'span_id': 's1', 'evidence_id': 'a', 'text': 'Already read.'}],
        'draft': {'limitations': ['A requested mechanism detail is missing']},
        'limitations': ['A requested mechanism detail is missing']}
    model = Model({'action': 'SEARCH', 'query': 'grouping operation details', 'reason': 'Another gap'})
    value = plan_evidence_reload('Explain mechanism', report, CANDIDATES, {'a'}, model, allow_search=True)
    assert value['action'] == 'SEARCH' and model.calls == 1


def test_unfitted_gap_reload_stops_before_another_writer_and_restores_partial(tmp_path):
    from test_reader_pipeline import factory
    class Provider:
        calls = 0
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            self.calls += 1
            payload = json.loads(messages[-1]['content'])
            if 'unread_candidates' in payload:
                value = {'action': 'SEARCH', 'query': 'grouping operation details', 'reason': 'Missing detail'}
            elif 'evidence' in payload:
                assert self.calls == 1, 'Unfitted evidence must not trigger a blind rewrite'
                value = {'facts': [{'id': 'f', 'text': 'A source fact.',
                    'evidence_ids': [payload['evidence'][0]['evidence_id']]}], 'direct_answer_id': 'f'}
            else:
                value = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                    'span_ids': [payload['facts'][0]['allowed_span_ids'][0]]}],
                    'coverage': 'PARTIAL', 'missing_aspects': ['Missing grouping operation']}
            return LLMResponse(text=json.dumps(value), finish_reason='stop', prompt_tokens=10, completion_tokens=10)
    class Search:
        def search(self, query, **kwargs):
            gap = 'grouping operation details' in query
            return [{'stable_id': 'gap' if gap else 'original', 'paper_sha256': 'sha',
                'source_uri': 'paper://sha/gap' if gap else 'paper://sha/original',
                'content': 'Detailed grouping operation. ' * 4000 if gap else 'A source fact.', 'page': 2 if gap else 1}]
    provider = Provider()
    runtime = factory(tmp_path, provider)
    runtime.search_adapter = Search()
    rid = runtime.enqueue(question='Explain grouping mechanism', metadata={
        'engine': 'answer_first_text', 'paper_sha256s': ['sha']})['run']['run_id']
    result = runtime.resume(rid)
    assert result['run']['metadata']['reader_recovery_fallback']['effective_action'] == 'STOP'
    assert result['final_answer']['answer'].startswith('A source fact.')
    assert provider.calls == 3 and result['run']['used_llm_calls'] == 3
    runtime.resume(rid)
    assert provider.calls == 3


@pytest.mark.parametrize('ids', [['a'], ['invented'], ['b', 'b'], []])
def test_recovery_rejects_repeated_or_fabricated_evidence(ids):
    with pytest.raises(ValueError):
        plan_evidence_reload('How?', {}, CANDIDATES, {'a'},
                             Model({'action': 'RELOAD', 'evidence_ids': ids, 'reason': 'Gap'}))


def test_exhausted_saved_candidates_stop_without_model():
    assert plan_evidence_reload('How?', {}, CANDIDATES, {'a', 'b'}, None)['provider_attempts'] == 0


def test_recovery_can_check_original_read_evidence_instead_of_trusting_a_bad_audit():
    class EvidenceCheckingModel(Model):
        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]['content'])
            self.payload = payload
            visible = payload.get('read_evidence', [])
            found = any(row['content'] == 'Annotation was performed by the Delta team.' for row in visible)
            self.value = {'action': 'REVISE' if found else 'STOP', 'evidence_ids': [],
                          'reason': 'Inspect the original, not the reviewer conclusion'}
            if found:
                self.value['revision_basis'] = [{'aspect': aspect, 'evidence_id': 'read',
                    'quote': visible[0]['content']} for aspect in payload['revision_aspects']]
            return super().complete(messages, **kwargs)
    model = EvidenceCheckingModel({})
    source = {'evidence_id': 'read', 'content': 'Annotation was performed by the Delta team.',
              'source_uri': 'synthetic://original'}
    report = {'audit': {'coverage': 'UNANSWERABLE'},
              'limitations': ['The reviewer incorrectly says the annotator is absent.']}
    result = plan_evidence_reload('Who annotated?', report, [source], {'read'}, model, allow_search=True)
    assert result['action'] == 'REVISE'
    assert model.payload['read_evidence'] == [source]


def test_recovery_read_window_is_bounded_whole_passages_and_discloses_omissions():
    class Capture(Model):
        def complete(self, messages, **kwargs):
            self.payload = json.loads(messages[-1]['content'])
            return super().complete(messages, **kwargs)
    rows = [{'evidence_id': str(i), 'source_uri': f'paper://p/{i}',
             'content': '原始证据' * 1000} for i in range(30)]
    model = Capture({'action': 'STOP', 'evidence_ids': [], 'reason': 'Synthetic size check'})
    result = plan_evidence_reload('What?', {'limitations': ['Gap']}, rows,
                                  {row['evidence_id'] for row in rows}, model)
    window = result['read_evidence_window']
    assert window['context_upper_bytes'] <= 16000
    assert window['omitted_evidence_ids'] and window['semantic_coverage'] == 'NOT_PROVEN'
    assert all(row in rows for row in model.payload['read_evidence'])


def test_gap_search_requires_explicit_permission_and_a_bounded_query():
    decision = {'action': 'SEARCH', 'query': 'training schedule and optimizer',
                'evidence_ids': [], 'reason': 'Requested training conditions are missing'}
    report = {'limitations': ['Training conditions unavailable']}
    with pytest.raises(ValueError):
        plan_evidence_reload('How was it trained?', report, CANDIDATES, {'a', 'b'}, Model(decision))
    result = plan_evidence_reload('How was it trained?', report, CANDIDATES, {'a', 'b'},
                                  Model(decision), allow_search=True)
    assert result['action'] == 'SEARCH'
    assert result['query'] == 'training schedule and optimizer'


@pytest.mark.parametrize('query', ['', 'x'*401, 'How was it trained?'])
def test_gap_search_rejects_empty_oversize_or_unchanged_queries(query):
    with pytest.raises(ValueError):
        plan_evidence_reload('How was it trained?', {'limitations': ['Gap']}, [], set(),
            Model({'action': 'SEARCH', 'query': query, 'reason': 'Missing conditions'}), allow_search=True)


@pytest.mark.parametrize('kind', ['outside', 'changed', 'duplicate', 'empty'])
def test_gap_lookup_validates_sources_and_does_not_reintroduce_saved_content(kind):
    from paperpilot_agentic.reader_recovery import retrieve_gap_evidence
    original = {'evidence_id': 'evidence_saved', 'stable_id': 'a', 'paper_sha256': 'p',
                'source_uri': 'paper://p/a', 'content': 'Original source.'}
    class Search:
        def search(self, query, **kwargs):
            if kind == 'empty':
                return []
            if kind == 'outside':
                return [{**original, 'paper_sha256': 'other', 'source_uri': 'paper://other/a'}]
            return [{**original, 'content': 'Changed source.' if kind == 'changed' else original['content']}]
    if kind in {'outside', 'changed'}:
        with pytest.raises(ValueError):
            retrieve_gap_evidence(Search(), 'Original question', 'Missing details', {'p'}, [original])
    else:
        assert retrieve_gap_evidence(Search(), 'Original question', 'Missing details', {'p'}, [original]) == []


def test_scoped_empty_initial_retrieval_can_recover_through_one_gap_query(tmp_path):
    from test_reader_pipeline import factory
    class Provider:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            payload = json.loads(messages[-1]['content'])
            if 'unread_candidates' in payload:
                value = {'action': 'SEARCH', 'query': 'optimizer implementation details', 'reason': 'Missing requested training detail'}
            elif 'evidence' in payload:
                value = {'facts': [{'id': 'f', 'text': 'Training uses AdamW.',
                                    'evidence_ids': [payload['evidence'][0]['evidence_id']]}], 'direct_answer_id': 'f'}
            else:
                value = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                         'span_ids': [payload['facts'][0]['allowed_span_ids'][0]]}], 'coverage': 'COMPLETE', 'missing_aspects': []}
            return LLMResponse(text=json.dumps(value), finish_reason='stop', prompt_tokens=10, completion_tokens=10)
    class Search:
        def search(self, query, **kwargs):
            if 'optimizer implementation details' not in query:
                return []
            return [{'stable_id': 'train', 'paper_sha256': 'sha', 'source_uri': 'paper://sha/train',
                     'content': 'Training uses AdamW.', 'page': 4}]
    runtime = factory(tmp_path, Provider())
    runtime.search_adapter = Search()
    run = runtime.enqueue(question='How was Alpha trained?',
        metadata={'engine': 'answer_first_text', 'paper_sha256s': ['sha']})
    result = runtime.resume(run['run']['run_id'])
    assert result['final_answer']['answer'].startswith('Training uses AdamW.')
    assert result['run']['used_tool_calls'] == 2
    assert result['run']['metadata']['reader_gap_search']['status'] == 'NEW_EVIDENCE'


def test_revision_can_use_existing_sources_but_cannot_smuggle_new_ids():
    report = {'limitations': ['Ambiguous data stage'], 'audit': {}}
    decision = plan_evidence_reload('How many?', report, CANDIDATES, {'a', 'b'},
                                   Model({'action': 'REVISE', 'reason': 'Label both stages',
                                       'revision_basis': [{'aspect': 'Ambiguous data stage',
                                           'evidence_id': 'a', 'quote': 'Already read.'}]}))
    assert decision['action'] == 'REVISE'
    assert decision['evidence_ids'] == []
    with pytest.raises(ValueError):
        plan_evidence_reload('How many?', report, CANDIDATES, {'a'},
                             Model({'action': 'REVISE', 'reason': 'Label stages', 'evidence_ids': ['b']}))


@pytest.mark.parametrize('fail_recovery', [False, True])
@pytest.mark.parametrize('max_tokens', [100_000, 200_000])
@pytest.mark.parametrize('action,search_result', [
    ('RELOAD','new'), ('REVISE','new'), ('SEARCH','new'),
    ('SEARCH','empty'), ('SEARCH','duplicate'), ('SEARCH','outside')])
def test_runtime_reloads_then_reaudits_without_publishing_first_round(
        tmp_path, fail_recovery, action, search_result, max_tokens):
    from test_reader_pipeline import factory
    class Provider:
        calls = 0
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            self.calls += 1
            assert runtime.db.get_run(rid)['status'] == 'RUNNING'
            assert 'final_answer' not in runtime.db.get_run(rid)['metadata']
            if fail_recovery and self.calls == 3:
                raise ValueError('Invalid decision')
            payload = json.loads(messages[-1]['content'])
            if self.calls in {1, 4}:
                if self.calls == 4:
                    assert payload['prior_review_untrusted']['audit']['coverage'] == 'PARTIAL'
                eid = payload['evidence'][0]['evidence_id']
                value = {'facts': [{'id': 'f', 'text': f'Answer round {self.calls}.',
                                    'evidence_ids': [eid]}], 'direct_answer_id': 'f'}
            elif self.calls == 3:
                value = {'action': action, 'reason': 'Resolve incomplete answer',
                         'evidence_ids': [payload['unread_candidates'][0]['evidence_id']] if action == 'RELOAD' else []}
                if action == 'SEARCH':
                    value['query'] = 'missing mechanism details'
                if action == 'REVISE':
                    original = payload['read_evidence'][0]
                    value['revision_basis'] = [{'aspect': aspect, 'evidence_id': original['evidence_id'],
                        'quote': original['content']} for aspect in payload['revision_aspects']]
            else:
                sid = payload['facts'][0]['allowed_span_ids'][0]
                value = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED',
                    'relevance': 'DIRECT', 'span_ids': [sid]}],
                    'coverage': 'PARTIAL' if self.calls == 2 else 'COMPLETE',
                    'missing_aspects': ['Missing mechanism'] if self.calls == 2 else []}
            return LLMResponse(text=json.dumps(value), finish_reason='stop', prompt_tokens=10, completion_tokens=10)
    class Search:
        def search(self, query, **kwargs):
            if 'missing mechanism details' in query:
                assert kwargs['paper_sha256s'] == {'sha'}
                assert kwargs['top_k'] == 8
                if search_result == 'empty':
                    return []
                if search_result == 'duplicate':
                    return [{'stable_id': '1', 'paper_sha256': 'sha', 'source_uri': 'paper://sha/1',
                             'page': 1, 'content': '1'*20000}]
                if search_result == 'outside':
                    return [{'stable_id': '3', 'paper_sha256': 'other', 'source_uri': 'paper://other/3',
                             'page': 3, 'content': 'Source outside requested papers.'}]
                return [{'stable_id': '3', 'paper_sha256': 'sha',
                         'source_uri': 'paper://sha/3', 'page': 3, 'content': 'A newly retrieved mechanism.'}]
            return [{'stable_id': str(i), 'paper_sha256': 'sha', 'source_uri': f'paper://sha/{i}',
                     'page': i, 'content': ('A saved mechanism original. ' * 100
                         if action == 'REVISE' and i == 1 else str(i) * 20000)} for i in (1, 2)]
    provider = Provider()
    runtime = factory(tmp_path, provider)
    runtime.search_adapter = Search()
    # The initial shared packet is capped at 24 KB, leaving an unread 20 KB
    # passage at either budget. Budget is not a proxy for packet selection:
    # verify both affordable recovery and a safe stop before mandatory calls.
    from paperpilot_agentic.models import Budget
    rid = runtime.enqueue(question='Explain mechanism', metadata={'engine': 'answer_first_text'},
                          budget=Budget(max_tokens=max_tokens))['run']['run_id']
    result = runtime.resume(rid)
    budget_stop = max_tokens == 100_000 and action == 'RELOAD' and not fail_recovery
    stopped = fail_recovery or budget_stop or (action == 'SEARCH' and search_result != 'new')
    assert provider.calls == (3 if stopped else 5)
    assert result['final_answer']['answer'].startswith('Answer round 1.' if stopped else 'Answer round 4.')
    # A rejected preflight still persists its second-round report, but sends
    # no Writer/Auditor request and restores the first audited answer.
    assert len(result['run']['metadata']['reader_rounds']) == (2 if budget_stop or not stopped else 1)
    if fail_recovery or (action == 'SEARCH' and search_result == 'outside'):
        assert result['run']['metadata']['reader_recovery_error'] == 'ValueError'
    elif budget_stop:
        assert result['run']['metadata']['reader_recovery_error'] == 'LLMBudgetExhausted'
        assert result['run']['budget']['max_tokens'] == 100_000
        assert result['run']['metadata']['reader_recovery_failure_stage'] == 'reader_revision'
        failed_round = result['run']['metadata']['reader_rounds'][-1]
        assert failed_round['error_type'] == 'LLMBudgetExhausted'
        assert failed_round['provider_attempts'] == 0
    else:
        assert result['run']['metadata']['reader_recovery']['action'] == action
        if action == 'SEARCH' and search_result in {'empty', 'duplicate'}:
            assert result['run']['metadata']['reader_gap_search']['status'] == 'STOP_NO_NEW_EVIDENCE'
    assert result['run']['used_tool_calls'] == (1 if fail_recovery or action == 'REVISE' else 2)
    assert len(result['evidence']) == (3 if action == 'SEARCH' and not stopped else 2)
    runtime.resume(rid)
    assert provider.calls == (3 if stopped else 5)
