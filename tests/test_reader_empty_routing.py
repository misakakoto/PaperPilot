"""An empty navigator must not silently discard locally available method text."""
import json
from types import SimpleNamespace
import pytest

from paperpilot_agentic.providers import LLMResponse
from test_durable_reader import AnswerProvider
from test_reader_pipeline import factory


@pytest.mark.parametrize('page_end,expected_page', [(7, 7), (None, None)])
def test_empty_router_reads_missing_original_before_writer(tmp_path, page_end, expected_page):
    class Provider(AnswerProvider):
        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]['content'])
            if 'outline' in payload:
                kwargs['before_attempt']()
                self.calls += 1
                return LLMResponse(text=json.dumps({'section_ids': [],
                    'reason': 'Headings are not factual content; originals are needed.'}), finish_reason='stop')
            if 'evidence' in payload:
                assert any('Token grouping uses alternating positions.' in row['content']
                           for row in payload['evidence']), 'Missing method never reached Writer'
            return super().complete(messages, **kwargs)

    class Store:
        def paper_outline(self, sha):
            # PDF extraction can produce a sentence fragment as a section title.
            return [{'section_id': 'body', 'title': 'model the relationship between',
                     'chars': 90, 'chunk_ids': ['detail']}]

        def get_chunks(self, ids):
            return [SimpleNamespace(paper_sha256='sha', stable_id='detail', title='Alpha',
                section='model the relationship between',
                text='Token grouping uses alternating positions.', page_start=7,
                page_end=page_end)] if 'detail' in ids else []

    provider = Provider()
    runtime = factory(tmp_path, provider)
    runtime.search_adapter.store = Store()
    rid = runtime.enqueue(question='How does token grouping work?', metadata={
        'engine': 'answer_first_text', 'paper_sha256s': ['sha']})['run']['run_id']
    result = runtime.resume(rid)
    assert result['run']['status'] == 'COMPLETED', result['run']['metadata']
    originals = [row for row in result['evidence']
                 if 'Token grouping uses alternating positions.' in row['content']]
    assert originals and all(row.get('page') == expected_page for row in originals)
    assert provider.calls == 3  # no second navigation request
    assert runtime.resume(rid)['final_answer'] == result['final_answer']
    assert provider.calls == 3


def test_nonempty_router_also_reloads_original_operation_definitions(tmp_path):
    class Provider(AnswerProvider):
        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]['content'])
            if 'outline' in payload:
                kwargs['before_attempt']()
                self.calls += 1
                return LLMResponse(text='{"section_ids":["caption"]}', finish_reason='stop')
            if 'evidence' in payload:
                assert any('Alpha modeling is cross-ordering tokens.' in row['content']
                           for row in payload['evidence']), 'Definition omitted upstream'
            return super().complete(messages, **kwargs)

    class Store:
        def paper_outline(self, sha):
            return [{'section_id': cid, 'title': 'Methods', 'chars': 70, 'chunk_ids': [cid]}
                    for cid in ('caption', 'definition')]

        def get_chunks(self, ids):
            return [SimpleNamespace(paper_sha256='sha', stable_id=cid, title='Paper',
                section='Methods', text='Alpha mechanism shown in Fig. 4.' if cid == 'caption'
                else 'Alpha modeling is cross-ordering tokens.', page_start=4) for cid in ids]

    provider = Provider()
    runtime = factory(tmp_path, provider)
    runtime.search_adapter.store = Store()
    rid = runtime.enqueue(question='Explain Alpha token organization.', metadata={
        'engine': 'answer_first_text', 'paper_sha256s': ['sha']})['run']['run_id']
    result = runtime.resume(rid)
    assert result['run']['status'] == 'COMPLETED', result['run']['metadata']
    assert provider.calls == 3
    navigation = result['run']['metadata']['reader_routing']['method_definition_read']
    assert navigation['provider_attempts'] == 0 and navigation['semantic_coverage'] == 'NOT_PROVEN'
    assert runtime.resume(rid)['final_answer'] == result['final_answer']
    assert provider.calls == 3
