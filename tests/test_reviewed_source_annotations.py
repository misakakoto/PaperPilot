"""Reviewed legends must reach both stages without rewriting old sources."""
from copy import deepcopy
import json

import pytest

from paperpilot_agentic.answer_pipeline import (
    initialize, AnswerOptions, writer_messages, auditor_messages,
)
from paperpilot_agentic.reader_visual import named_table_content
from paperpilot_agentic.reviewed_sources import ReviewedSources
from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.answer_contracts import evidence_spans
from paperpilot_agentic.providers import LLMResponse
from test_reviewed_sources import artifact, review


LITERAL = '† denotes an unavailable observation.'


def annotation_review():
    return {'reviewer_id': 'Source crop inspection', 'reviewer_kind': 'ai',
            'reviewed_at': '2026-10-19', 'checks': [
                {'dimension': d, 'result': 'PASS', 'note': 'Checked the bound crop.'}
                for d in ['identity', 'location', 'literal_text']]}


def setup(registry):
    item = artifact()
    parent = registry.register(item, review())
    return item, parent


def add(registry, item, parent, **kwargs):
    return registry.register_annotation(parent, item.source, kind='legend', text=LITERAL,
                                        review=annotation_review(), **kwargs)


def test_literal_annotation_reaches_writer_and_auditor_and_preserves_parent(tmp_path):
    registry = ReviewedSources(tmp_path)
    item, parent = setup(registry)
    parent_bytes = (tmp_path / (parent + '.json')).read_bytes()
    before = registry.identity()
    annotation = add(registry, item, parent)
    assert registry.identity() != before
    assert add(registry, item, parent) == annotation
    assert (tmp_path / (parent + '.json')).read_bytes() == parent_bytes
    evidence = {'evidence_id': 'e', 'source_uri': 'paper://p/layout/o',
                'content': 'old', 'provenance': {'nearby_caption_text': {'text': 'OCR legend'}}}
    assert registry.apply(evidence, item.source)
    evidence['content'] = named_table_content(evidence, 'Table I')
    assert LITERAL in evidence['content'] and 'OCR legend' in evidence['content']
    notes = evidence['provenance']['vision_metadata']['literal_source_annotations']
    assert notes[0]['annotation_id'] == annotation
    assert notes[0]['source'] == item.source.model_dump(mode='json')
    assert notes[0]['review']['reviewer_kind'] == 'ai'
    context = initialize('Explain the table marker.', [evidence], object(), AnswerOptions(), None)
    assert LITERAL in json.dumps(writer_messages(context), ensure_ascii=False)
    audit = auditor_messages(context, {'facts': [], 'sources': [evidence]}, [])
    assert LITERAL in json.dumps(audit, ensure_ascii=False)
    assert not context.report.get('all_facts_supported')


@pytest.mark.parametrize('field,value', [('crop_sha256', 'changed'), ('paper_sha256', 'other'),
                                      ('page', 3), ('object_id', 'other')])
def test_registration_cannot_attach_to_another_source(tmp_path, field, value):
    registry = ReviewedSources(tmp_path)
    item, parent = setup(registry)
    region = item.source.model_copy(update={field: value})
    with pytest.raises(ValueError):
        registry.register_annotation(parent, region, kind='legend', text=LITERAL,
                                     review=annotation_review())


def test_tampered_annotation_or_conflicting_legend_is_not_reused(tmp_path):
    registry = ReviewedSources(tmp_path)
    item, parent = setup(registry)
    identifier = add(registry, item, parent)
    with pytest.raises(ValueError, match='Conflicting'):
        registry.register_annotation(parent, item.source, kind='legend', text='A different legend.',
                                     review=annotation_review())
    path = tmp_path / 'annotations' / parent / (identifier + '.json')
    path.write_text(path.read_text(encoding='utf-8').replace(LITERAL, 'Changed meaning.'), encoding='utf-8')
    with pytest.raises(ValueError, match='identity'):
        registry.apply({'content': 'old', 'provenance': {}}, item.source)


def test_missing_literal_review_and_path_like_parent_are_rejected(tmp_path):
    registry = ReviewedSources(tmp_path)
    item, parent = setup(registry)
    invalid = deepcopy(annotation_review())
    invalid['checks'][-1]['result'] = 'UNKNOWN'
    for identifier, check in [(parent, invalid), ('../source', annotation_review())]:
        with pytest.raises(ValueError):
            registry.register_annotation(identifier, item.source, kind='legend', text=LITERAL,
                                         review=check)


def test_real_answer_flow_reads_annotation_in_both_physical_messages(tmp_path):
    registry = ReviewedSources(tmp_path)
    item, parent = setup(registry)
    evidence = {'evidence_id': 'e', 'source_uri': 'paper://p/layout/o',
                'content': 'old', 'provenance': {}}
    registry.apply(evidence, item.source)
    evidence['content'] = named_table_content(evidence, 'Table I')
    before = initialize('Explain Table I marker.', [evidence], object(), AnswerOptions(), None)
    add(registry, item, parent)
    registry.apply(evidence, item.source)
    evidence['content'] = named_table_content(evidence, 'Table I')
    after = initialize('Explain Table I marker.', [evidence], object(), AnswerOptions(), None)
    assert before.report['input_sha256'] != after.report['input_sha256']
    source_span = next(s['span_id'] for s in evidence_spans([evidence]) if LITERAL in s['text'])

    class Model:
        calls = 0

        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            self.calls += 1
            assert LITERAL in json.dumps(messages, ensure_ascii=False)
            payload = json.loads(messages[-1]['content'])
            if self.calls == 1:
                value = {'facts': [{'id': 'f1', 'kind': 'SOURCE_FACT',
                    'text': 'The marker denotes an unavailable observation.',
                    'evidence_ids': [payload['evidence'][0]['evidence_id']]}],
                    'direct_answer_id': 'f1', 'limitations': []}
            else:
                value = {'assessments': [{'fact_id': 'f1', 'verdict': 'SUPPORTED',
                    'relevance': 'DIRECT', 'span_ids': [source_span]}],
                    'coverage': 'COMPLETE', 'missing_aspects': []}
            return LLMResponse(json.dumps(value), finish_reason='stop')

    model = Model()
    report = answer_then_audit('Explain Table I marker.', [evidence], model)
    assert model.calls == 2
    assert report['status'] == 'REVIEW_REQUIRED', report.get('failure_detail')
