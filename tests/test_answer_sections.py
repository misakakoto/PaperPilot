from copy import deepcopy
import pytest

from paperpilot_agentic.answer_audit import audited_presentation
from paperpilot_agentic.answer_layout import apply_requested_layout


def candidate():
    draft = {'facts': [
        {'id': 'context', 'text': 'The measurements use the same validation split.', 'evidence_ids': ['e2']},
        {'id': 'answer', 'text': 'The reported difference is 2 F1 points.', 'evidence_ids': ['e1']},
    ], 'direct_answer_id': 'context', 'limitations': ['The test-set effect is uncertain from the current evidence.']}
    audit = {'assessments': [
        {'fact_id': 'context', 'verdict': 'SUPPORTED', 'relevance': 'NECESSARY_CONTEXT'},
        {'fact_id': 'answer', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT'},
    ], 'coverage': 'PARTIAL', 'missing_aspects': []}
    return draft, audit


def test_reviewed_answer_and_necessary_context_are_distinguishable_without_deletion():
    draft, audit = candidate()
    original = deepcopy((draft, audit))
    report = audited_presentation(draft, audit)
    assert report['answer_sections'] == [
        {'kind': 'direct', 'text': 'The reported difference is 2 F1 points. [e1]'},
        {'kind': 'context', 'text': 'The measurements use the same validation split. [e2]'},
    ]
    assert report['answer'] == '\n\n'.join(s['text'] for s in report['answer_sections'])
    assert report['limitations'] == draft['limitations']
    assert (draft, audit) == original


def test_context_is_never_promoted_to_direct_answer():
    draft, audit = candidate()
    audit['assessments'][1]['relevance'] = 'TANGENTIAL'
    report = audited_presentation(draft, audit)
    assert report['answer_scope'] == 'BACKGROUND_ONLY'
    assert report['answer_sections'] == [
        {'kind': 'context', 'text': 'The measurements use the same validation split. [e2]'}]


def test_requested_numbered_layout_is_not_replaced_by_sections():
    draft, audit = candidate()
    report = audited_presentation(draft, audit)
    result = apply_requested_layout('用三点回答', report, None)
    assert result['presentation_contract']['status'] == 'SATISFIED'
    assert not result.get('answer_sections')
    assert 'test-set effect' in result['answer']


def test_numbered_layout_clears_sections_when_merged_then_published(tmp_path):
    from paperpilot_agentic.db import RuntimeDatabase
    from paperpilot_agentic.durable_reader import publish_reader_answer
    from paperpilot_agentic.models import Budget
    draft, audit = candidate()
    report = audited_presentation(draft, audit)
    # The reader merges presentation updates into its existing report.
    report.update(apply_requested_layout('用一点回答', report, None))
    report['status'] = 'REVIEW_REQUIRED'
    db = RuntimeDatabase(tmp_path / 'runtime.db')
    rid = db.create_run('用一点回答', Budget())
    published = publish_reader_answer(db, rid, report)
    assert published['answer'].startswith('1. ')
    assert 'test-set effect' in published['answer']
    assert 'answer_sections' not in published


def test_publication_persists_sections_without_rewriting_historical_results(tmp_path):
    from paperpilot_agentic.db import RuntimeDatabase
    from paperpilot_agentic.durable_reader import publish_reader_answer
    from paperpilot_agentic.models import Budget
    db = RuntimeDatabase(tmp_path / 'runtime.db')
    rid = db.create_run('Question?', Budget())
    value = {'status': 'REVIEW_REQUIRED', 'answer': 'Answer.\n\nCondition.',
             'answer_sections': [{'kind': 'direct', 'text': 'Answer.'},
                                 {'kind': 'context', 'text': 'Condition.'}]}
    published = publish_reader_answer(db, rid, value)
    reopened = RuntimeDatabase(db.path).get_run(rid)['metadata']['final_answer']
    assert reopened['answer_sections'] == published['answer_sections'] == value['answer_sections']
    historical = publish_reader_answer(db, rid, {'status': 'REVIEW_REQUIRED', 'answer': 'Old answer.'})
    assert 'answer_sections' not in historical


@pytest.mark.parametrize('sections', [None, [{'kind': 'direct', 'text': 'Altered answer.'}],
    [{'kind': 'new', 'text': 'Answer.'}], [{'kind': 'direct', 'text': ''}],
    [{'kind': 'direct', 'text': 'Answer.'}, {'kind': 'direct', 'text': 'Extra.'}]])
def test_malformed_sections_cannot_add_or_delete_published_prose(tmp_path, sections):
    from paperpilot_agentic.db import RuntimeDatabase
    from paperpilot_agentic.durable_reader import publish_reader_answer
    from paperpilot_agentic.models import Budget
    db = RuntimeDatabase(tmp_path / 'runtime.db')
    rid = db.create_run('Question?', Budget())
    report = {'status': 'REVIEW_REQUIRED', 'answer': 'Answer.', 'answer_sections': sections}
    if sections is None:
        assert publish_reader_answer(db, rid, report)['answer'] == 'Answer.'
    else:
        with pytest.raises(ValueError, match='preserve'):
            publish_reader_answer(db, rid, report)
