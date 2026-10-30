from evals.qasper.freeze_unseen import select_unseen
import json
import sqlite3
import pytest


def test_previously_diagnosed_questions_cannot_be_relabelled_as_unseen():
    dataset = {'paper': {'qas': [{'question_id': 'used', 'question': 'Original question?',
        'answers': [{'answer': {'unanswerable': True}}]}]}}
    result = select_unseen(dataset, {'used'}, per_type=1, seed='fixed')
    assert result['status'] == 'INSUFFICIENT_UNSEEN_CASES'
    assert result['cases'] == []
    assert result['excluded_question_ids'] == ['used']
    assert result['available_by_type']['unanswerable'] == 0


def test_stratified_freeze_keeps_original_questions_and_balanced_types():
    answers = [{'extractive_spans':['A']}, {'free_form_answer':'A'}, {'yes_no':False}, {'unanswerable':True}]
    qs = [{'question_id':f'{kind}-{i}', 'question':f'Original {kind} {i}?',
           'answers':[{'answer':answer}]} for kind,answer in enumerate(answers) for i in range(12)]
    result = select_unseen({'p':{'qas':qs}}, set())
    assert result['status'] == 'FROZEN'
    assert len(result['cases']) == 40
    for kind in ('extractive','abstractive','boolean','unanswerable'):
        assert sum(r['answer_type']==kind for r in result['cases']) == 10
    assert {r['question'] for r in result['cases']} <= {q['question'] for q in qs}


def test_history_audit_covers_id_arrays_and_original_questions_in_databases(tmp_path):
    from evals.qasper.freeze_unseen import audit_history
    q1,q2='a'*40,'b'*40
    dataset=tmp_path/'input.json'
    dataset.write_text(json.dumps({'p':{'qas':[
        {'question_id':q1,'question':'First original question?'},
        {'question_id':q2,'question':'Second original question?'}]}}),encoding='utf-8')
    history=tmp_path/'history'
    history.mkdir()
    (history/'manifest.json').write_text(json.dumps({'question_ids':[q1]}),encoding='utf-8')
    with sqlite3.connect(history/'runtime.sqlite3') as c:
        c.execute('CREATE TABLE runs(question TEXT)')
        c.execute('INSERT INTO runs VALUES(?)',('Second original question?',))
    used,audit=audit_history(dataset,[history])
    assert used == {q1,q2}
    assert audit['errors'] == []


def test_failed_history_reads_cannot_be_silently_dropped(tmp_path):
    from evals.qasper.freeze_unseen import finalize_reviewed_audit
    previous=tmp_path/'audit.json'
    previous.write_text(json.dumps({'history_audit':{'errors':[{'path':'unknown-results'}]}}),encoding='utf-8')
    with pytest.raises(ValueError,match='Every failed read'):
        finalize_reviewed_audit(tmp_path/'dataset.json',previous,tmp_path/'frozen.json',{})


def test_additional_holdout_excludes_all_frozen_cases_even_if_unexecuted(tmp_path):
    import hashlib
    from evals.qasper.freeze_unseen import freeze_additional
    answers = [{'extractive_spans':['A']}, {'free_form_answer':'A'}, {'yes_no':False}, {'unanswerable':True}]
    data = {'p': {'qas': [{'question_id':f'{kind}-{i}', 'question':f'Original {kind} {i}?',
        'answers':[{'answer':answer}]} for kind,answer in enumerate(answers) for i in range(30)]}}
    dataset = tmp_path/'dataset.json'
    dataset.write_text(json.dumps(data), encoding='utf-8')
    first = select_unseen(data, {'0-0'})
    first['dataset_sha256'] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    previous = tmp_path/'first.json'
    previous.write_text(json.dumps(first), encoding='utf-8')
    second = freeze_additional(dataset, previous, tmp_path/'second.json', seed='independent')
    first_ids = {r['question_id'] for r in first['cases']}
    second_ids = {r['question_id'] for r in second['cases']}
    assert len(first_ids) == len(second_ids) == 40
    assert first_ids.isdisjoint(second_ids)
    assert first_ids | {'0-0'} <= set(second['excluded_question_ids'])
    assert second['previous_selection_count'] == 40


def test_future_audits_do_not_hide_previously_dated_evaluation_folders(tmp_path):
    from evals.qasper.freeze_unseen import audit_history
    dataset = tmp_path/'dataset.json'
    question_id = 'a'*40
    dataset.write_text(json.dumps({'p': {'qas': [{'question_id': question_id, 'question': 'Original?'}]}}),
                       encoding='utf-8')
    outputs = tmp_path/'outputs'
    previous = outputs/'research-task-evaluation-20260927-previous'
    previous.mkdir(parents=True)
    (previous/'protocol.json').write_text(json.dumps({'question_id': question_id}), encoding='utf-8')
    used, _ = audit_history(dataset, [outputs])
    assert used == {question_id}
