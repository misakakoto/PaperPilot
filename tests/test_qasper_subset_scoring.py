import json

import pytest

from evals.qasper.score_reader_subset import score


def save(path, data):
    path.write_text(json.dumps(data), encoding='utf-8')


@pytest.fixture
def experiment(tmp_path):
    dataset = tmp_path/'dataset.json'
    save(dataset, {'p': {'qas': [{'question_id': 'q1', 'question': 'Original question?'}]}})
    # Only a scoring-contract stub; not a replacement for official QASPER scoring.
    evaluator = tmp_path/'evaluator.py'
    evaluator.write_text("def get_answers_and_evidence(dataset, flag):\n    return {'q1': [{'answer': 'yes'}]}\n"
                         "def token_f1_score(prediction, reference):\n    return float(prediction == reference)\n",
                         encoding='utf-8')
    workspace = tmp_path/'case'
    workspace.mkdir()
    save(workspace/'manifest.json', {'question_id': 'q1', 'paper_id': 'p',
        'question': 'Original question?', 'gold_answers_sent': False, 'gold_evidence_sent': False})
    save(workspace/'reader-result.json', {'question': 'Original question?',
        'reader': {'answer': 'yes', 'status': 'REVIEW_REQUIRED'},
        'persisted_run': {'status': 'COMPLETED', 'used_llm_calls': 2, 'used_tokens': 10}})
    return workspace, dataset, evaluator


def test_completed_prediction_scored_without_claiming_factual_correctness(experiment):
    workspace, dataset, evaluator = experiment
    result = score([workspace], dataset, evaluator)
    assert result['raw_f1'] == 1.0
    assert result['official_benchmark_comparable'] is False
    assert result['rows'][0]['result_sha256']


@pytest.mark.parametrize('status', ['FAILED', 'RUNNING', 'BUDGET_EXHAUSTED', 'CANCELLED'])
def test_noncompleted_answer_matching_gold_still_gets_zero(experiment, status):
    workspace, dataset, evaluator = experiment
    original = json.loads((workspace/'reader-result.json').read_text())
    original['persisted_run']['status'] = status
    save(workspace/'reader-result.json', original)
    result = score([workspace], dataset, evaluator)
    assert result['count'] == 1 and result['raw_f1'] == 0
    assert result['rows'][0]['original_answer'] == 'yes'
    assert result['invalid_predictions'] == 1


def test_missing_result_keeps_requested_question_in_denominator(experiment):
    workspace, dataset, evaluator = experiment
    (workspace/'reader-result.json').unlink()
    result = score([workspace], dataset, evaluator)
    assert result['count'] == 1 and result['raw_f1'] == 0
    assert result['rows'][0]['run_status'] == 'MISSING'
    assert result['rows'][0]['calls'] is None


@pytest.mark.parametrize('change', ['question', 'paper', 'dataset_hash', 'gold_evidence'])
def test_changed_identity_or_oracle_input_rejected(experiment, change):
    workspace, dataset, evaluator = experiment
    manifest = json.loads((workspace/'manifest.json').read_text())
    if change == 'question':
        manifest['question'] = 'Easier question?'
        result = json.loads((workspace/'reader-result.json').read_text())
        result['question'] = manifest['question']
        save(workspace/'reader-result.json', result)
    elif change == 'paper':
        manifest['paper_id'] = 'other'
    elif change == 'dataset_hash':
        manifest['dataset_sha256'] = 'wrong'
    else:
        manifest['gold_evidence_sent'] = True
    save(workspace/'manifest.json', manifest)
    with pytest.raises(ValueError):
        score([workspace], dataset, evaluator)


def test_duplicate_case_rejected(experiment):
    workspace, dataset, evaluator = experiment
    with pytest.raises(ValueError):
        score([workspace, workspace], dataset, evaluator)
