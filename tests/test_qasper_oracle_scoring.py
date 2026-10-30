import copy

import pytest

from evals.qasper.score_oracle import score


class EvaluatorStub:
    """Tests orchestration only; real official script is checked separately offline."""

    @staticmethod
    def get_answers_and_evidence(data, text_only):
        return {qa['question_id']: [{'answer': 'Yes'}]
                for paper in data.values() for qa in paper['qas']}

    @staticmethod
    def normalize_answer(text):
        return text.lower().strip()

    @staticmethod
    def token_f1_score(prediction, reference):
        return float(prediction.lower().strip() == reference.lower().strip())

    @staticmethod
    def evaluate(gold, predicted):
        return {'Answer F1': sum(EvaluatorStub.token_f1_score(p['answer'], 'Yes')
                                 for p in predicted.values()) / len(gold),
                'Missing predictions': len(gold) - len(predicted)}


def fixture():
    dataset = {'p': {'qas': [{'question_id': q, 'question': q} for q in ('a', 'b')]}}
    report = {'kind': 'qasper_v0.3_oracle_evidence_reader_dev_smoke', 'dataset_split': 'dev',
              'cases': [{'paper_id': 'p', 'question_id': q, 'question': q,
                         'answer_type': 'boolean', 'prediction': 'Yes',
                         'usage': {'finish_reason': 'stop'}} for q in ('a', 'b')]}
    return dataset, report


@pytest.mark.parametrize('failure', ['missing', 'truncated', 'error'])
def test_failed_prediction_stays_in_denominator(failure):
    data, report = fixture()
    if failure == 'missing':
        report['cases'][1]['prediction'] = None
    elif failure == 'truncated':
        report['cases'][1]['usage']['finish_reason'] = 'length'
    else:
        report['cases'][1]['error'] = 'TimeoutError'
    result = score(report, data, EvaluatorStub)
    assert result['answer_f1'] == result['exact_match'] == .5
    assert result['missing_or_failed'] == 1
    assert result['evidence_f1'] is None
    assert result['official_benchmark_comparable'] is False


def test_duplicate_unknown_and_changed_questions_rejected():
    data, original = fixture()
    for mutation in ('duplicate', 'unknown', 'changed'):
        report = copy.deepcopy(original)
        if mutation == 'duplicate':
            report['cases'][1] = report['cases'][0]
        elif mutation == 'unknown':
            report['cases'][1]['question_id'] = 'unknown'
        else:
            report['cases'][1]['question'] = 'reworded'
        with pytest.raises(ValueError):
            score(report, data, EvaluatorStub)
