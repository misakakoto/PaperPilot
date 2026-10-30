import pytest

from evals.qasper.validate_report import validate_report


def fixture_report():
    return {
        "benchmark_name": "qasper-v0.3",
        "dataset_split": "test",
        "case_count": 2,
        "successful_predictions": 1,
        "failed_predictions": 1,
        "answer_f1": 0.5,
        "answer_exact_match": 0.5,
        "evidence_f1": 0.25,
        "cases": [
            {"question_id": "q1", "status": "scored", "answer_f1": 1,
             "exact_match": 1, "evidence_f1": 0.5},
            {"question_id": "q2", "status": "failed", "answer_f1": 0,
             "exact_match": 0, "evidence_f1": 0},
        ],
    }


def test_recomputes_historical_metrics_without_claiming_current_score():
    result = validate_report(fixture_report())
    assert result["passed"]
    assert result["recomputed"]["answer_f1"] == 0.5
    assert "does not rerun" in result["scope"]


def test_rejects_changed_aggregate_and_duplicate_question():
    report = fixture_report()
    report["answer_f1"] = 0.9
    with pytest.raises(ValueError, match="Aggregate metric mismatch"):
        validate_report(report)
    report = fixture_report()
    report["cases"][1]["question_id"] = "q1"
    with pytest.raises(ValueError, match="duplicated"):
        validate_report(report)
