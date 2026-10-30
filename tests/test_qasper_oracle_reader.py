import json

from evals.qasper.oracle_reader import build_cases, messages, oracle_evidence


def qa(kind):
    answer = {"unanswerable": kind == "unanswerable", "extractive_spans": [],
              "free_form_answer": "", "yes_no": None, "evidence": [f"evidence-{kind}"]}
    if kind == "extractive":
        answer["extractive_spans"] = ["secret gold span"]
    elif kind == "abstractive":
        answer["free_form_answer"] = "secret gold summary"
    elif kind == "boolean":
        answer["yes_no"] = True
    return {"question_id": kind, "question": f"question-{kind}",
            "answers": [{"answer": answer}]}


def test_oracle_evidence_deduplicates_across_annotations():
    value = qa("extractive")
    value["answers"].append(value["answers"][0])
    assert oracle_evidence(value) == ["evidence-extractive"]


def test_balanced_selection_is_deterministic_and_prompt_hides_gold_answers():
    dataset = {"paper": {"qas": [qa(kind) for kind in
               ("extractive", "abstractive", "boolean", "unanswerable")]}}
    cases = build_cases(dataset, limit=4)
    assert [item["answer_type"] for item in cases] == [
        "extractive", "abstractive", "boolean", "unanswerable"
    ]
    rendered = json.dumps(messages(cases[0]), ensure_ascii=False)
    assert "secret gold span" not in rendered
    assert "evidence-extractive" in rendered
