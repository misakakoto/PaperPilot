"""Regression: a content-review failure must not erase a structurally valid draft."""
import json

import pytest

from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.intelligence import LLMAnswerSynthesizer
from paperpilot_agentic.llm_runtime import (
    AnswerDraftRejected, LLMStructuredOutputError, PersistentLLMRuntime,
)
from paperpilot_agentic.models import Budget
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.providers import LLMResponse


class Model:
    model = "fixture"

    def __init__(self):
        self.calls = 0

    def complete(self, messages, **kwargs):
        kwargs["before_attempt"]()
        self.calls += 1
        return LLMResponse(text=json.dumps({"answer": f"draft-{self.calls}"}),
                           prompt_tokens=10, completion_tokens=10, finish_reason="stop")


@pytest.mark.parametrize("repair_cap", [0, 1])
def test_exhausted_repair_preserves_content_draft(tmp_path, repair_cap):
    db = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    run_id = db.create_run("question", Budget(max_llm_calls=4, max_repair_calls=repair_cap))
    model = Model()
    runtime = PersistentLLMRuntime(db, model)

    def validate(value):
        raise AnswerDraftRejected("mechanism review", value)

    with pytest.raises(LLMStructuredOutputError) as error:
        runtime.complete_json(run_id, [], policy="synthesizer", validator=validate)
    assert error.value.rejected_answer["answer"] == f"draft-{1 + repair_cap}"
    assert model.calls == 1 + repair_cap


def test_invalid_citation_does_not_produce_deliverable_draft(tmp_path):
    db = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    run_id = db.create_run("question", Budget(max_llm_calls=4))

    def validate(value):
        raise ValueError("evidence not linked to claim")

    with pytest.raises(LLMStructuredOutputError) as error:
        PersistentLLMRuntime(db, Model()).complete_json(
            run_id, [], policy="synthesizer", validator=validate,
        )
    assert error.value.rejected_answer is None


def test_draft_is_persisted_with_warning_and_not_answered(tmp_path):
    class Synthesizer:
        calls = 0

        def synthesize(self, **kwargs):
            self.calls += 1
            raise AnswerDraftRejected("mechanism review", {
                "answer": "Parallel 的机制解释需要核验。",
                "citation_ids": [], "covered_claim_ids": [],
                "review_claim_ids": [],
            })

    synthesizer = Synthesizer()
    runtime = ResearchAgentRuntime(db_path=tmp_path / "runtime.sqlite3",
                                   workspace=tmp_path, answer_synthesizer=synthesizer)
    run_id = runtime.db.create_run("如何处理？", Budget())
    assert runtime._ensure_final_answer(run_id).value == "DEGRADED"
    answer = runtime.db.get_run(run_id)["metadata"]["final_answer"]
    assert answer["original_answer"] == "Parallel 的机制解释需要核验。"
    assert "待核验草稿" in answer["answer"]
    assert answer["review_required"] is True
    assert answer["mode"] == "llm-draft-review-required-v1"
    runtime._ensure_final_answer(run_id)
    assert synthesizer.calls == 1


def test_real_synthesis_preserves_table_when_mechanism_repair_fails(tmp_path):
    class RepeatedBadMechanism(Model):
        def complete(self, messages, **kwargs):
            kwargs["before_attempt"]()
            self.calls += 1
            return LLMResponse(text=json.dumps({"sections": [
                {"claim_id": "claim_method", "text": "Parallel 使用独立分支融合时相特征。",
                 "evidence_ids": ["evidence_method"]},
                {"claim_id": "claim_table", "text": "基线 F1 为 81.11，完整组合 F1 为 82.83。",
                 "evidence_ids": ["evidence_table"]},
            ]}, ensure_ascii=False), prompt_tokens=10, completion_tokens=10,
                finish_reason="stop")

    db = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    run_id = db.create_run("Parallel 机制如何组织特征？请给出消融结果。", Budget(max_llm_calls=4))
    model = RepeatedBadMechanism()
    claims = [
        {"claim_id": "claim_method", "statement": "Parallel mechanism", "status": "SUPPORTED"},
        {"claim_id": "claim_table", "statement": "Ablation results", "status": "SUPPORTED"},
    ]
    evidence = [
        {"evidence_id": "evidence_method", "claim_id": "claim_method", "polarity": "SUPPORTS",
         "content": "Parallel concatenates tokens in the channel dimension for joint modeling."},
        {"evidence_id": "evidence_table", "claim_id": "claim_table", "polarity": "SUPPORTS",
         "content": "Baseline F1 81.11; full combination F1 82.83."},
    ]
    with pytest.raises(LLMStructuredOutputError) as failure:
        LLMAnswerSynthesizer(PersistentLLMRuntime(db, model)).synthesize(
            question="Parallel 机制如何组织特征？请给出消融结果。", claims=claims,
            evidence=evidence, run_id=run_id,
        )
    draft = failure.value.rejected_answer
    assert draft is not None
    assert "81.11" in draft["section_text_map"]["claim_table"]
    assert "82.83" in draft["section_text_map"]["claim_table"]
    assert draft["review_claim_ids"] == ["claim_method"]
    assert model.calls == 2
