from types import SimpleNamespace

import pytest

from paperpilot_agentic.direct_answer import answer_from_evidence, prepare_evidence


class Model:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.messages = messages
        kwargs["before_attempt"]()
        self.calls += 1
        assert kwargs["json_mode"] is False
        return SimpleNamespace(text=self.text, finish_reason="stop", model="fixture",
                               prompt_tokens=20, completion_tokens=10)


def test_reader_keeps_answer_and_flags_unknown_citations():
    model = Model("结果为 80 [evidence_missing]。")
    result = answer_from_evidence("结果？", [{"evidence_id": "evidence_a", "content": "80"}], model)
    assert result["answer"] == model.text
    assert result["structural_issues"] == ["unknown_citations"]
    assert result["semantic_correctness"] == "NOT_EVALUATED"
    assert result["provider_attempts"] == 1


def test_valid_citation_does_not_imply_correct_answer():
    model = Model("错误结论 [evidence_a]。")
    result = answer_from_evidence("结果？", [{"evidence_id": "evidence_a", "content": "80"}], model)
    assert result["structural_checks_passed"]
    assert result["status"] == "REVIEW_REQUIRED"


def test_missing_evidence_does_not_call_provider():
    model = Model("unused")
    assert answer_from_evidence("问题", [], model)["status"] == "NO_EVIDENCE"
    assert model.calls == 0


def test_context_overflow_and_conflicting_ids_are_explicit():
    with pytest.raises(ValueError, match="character cap"):
        prepare_evidence([{"evidence_id": "a", "content": "long"}], max_context_chars=1)
    with pytest.raises(ValueError, match="Conflicting"):
        prepare_evidence([{"evidence_id": "a", "content": "first"},
                          {"evidence_id": "a", "content": "second"}])


def test_retries_cannot_exceed_one_attempt():
    class RetryModel:
        def complete(self, messages, **kwargs):
            kwargs["before_attempt"]()
            kwargs["before_attempt"]()
    with pytest.raises(RuntimeError, match="one upstream"):
        answer_from_evidence("问题", [{"evidence_id": "a", "content": "fact"}], RetryModel())


def test_prompt_policy_is_explicit_and_fingerprinted_without_changing_evidence():
    rows = [{"evidence_id": "a", "content": "Source fact"}]
    original = Model("Answer [a]")
    concise = Model("Answer [a]")
    first = answer_from_evidence("What is reported?", rows, original)
    second = answer_from_evidence("What is reported?", rows, concise, concise_policy=True)
    assert first["input_sha256"] == second["input_sha256"]
    assert first["prompt_sha256"] != second["prompt_sha256"]
    assert first["reader_protocol"] == "direct-original-alias-v2"
    assert second["reader_protocol"] == "direct-concise-alias-v2"
    assert original.messages[1] == concise.messages[1]
    assert "English questions require English answers" in concise.messages[0]["content"]
    assert second["semantic_correctness"] == "NOT_EVALUATED"
    assert second["answer"] == concise.text


def test_empty_evidence_still_records_protocol_without_spending():
    model = Model("unused")
    result = answer_from_evidence("What?", [], model, concise_policy=True)
    assert result["reader_protocol"] == "direct-concise-alias-v2"
    assert len(result["prompt_sha256"]) == 64
    assert model.calls == 0


def test_prose_aliases_restore_only_exact_citation_tokens():
    identifier = 'evidence_' + 'a' * 32
    model = Model('ref1 is ordinary text. Fact [ref1]. Unknown [ref999].')
    result = answer_from_evidence('Question?', [{'evidence_id': identifier, 'content': 'Fact'}], model)
    assert result['raw_answer'] == model.text
    assert result['answer'] == f'ref1 is ordinary text. Fact [{identifier}]. Unknown [ref999].'
    assert result['unknown_citation_ids'] == ['ref999']
    assert result['structural_issues'] == ['unknown_citations']
