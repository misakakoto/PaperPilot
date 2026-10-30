import json

import pytest

from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.research_contracts import diagnosis_action_schema, parse_diagnosis_action
from test_diagnosis_report_v2 import provider
from test_imported_diagnosis_harness import setup


def revised_provider(reader, *, reject=0, invalid_ref=False, missing_review=False):
    normal = provider(reader)
    finish_calls = []

    def complete(messages, **kwargs):
        response = normal(messages, **kwargs)
        value = json.loads(response.text)
        if value["action"] == "report":
            # Both legitimate and mistaken expressions must reach content review,
            # rather than a keyword rule claiming a factual verdict.
            value["arguments"]["hypotheses"][0]["text"] = (
                "F1下降可能需要激活观察解释；目前不能确定原因。"
            )
            if invalid_ref:
                value["arguments"]["hypotheses"][0]["observation_ids"] = ["pattern/fake"]
        elif value["action"] == "finish":
            finish_calls.append(value)
            if not missing_review:
                needs = len(finish_calls) <= reject
                value["arguments"]["content_review"] = {
                    "verdict": "NEEDS_REVISION" if needs else "ACCEPT_TENTATIVE",
                    "issues": [
                        "The proposed observation does not distinguish the alternative causes."
                    ]
                    if needs
                    else [],
                    "rationale": "Compare the hypothesis and discriminating measurement with the host observations.",
                }
                task = json.loads(next(m['content'] for m in messages if m['role'] == 'user'))
                if 'scientific_review_context' in task:
                    value['arguments']['content_review']['checks'] = [
                        {'aspect': aspect, 'status': 'SUPPORTED',
                         'statement': 'Tentative explanation; no mechanism identified from metrics alone',
                         'basis_refs': ['analysis/hypotheses', 'analysis/experiment_contract']}
                        for aspect in ('source_scope','causal_identification','discriminating_measurement','controlled_comparison')]
        return LLMResponse(
            text=json.dumps(value),
            model=reader.model,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=10,
        )

    return complete


@pytest.mark.parametrize("mode", ["summary", "tools"])
@pytest.mark.parametrize("reject", [0, 1, 2])
def test_v3_content_revision_is_bounded_and_preserves_facts(tmp_path, mode, reject):
    _, harness, reader, binding = setup(tmp_path)
    reader.complete = revised_provider(reader, reject=reject)
    run_id = harness.start_imported_diagnosis(
        binding["bundle_id"],
        "content-review-v3",
        comparison_mode=mode,
        report_protocol="metric-facts-v3",
    )["run"]["run_id"]
    result = harness.advance(run_id)
    episode = result["research_episode"]
    assert episode["phase"] == ("PARTIAL" if reject == 2 else "COMPLETED"), episode
    assert episode.get("analysis_revisions", 0) == min(reject, 1)
    assert len(episode["analysis_history"]) == min(reject, 1)
    assert episode["analysis"]["observations"]
    assert episode["assurance"]["facts"] == "HOST_RECOMPUTED"
    assert episode["assurance"]["inference"] == "NOT_INDEPENDENTLY_VERIFIED"
    assert "宿主生成的观测" in result["final_answer"]["answer"]
    if reject == 2:
        assert "F1下降可能" not in result["final_answer"]["answer"]
        assert "缺口" in result["final_answer"]["answer"]
    else:
        assert "F1下降可能" in result["final_answer"]["answer"]
    assert episode["turn"] == (3 if mode == "summary" else 4) + (2 if reject else 0)
    assert episode["new_gpu_starts"] == 0
    calls = len(reader.messages)
    harness.advance(run_id)
    assert len(reader.messages) == calls


@pytest.mark.parametrize("invalid_ref,missing_review", [(True, False), (False, True)])
def test_v3_still_blocks_unknown_sources_and_missing_content_review(
    tmp_path, invalid_ref, missing_review
):
    _, harness, reader, binding = setup(tmp_path)
    reader.complete = revised_provider(
        reader, invalid_ref=invalid_ref, missing_review=missing_review
    )
    rid = harness.start_imported_diagnosis(
        binding["bundle_id"],
        "content-invalid-v3",
        comparison_mode="summary",
        report_protocol="metric-facts-v3",
    )["run"]["run_id"]
    result = harness.advance(rid)
    assert result["research_episode"]["phase"] == "PARTIAL"
    assert "F1下降可能" not in result["final_answer"]["answer"]
    assert result["research_episode"]["retained_host_observations"]
    assert result["research_episode"]["new_gpu_starts"] == 0


def test_prompt_schema_and_parser_share_stop_decisions_and_role_permissions():
    schema = diagnosis_action_schema("lead")["finish"]["arguments"]
    decisions = schema["properties"]["decision"]["enum"]
    assert "STOP_INSUFFICIENT_EVIDENCE" in decisions
    assert "content_review" in schema["required"]
    for action in ["train", "approve", "submit_patch"]:
        with pytest.raises(ValueError):
            parse_diagnosis_action(
                "lead", {"action": action, "reason": "Not authorized", "arguments": {}}
            )
