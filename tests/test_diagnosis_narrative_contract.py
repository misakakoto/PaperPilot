"""Exercise rejected narrative + correction through the real diagnosis harness."""

import json
import pytest

from paperpilot_agentic.providers import LLMResponse
from test_diagnosis_report_v2 import provider
from test_imported_diagnosis_harness import setup
from paperpilot_agentic.diagnostic_facts import validate_narrative_contract


@pytest.mark.parametrize("prose", ["在高覆盖度组，所有种子的召回率均下降。", "各组F1和IoU下降。"])
def test_observed_trend_cannot_reenter_through_free_hypothesis(tmp_path, prose):
    runtime, harness, reader, binding = setup(tmp_path)
    normal = provider(reader)
    attempted = []

    def complete(messages, **kwargs):
        response = normal(messages, **kwargs)
        value = json.loads(response.text)
        if value["action"] == "report" and not attempted:
            attempted.append(True)
            value["arguments"]["hypotheses"][0]["text"] = prose
            return LLMResponse(
                text=json.dumps(value),
                model=reader.model,
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=10,
            )
        return response

    reader.complete = complete
    rid = harness.start_imported_diagnosis(
        binding["bundle_id"],
        "narrative-real-failure",
        comparison_mode="tools",
        report_protocol="metric-facts-v2",
    )["run"]["run_id"]
    result = harness.advance(rid)
    episode = result["research_episode"]
    assert episode["phase"] == "COMPLETED", episode
    assert episode["turn"] == 5, "Bad observed-trend prose was accepted without a correction"
    assert prose not in result["final_answer"]["answer"]
    assert episode["analysis"]["observations"]
    assert episode["new_gpu_starts"] == 0
    calls = len(reader.messages)
    harness.advance(rid)
    assert len(reader.messages) == calls


@pytest.mark.parametrize(
    "text",
    [
        "种子303的F1下降主要由召回率大幅下降驱动。",
        "The high-coverage group recall fell across all seeds.",
        "Precision and recall both improved.",
        "P/R trade-off explains the mechanism.",
        "不同分组的精确率和召回率均上升。",
        "种子303的F1下降。",
        "结果IoU提升。",
        "在组内Precision-Recall权衡发生变化。",
        "当前Recall下降。",
    ],
)
def test_trend_prose_is_a_representation_violation_even_if_correct(text):
    with pytest.raises(ValueError, match="OBSERVATION_PROSE_REQUIRES_ID.*representation rejection"):
        validate_narrative_contract("report", "Use observations.", {"hypotheses": [{"text": text}]})


def test_mechanism_scope_limits_and_future_predictions_are_not_called_false():
    result = validate_narrative_contract(
        "report",
        "Request a controlled measurement.",
        {
            "hypotheses": [
                {
                    "text": "Possible gradient routing changes require measurement.",
                    "observation_ids": ["pattern/coverage/1/f1"],
                    "alternatives": ["Initialization and optimizer dynamics remain unmeasured."],
                }
            ],
            "limitations": ["Precision and recall alone do not identify the cause."],
            "next_experiment": {
                "expected_patterns": [
                    "If the hypothesis holds, F1 may rise; otherwise no consistent shift."
                ],
                "falsification": "Under the controlled intervention, recall fails to increase.",
            },
        },
    )
    assert result["assurance"].startswith("LEXICAL_REPRESENTATION_ONLY")
    validate_narrative_contract(
        "delegate_analysis",
        "Request correction.",
        {"feedback": "Correct the unsupported claim: recall decreased for every seed."},
    )


@pytest.mark.parametrize(
    "prose",
    [
        "The F16 block is a possible cause of the drop.",
        "myPrecisionModule may change the upstream signal.",
        "The classifier1 module may change the downsampling path.",
    ],
)
def test_metric_substrings_in_module_identifiers_are_not_metrics(prose):
    validate_narrative_contract("report", "Request measurement.", {"hypotheses": [{"text": prose}]})


def test_tool_purpose_is_unverified_trace_not_a_rejected_observation(tmp_path):
    _, harness, reader, binding = setup(tmp_path)
    normal = provider(reader)

    def complete(messages, **kwargs):
        response = normal(messages, **kwargs)
        value = json.loads(response.text)
        if value["action"] in {"inspect_experiment", "delegate_analysis"}:
            value["reason"] = "检查召回率是否下降，并请分析者验证F1变化。"
            return LLMResponse(
                text=json.dumps(value),
                model=reader.model,
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=10,
            )
        return response

    reader.complete = complete
    rid = harness.start_imported_diagnosis(
        binding["bundle_id"],
        "tool-purpose-is-not-fact",
        comparison_mode="tools",
        report_protocol="metric-facts-v2",
    )["run"]["run_id"]
    result = harness.advance(rid)
    episode = result["research_episode"]
    assert episode["phase"] == "COMPLETED", episode
    assert episode["turn"] == 4
    tools = [
        d
        for d in episode["decisions"]
        if d["action"] in {"inspect_experiment", "delegate_analysis"}
    ]
    assert tools and all(
        d["reason_assurance"] == "UNVERIFIED_TOOL_PURPOSE_NOT_EVIDENCE" for d in tools
    )
    assert all(d["narrative_check"]["unverified_trace_paths"] for d in tools)
    assert "检查召回率是否下降" not in result["final_answer"]["answer"]
    assert episode["analysis"]["observations"] and episode["new_gpu_starts"] == 0


def test_unverified_tool_purpose_is_not_certified_even_when_it_asserts_a_trend():
    check = validate_narrative_contract(
        "inspect_experiment", "所有组Recall下降。", {"view": "coverage"}
    )
    assert check["unverified_trace_paths"] == ["inspect_experiment.reason"]
    assert "inspect_experiment.reason" not in check["checked_paths"]
    assert "not semantic" in check["assurance"]


@pytest.mark.parametrize("field", ["reason", "rationale"])
def test_lead_cannot_publish_a_second_free_trend_summary(tmp_path, field):
    _, harness, reader, binding = setup(tmp_path)
    normal = provider(reader)
    attempted = []

    def complete(messages, **kwargs):
        response = normal(messages, **kwargs)
        value = json.loads(response.text)
        if value["action"] == "finish" and not attempted:
            attempted.append(True)
            if field == "reason":
                value["reason"] = "所有种子的召回率均下降。"
            else:
                value["arguments"]["rationale"] = "所有种子的召回率均下降。"
            return LLMResponse(
                text=json.dumps(value),
                model=reader.model,
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=10,
            )
        return response

    reader.complete = complete
    rid = harness.start_imported_diagnosis(
        binding["bundle_id"],
        "lead-narrative-" + field,
        comparison_mode="summary",
        report_protocol="metric-facts-v2",
    )["run"]["run_id"]
    result = harness.advance(rid)
    assert result["research_episode"]["phase"] == "COMPLETED"
    assert result["research_episode"]["turn"] == 4
    assert "所有种子的召回率均下降" not in result["research_episode"]["stop_reason"]
    assert result["research_episode"]["decisions"][-1]["narrative_check"]["version"]


def test_repeated_bad_prose_stops_but_preserves_host_observations(tmp_path):
    _, harness, reader, binding = setup(tmp_path)
    normal = provider(reader)

    def complete(messages, **kwargs):
        response = normal(messages, **kwargs)
        value = json.loads(response.text)
        if value["action"] == "report":
            value["arguments"]["hypotheses"][0]["text"] = "所有种子的召回率均下降。"
            return LLMResponse(
                text=json.dumps(value),
                model=reader.model,
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=10,
            )
        return response

    reader.complete = complete
    rid = harness.start_imported_diagnosis(
        binding["bundle_id"],
        "repeated-narrative-error",
        comparison_mode="summary",
        report_protocol="metric-facts-v2",
    )["run"]["run_id"]
    result = harness.advance(rid)
    episode = result["research_episode"]
    assert episode["phase"] == "PARTIAL"
    assert episode["analysis"] is None
    assert episode["retained_host_observations"]
    assert all(v["source_refs"] for v in episode["retained_host_observations"].values())
    assert episode["turn"] == 3
    assert "宿主生成的观测" in result["final_answer"]["answer"]
    assert "所有种子的召回率均下降" not in result["final_answer"]["answer"]
    assert "没有发布机制解释" in result["final_answer"]["answer"]
    assert episode["new_gpu_starts"] == 0
    calls = len(reader.messages)
    harness.advance(rid)
    assert len(reader.messages) == calls
