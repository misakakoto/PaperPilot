import json

import pytest

from paperpilot_agentic.providers import LLMResponse
from test_imported_diagnosis_harness import setup


def provider(reader, *, stop=False):
    def complete(messages, before_attempt=None, **kwargs):
        before_attempt()
        reader.messages.append(messages)
        task = json.loads(next(m["content"] for m in messages if m["role"] == "user"))
        if "lead" in messages[0]["content"].split(".")[0]:
            action, args = (
                (
                    "finish",
                    {
                        "decision": "STOP",
                        "rationale": "Observed predictions cannot establish an internal cause.",
                    },
                )
                if task["independent_analysis"]
                else ("delegate_analysis", {})
            )
        elif "coverage" in task["available_views"] and "coverage" not in task["observations"]:
            action, args = "inspect_experiment", {"view": "coverage"}
        else:
            ids = task["required_observation_ids"]
            chosen = list(task["host_observation_catalog"])
            refs = task["required_report_fact_refs"]
            extra = next(
                (
                    v["source_refs"][0]
                    for k, v in task["host_observation_catalog"].items()
                    if "/coverage/" in k
                ),
                refs[0],
            )
            action, args = (
                "report",
                {
                    "fact_refs": refs,
                    "observation_ids": chosen,
                    "hypotheses": []
                    if stop
                    else [
                        {
                            "text": "A possible response difference requires controlled measurement.",
                            "observation_ids": ids,
                            "alternatives": ["Unmeasured optimization dynamics"],
                        }
                    ],
                    "limitations": ["Imported training is user-declared; no new training."],
                    "next_experiment": {
                        "hypothesis": "Measure a controlled response difference before making a causal claim.",
                        "support_refs": [refs[0]],
                        "counter_refs": [extra],
                        "missing_observations": ["Layer responses"],
                        "falsification": "No predicted response difference under the controlled intervention.",
                        "decision": "STOP_INSUFFICIENT_EVIDENCE" if stop else "PROPOSE_EXPERIMENT",
                        "intervention": "Observe a bounded isolated intervention; execution is not authorized here.",
                        "controls": ["Same inputs and weights"],
                        "measurements": ["Layer response"],
                        "expected_patterns": ["Predicted response difference versus no difference"],
                        "alternatives_remaining": [
                            "Optimization confound not independently measured"
                        ],
                    },
                },
            )
        return LLMResponse(
            text=json.dumps(
                {"action": action, "reason": "Use host observations only.", "arguments": args}
            ),
            model=reader.model,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=10,
        )

    return complete


@pytest.mark.parametrize("mode", ["summary", "tools"])
@pytest.mark.parametrize("stop", [False, True])
def test_v2_projects_host_patterns_into_report_and_lead_summary_without_training(
    tmp_path, mode, stop
):
    runtime, harness, reader, binding = setup(tmp_path)
    reader.complete = provider(reader, stop=stop)
    rid = harness.start_imported_diagnosis(
        binding["bundle_id"],
        "new-observation-report-v2",
        comparison_mode=mode,
        report_protocol="metric-facts-v2",
    )["run"]["run_id"]
    result = harness.advance(rid)
    episode = result["research_episode"]
    assert episode["phase"] == "COMPLETED", episode
    analysis = episode["analysis"]
    assert analysis["report_protocol"] == "metric-facts-v2"
    assert analysis["observations"] and analysis["projection_record"]["source_refs"]
    assert all(v["text"] in episode["stop_reason"] for v in analysis["observations"].values())
    assert "宿主生成的观测" in result["final_answer"]["answer"]
    assert analysis["experiment_contract"]["alternatives_remaining"]
    assert not runtime.db.list_approvals(rid) and episode["new_gpu_starts"] == 0
    calls = len(reader.messages)
    harness.advance(rid)
    assert len(reader.messages) == calls
    with pytest.raises(ValueError, match="different inputs"):
        harness.start_imported_diagnosis(
            binding["bundle_id"], "new-observation-report-v2", comparison_mode=mode
        )


def test_unknown_observation_and_legacy_experiment_cannot_pass_v2_contract(tmp_path):
    _, harness, reader, binding = setup(tmp_path)
    reader.complete = provider(reader)
    rid = harness.start_imported_diagnosis(
        binding["bundle_id"], "invalid-pattern-report-v2", report_protocol="metric-facts-v2"
    )["run"]["run_id"]
    from paperpilot_agentic.research_diagnosis import DiagnosisHarness

    d = DiagnosisHarness(harness.runtime)
    s = d._load(rid)
    with pytest.raises(ValueError, match="existing host"):
        d._prepare_v2_report(
            s,
            {
                "fact_refs": [],
                "observation_ids": ["invented"],
                "hypotheses": [],
                "limitations": ["unknown"],
                "next_experiment": {},
            },
        )
