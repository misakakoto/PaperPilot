import json

import pytest

from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.diagnostic_validation import project_observed_refs
from test_diagnosis_report_v2 import provider
from test_imported_diagnosis_harness import setup


@pytest.mark.parametrize("mode", ["summary", "tools"])
@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize("stop", [False, True])
def test_registered_pattern_refs_project_through_actual_harness(tmp_path, mode, invalid, stop):
    _, harness, reader, binding = setup(tmp_path)
    normal = provider(reader, stop=stop)
    submitted = []

    def complete(messages, **kwargs):
        response = normal(messages, **kwargs)
        value = json.loads(response.text)
        if value["action"] == "report":
            task = json.loads(next(m["content"] for m in messages if m["role"] == "user"))
            args = value["arguments"]
            args.pop("fact_refs")
            args["next_experiment"]["support_refs"] = (
                ["pattern/not-an-observation"] if invalid else [task["required_observation_ids"][0]]
            )
            args["next_experiment"]["counter_refs"] = [
                next(
                    (k for k in task["required_observation_ids"] if "/coverage/" in k),
                    task["required_observation_ids"][-1],
                )
            ]
            submitted.append(json.loads(json.dumps(args)))
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
        "registered-ref-projection",
        comparison_mode=mode,
        report_protocol="metric-facts-v2",
    )["run"]["run_id"]
    result = harness.advance(rid)
    episode = result["research_episode"]
    if invalid:
        assert episode["phase"] == "PARTIAL"
        assert episode["analysis"] is None
        assert "unknown" in episode["stop_reason"].lower()
        assert episode["retained_host_observations"]
    else:
        assert episode["phase"] == "COMPLETED", episode
        analysis = episode["analysis"]
        refs = analysis["experiment_contract"]["support_refs"]
        assert refs == ["paired/0/delta/f1"]
        assert all(r in analysis["facts"] for r in refs)
        assert (
            analysis["projection_record"]["reference_resolution"]["selections"]["support_refs"]
            == submitted[-1]["next_experiment"]["support_refs"]
        )
        assert submitted[-1]["next_experiment"]["support_refs"][0].startswith("pattern/")
        assert episode["turn"] == (4 if mode == "tools" else 3)
    assert episode["new_gpu_starts"] == 0
    calls = len(reader.messages)
    harness.advance(rid)
    assert len(reader.messages) == calls


def test_reference_projection_preserves_numeric_compatibility_and_exact_membership():
    numeric = {"paired/0/delta/precision": 0.01, "paired/0/delta/recall": 0.02}
    patterns = {"pattern/paired/0/precision_recall": {"source_refs": list(numeric)}}
    values = ["pattern/paired/0/precision_recall", "paired/0/delta/precision"]
    projected, trace = project_observed_refs(values, numeric, patterns, label="support_refs")
    assert projected == list(numeric)
    assert trace[0]["numeric_refs"] == list(numeric)
    assert trace[1]["selected_id"] == values[1]
    for invalid in [["pattern/paired/0/precision_recal"], [0.99], ["paired/0/delta/f1"]]:
        with pytest.raises(ValueError, match="unknown observed"):
            project_observed_refs(invalid, numeric, patterns, label="support_refs")
