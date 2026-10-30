import json

from test_diagnosis_content_review import revised_provider
from test_imported_diagnosis_harness import setup


def test_review_packet_projects_duplicate_analysis_and_keeps_complete_source_rows(tmp_path):
    _, harness, reader, binding = setup(tmp_path, seeds=(3, 5, 9))
    reader.complete = revised_provider(reader, reject=1)
    rid = harness.start_imported_diagnosis(binding["bundle_id"], "prompt-analysis-projection",
        comparison_mode="tools", report_protocol="metric-facts-v3")["run"]["run_id"]
    result = harness.advance(rid)
    assert result["research_episode"]["phase"] == "COMPLETED"
    packet = next(json.loads(m["content"]) for messages in reversed(reader.messages)
                  for m in messages if m["role"] == "user" and "independent_analysis" in m["content"])
    analysis = result["research_episode"]["analysis"]
    projected = packet["independent_analysis"]
    assert "facts" not in projected
    assert "observations" not in projected
    assert "interpretations" not in projected
    assert "next_experiment" not in projected
    assert projected["hypotheses"] == analysis["hypotheses"]
    assert projected["limitations"] == analysis["limitations"]
    assert projected["experiment_contract"] == analysis["experiment_contract"]
    assert set(projected["observation_ids"]) == set(analysis["observations"])
    assert set(projected["fact_refs"]) == set(analysis["facts"])
    assert packet["analysis_feedback"]  # A current correction must not disappear.
    for ref, expected in analysis["facts"].items():
        view, *parts = ref.split("/")
        actual = packet["observations"][view]["rows"]
        for part in parts:
            actual = actual[int(part)] if isinstance(actual, list) else actual[part]
        assert actual == expected
    for key, value in analysis["observations"].items():
        assert packet["host_observation_catalog"][key]["source_refs"] == value["source_refs"]
    assert result["research_episode"]["seeds"] == [3, 5, 9]
    assert result["research_episode"]["analysis_history"]
    calls = len(reader.messages)
    harness.advance(rid)
    assert len(reader.messages) == calls
