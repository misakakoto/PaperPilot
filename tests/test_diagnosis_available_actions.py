import json

import pytest
from jsonschema import Draft202012Validator

from paperpilot_agentic.research_diagnosis import DiagnosisHarness
from test_diagnosis_content_review import revised_provider
from test_imported_diagnosis_harness import setup


def packet(harness, rid):
    state = harness._load(rid)
    _, messages = harness._context(rid, state)
    return state, json.loads(messages[1]["content"])


def test_current_state_controls_prompt_actions_and_execution(tmp_path):
    _, facade, reader, binding = setup(tmp_path)
    reader.complete = revised_provider(reader)
    rid = facade.start_imported_diagnosis(binding["bundle_id"], "state-bound-tools-001",
        comparison_mode="tools", report_protocol="metric-facts-v3")["run"]["run_id"]
    harness = DiagnosisHarness(facade.runtime)
    state, task = packet(harness, rid)
    assert "finish" not in task["schema"]
    assert set(harness.allowed_actions(state)) == set(task["schema"])
    assert task["schema"]["inspect_experiment"]["arguments"]["properties"]["view"]["enum"] == ["coverage", "samples"]
    envelope = task["output_envelope"]
    assert "reason" in envelope["required"]
    assert not Draft202012Validator(envelope).is_valid({"action":"delegate_analysis", "arguments":{}})
    with pytest.raises(ValueError, match="not available"):
        harness._apply(rid, state, {"action":"finish", "reason":"Premature", "arguments":{
            "decision":"STOP", "rationale":"No analysis", "content_review":{
                "verdict":"ACCEPT_TENTATIVE", "issues":[], "rationale":"Unfounded"}}}, 1)
    while not state["analysis"]:
        facade.advance(rid, max_turns=1)
        state = harness._load(rid)
    state, task = packet(harness, rid)
    assert "finish" in task["schema"]
    delegate = task["schema"]["delegate_analysis"]["arguments"]
    assert "feedback" in delegate["required"]
    assert not Draft202012Validator(delegate).is_valid({})
    assert not Draft202012Validator(delegate).is_valid({"feedback":" "})
    assert task["schema"]["inspect_experiment"]["arguments"]["properties"]["view"]["enum"] == ["samples"]
    with pytest.raises(ValueError, match="not available"):
        harness._apply(rid,state,{"action":"inspect_experiment", "reason":"Duplicate", "arguments":{"view":"coverage"}},state["turn"]+1)
    result=facade.advance(rid)
    assert result["research_episode"]["phase"]=="COMPLETED"
    assert result["research_episode"]["new_gpu_starts"]==0
