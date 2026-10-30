import json
import pytest
from paperpilot_agentic.providers import LLMResponse
from test_imported_diagnosis_harness import setup
from test_diagnosis_report_v2 import provider
from paperpilot_agentic.diagnostic_validation import resolve_finish_decision


@pytest.mark.parametrize(
    "decision",
    ["STOP_INSUFFICIENT_EVIDENCE", "NEED_MORE_EVIDENCE", "PROPOSE_EXPERIMENT", "unknown"],
)
def test_finish_decision(tmp_path, decision):
    _, harness, reader, binding = setup(tmp_path)
    normal = provider(reader, stop=True)

    def complete(messages, **kwargs):
        response = normal(messages, **kwargs)
        value = json.loads(response.text)
        if value["action"] == "finish":
            value["arguments"]["decision"] = decision
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
        binding["bundle_id"], "finish-contract-" + decision, report_protocol="metric-facts-v2"
    )["run"]["run_id"]
    result = harness.advance(rid)
    episode = result["research_episode"]
    if decision in {"STOP_INSUFFICIENT_EVIDENCE", "NEED_MORE_EVIDENCE"}:
        assert episode["phase"] == "COMPLETED", episode["stop_reason"]
        trace = episode["decisions"][-1]["decision_resolution"]
        assert trace["submitted_decision"] == decision
        assert trace["canonical_decision"] == "NEED_MORE_EVIDENCE"
        assert episode["stop_reason"].startswith("NEED_MORE_EVIDENCE\n")
    else:
        assert episode["phase"] == "PARTIAL"
    assert episode["new_gpu_starts"] == 0
    calls = len(reader.messages)
    harness.advance(rid)
    assert len(reader.messages) == calls


@pytest.mark.parametrize("value", ["stop", "STOP_AND_TRAIN", "", None, [], 1])
def test_unknown_finish_decisions_are_not_guessed(value):
    with pytest.raises(ValueError, match="finish.decision must be a registered decision"):
        resolve_finish_decision(value)
