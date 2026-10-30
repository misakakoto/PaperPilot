import json

import pytest

from paperpilot_agentic.providers import LLMResponse
from test_diagnosis_content_review import revised_provider
from test_imported_diagnosis_harness import setup


@pytest.mark.parametrize("decision", ["STOP_INSUFFICIENT_EVIDENCE", "PROPOSE_EXPERIMENT"])
def test_final_decision_projects_host_observations_without_republishing_trace(tmp_path, decision):
    _, harness, reader, binding = setup(tmp_path)
    original = revised_provider(reader)

    def complete(messages, **kwargs):
        response = original(messages, **kwargs)
        action = json.loads(response.text)
        if action["action"] == "finish":
            action["arguments"]["decision"] = decision
            action["arguments"]["rationale"] = "UNVERIFIED_FALSE_TREND in decision trace"
        return LLMResponse(text=json.dumps(action), model=reader.model, finish_reason="stop",
                           prompt_tokens=10, completion_tokens=10)

    reader.complete = complete
    rid = harness.start_imported_diagnosis(binding["bundle_id"], "final-decision-projection",
        comparison_mode="summary", report_protocol="metric-facts-v3")["run"]["run_id"]
    result = harness.advance(rid)
    episode = result["research_episode"]
    assert episode["phase"] == "COMPLETED"
    canonical = "NEED_MORE_EVIDENCE" if decision == "STOP_INSUFFICIENT_EVIDENCE" else decision
    final = episode["final_decision"]
    assert final["decision"] == canonical
    assert final["analyst_recommendation"] == "PROPOSE_EXPERIMENT"
    assert final["execution_authorized"] is False
    answer = result["final_answer"]["answer"]
    assert "负责人最终决策：" + canonical in answer
    assert "UNVERIFIED_FALSE_TREND" not in answer
    assert "UNVERIFIED_FALSE_TREND" not in episode["stop_reason"]
    assert "UNVERIFIED_FALSE_TREND" not in str(result["final_answer"]["limitations"])
    for observation in episode["analysis"]["observations"].values():
        assert observation["text"] in answer
    assert "F1下降可能" in answer  # Tentative hypotheses must not all be hidden to pass.
    assert "Layer response" in answer  # Preserve the analyst's useful proposed measurement.
    if canonical == "NEED_MORE_EVIDENCE":
        assert "分析者备选方案（负责人暂缓，未授权执行）" in answer
        assert "负责人采纳的实验建议" not in answer
    else:
        assert "负责人采纳的实验建议（仍未授权执行）" in answer
    with harness.db.connect() as conn:
        turns = conn.execute("SELECT raw_json FROM research_episode_turns WHERE run_id=?", (rid,)).fetchall()
    assert "UNVERIFIED_FALSE_TREND" in str([dict(t) for t in turns])
    calls = len(reader.messages)
    harness.advance(rid)
    assert len(reader.messages) == calls
    assert episode["new_gpu_starts"] == 0
