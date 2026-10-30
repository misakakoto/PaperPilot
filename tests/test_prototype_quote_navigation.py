import copy
import pytest
from paperpilot_agentic.prototype_workspace import new_workspace, plan, complete
from paperpilot_agentic.prototype_analysis import inspect_quotes


def source():
    w = new_workspace("navigation")
    plan(
        w,
        [
            {
                "id": "paper",
                "goal": "source",
                "capability": "read_literature",
                "depends_on": [],
                "acceptance": "original",
            }
        ],
    )
    identity = complete(
        w,
        "paper",
        "read_literature",
        {
            "evidence": [
                {
                    "content": "Alpha method is explicit. Beta method is different. Gamma is a limitation.",
                    "evidence_type": "TEXT",
                    "source_uri": "fixture://paper",
                    "page": 1,
                }
            ]
        },
    )
    return w, identity


def test_returned_span_offset_is_a_valid_input_and_next_action_advances():
    w, identity = source()
    before = copy.deepcopy(w)
    first = inspect_quotes(w, {"artifact_id": identity, "span_offset": 0, "limit": 1})
    next_action = first["next_action"]
    second = inspect_quotes(w, next_action["arguments"])
    assert next_action["reason"] and second["span_offset"] == 1
    assert first["spans"][0]["quote_id"] != second["spans"][0]["quote_id"]
    assert w == before


def test_legacy_offsets_remain_exact_and_ambiguous_names_are_rejected():
    w, identity = source()
    a = inspect_quotes(w, {"artifact_id": identity, "offset": 1, "limit": 1})
    b = inspect_quotes(w, {"artifact_id": identity, "span_offset": 1, "limit": 1})
    assert a == b
    with pytest.raises(ValueError, match="both"):
        inspect_quotes(w, {"artifact_id": identity, "offset": 0, "span_offset": 1})


def test_query_selects_literal_method_spans_without_changing_quote_identity():
    from paperpilot_agentic.prototype_analysis import spans

    w, identity = source()
    before = copy.deepcopy(w)
    result = inspect_quotes(w, {"artifact_id": identity, "query": "Beta different", "limit": 1})
    assert result["spans"][0]["text"] == "Beta method is different."
    originals = {s["quote_id"]: s for s in spans(identity, w["artifacts"][identity])}
    assert result["spans"][0]["text"] == originals[result["spans"][0]["quote_id"]]["text"]
    assert w == before
    empty = inspect_quotes(w, {"artifact_id": identity, "query": "unmatchedterm"})
    assert empty["spans"] == [] and empty["next_action"] is None


def test_missing_rationale_normalization_preserves_original_request_and_parameters():
    from paperpilot_agentic.prototype_harness import PrototypeHarness

    w, identity = source()
    state = {
        "work": w,
        "sources": {},
        "mode": "adaptive",
        "literature_analysis_policy": "source-spans-analysis-v1",
        "action_envelope_policy": "optional-rationale-v1",
    }
    h = PrototypeHarness.__new__(PrototypeHarness)
    value = {
        "action": "inspect_quotes",
        "arguments": {"artifact_id": identity, "span_offset": 0, "limit": 1},
    }
    original = copy.deepcopy(value)
    result = h._apply("offline", state, value, 1)
    assert result["status"] == "OK" and value == original
    assert state["action_normalizations"][-1]["field"] == "reason"
    assert state["work"] == w


@pytest.mark.parametrize("case", ["legacy", "extra_field", "invalid_arguments"])
def test_optional_rationale_does_not_relax_action_or_argument_validation(case):
    from paperpilot_agentic.prototype_harness import PrototypeHarness

    w, identity = source()
    state = {
        "work": w,
        "sources": {},
        "mode": "adaptive",
        "literature_analysis_policy": "source-spans-analysis-v1",
        "action_envelope_policy": "optional-rationale-v1",
    }
    value = {"action": "inspect_quotes", "arguments": {"artifact_id": identity, "span_offset": 0}}
    if case == "legacy":
        state.pop("action_envelope_policy")
    elif case == "extra_field":
        value["unregistered"] = "x"
    else:
        value["arguments"]["unknown_parameter"] = "x"
    before = copy.deepcopy(state)
    with pytest.raises(ValueError):
        PrototypeHarness.__new__(PrototypeHarness)._apply("offline", state, value, 1)
    assert state == before


@pytest.mark.parametrize(
    "query", [None, True, [], "", " ", "x" * 501, " ".join(str(i) for i in range(33))]
)
def test_invalid_query_does_not_change_source(query):
    w, identity = source()
    before = copy.deepcopy(w)
    with pytest.raises(ValueError):
        inspect_quotes(w, {"artifact_id": identity, "query": query})
    assert w == before


def test_query_pagination_preserves_query_and_original_quote_ids():
    w, identity = source()
    first = inspect_quotes(w, {"artifact_id": identity, "query": "method", "limit": 1})
    assert first["next_action"]["arguments"]["query"] == "method"
    second = inspect_quotes(w, first["next_action"]["arguments"])
    assert first["spans"][0]["quote_id"] != second["spans"][0]["quote_id"]
    assert second["next_action"] is None
