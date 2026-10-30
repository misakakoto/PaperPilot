import copy
import pytest
from paperpilot_agentic import prototype_design_contract as contract
from test_prototype_analysis import prepared, proposal
from test_prototype_literature import work


@pytest.mark.parametrize(
    "case", ["valid", "missing", "ordinal", "probabilities", "boolean_channel"]
)
def test_design_interface_gate_is_atomic_and_versioned(tmp_path, monkeypatch, case):
    _, h, rid, state, identity = prepared(tmp_path, monkeypatch)
    work(h, rid, state, "analysis", proposal(state, identity))
    aid = state["work"]["tasks"]["analysis"]["output"]
    finding = state["work"]["artifacts"][aid]["content"]["findings"][0]
    state["design_contract_policy"] = contract.POLICY
    content = {
        "architecture": "Falsifiable binary mechanism",
        "tensor_interfaces": copy.deepcopy(contract.INTERFACE),
        "literature_basis": [
            {"artifact_id": aid, "quote_id": finding["quote_id"], "interpretation": "hypothesis"}
        ],
    }
    if case == "missing":
        content.pop("tensor_interfaces")
    if case == "ordinal":
        content["tensor_interfaces"]["output"] = ["B", 4, "H", "W"]
    if case == "probabilities":
        content["tensor_interfaces"]["output_semantics"] = "sigmoid_probabilities"
    if case == "boolean_channel":
        content["tensor_interfaces"]["output"][1] = True
    before = copy.deepcopy(state)
    if case == "valid":
        work(h, rid, state, "design", content)
        assert state["work"]["tasks"]["design"]["status"] == "DONE"
    else:
        with pytest.raises(ValueError, match="binary.*interface"):
            work(h, rid, state, "design", content)
        assert state == before


def test_legacy_design_without_typed_contract_still_readable():
    contract.check({}, {"architecture": "legacy"})
    with pytest.raises(ValueError, match="Unknown"):
        contract.check({"design_contract_policy": "invented"}, {})


def test_new_design_prompt_supplies_literal_json_contract(tmp_path, monkeypatch):
    import json
    from paperpilot_agentic import prototype_prompt

    _, h, rid, state, identity = prepared(tmp_path, monkeypatch)
    work(h, rid, state, "analysis", proposal(state, identity))
    state.update(
        design_contract_policy="literal-binary-design-interface-v2",
        design_context_policy="direct-design-context-v1",
        instruction_policy=prototype_prompt.POLICY,
    )
    _, messages = h._context(rid, state)
    assert (
        json.dumps(contract.INTERFACE, ensure_ascii=False, sort_keys=True) in messages[0]["content"]
    )
    assert json.loads(messages[-1]["content"])["required_tensor_interfaces"] == contract.INTERFACE
