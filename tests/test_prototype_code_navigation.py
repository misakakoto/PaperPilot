import copy
import pytest
from paperpilot_agentic import prototype_code_navigation as nav
from paperpilot_agentic.prototype_workspace import digest

SOURCE = 'raise RuntimeError("must never execute")\n\nclass A:\n    def forward(self, x):\n        return x + 1\n\nclass B:\n    def forward(self, x):\n        return x - 1\n'


def owned():
    artifact = {
        "name": "code",
        "kind": "implement",
        "inputs": [],
        "content": {"source": SOURCE, "config": {}},
    }
    identity = digest(artifact)
    return {
        "revision": 2,
        "artifacts": {identity: artifact},
        "tasks": {"code": {"status": "DONE", "output": identity}},
    }, identity


def test_exact_qualified_symbol_is_in_original_and_read_does_not_execute():
    work, identity = owned()
    before = copy.deepcopy(work)
    view = nav.inspect_code(work, {"artifact_id": identity, "symbol": "B.forward"})
    assert view["source"] == "    def forward(self, x):\n        return x - 1\n"
    assert view["start_line"] == 8 and view["end_line"] == 9
    assert view["unique_in_source"] is True and view["artifact_id"] == identity
    assert work == before


@pytest.mark.parametrize(
    "args",
    [
        {"line": True},
        {"line": 0},
        {"symbol": "forward"},
        {"symbol": "A.forward", "line": 4},
        {"line": 4, "path": "/other/model.py"},
        {"line": 4, "max_characters": 999999},
    ],
)
def test_invalid_or_ambiguous_requests_do_not_mutate(args):
    work, identity = owned()
    before = copy.deepcopy(work)
    with pytest.raises(ValueError):
        nav.inspect_code(work, {"artifact_id": identity, **args})
    assert work == before


def test_line_reads_are_bounded_and_foreign_or_changed_versions_rejected():
    work, identity = owned()
    view = nav.inspect_code(
        work, {"artifact_id": identity, "line": 8, "context_lines": 1, "max_characters": 128}
    )
    assert view["start_line"] == 7 and view["end_line"] == 9 and len(view["source"]) <= 128
    with pytest.raises(ValueError):
        nav.inspect_code(work, {"artifact_id": "foreign", "line": 1})
    work["artifacts"][identity]["content"]["source"] += "\n# changed"
    with pytest.raises(ValueError, match="identity"):
        nav.inspect_code(work, {"artifact_id": identity, "line": 1})


def test_navigation_is_opt_in_and_dispatches_through_harness(tmp_path, monkeypatch):
    from test_prototype_literal_patch import ready_patch

    _, h, rid, state, base = ready_patch(tmp_path, monkeypatch)
    assert "inspect_code" not in h.allowed_actions(state)
    state["code_navigation_policy"] = nav.POLICY
    assert "inspect_code" in h.allowed_actions(state)
    result = h._apply(
        rid,
        state,
        {
            "action": "inspect_code",
            "reason": "locate code",
            "arguments": {"artifact_id": base, "symbol": "build_model"},
        },
        3,
    )
    assert result["source"].startswith("def build_model")


def test_real_receipt_focus_requires_exact_code_binding():
    work, base = owned()
    receipt = {
        "name": "check",
        "kind": "validate",
        "inputs": [base],
        "content": {
            "passed": False,
            "validated_code_id": base,
            "validation_contract": {"kind": "unbound"},
            "execution": {
                "metrics": {"traceback": 'File "/workspace/model.py", line 8, in forward'}
            },
        },
    }
    identity = digest(receipt)
    work["artifacts"][identity] = receipt
    state = {
        "code_navigation_policy": nav.POLICY,
        "work": work,
        "validation_contract": {"kind": "unbound"},
        "validation_feedback": {"code_id": base, "result": receipt["content"]},
    }
    assert nav.repair_locations(state, base)[0]["receipt_artifact_id"] == identity
    state["validation_feedback"]["code_id"] = "other"
    assert nav.repair_locations(state, base) == []
