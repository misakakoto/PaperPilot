import copy
import json

import pytest

from paperpilot_agentic.prototype_candidate import PATCH_POLICY, normalize, patch_base
from paperpilot_agentic.prototype_prompt import POLICY, TASK_POLICY
from paperpilot_agentic.prototype_workspace import plan, task_rows, complete
from paperpilot_agentic.prototype_harness import validation_context
from test_prototype_candidate_policy import prepared, work, SOURCE
from test_prototype_adaptive_repair import failed


def ready_patch(tmp_path, monkeypatch):
    rt, h, rid, state = prepared(tmp_path, monkeypatch)
    work(h, rid, state, "code", {"source": SOURCE, "config": {}})
    base = state["work"]["tasks"]["code"]["output"]
    rows = task_rows(state["work"])
    next(t for t in rows if t["id"] == "code")["goal"] = "repair same candidate"
    plan(state["work"], rows)
    state["candidate_patch_policy"] = PATCH_POLICY
    return rt, h, rid, state, base


def test_patch_is_bound_to_exact_current_code_and_creates_new_artifact(tmp_path, monkeypatch):
    _, h, rid, state, base = ready_patch(tmp_path, monkeypatch)
    before = copy.deepcopy(state["work"]["artifacts"][base])
    work(
        h,
        rid,
        state,
        "code",
        {
            "base_artifact_id": base,
            "replacements": [{"old": "return None", "new": "return 'fixed'"}],
            "config": {"enabled": True},
        },
    )
    current = state["work"]["tasks"]["code"]["output"]
    assert current != base and state["work"]["artifacts"][base] == before
    assert state["work"]["artifacts"][current]["content"] == {
        "source": SOURCE.replace("return None", "return 'fixed'"),
        "config": {"enabled": True},
    }
    assert state["work"]["tasks"]["check"]["status"] == "PENDING"
    assert state["implementation_normalizations"][-1]["base_artifact_id"] == base


@pytest.mark.parametrize("old", ["", "missing code", " "])
def test_ambiguous_or_absent_literal_patches_are_rejected_without_mutating_base(
    tmp_path, monkeypatch, old
):
    _, _, _, state, base = ready_patch(tmp_path, monkeypatch)
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match="exactly once"):
        normalize(
            state,
            {"base_artifact_id": base, "replacements": [{"old": old, "new": "x"}], "config": {}},
            task_id="code",
        )
    assert state == before


def test_foreign_base_or_changed_design_cannot_reuse_old_source_patch(tmp_path, monkeypatch):
    _, _, _, state, base = ready_patch(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="current unchanged base"):
        normalize(
            state,
            {
                "base_artifact_id": "foreign",
                "replacements": [{"old": "None", "new": "1"}],
                "config": {},
            },
            task_id="code",
        )
    rows = task_rows(state["work"])
    next(t for t in rows if t["id"] == "design")["goal"] = "different design"
    plan(state["work"], rows)
    complete(state["work"], "design", "design", {"architecture": "changed"})
    assert patch_base(state, "code") is None


def test_task_summaries_are_read_only_and_exact_plan_can_be_inspected():
    h, state = failed()
    state.update(
        instruction_policy=POLICY,
        task_definition_policy=TASK_POLICY,
        task_context_policy="serial-task-v1",
        context_policy="artifact-table-v1",
    )
    before = copy.deepcopy(state)
    _, messages = h._context("offline", state)
    payload = json.loads(messages[-1]["content"])
    assert "goal" not in payload["tasks"]["read"] and "goal" in state["work"]["tasks"]["read"]
    assert "inspect_plan" in payload["available_actions"]
    result = h._apply(
        "offline",
        state,
        {"action": "inspect_plan", "reason": "exact definitions", "arguments": {}},
        8,
    )
    assert result["tasks"] == task_rows(state["work"]) and state == before
    assert h._decision_reserve_class(state) == "repair"


def test_traceback_view_keeps_exact_candidate_frames_without_framework_noise():
    trace = 'Traceback (most recent call last):\n  File "/lib/torch.py", line 100, in forward\n    helper(x)\n  File "/workspace/model.py", line 25, in forward\n    out = x * gate\nRuntimeError: channels differ\n'
    raw = {
        "passed": False,
        "execution": {"metrics": {"error": "channels differ", "traceback": trace}},
    }
    before = copy.deepcopy(raw)
    view = validation_context(raw, compact_traceback=True)
    metrics = view["execution"]["metrics"]
    assert "traceback" not in metrics and metrics["error"] == "channels differ"
    assert metrics["traceback_view"]["candidate_frames"] == [
        {
            "file": "/workspace/model.py",
            "line": 25,
            "function": "forward",
            "source": "out = x * gate",
        }
    ]
    assert raw == before and validation_context(raw)["execution"]["metrics"]["traceback"] == trace


def test_review_source_view_has_identical_ast_and_keeps_original_source(monkeypatch):
    from paperpilot_agentic.prototype_prompt import project_review_code
    import ast

    artifact = {
        "name": "code",
        "kind": "implement",
        "inputs": [],
        "content": {
            "source": "# retained original comment\n\ndef build_model(config):\n    return None\n",
            "config": {},
        },
    }
    from paperpilot_agentic.prototype_workspace import digest

    identity = digest(artifact)
    original = copy.deepcopy(artifact)
    view = project_review_code(identity, artifact)
    assert ast.dump(ast.parse(view["content"]["source"])) == ast.dump(
        ast.parse(artifact["content"]["source"])
    )
    assert view["content"]["context_view"]["same_ast"] is True
    assert artifact == original and digest(artifact) == identity


def test_exhausted_implementation_repairs_release_no_new_attempts_for_final_reporting():
    h, state = failed()
    state.update(instruction_policy=POLICY)
    state["work"]["repairs"] = 3
    assert h._decision_reserve_class(state) == "repair"
    assert "repair" not in h.allowed_actions(state)


def test_finished_failed_work_uses_receipts_for_repair_routing_without_replaying_design():
    from test_prototype_completion import finished
    from paperpilot_agentic.prototype_repair import POLICY as REPAIR_POLICY
    h, state, rows = finished()
    state.update(instruction_policy=POLICY, repair_policy=REPAIR_POLICY)
    state['work']['repairs'] = 1
    rows[-1]['goal'] = 'review actual mismatch'
    plan(state['work'], rows)
    complete(state['work'], 'review', 'review', {
        'implementation_consistent': False, 'issues': ['wrong gate'], 'limitations': []})
    before = copy.deepcopy(state)
    _, messages = h._context('offline', state)
    payload = json.loads(messages[-1]['content'])
    assert payload['completion_context']['design']['content'] == {
        'full_artifact_id': state['work']['tasks']['design']['output'],
        'view': 'repair-routing-reference'}
    assert payload['completion_context']['review']['content']['issues'] == ['wrong gate']
    assert h._decision_output_tokens(state) == 1000
    assert state == before


def test_patch_source_excerpts_are_exact_and_never_mutate_full_candidate():
    from paperpilot_agentic.prototype_prompt import project_patch_code
    source = "import torch\nclass Model:\n    def __init__(self):\n        self.large = '" + 'x'*1000 + "'\n    def forward(self, a, b):\n        return a + b\n"
    artifact = {'kind': 'implement', 'content': {'source': source, 'config': {}}}
    before = copy.deepcopy(artifact)
    view = project_patch_code('original', artifact)
    assert view['content']['context_view']['representation'] == 'EXACT_FORWARD_EXCERPTS_NOT_COMPLETE_SOURCE'
    assert 'source' not in view['content']
    excerpts = view['content']['source_excerpts']
    assert {'symbol': 'Model.forward', 'source': '    def forward(self, a, b):\n        return a + b\n'} in excerpts
    assert all(item['source'] in source for item in excerpts)
    assert all('# '+item['symbol'] not in item['source'] for item in excerpts)
    assert view['content']['context_view']['omitted_methods_line_ranges'] == [['Model.__init__', 3, 4]]
    assert artifact == before


def test_patch_context_keeps_review_failures_bound_to_the_exact_base(tmp_path, monkeypatch):
    from paperpilot_agentic.prototype_prompt import POLICY as FOCUSED_POLICY
    from paperpilot_agentic.prototype_repair import POLICY as REPAIR_POLICY
    from paperpilot_agentic.prototype_context import POLICY as TABLE_POLICY, SERIAL_POLICY
    _, h, rid, state = prepared(tmp_path, monkeypatch)
    rows = task_rows(state['work'])
    rows.append({'id': 'review', 'goal': 'review', 'capability': 'review',
                 'depends_on': ['design', 'code', 'check'], 'acceptance': 'fixed'})
    plan(state['work'], rows)
    work(h, rid, state, 'code', {'source': SOURCE, 'config': {}})
    code = state['work']['tasks']['code']['output']
    complete(state['work'], 'check', 'validate', {'passed': False, 'validated_code_id': code,
             'failure_kind': 'IMPLEMENTATION', 'error': 'channels differ'})
    receipt = complete(state['work'], 'review', 'review', {
        'implementation_consistent': False, 'issues': ['declared channel gate differs'], 'limitations': []})
    state.update(repair_policy=REPAIR_POLICY, instruction_policy=FOCUSED_POLICY,
                 candidate_patch_policy=PATCH_POLICY, context_policy=TABLE_POLICY,
                 task_context_policy=SERIAL_POLICY)
    h._apply(rid, state, {'action': 'repair', 'arguments': {'target_task_id': 'code'}, 'reason': 'actual failure'}, 8)
    before = copy.deepcopy(state)
    _, messages = h._context(rid, state)
    payload = json.loads(messages[-1]['content'])
    assert payload['repair_review_feedback'][receipt] == {'code_artifact_id': code,
        'implementation_consistent': False, 'issues': ['declared channel gate differs']}
    assert state == before
