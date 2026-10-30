import ast
import copy
import hashlib
import json

import pytest

from paperpilot_agentic import prototype_prompt as prompt
from paperpilot_agentic.prototype_workspace import digest, plan
from test_prototype_completion import finished

SOURCE = """# preserved author header
import math

def helper(x):
    return math.sqrt(x)

class Base:
    def forward(self, x):
        return helper(x)

class SNUNet_ECAM(Base):
    pass

class Unrelated:
    pass
"""


def source_artifact():
    return {
        "name": "read",
        "kind": "read",
        "inputs": [],
        "content": {
            "evidence_id": "code",
            "evidence_type": "CODE",
            "content": SOURCE,
            "source_uri": "public-pinned-source",
            "unused_bookkeeping": None,
            "provenance": {"license_text": "Exact license text", "license_url": "pinned-license"},
        },
    }


def test_pinned_view_keeps_transitive_dependencies_exact_and_original_immutable(monkeypatch):
    monkeypatch.setattr(prompt, "SOURCE_SHA", hashlib.sha256(SOURCE.encode()).hexdigest())
    artifact = source_artifact()
    identity = digest(artifact)
    before = copy.deepcopy(artifact)
    view = prompt.project_source(identity, artifact)
    assert artifact == before and digest(artifact) == identity
    source = view["content"]["content"]
    tree = ast.parse(source)
    assert [n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))] == [
        "helper",
        "Base",
        "SNUNet_ECAM",
    ]
    assert source.startswith("# preserved author header")
    assert view["content"]["provenance"]["license_text"] == "Exact license text"
    metadata = view["content"]["context_view"]
    assert metadata["complete_file"] is False and metadata["raw_artifact_id"] == identity
    assert metadata["omitted_top_level_symbols"] == ["Unrelated"]
    for first, last in metadata["included_line_ranges"]:
        assert "".join(SOURCE.splitlines(keepends=True)[first - 1 : last]) in source
    assert (
        "Unrelated" in artifact["content"]["content"]
        and "unused_bookkeeping" not in view["content"]
    )
    compile(source, "excerpt.py", "exec")  # Parse only; never import generated source.


def test_unknown_or_modified_sources_are_never_silently_pruned(monkeypatch):
    monkeypatch.setattr(prompt, "SOURCE_SHA", hashlib.sha256(SOURCE.encode()).hexdigest())
    artifact = source_artifact()
    artifact["content"]["content"] += "\n# changed"
    assert prompt.project_source(digest(artifact), artifact) is artifact
    artifact = source_artifact()
    artifact["content"]["evidence_type"] = "TEXT"
    assert prompt.project_source(digest(artifact), artifact) is artifact


def context_state():
    harness, state, rows = finished()
    rows[2]["goal"] = "implement new model"
    plan(state["work"], rows)
    state.update(
        instruction_policy=prompt.POLICY,
        task_context_policy="serial-task-v1",
        context_policy="artifact-table-v1",
    )
    return harness, state


def test_active_implementation_receives_code_contract_without_unrelated_tool_schemas():
    harness, state = context_state()
    before = copy.deepcopy(state)
    _, messages = harness._context("unused", state)
    system = messages[0]["content"]
    assert "complete model.py" in system and "Bx1xHxW" in system and "No training" in system
    assert "FULL REPLACEMENT" in system and "ALL implementation_source_ids" in system
    assert "published_from" not in system and "pages:[1..3" not in system
    assert json.loads(messages[-1]["content"])["instruction_policy"] == prompt.POLICY
    assert state == before


def test_bound_implementation_keeps_fusion_only_contract():
    state = {"work": {"tasks": {"code": {"capability": "implement"}}}, "profile_id": "pinned"}
    text = prompt.instructions(state, {"active_task_id": "code", "available_actions": ["work"]})
    assert "fusion_body" in text and "complete model.py" not in text


def test_active_search_has_explicit_work_target_and_scalar_query(tmp_path, monkeypatch):
    from test_prototype_literature import setup
    _, h, rid, state, _ = setup(tmp_path, monkeypatch)
    before = copy.deepcopy(state)
    _, messages = h._context(rid, state)
    assert 'arguments.task_id' in messages[0]['content']
    assert 'never a list' in messages[0]['content']
    assert 'active_task_id' in messages[0]['content']
    assert json.loads(messages[-1]['content'])['active_task_id'] == 'search'
    assert state == before


def test_initial_research_plan_example_is_admitted_without_inventing_missing_edges(tmp_path, monkeypatch):
    from test_prototype_literature import setup
    from paperpilot_agentic.prototype_workspace import new_workspace
    _, h, rid, state, _ = setup(tmp_path, monkeypatch)
    state.update(initial_plan_policy='source-bound-initial-plan-v1',
                 literature_analysis_policy='source-span-analysis-v1',
                 literature_review_policy='source-claim-review-v1')
    from paperpilot_agentic.prototype_analysis import POLICY
    state['literature_analysis_policy'] = POLICY
    state['work'] = new_workspace('Research a falsifiable building change detection prototype')
    before = copy.deepcopy(state)
    _, messages = h._context(rid, state)
    text = messages[0]['content']
    assert 'Valid initial plan:\n' in text
    example = json.loads(text.split('Valid initial plan:\n')[1].split('\nEnd example.')[0])
    assert state == before
    h._apply(rid, state, example, 1)
    tasks = state['work']['tasks']
    assert len([t for t in tasks.values() if t['capability'] == 'read']) == len(state['sources'])
    assert {'design', 'implement', 'validate', 'review'} <= {t['capability'] for t in tasks.values()}
    assert not state['plan_elaborations'][-1]['added_edges']


def test_unknown_prompt_policy_is_rejected_before_context_use():
    harness, state = context_state()
    state["instruction_policy"] = "unknown"
    with pytest.raises(ValueError, match="instruction policy"):
        harness._context("unused", state)


def test_research_outline_keeps_original_source_and_exact_constructor_declarations():
    import ast
    import copy
    import json
    from paperpilot_agentic.prototype_prompt import project_research_source, RESEARCH_SOURCE_POLICY
    from paperpilot_agentic.reference_methods import SOURCE_SHA
    from pathlib import Path
    import hashlib
    baseline = Path('outputs/prototype-holdout-qwen37-v1-20261024/baseline.py')
    if not baseline.exists():
        import pytest
        pytest.skip('Pinned baseline snapshot not available')
    source = baseline.read_text(encoding='utf-8')
    assert hashlib.sha256(source.encode()).hexdigest() == SOURCE_SHA
    artifact = {'kind': 'read', 'content': {'evidence_type': 'CODE', 'content': source}}
    before = copy.deepcopy(artifact)
    view = project_research_source('original', artifact)
    assert view['content']['context_view']['policy'] == RESEARCH_SOURCE_POLICY
    assert view['content']['context_view']['complete_source'] is False
    assert 'content' not in view['content']
    names = {x['class'] for x in view['content']['source_outline']}
    assert 'SNUNet_ECAM' in names and 'Siam_NestedUNet_Conc' not in names
    assignments = {ast.unparse(n) for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Assign)}
    assert all(d in assignments for x in view['content']['source_outline'] for d in x['constructor_assignments'])
    assert len(json.dumps(view)) < len(json.dumps(artifact))
    assert artifact == before
