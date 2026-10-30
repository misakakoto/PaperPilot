"""Design admission against exact, versioned source-claim review receipts."""

import copy
import json

import pytest

from paperpilot_agentic import prototype_literature_review as review
from paperpilot_agentic.prototype_workspace import task_rows

from test_prototype_analysis import prepared, proposal
from test_prototype_literature import work


def test_research_design_requires_current_claim_review(tmp_path, monkeypatch):
    _, h, rid, state, identity = prepared(
        tmp_path,
        monkeypatch,
        text="Future work will introduce reinforcement learning for feature decoupling.",
    )
    state["literature_review_policy"] = "source-claim-review-v1"
    content = proposal(state, identity)
    content["findings"][0]["interpretation"] = (
        "The paper already implements reinforcement learning."
    )
    work(h, rid, state, "analysis", content)
    analysis_id = state["work"]["tasks"]["analysis"]["output"]
    finding = state["work"]["artifacts"][analysis_id]["content"]["findings"][0]
    with pytest.raises(ValueError, match="review"):
        work(
            h,
            rid,
            state,
            "design",
            {
                "architecture": "Use the claimed implemented mechanism",
                "literature_basis": [
                    {
                        "artifact_id": analysis_id,
                        "quote_id": finding["quote_id"],
                        "interpretation": "The mechanism was implemented in the source",
                    }
                ],
            },
        )


def reviewed(tmp_path, monkeypatch):
    rt, h, rid, state, identity = prepared(
        tmp_path,
        monkeypatch,
        text="Future work will introduce reinforcement learning. This is a proposal, not an experiment.",
    )
    state["literature_review_policy"] = review.POLICY
    rows = task_rows(state["work"])
    rows[-1]["depends_on"].append("claims")
    rows.insert(
        -1,
        {
            "id": "claims",
            "goal": "critique source claims",
            "capability": review.CAPABILITY,
            "depends_on": ["analysis"],
            "acceptance": "exact finding review",
        },
    )
    h._apply(
        rid, state, {"action": "plan", "reason": "source review", "arguments": {"tasks": rows}}, 2
    )
    work(h, rid, state, "analysis", proposal(state, identity))
    return rt, h, rid, state, identity


def judgment(state, *, supported):
    p = review.packet(state["work"], state["work"]["tasks"]["claims"])
    return {
        "checks": [
            {
                "finding_index": f["finding_index"],
                "quote_id": f["quote_id"],
                "source_role": "future_work",
                "interpretation_supported": supported,
                "reason": "The passage describes future work",
                "required_correction": ""
                if supported
                else "Describe this mechanism as a future proposal",
            }
            for f in p["findings"]
        ],
        "coverage": {
            "adequate_for_design": True,
            "reason": "Bounded hypothetical design only",
            "limitations": ["No training or novelty evidence"],
        },
    }


def test_failed_claim_review_blocks_design_and_repair_preserves_unrelated_inputs(
    tmp_path, monkeypatch
):
    _, h, rid, state, _ = reviewed(tmp_path, monkeypatch)
    work(h, rid, state, "claims", judgment(state, supported=False))
    receipt = state["work"]["tasks"]["claims"]["output"]
    assert state["work"]["artifacts"][receipt]["content"]["passed"] is False
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match="passing current literature review"):
        work(h, rid, state, "design", {})
    assert state == before and "repair" in h.allowed_actions(state)
    h._apply(
        rid,
        state,
        {
            "action": "repair",
            "reason": "Correct future-work claim",
            "arguments": {"target_task_id": "analysis"},
        },
        7,
    )
    assert state["work"]["tasks"]["analysis"]["status"] == "PENDING"
    assert state["work"]["tasks"]["claims"]["status"] == "PENDING"
    assert state["work"]["tasks"]["design"]["status"] == "PENDING"
    assert state["work"]["artifacts"] == before["work"]["artifacts"]
    assert state["work"]["tasks"]["read"] == before["work"]["tasks"]["read"]
    assert state["work"]["tasks"]["paper"] == before["work"]["tasks"]["paper"]
    assert state["work"]["repairs"] == before["work"]["repairs"]
    _, messages = h._context(rid, state)
    assert (
        json.loads(messages[-1]["content"])["literature_review_feedback"][receipt]["passed"]
        is False
    )


def test_corrected_analysis_needs_fresh_review_before_design(tmp_path, monkeypatch):
    _, h, rid, state, identity = reviewed(tmp_path, monkeypatch)
    work(h, rid, state, "claims", judgment(state, supported=False))
    old = state["work"]["tasks"]["claims"]["output"]
    h._apply(
        rid,
        state,
        {
            "action": "repair",
            "reason": "fix attribution",
            "arguments": {"target_task_id": "analysis"},
        },
        7,
    )
    content = proposal(state, identity)
    content["findings"][0]["interpretation"] = (
        "This is explicitly future work; we borrow it as an untested hypothesis."
    )
    work(h, rid, state, "analysis", content)
    with pytest.raises(ValueError, match="review"):
        review.require_design(state, state["work"]["tasks"]["design"])
    work(h, rid, state, "claims", judgment(state, supported=True))
    new = state["work"]["tasks"]["claims"]["output"]
    assert old != new
    analysis_id = state["work"]["tasks"]["analysis"]["output"]
    f = state["work"]["artifacts"][analysis_id]["content"]["findings"][0]
    work(
        h,
        rid,
        state,
        "design",
        {
            "literature_basis": [
                {
                    "artifact_id": analysis_id,
                    "quote_id": f["quote_id"],
                    "interpretation": "Hypothesis only",
                }
            ]
        },
    )
    assert new in state["work"]["artifacts"][state["work"]["tasks"]["design"]["output"]]["inputs"]
    assert state["work"]["artifacts"][new]["content"]["scientific_validity"] == "NOT_VERIFIED"


@pytest.mark.parametrize(
    "case", ["missing", "duplicate", "foreign_quote", "self_pass", "no_correction", "wrong_role"]
)
def test_review_cannot_omit_findings_or_self_certify(tmp_path, monkeypatch, case):
    _, h, rid, state, _ = reviewed(tmp_path, monkeypatch)
    content = judgment(state, supported=False)
    if case == "missing":
        content["checks"] = []
    elif case == "duplicate":
        content["checks"] *= 2
    elif case == "foreign_quote":
        content["checks"][0]["quote_id"] = "foreign"
    elif case == "self_pass":
        content["passed"] = True
    elif case == "no_correction":
        content["checks"][0]["required_correction"] = ""
    else:
        content["checks"][0]["source_role"] = []
    before = copy.deepcopy(state)
    with pytest.raises(ValueError):
        work(h, rid, state, "claims", content)
    assert state == before


def test_insufficient_coverage_is_a_blocking_model_judgment(tmp_path, monkeypatch):
    _, h, rid, state, _ = reviewed(tmp_path, monkeypatch)
    content = judgment(state, supported=True)
    content["coverage"].update(
        adequate_for_design=False, reason="Only damage grading has been read"
    )
    work(h, rid, state, "claims", content)
    a = state["work"]["artifacts"][state["work"]["tasks"]["claims"]["output"]]["content"]
    assert a["passed"] is False and a["issues"] == ["Only damage grading has been read"]
    assert a["assurance"] == "MODEL_JUDGMENT_NOT_HOST_VERIFIED"


def test_plan_requires_review_of_the_same_analysis_and_legacy_remains_readable(
    tmp_path, monkeypatch
):
    _, _, _, state, _ = reviewed(tmp_path, monkeypatch)
    rows = task_rows(state["work"])
    next(t for t in rows if t["id"] == "design")["depends_on"].remove("claims")
    with pytest.raises(ValueError, match="direct review coverage"):
        review.check_plan(state, rows)
    legacy = copy.deepcopy(state)
    legacy.pop("literature_review_policy")
    legacy_rows = [t for t in rows if t["id"] != "claims"]
    review.check_plan(legacy, legacy_rows)
    with pytest.raises(ValueError, match="not registered"):
        review.check_plan(legacy, rows)


def test_review_packet_rejects_modified_original_sources(tmp_path, monkeypatch):
    _, _, _, state, identity = reviewed(tmp_path, monkeypatch)
    state["work"]["artifacts"][identity]["content"]["evidence"][0]["content"] += "Changed source"
    with pytest.raises(ValueError, match="identity changed"):
        review.packet(state["work"], state["work"]["tasks"]["claims"])


def test_unknown_review_policy_cannot_be_ignored(tmp_path, monkeypatch):
    _, h, rid, state, _ = reviewed(tmp_path, monkeypatch)
    state["literature_review_policy"] = "unknown-policy"
    with pytest.raises(ValueError, match="Unknown literature review policy"):
        h._context(rid, state)


@pytest.mark.parametrize("matching_page", [True, False])
def test_local_fragment_requires_matching_full_page_context(matching_page):
    from paperpilot_agentic.prototype_workspace import new_workspace, plan, complete
    from paperpilot_agentic.prototype_analysis import resolve, spans

    work_state = new_workspace("research")
    rows = [
        {"id": k, "goal": k, "capability": c, "depends_on": d, "acceptance": "source"}
        for k, c, d in [
            ("search", "search_literature", []),
            ("fragment", "read_literature", ["search"]),
            ("analysis", "analyze_literature", ["fragment"]),
            ("pages", "read_literature", ["search"]),
            ("review", "review_literature", ["analysis", "pages"]),
        ]
    ]
    plan(work_state, rows)
    complete(work_state, "search", "search_literature", {"ok": True, "evidence": []})
    sha = "a" * 64
    uri = "paper://" + sha + "/chunk"
    raw = complete(
        work_state,
        "fragment",
        "read_literature",
        {
            "ok": True,
            "evidence": [
                {
                    "content": "A new mechanism is introduced.",
                    "source_uri": uri,
                    "page": 1,
                    "evidence_type": "TEXT",
                }
            ],
        },
    )
    proposal = {
        "findings": [
            {
                "artifact_id": raw,
                "quote_id": spans(raw, work_state["artifacts"][raw])[0]["quote_id"],
                "interpretation": "Already implemented",
            }
        ],
        "gaps": [],
        "directions": [
            {"hypothesis": "A hypothesis", "mechanism": "A mechanism", "falsification": "A control"}
        ],
    }
    complete(
        work_state,
        "analysis",
        "analyze_literature",
        resolve(work_state, work_state["tasks"]["analysis"], proposal, analysis=True),
    )
    task = work_state["tasks"]["review"]
    p = review.packet(work_state, task)
    content = {
        "checks": [
            {
                "finding_index": 0,
                "quote_id": p["findings"][0]["quote_id"],
                "source_role": "implemented_method",
                "interpretation_supported": True,
                "reason": "Model may be wrong",
                "required_correction": "",
            }
        ],
        "coverage": {"adequate_for_design": True, "reason": "Model judgment", "limitations": []},
    }
    assert review.resolve(work_state, task, content)["passed"] is False
    complete(
        work_state,
        "pages",
        "read_literature",
        {
            "ok": True,
            "evidence": [
                {
                    "content": "Future work is planned. A new mechanism is introduced.",
                    "source_uri": uri + "/page",
                    "page": 1,
                    "provenance": {
                        "complete_page": True,
                        "paper_sha256": sha if matching_page else "b" * 64,
                    },
                }
            ],
        },
    )
    p = review.packet(work_state, task)
    if matching_page:
        assert p["missing_page_context"] == [] and len(p["page_contexts"]) == 1
        assert "Future work" in next(iter(p["page_contexts"].values()))["text"]
    else:
        assert p["missing_page_context"] and not p["page_contexts"]
        assert review.resolve(work_state, task, content)["passed"] is False


def test_review_packet_is_shared_once_and_extra_analysis_fields_are_not_lost(tmp_path, monkeypatch):
    from paperpilot_agentic.prototype_workspace import plan, task_rows
    from paperpilot_agentic.prototype_context import POLICY as TABLE_POLICY

    _, h, rid, state, identity = reviewed(tmp_path, monkeypatch)
    rows = task_rows(state["work"])
    next(t for t in rows if t["id"] == "analysis")["goal"] += " with explicit extra claim"
    plan(state["work"], rows)
    content = proposal(state, identity)
    content["extra_research_claim"] = "Unverified extra assumption requiring review"
    work(h, rid, state, "analysis", content)
    state.update(
        literature_review_context_policy="single-review-packet-v1", context_policy=TABLE_POLICY
    )
    before = copy.deepcopy(state)
    _, messages = h._context(rid, state)
    payload = json.loads(messages[-1]["content"])
    packet = payload["literature_review_packet"]
    aid = packet["analysis_artifact_id"]
    assert payload["context_artifacts"][aid]["content"]["view_ref"] == "literature_review_packet"
    assert (
        packet["additional_analysis_fields"]["extra_research_claim"]
        == content["extra_research_claim"]
    )
    assert packet["findings"][0]["interpretation"] == content["findings"][0]["interpretation"]
    assert state == before


def test_missing_evidence_requires_new_reading_instead_of_text_only_repair(tmp_path, monkeypatch):
    _, h, rid, state, _ = reviewed(tmp_path, monkeypatch)
    content = judgment(state, supported=False)
    content["coverage"].update(
        adequate_for_design=False, reason="Methods and other papers have not been read"
    )
    work(h, rid, state, "claims", content)
    assert not review.repair_options(state)
    _, messages = h._context(rid, state)
    gaps = json.loads(messages[-1]["content"])["literature_evidence_gaps"]
    assert len(gaps) == 1 and "Replan" in gaps[0]["required_action"]
    before = copy.deepcopy(state)
    with pytest.raises(ValueError):
        h._apply(
            rid,
            state,
            {
                "action": "repair",
                "reason": "rewrite without new evidence",
                "arguments": {"target_task_id": "analysis"},
            },
            8,
        )
    assert state == before


def test_evidence_gap_context_is_a_readable_replanning_contract(tmp_path,monkeypatch):
    _,h,rid,state,_=reviewed(tmp_path,monkeypatch)
    content=judgment(state,supported=True)
    content['coverage'].update(adequate_for_design=False,reason='Need the methods section and another source')
    work(h,rid,state,'claims',content)
    state['evidence_replanning_policy']='evidence-gap-planning-v1'
    before=copy.deepcopy(state)
    _,messages=h._context(rid,state)
    payload=json.loads(messages[-1]['content'])
    assert payload['evidence_replanning'] and payload['active_task_id'] is None
    assert payload['ready_task_ids'] == []
    assert payload['blocked_for_evidence_task_ids']
    assert payload['plan_to_revise']==task_rows(state['work'])
    assert 'plan' in payload['available_actions'] and 'work' not in payload['available_actions']
    assert 'raw_artifact_id' in payload['completion_context']['analysis']['content']
    assert state==before


@pytest.mark.parametrize('mutation', ['no_reading', 'removed_review', 'weaker_acceptance'])
def test_evidence_replanning_cannot_bypass_failed_review(tmp_path, monkeypatch, mutation):
    _, h, rid, state, _ = reviewed(tmp_path, monkeypatch)
    content = judgment(state, supported=True)
    content['coverage'].update(adequate_for_design=False, reason='Missing methods')
    work(h, rid, state, 'claims', content)
    state['evidence_replanning_policy'] = 'evidence-gap-planning-v1'
    rows = task_rows(state['work'])
    if mutation == 'removed_review':
        rows = [t for t in rows if t['id'] != 'claims']
        for row in rows:
            row['depends_on'] = [d for d in row['depends_on'] if d != 'claims']
    elif mutation == 'weaker_acceptance':
        next(t for t in rows if t['id'] == 'analysis')['acceptance'] = 'Any text is enough'
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match='Evidence replanning'):
        h._apply(rid, state, {'action': 'plan', 'reason': 'skip reading', 'arguments': {'tasks': rows}}, 8)
    assert state == before


def test_evidence_extension_preserves_observations_and_invalidates_consumers(tmp_path, monkeypatch):
    _, h, rid, state, _ = reviewed(tmp_path, monkeypatch)
    content = judgment(state, supported=True)
    content['coverage'].update(adequate_for_design=False, reason='Missing methods')
    work(h, rid, state, 'claims', content)
    state['evidence_replanning_policy'] = 'evidence-gap-planning-v1'
    before = copy.deepcopy(state['work'])
    search = next(k for k, t in before['tasks'].items() if t['capability'] == 'search_literature')
    addition = {'id': 'read_methods', 'capability': 'read_literature', 'depends_on': [search],
                'goal': 'Read method and ablation pages of discovered papers', 'acceptance': 'Bounded source pages retained'}
    h._apply(rid, state, {'action': 'gather_evidence', 'reason': 'Need methods',
                         'arguments': {'analysis_task_id': 'analysis', 'tasks': [addition]}}, 8)
    after = state['work']
    assert after['artifacts'] == before['artifacts']
    assert after['tasks']['analysis']['depends_on'] == before['tasks']['analysis']['depends_on'] + ['read_methods']
    assert all(after['tasks'][k]['status'] == 'PENDING' for k in ('analysis', 'claims', 'design'))
    for key, task in before['tasks'].items():
        if task['capability'] in {'read', 'search_literature', 'read_literature'}:
            assert after['tasks'][key] == task
    assert not review.needs_replanning(state)
    _, messages = h._context(rid, state)
    memory = json.loads(messages[-1]['content'])['literature_reading_memory']
    assert memory['completed_reads'][0]['observed_pages'] == [3]
    assert memory['assigned_evidence_gaps'][0]['coverage']['reason'] == 'Missing methods'
    assert memory['assigned_evidence_gaps'][0]['status'].startswith('HISTORICAL')


def test_null_plan_rows_are_validation_errors_not_crashes():
    from paperpilot_agentic.prototype_workspace import new_workspace, plan
    with pytest.raises(ValueError, match='Invalid task fields'):
        plan(new_workspace('Research'), [None])


def test_rejected_evidence_extension_does_not_mutate_live_dependencies(tmp_path, monkeypatch):
    _, h, rid, state, _ = reviewed(tmp_path, monkeypatch)
    content = judgment(state, supported=True)
    content['coverage'].update(adequate_for_design=False, reason='Missing methods')
    work(h, rid, state, 'claims', content)
    state['evidence_replanning_policy'] = 'evidence-gap-planning-v1'
    before = copy.deepcopy(state)
    addition = {'id': 'read_methods', 'capability': 'read_literature', 'depends_on': [],
                'goal': 'Read methods', 'acceptance': 'Bounded source pages retained'}
    with pytest.raises(ValueError, match='dependency missing'):
        h._apply(rid, state, {'action': 'gather_evidence', 'reason': 'Need methods',
                             'arguments': {'analysis_task_id': 'analysis', 'tasks': [addition]}}, 8)
    assert state == before
    assert review.needs_replanning(state)


def test_extension_only_policy_exposes_no_full_plan_replacement(tmp_path, monkeypatch):
    _, h, rid, state, _ = reviewed(tmp_path, monkeypatch)
    content = judgment(state, supported=True)
    content['coverage'].update(adequate_for_design=False, reason='Missing methods')
    work(h, rid, state, 'claims', content)
    state['evidence_replanning_policy'] = 'evidence-gap-extension-v2'
    _, messages = h._context(rid, state)
    payload = json.loads(messages[-1]['content'])
    assert 'gather_evidence' in payload['available_actions']
    assert 'plan' not in payload['available_actions']
    assert 'Valid request example:' in messages[0]['content']
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match='Action unavailable'):
        h._apply(rid, state, {'action': 'plan', 'reason': 'replace all tasks',
                             'arguments': {'tasks': task_rows(state['work'])}}, 8)
    assert state == before
