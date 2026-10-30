from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
from paperpilot_agentic.audit import source_attribution_audit
from paperpilot_agentic.builtin_tools import (
    _explicit_layout_label,
    _filter_named_scholarly_items,
    _matches_explicit_layout_label,
    _prioritize_visual_candidates,
)
from paperpilot_agentic.claim_graph import ClaimGraph, EvidenceGapScheduler
from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.evidence import EvidenceLedger
from paperpilot_agentic.intelligence import (
    EvidenceDrivenController,
    LLMAnswerSynthesizer,
    LLMClaimPlanner,
    _select_synthesis_evidence,
)
from paperpilot_agentic.llm_runtime import (
    LLMBudgetExhausted,
    LLMStructuredOutputError,
    PersistentLLMRuntime,
)
from paperpilot_agentic.models import (
    ActionCandidate,
    Budget,
    EvidencePolarity,
    LLM_POLICIES,
    RunStatus,
)
from paperpilot_agentic.orchestrator import (
    ResearchAgentRuntime,
    _visual_table_matches_claim,
)
from paperpilot_agentic.pricing import (
    DEEPSEEK_PRICING_SOURCE,
    TokenPriceCNY,
    estimate_deepseek_cost_cny,
)
from paperpilot_agentic.verification import EvidenceAssessment, LLMEvidenceVerifier


class _JsonModel:
    model = "deepseek-v4-flash"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    def complete(self, _messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        callback = kwargs.get("before_attempt")
        if callable(callback):
            callback()
        self.calls += 1
        return LLMResponse(
            text=json.dumps(self.payload, ensure_ascii=False),
            prompt_tokens=100,
            cached_tokens=20,
            completion_tokens=50,
            reasoning_tokens=30,
            finish_reason="stop",
            model=self.model,
        )


def test_synthesizer_uses_json_focused_policy() -> None:
    assert LLM_POLICIES["synthesizer"].thinking_enabled is False


def test_json_repair_receives_bounded_original_contract(tmp_path: Path) -> None:
    class RepairAwareModel:
        model = "qwen-fixture"

        def __init__(self) -> None:
            self.calls: list[list[dict[str, str]]] = []

        def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
            callback = kwargs.get("before_attempt")
            if callable(callback):
                callback()
            self.calls.append(messages)
            if len(self.calls) == 1:
                text = json.dumps({"section": "wrong-claim-id"})
            else:
                repair_payload = json.loads(messages[-1]["content"])
                assert "claim_demo" in json.dumps(
                    repair_payload["original_task"], ensure_ascii=False
                )
                text = json.dumps({"claim_id": "claim_demo"})
            return LLMResponse(
                text=text,
                prompt_tokens=20,
                completion_tokens=10,
                finish_reason="stop",
                model=self.model,
            )

    db = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    run_id = db.create_run("repair contract", Budget(max_llm_calls=4))
    model = RepairAwareModel()
    runtime = PersistentLLMRuntime(db, model)
    value = runtime.complete_json(
        run_id,
        [
            {"role": "system", "content": "Return the required claim_id."},
            {"role": "user", "content": '{"required_claim_id":"claim_demo"}'},
        ],
        policy="planner",
        validator=lambda item: (
            item
            if item.get("claim_id") == "claim_demo"
            else (_ for _ in ()).throw(ValueError("unknown claim_id"))
        ),
    )

    assert value == {"claim_id": "claim_demo"}
    assert [item["stage"] for item in db.list_llm_calls(run_id)] == [
        "planner",
        "json_repair",
    ]
    invalid_events = [
        item
        for item in db.list_events(run_id)
        if item["event_type"] == "llm.structured_output.invalid"
    ]
    assert len(invalid_events) == 1
    assert invalid_events[0]["payload"]["stage"] == "planner"
    assert invalid_events[0]["payload"]["phase"] == "initial"
    assert "wrong-claim-id" in invalid_events[0]["payload"]["invalid_output"]
    assert invalid_events[0]["payload"]["private_runtime_diagnostic"] is True


def test_json_repair_persists_both_invalid_outputs_privately(tmp_path: Path) -> None:
    class TwiceInvalidModel:
        model = "qwen-fixture"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, _messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
            callback = kwargs.get("before_attempt")
            if callable(callback):
                callback()
            self.calls += 1
            return LLMResponse(
                text=json.dumps({"attempt": self.calls, "claim_id": "wrong"}),
                prompt_tokens=20,
                completion_tokens=10,
                finish_reason="stop",
                model=self.model,
            )

    db = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    run_id = db.create_run("repair diagnostics", Budget(max_llm_calls=4))
    runtime = PersistentLLMRuntime(db, TwiceInvalidModel())

    with pytest.raises(LLMStructuredOutputError, match="JSON repair failed"):
        runtime.complete_json(
            run_id,
            [{"role": "user", "content": "return claim_demo"}],
            policy="planner",
            validator=lambda item: (
                item
                if item.get("claim_id") == "claim_demo"
                else (_ for _ in ()).throw(ValueError("unknown claim_id"))
            ),
        )

    invalid_events = [
        item
        for item in db.list_events(run_id)
        if item["event_type"] == "llm.structured_output.invalid"
    ]
    assert [item["payload"]["phase"] for item in invalid_events] == [
        "initial",
        "repair",
    ]
    assert '"attempt": 1' in invalid_events[0]["payload"]["invalid_output"]
    assert '"attempt": 2' in invalid_events[1]["payload"]["invalid_output"]


def test_follow_up_reuses_conversation_but_not_parent_evidence(tmp_path: Path) -> None:
    runtime = ResearchAgentRuntime(
        db_path=tmp_path / "runtime.sqlite3",
        workspace=tmp_path,
    )
    parent_id = runtime.db.create_run(
        "Compare Paper A and Paper B.",
        Budget(),
        {"conversation_id": "conversation_demo"},
    )
    runtime.db.update_run_metadata(
        parent_id,
        {
            "final_answer": {
                "answer": "Paper A models tokens; Paper B aligns features.",
                "citation_ids": [],
                "research_outcome": "ANSWERED",
            }
        },
    )
    runtime.db.update_run_status(parent_id, RunStatus.COMPLETED)

    child = runtime.enqueue_follow_up(
        parent_run_id=parent_id,
        question="Which idea fits a misregistered dataset?",
        claims=[
            {
                "claim_id": "claim_follow_up",
                "statement": "Paper B directly targets spatial misregistration.",
                "required_evidence_types": ["TEXT"],
                "gaps": ["mechanism"],
            }
        ],
        metadata={"source": "test"},
    )

    metadata = child["run"]["metadata"]
    assert metadata["conversation_id"] == "conversation_demo"
    assert metadata["parent_run_id"] == parent_id
    assert metadata["conversation_context_is_evidence"] is False
    assert metadata["conversation_context"]["prior_question"] == (
        "Compare Paper A and Paper B."
    )
    assert "aligns features" in metadata["conversation_context"]["prior_answer"]
    assert metadata["initial_claims"][0]["claim_id"] == "claim_follow_up"
    assert child["evidence"] == []


def test_verified_visual_table_cannot_close_a_different_table_number() -> None:
    assert _visual_table_matches_claim(
        "Table VII of FlowMamba reports the MFA ablation.",
        "TABLE VII ABLATION STUDY OF MFA",
    )
    assert not _visual_table_matches_claim(
        "Table VII of FlowMamba reports the MFA ablation.",
        "TABLE XII ABLATION STUDY OF SPATIOTEMPORAL MODELING",
    )


def test_synthesis_context_is_claim_fair_and_preserves_polarity_diversity() -> None:
    claims = [
        {"claim_id": f"claim_{index}", "statement": f"Claim {index}"}
        for index in range(6)
    ]
    evidence: list[dict[str, Any]] = []
    # Deliberately append all items for early claims first. A flat [:30]
    # selection would exclude the final three claims completely.
    for claim_index in range(6):
        for item_index in range(10):
            evidence.append(
                {
                    "evidence_id": f"evidence_{claim_index}_{item_index}",
                    "claim_id": f"claim_{claim_index}",
                    "polarity": "CONTRADICTS" if item_index == 9 else "SUPPORTS",
                    "evidence_type": "TABLE" if item_index == 8 else "TEXT",
                    "confidence": 0.99 if item_index == 8 else 0.7,
                    "content": "x" * 1_000,
                    "provenance": {
                        "verification": {
                            "verifier": "runtime-verified-visual-table-v1"
                            if item_index == 8
                            else "llm-entailment-v1",
                            "degraded": False,
                        }
                    },
                }
            )

    selected, stats = _select_synthesis_evidence(claims, evidence)
    represented = {str(item["claim_id"]) for item in selected}

    assert represented == {f"claim_{index}" for index in range(6)}
    assert stats["policy"] == "claim-fair-polarity-aware-v1"
    assert stats["selected_evidence_count"] == 30
    assert stats["omitted_evidence_count"] == 30
    assert stats["represented_claim_count"] == 6
    for claim_index in range(6):
        claim_items = [
            item for item in selected if item["claim_id"] == f"claim_{claim_index}"
        ]
        assert any(item["evidence_type"] == "TABLE" for item in claim_items)
        assert any(item["polarity"] == "CONTRADICTS" for item in claim_items)


def test_synthesis_uses_per_claim_polarity_for_shared_evidence() -> None:
    claims = [
        {"claim_id": "claim_mechanism", "statement": "mechanism"},
        {"claim_id": "claim_ablation", "statement": "ablation"},
    ]
    evidence = [
        {
            "evidence_id": "evidence_shared",
            "claim_ids": ["claim_mechanism", "claim_ablation"],
            # The flat evidence label reflects the first verification only.
            "polarity": "NEUTRAL",
            "claim_polarities": {
                "claim_mechanism": "NEUTRAL",
                "claim_ablation": "SUPPORTS",
            },
            "evidence_type": "TEXT",
            "confidence": 0.9,
            "content": "The baseline is 80.14% and MFA reaches 80.26%.",
            "provenance": {"verification": {"verifier": "llm-entailment-v1"}},
        }
    ]

    selected, stats = _select_synthesis_evidence(claims, evidence)

    assert stats["selected_evidence_count"] == 1
    assert selected[0]["claim_ids"] == ["claim_ablation"]
    assert selected[0]["polarity"] == "SUPPORTS"
    assert selected[0]["claim_polarities"] == {"claim_ablation": "SUPPORTS"}


def test_official_deepseek_flash_cost_matches_paid_smoke_usage() -> None:
    beijing = ZoneInfo("Asia/Shanghai")
    estimate = estimate_deepseek_cost_cny(
        model="deepseek-v4-flash",
        prompt_tokens=38_679,
        cached_tokens=0,
        completion_tokens=8_939,
        at=datetime(2026, 8, 26, 21, 26, tzinfo=beijing),
    )
    assert estimate is not None
    assert estimate.pricing_tier == "off_peak"
    assert estimate.pricing_source == DEEPSEEK_PRICING_SOURCE
    assert estimate.amount_cny == pytest.approx(0.098244)
    beijing = ZoneInfo("Asia/Shanghai")
    peak = estimate_deepseek_cost_cny(
        model="deepseek-v4-pro",
        prompt_tokens=1_000,
        cached_tokens=250,
        completion_tokens=100,
        at=datetime(2026, 8, 26, 10, 0, tzinfo=beijing),
    )
    assert peak is not None
    assert peak.pricing_tier == "peak"
    assert peak.amount_cny == pytest.approx((250 * 0.30 + 750 * 9.0 + 100 * 27.0) / 1_000_000)

    override = estimate_deepseek_cost_cny(
        model="private-deepseek-compatible-model",
        prompt_tokens=1_000,
        cached_tokens=0,
        completion_tokens=100,
        at=datetime(2026, 8, 26, 21, 0, tzinfo=beijing),
        override_rates=TokenPriceCNY(1.0, 2.0, 3.0),
    )
    assert override is not None
    assert override.pricing_source.startswith("caller-supplied")
    assert override.pricing_version == "runtime-rate-override"


def test_runtime_persists_cny_cost_without_double_charging_reasoning(tmp_path: Path) -> None:
    db = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    run_id = db.create_run("cost accounting", Budget(max_llm_calls=3))
    model = _JsonModel({"ok": True})
    runtime = PersistentLLMRuntime(db, model)
    runtime.complete(run_id, [{"role": "user", "content": "test"}], policy="planner")

    call = db.list_llm_calls(run_id)[0]
    run = db.get_run(run_id)
    assert call["reasoning_tokens"] == 30
    assert call["completion_tokens"] == 50
    assert call["total_tokens"] == 150
    assert call["estimated_cost_cny"] > 0
    assert run["estimated_cost_cny"] == pytest.approx(call["estimated_cost_cny"])


def test_prompt_upper_bound_prevents_a_real_token_budget_overshoot(tmp_path: Path) -> None:
    db = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    run_id = db.create_run("hard token preflight", Budget(max_tokens=27_000))
    model = _JsonModel({"ok": True})
    runtime = PersistentLLMRuntime(db, model)
    with pytest.raises(LLMBudgetExhausted, match="input/output upper bound"):
        runtime.complete(
            run_id,
            [{"role": "user", "content": "x" * 2_000}],
            policy="planner",
        )
    assert model.calls == 0
    assert db.get_run(run_id)["used_llm_calls"] == 0


def test_controller_semantic_failure_uses_local_fallback_without_json_repair(tmp_path: Path) -> None:
    db = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    run_id = db.create_run("controller fallback", Budget(max_llm_calls=4))
    model = _JsonModel(
        {
            "action": "SEARCH",
            "queries": ["already searched"],
            "reason": "repeat",
            "expected_information_gain": 0.8,
        }
    )
    controller = EvidenceDrivenController(PersistentLLMRuntime(db, model))
    decision = controller.refine_search(
        run_id=run_id,
        claim={"statement": "Method A improves Dataset B.", "metadata": {}},
        gap="primary_source",
        previous_queries=["already searched"],
        evidence=[],
        budget={},
    )

    assert decision["action"] == "REFORMULATE"
    assert "controller_fallback_reason" in decision
    assert model.calls == 1
    assert [item["stage"] for item in db.list_llm_calls(run_id)] == ["controller"]
    qualitative = EvidenceDrivenController._validate_decision(
        {
            "action": "SEARCH",
            "queries": ["novel query"],
            "reason": "useful",
            "expected_information_gain": "High",
        },
        [],
    )
    assert qualitative["expected_information_gain"] == pytest.approx(0.8)


def test_controller_preserves_last_research_call_for_verifier(tmp_path: Path) -> None:
    db = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    run_id = db.create_run("budget-aware controller", Budget(max_llm_calls=2))
    model = _JsonModel({"action": "ABSTAIN", "queries": []})
    controller = EvidenceDrivenController(PersistentLLMRuntime(db, model))

    decision = controller.refine_search(
        run_id=run_id,
        claim={"statement": "Flow alignment compensates displacement.", "metadata": {}},
        gap="mechanism",
        previous_queries=[],
        evidence=[],
        budget={
            "max_llm_calls": 2,
            "synthesis_reserve_calls": 1,
            "repair_reserve_calls": 0,
            "used": {"used_llm_calls": 0},
        },
    )

    assert decision["action"] == "REFORMULATE"
    assert decision["controller"] == "budget-aware-deterministic-controller-v1"
    assert model.calls == 0
    assert db.list_llm_calls(run_id) == []


def test_only_local_safe_tools_continue_after_research_llm_exhaustion() -> None:
    layout = ActionCandidate(
        tool_name="paper.inspect_layout",
        arguments={},
        claim_id="claim_table",
        gap="numeric_results",
        reason="reuse exact crop",
        expected_gain=0.9,
    )
    search = ActionCandidate(
        tool_name="paper.search",
        arguments={},
        claim_id="claim_text",
        gap="primary_source",
        reason="requires semantic verification",
        expected_gain=0.7,
    )
    assert ResearchAgentRuntime._can_run_after_research_llm_exhaustion(layout)
    assert not ResearchAgentRuntime._can_run_after_research_llm_exhaustion(search)


def test_verifier_budget_exhaustion_retains_neutral_auditable_evidence(
    tmp_path: Path,
) -> None:
    class ExhaustedVerifier:
        max_items_per_tool = 6

        def assess_many(self, *_args: Any, **_kwargs: Any) -> list[Any]:
            raise LLMBudgetExhausted("research calls exhausted")

    runtime = ResearchAgentRuntime(
        db_path=tmp_path / "runtime.sqlite3",
        workspace=tmp_path / "workspace",
        evidence_verifier=ExhaustedVerifier(),
    )
    run_id = runtime.db.create_run("budgeted evidence", Budget())
    claim = runtime.graph.add_claim(
        run_id=run_id,
        statement="Table XII reports F1 82.83.",
        gaps=["numeric_results"],
        required_evidence_types=["TABLE"],
    )
    verified = runtime._verify_evidence(
        [
            {
                "evidence_type": "TABLE",
                "content": "Unverified parser text: F1 82.83.",
                "source_uri": "paper://fixture/table-xii",
                "confidence": 0.82,
                "provenance": {"gap": "numeric_results"},
            }
        ],
        default_claim_id=claim.claim_id,
        run_id=run_id,
    )
    assert verified[0]["polarity"] == "NEUTRAL"
    audit = verified[0]["provenance"]["verification"]
    assert audit["verifier"] == "runtime-verifier-budget-exhausted-v1"
    assert audit["degraded"] is True
    assert "LLMBudgetExhausted" in audit["error"]


def test_run_level_json_repair_cap_is_shared_across_stages(tmp_path: Path) -> None:
    class BrokenModel(_JsonModel):
        def complete(self, _messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
            callback = kwargs.get("before_attempt")
            if callable(callback):
                callback()
            self.calls += 1
            return LLMResponse(
                text="{broken",
                prompt_tokens=10,
                completion_tokens=10,
                finish_reason="stop",
                model=self.model,
            )

    db = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    run_id = db.create_run("repair cap", Budget(max_llm_calls=6, max_repair_calls=1))
    model = BrokenModel({})
    runtime = PersistentLLMRuntime(db, model)
    with pytest.raises(LLMStructuredOutputError):
        runtime.complete_json(run_id, [{"role": "user", "content": "plan"}], policy="planner")
    with pytest.raises(LLMStructuredOutputError, match="cap is exhausted"):
        runtime.complete_json(
            run_id,
            [{"role": "user", "content": "control"}],
            policy="controller",
        )
    assert [item["stage"] for item in db.list_llm_calls(run_id)] == [
        "planner",
        "json_repair",
        "controller",
    ]
    assert model.calls == 3


def test_planner_separates_paper_experiments_and_neutralizes_invented_components() -> None:
    model = _JsonModel(
        {
            "objective": "Explain ChangeMamba from the paper.",
            "constraints": ["仅依据论文原文"],
            "output_requirements": ["区分方法与实验结论"],
            "candidate_queries": ["ChangeMamba spatiotemporal state space"],
            "claims": [
                {
                    "statement": "ChangeMamba 的核心机制包括双分支、特征融合",
                    "importance": 1.0,
                    "gaps": ["experiment_evaluation"],
                    "candidate_queries": ["ChangeMamba ablation"],
                }
            ],
        }
    )
    plan = LLMClaimPlanner(model).plan(
        "ChangeMamba 如何利用时空状态空间建模，并有哪些论文消融实验支持？"
    )
    claim = plan.claims[0]
    assert "双分支" not in claim["statement"]
    assert "特征融合" not in claim["statement"]
    assert claim["metadata"]["planner_claim_neutralized"] is True
    assert claim["gaps"] == ["numeric_results"]
    assert claim["required_evidence_types"] == ["TABLE"]
    assert claim["metadata"]["evidence_contract"]["closure"] == (
        "matching_gap_and_evidence_type"
    )


def test_planner_neutralizes_domain_drift_not_present_in_question() -> None:
    model = _JsonModel(
        {
            "objective": "Compare two remote-sensing papers.",
            "constraints": [],
            "output_requirements": [],
            "candidate_queries": [],
            "claims": [
                {
                    "statement": "FlowMamba 的 MFA 旨在解决视频帧运动对齐问题。",
                    "gaps": ["primary_source"],
                    "candidate_queries": ["FlowMamba MFA alignment"],
                }
            ],
        }
    )
    plan = LLMClaimPlanner(model).plan(
        "在遥感影像中，FlowMamba 的 MFA 解决什么问题？"
    )
    claim = plan.claims[0]
    assert "视频" not in claim["statement"]
    assert claim["metadata"]["planner_domain_neutralized"] is True


def test_planner_neutralizes_unrequested_details_from_fresh_paid_trace() -> None:
    model = _JsonModel(
        {
            "objective": "Compare ChangeMamba and FlowMamba.",
            "constraints": ["只使用可核验证据"],
            "output_requirements": ["给出两篇论文各自的消融数字"],
            "candidate_queries": [],
            "claims": [
                {
                    "statement": (
                        "ChangeMamba 的时空关系建模模块旨在解决时空上下文建模不足，"
                        "其核心机制是对时间维度和空间维度进行联合建模。"
                    ),
                    "gaps": ["primary_source"],
                    "candidate_queries": ["ChangeMamba mechanism"],
                },
                {
                    "statement": (
                        "FlowMamba 的光流对齐模块旨在解决双时相图像像素级对齐问题，"
                        "其核心机制是估计光流场并进行特征扭曲（warping）。"
                    ),
                    "gaps": ["primary_source"],
                    "candidate_queries": ["FlowMamba alignment"],
                },
                {
                    "statement": (
                        "ChangeMamba 论文中报告了时空关系建模模块的消融实验，"
                        "通过移除或替换模块验证其有效性，并给出具体指标。"
                    ),
                    "gaps": ["numeric_results"],
                    "candidate_queries": ["ChangeMamba ablation"],
                },
                {
                    "statement": (
                        "FlowMamba 论文中报告了光流对齐模块的消融实验，"
                        "通过移除或替换模块验证其有效性，并给出具体指标。"
                    ),
                    "gaps": ["numeric_results"],
                    "candidate_queries": ["FlowMamba ablation"],
                },
            ],
        }
    )
    question = (
        "请比较 ChangeMamba 的时空关系建模与 FlowMamba 的光流对齐模块："
        "它们分别要解决什么问题、核心机制有什么不同，并引用两篇论文各自的"
        "消融实验数字说明模块是否有效。"
    )
    plan = LLMClaimPlanner(model).plan(question)
    assert len(plan.claims) == 4
    mechanism_claims = [
        claim for claim in plan.claims if claim["gaps"] == ["primary_source"]
    ]
    numeric_claims = [
        claim for claim in plan.claims if claim["gaps"] == ["numeric_results"]
    ]
    assert all("明确" in claim["statement"] for claim in mechanism_claims)
    assert all("时间维度" not in claim["statement"] for claim in mechanism_claims)
    assert all("光流场" not in claim["statement"] for claim in mechanism_claims)
    assert all(claim["metadata"]["planner_mechanism_neutralized"] for claim in mechanism_claims)
    assert all(
        "移除" not in claim["statement"]
        and "去除" not in claim["statement"]
        and "替换" not in claim["statement"]
        for claim in numeric_claims
    )
    assert all(claim["metadata"]["planner_ablation_design_neutralized"] for claim in numeric_claims)


def test_planner_neutralizes_operates_on_details_from_interview_trace() -> None:
    model = _JsonModel(
        {
            "objective": "Compare two local papers.",
            "constraints": ["Use only the named papers."],
            "output_requirements": ["Separate facts from inference."],
            "candidate_queries": [],
            "claims": [
                {
                    "statement": (
                        "In ChangeMamba, the Sequential, Cross, and Parallel token "
                        "organization strategies operate on specific intermediate "
                        "representations to address temporal dependency modeling or "
                        "cross-modal interaction."
                    ),
                    "gaps": ["primary_source"],
                    "candidate_queries": [
                        "ChangeMamba Sequential Cross Parallel token organization"
                    ],
                }
            ],
        }
    )
    question = (
        "仅依据 ChangeMamba，说明 Sequential、Cross、Parallel 三种 token 组织分别"
        "作用于什么表示、解决什么问题。"
    )

    claim = LLMClaimPlanner(model).plan(question).claims[0]

    assert "specific intermediate" not in claim["statement"]
    assert "cross-modal interaction" not in claim["statement"]
    assert all(name in claim["statement"] for name in ("Sequential", "Cross", "Parallel"))
    assert claim["metadata"]["planner_mechanism_neutralized"] is True


def test_planner_neutralizes_organizes_by_guesses_from_paid_trace() -> None:
    model = _JsonModel(
        {
            "objective": "Explain three ChangeMamba mechanisms.",
            "constraints": ["Use only the paper."],
            "output_requirements": ["Separate method and results."],
            "candidate_queries": [],
            "claims": [
                {
                    "statement": (
                        "The Sequential mechanism in ChangeMamba organizes bi-temporal "
                        "features by processing them in a specific sequential order."
                    ),
                    "gaps": ["primary_source"],
                    "candidate_queries": ["ChangeMamba Sequential token organization"],
                },
                {
                    "statement": (
                        "The Cross mechanism in ChangeMamba organizes bi-temporal features "
                        "by enabling interaction or cross-attention between time phases."
                    ),
                    "gaps": ["primary_source"],
                    "candidate_queries": ["ChangeMamba Cross token organization"],
                },
                {
                    "statement": (
                        "The Parallel mechanism in ChangeMamba organizes bi-temporal "
                        "features through parallel branches before fusion."
                    ),
                    "gaps": ["primary_source"],
                    "candidate_queries": ["ChangeMamba Parallel token organization"],
                },
            ],
        }
    )
    question = (
        "仅依据 ChangeMamba，说明 Sequential、Cross、Parallel 三种机制如何组织"
        "双时相特征。"
    )

    claims = LLMClaimPlanner(model).plan(question).claims

    assert len(claims) == 3
    assert ["Sequential", "Cross", "Parallel"] == [
        next(name for name in ("Sequential", "Cross", "Parallel") if name in claim["statement"])
        for claim in claims
    ]
    assert all("organizes" not in claim["statement"] for claim in claims)
    assert all("parallel branches" not in claim["statement"] for claim in claims)
    assert all(
        claim["metadata"]["planner_mechanism_neutralized"] is True
        for claim in claims
    )


def test_planner_does_not_build_a_subject_from_an_assertive_verb_tail() -> None:
    model = _JsonModel(
        {
            "objective": "Explain one ChangeMamba mechanism.",
            "constraints": ["Use only the paper."],
            "output_requirements": [],
            "candidate_queries": [],
            "claims": [
                {
                    "statement": (
                        "The paper defines the Sequential mechanism's specific process "
                        "for organizing bi-temporal tokens in temporal order."
                    ),
                    "gaps": ["primary_source"],
                    "candidate_queries": ["ChangeMamba Sequential token organization"],
                }
            ],
        }
    )
    question = "ChangeMamba 的 Sequential 机制如何组织双时相特征？"

    claim = LLMClaimPlanner(model).plan(question).claims[0]

    assert claim["statement"] == "论文原文明确说明了 Sequential 机制作用的输入或表示变换。"
    assert "paper defines" not in claim["statement"].casefold()
    assert claim["metadata"]["planner_mechanism_neutralized"] is True


def test_planner_removes_invented_acronym_expansion_and_clarification_tail() -> None:
    model = _JsonModel(
        {
            "objective": "Compare two alignment mechanisms.",
            "constraints": [],
            "output_requirements": [],
            "candidate_queries": [],
            "claims": [
                {
                    "statement": (
                        "In FlowMamba, the Multi-scale Feature Alignment (MFA) module "
                        "uses optical flow to align features."
                    ),
                    "gaps": ["primary_source"],
                    "candidate_queries": ["FlowMamba MFA optical flow"],
                },
                {
                    "statement": (
                        "ChangeMamba uses token organization, but the primary source "
                        "must clarify if it explicitly handles pixel misalignment."
                    ),
                    "gaps": ["primary_source"],
                    "candidate_queries": ["ChangeMamba token organization"],
                },
            ],
        }
    )
    question = (
        "FlowMamba 的 MFA 与 ChangeMamba 的 token 组织中，哪一种更直接处理"
        "双时相空间错位？"
    )

    plan = LLMClaimPlanner(model).plan(question)

    assert len(plan.claims) == 2
    flow, change = plan.claims
    assert "Multi-scale" not in flow["statement"]
    assert "MFA" in flow["statement"]
    assert flow["metadata"]["planner_acronym_expansion_neutralized"] is True
    assert "must clarify" not in change["statement"]
    assert change["metadata"]["planner_clarification_tail_stripped"] is True


def test_planner_drops_inferred_absence_comparison_from_follow_up_trace() -> None:
    model = _JsonModel(
        {
            "objective": "Compare which design handles misalignment more directly.",
            "constraints": ["Use source facts only."],
            "output_requirements": ["Separate facts from inference."],
            "candidate_queries": [],
            "claims": [
                {
                    "statement": (
                        "ChangeMamba defines Sequential, Cross, and Parallel token "
                        "organization for bi-temporal features."
                    ),
                    "gaps": ["primary_source"],
                    "candidate_queries": ["ChangeMamba Sequential Cross Parallel"],
                },
                {
                    "statement": (
                        "FlowMamba uses optical-flow alignment to mitigate "
                        "bi-temporal misregistration."
                    ),
                    "gaps": ["primary_source"],
                    "candidate_queries": ["FlowMamba optical flow misregistration"],
                },
                {
                    "statement": (
                        "The comparison of directness relies on explicit geometric "
                        "warping versus token reordering. The primary sources must be "
                        "checked to confirm ChangeMamba lacks explicit alignment."
                    ),
                    "gaps": ["primary_source"],
                    "candidate_queries": ["ChangeMamba geometric correction"],
                },
            ],
        }
    )
    question = (
        "基于两篇论文，哪一种设计更直接处理双时相空间错位？请区分论文事实与"
        "比较性推断，并说明结论边界。"
    )

    plan = LLMClaimPlanner(model).plan(question)

    assert len(plan.claims) == 2
    assert all("versus" not in item["statement"] for item in plan.claims)
    assert any("综合阶段" in item for item in plan.output_requirements)


def test_paid_a2_failure_plan_replay_becomes_two_typed_table_claims() -> None:
    """Replay the failure shape observed in the 2026-08-29 paid A2 run."""

    model = _JsonModel(
        {
            "objective": "Compare ChangeMamba and FlowMamba.",
            "constraints": ["仅依据两篇论文原文"],
            "output_requirements": ["给出各自消融结果"],
            "candidate_queries": [],
            "claims": [
                {
                    "statement": "ChangeMamba 提出时空 token 建模，并有相应消融实验。",
                    "gaps": ["paper_experiment"],
                    "candidate_queries": ["ChangeMamba token modeling ablation"],
                },
                {
                    "statement": "FlowMamba 提出 MFA 对齐模块，并有相应消融实验。",
                    "gaps": ["paper_experiment"],
                    "candidate_queries": ["FlowMamba MFA ablation"],
                },
                {
                    "statement": "ChangeMamba 的时空建模旨在解决视频变化检测。",
                    "gaps": ["primary_source"],
                    "candidate_queries": ["ChangeMamba mechanism"],
                },
                {
                    "statement": "FlowMamba 的 MFA 旨在解决视频帧运动对齐。",
                    "gaps": ["primary_source"],
                    "candidate_queries": ["FlowMamba MFA purpose"],
                },
            ],
        }
    )
    question = (
        "仅依据 ChangeMamba 与 FlowMamba 两篇原文，对比前者的时空 token 建模与"
        "后者的 MFA 光流对齐：它们各自解决什么问题，哪些各自论文中的消融结果支持"
        "对应模块？不要把一篇论文的模块或结果归到另一篇。"
    )
    plan = LLMClaimPlanner(model).plan(question)
    numeric_claims = [
        claim for claim in plan.claims if claim["gaps"] == ["numeric_results"]
    ]
    mechanism_claims = [
        claim for claim in plan.claims if claim["gaps"] == ["primary_source"]
    ]
    assert len(numeric_claims) == 2
    assert all(claim["required_evidence_types"] == ["TABLE"] for claim in numeric_claims)
    assert len(mechanism_claims) == 2
    assert all("视频" not in claim["statement"] for claim in mechanism_claims)


def test_text_cannot_close_numeric_gap_but_matching_table_can(tmp_path: Path) -> None:
    db = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    run_id = db.create_run("typed evidence gap", Budget())
    graph = ClaimGraph(db)
    claim = graph.add_claim(
        run_id=run_id,
        statement="ChangeMamba Table XII reports its ablation values.",
        gaps=["numeric_results"],
        required_evidence_types=["TABLE"],
        metadata={"pdf_paths": [str(tmp_path / "ChangeMamba.pdf")]},
    )
    ledger = EvidenceLedger(db)
    ledger.add(
        run_id=run_id,
        claim_id=claim.claim_id,
        evidence_type="TEXT",
        content=(
            "The paper contains ablation studies. They are spatiotemporal sequential "
            "modeling, spatiotemporal cross modeling, and spatial-temporal parallel modeling."
        ),
        polarity="SUPPORTS",
        source_uri="paper://fixture/method-text",
        confidence=0.9,
        provenance={"gap": "numeric_results"},
    )
    after_text = graph.update_from_evidence(claim.claim_id)
    assert after_text.gaps == ["numeric_results"]
    next_action = EvidenceGapScheduler(db, graph).select_next(run_id, budget=Budget())
    assert next_action is not None
    assert next_action.tool_name == "paper.search"
    graph.mark_attempt(claim.claim_id, "numeric_results", success=True)
    layout_action = EvidenceGapScheduler(db, graph).select_next(run_id, budget=Budget())
    assert layout_action is not None
    assert layout_action.tool_name == "paper.inspect_layout"
    assert "spatiotemporal sequential modeling" in layout_action.arguments["query"]

    ledger.add(
        run_id=run_id,
        claim_id=claim.claim_id,
        evidence_type="TABLE",
        content="TABLE XII. Ablation study of the proposed components.",
        polarity="SUPPORTS",
        source_uri="paper://fixture/table-caption",
        confidence=0.9,
        provenance={"gap": "numeric_results"},
    )
    after_caption = graph.update_from_evidence(claim.claim_id)
    assert after_caption.gaps == ["numeric_results"]

    ledger.add(
        run_id=run_id,
        claim_id=claim.claim_id,
        evidence_type="TABLE",
        content="TABLE XII baseline F1 81.11; all mechanisms F1 82.83.",
        polarity="SUPPORTS",
        source_uri="paper://fixture/table-xii",
        confidence=0.9,
        provenance={"gap": "numeric_results"},
    )
    after_table = graph.update_from_evidence(claim.claim_id)
    assert after_table.gaps == []
    assert after_table.status.value == "SUPPORTED"


def test_strict_named_paper_numeric_claim_inspects_layout_before_text_search(
    tmp_path: Path,
) -> None:
    db = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    run_id = db.create_run("strict named paper table", Budget())
    graph = ClaimGraph(db)
    claim = graph.add_claim(
        run_id=run_id,
        statement="ChangeMamba Table XII reports its ablation values.",
        gaps=["numeric_results"],
        required_evidence_types=["TABLE"],
        metadata={
            "pdf_paths": [str(tmp_path / "ChangeMamba.pdf")],
            "paper_sha256s": ["a" * 64],
            "source_scope_mode": "strict_explicit_title",
            "candidate_queries": ["ChangeMamba ablation Table XII"],
        },
    )
    action = EvidenceGapScheduler(db, graph).select_next(run_id, budget=Budget())
    assert action is not None
    assert action.claim_id == claim.claim_id
    assert action.tool_name == "paper.inspect_layout"
    assert action.arguments["attempt_index"] == 1
    assert "ChangeMamba ablation" in action.arguments["query"]


def test_named_table_is_scheduled_before_higher_importance_method_search(
    tmp_path: Path,
) -> None:
    db = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    run_id = db.create_run("tight multi-claim provider budget", Budget())
    graph = ClaimGraph(db)
    table_claim = graph.add_claim(
        run_id=run_id,
        statement="ChangeMamba Table XII reports its ablation values.",
        importance=0.8,
        gaps=["numeric_results"],
        required_evidence_types=["TABLE"],
        metadata={
            "pdf_paths": [str(tmp_path / "ChangeMamba.pdf")],
            "paper_sha256s": ["a" * 64],
            "source_scope_mode": "strict_explicit_title",
            "candidate_queries": ["ChangeMamba Table XII ablation"],
        },
    )
    graph.add_claim(
        run_id=run_id,
        statement="ChangeMamba explains its token organization mechanism.",
        importance=0.9,
        gaps=["primary_source"],
        required_evidence_types=["TEXT"],
        metadata={
            "paper_sha256s": ["a" * 64],
            "candidate_queries": ["ChangeMamba token organization"],
        },
    )
    action = EvidenceGapScheduler(db, graph).select_next(run_id, budget=Budget())
    assert action is not None
    assert action.claim_id == table_claim.claim_id
    assert action.tool_name == "paper.inspect_layout"


def test_visual_budget_prioritizes_parser_incomplete_table() -> None:
    selected = [
        {"object_id": "parsed-high-rank"},
        {"object_id": "caption-only-target"},
        {"object_id": "parsed-low-rank"},
    ]
    evidence = [
        {
            "provenance": {
                "needs_visual_verification": False,
                "retrieval_score": 0.9,
            }
        },
        {
            "provenance": {
                "needs_visual_verification": True,
                "retrieval_score": 0.6,
            }
        },
        {
            "provenance": {
                "needs_visual_verification": False,
                "retrieval_score": 0.4,
            }
        },
    ]
    ordered = _prioritize_visual_candidates(selected, evidence)
    assert ordered[0][0]["object_id"] == "caption-only-target"


def test_explicit_layout_label_filters_decoy_tables() -> None:
    target = _explicit_layout_label(
        "Read ChangeMamba Table XII ablation values.",
        "table",
    )
    assert target == ("table", "XII")
    assert _matches_explicit_layout_label(
        {"caption": "TABLE XII", "text": "OA F1 IoU"},
        target,
    )
    assert not _matches_explicit_layout_label(
        {"caption": "TABLE VII", "text": "another ablation"},
        target,
    )


def test_layout_query_keeps_named_table_when_sibling_hints_exist(tmp_path: Path) -> None:
    db = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    run_id = db.create_run("named table with sibling hints", Budget())
    graph = ClaimGraph(db)
    table_claim = graph.add_claim(
        run_id=run_id,
        statement="ChangeMamba Table XII reports its ablation values.",
        gaps=["numeric_results"],
        required_evidence_types=["TABLE"],
        metadata={
            "pdf_paths": [str(tmp_path / "ChangeMamba.pdf")],
            "paper_sha256s": ["a" * 64],
            "source_scope_mode": "strict_explicit_title",
            "candidate_queries": ["ChangeMamba Table XII ablation"],
        },
    )
    sibling = graph.add_claim(
        run_id=run_id,
        statement="ChangeMamba uses an STSS mechanism.",
        gaps=["primary_source"],
        metadata={"paper_sha256s": ["a" * 64]},
    )
    EvidenceLedger(db).add(
        run_id=run_id,
        claim_id=sibling.claim_id,
        evidence_type="TEXT",
        content="The STSS block contains Sequential Cross Parallel mechanisms.",
        polarity="SUPPORTS",
        source_uri="paper://fixture/page/7",
        confidence=0.9,
        provenance={"paper_sha256": "a" * 64, "gap": "primary_source"},
    )
    scheduler = EvidenceGapScheduler(db, graph)
    action = next(
        item
        for item in scheduler._for_claim(table_claim, Budget())
        if item.tool_name == "paper.inspect_layout"
    )
    assert "Table XII" in action.arguments["query"]
    assert "ChangeMamba Table XII ablation" in action.arguments["query"]


def test_strict_visual_table_closes_numeric_gap_without_second_llm_call(
    tmp_path: Path,
) -> None:
    class RejectingVerifier:
        max_items_per_tool = 6

        def assess_many(self, *_args: Any, **_kwargs: Any) -> list[Any]:
            raise AssertionError("strict visual table must not consume a second verifier call")

    runtime = ResearchAgentRuntime(
        db_path=tmp_path / "runtime.sqlite3",
        workspace=tmp_path / "workspace",
        evidence_verifier=RejectingVerifier(),
    )
    run_id = runtime.db.create_run("visual numeric evidence", Budget())
    claim = runtime.graph.add_claim(
        run_id=run_id,
        statement="ChangeMamba Table XII reports the ablation results.",
        gaps=["numeric_results"],
        required_evidence_types=["TABLE"],
    )
    verified = runtime._verify_evidence(
        [
            {
                "evidence_type": "TABLE",
                "content": "TABLE XII baseline F1 81.11; all mechanisms F1 82.83.",
                "source_uri": "paper://fixture/layout/table-xii",
                "confidence": 0.91,
                "provenance": {
                    "gap": "numeric_results",
                    "vision_verified": True,
                    "vision_metadata": {"table_structure_valid": True},
                },
            }
        ],
        default_claim_id=claim.claim_id,
        run_id=run_id,
    )
    assert verified[0]["polarity"] == "SUPPORTS"
    assert (
        verified[0]["provenance"]["verification"]["verifier"]
        == "runtime-verified-visual-table-v1"
    )


def test_two_paper_run_covers_both_tables_without_semantic_verifier_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    change_sha = "a" * 64
    flow_sha = "b" * 64
    change_pdf = tmp_path / "ChangeMamba.pdf"
    flow_pdf = tmp_path / "FlowMamba.pdf"
    change_pdf.write_bytes(b"change-fixture")
    flow_pdf.write_bytes(b"flow-fixture")

    def fake_parse_pdf_layout(pdf_path: Path, output: Path, **_kwargs: Any) -> dict[str, Any]:
        output.mkdir(parents=True, exist_ok=True)
        artifact = output / "target-table.png"
        artifact.write_bytes(b"fixture-image")
        decoy_artifact = output / "decoy-table.png"
        decoy_artifact.write_bytes(b"decoy-image")
        is_change = "ChangeMamba" in pdf_path.name
        paper_sha = change_sha if is_change else flow_sha
        label = "TABLE XII ABLATION" if is_change else "TABLE VII MFA ABLATION"
        decoy_label = "TABLE XI" if is_change else "TABLE VI"
        manifest = {
            "parser": "fixture",
            "warnings": [],
            "objects": [
                {
                    "object_id": "target-table",
                    "object_type": "table",
                    "page": 16 if is_change else 10,
                    "bbox": [0, 0, 100, 100],
                    "text": label,
                    "caption": label,
                    "artifact_path": str(artifact),
                    "document_sha256": paper_sha,
                    "metadata": {"rows": 0, "caption_only": True},
                },
                {
                    "object_id": "decoy-table",
                    "object_type": "table",
                    "page": 15 if is_change else 9,
                    "bbox": [0, 0, 100, 100],
                    "text": f"{decoy_label} unrelated benchmark",
                    "caption": decoy_label,
                    "artifact_path": str(decoy_artifact),
                    "document_sha256": paper_sha,
                    "metadata": {"rows": 0, "caption_only": True},
                },
            ],
        }
        (output / "layout_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return manifest

    monkeypatch.setattr(
        "paperpilot_agentic.builtin_tools.parse_pdf_layout",
        fake_parse_pdf_layout,
    )

    class ScopedSearch:
        def search(
            self,
            query: str,
            top_k: int = 8,
            **kwargs: Any,
        ) -> list[dict[str, Any]]:
            del top_k
            paper_sha = str((kwargs.get("paper_sha256s") or [change_sha])[0])
            name = "ChangeMamba" if paper_sha == change_sha else "FlowMamba"
            return [
                {
                    "content": f"{name} method mechanism directly addresses the stated problem.",
                    "source_uri": f"paper://{paper_sha}/page/1",
                    "page": 1,
                    "paper_sha256": paper_sha,
                    "confidence": 0.95,
                    "title": name,
                }
            ]

    class StrictFixtureVision:
        def __init__(self) -> None:
            self.calls = 0

        def analyze(self, _path: str, prompt: str, _metadata: Any = None) -> dict[str, Any]:
            self.calls += 1
            is_change = "ChangeMamba" in prompt
            content = (
                "TABLE XII baseline F1 81.11; all mechanisms F1 82.83."
                if is_change
                else "TABLE VII MFA-only F1overall 80.26; baseline 80.14."
            )
            return {
                "description": content,
                "confidence": 0.93,
                "verified": True,
                "metadata": {
                    "table_structure_valid": True,
                    "evidence_relevant": True,
                    "schema": "fixture-strict-visual",
                },
            }

    class CountingTextVerifier:
        max_items_per_tool = 6

        def __init__(self) -> None:
            self.calls = 0

        def assess_many(
            self, _claim: str, evidence: list[dict[str, Any]], **_kwargs: Any
        ) -> list[EvidenceAssessment]:
            self.calls += 1
            return [
                EvidenceAssessment(
                    polarity=EvidencePolarity.SUPPORTS,
                    confidence=0.93,
                    relevant=True,
                    reason="fixture text supports mechanism",
                    verifier="fixture-text-verifier",
                    claim_coverage=1.0,
                )
                for _ in evidence
            ]

    vision = StrictFixtureVision()
    verifier = CountingTextVerifier()
    runtime = ResearchAgentRuntime(
        db_path=tmp_path / "runtime.sqlite3",
        workspace=tmp_path / "workspace",
        search_adapter=ScopedSearch(),
        vision_adapter=vision,
        evidence_verifier=verifier,
    )
    result = runtime.start(
        question="Compare ChangeMamba and FlowMamba mechanisms and ablations.",
        claims=[
            {
                "statement": "ChangeMamba reports its ablation values.",
                "gaps": ["numeric_results"],
                "required_evidence_types": ["TABLE"],
                "metadata": {
                    "paper_sha256s": [change_sha],
                    "pdf_paths": [str(change_pdf)],
                    "source_scope_mode": "strict_explicit_title",
                    "candidate_queries": ["ChangeMamba Table XII ablation"],
                },
            },
            {
                "statement": "FlowMamba reports its MFA ablation values.",
                "gaps": ["numeric_results"],
                "required_evidence_types": ["TABLE"],
                "metadata": {
                    "paper_sha256s": [flow_sha],
                    "pdf_paths": [str(flow_pdf)],
                    "source_scope_mode": "strict_explicit_title",
                    "candidate_queries": ["FlowMamba Table VII MFA ablation"],
                },
            },
        ],
        budget=Budget(max_steps=12, max_tool_calls=8),
    )

    assert result["run"]["status"] == "COMPLETED"
    numeric_claims = [
        item
        for item in result["claim_graph"]["claims"]
        if "ablation values" in item["statement"]
    ]
    assert len(numeric_claims) == 2
    assert {item["status"] for item in numeric_claims} == {"SUPPORTED"}
    # Only the two explicitly named tables are sent to vision; the two decoy
    # tables remain local layout evidence and consume no provider calls.
    assert vision.calls == 2
    assert verifier.calls == 0
    visual_verifiers = {
        (item.get("provenance") or {}).get("verification", {}).get("verifier")
        for item in result["evidence"]
        if item["evidence_type"] == "TABLE"
    }
    assert visual_verifiers == {"runtime-verified-visual-table-v1"}


def test_numeric_layout_hints_reuse_only_same_paper_method_evidence(
    tmp_path: Path,
) -> None:
    db = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    run_id = db.create_run("same-paper discovery", Budget())
    graph = ClaimGraph(db)
    paper_a = "a" * 64
    paper_b = "b" * 64
    method_a = graph.add_claim(
        run_id=run_id,
        statement="Paper A mechanism",
        gaps=["primary_source"],
        metadata={"paper_sha256s": [paper_a]},
    )
    method_b = graph.add_claim(
        run_id=run_id,
        statement="Paper B mechanism",
        gaps=["primary_source"],
        metadata={"paper_sha256s": [paper_b]},
    )
    numeric = graph.add_claim(
        run_id=run_id,
        statement="Paper A ablation",
        gaps=["numeric_results"],
        required_evidence_types=["TABLE"],
        metadata={
            "paper_sha256s": [paper_a],
            "pdf_paths": [str(tmp_path / "paper-a.pdf")],
            "candidate_queries": ["Paper A ablation"],
        },
    )
    ledger = EvidenceLedger(db)
    ledger.add(
        run_id=run_id,
        claim_id=method_a.claim_id,
        evidence_type="TEXT",
        content="They are sequential modeling, cross modeling, and parallel modeling.",
        polarity="SUPPORTS",
        source_uri="paper://a/method",
        confidence=0.9,
        provenance={"gap": "primary_source", "paper_sha256": paper_a},
    )
    ledger.add(
        run_id=run_id,
        claim_id=method_b.claim_id,
        evidence_type="TEXT",
        content="They are unrelated optical flow offsets and another module.",
        polarity="SUPPORTS",
        source_uri="paper://b/method",
        confidence=0.9,
        provenance={"gap": "primary_source", "paper_sha256": paper_b},
    )
    scheduler = EvidenceGapScheduler(db, graph)
    hints = scheduler._discovery_hints(numeric, "numeric_results")
    assert any("sequential modeling" in item for item in hints)
    assert all("optical flow" not in item for item in hints)

    graph.mark_attempt(numeric.claim_id, "numeric_results", success=False)
    action = scheduler._action_for_gap(numeric, "numeric_results", attempt_index=1)
    assert action is not None and action.tool_name == "paper.inspect_layout"
    assert "sequential modeling" in action.arguments["query"]


def test_structured_synthesis_renders_claim_linked_citations() -> None:
    model = _JsonModel(
        {
            "sections": [
                {
                    "claim_id": "claim_1",
                    "text": "Table XII shows the combined design performs best.",
                    "evidence_ids": ["evidence_1"],
                }
            ],
            "unresolved": [],
        }
    )
    answer = LLMAnswerSynthesizer(model).synthesize(
        question="Which ablation supports the design?",
        claims=[
            {
                "claim_id": "claim_1",
                "statement": "The combined design performs best.",
                "status": "SUPPORTED",
                "confidence": 0.9,
                "gaps": [],
                "metadata": {},
            }
        ],
        evidence=[
            {
                "evidence_id": "evidence_1",
                "claim_id": "claim_1",
                "polarity": "SUPPORTS",
                "evidence_type": "TABLE",
                "source_uri": "paper://fixture/table-xii",
                "content": "Baseline F1 81.11; all mechanisms F1 82.83.",
            }
        ],
    )
    assert answer["mode"] == "llm-grounded-synthesis-v3"
    assert answer["citation_ids"] == ["evidence_1"]
    assert answer["answer"].endswith("[evidence_1]")


def test_synthesis_prompt_includes_evidence_grounded_mechanism_requirements() -> None:
    class PromptCaptureModel:
        model = "qwen-fixture"

        def __init__(self) -> None:
            self.messages: list[dict[str, str]] = []

        def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
            callback = kwargs.get("before_attempt")
            if callable(callback):
                callback()
            self.messages = messages
            return LLMResponse(
                text=json.dumps(
                    {
                        "sections": [
                            {
                                "claim_id": "claim_parallel",
                                "text": (
                                    "Parallel 沿通道维度拼接两个时相 token，"
                                    "再对其进行联合建模。"
                                ),
                                "evidence_ids": ["evidence_parallel"],
                            }
                        ],
                        "unresolved": [],
                    },
                    ensure_ascii=False,
                ),
                prompt_tokens=100,
                completion_tokens=50,
                finish_reason="stop",
                model=self.model,
            )

    model = PromptCaptureModel()
    answer = LLMAnswerSynthesizer(model).synthesize(
        question="Parallel 机制如何组织双时相特征？",
        claims=[
            {
                "claim_id": "claim_parallel",
                "statement": (
                    "The primary source contains a concrete definition of the input or "
                    "representation transformation used by the Parallel mechanism."
                ),
                "status": "SUPPORTED",
                "confidence": 0.9,
                "gaps": ["primary_source"],
                "metadata": {},
            }
        ],
        evidence=[
            {
                "evidence_id": "evidence_parallel",
                "claim_id": "claim_parallel",
                "claim_ids": ["claim_parallel"],
                "claim_polarities": {"claim_parallel": "SUPPORTS"},
                "polarity": "SUPPORTS",
                "evidence_type": "TEXT",
                "source_uri": "paper://fixture/parallel",
                "content": (
                    "Parallel modeling concatenates temporal tokens in the channel "
                    "dimension and then performs joint modeling."
                ),
            }
        ],
    )

    payload = json.loads(model.messages[-1]["content"])
    prompt_claim = payload["claims"][0]
    assert "Never infer an acronym's full form" in model.messages[0]["content"]
    assert "retain the acronym without expanding it" in model.messages[0]["content"]
    assert prompt_claim["required_operation_concepts"] == {
        "Parallel": ["channel_dimension", "concatenation", "joint_modeling"]
    }
    assert prompt_claim["grounded_operation_hints"] == {
        "Parallel": [
            "两个时相表示的拼接操作发生在通道维度",
            "拼接两个时相的 token/表示",
            "对两个时相的表示进行联合建模",
        ]
    }
    assert answer["mode"] == "llm-grounded-synthesis-v3"


def test_synthesis_repairs_generic_mechanism_answer_from_original_evidence(
    tmp_path: Path,
) -> None:
    class MechanismRepairModel:
        model = "qwen-fixture"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
            callback = kwargs.get("before_attempt")
            if callable(callback):
                callback()
            self.calls += 1
            text = (
                {
                    "sections": [
                        {
                            "claim_id": "claim_parallel",
                            "text": "Parallel 通过并行分支处理特征并进行融合。",
                            "evidence_ids": ["evidence_parallel"],
                        }
                    ],
                    "unresolved": [],
                }
                if self.calls == 1
                else {
                    "sections": [
                        {
                            "claim_id": "claim_parallel",
                            "text": (
                                "Parallel 沿通道维度拼接两个时相的 token，"
                                "随后进行联合建模。"
                            ),
                            "evidence_ids": ["evidence_parallel"],
                        }
                    ],
                    "unresolved": [],
                }
            )
            return LLMResponse(
                text=json.dumps(text, ensure_ascii=False),
                prompt_tokens=100,
                completion_tokens=50,
                finish_reason="stop",
                model=self.model,
            )

    db = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    run_id = db.create_run(
        "Parallel 机制如何组织双时相特征？",
        Budget(
            max_llm_calls=2,
            max_tokens=20_000,
            synthesis_reserve_tokens=4_000,
            repair_reserve_tokens=4_000,
            synthesis_reserve_calls=1,
            repair_reserve_calls=1,
            max_repair_calls=1,
        ),
    )
    model = MechanismRepairModel()
    answer = LLMAnswerSynthesizer(PersistentLLMRuntime(db, model)).synthesize(
        run_id=run_id,
        question="Parallel 机制如何组织双时相特征？",
        claims=[
            {
                "claim_id": "claim_parallel",
                "statement": "The Parallel mechanism in ChangeMamba must be established.",
                "status": "SUPPORTED",
                "confidence": 0.9,
                "gaps": ["primary_source"],
                "metadata": {},
            }
        ],
        evidence=[
            {
                "evidence_id": "evidence_parallel",
                "claim_id": "claim_parallel",
                "claim_ids": ["claim_parallel"],
                "claim_polarities": {"claim_parallel": "SUPPORTS"},
                "polarity": "SUPPORTS",
                "evidence_type": "TEXT",
                "source_uri": "paper://fixture/parallel",
                "content": (
                    "Parallel modeling is done by concatenating temporal tokens in the "
                    "channel dimension and then performing joint modeling."
                ),
            }
        ],
    )

    assert "通道维度拼接" in answer["answer"]
    assert "联合建模" in answer["answer"]
    assert [item["stage"] for item in db.list_llm_calls(run_id)] == [
        "synthesizer",
        "json_repair",
    ]


def test_finalized_partial_requires_verified_non_neutral_evidence(tmp_path: Path) -> None:
    db = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    run_id = db.create_run("partial semantics", Budget())
    graph = ClaimGraph(db)
    claim = graph.add_claim(
        run_id=run_id,
        statement="Method A improves Dataset B.",
        gaps=["primary_source", "numeric_results"],
        required_evidence_types=["TEXT"],
    )
    EvidenceLedger(db).add(
        run_id=run_id,
        claim_id=claim.claim_id,
        evidence_type="TEXT",
        content="The primary paper describes Method A on Dataset B.",
        polarity="SUPPORTS",
        source_uri="paper://fixture/chunk-1",
        confidence=0.9,
        provenance={"gap": "primary_source"},
    )
    assert graph.update_from_evidence(claim.claim_id).status.value == "PARTIAL"
    finalized = graph.finalize_unresolved(claim.claim_id, "numeric table still missing")
    assert finalized.status.value == "PARTIAL"
    assert finalized.metadata["finalized"] is True


def _add_strict_semantic_fixture(
    ledger: EvidenceLedger,
    *,
    run_id: str,
    claim_id: str,
    claim_sha: str,
    evidence_sha: str | None = None,
    verifier: str = "llm-entailment-v1",
    degraded: bool = False,
    semantic_entailment: bool | None = True,
    suffix: str = "",
) -> None:
    quote = (
        "The MFA module aligns pre- and post-disaster features to mitigate "
        "misregistration in bi-temporal building damage assessment."
    )
    verification: dict[str, Any] = {
        "polarity": "SUPPORTS",
        "confidence": 0.91,
        "relevant": True,
        "reason": "Direct semantic support.",
        "verifier": verifier,
        "claim_coverage": 0.0,
        "degraded": degraded,
        "grounded_quote": quote,
        "quote_match": "exact",
    }
    if semantic_entailment is not None:
        verification["semantic_entailment"] = semantic_entailment
    ledger.add(
        run_id=run_id,
        claim_id=claim_id,
        evidence_type="TEXT",
        content=f"{quote} {suffix}".strip(),
        polarity="SUPPORTS",
        source_uri=f"paper://{evidence_sha or claim_sha}/method/{suffix or 'one'}",
        confidence=0.91,
        provenance={
            "gap": "primary_source",
            "paper_sha256": evidence_sha or claim_sha,
            "verification": verification,
        },
    )


def test_chinese_claim_closes_from_grounded_english_semantic_verdict(
    tmp_path: Path,
) -> None:
    db = RuntimeDatabase(tmp_path / "cross-lingual.sqlite3")
    run_id = db.create_run("cross-language evidence", Budget())
    graph = ClaimGraph(db)
    paper_sha = "a" * 64
    claim = graph.add_claim(
        run_id=run_id,
        statement="FlowMamba 的 MFA 模块通过对齐灾前灾后特征缓解双时相影像错位。",
        gaps=["primary_source"],
        required_evidence_types=["TEXT"],
        metadata={
            "paper_sha256s": [paper_sha],
            "source_scope_mode": "strict_explicit_title",
        },
    )
    _add_strict_semantic_fixture(
        EvidenceLedger(db),
        run_id=run_id,
        claim_id=claim.claim_id,
        claim_sha=paper_sha,
    )

    updated = graph.update_from_evidence(claim.claim_id)

    assert updated.status.value == "SUPPORTED"
    assert updated.gaps == []
    assert updated.confidence == pytest.approx(0.91)
    assert updated.metadata["closure_gate_satisfied"] is True
    assert len(updated.metadata["semantic_adjudication_evidence_ids"]) == 1


@pytest.mark.parametrize(
    ("case", "evidence_sha", "verifier", "degraded", "semantic_entailment"),
    [
        ("lexical", None, "conservative-lexical-v1", False, False),
        ("wrong-paper", "b" * 64, "llm-entailment-v1", False, True),
        ("degraded", None, "llm-entailment-v1", True, True),
    ],
)
def test_strict_claim_does_not_close_from_untrusted_semantic_paths(
    tmp_path: Path,
    case: str,
    evidence_sha: str | None,
    verifier: str,
    degraded: bool,
    semantic_entailment: bool,
) -> None:
    db = RuntimeDatabase(tmp_path / f"{case}.sqlite3")
    run_id = db.create_run(case, Budget())
    graph = ClaimGraph(db)
    paper_sha = "a" * 64
    claim = graph.add_claim(
        run_id=run_id,
        statement="FlowMamba 的 MFA 模块用于缓解错位。",
        gaps=["primary_source"],
        required_evidence_types=["TEXT"],
        metadata={"paper_sha256s": [paper_sha]},
    )
    ledger = EvidenceLedger(db)
    # Two strong-looking rows prove that quantity/lexical confidence cannot
    # bypass the strict closure gate.
    for index in range(2):
        _add_strict_semantic_fixture(
            ledger,
            run_id=run_id,
            claim_id=claim.claim_id,
            claim_sha=paper_sha,
            evidence_sha=evidence_sha,
            verifier=verifier,
            degraded=degraded,
            semantic_entailment=semantic_entailment,
            suffix=str(index),
        )

    updated = graph.update_from_evidence(claim.claim_id)

    assert updated.status.value == "PARTIAL"
    assert updated.gaps == ["primary_source"]
    assert updated.metadata["closure_gate_satisfied"] is False
    assert updated.metadata["semantic_adjudication_evidence_ids"] == []


def test_degraded_and_outcome_events_are_idempotent(tmp_path: Path) -> None:
    runtime = ResearchAgentRuntime(db_path=tmp_path / "runtime.sqlite3", workspace=tmp_path)
    initial = runtime.start(
        question="No evidence claim",
        claims=[{"statement": "No evidence claim", "gaps": ["primary_source"]}],
        auto_advance=False,
    )
    run_id = initial["run"]["run_id"]
    runtime._mark_degraded(run_id, "fixture", "same failure")
    runtime._mark_degraded(run_id, "fixture", "same failure")
    runtime._update_research_outcome(run_id)
    runtime._update_research_outcome(run_id)
    events = runtime.db.list_events(run_id)
    assert sum(item["event_type"] == "run.degraded" for item in events) == 1
    assert sum(item["event_type"] == "run.outcome" for item in events) == 1

    budget_db = RuntimeDatabase(tmp_path / "verifier-budget.sqlite3")
    budget_run = budget_db.create_run("reserved verifier budget", Budget(max_llm_calls=2))
    model = _JsonModel({"assessments": []})
    verifier = LLMEvidenceVerifier(PersistentLLMRuntime(budget_db, model))
    assessments = verifier.assess_many(
        "Method A is effective.",
        [{"evidence_type": "TEXT", "content": "Method A is effective.", "confidence": 0.9}],
        run_id=budget_run,
    )
    assert model.calls == 0
    assert assessments[0].polarity.value == "NEUTRAL"
    assert assessments[0].verifier == "runtime-verifier-budget-v2"
    assert assessments[0].degraded is False


def test_pdf_resolution_requires_matching_indexed_sha(tmp_path: Path) -> None:
    pdf = tmp_path / "Named Paper.pdf"
    pdf.write_bytes(b"fixture-pdf-bytes")
    import hashlib

    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()

    class Store:
        def paper_metadata(self, paper_sha256s: set[str]) -> list[dict[str, Any]]:
            return [
                {
                    "sha256": value,
                    "file_name": "Named Paper.pdf",
                    "title": "Named Paper Canonical Title",
                }
                for value in paper_sha256s
            ]

    adapter = PaperPilotKnowledgeSearchAdapter(Store(), paper_roots=[tmp_path])
    assert adapter.resolve_pdf_paths({digest}) == [str(pdf.resolve())]
    assert adapter.resolve_pdf_paths({"0" * 64}) == []
    assert adapter.resolve_paper_titles({digest}) == ["Named Paper Canonical Title"]


def test_named_scholar_filter_rejects_same_name_cross_domain_collision() -> None:
    expected = [
        "FlowMamba: Building Damage Assessment via Optics Flow-Based State Space Model"
    ]
    kept, filtered = _filter_named_scholarly_items(
        [
            {
                "title": expected[0],
                "url": "https://example.test/remote-sensing",
            },
            {
                "title": "FlowMamba: Learning Point Cloud Scene Flow with Global Motion Propagation",
                "url": "https://example.test/point-cloud",
            },
        ],
        expected,
    )
    assert filtered == 1
    assert [item["url"] for item in kept] == [
        "https://example.test/remote-sensing"
    ]


def test_named_paper_resolution_injects_layout_path_for_tools(tmp_path: Path) -> None:
    pdf = tmp_path / "ChangeMamba.pdf"
    pdf.write_bytes(b"fixture")
    paper_sha = "a" * 64

    class ScopedAdapter:
        def resolve_explicit_title_scope(self, _queries: Any, **_: Any) -> set[str]:
            return {paper_sha}

        def resolve_pdf_paths(self, values: set[str]) -> list[str]:
            return [str(pdf.resolve())] if values == {paper_sha} else []

        def search(self, _query: str, top_k: int = 8, **_: Any) -> list[dict[str, Any]]:
            del top_k
            return []

    runtime = ResearchAgentRuntime(
        db_path=tmp_path / "runtime.sqlite3",
        workspace=tmp_path / "workspace",
        search_adapter=ScopedAdapter(),
    )
    result = runtime.start(
        question="请仅依据 ChangeMamba 原文回答。",
        claims=[{"statement": "ChangeMamba 的表格实验", "gaps": ["numeric_results"]}],
        metadata={"allow_external_search": True},
        auto_advance=False,
    )
    claim = result["claim_graph"]["claims"][0]
    assert claim["metadata"]["paper_sha256s"] == [paper_sha]
    assert claim["metadata"]["pdf_paths"] == [str(pdf.resolve())]
    assert result["run"]["metadata"]["layout_pdf_count"] == 1
    assert result["run"]["metadata"]["allow_external_search"] is False
    assert claim["metadata"]["allow_external_search"] is False
    assert claim["metadata"]["max_gap_attempts"] == 2
    first = runtime.scheduler.select_next(result["run"]["run_id"], budget=Budget())
    assert first is not None and first.tool_name == "paper.inspect_layout"
    runtime.graph.mark_attempt(claim["claim_id"], "numeric_results", success=False)
    second = runtime.scheduler.select_next(result["run"]["run_id"], budget=Budget())
    assert second is not None and second.tool_name == "paper.search"
    assert ResearchAgentRuntime._strict_source_requested(
        "仅依据 ChangeMamba 表 XII 回答。", None
    )

    flow_pdf = tmp_path / "FlowMamba.pdf"
    flow_pdf.write_bytes(b"flow-fixture")
    flow_sha = "b" * 64

    class DualScopedAdapter:
        def resolve_explicit_title_scope(self, queries: Any, **_: Any) -> set[str]:
            text = " ".join(queries) if isinstance(queries, list) else str(queries)
            scope: set[str] = set()
            if "ChangeMamba" in text:
                scope.add(paper_sha)
            if "FlowMamba" in text:
                scope.add(flow_sha)
            return scope

        def resolve_pdf_paths(self, values: set[str]) -> list[str]:
            mapping = {paper_sha: pdf, flow_sha: flow_pdf}
            return [str(mapping[value].resolve()) for value in sorted(values)]

        def search(self, _query: str, top_k: int = 8, **_: Any) -> list[dict[str, Any]]:
            del top_k
            return []

    dual_runtime = ResearchAgentRuntime(
        db_path=tmp_path / "dual-runtime.sqlite3",
        workspace=tmp_path / "dual-workspace",
        search_adapter=DualScopedAdapter(),
    )
    dual = dual_runtime.start(
        question="请比较 ChangeMamba 与 FlowMamba 两篇论文的方法与实验。",
        claims=[
            {"statement": "ChangeMamba 的时空模块", "gaps": ["primary_source"]},
            {"statement": "FlowMamba 的 MFA 模块", "gaps": ["primary_source"]},
        ],
        metadata={"allow_external_search": True},
        auto_advance=False,
    )
    by_statement = {
        item["statement"]: item["metadata"] for item in dual["claim_graph"]["claims"]
    }
    assert by_statement["ChangeMamba 的时空模块"]["paper_sha256s"] == [paper_sha]
    assert by_statement["FlowMamba 的 MFA 模块"]["paper_sha256s"] == [flow_sha]
    assert by_statement["ChangeMamba 的时空模块"]["source_scope_mode"] == (
        "claim_explicit_title"
    )
    assert by_statement["FlowMamba 的 MFA 模块"]["source_scope_mode"] == (
        "claim_explicit_title"
    )
    assert by_statement["ChangeMamba 的时空模块"]["pdf_paths"] == [str(pdf.resolve())]
    assert by_statement["FlowMamba 的 MFA 模块"]["pdf_paths"] == [str(flow_pdf.resolve())]
    change_claim = next(
        item for item in dual["claim_graph"]["claims"]
        if item["statement"] == "ChangeMamba 的时空模块"
    )
    rejected = dual_runtime._verify_evidence(
        [
            {
                "evidence_type": "TEXT",
                "content": "FlowMamba introduces an MFA module.",
                "confidence": 0.9,
                "provenance": {"paper_sha256": flow_sha},
            }
        ],
        default_claim_id=change_claim["claim_id"],
        run_id=dual["run"]["run_id"],
    )
    assert rejected[0]["polarity"] == "NEUTRAL"
    assert rejected[0]["provenance"]["verification"]["verifier"] == "runtime-source-scope-v1"
    flow_claim = next(
        item for item in dual["claim_graph"]["claims"]
        if item["statement"] == "FlowMamba 的 MFA 模块"
    )
    evidence_id, _ = EvidenceLedger(dual_runtime.db).add(
        run_id=dual["run"]["run_id"],
        claim_id=change_claim["claim_id"],
        evidence_type="TEXT",
        content="ChangeMamba introduces a spatiotemporal module.",
        polarity=EvidencePolarity.SUPPORTS,
        source_uri=f"paper://{paper_sha}/p1",
        provenance={"paper_sha256": paper_sha},
    )
    dual_runtime.db.link_evidence(
        flow_claim["claim_id"],
        evidence_id,
        EvidencePolarity.NEUTRAL,
    )
    shared = next(
        item for item in dual_runtime.db.list_evidence(dual["run"]["run_id"])
        if item["evidence_id"] == evidence_id
    )
    assert shared["claim_polarities"] == {
        change_claim["claim_id"]: "SUPPORTS",
        flow_claim["claim_id"]: "NEUTRAL",
    }
    assert source_attribution_audit(dual_runtime.db, dual["run"]["run_id"])["status"] == "PASS"
