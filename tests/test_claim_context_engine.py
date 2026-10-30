from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from paperpilot_agentic.claim_graph import ClaimGraph, EvidenceGapScheduler
from paperpilot_agentic.context_engine import (
    ContextPolicy,
    ResearchContextEngine,
    audit_final_answer_quality,
    evidence_excerpt,
    estimate_context_tokens,
    mechanism_fidelity_issues,
    mechanism_operation_requirements,
)
from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.evidence import EvidenceLedger
from paperpilot_agentic.models import Budget, ClaimStatus, EvidencePolarity, EvidenceType
from paperpilot_agentic.orchestrator import ResearchAgentRuntime


class _EmptySearch:
    def search(self, query: str, top_k: int = 8, **_: Any) -> list[dict[str, Any]]:
        return []

    def search_many(
        self, queries: list[str], top_k: int = 8, **_: Any
    ) -> list[dict[str, Any]]:
        return []


def _seed_context_run(tmp_path: Path) -> tuple[RuntimeDatabase, str, list[str], list[str]]:
    db = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    run_id = db.create_run(
        "请比较六项机制；不得把 0.125 写成 0.15，并说明反证。",
        Budget(),
        {
            "research_plan": {
                "objective": "比较六项机制并找出证据边界",
                "constraints": ["不得伪造结果", "关键阈值必须是 0.125"],
                "output_requirements": ["逐项回答", "指出反证"],
            }
        },
    )
    graph = ClaimGraph(db)
    ledger = EvidenceLedger(db)
    claim_ids: list[str] = []
    evidence_ids: list[str] = []
    for index in range(6):
        claim = graph.add_claim(
            run_id=run_id,
            statement=f"机制 {index} 在目标数据集上有效。",
            gaps=["primary_source", "numeric_results"],
            metadata={
                "candidate_queries": [f"mechanism {index} result", f"mechanism {index} ablation"],
                "paper_sha256s": [f"paper-{index}-a", f"paper-{index}-b"],
            },
        )
        claim_ids.append(claim.claim_id)
        if index != 0:
            continue
        for item_index in range(12):
            evidence_id, _ = ledger.add(
                run_id=run_id,
                claim_id=claim.claim_id,
                evidence_type=(
                    EvidenceType.TABLE if item_index % 3 == 0 else EvidenceType.TEXT
                ),
                polarity=(
                    EvidencePolarity.CONTRADICTS
                    if item_index == 11
                    else EvidencePolarity.SUPPORTS
                ),
                content=(
                    f"Table row {item_index}: score=0.{item_index:03d}; "
                    + "long context token " * 180
                ),
                page=10 + item_index,
                confidence=0.9,
                provenance={
                    "paper_sha256": "paper-0-a" if item_index < 10 else "paper-0-b",
                    "verification": {
                        "verifier": "llm-entailment-v1",
                        "relevant": True,
                        "degraded": False,
                        "grounded_quote": f"score=0.{item_index:03d}",
                    },
                },
            )
            evidence_ids.append(evidence_id)
    return db, run_id, claim_ids, evidence_ids


def test_context_snapshot_roundtrip_compacts_without_losing_pinned_state(
    tmp_path: Path,
) -> None:
    db, run_id, claim_ids, evidence_ids = _seed_context_run(tmp_path)
    policy = ContextPolicy(
        controller_soft_tokens=512,
        controller_hard_tokens=3_200,
        synthesis_soft_tokens=1_024,
        synthesis_hard_tokens=8_000,
    )
    engine = ResearchContextEngine(db, policy)

    assembly = engine.build_controller_context(
        run_id,
        claim_id=claim_ids[0],
        gap="numeric_results",
        budget={"remaining_tokens": 20_000, "remaining_tool_calls": 5},
    )

    assert assembly.compressed is True
    assert assembly.estimated_tokens_after <= policy.controller_hard_tokens
    assert assembly.payload["target"]["claim_id"] == claim_ids[0]
    assert "0.125" in assembly.payload["pinned"]["constraints"][1]
    assert assembly.payload["pinned"]["summary_is_not_evidence"] is True
    assert set(assembly.lineage["evidence_ids"]).issubset(set(evidence_ids))
    restored = engine.restore(assembly.snapshot_id)
    assert restored["payload"] == assembly.payload
    assert restored["lineage"] == assembly.lineage
    assert db.list_context_snapshots(run_id, stage="controller")[0]["snapshot_id"] == assembly.snapshot_id
    with db.connect() as conn:
        version = conn.execute(
            "SELECT value FROM runtime_meta WHERE key='schema_version'"
        ).fetchone()["value"]
    assert version == "5"


def test_structured_excerpt_preserves_numeric_tail_and_grounded_quote() -> None:
    excerpt = evidence_excerpt(
        {
            "evidence_type": "TABLE",
            "content": "Table XII\n" + "irrelevant words " * 200 + "OA=91.7 F1=88.2",
            "provenance": {
                "verification": {"grounded_quote": "The ablation improves OA to 91.7."}
            },
        },
        max_chars=180,
    )
    assert "91.7" in excerpt
    assert "88.2" in excerpt
    assert "grounded" not in excerpt.casefold()


def test_scheduler_exposes_claim_coverage_and_paper_diversity_utility(
    tmp_path: Path,
) -> None:
    db = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    run_id = db.create_run("compare claims", Budget())
    graph = ClaimGraph(db)
    scheduler = EvidenceGapScheduler(db, graph)
    ledger = EvidenceLedger(db)
    partly_covered = graph.add_claim(
        run_id=run_id,
        statement="partly covered",
        gaps=["numeric_results"],
        confidence=0.2,
        metadata={
            "initial_gaps": ["primary_source", "numeric_results"],
            "paper_sha256s": ["paper-a", "paper-b"],
        },
    )
    uncovered = graph.add_claim(
        run_id=run_id,
        statement="uncovered",
        gaps=["primary_source", "numeric_results"],
        confidence=0.2,
        metadata={
            "paper_sha256s": ["paper-c", "paper-d"],
        },
    )
    ledger.add(
        run_id=run_id,
        claim_id=partly_covered.claim_id,
        evidence_type="TEXT",
        content="primary source",
        confidence=0.9,
        provenance={
            "paper_sha256": "paper-a",
            "verification": {
                "verifier": "llm-entailment-v1",
                "relevant": True,
                "degraded": False,
            },
        },
    )

    partly_actions = scheduler._for_claim(partly_covered, Budget())
    uncovered_actions = scheduler._for_claim(uncovered, Budget())

    assert all(item.utility_version == "claim-coverage-cost-v1" for item in uncovered_actions)
    assert max(item.coverage_gain for item in uncovered_actions) > max(
        item.coverage_gain for item in partly_actions
    )
    assert max(item.diversity_gain for item in partly_actions) > 0


def test_zero_api_run_persists_context_and_passes_answer_quality(tmp_path: Path) -> None:
    result = ResearchAgentRuntime(
        db_path=tmp_path / "runtime.sqlite3",
        workspace=tmp_path / "workspace",
        search_adapter=_EmptySearch(),
    ).start(
        question="方法 A 是否改善数据集 B？不得伪造结论。",
        claims=[
            {
                "statement": "方法 A 改善数据集 B。",
                "gaps": ["primary_source"],
                "metadata": {
                    "candidate_queries": ["method A dataset B", "method A ablation"],
                    "max_gap_attempts": 1,
                },
            }
        ],
    )

    stages = [item["stage"] for item in result["context_snapshots"]]
    assert "controller" in stages
    assert "synthesizer" in stages
    assert result["final_answer"]["answer_quality"]["passed"] is True
    assert result["final_answer"]["answer_quality"]["claim_coverage"] == 1.0
    assert result["run"]["used_llm_calls"] == 0


def test_quality_gate_replaces_incomplete_multiclaim_answer(tmp_path: Path) -> None:
    class IncompleteSynthesizer:
        def synthesize(
            self,
            *,
            question: str,
            claims: Sequence[Mapping[str, Any]],
            evidence: Sequence[Mapping[str, Any]],
            run_id: str | None = None,
        ) -> Mapping[str, Any]:
            del question, evidence, run_id
            return {
                "answer": "Only the first Claim was answered.",
                "citation_ids": [],
                "covered_claim_ids": [claims[0]["claim_id"]],
                "section_evidence_map": {},
            }

    result = ResearchAgentRuntime(
        db_path=tmp_path / "runtime.sqlite3",
        workspace=tmp_path / "workspace",
        search_adapter=_EmptySearch(),
        answer_synthesizer=IncompleteSynthesizer(),
    ).start(
        question="compare A and B",
        claims=[
            {
                "statement": "Claim A",
                "gaps": ["primary_source"],
                "metadata": {"max_gap_attempts": 0},
            },
            {
                "statement": "Claim B",
                "gaps": ["primary_source"],
                "metadata": {"max_gap_attempts": 0},
            },
        ],
    )

    final_answer = result["final_answer"]
    assert final_answer["mode"] == "deterministic-grounded-evidence-digest-v2"
    assert final_answer["answer_quality"]["passed"] is True
    assert final_answer["answer_quality"]["covered_claim_count"] == 2
    assert "missing_claim_sections" in final_answer["synthesis_error"]
    assert result["run"]["research_outcome"] == "DEGRADED"


def test_answer_quality_audit_rejects_cross_claim_citation() -> None:
    claims = [
        {"claim_id": "claim_a", "status": ClaimStatus.SUPPORTED.value},
        {"claim_id": "claim_b", "status": ClaimStatus.SUPPORTED.value},
    ]
    evidence = [
        {"evidence_id": "evidence_a", "claim_id": "claim_a"},
        {"evidence_id": "evidence_b", "claim_id": "claim_b"},
    ]
    quality = audit_final_answer_quality(
        {
            "answer": "bad mapping",
            "citation_ids": ["evidence_a", "evidence_b"],
            "covered_claim_ids": ["claim_a", "claim_b"],
            "section_evidence_map": {
                "claim_a": ["evidence_b"],
                "claim_b": ["evidence_a"],
            },
        },
        claims=claims,
        evidence=evidence,
    )
    assert quality["passed"] is False
    assert "cross_claim_citation" in quality["issues"]


def test_answer_quality_audit_rejects_ungrounded_numeric_literal() -> None:
    quality = audit_final_answer_quality(
        {
            "answer": "The score is 91.7.",
            "citation_ids": ["evidence_a"],
            "covered_claim_ids": ["claim_a"],
            "section_evidence_map": {"claim_a": ["evidence_a"]},
            "section_text_map": {"claim_a": "The score is 91.7."},
        },
        claims=[{"claim_id": "claim_a", "status": ClaimStatus.SUPPORTED.value}],
        evidence=[
            {
                "evidence_id": "evidence_a",
                "claim_id": "claim_a",
                "content": "The reported score is 90.1.",
            }
        ],
    )
    assert quality["passed"] is False
    assert quality["numeric_violations"] == {"claim_a": ["91.7"]}
    assert "ungrounded_numeric_literals" in quality["issues"]


def test_answer_quality_audit_accepts_grounded_percent_suffix() -> None:
    answer = "实验表明，该配置在论文报告数据集上的 F1 指标达到 82.83%，与表格结果一致。"
    quality = audit_final_answer_quality(
        {
            "answer": answer,
            "citation_ids": ["evidence_a"],
            "covered_claim_ids": ["claim_a"],
            "section_evidence_map": {"claim_a": ["evidence_a"]},
            "section_text_map": {"claim_a": answer},
        },
        claims=[{"claim_id": "claim_a", "status": ClaimStatus.SUPPORTED.value}],
        evidence=[
            {
                "evidence_id": "evidence_a",
                "claim_id": "claim_a",
                "polarity": "SUPPORTS",
                "content": "F1 (%) | 82.83",
            }
        ],
    )

    assert quality["passed"] is True
    assert quality["numeric_violations"] == {}


def test_answer_quality_audit_rejects_claim_echo_without_synthesis() -> None:
    quality = audit_final_answer_quality(
        {
            "answer": "ChangeMamba uses a state-space model.",
            "citation_ids": ["evidence_a"],
            "covered_claim_ids": ["claim_a"],
            "section_evidence_map": {"claim_a": ["evidence_a"]},
            "section_text_map": {
                "claim_a": "ChangeMamba uses a state-space model."
            },
        },
        claims=[
            {
                "claim_id": "claim_a",
                "statement": "ChangeMamba uses a state-space model.",
                "status": ClaimStatus.SUPPORTED.value,
            }
        ],
        evidence=[
            {
                "evidence_id": "evidence_a",
                "claim_id": "claim_a",
                "content": "The STSS block scans temporal and spatial dimensions selectively.",
            }
        ],
    )
    assert quality["passed"] is False
    assert quality["claim_echo_violations"] == ["claim_a"]
    assert "claim_echo_without_synthesis" in quality["issues"]


def test_answer_quality_audit_requires_available_result_values() -> None:
    quality = audit_final_answer_quality(
        {
            "answer": "The ablation supports the module.",
            "citation_ids": ["evidence_a"],
            "covered_claim_ids": ["claim_a"],
            "section_evidence_map": {"claim_a": ["evidence_a"]},
            "section_text_map": {"claim_a": "The ablation supports the module."},
        },
        claims=[
            {
                "claim_id": "claim_a",
                "statement": "Report the ablation result.",
                "status": ClaimStatus.SUPPORTED.value,
                "gaps": ["numeric_results"],
            }
        ],
        evidence=[
            {
                "evidence_id": "evidence_a",
                "claim_id": "claim_a",
                "evidence_type": "TABLE",
                "content": "The full model reaches F1 91.7 compared with 90.1.",
            }
        ],
    )
    assert quality["passed"] is False
    assert quality["missing_result_value_claim_ids"] == ["claim_a"]
    assert "grounded_result_values_omitted" in quality["issues"]


def test_mechanism_quality_gate_rejects_generic_branch_fusion_paraphrase() -> None:
    claims = [
        {
            "claim_id": "claim_parallel",
            "statement": (
                "The primary source contains a concrete definition of the input or "
                "representation transformation used by the Parallel mechanism in ChangeMamba."
            ),
            "status": ClaimStatus.SUPPORTED.value,
            "gaps": ["primary_source"],
        }
    ]
    evidence = [
        {
            "evidence_id": "evidence_parallel",
            "claim_id": "claim_parallel",
            "claim_ids": ["claim_parallel"],
            "claim_polarities": {"claim_parallel": "SUPPORTS"},
            "content": (
                "Parallel modeling is done by concatenating the tokens of the two "
                "temporal phases in the channel dimension and then performing joint modeling."
            ),
        }
    ]
    final_answer = {
        "answer": "Parallel 通过并行分支处理双时相特征，随后进行融合。",
        "citation_ids": ["evidence_parallel"],
        "covered_claim_ids": ["claim_parallel"],
        "section_evidence_map": {"claim_parallel": ["evidence_parallel"]},
        "section_text_map": {
            "claim_parallel": "Parallel 通过并行分支处理双时相特征，随后进行融合。"
        },
    }

    omissions = mechanism_fidelity_issues(
        final_answer,
        claims=claims,
        evidence=evidence,
        question="Parallel 机制如何组织双时相特征？",
    )
    quality = audit_final_answer_quality(
        final_answer,
        claims=claims,
        evidence=evidence,
        question="Parallel 机制如何组织双时相特征？",
    )

    assert omissions == {
        "claim_parallel": [
            "channel_dimension",
            "concatenation",
            "joint_modeling",
            "unsupported:branch_processing",
            "unsupported:fusion",
        ]
    }
    assert quality["passed"] is False
    assert "mechanism_operation_omitted" in quality["issues"]
    assert "mechanism_operation_unsupported" in quality["issues"]


def test_mechanism_quality_gate_accepts_grounded_chinese_translation() -> None:
    claims = [
        {
            "claim_id": "claim_parallel",
            "statement": "The Parallel mechanism in ChangeMamba must be established.",
            "status": ClaimStatus.SUPPORTED.value,
        }
    ]
    evidence = [
        {
            "evidence_id": "evidence_parallel",
            "claim_id": "claim_parallel",
            "claim_ids": ["claim_parallel"],
            "claim_polarities": {"claim_parallel": "SUPPORTS"},
            "content": (
                "Parallel modeling concatenates both temporal tokens in the channel "
                "dimension before performing joint modeling."
            ),
        }
    ]
    answer = "Parallel 沿通道维度拼接两个时相的 token，再进行联合建模。"
    final_answer = {
        "answer": answer,
        "citation_ids": ["evidence_parallel"],
        "covered_claim_ids": ["claim_parallel"],
        "section_evidence_map": {"claim_parallel": ["evidence_parallel"]},
        "section_text_map": {"claim_parallel": answer},
    }

    assert mechanism_fidelity_issues(
        final_answer,
        claims=claims,
        evidence=evidence,
        question="Parallel 机制如何组织双时相特征？",
    ) == {}


def test_mechanism_quality_gate_rejects_unsupported_independent_processing() -> None:
    claims = [
        {
            "claim_id": "claim_parallel",
            "statement": "The Parallel mechanism in ChangeMamba must be established.",
            "status": ClaimStatus.SUPPORTED.value,
        }
    ]
    evidence = [
        {
            "evidence_id": "evidence_parallel",
            "claim_id": "claim_parallel",
            "claim_ids": ["claim_parallel"],
            "claim_polarities": {"claim_parallel": "SUPPORTS"},
            "content": (
                "Parallel modeling concatenates both temporal tokens in the channel "
                "dimension before performing joint modeling."
            ),
        }
    ]
    answer_text = (
        "Parallel机制沿通道维独立处理两个时相表示，再拼接并联合建模。"
    )
    final_answer = {
        "answer": answer_text,
        "citation_ids": ["evidence_parallel"],
        "covered_claim_ids": ["claim_parallel"],
        "section_evidence_map": {"claim_parallel": ["evidence_parallel"]},
        "section_text_map": {"claim_parallel": answer_text},
    }

    fidelity = mechanism_fidelity_issues(
        final_answer,
        claims=claims,
        evidence=evidence,
        question="Parallel 机制如何组织双时相特征？",
    )
    quality = audit_final_answer_quality(
        final_answer,
        claims=claims,
        evidence=evidence,
        question="Parallel 机制如何组织双时相特征？",
    )

    assert fidelity == {"claim_parallel": ["unsupported:independent_processing"]}
    assert "mechanism_operation_unsupported" in quality["issues"]
    assert quality["mechanism_operation_unsupported"] == fidelity


def test_mechanism_requirements_are_scoped_to_each_named_evidence_span() -> None:
    claims = [
        {
            "claim_id": f"claim_{name.casefold()}",
            "statement": (
                "The primary source contains a concrete definition of the input or "
                f"representation transformation used by the {name} mechanism."
            ),
            "status": ClaimStatus.SUPPORTED.value,
        }
        for name in ("Sequential", "Cross", "Parallel")
    ]
    shared_id = "evidence_change_mamba"
    evidence = [
        {
            "evidence_id": shared_id,
            "content": (
                "Sequential modeling arranges tokens in temporal order. "
                "Cross modeling uses cross-ordering of temporal tokens. "
                "Parallel modeling concatenates temporal tokens in the channel "
                "dimension and then performs joint modeling."
            ),
        }
    ]

    requirements = mechanism_operation_requirements(
        claims=claims,
        evidence=evidence,
        question="Sequential、Cross、Parallel 分别如何组织双时相特征？",
        evidence_ids_by_claim={
            str(claim["claim_id"]): [shared_id] for claim in claims
        },
    )

    assert requirements == {
        "claim_sequential": {"Sequential": ["temporal_order"]},
        "claim_cross": {"Cross": ["cross_ordering"]},
        "claim_parallel": {
            "Parallel": [
                "channel_dimension",
                "concatenation",
                "joint_modeling",
            ]
        },
    }


def test_mechanism_quality_gate_handles_grouped_variant_claim() -> None:
    claim_id = "claim_grouped"
    claims = [
        {
            "claim_id": claim_id,
            "statement": (
                "Sequential, Cross, and Parallel token organization strategies "
                "must be established from ChangeMamba."
            ),
            "status": ClaimStatus.SUPPORTED.value,
        }
    ]
    evidence = [
        {
            "evidence_id": "evidence_grouped",
            "claim_ids": [claim_id],
            "claim_polarities": {claim_id: "SUPPORTS"},
            "content": (
                "Sequential modeling arranges tokens in temporal order. "
                "Cross modeling uses cross-ordering of temporal tokens. "
                "Parallel modeling concatenates temporal tokens in the channel "
                "dimension and then performs joint modeling."
            ),
        }
    ]
    weak_answer = {
        "section_evidence_map": {claim_id: ["evidence_grouped"]},
        "section_text_map": {
            claim_id: (
                "Sequential 按时间顺序排列，Cross 将时相 token 交错排列，"
                "Parallel 采用并行分支融合。"
            )
        },
    }

    assert mechanism_fidelity_issues(
        weak_answer,
        claims=claims,
        evidence=evidence,
        question=(
            "说明 Sequential、Cross、Parallel 三种机制如何组织双时相特征。"
        ),
    ) == {
        claim_id: [
            "Parallel:channel_dimension",
            "Parallel:concatenation",
            "Parallel:joint_modeling",
            "Parallel:unsupported:branch_processing",
            "Parallel:unsupported:fusion",
        ]
    }

    strong_answer = {
        **weak_answer,
        "section_text_map": {
            claim_id: (
                "Sequential机制将两个时相 token 按时间顺序连接；Cross机制将对应位置 "
                "token 交错排列；Parallel机制沿通道维拼接两时相 token 后联合建模。"
            )
        },
    }
    assert mechanism_fidelity_issues(
        strong_answer,
        claims=claims,
        evidence=evidence,
        question=(
            "说明 Sequential、Cross、Parallel 三种机制如何组织双时相特征。"
        ),
    ) == {}


def test_mechanism_quality_gate_allows_locally_negated_unsupported_terms() -> None:
    claim_id = "claim_grouped"
    claims = [
        {
            "claim_id": claim_id,
            "statement": "Sequential, Cross, and Parallel token organization in ChangeMamba.",
            "status": ClaimStatus.SUPPORTED.value,
        }
    ]
    evidence = [
        {
            "evidence_id": "evidence_grouped",
            "claim_ids": [claim_id],
            "claim_polarities": {claim_id: "SUPPORTS"},
            "content": (
                "Sequential modeling arranges tokens in temporal order. "
                "Cross modeling uses cross-ordering of temporal tokens. "
                "Parallel modeling concatenates temporal tokens in the channel "
                "dimension and then performs joint modeling."
            ),
        }
    ]
    corrective_answer = {
        "section_evidence_map": {claim_id: ["evidence_grouped"]},
        "section_text_map": {
            claim_id: (
                "Sequential并非线性处理，而是按时间顺序排列两个时相 token；"
                "Cross将时相 token 交错排列；Parallel不是通过独立分支融合，"
                "而是沿通道维拼接两时相 token 后联合建模。"
            )
        },
    }

    assert mechanism_fidelity_issues(
        corrective_answer,
        claims=claims,
        evidence=evidence,
        question="说明 Sequential、Cross、Parallel 三种机制如何组织双时相特征。",
    ) == {}


def test_mechanism_quality_gate_rejects_swapped_operations_despite_all_keywords() -> None:
    claim_id = "claim_grouped"
    claims = [
        {
            "claim_id": claim_id,
            "statement": "Sequential, Cross, and Parallel token organization in ChangeMamba.",
            "status": ClaimStatus.SUPPORTED.value,
        }
    ]
    evidence = [
        {
            "evidence_id": "evidence_grouped",
            "claim_ids": [claim_id],
            "claim_polarities": {claim_id: "SUPPORTS"},
            "content": (
                "Sequential modeling arranges tokens in temporal order. "
                "Cross modeling uses cross-ordering of temporal tokens. "
                "Parallel modeling concatenates temporal tokens in the channel "
                "dimension and then performs joint modeling."
            ),
        }
    ]
    swapped = {
        "section_evidence_map": {claim_id: ["evidence_grouped"]},
        "section_text_map": {
            claim_id: (
                "Sequential 沿通道维拼接并联合建模；Cross 将 token 交错排列；"
                "Parallel 按时间顺序连接。"
            )
        },
    }

    issues = mechanism_fidelity_issues(
        swapped,
        claims=claims,
        evidence=evidence,
        question="说明 Sequential、Cross、Parallel 三种机制如何组织双时相特征。",
    )

    assert issues == {
        claim_id: [
            "Sequential:temporal_order",
            "Parallel:channel_dimension",
            "Parallel:concatenation",
            "Parallel:joint_modeling",
        ]
    }


def test_claim_aware_compaction_beats_flat_prefix_on_late_claims(tmp_path: Path) -> None:
    db, run_id, claim_ids, _ = _seed_context_run(tmp_path)
    engine = ResearchContextEngine(
        db,
        ContextPolicy(
            controller_soft_tokens=512,
            controller_hard_tokens=3_200,
            synthesis_soft_tokens=1_024,
            synthesis_hard_tokens=8_000,
        ),
    )
    assembly = engine.build_controller_context(
        run_id,
        claim_id=claim_ids[-1],
        gap="primary_source",
        budget={},
    )
    raw_log = "\n".join("early tool output " * 80 for _ in range(20)) + "\n" + claim_ids[-1]
    flat_prefix = raw_log[:2_000]
    compact_text = str(assembly.payload)

    assert claim_ids[-1] not in flat_prefix
    assert claim_ids[-1] in compact_text
    assert "0.125" in compact_text
    assert estimate_context_tokens(assembly.payload) <= 3_200
