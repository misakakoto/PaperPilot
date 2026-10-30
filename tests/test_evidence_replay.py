from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from paperpilot_agentic.evidence_replay import (
    redacted_replay_report,
    replay_paid_evidence,
)


def _qwen_result(image: Path, label: str, value: str) -> dict:
    return {
        "description": f"{label} reports one row.\nVisible transcription: Model | Score\nBaseline | {value}",
        "confidence": 0.49,
        "verified": False,
        "metadata": {
            "finish_reason": "stop",
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "image_readable": True,
            "evidence_relevant": False,
            "reported_confidence": 0.95,
            "visible_values": [value],
            "table_rows": [
                {
                    "row_label": "Baseline",
                    "checked_columns": [],
                    "cells": {"Score": value},
                }
            ],
            "uncertain_fields": ["The metric definition is not provided in the crop."],
        },
    }


def test_zero_network_replay_binds_sources_and_closes_numeric_claims(
    tmp_path: Path,
) -> None:
    sha_a = "a" * 64
    sha_b = "b" * 64
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_a.write_bytes(b"crop-a")
    image_b.write_bytes(b"crop-b")
    base = {
        "run": {
            "run_id": "run_paid",
            "status": "BUDGET_EXHAUSTED",
            "research_outcome": "PARTIAL",
            "question": "Compare paper A and paper B tables.",
        },
        "claim_graph": {
            "claims": [
                {
                    "claim_id": "claim_a",
                    "statement": "Paper A reports a numeric result.",
                    "status": "UNRESOLVED",
                    "confidence": 0.0,
                    "gaps": ["numeric_results"],
                    "metadata": {"paper_sha256s": [sha_a]},
                },
                {
                    "claim_id": "claim_b",
                    "statement": "Paper B reports a numeric result.",
                    "status": "UNRESOLVED",
                    "confidence": 0.0,
                    "gaps": ["numeric_results"],
                    "metadata": {"paper_sha256s": [sha_b]},
                },
            ]
        },
        "evidence": [],
    }
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    stale_source = {
        "evidence_id": "paid_a",
        "claim_id": "claim_a",
        "claim_ids": ["claim_a"],
        "claim_polarities": {"claim_a": "SUPPORTS"},
        "evidence_type": "TABLE",
        "polarity": "SUPPORTS",
        "content": _qwen_result(image_a, "TABLE XII", "999.9")["description"],
        "confidence": 0.49,
        "page": 12,
        "source_uri": f"paper://{sha_a}/layout/table-xii",
        "artifact_path": str(image_a),
        "provenance": {
            "document_sha256": sha_a,
            "vision_verified": False,
            "vision_metadata": _qwen_result(image_a, "TABLE XII", "81.2")["metadata"],
        },
    }
    snapshot_source = dict(stale_source)
    snapshot_source["content"] = _qwen_result(image_a, "TABLE XII", "81.2")[
        "description"
    ]
    snapshot = {"evidence": [snapshot_source]}
    base["evidence"] = [stale_source]
    base_path.write_text(json.dumps(base), encoding="utf-8")
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    record_path = tmp_path / "record.json"
    record_path.write_text(
        json.dumps({"result": _qwen_result(image_b, "TABLE VII", "80.53")}),
        encoding="utf-8",
    )
    manifest = {
        "base_snapshot": str(base_path),
        "imports": [
            {
                "kind": "snapshot_qwen_evidence",
                "snapshot": str(snapshot_path),
                "evidence_id": "paid_a",
                "target_claim_id": "claim_a",
                "prompt": "Read TABLE XII.",
                "caption": "TABLE XII",
            },
            {
                "kind": "qwen_record",
                "record": str(record_path),
                "image": str(image_b),
                "target_claim_id": "claim_b",
                "paper_sha256": sha_b,
                "page": 7,
                "prompt": "Read TABLE VII.",
                "caption": "TABLE VII",
            },
        ],
    }

    result = replay_paid_evidence(manifest)
    redacted = redacted_replay_report(result)

    assert result["network_calls"] == 0
    assert result["paid_requests"] == 0
    assert result["replay_research_outcome"] == "ANSWERED"
    assert result["source_scope_violations"] == 0
    assert all(item["status"] == "SUPPORTED" for item in result["claims"])
    assert "81.2" in result["answer"]["answer"]
    assert "999.9" not in result["answer"]["answer"]
    assert all(item["evidence_id"] != "paid_a" for item in result["evidence"])
    assert "80.53" in result["answer"]["answer"]
    assert len(result["answer"]["citation_ids"]) == 2
    assert redacted["private_evidence_persisted"] is False
    assert "answer" not in redacted

    manifest["imports"][1]["paper_sha256"] = sha_a
    with pytest.raises(ValueError, match="source scope"):
        replay_paid_evidence(manifest)


def test_zero_network_replay_recomputes_cross_language_semantic_claim(
    tmp_path: Path,
) -> None:
    paper_sha = "c" * 64
    quote = "The proposed MFA module aligns pre- and post-disaster features."
    claim_id = "claim_mechanism"
    base = {
        "run": {
            "run_id": "run_paid_semantic",
            "status": "BUDGET_EXHAUSTED",
            "research_outcome": "PARTIAL",
            "question": "FlowMamba 的 MFA 模块解决什么问题？",
        },
        "claim_graph": {
            "claims": [
                {
                    "claim_id": claim_id,
                    "statement": "FlowMamba 的 MFA 模块用于对齐灾前灾后特征。",
                    "status": "PARTIAL",
                    "confidence": 0.36,
                    "gaps": [],
                    "required_evidence_types": ["TEXT"],
                    "metadata": {
                        "paper_sha256s": [paper_sha],
                        "source_scope_mode": "strict_explicit_title",
                    },
                }
            ]
        },
        "evidence": [
            {
                "evidence_id": "paid_semantic_evidence",
                "claim_id": claim_id,
                "claim_ids": [claim_id],
                "claim_polarities": {claim_id: "SUPPORTS"},
                "evidence_type": "TEXT",
                "polarity": "SUPPORTS",
                "content": quote,
                "source_uri": f"paper://{paper_sha}/p1",
                "confidence": 0.9,
                "provenance": {
                    "gap": "primary_source",
                    "paper_sha256": paper_sha,
                    "verification": {
                        "polarity": "SUPPORTS",
                        "confidence": 0.9,
                        "relevant": True,
                        "verifier": "llm-entailment-v1",
                        "claim_coverage": 0.0,
                        "degraded": False,
                        "grounded_quote": quote,
                        "quote_match": "exact",
                        "semantic_entailment": True,
                    },
                },
            }
        ],
    }
    base_path = tmp_path / "semantic-base.json"
    base_path.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")

    result = replay_paid_evidence({"base_snapshot": str(base_path), "imports": []})

    assert result["network_calls"] == 0
    assert result["paid_requests"] == 0
    assert result["replay_research_outcome"] == "ANSWERED"
    assert result["claims"][0]["status"] == "SUPPORTED"
    assert result["claims"][0]["confidence"] == pytest.approx(0.9)
    assert result["claims"][0]["metadata"][
        "semantic_adjudication_evidence_ids"
    ] == ["paid_semantic_evidence"]
