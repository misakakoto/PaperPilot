"""Regression tests for the slim, standalone PaperPilot checkout."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from paperpilot_agentic.config import (
    RuntimeSettings,
    read_api_key_file,
    read_dashscope_api_key_file,
)
from paperpilot_retrieval.retrieval import FastEmbedder
from paperpilot_agentic.web_app import create_app
from paperpilot_retrieval import PaperKnowledgeStore
from paperpilot_retrieval.models import PaperChunk, PaperDocument, PaperPage
from scripts.run_interview_demo_suite import (
    _four_call_follow_up_budget,
    _request_optional_run,
    _same_question,
    _scenario_issues,
    _validate_provider_connectivity,
    _validate_retrieval_preflight,
)


def test_interview_follow_up_budget_funds_two_fresh_paper_verifications() -> None:
    budget = _four_call_follow_up_budget(1_800)

    research_calls = (
        budget["max_llm_calls"]
        - budget["synthesis_reserve_calls"]
        - budget["repair_reserve_calls"]
    )
    assert research_calls == 3  # planner + one verifier for each named paper
    assert budget["max_repair_calls"] == 0


def test_provider_key_bundle_is_resolved_without_copying_secrets(tmp_path: Path) -> None:
    bundle = tmp_path / "keys.txt"
    bundle.write_text("deepseek-secret\nqwen-secret\n", encoding="utf-8")

    assert read_api_key_file(bundle) == "deepseek-secret"
    assert read_dashscope_api_key_file(bundle) == "qwen-secret"


def test_default_text_provider_is_qwen_with_external_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "keys.txt"
    bundle.write_text("deepseek-secret\nqwen: qwen-secret\n", encoding="utf-8")
    with (
        patch("paperpilot_agentic.config.load_dotenv", return_value=False),
        patch.dict(
            "os.environ",
            {
                "PAPERPILOT_API_KEY_FILE": str(bundle),
                "DEEPSEEK_API_KEY": "",
                "DASHSCOPE_API_KEY": "",
                "QWEN_API_KEY": "",
            },
            clear=False,
        ),
    ):
        os.environ.pop("PAPERPILOT_LLM_PROVIDER", None)
        os.environ.pop("PAPERPILOT_LLM_MODEL", None)
        os.environ.pop("QWEN_MODEL", None)
        settings = RuntimeSettings.from_env(require_api_key=True)

    assert settings.provider == "qwen"
    assert settings.model == "qwen3.5-plus-2026-02-15"
    assert settings.fallback_models == ()
    assert settings.api_key == "qwen-secret"


def test_local_bm25_index_round_trip(tmp_path: Path) -> None:
    store = PaperKnowledgeStore(tmp_path / "papers.sqlite3", namespace="test")
    document = PaperDocument(
        id="P1",
        file_name="change.pdf",
        sha256="a" * 64,
        title="ChangeMamba: Remote Sensing Change Detection With Spatiotemporal State Space Model",
        page_count=1,
        processed_pages=1,
        pages=[PaperPage(number=1, text="Selective state space change detection")],
    )
    chunk = PaperChunk(
        id="P1:p1:c1",
        paper_id="P1",
        page_start=1,
        page_end=1,
        section="Method",
        text="The selective state space module improves remote sensing change detection.",
        token_count_estimate=12,
    )

    synced = store.sync_documents([document], [chunk])
    hits = store.search_chunks("state space change detection", top_k=3, mode="bm25")

    assert synced.papers_added == 1
    assert synced.chunks_written == 1
    assert hits and hits[0].title == document.title
    assert hits[0].paper_sha256 == document.sha256
    assert hits[0].stable_id.startswith(f"C-{document.sha256[:16]}-")

    flow_document = PaperDocument(
        id="P2",
        file_name="flow.pdf",
        sha256="b" * 64,
        title="FlowMamba: Building Damage Assessment via Optics Flow-Based State Space Model",
        page_count=1,
        processed_pages=1,
        pages=[PaperPage(number=1, text="Mamba flow alignment")],
    )
    flow_chunk = PaperChunk(
        id="P2:p1:c1",
        paper_id="P2",
        page_start=1,
        page_end=1,
        section="Method",
        text="FlowMamba aligns bi-temporal features.",
        token_count_estimate=8,
    )
    store.sync_documents([flow_document], [flow_chunk])
    constrained = {document.sha256, flow_document.sha256}

    assert store.resolve_explicit_title_scope(
        "The ablation results in Table VII of the FlowMamba paper.",
        constrained,
    ) == {flow_document.sha256}
    assert store.resolve_explicit_title_scope(
        "The ablation results in Table XII of the ChangeMamba paper.",
        constrained,
    ) == {document.sha256}


def test_web_health_and_bootstrap_work_without_an_api_key(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    settings = RuntimeSettings(
        provider="qwen",
        api_key=None,
        base_url="https://example.invalid/compatible-mode/v1",
        model="qwen-test",
        thinking_enabled=False,
        temperature=0.0,
        max_tokens=100,
        timeout_seconds=1.0,
        workspace=workspace,
        runtime_db=workspace / "runtime.sqlite3",
        memory_db=None,
        memory_namespace="test",
        paper_roots=(workspace / "papers",),
    )

    with patch(
        "paperpilot_agentic.web_app.probe_provider_connectivity",
        return_value={
            "reachable": False,
            "probe": "tcp_connect_v1",
            "error_type": "fixture",
            "latency_ms": 0.1,
        },
    ):
        with TestClient(create_app(settings)) as client:
            assert client.get("/health").json() == {"status": "ok"}
            response = client.get("/api/bootstrap")
            assert response.status_code == 200
            assert response.json()["llm_configured"] is False
            assert Path(response.json()["source_root"]).resolve() == Path(__file__).resolve().parents[1]
            assert "launcher_id" in response.json()
            connectivity = client.get("/api/provider-connectivity").json()
            assert connectivity["reachable"] is False
            assert connectivity["billable_requests"] == 0
            assert client.get("/").status_code == 200
            assert client.post('/api/agentic/turns',json={'request_id':'csrf-attempt','question':'test'},
                headers={'Origin':'https://external.invalid'}).status_code==403
            assert client.get('/health',headers={'Host':'rebind.invalid'}).status_code==400
            import mimetypes
            mimetypes.init()
            from unittest.mock import patch as patch_mapping
            with patch_mapping.dict(mimetypes.types_map,{'.js':'text/plain'}):
                asset=client.get('/assets/workbench.js')
            assert asset.headers['content-type'].startswith('application/javascript')


def test_web_ui_groups_runs_as_conversations_and_defaults_to_follow_up() -> None:
    static_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "paperpilot_agentic"
        / "web_static"
    )
    html = (static_root / "index.html").read_text(encoding="utf-8")
    javascript = (static_root / "workbench.js").read_text(encoding="utf-8")

    assert "研究会话" in html
    assert 'id="thread"' in html
    assert 'id="follow-up-run"' not in html
    assert '/conversations?limit=30' in javascript
    assert '/runs?limit=200' not in javascript
    assert 'id="engine-select"' not in html
    assert 'id="stop"' in html and 'id="resume"' in html
    assert 'parent_run_id:turn.run_id' in javascript
    assert '/notebook?offset=' in javascript
    assert '未知' in javascript
    assert '不承诺任意模型自主训练' in html
    assert '/assets/vendor/katex/' in html and 'cdn.' not in html


def test_interview_suite_rejects_stale_question_and_semantic_answer_failures() -> None:
    scenarios_path = Path(__file__).resolve().parents[1] / "demo" / "interview_scenarios.json"
    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))["scenarios"]
    single = next(item for item in scenarios if item["kind"] == "single_paper")
    multi = next(item for item in scenarios if item["kind"] == "multi_paper")

    assert not _same_question(
        "只说明 ChangeMamba 的整体 STSS 结构。",
        single["question"],
    )
    bad_multi = {
        "run": {"status": "COMPLETED", "research_outcome": "ANSWERED"},
        "final_answer": {
            "answer": (
                "Sequential Cross Parallel MFA 92.35 82.83 80.53 "
                "Multi-scale Feature Alignment"
            ),
            "answer_quality": {"passed": True},
        },
        "claim_graph": {"claims": []},
    }

    issues = _scenario_issues(bad_multi, multi)

    assert (
        "missing_required_substring_group:80.14+80.26|80.43+80.53"
        in issues
    )
    assert "forbidden_substring:Multi-scale Feature Alignment" in issues
    assert "too_few_citations:0<4" in issues
    assert "too_few_supported_claims:0<4" in issues
    assert any(item.startswith("required_citation_source_missing:") for item in issues)


def test_interview_suite_treats_missing_historical_single_as_cache_miss(monkeypatch) -> None:
    def missing(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("HTTP 404 from PaperPilot: historical run not found")

    monkeypatch.setattr("scripts.run_interview_demo_suite._request_json", missing)

    assert _request_optional_run("http://127.0.0.1:8014", "run_missing") is None


def test_interview_suite_checks_cited_paper_scope_and_followup_lineage() -> None:
    scenarios_path = Path(__file__).resolve().parents[1] / "demo" / "interview_scenarios.json"
    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))["scenarios"]
    follow_up = next(item for item in scenarios if item["kind"] == "follow_up")
    change_sha, flow_sha = follow_up["answer_checks"]["required_citation_source_sha256s"]
    snapshot = {
        "run": {
            "status": "COMPLETED",
            "research_outcome": "ANSWERED",
            "metadata": {"parent_run_id": "run_parent", "conversation_id": "conversation_1"},
        },
        "final_answer": {
            "answer": (
                "论文事实：ChangeMamba 与 FlowMamba MFA。比较性推断：后者更直接。"
                "结论边界：不能证明精度更高。"
            ),
            "answer_quality": {"passed": True},
            "citation_ids": ["ev_change", "ev_flow"],
        },
        "claim_graph": {
            "claims": [{"status": "SUPPORTED"}, {"status": "SUPPORTED"}],
        },
        "evidence": [
            {"evidence_id": "ev_change", "provenance": {"document_sha256": change_sha}},
            {"evidence_id": "ev_flow", "provenance": {"document_sha256": flow_sha}},
        ],
    }

    assert _scenario_issues(snapshot, follow_up) == []
    snapshot["evidence"][1]["provenance"]["document_sha256"] = "f" * 64
    issues = _scenario_issues(snapshot, follow_up)
    assert f"required_citation_source_missing:{flow_sha}" in issues
    assert f"citation_source_out_of_scope:{'f' * 64}" in issues


def test_interview_suite_accepts_any_complete_controlled_ablation_pair() -> None:
    scenario = {
        "answer_checks": {
            "required_one_of_substring_groups": [
                ["80.14", "80.26"],
                ["80.43", "80.53"],
            ]
        }
    }
    snapshot = {
        "run": {},
        "final_answer": {"answer": "MFA changes F1 from 80.14 to 80.26."},
        "claim_graph": {"claims": []},
        "evidence": [],
    }
    assert _scenario_issues(snapshot, scenario) == []

    snapshot["final_answer"]["answer"] = "The table contains 80.14 and 80.53."
    issues = _scenario_issues(snapshot, scenario)
    assert issues == [
        "missing_required_substring_group:80.14+80.26|80.43+80.53"
    ]


def test_interview_suite_accepts_semantic_alternatives_for_mechanism_wording() -> None:
    scenario = {
        "answer_checks": {
            "required_any_substring_groups": [
                ["交叉排序", "交错排列", "交替组合"],
                ["通道维度", "通道维"],
            ]
        }
    }
    snapshot = {
        "run": {},
        "final_answer": {"answer": "Cross 将 token 交错排列；Parallel 沿通道维拼接。"},
        "claim_graph": {"claims": []},
        "evidence": [],
    }

    assert _scenario_issues(snapshot, scenario) == []

    snapshot["final_answer"]["answer"] = "Cross 进行交互；Parallel 使用并行分支。"
    assert _scenario_issues(snapshot, scenario) == [
        "missing_required_alternative:交叉排序|交错排列|交替组合",
        "missing_required_alternative:通道维度|通道维",
    ]


def test_interview_suite_requires_zero_network_retrieval_preflight(tmp_path) -> None:
    scenario_path = tmp_path / "scenarios.json"
    scenario_path.write_text("{}", encoding="utf-8")
    valid = {
        "kind": "zero_network_interview_demo_retrieval_preflight",
        "scenario_file": str(scenario_path),
        "probe_count": 1,
        "passed": True,
        "checks": [{"passed": True}],
        "network_calls": 0,
        "paid_requests": 0,
        "dense_index": {"configured": True, "complete": True},
    }

    _validate_retrieval_preflight(valid, scenario_path=scenario_path)
    invalid = {**valid, "network_calls": 1}
    with pytest.raises(RuntimeError, match="not zero-network"):
        _validate_retrieval_preflight(invalid, scenario_path=scenario_path)


def test_interview_suite_rejects_unreachable_provider_before_agent_run() -> None:
    _validate_provider_connectivity(
        {
            "reachable": True,
            "probe": "tcp_connect_v1",
            "billable_requests": 0,
        }
    )
    with pytest.raises(RuntimeError, match="cannot reach"):
        _validate_provider_connectivity(
            {
                "reachable": False,
                "probe": "tcp_connect_v1",
                "billable_requests": 0,
                "error_type": "ConnectionRefusedError",
            }
        )


def test_fastembed_cache_identity_includes_runtime_implementation() -> None:
    embedder = object.__new__(FastEmbedder)
    embedder._name = "org/model"
    embedder._model_sha256 = "a" * 64
    embedder._tokenizer_sha256 = "b" * 64
    embedder._fastembed_version = "0.8.0"
    embedder._backend_identity = "fastembed.text.pooled_embedding.PooledEmbedding"

    identity = embedder._cache_identity()

    assert identity is not None
    assert "@fastembed:0.8.0" in identity
    assert "@backend:fastembed.text.pooled_embedding.PooledEmbedding" in identity
    embedder._fastembed_version = "0.5.1"
    assert embedder._cache_identity() != identity


def test_production_source_has_no_legacy_package_dependency() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    offenders = []
    for path in source_root.rglob("*.py"):
        if "deep_research_agent" in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(source_root).as_posix())
    assert offenders == []
