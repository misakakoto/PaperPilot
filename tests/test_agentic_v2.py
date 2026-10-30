from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.models import Budget, EvidencePolarity
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.sandbox import ExperimentResult
from paperpilot_agentic.verification import LLMEvidenceVerifier


class _StaticAdapter:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def search(self, query: str, top_k: int = 8, **_: Any):
        return self.rows[:top_k]

    def search_many(self, queries: list[str], top_k: int = 8, **_: Any):
        return self.rows[:top_k]


class _StageModel:
    model = "stage-fixture"

    def __init__(self, *, malformed_planner: bool = False, experiment: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.malformed_planner = malformed_planner
        self.experiment = experiment

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        callback = kwargs.get("before_attempt")
        if callable(callback):
            callback()
        joined = "\n".join(item["content"] for item in messages)
        self.calls.append({"messages": messages, **kwargs})
        if "Repair the supplied model output" in joined:
            payload = self._plan_payload()
        elif "Create a research plan" in joined:
            payload = "{bad json" if self.malformed_planner else self._plan_payload()
        elif "strict scientific evidence verifier" in joined:
            user = json.loads(messages[-1]["content"])
            payload = {
                "assessments": [
                    {
                        "index": item["index"],
                        "polarity": "SUPPORTS",
                        "confidence": 0.9,
                        "quote": item["content"][:24],
                        "reason": "The passage directly states the claim.",
                    }
                    for item in user["evidence"]
                ]
            }
        elif "Answer the research question" in joined:
            user = json.loads(messages[-1]["content"])
            evidence_id = user["evidence"][0]["evidence_id"]
            payload = {"answer": f"证据支持该主张 [{evidence_id}]。"}
        elif "Choose the next bounded" in joined:
            payload = {
                "action": "ABSTAIN",
                "queries": [],
                "reason": "No novel query remains.",
                "expected_information_gain": 0,
                "stop_reason": "fixture_stop",
            }
        else:
            payload = {}
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return LLMResponse(
            text=text,
            prompt_tokens=10,
            completion_tokens=20,
            cached_tokens=2,
            reasoning_tokens=3 if kwargs.get("thinking_enabled") else 0,
            finish_reason="stop",
            request_id=f"req-{len(self.calls)}",
            model=self.model,
        )

    def _plan_payload(self) -> dict[str, Any]:
        claim: dict[str, Any] = {
            "statement": "Residual connections improve optimization stability.",
            "importance": 1.0,
            "gaps": ["primary_source"],
            "required_evidence_types": ["TEXT"],
            "candidate_queries": ["residual connection optimization", "skip connection stability"],
        }
        if self.experiment:
            claim["gaps"] = ["independent_experiment"]
            claim["required_evidence_types"] = ["EXPERIMENT"]
            claim["experiment_spec"] = {
                "name": "safe-fixture",
                "hypothesis": "The metric is at least 0.8.",
                "seeds": [7],
                "metrics": ["score"],
                "metric_files": ["metrics.json"],
                "acceptance_criteria": [
                    {"metric": "score", "operator": ">=", "value": 0.8}
                ],
                "resource_limits": {"timeout_seconds": 60, "memory_mb": 512},
            }
        return {
            "objective": "Evaluate residual connections.",
            "constraints": ["不得伪造结果"],
            "output_requirements": ["引用证据"],
            "candidate_queries": ["residual connection optimization"],
            "claims": [claim],
        }


class _ScholarlyFixture:
    def search(self, query: str, top_k: int = 8) -> dict[str, Any]:
        return {
            "query": query,
            "providers": ["fixture"],
            "errors": [],
            "cache_hit": False,
            "items": [
                {
                    "source_id": "fixture:1",
                    "provider": "fixture",
                    "title": "A candidate paper",
                    "abstract": "Discovery metadata only.",
                    "landing_url": "https://example.invalid/paper",
                    "full_text_url": None,
                }
            ][:top_k],
        }

    def fetch_full_text(self, url: str) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError(f"unexpected fetch: {url}")


class _AttestedDockerRunner:
    """Deterministic runner double for orchestration acceptance only.

    Production still uses SandboxExperimentRunner and must obtain a real Docker
    isolation attestation. This double lets the test cover approval, execution,
    evaluation and Claim Graph feedback on hosts without a Docker daemon.
    """

    def __init__(self) -> None:
        self.calls = 0

    def run(self, protocol: Any) -> ExperimentResult:
        self.calls += 1
        assert protocol.policy.backend == "docker"
        assert protocol.policy.network == "none"
        output = Path(protocol.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "fixture-result.json"
        result_path.write_text(
            json.dumps({"score": 0.91, "accepted": True}), encoding="utf-8"
        )
        return ExperimentResult(
            experiment_id="experiment_fixture",
            status="COMPLETED",
            backend="docker",
            exit_code=0,
            timed_out=False,
            started_at="2026-08-29T00:00:00+00:00",
            completed_at="2026-08-29T00:00:01+00:00",
            duration_seconds=1.0,
            protocol_hash=protocol.fingerprint,
            preregistration_path=str(output / "protocol.json"),
            result_path=str(result_path),
            stdout_path=str(output / "stdout.log"),
            stderr_path=str(output / "stderr.log"),
            metrics={"score": 0.91},
            criteria=[
                {
                    "metric": "score",
                    "operator": ">=",
                    "target": 0.8,
                    "observed": 0.91,
                    "passed": True,
                }
            ],
            accepted=True,
            source_hash_before="sha256:fixture",
            source_hash_after="sha256:fixture",
            source_unchanged=True,
            workspace_hash_after="sha256:workspace",
            network_isolated=True,
            isolation_notes=["deterministic test double; not a production Docker attestation"],
            artifacts=[],
        )


def test_stage_policies_usage_ledger_and_grounded_answer(tmp_path: Path) -> None:
    model = _StageModel()
    rows = [
        {
            "content": f"Residual connections improve optimization stability. Evidence {index}.",
            "source_uri": f"paper://fixture/chunk-{index}",
            "confidence": 0.9,
            "paper_sha256": "a" * 64,
            "stable_id": f"chunk-{index}",
            "page": index,
        }
        for index in range(1, 4)
    ]
    result = ResearchAgentRuntime(
        db_path=tmp_path / "runtime.sqlite3",
        workspace=tmp_path,
        search_adapter=_StaticAdapter(rows),
        llm=model,
    ).start(question="残差连接是否改善优化稳定性；不得伪造结果")

    assert result["run"]["status"] == "COMPLETED"
    assert result["run"]["research_outcome"] == "ANSWERED"
    assert result["claim_graph"]["claims"][0]["status"] == "SUPPORTED"
    assert "不得伪造结果" not in result["claim_graph"]["claims"][0]["statement"]
    assert result["final_answer"]["citation_ids"]
    assert result["run"]["used_llm_calls"] == 3
    assert result["run"]["used_tokens"] == 90
    assert [item["stage"] for item in result["llm_calls"]] == [
        "planner",
        "verifier",
        "synthesizer",
    ]
    assert [item["max_output_tokens"] for item in result["llm_calls"]] == [6000, 4000, 8000]
    # Structured synthesis stays non-thinking so the provider returns the
    # claim/evidence ownership contract instead of a truncated reasoning trace.
    assert [bool(item["thinking_enabled"]) for item in result["llm_calls"]] == [
        False,
        False,
        False,
    ]


def test_invalid_planner_json_repairs_once_without_punctuation_claims(tmp_path: Path) -> None:
    model = _StageModel(malformed_planner=True)
    runtime = ResearchAgentRuntime(
        db_path=tmp_path / "runtime.sqlite3",
        workspace=tmp_path,
        search_adapter=_StaticAdapter([]),
        llm=model,
    )
    result = runtime.start(
        question="BDA 算法的流程是什么；不得伪造结果",
        budget=Budget(max_llm_calls=4),
        auto_advance=False,
    )

    assert result["run"]["status"] == "RUNNING"
    assert result["run"]["used_llm_calls"] == 2
    assert [item["stage"] for item in result["llm_calls"]] == ["planner", "json_repair"]
    assert len(result["claim_graph"]["claims"]) == 1
    assert result["claim_graph"]["claims"][0]["statement"] == "Residual connections improve optimization stability."
    assert result["run"]["metadata"]["research_plan"]["constraints"] == ["不得伪造结果"]
    assert runtime._research_llm_call_limit(
        result["run"]["run_id"],
        Budget(max_llm_calls=4),
    ) == 3
    assert runtime.llm_runtime is not None
    runtime.llm_runtime.complete(
        result["run"]["run_id"],
        [{"role": "user", "content": "Return a bounded controller decision."}],
        policy="controller",
    )
    runtime.llm_runtime.complete(
        result["run"]["run_id"],
        [{"role": "user", "content": "Return a final synthesis object."}],
        policy="synthesizer",
    )
    assert runtime.db.get_run(result["run"]["run_id"])["used_llm_calls"] == 4


def test_strict_named_paper_scope_rejects_cross_paper_evidence(tmp_path: Path) -> None:
    target_sha = "a" * 64
    other_sha = "b" * 64

    class ScopedAdapter(_StaticAdapter):
        def resolve_explicit_title_scope(self, queries: Any, **_: Any) -> set[str]:
            values = [queries] if isinstance(queries, str) else list(queries)
            return {target_sha} if any("changemamba" in str(item).casefold() for item in values) else set()

    rows = [
        {
            "content": f"Residual connections improve optimization stability. Evidence {index}.",
            "source_uri": f"paper://{target_sha}/target-{index}",
            "confidence": 0.9,
            "paper_sha256": target_sha,
            "stable_id": f"target-{index}",
            "title": "ChangeMamba",
            "page": index,
        }
        for index in range(1, 4)
    ]
    rows.append(
        {
            "content": "Residual connections improve optimization stability in FlowMamba.",
            "source_uri": f"paper://{other_sha}/other-1",
            "confidence": 0.9,
            "paper_sha256": other_sha,
            "stable_id": "other-1",
            "title": "FlowMamba",
            "page": 10,
        }
    )
    result = ResearchAgentRuntime(
        db_path=tmp_path / "runtime.sqlite3",
        workspace=tmp_path,
        search_adapter=ScopedAdapter(rows),
        llm=_StageModel(),
    ).start(question="请仅依据论文原文说明 ChangeMamba 的机制。")

    assert result["run"]["metadata"]["source_scope_mode"] == "strict_explicit_title"
    assert result["run"]["metadata"]["paper_sha256s"] == [target_sha]
    outside = [
        item
        for item in result["evidence"]
        if (item["provenance"].get("paper_sha256") or "") == other_sha
    ]
    assert outside and all(item["polarity"] == "NEUTRAL" for item in outside)
    assert all(
        item["provenance"]["verification"]["verifier"] == "runtime-source-scope-v1"
        for item in outside
    )


def test_planner_double_failure_is_degraded_single_claim(tmp_path: Path) -> None:
    class Broken(_StageModel):
        def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
            callback = kwargs.get("before_attempt")
            if callable(callback):
                callback()
            self.calls.append({"messages": messages, **kwargs})
            return LLMResponse("{still bad", prompt_tokens=3, completion_tokens=4, finish_reason="stop")

    result = ResearchAgentRuntime(
        db_path=tmp_path / "runtime.sqlite3",
        workspace=tmp_path,
        search_adapter=_StaticAdapter([]),
        llm=Broken(),
    ).start(question="BDA 算法的流程是什么；不得伪造结果", auto_advance=False)

    claim = result["claim_graph"]["claims"][0]
    assert claim["statement"] == "BDA 算法的流程是什么"
    assert len(result["claim_graph"]["claims"]) == 1
    assert result["run"]["metadata"]["degraded_reasons"][0]["stage"] == "claim_planner"


def test_react_queries_do_not_repeat_after_empty_evidence(tmp_path: Path) -> None:
    result = ResearchAgentRuntime(
        db_path=tmp_path / "runtime.sqlite3",
        workspace=tmp_path,
        search_adapter=_StaticAdapter([]),
    ).start(
        question="Does method A improve dataset B?",
        claims=[
            {
                "statement": "Method A improves dataset B.",
                "gaps": ["primary_source"],
                "metadata": {
                    "candidate_queries": ["method A dataset B", "method A ablation"],
                    "max_gap_attempts": 2,
                },
            }
        ],
    )
    searches = [item for item in result["tool_calls"] if item["tool_name"] == "paper.search"]
    assert len(searches) == 2
    first = {item.casefold() for item in searches[0]["args"]["queries"]}
    second = {item.casefold() for item in searches[1]["args"]["queries"]}
    assert first.isdisjoint(second)
    assert result["run"]["research_outcome"] == "UNRESOLVED"


def test_scholarly_metadata_cannot_close_a_claim(tmp_path: Path) -> None:
    result = ResearchAgentRuntime(
        db_path=tmp_path / "runtime.sqlite3",
        workspace=tmp_path,
        search_adapter=_StaticAdapter([]),
        scholarly_adapter=_ScholarlyFixture(),
    ).start(
        question="Does method A improve dataset B?",
        claims=[
            {
                "statement": "Method A improves dataset B.",
                "gaps": ["primary_source"],
                "metadata": {
                    "candidate_queries": ["method A dataset B", "method A evaluation"],
                    "max_gap_attempts": 3,
                    "allow_external_search": True,
                },
            }
        ],
    )
    assert any(item["tool_name"] == "scholar.search" for item in result["tool_calls"]), [
        item["tool_name"] for item in result["tool_calls"]
    ]
    metadata = [item for item in result["evidence"] if item["evidence_type"] == "METADATA"]
    assert metadata and all(item["polarity"] == "NEUTRAL" for item in metadata)
    assert result["claim_graph"]["claims"][0]["status"] == "UNRESOLVED"


def test_natural_language_experiment_reaches_docker_approval(tmp_path: Path) -> None:
    source = tmp_path / "experiment"
    source.mkdir()
    (source / "experiment.py").write_text(
        "import json\nfrom pathlib import Path\nPath('metrics.json').write_text(json.dumps({'score': 0.9}))\n",
        encoding="utf-8",
    )
    result = ResearchAgentRuntime(
        db_path=tmp_path / "runtime.sqlite3",
        workspace=tmp_path / "workspace",
        search_adapter=_StaticAdapter([]),
        llm=_StageModel(experiment=True),
    ).start(
        question="请验证残差连接假设并运行预注册实验",
        metadata={
            "experiment_source_dir": str(source),
            "experiment_command": [sys.executable, "experiment.py"],
        },
    )

    assert result["run"]["status"] == "WAITING_APPROVAL"
    assert [item["tool_name"] for item in result["tool_calls"]] == [
        "experiment.plan",
        "experiment.run",
    ]
    request = result["pending_approvals"][0]["request"]
    assert request["arguments"]["policy"]["backend"] == "docker"
    assert request["arguments"]["policy"]["network"] == "none"
    assert request["arguments"]["policy"]["allow_local_fallback"] is False


def test_natural_language_experiment_approval_execution_and_evaluation_loop(
    tmp_path: Path,
) -> None:
    source = tmp_path / "experiment"
    source.mkdir()
    (source / "experiment.py").write_text("print('fixture')\n", encoding="utf-8")
    runner = _AttestedDockerRunner()
    runtime = ResearchAgentRuntime(
        db_path=tmp_path / "runtime.sqlite3",
        workspace=tmp_path / "workspace",
        search_adapter=_StaticAdapter([]),
        llm=_StageModel(experiment=True),
        experiment_runner=runner,
    )
    waiting = runtime.start(
        question="请针对残差连接的薄弱样本提出可证伪假设，预注册并运行对照实验",
        metadata={
            "experiment_source_dir": str(source),
            "experiment_command": [sys.executable, "experiment.py"],
        },
    )

    assert waiting["run"]["status"] == "WAITING_APPROVAL"
    approval_id = waiting["pending_approvals"][0]["approval_id"]
    runtime.approve(approval_id, actor="acceptance-test", comment="bounded fixture")
    completed = runtime.resume(waiting["run"]["run_id"])

    assert runner.calls == 1
    assert completed["run"]["status"] == "COMPLETED"
    assert completed["run"]["research_outcome"] == "ANSWERED"
    assert completed["claim_graph"]["claims"][0]["status"] == "SUPPORTED"
    assert [item["tool_name"] for item in completed["tool_calls"]] == [
        "experiment.plan",
        "experiment.run",
        "experiment.evaluate",
    ]
    evaluated = [
        item
        for item in completed["evidence"]
        if item["evidence_type"] == "EXPERIMENT"
    ]
    assert len(evaluated) == 1
    assert evaluated[0]["polarity"] == "SUPPORTS"
    assert evaluated[0]["provenance"]["network_isolated"] is True
    assert evaluated[0]["provenance"]["source_unchanged"] is True


def test_reported_usage_can_exhaust_the_hard_token_budget(tmp_path: Path) -> None:
    class Expensive(_StageModel):
        def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
            response = super().complete(messages, **kwargs)
            return LLMResponse(
                response.text,
                prompt_tokens=95_000,
                completion_tokens=6_000,
                finish_reason="stop",
                model=self.model,
            )

    result = ResearchAgentRuntime(
        db_path=tmp_path / "runtime.sqlite3",
        workspace=tmp_path,
        search_adapter=_StaticAdapter([]),
        llm=Expensive(),
    ).start(question="Evaluate residual connections", budget=Budget(max_tokens=100_000))
    assert result["run"]["status"] == "BUDGET_EXHAUSTED"
    assert result["run"]["used_tokens"] == 101_000
    assert result["claim_graph"]["claims"][0]["status"] == "UNRESOLVED"


def test_simple_question_is_not_overplanned_or_turned_into_experiment(tmp_path: Path) -> None:
    class OverPlanner(_StageModel):
        def _plan_payload(self) -> dict[str, Any]:
            claims = [
                {
                    "statement": f"MambaBCD factual claim {index}.",
                    "gaps": (
                        ["independent_experiment", "implementation", "formula"]
                        if index == 2
                        else ["primary_source"]
                    ),
                    "candidate_queries": [f"MambaBCD query {index}"],
                }
                for index in range(1, 7)
            ]
            claims.insert(
                2,
                {
                    "statement": "论文可能未提供运行速度分析。",
                    "gaps": ["counter_evidence"],
                    "candidate_queries": ["MambaBCD speed"],
                },
            )
            return {
                "objective": "Explain MambaBCD.",
                "constraints": ["不得伪造结果"],
                "output_requirements": ["说明证据不足部分"],
                "candidate_queries": ["MambaBCD architecture"],
                "claims": claims,
            }

    result = ResearchAgentRuntime(
        db_path=tmp_path / "runtime.sqlite3",
        workspace=tmp_path,
        search_adapter=_StaticAdapter([]),
        llm=OverPlanner(),
    ).start(
        question="MambaBCD 如何建模双时相信息？请说明证据不足部分。",
        auto_advance=False,
    )
    claims = result["claim_graph"]["claims"]
    assert len(claims) == 4
    assert all("可能未提供" not in item["statement"] for item in claims)
    assert all(
        not ({"independent_experiment", "experiment_evaluation", "implementation", "formula"} & set(item["gaps"]))
        for item in claims
    )
    assert all(item["required_evidence_types"] == ["TEXT"] for item in claims)


def test_pdf_normalized_quote_is_grounded_but_paraphrase_is_rejected() -> None:
    class QuoteModel:
        def __init__(self, quote: str) -> None:
            self.quote = quote

        def complete(self, messages: list[dict[str, str]], **_: Any) -> LLMResponse:
            return LLMResponse(
                json.dumps(
                    {
                        "assessments": [
                            {
                                "index": 0,
                                "polarity": "SUPPORTS",
                                "confidence": 0.9,
                                "quote": self.quote,
                                "reason": "The passage directly supports the claim.",
                            }
                        ]
                    }
                )
            )

    evidence = {
        "evidence_type": "TEXT",
        "content": "The archi-\ntectures achieve the highest\nand second highest F1 scores.",
        "source_uri": "paper://fixture/page/11",
        "confidence": 0.9,
    }
    grounded = LLMEvidenceVerifier(
        QuoteModel("The architectures achieve the highest and second highest F1 scores.")
    ).assess("The method achieves strong F1 scores.", evidence)
    assert grounded.polarity == EvidencePolarity.SUPPORTS
    assert grounded.quote_match == "pdf_dehyphenated"

    ligature_evidence = {
        "evidence_type": "TEXT",
        "content": "The module estimates the o ﬀ set and uses di ﬀ erential optical ﬂow alignment.",
        "source_uri": "paper://fixture/page/4",
        "confidence": 0.9,
    }
    ligature_grounded = LLMEvidenceVerifier(
        QuoteModel(
            "The module estimates the offset and uses differential optical flow alignment."
        )
    ).assess("The module uses optical-flow alignment.", ligature_evidence)
    assert ligature_grounded.polarity == EvidencePolarity.SUPPORTS
    assert ligature_grounded.quote_match == "pdf_compatibility_normalized"

    rejected = LLMEvidenceVerifier(QuoteModel("The method is universally superior."))
    rejected_result = rejected.assess("The method achieves strong F1 scores.", evidence)
    assert rejected_result.polarity == EvidencePolarity.NEUTRAL
    assert "quote was not grounded" in rejected_result.reason


def test_verifier_caps_one_tool_at_six_and_reserves_final_answer_call(tmp_path: Path) -> None:
    model = _StageModel()
    rows = [
        {
            "content": f"Residual connections improve optimization stability. Evidence {index}.",
            "source_uri": f"paper://{'b' * 64}/chunk-{index}",
            "confidence": 0.9,
            "paper_sha256": "b" * 64,
            "stable_id": f"chunk-{index}",
            "page": index,
        }
        for index in range(1, 9)
    ]
    result = ResearchAgentRuntime(
        db_path=tmp_path / "runtime.sqlite3",
        workspace=tmp_path,
        search_adapter=_StaticAdapter(rows),
        llm=model,
    ).start(
        question="残差连接是否改善优化稳定性？",
        budget=Budget(max_llm_calls=4),
    )
    assert result["run"]["status"] == "COMPLETED"
    assert [item["stage"] for item in result["llm_calls"]] == [
        "planner",
        "verifier",
        "synthesizer",
    ]
    modes = [
        (item["provenance"].get("verification") or {}).get("verifier")
        for item in result["evidence"]
    ]
    assert modes.count("llm-entailment-v1") == 6
    assert modes.count("runtime-verifier-budget-v2") == 2
    assert all(
        item["provenance"].get("fragment_hash")
        and item["provenance"].get("evidence_anchor")
        for item in result["evidence"]
    )
