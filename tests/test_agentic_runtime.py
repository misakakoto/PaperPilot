# ruff: noqa: E402

from __future__ import annotations

import json
import hashlib
import subprocess
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from paperpilot_agentic.intelligence import (
    DeterministicAnswerSynthesizer,
    LLMAnswerSynthesizer,
)
from paperpilot_agentic.layout import (
    _caption_context_bbox,
    fitz,
    parse_pdf_layout,
    search_layout_objects,
)
from paperpilot_agentic.models import Budget, EvidencePolarity, ExperimentProtocol, RunStatus
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.sandbox import SandboxExperimentRunner


class _StaticSearchAdapter:
    def __init__(self, items: list[dict]):
        self.items = items

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        return self.items[:top_k]


class _BrokenPlanner:
    def plan(self, question: str, metadata: dict | None = None) -> list[dict]:
        raise ValueError("invalid planner JSON")


class _CountingModel:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[dict[str, str]], **kwargs: object):
        self.calls += 1
        raise AssertionError("model must not synthesize without verified evidence")


class _AnswerCaptureModel:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def complete(self, messages: list[dict[str, str]], **kwargs: object):
        self.messages = messages
        return SimpleNamespace(
            text=json.dumps({"answer": "Grounded result [evidence_1]"})
        )


class _CapturingPlanner:
    def __init__(self) -> None:
        self.metadata: dict | None = None

    def plan(self, question: str, metadata: dict | None = None) -> list[dict]:
        self.metadata = dict(metadata or {})
        return [{"statement": question, "gaps": ["primary_source"]}]


class _FixtureVisionAdapter:
    def __init__(self) -> None:
        self.calls = 0
        self.reaudits = 0

    def analyze(self, image_path: str, prompt: str, metadata: dict | None = None) -> dict:
        self.calls += 1
        return {
            "description": "The visible table reports mIoU 68.50 and Recall 80.39.",
            "confidence": 0.9,
            "verified": True,
            "metadata": {
                "fixture": True,
                "image_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
            },
        }

    def reaudit(
        self,
        original_result: dict,
        image_path: str,
        *,
        prompt: str,
        metadata: dict | None = None,
    ) -> dict:
        del prompt, metadata
        self.reaudits += 1
        expected = original_result["metadata"]["image_sha256"]
        if hashlib.sha256(Path(image_path).read_bytes()).hexdigest() != expected:
            raise ValueError("image mismatch")
        return original_result


class _FailingVisionAdapter:
    model = "qwen-fixture-failure"
    max_output_tokens = 4_000

    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, image_path: str, prompt: str, metadata: dict | None = None) -> dict:
        del image_path, prompt, metadata
        self.calls += 1
        error = TypeError("Qwen vision response must be a JSON object")
        error.vision_usage = {'prompt_tokens': 120, 'completion_tokens': 45,
                              'finish_reason': 'stop'}
        raise error


class DurableApprovalExperimentTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, list[dict]]:
        workspace = root / "workspace"
        source = root / "experiment_source"
        workspace.mkdir()
        source.mkdir()
        script = source / "experiment.py"
        script.write_text(
            "import json\n"
            "from pathlib import Path\n"
            "Path('metrics.json').write_text(json.dumps({'recall_at_10': 0.8037, 'rerank_reduction': 0.431}))\n",
            encoding="utf-8",
        )
        claims = [
            {
                "statement": "Dynamic reranking preserves recall while reducing Cross-Encoder calls.",
                "gaps": ["independent_experiment"],
                "required_evidence_types": ["EXPERIMENT"],
                "metadata": {
                    "experiment": {
                        "name": "dynamic-reranking-reproduction",
                        "command": [sys.executable, "experiment.py"],
                        "source_dir": str(source),
                        "metric_files": ["metrics.json"],
                        "hypothesis": "Recall remains above 0.80 and reranking calls fall by at least 0.40.",
                        "acceptance_criteria": [
                            {"metric": "recall_at_10", "operator": ">=", "value": 0.80},
                            {"metric": "rerank_reduction", "operator": ">=", "value": 0.40},
                        ],
                        "policy": {
                            "backend": "local",
                            "network": "host",
                            "allow_local_fallback": True,
                            "timeout_seconds": 30,
                            "cpu_seconds": 30,
                            "memory_mb": 1024,
                        },
                    }
                },
            }
        ]
        return workspace, source, claims

    def test_pause_approve_cross_instance_resume_and_idempotency(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace, source, claims = self._fixture(root)
            db = root / "runtime.sqlite3"
            before = (source / "experiment.py").read_bytes()

            first = ResearchAgentRuntime(db_path=db, workspace=workspace)
            waiting = first.start(question="Reproduce the dynamic-reranking claim", claims=claims)
            self.assertEqual(waiting["run"]["status"], RunStatus.WAITING_APPROVAL.value)
            self.assertEqual(len(waiting["pending_approvals"]), 1)
            run_id = waiting["run"]["run_id"]
            approval_id = waiting["pending_approvals"][0]["approval_id"]

            # A second object simulates a separate process making the human decision.
            approver = ResearchAgentRuntime(db_path=db, workspace=workspace)
            approver.approve(approval_id, actor="unit-test", comment="approved")

            # A third object resumes from durable SQLite state.
            resumed_runtime = ResearchAgentRuntime(db_path=db, workspace=workspace)
            completed = resumed_runtime.resume(run_id)
            self.assertEqual(completed["run"]["status"], RunStatus.COMPLETED.value)
            claim = completed["claim_graph"]["claims"][0]
            # Explicit local fallback exercises durability but is not admissible
            # scientific evidence because it cannot prove network/host isolation.
            self.assertEqual(claim["status"], "UNRESOLVED")
            self.assertEqual(claim["gaps"], ["executed_experiment_result"])
            self.assertEqual(completed["run"]["research_outcome"], "UNRESOLVED")
            self.assertEqual(len(completed["tool_calls"]), 3)
            self.assertEqual(completed["tool_calls"][0]["attempt"], 1)
            self.assertEqual(len(completed["evidence"]), 2)
            self.assertEqual(completed["evidence"][1]["evidence_type"], "EXPERIMENT")
            self.assertEqual(completed["evidence"][1]["polarity"], "NEUTRAL")
            self.assertTrue(completed["evidence"][0]["provenance"]["source_unchanged"])
            self.assertEqual(before, (source / "experiment.py").read_bytes())
            self.assertTrue(any(span["name"] == "tool.experiment.run" for span in completed["trace"]))
            tool_span = next(
                span for span in completed["trace"] if span["name"] == "tool.experiment.run"
            )
            parent_span = next(
                span
                for span in completed["trace"]
                if span["span_id"] == tool_span["parent_span_id"]
            )
            self.assertEqual(parent_span["name"], "agent.run")
            self.assertEqual(parent_span["trace_id"], tool_span["trace_id"])
            self.assertTrue(all(span["status"] == "OK" for span in completed["trace"]))

            # Re-resume is a no-op: the side effect is not executed twice.
            again = ResearchAgentRuntime(db_path=db, workspace=workspace).resume(run_id)
            self.assertEqual(len(again["tool_calls"]), 3)
            self.assertEqual(again["tool_calls"][0]["attempt"], 1)
            self.assertEqual(len(again["evidence"]), 2)

    def test_cli_cross_process_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace, _, claims = self._fixture(root)
            db = root / "runtime.sqlite3"
            claims_path = root / "claims.json"
            claims_path.write_text(json.dumps(claims), encoding="utf-8")
            launcher = ROOT / "scripts" / "paperpilot_agentic.py"

            def invoke(*args: str) -> dict:
                command = [
                    sys.executable,
                    str(launcher),
                    "--db",
                    str(db),
                    "--workspace",
                    str(workspace),
                    *args,
                ]
                # This is an offline recovery test. The developer's .env must
                # not connect the subprocess to their live library or providers.
                environment = {key: value for key, value in os.environ.items()
                               if not key.startswith(('PAPERPILOT_', 'DASHSCOPE_', 'QWEN_', 'DEEPSEEK_'))}
                environment['PYTHON_DOTENV_DISABLED'] = '1'
                result = subprocess.run(command, cwd=ROOT, env=environment,
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, msg=f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
                return json.loads(result.stdout)

            waiting = invoke("start", "--question", "cross process", "--claims-json", str(claims_path))
            run_id = waiting["run"]["run_id"]
            approval_id = waiting["pending_approvals"][0]["approval_id"]
            invoke("approve", approval_id, "--actor", "cli-test")
            completed = invoke("resume", run_id)
            self.assertEqual(completed["run"]["status"], "COMPLETED")
            self.assertEqual(completed["claim_graph"]["claims"][0]["status"], "UNRESOLVED")

    def test_optional_opentelemetry_export_matches_durable_trace(self) -> None:
        try:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )
        except ImportError:
            self.skipTest("OpenTelemetry SDK is an optional dependency")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            exporter = InMemorySpanExporter()
            provider = TracerProvider()
            provider.add_span_processor(SimpleSpanProcessor(exporter))
            runtime = ResearchAgentRuntime(
                db_path=root / "runtime.sqlite3",
                workspace=root,
                search_adapter=_StaticSearchAdapter(
                    [
                        {
                            "content": "Residual connections improve gradient flow.",
                            "source_uri": "paper://fixture/chunk-1",
                            "confidence": 0.9,
                        }
                    ]
                ),
            )
            runtime.tracer._otel_tracer = provider.get_tracer("paperpilot.test")

            result = runtime.start(
                question="Residual connections improve gradient flow."
            )
            provider.force_flush()

            exported_names = {span.name for span in exporter.get_finished_spans()}
            durable_names = {span["name"] for span in result["trace"]}
            self.assertIn("agent.run", exported_names)
            self.assertIn("tool.paper.search", exported_names)
            self.assertTrue(exported_names.issubset(durable_names))


class EvidenceVerificationTest(unittest.TestCase):
    def test_claim_dependency_blocks_downstream_action(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = ResearchAgentRuntime(
                db_path=root / "runtime.sqlite3",
                workspace=root,
                search_adapter=_StaticSearchAdapter([]),
            )
            initial = runtime.start(
                question="Resolve prerequisites before the downstream claim",
                claims=[
                    {"statement": "Downstream conclusion", "gaps": ["primary_source"]},
                    {"statement": "Required premise", "gaps": ["primary_source"]},
                ],
                edges=[{"source": 0, "target": 1, "relation": "depends_on"}],
                auto_advance=False,
            )
            claims = initial["claim_graph"]["claims"]
            action = runtime.scheduler.select_next(
                initial["run"]["run_id"],
                budget=Budget(),
            )

            self.assertIsNotNone(action)
            self.assertEqual(action.claim_id, claims[1]["claim_id"])

    def test_local_paths_are_hidden_from_planner_but_retained_for_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            planner = _CapturingPlanner()
            result = ResearchAgentRuntime(
                db_path=root / "runtime.sqlite3",
                workspace=root,
                search_adapter=_StaticSearchAdapter([]),
                claim_planner=planner,
            ).start(
                question="Inspect the selected paper",
                metadata={
                    "pdf_paths": [str(root / "private.pdf")],
                    "paper_names": ["paper.pdf"],
                },
            )

            self.assertNotIn("pdf_paths", planner.metadata or {})
            self.assertEqual(
                result["claim_graph"]["claims"][0]["metadata"]["pdf_paths"],
                [str(root / "private.pdf")],
            )

            answer_model = _AnswerCaptureModel()
            synthesized = LLMAnswerSynthesizer(answer_model).synthesize(
                question="Inspect the selected paper",
                claims=[
                    {
                        "claim_id": "claim_1",
                        "statement": "A grounded claim",
                        "status": "SUPPORTED",
                        "metadata": {"pdf_paths": [str(root / "private.pdf")]},
                    }
                ],
                evidence=[
                    {
                        "evidence_id": "evidence_1",
                        "claim_id": "claim_1",
                        "polarity": "SUPPORTS",
                        "evidence_type": "TEXT",
                        "source_uri": str(root / "private.pdf"),
                        "content": "A grounded claim",
                    }
                ],
            )
            self.assertEqual(synthesized["mode"], "llm-grounded-synthesis-v2")
            self.assertNotIn(str(root), json.dumps(answer_model.messages))

    def test_planner_failure_falls_back_and_ungrounded_answer_skips_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = _CountingModel()
            result = ResearchAgentRuntime(
                db_path=root / "runtime.sqlite3",
                workspace=root,
                search_adapter=_StaticSearchAdapter([]),
                claim_planner=_BrokenPlanner(),
                answer_synthesizer=LLMAnswerSynthesizer(model),
            ).start(question="Does the method improve robustness?")

            self.assertEqual(result["run"]["status"], "COMPLETED")
            self.assertIn("claim_planner_fallback", result["run"]["metadata"])
            self.assertEqual(result["run"]["research_outcome"], "DEGRADED")
            self.assertEqual(
                result["final_answer"]["mode"],
                "deterministic-grounded-evidence-digest-v2",
            )
            self.assertEqual(result["final_answer"]["citation_ids"], [])
            self.assertEqual(model.calls, 0)

    def test_deterministic_fallback_keeps_verified_table_values_visible(self) -> None:
        answer = DeterministicAnswerSynthesizer().synthesize(
            question="Which ablation supports the design?",
            claims=[
                {
                    "claim_id": "claim_1",
                    "statement": "Table XII reports the component ablation.",
                    "status": "PARTIAL",
                }
            ],
            evidence=[
                {
                    "evidence_id": "evidence_1",
                    "claim_id": "claim_1",
                    "polarity": "SUPPORTS",
                    "evidence_type": "TABLE",
                    "page": 16,
                    "confidence": 0.9,
                    "content": (
                        "Visible transcription:\n"
                        "Baseline | F1 81.11\nAll mechanisms | F1 82.83"
                    ),
                    "provenance": {},
                }
            ],
        )
        self.assertEqual(
            answer["mode"], "deterministic-grounded-evidence-digest-v2"
        )
        self.assertIn("F1 81.11", answer["answer"])
        self.assertIn("F1 82.83", answer["answer"])
        self.assertIn("[evidence_1]", answer["answer"])
        self.assertIn("回答（证据直出降级模式）", answer["answer"])
        self.assertIn("部分证实：Table XII", answer["answer"])
        self.assertIn("依据（TABLE，第 16 页）", answer["answer"])

    def test_deterministic_fallback_uses_each_mechanisms_own_source_span(self) -> None:
        mechanism_names = ("Sequential", "Cross", "Parallel")
        claims = [
            {
                "claim_id": f"claim_{name.casefold()}",
                "statement": (
                    "The primary source contains a concrete definition of the input or "
                    f"representation transformation used by the {name} mechanism."
                ),
                "status": "SUPPORTED",
                "gaps": ["primary_source"],
            }
            for name in mechanism_names
        ]
        claims.append(
            {
                "claim_id": "claim_table",
                "statement": "Table XII reports the baseline and mechanism ablation.",
                "status": "SUPPORTED",
                "gaps": ["numeric_results"],
            }
        )
        mechanism_claim_ids = [str(item["claim_id"]) for item in claims[:3]]
        answer = DeterministicAnswerSynthesizer().synthesize(
            question=(
                "说明 Sequential、Cross、Parallel 如何组织双时相特征，并结合 "
                "Table XII 回答。请将方法机制和实验结论分开回答。"
            ),
            claims=claims,
            evidence=[
                {
                    "evidence_id": "evidence_mechanisms",
                    "claim_ids": mechanism_claim_ids,
                    "claim_polarities": {
                        claim_id: "SUPPORTS" for claim_id in mechanism_claim_ids
                    },
                    "polarity": "SUPPORTS",
                    "evidence_type": "TEXT",
                    "page": 7,
                    "confidence": 0.9,
                    "content": (
                        "Sequential modeling arranges tokens in temporal order. "
                        "Cross modeling uses cross-ordering of temporal tokens. "
                        "Parallel modeling concatenates temporal tokens in the channel "
                        "dimension and then performs joint modeling."
                    ),
                    "provenance": {
                        "verification": {
                            "grounded_quote": (
                                "Parallel modeling concatenates temporal tokens."
                            )
                        }
                    },
                },
                {
                    "evidence_id": "evidence_table",
                    "claim_id": "claim_table",
                    "polarity": "SUPPORTS",
                    "evidence_type": "TABLE",
                    "page": 16,
                    "confidence": 0.9,
                    "content": "Visible transcription:\nBaseline | F1 81.11",
                    "provenance": {},
                },
            ],
        )

        self.assertIn("方法机制", answer["answer"])
        self.assertIn("实验结论", answer["answer"])
        self.assertIn("Sequential 原文", answer["answer"])
        self.assertIn("tokens in temporal order", answer["section_text_map"]["claim_sequential"])
        self.assertIn("cross-ordering", answer["section_text_map"]["claim_cross"])
        self.assertIn("channel dimension", answer["section_text_map"]["claim_parallel"])
        self.assertNotIn(
            "Parallel modeling",
            answer["section_text_map"]["claim_sequential"],
        )

    def test_deterministic_fallback_focuses_long_ablation_text_on_results(self) -> None:
        answer = DeterministicAnswerSynthesizer().synthesize(
            question="What does the MFA ablation show?",
            claims=[
                {
                    "claim_id": "claim_1",
                    "statement": "FlowMamba reports the MFA ablation.",
                    "status": "PARTIAL",
                }
            ],
            evidence=[
                {
                    "evidence_id": "evidence_1",
                    "claim_id": "claim_1",
                    "polarity": "NEUTRAL",
                    "claim_polarities": {"claim_1": "SUPPORTS"},
                    "evidence_type": "TEXT",
                    "page": 10,
                    "confidence": 0.82,
                    "content": (
                        ("preceding parameter analysis " * 100)
                        + "F. Ablation Study The baseline model achieves 80.14%. "
                        + "Integrating the MFA module increases the score to 80.26%."
                    ),
                    "provenance": {},
                }
            ],
        )

        assert "The baseline model achieves 80.14%" in answer["answer"]
        assert "MFA module increases the score to 80.26%" in answer["answer"]
        assert "preceding parameter analysis" not in answer["answer"]

    def test_deterministic_fallback_does_not_render_title_only_chunk(self) -> None:
        answer = DeterministicAnswerSynthesizer().synthesize(
            question="How does the method work?",
            claims=[
                {
                    "claim_id": "claim_1",
                    "statement": "The method aligns multitemporal features.",
                    "status": "SUPPORTED",
                }
            ],
            evidence=[
                {
                    "evidence_id": "evidence_title",
                    "claim_id": "claim_1",
                    "polarity": "SUPPORTS",
                    "evidence_type": "TEXT",
                    "page": 1,
                    "confidence": 0.99,
                    "content": "PaperName: A Short Descriptive Paper Title",
                    "provenance": {},
                },
                {
                    "evidence_id": "evidence_method",
                    "claim_id": "claim_1",
                    "polarity": "SUPPORTS",
                    "evidence_type": "TEXT",
                    "page": 4,
                    "confidence": 0.8,
                    "content": (
                        "The alignment module predicts a dense displacement field and "
                        "warps the post-event features into the pre-event coordinate frame."
                    ),
                    "provenance": {},
                },
            ],
        )
        self.assertNotIn("A Short Descriptive Paper Title", answer["answer"])
        self.assertIn("dense displacement field", answer["answer"])
        self.assertIn("依据（TEXT，第 4 页）", answer["answer"])

    def test_unrelated_retrieval_cannot_support_claim_or_multiply_on_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = _StaticSearchAdapter(
                [
                    {
                        "content": f"Bananas are yellow fruit number {index}.",
                        "source_uri": f"paper://banana/{index}",
                        "confidence": 0.9,
                    }
                    for index in range(8)
                ]
            )
            result = ResearchAgentRuntime(
                db_path=root / "runtime.sqlite3",
                workspace=root,
                search_adapter=adapter,
            ).start(question="Module X improves mIoU on Dataset Y.")

            claim = result["claim_graph"]["claims"][0]
            self.assertEqual(result["run"]["status"], "COMPLETED")
            self.assertEqual(claim["status"], "UNRESOLVED")
            self.assertEqual(len(result["evidence"]), 8)
            self.assertTrue(all(item["polarity"] == "NEUTRAL" for item in result["evidence"]))

    def test_multiple_relevant_sources_can_support_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = _StaticSearchAdapter(
                [
                    {
                        "content": "Module X improves mIoU on Dataset Y under the frozen protocol.",
                        "source_uri": f"paper://relevant/{index}",
                        "confidence": 0.9,
                    }
                    for index in range(3)
                ]
            )
            result = ResearchAgentRuntime(
                db_path=root / "runtime.sqlite3",
                workspace=root,
                search_adapter=adapter,
            ).start(question="Module X improves mIoU on Dataset Y.")

            claim = result["claim_graph"]["claims"][0]
            self.assertEqual(claim["status"], "SUPPORTED")
            self.assertEqual(claim["gaps"], [])

    def test_cross_run_deduplicated_evidence_is_visible_and_finalizes_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = _StaticSearchAdapter(
                [
                    {
                        "content": "A retrieved passage that is relevant but not direct enough.",
                        "source_uri": "paper://shared/chunk-1",
                        "confidence": 0.8,
                    }
                ]
            )
            runtime = ResearchAgentRuntime(
                db_path=root / "runtime.sqlite3",
                workspace=root,
                search_adapter=adapter,
            )
            first = runtime.start(question="Does module X improve dataset Y?")
            second = runtime.start(question="Does module X improve dataset Y?")

            self.assertEqual(len(first["evidence"]), 1)
            self.assertEqual(len(second["evidence"]), 1)
            self.assertEqual(
                first["evidence"][0]["source_uri"],
                second["evidence"][0]["source_uri"],
            )
            self.assertEqual(second["claim_graph"]["claims"][0]["status"], "UNRESOLVED")
            self.assertLessEqual(len(second["steps"]), 3)
            self.assertEqual(
                sum(
                    item["tool_name"] == "claim.finalize"
                    for item in second["tool_calls"]
                ),
                1,
            )

    def test_balanced_support_and_contradiction_remains_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = ResearchAgentRuntime(db_path=root / "runtime.sqlite3", workspace=root)
            initial = runtime.start(
                question="Claim with conflicting evidence",
                claims=[{"statement": "Claim with conflicting evidence", "gaps": ["primary_source"]}],
                auto_advance=False,
            )
            claim_id = initial["claim_graph"]["claims"][0]["claim_id"]
            for polarity in (EvidencePolarity.SUPPORTS, EvidencePolarity.CONTRADICTS):
                for index in range(3):
                    runtime.ledger.add(
                        run_id=initial["run"]["run_id"],
                        claim_id=claim_id,
                        evidence_type="TEXT",
                        content=f"Independent {polarity.value} evidence {index}",
                        source_uri=f"paper://{polarity.value.lower()}/{index}",
                        polarity=polarity,
                        confidence=0.9,
                        provenance={"gap": "primary_source"},
                    )

            claim = runtime.graph.update_from_evidence(claim_id)
            self.assertEqual(claim.status.value, "PARTIAL")
            self.assertTrue(claim.metadata["evidence_conflict"])


class SandboxSafetyTest(unittest.TestCase):
    def test_local_execution_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            (source / "experiment.py").write_text("print('ok')\n", encoding="utf-8")
            protocol = ExperimentProtocol.from_dict(
                {
                    "name": "deny-local-default",
                    "command": [sys.executable, "experiment.py"],
                    "source_dir": str(source),
                    "output_dir": str(root / "output"),
                    "policy": {"backend": "local", "network": "none"},
                }
            )
            with self.assertRaisesRegex(RuntimeError, "local fallback is disabled"):
                SandboxExperimentRunner().run(protocol)

    def test_local_output_limit_marks_experiment_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            (source / "experiment.py").write_text(
                "from pathlib import Path\nPath('large.bin').write_bytes(b'x' * (2 * 1024 * 1024))\n",
                encoding="utf-8",
            )
            protocol = ExperimentProtocol.from_dict(
                {
                    "name": "bounded-output",
                    "command": [sys.executable, "experiment.py"],
                    "source_dir": str(source),
                    "output_dir": str(root / "output"),
                    "policy": {
                        "backend": "local",
                        "network": "host",
                        "allow_local_fallback": True,
                        "max_output_mb": 1,
                        "timeout_seconds": 10,
                    },
                }
            )
            result = SandboxExperimentRunner().run(protocol)
            self.assertEqual(result.status, "FAILED")
            self.assertIn("output exceeded", result.error or "")


@unittest.skipIf(fitz is None, "PyMuPDF not installed")
class LayoutEvidenceTest(unittest.TestCase):
    def test_caption_table_crop_is_bounded_to_detected_column(self) -> None:
        page = SimpleNamespace(rect=SimpleNamespace(width=600.0, height=800.0))
        blocks = [
            ([40.0, 40.0 + index * 20, 270.0, 52.0 + index * 20], "left")
            for index in range(3)
        ] + [
            ([330.0, 40.0 + index * 20, 570.0, 52.0 + index * 20], "right")
            for index in range(3)
        ]

        bbox = _caption_context_bbox(page, [150.0, 200.0, 210.0, 210.0], blocks, "table")

        self.assertEqual(bbox, [0.0, 188.0, 318.0, 350.0])

    def test_centered_caption_table_crop_spans_both_columns(self) -> None:
        page = SimpleNamespace(rect=SimpleNamespace(width=600.0, height=800.0))
        blocks = [
            ([40.0, 40.0 + index * 20, 270.0, 52.0 + index * 20], "left")
            for index in range(3)
        ] + [
            ([330.0, 40.0 + index * 20, 570.0, 52.0 + index * 20], "right")
            for index in range(3)
        ]

        bbox = _caption_context_bbox(page, [280.0, 200.0, 320.0, 210.0], blocks, "table")

        self.assertEqual(bbox, [0.0, 188.0, 600.0, 390.0])

    def test_layout_manifest_contains_coordinates_and_rendered_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / "paper.pdf"
            doc = fitz.open()
            page = doc.new_page(width=595, height=842)
            page.insert_text((60, 70), "4.2 Experimental Results", fontsize=16)
            page.insert_text((60, 110), "The proposed method improves mIoU under the frozen protocol.", fontsize=10)
            # Draw a simple table and add a caption; recent PyMuPDF versions detect it.
            x0, y0, x1, y1 = 60, 150, 500, 250
            for y in (y0, 185, 220, y1):
                page.draw_line((x0, y), (x1, y))
            for x in (x0, 250, 370, x1):
                page.draw_line((x, y0), (x, y1))
            page.insert_text((70, 175), "Model")
            page.insert_text((270, 175), "mIoU")
            page.insert_text((390, 175), "Recall")
            page.insert_text((70, 210), "Ours")
            page.insert_text((270, 210), "68.50")
            page.insert_text((390, 210), "80.39")
            page.insert_text((60, 275), "Table 2. Frozen-protocol results.", fontsize=9)
            page.insert_text((60, 330), "Figure 3. Qualitative comparison of damaged buildings.", fontsize=9)
            page.insert_text((60, 380), "L = L_ce + α * L_boundary", fontsize=11)
            doc.save(str(pdf))
            doc.close()

            manifest = parse_pdf_layout(pdf, root / "layout", render_regions=True)
            self.assertEqual(manifest.page_count, 1)
            self.assertTrue(manifest.objects)
            self.assertTrue(all(len(item.bbox) == 4 for item in manifest.objects))
            self.assertTrue(any(item.object_type == "figure" for item in manifest.objects))
            self.assertTrue(any(item.object_type == "equation" for item in manifest.objects))
            rendered = [item for item in manifest.objects if item.artifact_path]
            self.assertTrue(rendered)
            self.assertTrue(all(Path(item.artifact_path).is_file() for item in rendered))
            caption_objects = [
                item for item in manifest.objects if item.metadata.get("caption_only")
            ]
            self.assertTrue(caption_objects)
            self.assertTrue(all(not item.metadata.get("vision_verified") for item in caption_objects))
            self.assertTrue((root / "layout" / "layout_manifest.json").is_file())

            ranked = search_layout_objects(
                {
                    "objects": [
                        {
                            "object_id": "table-xi",
                            "page": 1,
                            "object_type": "table",
                            "bbox": [100, 40, 200, 50],
                            "text": "TABLE XI",
                            "caption": "TABLE XI",
                        },
                        {
                            "object_id": "text-xi",
                            "page": 1,
                            "object_type": "text",
                            "bbox": [50, 60, 250, 75],
                            "text": "Comparison with baseline methods",
                        },
                        {
                            "object_id": "table-xii",
                            "page": 1,
                            "object_type": "table",
                            "bbox": [100, 200, 200, 210],
                            "text": "TABLE XII",
                            "caption": "TABLE XII",
                        },
                        {
                            "object_id": "text-xii",
                            "page": 1,
                            "object_type": "text",
                            "bbox": [50, 215, 250, 235],
                            "text": "Ablation study of spatiotemporal modeling mechanisms on SYSU",
                        },
                    ]
                },
                query="ablation spatiotemporal mechanisms SYSU",
                object_type="table",
                top_k=1,
            )
            self.assertEqual(ranked[0]["object_id"], "table-xii")
            self.assertIn("Ablation study", ranked[0]["retrieval_context"])
            chinese_label = search_layout_objects(
                {
                    "objects": [
                        {
                            "object_id": "table-xi",
                            "page": 1,
                            "object_type": "table",
                            "bbox": [100, 40, 200, 50],
                            "text": "TABLE XI",
                            "caption": "TABLE XI",
                        },
                        {
                            "object_id": "table-xii",
                            "page": 1,
                            "object_type": "table",
                            "bbox": [100, 200, 200, 210],
                            "text": "TABLE XII",
                            "caption": "TABLE XII",
                        },
                    ]
                },
                query="仅依据表 XII 回答",
                object_type="table",
                top_k=1,
            )
            self.assertEqual(chinese_label[0]["object_id"], "table-xii")

            vision = _FixtureVisionAdapter()
            result = ResearchAgentRuntime(
                db_path=root / "runtime.sqlite3",
                workspace=root / "workspace",
                vision_adapter=vision,
            ).start(
                question="Table 2 reports mIoU 68.50 and Recall 80.39.",
                claims=[
                    {
                        "statement": "Table 2 reports mIoU 68.50 and Recall 80.39.",
                        "gaps": ["numeric_results"],
                        "metadata": {"pdf_paths": [str(pdf)]},
                    },
                    {
                        "statement": "The ablation table supports the reported numeric result.",
                        "gaps": ["numeric_results"],
                        "metadata": {"pdf_paths": [str(pdf)]},
                    },
                ],
            )
            self.assertTrue(
                any(
                    item["tool_name"] == "paper.inspect_layout"
                    for item in result["tool_calls"]
                ),
                [(item['tool_name'], item.get('status'), item.get('error'),
                  (item.get('result') or {}).get('output')) for item in result['tool_calls']],
            )
            self.assertGreater(vision.calls, 0)
            self.assertLessEqual(vision.calls, 2)
            self.assertEqual(result["run"]["used_llm_calls"], vision.calls)
            self.assertEqual(
                len(
                    [
                        item
                        for item in result["llm_calls"]
                        if item["stage"] == "vision"
                    ]
                ),
                vision.calls,
            )
            self.assertTrue(
                any(
                    int((item.get("result") or {}).get("output", {}).get("vision_cache_hits") or 0)
                    > 0
                    for item in result["tool_calls"]
                ),
                [(item['tool_name'], item.get('status'), item.get('error'),
                  (item.get('result') or {}).get('output')) for item in result['tool_calls']],
            )
            self.assertTrue(result["evidence"])
            self.assertTrue(
                all(item["source_uri"].startswith("paper://") for item in result["evidence"])
            )
            self.assertTrue(
                any(
                    item["provenance"].get("analysis_mode") == "multimodal_adapter"
                    for item in result["evidence"]
                )
            )

            calls_before_second_run = vision.calls
            cached_result = ResearchAgentRuntime(
                db_path=root / "runtime.sqlite3",
                workspace=root / "workspace",
                vision_adapter=vision,
            ).start(
                question="Re-read Table 2 using the exact persisted crop.",
                claims=[
                    {
                        "statement": "Table 2 reports mIoU 68.50 and Recall 80.39.",
                        "gaps": ["numeric_results"],
                        "metadata": {"pdf_paths": [str(pdf)]},
                    }
                ],
            )
            self.assertEqual(vision.calls, calls_before_second_run)
            self.assertGreater(vision.reaudits, 0)
            self.assertTrue(
                any(
                    int(
                        (item.get("result") or {}).get("output", {}).get(
                            "vision_cross_run_cache_hits"
                        )
                        or 0
                    )
                    > 0
                    for item in cached_result["tool_calls"]
                )
            )
            self.assertTrue(
                any(
                    item["provenance"].get("vision_cache_scope")
                    == "cross_run_exact_crop"
                    for item in cached_result["evidence"]
                )
            )

            fresh_result = ResearchAgentRuntime(
                db_path=root / "runtime.sqlite3",
                workspace=root / "workspace",
                vision_adapter=vision,
            ).start(
                question="Read Table 2 with the current adapter, without cross-run reuse.",
                metadata={"disable_cross_run_visual_cache": True},
                claims=[{
                    "statement": "Table 2 reports mIoU 68.50 and Recall 80.39.",
                    "gaps": ["numeric_results"],
                    "metadata": {"pdf_paths": [str(pdf)]},
                }],
            )
            self.assertGreater(vision.calls, calls_before_second_run)
            self.assertFalse(any(
                (item.get("result") or {}).get("output", {}).get("vision_cross_run_cache_hits")
                for item in fresh_result["tool_calls"]
            ))

            failing_vision = _FailingVisionAdapter()
            failed_visual = ResearchAgentRuntime(
                db_path=root / "failed-vision.sqlite3",
                workspace=root / "failed-vision-workspace",
                vision_adapter=failing_vision,
            ).start(
                question="Read Table 2 without inventing unreadable cells.",
                claims=[
                    {
                        "statement": "Table 2 reports the ablation values.",
                        "gaps": ["numeric_results"],
                        "metadata": {"pdf_paths": [str(pdf)]},
                    }
                ],
                budget=Budget(
                    max_steps=4,
                    max_tool_calls=2,
                    max_llm_calls=4,
                    synthesis_reserve_calls=1,
                    repair_reserve_calls=1,
                ),
            )
            self.assertNotEqual(failed_visual["run"]["status"], "FAILED")
            self.assertEqual(
                failed_visual["run"]["used_llm_calls"], failing_vision.calls
            )
            failed_rows = [
                item
                for item in failed_visual["llm_calls"]
                if item["stage"] == "vision"
            ]
            self.assertEqual(len(failed_rows), failing_vision.calls)
            self.assertTrue(all(item['total_tokens'] == 165 for item in failed_rows))
            self.assertEqual(failed_visual['run']['used_tokens'], 165 * failing_vision.calls)
            self.assertTrue(all(item["status"] == "FAILED" for item in failed_rows))
            self.assertGreater(failing_vision.calls, 0)
            self.assertTrue(
                any(
                    (item.get("result") or {}).get("output", {}).get("vision_errors")
                    for item in failed_visual["tool_calls"]
                ),
                failed_visual["tool_calls"],
            )


if __name__ == "__main__":
    unittest.main()
