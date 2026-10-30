from __future__ import annotations

import json
from typing import Any

import pytest

from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.paid_synthesis import (
    SingleUpstreamAttemptLLM,
    synthesize_verified_replay_once,
)


class _OneShotModel:
    model = "fixture"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.provider_attempts = 0
        self.thinking_enabled_values: list[bool | None] = []

    def complete(self, _messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.thinking_enabled_values.append(kwargs.get("thinking_enabled"))
        callback = kwargs.get("before_attempt")
        if callable(callback):
            callback()
        self.provider_attempts += 1
        return LLMResponse(
            text=json.dumps(self.payload),
            prompt_tokens=100,
            completion_tokens=50,
            finish_reason="stop",
            model=self.model,
        )


def _replay() -> dict[str, Any]:
    return {
        "kind": "zero_network_paid_evidence_replay",
        "replay_research_outcome": "ANSWERED",
        "source_scope_violations": 0,
        "network_calls": 0,
        "paid_requests": 0,
        "question": "What does the method do?",
        "claims": [
            {
                "claim_id": "claim_1",
                "statement": "The method aligns features.",
                "status": "SUPPORTED",
                "confidence": 0.9,
                "gaps": [],
                "metadata": {},
            }
        ],
        "evidence": [
            {
                "evidence_id": "evidence_1",
                "claim_id": "claim_1",
                "claim_ids": ["claim_1"],
                "polarity": "SUPPORTS",
                "evidence_type": "TEXT",
                "source_uri": "paper://fixture/p1",
                "confidence": 0.9,
                "content": "The method aligns pre- and post-event features.",
                "provenance": {
                    "verification": {
                        "verifier": "llm-entailment-v1",
                        "degraded": False,
                    }
                },
            }
        ],
    }


def test_paid_replay_synthesis_uses_exactly_one_attempt() -> None:
    inner = _OneShotModel(
        {
            "sections": [
                {
                    "claim_id": "claim_1",
                    "text": (
                        "The cited passage explains that the method explicitly connects "
                        "pre-event and post-event representations during alignment."
                    ),
                    "evidence_ids": ["evidence_1"],
                }
            ],
            "unresolved": [],
        }
    )
    model = SingleUpstreamAttemptLLM(inner)

    result = synthesize_verified_replay_once(_replay(), model)

    assert result["answer"]["mode"] == "llm-grounded-synthesis-v3"
    assert result["citation_ids"] == ["evidence_1"]
    assert model.high_level_calls == 1
    assert model.upstream_attempts == 1
    assert inner.provider_attempts == 1
    with pytest.raises(RuntimeError, match="one high-level"):
        model.complete([])


def test_paid_replay_synthesis_fails_without_retry_on_bad_citation() -> None:
    inner = _OneShotModel(
        {
            "sections": [
                {
                    "claim_id": "claim_1",
                    "text": "Unsupported citation.",
                    "evidence_ids": ["evidence_other"],
                }
            ],
            "unresolved": [],
        }
    )
    model = SingleUpstreamAttemptLLM(inner)

    with pytest.raises(ValueError, match="not linked"):
        synthesize_verified_replay_once(_replay(), model)

    assert model.high_level_calls == 1
    assert model.upstream_attempts == 1
    assert inner.provider_attempts == 1


def test_paid_replay_synthesis_rejects_number_absent_from_cited_evidence() -> None:
    inner = _OneShotModel(
        {
            "sections": [
                {
                    "claim_id": "claim_1",
                    "text": "The method improves the score to 99.9.",
                    "evidence_ids": ["evidence_1"],
                }
            ],
            "unresolved": [],
        }
    )
    model = SingleUpstreamAttemptLLM(inner)

    with pytest.raises(ValueError, match="numeric literals"):
        synthesize_verified_replay_once(_replay(), model)

    assert model.upstream_attempts == 1
    assert inner.provider_attempts == 1


def test_paid_replay_synthesis_rejects_unresolved_preflight_without_call() -> None:
    replay = _replay()
    replay["claims"][0]["status"] = "PARTIAL"
    inner = _OneShotModel({})
    model = SingleUpstreamAttemptLLM(inner)

    with pytest.raises(ValueError, match="must be resolved"):
        synthesize_verified_replay_once(replay, model)

    assert model.high_level_calls == 0
    assert model.upstream_attempts == 0
    assert inner.provider_attempts == 0


def test_single_attempt_wrapper_can_force_thinking_off() -> None:
    inner = _OneShotModel(
        {
            "sections": [
                {
                    "claim_id": "claim_1",
                    "text": (
                        "The cited passage explains that the method explicitly connects "
                        "pre-event and post-event representations during alignment."
                    ),
                    "evidence_ids": ["evidence_1"],
                }
            ],
            "unresolved": [],
        }
    )
    model = SingleUpstreamAttemptLLM(inner, force_thinking_enabled=False)

    synthesize_verified_replay_once(_replay(), model)

    assert inner.thinking_enabled_values == [False]


def test_paid_synthesis_adds_explicitly_requested_epistemic_boundary() -> None:
    replay = _replay()
    replay["question"] = "Why is this mechanism more direct, and what cannot be proven?"
    inner = _OneShotModel(
        {
            "sections": [
                {
                    "claim_id": "claim_1",
                    "text": "The method aligns features more directly.",
                    "evidence_ids": ["evidence_1"],
                }
            ],
            "unresolved": [],
        }
    )

    result = synthesize_verified_replay_once(
        replay,
        SingleUpstreamAttemptLLM(inner),
    )

    assert "不能证明" in result["answer"]["answer"]
    assert result["answer"]["deterministic_contract_additions"] == [
        "explicit_epistemic_boundary"
    ]


def test_paid_synthesis_rejects_english_answer_to_chinese_question() -> None:
    replay = _replay()
    replay["question"] = "这个方法如何对齐双时相特征，并说明不能证明什么？"
    inner = _OneShotModel(
        {
            "sections": [
                {
                    "claim_id": "claim_1",
                    "text": "The method aligns pre-event and post-event features.",
                    "evidence_ids": ["evidence_1"],
                }
            ],
            "unresolved": [],
        }
    )

    with pytest.raises(ValueError, match="response_language_mismatch"):
        synthesize_verified_replay_once(
            replay,
            SingleUpstreamAttemptLLM(inner),
        )


def test_paid_synthesis_requires_explicit_requested_answer_sections() -> None:
    replay = _replay()
    replay["question"] = (
        "请明确分成“论文事实”“比较性推断”“结论边界”回答，说明不能证明什么。"
    )
    inner = _OneShotModel(
        {
            "sections": [
                {
                    "claim_id": "claim_1",
                    "text": "论文说明该方法会对齐事件前后的特征。",
                    "evidence_ids": ["evidence_1"],
                }
            ],
            "unresolved": [],
        }
    )

    with pytest.raises(ValueError, match="requested_answer_sections_missing"):
        synthesize_verified_replay_once(
            replay,
            SingleUpstreamAttemptLLM(inner),
        )


@pytest.mark.parametrize(
    "question",
    [
        "请将方法机制和实验结论分开回答。",
        "请区分论文事实与比较性推断。",
    ],
)
def test_paid_synthesis_enforces_additional_requested_section_phrasings(
    question: str,
) -> None:
    replay = _replay()
    replay["question"] = question
    inner = _OneShotModel(
        {
            "sections": [
                {
                    "claim_id": "claim_1",
                    "text": "论文说明该方法会对齐事件前后的特征。",
                    "evidence_ids": ["evidence_1"],
                }
            ],
            "unresolved": [],
        }
    )

    with pytest.raises(ValueError, match="requested_answer_sections_missing"):
        synthesize_verified_replay_once(
            replay,
            SingleUpstreamAttemptLLM(inner),
        )


def test_paid_synthesis_deterministically_groups_method_and_results() -> None:
    replay = _replay()
    replay["question"] = "请将方法机制和实验结论分开回答。"
    replay["claims"][0].update(
        statement="The method cross-orders tokens.",
        gaps=["primary_source"],
    )
    replay["evidence"][0]["content"] = "The method cross-orders temporal tokens."
    replay["claims"].append(
        {
            "claim_id": "claim_2",
            "statement": "The ablation reports an F1 value.",
            "status": "SUPPORTED",
            "confidence": 0.9,
            "gaps": ["numeric_results"],
            "metadata": {},
        }
    )
    replay["evidence"].append(
        {
            "evidence_id": "evidence_2",
            "claim_id": "claim_2",
            "claim_ids": ["claim_2"],
            "polarity": "SUPPORTS",
            "evidence_type": "TABLE",
            "source_uri": "paper://fixture/p2",
            "confidence": 0.9,
            "content": "The reported F1 value is 82.24.",
            "provenance": {
                "verification": {
                    "verifier": "llm-entailment-v1",
                    "degraded": False,
                }
            },
        }
    )
    inner = _OneShotModel(
        {
            "sections": [
                {
                    "claim_id": "claim_1",
                    "text": "该方法将两个时相的特征 token 逐位置交叉排序，再送入状态空间模块建模。",
                    "evidence_ids": ["evidence_1"],
                },
                {
                    "claim_id": "claim_2",
                    "text": "对应消融实验报告该配置的 F1 为 82.24，数值来自所引用表格。",
                    "evidence_ids": ["evidence_2"],
                },
            ],
            "unresolved": [],
        }
    )

    result = synthesize_verified_replay_once(
        replay,
        SingleUpstreamAttemptLLM(inner),
    )

    answer = result["answer"]["answer"]
    assert answer.index("方法机制") < answer.index("实验结论")
    assert result["answer_quality"]["passed"] is True


def test_paid_synthesis_renders_cross_paper_inference_with_both_sources() -> None:
    replay = _replay()
    replay["question"] = "请区分论文事实与比较性推断。"
    replay["evidence"][0]["source_uri"] = "paper://paper_a/p1"
    replay["claims"].append(
        {
            "claim_id": "claim_2",
            "statement": "The other method warps features.",
            "status": "SUPPORTED",
            "confidence": 0.9,
            "gaps": ["primary_source"],
            "metadata": {},
        }
    )
    replay["evidence"].append(
        {
            "evidence_id": "evidence_2",
            "claim_id": "claim_2",
            "claim_ids": ["claim_2"],
            "polarity": "SUPPORTS",
            "evidence_type": "TEXT",
            "source_uri": "paper://paper_b/p2",
            "confidence": 0.9,
            "content": "The other method explicitly warps bi-temporal features.",
            "provenance": {
                "verification": {
                    "verifier": "llm-entailment-v1",
                    "degraded": False,
                }
            },
        }
    )
    payload = {
        "sections": [
            {
                "claim_id": "claim_1",
                "text": "论文 A 说明该方法会在模型内部对齐事件前后两个时相的特征表示。",
                "evidence_ids": ["evidence_1"],
            },
            {
                "claim_id": "claim_2",
                "text": "论文 B 说明该方法利用估计的位移场显式扭曲双时相特征。",
                "evidence_ids": ["evidence_2"],
            },
        ],
        "comparative_inference": {
            "text": "与一般特征对齐相比，显式扭曲更直接作用于空间错位。",
            "evidence_ids": ["evidence_1", "evidence_2"],
        },
        "unresolved": [],
    }
    inner = _OneShotModel(payload)

    result = synthesize_verified_replay_once(
        replay,
        SingleUpstreamAttemptLLM(inner),
    )

    answer = result["answer"]
    assert "论文事实" in answer["answer"]
    assert "比较性推断" in answer["answer"]
    assert answer["synthesis_section_evidence_map"]["比较性推断"] == [
        "evidence_1",
        "evidence_2",
    ]
    assert result["answer_quality"]["passed"] is True

    payload["comparative_inference"]["evidence_ids"] = ["evidence_2"]
    with pytest.raises(ValueError, match="omitted evidence from a compared paper"):
        synthesize_verified_replay_once(
            replay,
            SingleUpstreamAttemptLLM(_OneShotModel(payload)),
        )
