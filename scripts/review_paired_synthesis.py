"""Offline replay of a recorded structured response through current validation."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paperpilot_agentic.context_engine import audit_final_answer_quality  # noqa: E402
from paperpilot_agentic.intelligence import LLMAnswerSynthesizer  # noqa: E402
from paperpilot_agentic.providers import LLMResponse  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8-sig"))
    recording = json.loads(args.recording.read_text(encoding="utf-8-sig"))

    class Replay:
        def complete(self, messages, **kwargs):
            if messages != recording["arms"]["structured"]["messages"]:
                raise ValueError("Prompt drift: cannot replay this response against changed input")
            return LLMResponse(text=recording["results"]["structured"]["text"], finish_reason="stop")

    question = snapshot["run"]["question"]
    claims = snapshot["claim_graph"]["claims"]
    evidence = snapshot["evidence"]
    result = {"network_calls": 0, "paid_requests": 0}
    try:
        answer = LLMAnswerSynthesizer(Replay()).synthesize(
            question=question, claims=claims, evidence=evidence,
        )
        result["rendered_answer"] = answer
        result["quality"] = audit_final_answer_quality(
            answer, claims=claims, evidence=evidence, question=question,
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = ["# 同证据回答对照（真实付费输出）", "", "## 原问题", "", question,
                "", "## A：直接中文回答（原样保留）", "",
                recording["results"]["direct"]["text"],
                "", "## B：结构化回答（原响应经原有渲染器展示）", "",
                (result.get("rendered_answer") or {}).get("answer", "未通过渲染，见 JSON 原始记录"),
                "", "## 验证范围", "",
                "仅固定证据的综合阶段，未重新运行检索、规划或视觉读表。"
                "质量门通过不等于大样本正确率。A 的额外数据集猜测和公式重建仍需警惕。"]
    args.output.with_suffix(".md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "quality": result.get("quality"),
                      "error": result.get("error")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
