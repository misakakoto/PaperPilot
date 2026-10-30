"""Deterministic XOR interaction ablation used only for Docker acceptance."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        (first, second, int(first == second))
        for _ in range(64)
        for first, second in ((-1, -1), (-1, 1), (1, -1), (1, 1))
    ]
    random.Random(args.seed).shuffle(rows)
    baseline_correct = sum(int(int(first > 0) == label) for first, _, label in rows)
    candidate_correct = sum(
        int(int(first * second > 0) == label) for first, second, label in rows
    )
    baseline = baseline_correct / len(rows)
    candidate = candidate_correct / len(rows)
    metrics = {
        "seed": args.seed,
        "samples": len(rows),
        "baseline_accuracy": baseline,
        "candidate_accuracy": candidate,
        "improvement": candidate - baseline,
    }
    args.output.write_text(
        json.dumps(metrics, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
