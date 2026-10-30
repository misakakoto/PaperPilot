"""Zero-network retrieval preflight for the tracked interview scenarios."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paperpilot_agentic.bootstrap import build_runtime  # noqa: E402
from paperpilot_agentic.config import RuntimeSettings  # noqa: E402


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="demo/interview_scenarios.json")
    parser.add_argument("--output", default=".paperpilot/demo/retrieval_preflight.json")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    scenario_path = Path(args.scenarios).expanduser().resolve()
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    settings = RuntimeSettings.from_env()
    runtime = build_runtime(settings)
    adapter = runtime.search_adapter
    resolver = getattr(adapter, "resolve_explicit_title_scope", None)
    if not callable(resolver):
        raise RuntimeError("Configured search adapter cannot resolve paper titles")

    embedder = getattr(adapter, "embedder", None)
    store = getattr(adapter, "store", None)
    dense_index: dict[str, Any] = {"configured": embedder is not None}
    if embedder is not None:
        if store is None or not callable(getattr(store, "stats", None)):
            raise RuntimeError("Dense retrieval is configured without durable index stats")
        identity = str(embedder.cache_identity)
        stats = dict(store.stats())
        active_chunks = int(stats.get("chunks") or 0)
        indexed_chunks = int((stats.get("embeddings") or {}).get(identity) or 0)
        dense_index.update(
            {
                "identity": identity,
                "active_chunks": active_chunks,
                "indexed_chunks": indexed_chunks,
                "complete": active_chunks > 0 and indexed_chunks == active_chunks,
            }
        )

    checks: list[dict[str, Any]] = []
    for probe in payload.get("retrieval_probes", []):
        title = str(probe["paper_title"])
        scope = set(resolver([title]))
        if len(scope) != 1:
            raise RuntimeError(f"Expected one indexed paper for {title!r}, got {len(scope)}")
        hits = list(
            adapter.search(
                str(probe["query"]),
                top_k=max(1, int(args.top_k)),
                paper_sha256s=scope,
            )
        )
        combined = _normalize("\n".join(str(item.get("content") or "") for item in hits))
        groups = [
            [str(term) for term in group]
            for group in probe.get("required_term_groups", [])
        ]
        group_results = [
            {
                "alternatives": group,
                "matched": any(_normalize(term) in combined for term in group),
            }
            for group in groups
        ]
        check = {
            "name": probe["name"],
            "paper_sha256": next(iter(scope)),
            "hit_count": len(hits),
            "pages": [item.get("page") for item in hits],
            "stable_ids": [item.get("stable_id") for item in hits],
            "term_groups": group_results,
            "passed": bool(hits) and all(item["matched"] for item in group_results),
        }
        checks.append(check)

    result = {
        "schema_version": 1,
        "kind": "zero_network_interview_demo_retrieval_preflight",
        "scenario_file": str(scenario_path),
        "probe_count": len(checks),
        "passed": (
            bool(checks)
            and all(item["passed"] for item in checks)
            and dense_index.get("configured") is True
            and dense_index.get("complete") is True
        ),
        "checks": checks,
        "dense_index": dense_index,
        "network_calls": 0,
        "paid_requests": 0,
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
