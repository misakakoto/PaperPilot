from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping
from types import SimpleNamespace

from .adapters import (
    PaperSearchAdapter,
    VisionEvidenceAdapter,
    make_search_adapter,
    make_vision_adapter,
)
from .layout import parse_pdf_layout, search_layout_objects
from .models import (
    EvidencePolarity,
    EvidenceType,
    ExperimentProtocol,
    RiskLevel,
    SandboxPolicy,
    ToolResult,
    stable_hash,
)
from .sandbox import SandboxExperimentRunner
from .scholarly import ScholarlySearchAdapter
from .layout_identity import (
    _prioritize_visual_candidates as _prioritize_visual_candidates,
    _explicit_layout_label as _explicit_layout_label,
    _matches_explicit_layout_label as _matches_explicit_layout_label,
)
from .tooling import ToolContext, ToolRegistry, ToolSpec


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_output_dir(context: ToolContext, requested: str | None, name: str) -> Path:
    run_root = context.workspace / ".paperpilot_agentic" / "runs" / context.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    if not requested:
        target = run_root / name
    else:
        candidate = Path(requested).expanduser()
        target = candidate.resolve() if candidate.is_absolute() else (run_root / candidate).resolve()
        try:
            target.relative_to(run_root.resolve())
        except ValueError as exc:
            raise PermissionError(f"Output directory must stay under {run_root}") from exc
    target.mkdir(parents=True, exist_ok=True)
    return target


def _adapter_for_run(context, base):
    metadata = context.db.get_run(context.run_id)['metadata']
    if metadata.get('retrieval_profile') not in {'token_evidence_pilot', 'paper_evidence_v2'}:
        return base
    from .structured_retrieval import select_retriever
    selected, _ = select_retriever(SimpleNamespace(workspace=context.workspace, search_adapter=base), metadata)
    return selected


def _token_paper_scope(context):
    metadata = context.db.get_run(context.run_id)['metadata']
    if metadata.get('retrieval_profile') not in {'token_evidence_pilot', 'paper_evidence_v2'}:
        return None
    value = metadata.get('resolved_paper_sha256s', metadata.get('paper_sha256s'))
    return set(value) if value is not None else None


def _search_handler(adapter: PaperSearchAdapter):
    base = adapter
    def handler(context: ToolContext, args: Mapping[str, Any]) -> ToolResult:
        adapter = _adapter_for_run(context, base)
        query = str(args["query"])
        queries = [
            str(item).strip()
            for item in args.get("queries", [query])
            if str(item).strip()
        ]
        queries = list(dict.fromkeys(queries or [query]))[:6]
        top_k = max(1, min(100, int(args.get("top_k", 8))))
        raw_hashes = args.get("paper_sha256s") or []
        paper_sha256s = {str(value) for value in raw_hashes} or None
        scope = _token_paper_scope(context)
        if scope is not None:
            if paper_sha256s is not None and not paper_sha256s <= scope:
                raise ValueError('Search escaped requested paper scope')
            paper_sha256s = scope if paper_sha256s is None else paper_sha256s
            if not paper_sha256s:
                return ToolResult(ok=True, output={'result_count': 0, 'evidence_count': 0}, evidence=[])
        search_many = getattr(adapter, "search_many", None)
        if callable(search_many):
            items = list(
                search_many(queries, top_k=top_k, paper_sha256s=paper_sha256s)
            )
        else:
            items = list(
                adapter.search(query, top_k=top_k, paper_sha256s=paper_sha256s)
                if paper_sha256s
                else adapter.search(query, top_k=top_k)
            )
        evidence: list[dict[str, Any]] = []
        for rank, raw in enumerate(items, start=1):
            item = dict(raw)
            content = str(item.get("content") or item.get("text") or "").strip()
            if not content:
                continue
            confidence = float(item.get("confidence", item.get("score", 0.7)))
            if confidence > 1.0:
                confidence = 1.0
            paper_sha256 = item.get("paper_sha256")
            page = item.get("page")
            fragment_hash = stable_hash(
                "paper-fragment-v1",
                paper_sha256,
                page,
                item.get("stable_id"),
                content,
            )
            evidence.append(
                {
                    "evidence_type": (item.get('evidence_type', EvidenceType.TEXT.value)
                                      if type(adapter).__name__ == 'PaperEvidenceAdapter' else EvidenceType.TEXT.value),
                    "content": content,
                    "source_uri": item.get("source_uri") or item.get("path"),
                    "page": item.get("page"),
                    "confidence": max(0.25, min(0.9, confidence)),
                    # Retrieval produces candidates, not entailment decisions.  The
                    # runtime verifier assigns SUPPORTS/CONTRADICTS after comparing
                    # each candidate with the active claim.
                    "polarity": EvidencePolarity.NEUTRAL.value,
                    "provenance": {
                        "rank": item.get("rank", rank),
                        "retrieval_score": item.get("score"),
                        "query": query,
                        "queries": queries,
                        "gap": args.get("gap"),
                        "attempt_index": args.get("attempt_index"),
                        "paper_scope_size": len(paper_sha256s or ()),
                        "adapter": type(adapter).__name__,
                        "paper_sha256": paper_sha256,
                        "document_sha256": paper_sha256,
                        "stable_id": item.get("stable_id"),
                        "fragment_hash": fragment_hash,
                        "evidence_anchor": (
                            f"paper://{paper_sha256}/page/{page}/fragment/{fragment_hash}"
                            if paper_sha256 and page is not None
                            else None
                        ),
                        "file_name": item.get("file_name"),
                        "title": item.get("title"),
                        "page_end": item.get("page_end"),
                        "channels": item.get("channels"),
                        "memory_ids": item.get("memory_ids"),
                        "explicit_title_scope": bool(item.get("explicit_title_scope")),
                        "resolved_title_scope_size": int(
                            item.get("resolved_title_scope_size") or 0
                        ),
                        **{k: v for k, v in (item.get('provenance') or {}).items()
                           if k in {'index_identity', 'retrieval_profile', 'parent_id', 'source_segments', 'input_tokens',
                                    'unit_id', 'matched_window_id', 'claim_role', 'page_range', 'assurance'}},
                    },
                }
            )
        return ToolResult(
            ok=True,
            output={
                "query": query,
                "queries": queries,
                "result_count": len(items),
                "evidence_count": len(evidence),
            },
            evidence=evidence,
        )

    return handler


def _layout_handler(vision_adapter: VisionEvidenceAdapter | None):
    from .layout_extraction import ExtractionRequest, LayoutExtraction
    extraction = LayoutExtraction(vision_adapter, layout_parser=parse_pdf_layout,
        layout_search=search_layout_objects, output_dir=_safe_output_dir, file_hash=_sha256)
    def handler(context, args):
        return extraction.extract(ExtractionRequest(context, args))
    return handler


def _code_handler(context: ToolContext, args: Mapping[str, Any]) -> ToolResult:
    path = Path(str(args["path"])).expanduser().resolve()
    if not path.exists():
        return ToolResult(ok=False, error=f"Code path not found: {path}")
    query_tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", str(args.get("query", "")))}
    files = [path] if path.is_file() else list(path.rglob("*.py"))
    findings: list[tuple[float, dict[str, Any]]] = []
    for file_path in files[:2_000]:
        if any(part in {".git", ".venv", "venv", "node_modules", "__pycache__"} for part in file_path.parts):
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(text, filename=str(file_path))
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            start = max(1, int(getattr(node, "lineno", 1)))
            end = max(start, int(getattr(node, "end_lineno", start)))
            lines = text.splitlines()
            snippet = "\n".join(lines[start - 1 : min(end, start + 80)])
            tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", f"{node.name} {snippet}")}
            overlap = len(tokens & query_tokens) / max(1, len(query_tokens))
            score = overlap + (0.25 if node.name.lower() in query_tokens else 0.0)
            findings.append(
                (
                    score,
                    {
                        "file": (
                            file_path.relative_to(path if path.is_dir() else path.parent)
                            .as_posix()
                        ),
                        "source_uri": f"code://{_sha256(file_path)}",
                        "symbol": node.name,
                        "symbol_type": type(node).__name__,
                        "line_start": start,
                        "line_end": end,
                        "snippet": snippet[:8_000],
                        "score": score,
                    },
                )
            )
    findings.sort(key=lambda pair: pair[0], reverse=True)
    selected = [item for _, item in findings[: max(1, min(20, int(args.get("top_k", 8))))]]
    evidence = [
        {
            "evidence_type": EvidenceType.CODE.value,
            "content": (
                f"{item['symbol_type']} {item['symbol']} "
                f"({item['file']}:{item['line_start']}-{item['line_end']})\n{item['snippet']}"
            ),
            "source_uri": item["source_uri"],
            "confidence": 0.58,
            "provenance": {
                "symbol": item["symbol"],
                "line_start": item["line_start"],
                "line_end": item["line_end"],
                "static_only": True,
                "gap": args.get("gap"),
                "attempt_index": args.get("attempt_index"),
            },
        }
        for item in selected
    ]
    return ToolResult(ok=True, output={"finding_count": len(selected), "static_only": True}, evidence=evidence)


def _experiment_handler(
    context: ToolContext,
    args: Mapping[str, Any],
    *,
    experiment_runner: SandboxExperimentRunner | None = None,
) -> ToolResult:
    payload = dict(args)
    claim_id = payload.pop("claim_id", None)
    gap = payload.pop("gap", None)
    attempt_index = payload.pop("attempt_index", None)
    nested = payload.pop("protocol", None)
    if isinstance(nested, Mapping):
        nested_payload = dict(nested)
        nested_payload.update({key: value for key, value in payload.items() if key not in nested_payload})
        payload = nested_payload
    source = Path(str(payload.get("source_dir", ""))).expanduser().resolve()
    output = _safe_output_dir(context, payload.get("output_dir"), "experiments")
    payload["source_dir"] = str(source)
    payload["output_dir"] = str(output)
    protocol = ExperimentProtocol.from_dict(payload)
    runner = experiment_runner
    if runner is None or isinstance(runner, SandboxExperimentRunner):
        runner = SandboxExperimentRunner(cancel_requested=context.cancelled)
    result = runner.run(protocol)
    confidence = 0.45
    evidence = [
        {
            "claim_id": claim_id,
            # Execution output is auditable but cannot close the claim until the
            # preregistered protocol and acceptance criteria are evaluated.
            "evidence_type": EvidenceType.METADATA.value,
            "content": json.dumps(
                {
                    "experiment_id": result.experiment_id,
                    "status": result.status,
                    "metrics": result.metrics,
                    "criteria": result.criteria,
                    "accepted": result.accepted,
                    "backend": result.backend,
                    "network_isolated": result.network_isolated,
                    "source_unchanged": result.source_unchanged,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "source_uri": result.result_path,
            "artifact_path": result.result_path,
            "confidence": confidence,
            "polarity": EvidencePolarity.NEUTRAL.value,
            "provenance": {
                "gap": gap,
                "attempt_index": attempt_index,
                "experiment_id": result.experiment_id,
                "protocol_hash": result.protocol_hash,
                "source_hash_before": result.source_hash_before,
                "source_hash_after": result.source_hash_after,
                "source_unchanged": result.source_unchanged,
                "backend": result.backend,
                "network_isolated": result.network_isolated,
                "preregistration_path": result.preregistration_path,
                "execution_pending_evaluation": True,
            },
        }
    ]
    return ToolResult(
        ok=result.status == "COMPLETED",
        output={
            **result.to_dict(),
            "protocol": protocol.to_dict(),
            "protocol_hash": protocol.fingerprint,
        },
        evidence=evidence,
        artifacts=result.artifacts,
        error=result.error,
        retryable=False,
    )


def _experiment_plan_handler(context: ToolContext, args: Mapping[str, Any]) -> ToolResult:
    spec = dict(args.get("spec") or {})
    source_dir = str(args.get("source_dir") or "").strip()
    command = args.get("command")
    if not source_dir or not isinstance(command, list) or not command:
        return ToolResult(
            ok=True,
            output={
                "ready_for_execution": False,
                "reason": "A server-approved source directory and argv command are required.",
                "spec": spec,
            },
        )
    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("Approved experiment source directory does not exist")
    limits = spec.get("resource_limits") if isinstance(spec.get("resource_limits"), Mapping) else {}
    seeds = spec.get('seeds', [])
    if (not isinstance(seeds, list) or any(type(value) is not int for value in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('Seeds must be a list of distinct integers; invalid values cannot be silently dropped')
    bound_seed = '__PAPERPILOT_SEED__' in command[1:]
    if len(seeds) > 1 and not bound_seed:
        return ToolResult(ok=True, output={
            'ready_for_execution': False, 'spec': spec,
            'reason': 'Multiple seeds require the server-approved argv to contain one standalone __PAPERPILOT_SEED__ argument; no command is silently rewritten.'})
    if bound_seed and not seeds:
        raise ValueError('A seed-bound command requires at least one seed')
    criteria = spec.get("acceptance_criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("A preregistered experiment requires acceptance_criteria")
    metric_files = spec.get("metric_files")
    if not isinstance(metric_files, list) or not metric_files:
        metric_files = ["metrics.json"]
    policy = SandboxPolicy(
        backend="docker",
        docker_image=str(args.get('docker_image') or 'python:3.11-slim'),
        timeout_seconds=max(30, min(3_600, int(limits.get(
            "timeout_seconds", 480 // len(seeds) if bound_seed else 600)))),
        cpu_seconds=max(30, min(3_600, int(limits.get("cpu_seconds", 600)))),
        memory_mb=max(256, min(16_384, int(limits.get("memory_mb", 2_048)))),
        max_output_mb=max(1, min(256, int(limits.get("max_output_mb", 64)))),
        network="none",
        allow_local_fallback=False,
    )
    output = _safe_output_dir(context, args.get("output_dir"), "experiments")
    protocol = ExperimentProtocol(
        name=str(spec.get("name") or spec.get("hypothesis") or "paperpilot-experiment")[:200],
        command=[str(item) for item in command],
        source_dir=str(source),
        output_dir=str(output),
        metric_files=[str(item) for item in metric_files],
        acceptance_criteria=[dict(item) for item in criteria if isinstance(item, Mapping)],
        hypothesis=str(spec.get("hypothesis") or "")[:2_000],
        seed=seeds[0] if seeds and not bound_seed else None,
        seed_sweep=list(seeds) if bound_seed else [],
        study_design=json.loads(json.dumps({key: spec[key] for key in (
            'baseline', 'variants', 'dataset', 'data_split', 'controlled_variables', 'metrics', 'seeds')
            if key in spec}, allow_nan=False)),
        policy=policy,
    )
    if protocol.seed_sweep:
        from .seed_sweep import validate_seed_sweep

        validate_seed_sweep(protocol)
    preregistration = output / f"protocol-{protocol.fingerprint.split(':')[-1][:16]}.json"
    preregistration.write_text(
        json.dumps(protocol.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return ToolResult(
        ok=True,
        output={
            "ready_for_execution": True,
            "protocol": protocol.to_dict(),
            "protocol_hash": protocol.fingerprint,
            "preregistration_path": str(preregistration),
            "execution_scope": (
                "seed_sweep; all_seeds_must_meet_criteria; reported_seed_checked; not_a_significance_test"
                if protocol.seed_sweep else
                "single_approved_command; design_fields_are_preregistered_not_execution_attestations"
            ),
        },
        artifacts=[
            {
                "kind": "experiment-preregistration",
                "path": str(preregistration),
                "sha256": _sha256(preregistration),
                "metadata": {"protocol_hash": protocol.fingerprint},
            }
        ],
    )


def _experiment_metric_summary(value: Any, depth: int = 0) -> Any:
    """Keep measured scalars visible; detailed rows remain in the result artifact.

    Large ranking arrays must not crowd later seeds or configuration out of the
    synthesis context. This is a presentation projection, never score mutation.
    """
    if isinstance(value, Mapping) and depth < 4:
        return {str(key): _experiment_metric_summary(item, depth + 1)
                for key, item in value.items()
                if isinstance(item, Mapping) or item is None or type(item) in {str, int, float, bool}}
    if type(value) is str:
        return value[:500]
    if value is None or type(value) in {int, float, bool}:
        return value
    return None


def _experiment_evaluate_handler(context: ToolContext, args: Mapping[str, Any]) -> ToolResult:
    del context
    protocol = ExperimentProtocol.from_dict(dict(args.get("protocol") or {}))
    result = dict(args.get("result") or {})
    protocol_hash = str(result.get("protocol_hash") or "")
    if protocol_hash != protocol.fingerprint:
        raise ValueError("Experiment result protocol fingerprint does not match preregistration")
    source_unchanged = bool(result.get("source_unchanged"))
    network_isolated = bool(result.get("network_isolated"))
    completed = str(result.get("status")) == "COMPLETED"
    accepted = result.get("accepted")
    criteria = result.get('criteria') or []
    evaluable = (type(accepted) is bool and isinstance(criteria, list) and bool(criteria)
                 and all(isinstance(row, Mapping) and not row.get('error')
                         and type(row.get('passed')) is bool for row in criteria))
    if not evaluable:
        accepted = None
    trustworthy = completed and source_unchanged and network_isolated
    if not trustworthy:
        polarity = EvidencePolarity.NEUTRAL
        reason = "Experiment result is inadmissible because an integrity or isolation boundary failed."
    elif not evaluable:
        polarity = EvidencePolarity.NEUTRAL
        reason = "Experiment is inconclusive: preregistered criteria could not be evaluated; this does not refute the hypothesis."
    elif accepted:
        polarity = EvidencePolarity.SUPPORTS
        reason = "Preregistered criteria passed in an isolated sandbox."
    else:
        polarity = EvidencePolarity.CONTRADICTS
        reason = "Preregistered criteria did not pass in the isolated sandbox."
    return ToolResult(
        ok=True,
        output={
            "protocol_hash": protocol_hash,
            "accepted": accepted,
            "trustworthy": trustworthy,
            "evaluable": evaluable,
            "polarity": polarity.value,
            "reason": reason,
        },
        evidence=[
            {
                "claim_id": args.get("claim_id"),
                "evidence_type": EvidenceType.EXPERIMENT.value,
                "content": json.dumps(
                    {
                        "metrics": _experiment_metric_summary(result.get("metrics") or {}),
                        "detail_artifact": result.get("result_path"),
                        "criteria": result.get("criteria") or [],
                        "accepted": accepted,
                        "trustworthy": trustworthy,
                        "reason": reason,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "source_uri": result.get("result_path"),
                "artifact_path": result.get("result_path"),
                "confidence": 0.95 if trustworthy else 0.4,
                "polarity": polarity.value,
                "provenance": {
                    "gap": args.get("gap"),
                    "protocol_hash": protocol_hash,
                    "source_hash_before": result.get("source_hash_before"),
                    "source_hash_after": result.get("source_hash_after"),
                    "source_unchanged": source_unchanged,
                    "network_isolated": network_isolated,
                    "evaluation": "preregistered-deterministic-v2-inconclusive",
                },
            }
        ],
    )


_TITLE_STOPWORDS = {
    "a", "an", "and", "for", "in", "of", "on", "the", "to", "via", "with",
    "approach", "method", "model", "network", "paper", "study",
}


def _title_match_score(candidate: str, expected: str) -> float:
    """Conservative title similarity for a named-paper scholarly fallback."""

    def tokens(value: str) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", value)
            if len(token) > 1 and token.casefold() not in _TITLE_STOPWORDS
        }

    candidate_tokens = tokens(candidate)
    expected_tokens = tokens(expected)
    if not candidate_tokens or not expected_tokens:
        return 0.0
    folded_candidate = re.sub(r"\W+", "", candidate).casefold()
    folded_expected = re.sub(r"\W+", "", expected).casefold()
    if folded_candidate == folded_expected:
        return 1.0
    return len(candidate_tokens & expected_tokens) / max(
        1,
        min(len(candidate_tokens), len(expected_tokens)),
    )


def _filter_named_scholarly_items(
    items: list[Mapping[str, Any]],
    expected_titles: list[str],
) -> tuple[list[dict[str, Any]], int]:
    if not expected_titles:
        return [dict(item) for item in items], 0
    kept: list[dict[str, Any]] = []
    for raw in items:
        item = dict(raw)
        title = str(item.get("title") or "").strip()
        score = max((_title_match_score(title, expected) for expected in expected_titles), default=0.0)
        if score < 0.55:
            continue
        item["expected_title_match_score"] = round(score, 6)
        kept.append(item)
    return kept, len(items) - len(kept)


def _scholar_search_handler(adapter: ScholarlySearchAdapter):
    def handler(context: ToolContext, args: Mapping[str, Any]) -> ToolResult:
        del context
        query = str(args.get("query") or "").strip()
        options = {k: args[k] for k in ('published_from', 'published_to', 'sort', 'require_full_text') if k in args}
        result = adapter.search(query, top_k=max(1, min(20, int(args.get("top_k", 8)))), **options)
        expected_titles = [
            str(value).strip()
            for value in args.get("expected_titles", [])
            if str(value).strip()
        ][:8]
        candidates, filtered_out = _filter_named_scholarly_items(
            list(result.get("items", [])),
            expected_titles,
        )
        evidence: list[dict[str, Any]] = []
        for rank, item in enumerate(candidates, start=1):
            title = str(item.get("title") or "").strip()
            abstract = str(item.get("abstract") or "").strip()
            if not title:
                continue
            evidence.append(
                {
                    "claim_id": args.get("claim_id"),
                    "evidence_type": EvidenceType.METADATA.value,
                    "content": f"{title}\n{abstract}".strip(),
                    "source_uri": item.get("landing_url"),
                    "confidence": 0.35,
                    "polarity": EvidencePolarity.NEUTRAL.value,
                    "provenance": {
                        "gap": args.get("gap"),
                        "query": query,
                        "rank": rank,
                        "provider": item.get("provider"),
                        "source_id": item.get("source_id"),
                        "doi": item.get("doi"),
                        "title": title,
                        "full_text_url": item.get("full_text_url"),
                        "discovery_only": True,
                        "publication_date": item.get('publication_date'),
                        "date_basis": item.get('date_basis'),
                        "search_scope": result.get('search_scope'),
                    },
                }
            )
        return ToolResult(
            ok=True,
            output={
                "query": query,
                "providers": result.get("providers", []),
                "errors": result.get("errors", []),
                "cache_hit": result.get("cache_hit", False),
                "candidates": candidates,
                "expected_titles": expected_titles,
                "title_mismatch_filtered": filtered_out,
                "search_scope": result.get('search_scope'),
                "rejected_items": result.get('rejected_items', []),
            },
            evidence=evidence,
        )

    return handler


def _scholar_fetch_handler(adapter: ScholarlySearchAdapter):
    def handler(context: ToolContext, args: Mapping[str, Any]) -> ToolResult:
        del context
        result = adapter.fetch_full_text(str(args.get("url") or ""))
        evidence = [
            {
                "claim_id": args.get("claim_id"),
                "evidence_type": EvidenceType.TEXT.value,
                "content": str(page["content"]),
                "source_uri": result["url"],
                "page": page["page"],
                "confidence": 0.7,
                "polarity": EvidencePolarity.NEUTRAL.value,
                "provenance": {
                    "gap": args.get("gap"),
                    "source_id": args.get("source_id"),
                    "title": args.get("title"),
                    "document_sha256": result["sha256"],
                    "paper_sha256": result["sha256"],
                    "external_full_text": True,
                },
            }
            for page in result["pages"]
        ]
        return ToolResult(
            ok=True,
            output={
                "url": result["url"],
                "document_sha256": result["sha256"],
                "page_chunk_count": len(evidence),
            },
            evidence=evidence,
        )

    return handler


def _finalize_handler(context: ToolContext, args: Mapping[str, Any]) -> ToolResult:
    return ToolResult(ok=True, output={"claim_id": args.get("claim_id"), "reason": args.get("reason")})


def register_builtin_tools(
    registry: ToolRegistry,
    *,
    workspace: str | Path,
    search_adapter: PaperSearchAdapter | None = None,
    scholarly_adapter: ScholarlySearchAdapter | None = None,
    vision_adapter: VisionEvidenceAdapter | None = None,
    experiment_runner: SandboxExperimentRunner | None = None,
) -> None:
    def pilot_navigation(context, args, *, read=False):
        from .paper_structure import PaperTextNavigator
        metadata = context.db.get_run(context.run_id)['metadata']
        if metadata.get('retrieval_profile') != 'structured_pilot':
            raise ValueError('Literal paper navigation requires the opt-in structured pilot')
        from .structured_retrieval import pilot_identity
        if pilot_identity(context.workspace/'structured-pilot') != metadata.get('retrieval_identity'):
            raise ValueError('Frozen pilot identity changed')
        sha = args['paper_sha256']
        scope = metadata.get('resolved_paper_sha256s') or metadata.get('paper_sha256s')
        if scope and sha not in scope:
            raise ValueError('Navigation escaped requested paper scope')
        navigator = PaperTextNavigator(context.workspace/'structured-pilot')
        if read:
            return navigator.read(sha, args['paragraph_ids'])
        return navigator.find_text(sha, args['query'])

    registry.register(ToolSpec(name='paper.find_text',
        description='Literal search of registered paper text. Matches are navigation, not verified facts.',
        handler=lambda context, args: pilot_navigation(context, args),
        input_schema={'type': 'object', 'properties': {'paper_sha256': {'type': 'string'},
            'query': {'type': 'string'}}, 'required': ['paper_sha256', 'query'], 'additionalProperties': False}))
    registry.register(ToolSpec(name='paper.read_passages',
        description='Read bounded originals by registered paragraph IDs; source locations and omissions retained.',
        handler=lambda context, args: pilot_navigation(context, args, read=True),
        input_schema={'type': 'object', 'properties': {'paper_sha256': {'type': 'string'},
            'paragraph_ids': {'type': 'array', 'items': {'type': 'string'}}},
            'required': ['paper_sha256', 'paragraph_ids'], 'additionalProperties': False}))
    from .reference_methods import register_reference_tools
    register_reference_tools(registry)
    adapter = search_adapter or make_search_adapter(workspace)
    from .paper_evidence_tools import register_paper_evidence_tools
    register_paper_evidence_tools(registry, adapter, _adapter_for_run, _token_paper_scope)
    def read_token_context(context, args):
        metadata = context.db.get_run(context.run_id)['metadata']
        if metadata.get('retrieval_profile') != 'token_evidence_pilot':
            raise ValueError('Parent context requires the opt-in token evidence profile')
        selected = _adapter_for_run(context, adapter)
        result = selected.read_context(args['child_ids'], max_tokens=args.get('max_tokens', 2048),
                                       paper_sha256s=_token_paper_scope(context), view=args.get('view', 'full'))
        return ToolResult(ok=True, output={k: v for k, v in result.items() if k != 'evidence'},
                          evidence=result['evidence'])
    registry.register(ToolSpec(name='paper.read_context',
        description='Read complete original parent paragraphs for owned search child IDs, with a token cap and explicit omissions. Optional source_handles view keeps source URIs/child IDs while full locators remain readable with view=full. Requires token_evidence_pilot.',
        handler=read_token_context,
        input_schema={'type': 'object', 'properties': {'child_ids': {'type': 'array', 'items': {'type': 'string'}},
            'max_tokens': {'type': 'integer'}, 'view': {'type': 'string', 'enum': ['full', 'source_handles']}},
            'required': ['child_ids'], 'additionalProperties': False}))
    from .local_paper_pages import read_pages
    def read_registered_pages(context, args):
        scope = _token_paper_scope(context)
        if scope is not None and args['paper_sha256'] not in scope:
            raise ValueError('Page read escaped requested paper scope')
        return read_pages(_adapter_for_run(context, adapter), args)
    registry.register(ToolSpec(name='paper.read_pages',
        description='Read 1..3 pages from a SHA-verified registered local PDF in configured corpus roots.',
        handler=read_registered_pages,
        input_schema={'type': 'object', 'properties': {'paper_sha256': {'type': 'string'},
            'pages': {'type': 'array', 'items': {'type': 'integer'}}},
            'required': ['paper_sha256', 'pages'], 'additionalProperties': False}))
    academic = scholarly_adapter or ScholarlySearchAdapter(
        Path(workspace) / ".paperpilot_agentic" / "scholarly-cache"
    )
    effective_vision_adapter = vision_adapter or make_vision_adapter()
    registry.register(
        ToolSpec(
            name="paper.search",
            description="Search PaperPilot's evidence corpus through the configured retriever adapter.",
            handler=_search_handler(adapter),
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 6,
                    },
                    "top_k": {"type": "integer"},
                    "claim_id": {"type": "string"},
                    "gap": {"type": "string"},
                    "attempt_index": {"type": "integer"},
                    "paper_sha256s": {
                        "type": "array",
                        "items": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
                    },
                },
            },
            risk_level=RiskLevel.READ_ONLY,
            max_attempts=2,
            tags=("paper", "retrieval", "evidence"),
        )
    )
    registry.register(
        ToolSpec(
            name="paper.inspect_layout",
            description="Parse a PDF into page/bounding-box objects and retrieve text, table, figure or equation evidence.",
            handler=_layout_handler(effective_vision_adapter),
            input_schema={
                "type": "object",
                "required": ["pdf_path"],
                "properties": {
                    "pdf_path": {"type": "string"},
                    "object_type": {"type": "string", "enum": ["text", "table", "figure", "equation"]},
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                    "claim_id": {"type": "string"},
                    "gap": {"type": "string"},
                    "attempt_index": {"type": "integer"},
                },
            },
            risk_level=RiskLevel.READ_ONLY,
            max_attempts=2,
            tags=("pdf", "layout", "multimodal"),
        )
    )
    registry.register(
        ToolSpec(
            name="code.inspect",
            description="Perform read-only AST inspection of repository code; no project code is imported or executed.",
            handler=_code_handler,
            input_schema={
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string"},
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                    "claim_id": {"type": "string"},
                    "gap": {"type": "string"},
                    "attempt_index": {"type": "integer"},
                },
            },
            risk_level=RiskLevel.READ_ONLY,
            max_attempts=2,
            tags=("code", "static-analysis"),
        )
    )
    registry.register(
        ToolSpec(
            name="scholar.search",
            description="Search traceable scholarly APIs after local evidence is insufficient; metadata remains discovery-only.",
            handler=_scholar_search_handler(academic),
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                    "published_from": {"type": "string"},
                    "published_to": {"type": "string"},
                    "sort": {"type": "string", "enum": ["relevance", "newest"]},
                    "require_full_text": {"type": "boolean"},
                    "expected_titles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 8,
                    },
                    "claim_id": {"type": "string"},
                    "gap": {"type": "string"},
                    "attempt_index": {"type": "integer"},
                },
            },
            risk_level=RiskLevel.READ_ONLY,
            max_attempts=2,
            tags=("scholarly", "metadata", "discovery"),
        )
    )
    registry.register(
        ToolSpec(
            name="scholar.fetch",
            description="Fetch and parse PDF full text from an allowlisted scholarly HTTPS host.",
            handler=_scholar_fetch_handler(academic),
            input_schema={
                "type": "object",
                "required": ["url"],
                "properties": {
                    "url": {"type": "string"},
                    "source_id": {"type": ["string", "null"]},
                    "title": {"type": ["string", "null"]},
                    "claim_id": {"type": "string"},
                    "gap": {"type": "string"},
                    "attempt_index": {"type": "integer"},
                },
            },
            risk_level=RiskLevel.READ_ONLY,
            max_attempts=2,
            tags=("scholarly", "full-text", "evidence"),
        )
    )
    registry.register(
        ToolSpec(
            name="experiment.plan",
            description="Compile a hypothesis into a server-bounded preregistered Docker experiment protocol.",
            handler=_experiment_plan_handler,
            input_schema={
                "type": "object",
                "required": ["spec"],
                "properties": {
                    "spec": {"type": "object"},
                    "source_dir": {"type": ["string", "null"]},
                    "command": {"type": ["array", "null"]},
                    "output_dir": {"type": ["string", "null"]},
                    "dataset": {},
                    "docker_image": {"type": ["string", "null"]},
                    "claim_id": {"type": "string"},
                    "gap": {"type": "string"},
                    "attempt_index": {"type": "integer"},
                },
            },
            risk_level=RiskLevel.READ_ONLY,
            max_attempts=1,
            tags=("experiment", "preregistration", "read-only"),
        )
    )
    registry.register(
        ToolSpec(
            name="experiment.run",
            description="Execute a preregistered experiment in an isolated copied workspace with resource limits.",
            handler=lambda context, args: _experiment_handler(
                context,
                args,
                experiment_runner=experiment_runner,
            ),
            input_schema={
                "type": "object",
                "required": ["name", "command", "source_dir"],
                "properties": {
                    "name": {"type": "string"},
                    "command": {"type": "array"},
                    "source_dir": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "metric_files": {"type": "array"},
                    "acceptance_criteria": {"type": "array"},
                    "hypothesis": {"type": "string"},
                    "seed": {"type": ["integer", "null"]},
                    "study_design": {"type": "object"},
                    "seed_sweep": {"type": "array", "items": {"type": "integer"}},
                    "environment": {"type": "object"},
                    "policy": {"type": "object"},
                    "claim_id": {"type": "string"},
                    "gap": {"type": "string"},
                    "attempt_index": {"type": "integer"},
                    "protocol": {"type": "object"},
                },
            },
            risk_level=RiskLevel.HIGH,
            side_effects=True,
            requires_approval=True,
            timeout_seconds=600,
            max_attempts=2,
            tags=("experiment", "sandbox", "hitl"),
        )
    )
    registry.register(
        ToolSpec(
            name="experiment.evaluate",
            description="Evaluate sandbox results against the exact preregistered protocol fingerprint.",
            handler=_experiment_evaluate_handler,
            input_schema={
                "type": "object",
                "required": ["protocol", "result"],
                "properties": {
                    "protocol": {"type": "object"},
                    "result": {"type": "object"},
                    "claim_id": {"type": "string"},
                    "gap": {"type": "string"},
                    "attempt_index": {"type": "integer"},
                },
            },
            risk_level=RiskLevel.READ_ONLY,
            max_attempts=1,
            tags=("experiment", "evaluation", "evidence"),
        )
    )
    registry.register(
        ToolSpec(
            name="claim.finalize",
            description="Finalize a claim after its evidence gaps are closed or exhausted.",
            handler=_finalize_handler,
            input_schema={
                "type": "object",
                "required": ["claim_id"],
                "properties": {
                    "claim_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "attempt_index": {"type": "integer"},
                },
            },
            risk_level=RiskLevel.READ_ONLY,
            max_attempts=2,
            tags=("claim", "control"),
        )
    )
