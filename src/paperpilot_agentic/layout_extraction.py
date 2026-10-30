"""Persistent layout extraction behind one tool interface.

Identity/selection, cached revalidation, and provider dispatch are private steps.
The legacy tool payload and billable attempt hooks remain compatible.
"""
from __future__ import annotations
import inspect
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from .models import Budget, EvidenceType, ToolResult
from .llm_runtime import LLMBudgetExhausted
from .layout_identity import _prioritize_visual_candidates, _explicit_layout_label, _matches_explicit_layout_label
from .source_references import PLACEHOLDER
from .table_context import nearby_caption
from .structured_extraction import SourceRegion, annotate_observation, config_sha

@dataclass
class ExtractionStats:
    vision_api_calls: int = 0
    vision_prompt_tokens: int = 0
    vision_completion_tokens: int = 0
    vision_skipped: int = 0
    vision_cache_hits: int = 0
    vision_cross_run_cache_hits: int = 0
    vision_errors: list[str] = field(default_factory=list)
    local_parse_attempts: int = 0
    local_cache_hits: int = 0
    local_duration_ms: float = 0
    local_sdk_requests: int = 0
    local_sdk_requests_unknown: int = 0
    reviewed_source_hits: int = 0

@dataclass
class RegionRead:
    item: Mapping[str, Any]
    evidence: dict[str, Any]
    image: str
    prompt: str
    metadata: dict[str, Any]

@dataclass
class VisualReadSession:
    context: Any
    args: Mapping[str, Any]
    stats: ExtractionStats
    limits: tuple[int, int, int]
    prior_calls: int
    prior_by_artifact: dict
    prior_documents: dict
    current_documents: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionRequest:
    context: Any
    arguments: Mapping[str, Any]

class LayoutExtraction:
    def __init__(self, adapter, *, layout_parser, layout_search, output_dir, file_hash):
        self.adapter = adapter
        self.parse_layout = layout_parser
        self.search_layout = layout_search
        self.output_dir = output_dir
        self.file_hash = file_hash

    def extract(self, request: ExtractionRequest) -> ToolResult:
        context, args = request.context, request.arguments
        selection = self._prepare(context, args)
        if isinstance(selection, ToolResult):
            return selection
        pdf, manifest_path, manifest, object_type, explicit_label, unresolved_reference, selected, evidence = selection
        stats = ExtractionStats()
        session = self._session(context, args, stats)
        self._read_regions(session, selected, evidence)
        for item in evidence:
            if item['object_type'] not in {'table', 'equation', 'figure'} or not item.get('artifact_path'):
                continue
            try:
                annotate_observation(item, region=self._source_region(item),
                    parser=type(self.adapter).__name__ if self.adapter else 'layout')
            except (ValueError, OSError) as exc:
                item['provenance']['structured_observation_error'] = str(exc)
                item['provenance']['vision_verified'] = False
        vision_adapter = self.adapter
        per_tool_limit, per_run_limit, per_document_limit = session.limits
        prior_vision_calls = session.prior_calls
        visual_candidates = _prioritize_visual_candidates(selected, evidence)
        artifacts = [
            {
                "kind": "layout_manifest",
                "path": str(manifest_path.resolve()),
                "sha256": self.file_hash(manifest_path),
                "metadata": {"pdf_path": str(pdf), "object_type": object_type},
            }
        ]
        artifacts.extend(
            {
                "kind": f"layout_{item.get('object_type', 'object')}",
                "path": item["artifact_path"],
                "metadata": {"object_id": item.get("object_id"), "page": item.get("page")},
            }
            for item in selected
            if item.get("artifact_path")
        )
        return ToolResult(
            ok=True,
            output={
                "manifest_path": str(manifest_path.resolve()),
                "selected_count": len(selected),
                "layout_target_status": (
                    "MATCHED" if selected else "NOT_FOUND"
                ) if explicit_label else ("UNRESOLVED_REFERENCE" if unresolved_reference else "SEMANTIC_SEARCH"),
                "parser": manifest.parser if hasattr(manifest, "parser") else manifest.get("parser"),
                "warnings": manifest.warnings if hasattr(manifest, "warnings") else manifest.get("warnings", []),
                "vision_adapter": type(vision_adapter).__name__ if vision_adapter else None,
                "vision_api_calls": stats.vision_api_calls,
                "vision_prompt_tokens": stats.vision_prompt_tokens,
                "vision_completion_tokens": stats.vision_completion_tokens,
                "vision_skipped": stats.vision_skipped,
                "vision_cache_hits": stats.vision_cache_hits,
                "vision_cross_run_cache_hits": stats.vision_cross_run_cache_hits,
                "vision_target_label": (
                    list(explicit_label) if vision_adapter is not None and explicit_label else None
                ),
                "vision_candidate_count": (
                    len(visual_candidates) if vision_adapter is not None else 0
                ),
                "vision_errors": stats.vision_errors,
                "local_parse_attempts": stats.local_parse_attempts,
                "local_cache_hits": stats.local_cache_hits,
                "local_duration_ms": stats.local_duration_ms,
                "local_sdk_requests": stats.local_sdk_requests,
                "local_sdk_requests_unknown": stats.local_sdk_requests_unknown,
                "reviewed_source_hits": stats.reviewed_source_hits,
                "vision_call_limits": {
                    "per_tool": per_tool_limit,
                    "per_run": per_run_limit,
                    "per_document": per_document_limit,
                    "prior_run_calls": prior_vision_calls,
                },
            },
            evidence=evidence,
            artifacts=artifacts,
        )

    def _prepare(self, context, args):
        pdf = Path(str(args["pdf_path"])).expanduser().resolve()
        if not pdf.is_file():
            return ToolResult(ok=False, error=f"PDF not found: {pdf}")
        output = self.output_dir(context, None, f"layout/{pdf.stem}-{self.file_hash(pdf)[:12]}")
        manifest_path = output / "layout_manifest.json"
        if manifest_path.is_file():
            manifest: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = self.parse_layout(pdf, output, render_regions=True)
        object_type = str(args.get("object_type") or "") or None
        query = str(args.get("query") or "")
        explicit_label = _explicit_layout_label(query, object_type)
        unresolved_reference = explicit_label is None and bool(PLACEHOLDER.search(query))
        manifest_data = manifest.to_dict() if hasattr(manifest, 'to_dict') else manifest
        top_k = max(1, int(args.get("top_k", 8)))
        selected = self.search_layout(
            manifest,
            query=query,
            object_type=object_type,
            # Exact identity is a constraint, not a ranking boost. Searching all
            # local objects avoids losing the target before applying top_k.
            top_k=len(manifest_data.get('objects', [])) if explicit_label else top_k,
        )
        if explicit_label is not None:
            selected = [
                item
                for item in selected
                if _matches_explicit_layout_label(item, explicit_label)
            ][:top_k]
        elif unresolved_reference:
            # A placeholder-only request has no physical table identity. Do not
            # spend a visual call on a semantic neighbour and pretend it maps.
            selected = []
        evidence: list[dict[str, Any]] = []
        for item in selected:
            kind = str(item.get("object_type", "text"))
            evidence_type = {
                "table": EvidenceType.TABLE,
                "figure": EvidenceType.FIGURE,
                "equation": EvidenceType.EQUATION,
            }.get(kind, EvidenceType.TEXT)
            content = str(item.get("text") or item.get("caption") or "").strip()
            retrieval_context = str(item.get("retrieval_context") or "").strip()
            if retrieval_context and retrieval_context not in content:
                content = f"{content}\nNearby extracted text: {retrieval_context}".strip()
            if not content:
                content = f"{kind} object on page {item.get('page')}"
            metadata = dict(item.get("metadata") or {})
            vision_verified = bool(metadata.get("vision_verified", False))
            caption_only = bool(metadata.get("caption_only", False))
            rows = int(metadata.get("rows") or 0)
            needs_visual_verification = (
                kind in {"figure", "equation"}
                or caption_only
                or (kind == "table" and rows <= 0)
            )
            unverified_visual = needs_visual_verification and not vision_verified
            document_sha = item.get("document_sha256") or self.file_hash(pdf)
            object_id = item.get("object_id") or f"p{item.get('page')}-{kind}"
            evidence.append(
                {
                    "evidence_type": evidence_type.value,
                    "content": content,
                    "source_uri": f"paper://{document_sha}/layout/{object_id}",
                    "page": item.get("page"),
                    "bbox": item.get("bbox"),
                    "object_type": kind,
                    "artifact_path": item.get("artifact_path"),
                    "confidence": 0.45 if unverified_visual else (0.82 if kind == "table" else 0.7),
                    "provenance": {
                        "document_sha256": item.get("document_sha256"),
                        "layout_object_id": item.get("object_id"),
                        "layout_source_label": _explicit_layout_label(str(item.get('caption') or item.get('text') or ''), kind),
                        "layout_target_binding": ({
                            "label": list(explicit_label),
                            "object_id": object_id,
                            "document_sha256": document_sha,
                            "basis": "own_caption_or_heading",
                        } if explicit_label else None),
                        "section_path": item.get("section_path"),
                        "parser": item.get("parser"),
                        "render_bbox": metadata.get('render_context_bbox') or item.get('bbox'),
                        "retrieval_score": item.get("retrieval_score"),
                        "gap": args.get("gap"),
                        "attempt_index": args.get("attempt_index"),
                        "analysis_mode": "caption_context_only" if unverified_visual else "layout_text_extraction",
                        "vision_verified": vision_verified,
                        "needs_visual_verification": needs_visual_verification,
                        "caption_only": caption_only,
                        "nearby_caption_text": nearby_caption(manifest, item),
                        "retrieval_context_object_ids": item.get("retrieval_context_object_ids") or [],
                        "retrieval_context_bboxes": item.get("retrieval_context_bboxes") or [],
                    },
                }
            )


        return pdf, manifest_path, manifest, object_type, explicit_label, unresolved_reference, selected, evidence

    def _session(self, context, args, stats):
        per_tool_limit = max(
            0, min(8, int(os.getenv("PAPERPILOT_MAX_VISION_REGIONS_PER_TOOL", "2")))
        )
        per_run_limit = max(
            0, min(40, int(os.getenv("PAPERPILOT_MAX_VISION_CALLS_PER_RUN", "6")))
        )
        per_document_limit = max(
            0, min(8, int(os.getenv("PAPERPILOT_MAX_VISION_REGIONS_PER_DOCUMENT", "2")))
        )
        prior_vision_calls = 0
        prior_vision_by_artifact: dict[str, Mapping[str, Any]] = {}
        prior_document_artifacts: dict[str, set[str]] = {}
        for prior_call in context.db.list_tool_calls(context.run_id):
            result = prior_call.get("result") or {}
            output = result.get("output") or {}
            prior_vision_calls += int(output.get("vision_api_calls") or 0)
            for prior_item in result.get("evidence") or []:
                prior_provenance = dict(prior_item.get("provenance") or {})
                prior_artifact = str(prior_item.get("artifact_path") or "").strip()
                if not prior_artifact or not prior_provenance.get("vision_adapter"):
                    continue
                artifact_key = str(Path(prior_artifact).expanduser().resolve())
                prior_vision_by_artifact.setdefault(artifact_key, prior_item)
                document_key = str(
                    prior_provenance.get("document_sha256") or ""
                ).strip()
                if document_key:
                    prior_document_artifacts.setdefault(document_key, set()).add(
                        artifact_key
                    )


        return VisualReadSession(context, args, stats, (per_tool_limit, per_run_limit, per_document_limit), prior_vision_calls, prior_vision_by_artifact, prior_document_artifacts)

    def _read_regions(self, session, selected, evidence):
        from .reviewed_sources import ReviewedSources
        registry = ReviewedSources.configured()
        reviewed = set()
        for item in evidence:
            if item.get('object_type') != 'table' or not item.get('artifact_path'):
                continue
            try:
                source = self._source_region(item)
                if registry.resolve(source) is not None:
                    if source.paper_sha256 != self.file_hash(Path(session.args['pdf_path'])):
                        raise ValueError('Source PDF changed before reviewed asset read')
                    registry.apply(item, source)
                    reviewed.add(id(item))
                    session.stats.reviewed_source_hits += 1
            except (ValueError, OSError) as exc:
                item['provenance']['reviewed_source_error'] = str(exc)
                # Conflicting assets must not silently become a cloud request.
                reviewed.add(id(item))
                item['provenance']['vision_verified'] = False
        context, vision_adapter = session.context, self.adapter
        prior_vision_calls = session.prior_calls
        prior_document_artifacts = session.prior_documents
        current_document_artifacts = session.current_documents
        per_tool_limit, per_run_limit, per_document_limit = session.limits
        query = str(session.args.get("query") or "")
        if vision_adapter is not None:
            visual_candidates = _prioritize_visual_candidates(selected, evidence)
            for item, evidence_item in visual_candidates:
                if id(evidence_item) in reviewed:
                    continue
                artifact_path = item.get("artifact_path")
                if not artifact_path or item.get("object_type") not in {"figure", "table", "equation"}:
                    continue
                # For numeric-result questions, use the configured visual
                # adapter as a second reader even when the PDF parser produced
                # rows. Ablation tables often encode checkmarks or row grouping
                # as glyphs that text extraction silently loses.
                visual_requested = bool(
                    evidence_item["provenance"].get("needs_visual_verification")
                    or (
                        str(session.args.get("gap") or "") == "numeric_results"
                        and item.get("object_type") == "table"
                    )
                )
                if not visual_requested:
                    continue
                artifact_key = str(Path(str(artifact_path)).expanduser().resolve())
                vision_prompt = (
                    "Inspect this scholarly-paper region as evidence for the claim. "
                    "Describe only visible content, transcribe relevant values, identify ambiguity, "
                    f"and assess this query: {query}"
                )
                original_question = context.db.get_run(context.run_id).get('question')
                if original_question:
                    vision_prompt += (
                        '\nOriginal user question (extraction scope, not source evidence): '
                        + str(original_question)
                        + '\nPreserve separate leaf condition columns needed for this question; '
                        'never collapse them into their spanning group heading.')
                vision_context = {
                    "object_id": item.get("object_id"),
                    "object_type": item.get("object_type"),
                    "page": item.get("page"),
                    "caption": item.get("caption"),
                    "section_path": item.get("section_path"),
                    "run_id": context.run_id,
                }
                region = RegionRead(item, evidence_item, artifact_key, vision_prompt, vision_context)
                if getattr(self.adapter, 'local_execution', False):
                    try:
                        region.metadata.update(pdf_path=str(Path(session.args['pdf_path']).resolve()),
                            source_region=self._source_region(evidence_item).model_dump(mode='json'))
                    except (ValueError, OSError) as exc:
                        evidence_item['provenance'].update(vision_verified=False,
                            analysis_mode='local_source_failed', vision_error=str(exc))
                        session.stats.vision_errors.append(str(exc))
                        continue
                if self._reuse(session, region):
                    continue
                document_key = str(evidence_item["provenance"].get("document_sha256") or "").strip()
                prior_document_count = len(
                    prior_document_artifacts.get(document_key, set())
                )
                current_document_count = len(
                    current_document_artifacts.get(document_key, set())
                )
                if (
                    session.stats.vision_api_calls + session.stats.local_parse_attempts + session.stats.local_cache_hits >= per_tool_limit
                    or prior_vision_calls + session.stats.vision_api_calls >= per_run_limit
                    or prior_document_count + current_document_count >= per_document_limit
                ):
                    session.stats.vision_skipped += 1
                    evidence_item["provenance"].update(
                        {
                            "vision_verified": False,
                            "vision_skip_reason": (
                                "bounded_document_visual_budget"
                                if prior_document_count + current_document_count
                                >= per_document_limit
                                else "bounded_visual_call_budget"
                            ),
                        }
                    )
                    continue
                self._analyze(session, region)

    def _reuse(self, session, region):
        if getattr(self.adapter, 'local_execution', False):
            return False  # Local raw cache is validated inside the adapter.
        context = session.context
        reaudit = getattr(self.adapter, 'reaudit', None)
        try:
            image_sha = self.file_hash(Path(region.image))
        except OSError:
            return False  # Normal dispatch records an unavailable source without accepting cache.
        candidates = []
        same = session.prior_by_artifact.get(region.image)
        if same is not None:
            candidates.append((same, 'same_run_exact_crop'))
        fresh = context.db.get_run(context.run_id)['metadata'].get('disable_cross_run_visual_cache') is True
        if callable(reaudit) and not fresh:
            candidates.extend((row, 'cross_run_exact_crop') for row in
                context.db.find_visual_evidence_by_image_sha(image_sha,
                    document_sha256=region.evidence['provenance'].get('document_sha256'),
                    exclude_run_id=context.run_id))
        for cached, scope in candidates:
            provenance = cached.get('provenance') or {}
            meta = dict(provenance.get('vision_metadata') or {})
            identity = provenance.get('parser_cache_identity')
            # New observations bind parser configuration and the complete source region.
            # Legacy Qwen records can be read only by the current offline re-auditor.
            if identity is not None and identity != self._cache_identity():
                continue
            if identity is None and (not callable(reaudit) or
                    provenance.get('vision_adapter') != type(self.adapter).__name__ or
                    (getattr(self.adapter, 'model', None) and meta.get('model') != self.adapter.model)):
                continue
            if meta.get('image_sha256') != image_sha:
                continue
            source = (provenance.get('structured_extraction') or {}).get('source')
            if source and source != self._source_region(region.evidence).model_dump(mode='json'):
                continue
            original = {'description': str(cached.get('content') or ''),
                'confidence': float(cached.get('confidence') or 0),
                'verified': bool(provenance.get('vision_verified')), 'metadata': meta}
            if callable(reaudit):
                try:
                    visual = reaudit(original, region.image, prompt=region.prompt, metadata=region.metadata)
                except (ValueError, TypeError, OSError):
                    continue
            elif provenance.get('visual_request_sha256') == config_sha(region.prompt):
                visual = original
            else:
                continue
            details = dict(visual.get('metadata') or {})
            verified = bool(visual.get('verified'))
            irrelevant = details.get('offline_reaudit') and details.get('evidence_relevant') is False
            if not verified and not irrelevant:
                continue
            if verified and visual.get('description'):
                region.evidence.update(content=visual['description'], confidence=float(visual['confidence']))
            region.evidence['provenance'].update(
                analysis_mode='multimodal_cached_reaudit' if verified else 'multimodal_cached_irrelevant',
                vision_verified=verified, vision_adapter=type(self.adapter).__name__,
                vision_metadata=details, vision_reused=True, vision_cache_scope=scope,
                vision_cache_source_evidence_id=cached.get('evidence_id'),
                vision_cache_source_run_id=cached.get('run_id'),
                parser_cache_identity=self._cache_identity(), visual_request_sha256=config_sha(region.prompt))
            if irrelevant:
                region.evidence['provenance']['vision_skip_reason'] = 'exact_crop_irrelevant'
            session.stats.vision_cache_hits += 1
            session.stats.vision_cross_run_cache_hits += int(scope == 'cross_run_exact_crop')
            return True
        return False

    def _analyze(self, session, region):
        if getattr(self.adapter, 'local_execution', False):
            self._analyze_local(session, region)
            return
        context, vision_adapter = session.context, self.adapter
        bind = (getattr(vision_adapter, 'for_run', None)
                if callable(getattr(type(vision_adapter), 'for_run', None)) else None)
        if callable(bind):
            vision_adapter = bind(context.db, context.run_id)
        durable_dispatch = bool(getattr(vision_adapter, 'durable_dispatch', False))
        prior_vision_calls = session.prior_calls
        current_document_artifacts = session.current_documents
        per_tool_limit, per_run_limit, per_document_limit = session.limits
        evidence_item = region.evidence
        artifact_key, vision_prompt, vision_context = region.image, region.prompt, region.metadata
        document_key = str(evidence_item["provenance"].get("document_sha256") or "").strip()
        run = context.db.get_run(context.run_id)
        budget = Budget.from_dict(run.get("budget"))
        # Vision and text calls share one durable provider-attempt
        # counter.  Protect the same synthesis/repair slots used by the
        # text runtime so a table reader cannot consume the final answer.
        research_provider_cap = max(
            0,
            budget.max_llm_calls
            - budget.synthesis_reserve_calls
            - budget.repair_reserve_calls,
        )
        if budget.unlimited_llm_usage:
            research_provider_cap = None
        reserved_attempt = False

        def reserve_visual_attempt():
            nonlocal reserved_attempt
            try:
                if (session.stats.vision_api_calls >= per_tool_limit
                        or prior_vision_calls + session.stats.vision_api_calls >= per_run_limit):
                    raise RuntimeError('Bounded visual call budget exhausted')
                if not durable_dispatch:
                    context.db.reserve_llm_attempt(context.run_id, max_llm_calls=research_provider_cap)
            except RuntimeError as exc:
                raise LLMBudgetExhausted(str(exc)) from exc
            if not durable_dispatch:
                visual_dispatched()

        def visual_dispatched():
            nonlocal reserved_attempt
            reserved_attempt = True
            session.stats.vision_api_calls += 1

        started = time.perf_counter()
        model_name = str(
            getattr(vision_adapter, "model", type(vision_adapter).__name__)
        )
        max_output_tokens = int(
            getattr(vision_adapter, "max_output_tokens", 4_000)
        )
        try:
            kwargs = {}
            if 'before_attempt' in inspect.signature(vision_adapter.analyze).parameters:
                kwargs['before_attempt'] = reserve_visual_attempt
                if durable_dispatch:
                    kwargs['on_dispatch'] = visual_dispatched
            else:
                # Compatibility adapters without a dispatch hook retain
                # their conservative invocation-level accounting.
                reserve_visual_attempt()
            visual = vision_adapter.analyze(
                artifact_key,
                vision_prompt,
                vision_context,
                **kwargs,
            )
        except Exception as exc:
            if not reserved_attempt:
                session.stats.vision_skipped += 1
                evidence_item['provenance'].update(
                    vision_verified=False, vision_skip_reason=(
                        'shared_provider_budget_exhausted' if isinstance(exc, LLMBudgetExhausted)
                        else 'provider_pre_dispatch_rejected'),
                    vision_error=str(exc)[:500])
                session.stats.vision_errors.append(str(exc)[:500])
                return
            vision_attempt = prior_vision_calls + session.stats.vision_api_calls
            duration_ms = (time.perf_counter() - started) * 1_000
            from .provider_diagnostics import safe_error_text
            error = safe_error_text(exc)[:1_000]
            failed_usage = getattr(exc, 'vision_usage', {})
            failed_prompt = int(failed_usage.get('prompt_tokens') or 0)
            failed_output = int(failed_usage.get('completion_tokens') or 0)
            session.stats.vision_prompt_tokens += failed_prompt
            session.stats.vision_completion_tokens += failed_output
            record_call = (lambda **kwargs: None) if durable_dispatch else context.db.record_llm_call
            record_call(
                run_id=context.run_id,
                stage="vision",
                attempt=vision_attempt,
                status="FAILED",
                model=model_name,
                prompt_tokens=failed_prompt,
                completion_tokens=failed_output,
                finish_reason=failed_usage.get('finish_reason'),
                thinking_enabled=False,
                max_output_tokens=max_output_tokens,
                duration_ms=duration_ms,
                provider_attempts=int(failed_usage.get('provider_attempts', 1)),
                error=error,
            )
            evidence_item["provenance"].update(
                {
                    "analysis_mode": "multimodal_adapter_failed",
                    "vision_verified": False,
                    "vision_adapter": type(vision_adapter).__name__,
                    "vision_error": error,
                    "vision_schema_issues": getattr(exc, 'vision_schema_issues', []),
                }
            )
            session.stats.vision_errors.append(error)
            return
        if document_key:
            current_document_artifacts.setdefault(document_key, set()).add(
                artifact_key
            )
        vision_attempt = prior_vision_calls + session.stats.vision_api_calls
        visual_metadata = dict(visual.get("metadata") or {})
        session.stats.vision_prompt_tokens += int(visual_metadata.get("prompt_tokens") or 0)
        session.stats.vision_completion_tokens += int(
            visual_metadata.get("completion_tokens") or 0
        )
        record_call = (lambda **kwargs: None) if durable_dispatch else context.db.record_llm_call
        record_call(
            run_id=context.run_id,
            stage="vision",
            attempt=vision_attempt,
            status="COMPLETED",
            model=str(visual_metadata.get("model") or model_name),
            thinking_enabled=False,
            max_output_tokens=max_output_tokens,
            prompt_tokens=int(visual_metadata.get("prompt_tokens") or 0),
            completion_tokens=int(
                visual_metadata.get("completion_tokens") or 0
            ),
            finish_reason=str(
                visual_metadata.get("finish_reason") or "unknown"
            ),
            duration_ms=(time.perf_counter() - started) * 1_000,
            provider_attempts=int(visual_metadata.get('provider_attempts', 1)),
        )
        description = str(visual.get("description") or visual.get("content") or "").strip()
        if description:
            evidence_item["content"] = description
        verified = bool(visual.get("verified", False))
        visual_confidence = max(
            0.0, min(1.0, float(visual.get("confidence", 0.0)))
        )
        if verified:
            evidence_item["confidence"] = max(
                float(evidence_item.get("confidence", 0.0)),
                visual_confidence,
            )
        else:
            evidence_item["confidence"] = min(
                float(evidence_item.get("confidence", 0.45)),
                visual_confidence,
            )
        evidence_item["provenance"].update(
            {
                "analysis_mode": (
                    "multimodal_adapter"
                    if verified
                    else "multimodal_adapter_unverified"
                ),
                "vision_verified": verified,
                "vision_adapter": type(vision_adapter).__name__,
                "vision_metadata": visual_metadata,
                "parser_cache_identity": self._cache_identity(),
                "visual_request_sha256": config_sha(region.prompt),
            }
        )

    def _source_region(self, item):
        provenance = item['provenance']
        binding = provenance.get('layout_target_binding') or {}
        label = binding.get('label') or provenance.get('layout_source_label') or []
        return SourceRegion(paper_sha256=provenance.get('document_sha256') or '',
            object_id=provenance.get('layout_object_id') or '',
            label=({'table': 'Table ', 'equation': 'Equation ', 'figure': 'Figure '}.get(label[0], '')
                   + label[-1]) if label else '',
            page=item['page'], bbox=provenance.get('render_bbox') or item['bbox'],
            crop_sha256=self.file_hash(Path(item['artifact_path'])))

    def _cache_identity(self):
        return config_sha({'adapter': type(self.adapter).__name__, 'model': getattr(self.adapter, 'model', None),
            'fallback_models': getattr(self.adapter, 'fallback_models', ()),
            'max_output': getattr(self.adapter, 'max_output_tokens', None),
            'provider': getattr(self.adapter, 'base_url', None),
            'image_pixels': getattr(self.adapter, 'min_image_pixels', None),
            'image_limit': getattr(self.adapter, 'max_image_bytes', None),
            'parser_version': getattr(self.adapter, 'parser_version', None),
            'identity': getattr(self.adapter, 'identity', None), 'validation': 'structured-region-v1'})

    def _analyze_local(self, session, region):
        started = time.perf_counter()
        try:
            visual = self.adapter.analyze(region.image, region.prompt, region.metadata)
            meta = visual['metadata']
            session.stats.local_parse_attempts += int(meta.get('local_parse_attempts', 1))
            session.stats.local_cache_hits += int(bool(meta.get('local_cache_hit')))
            if meta.get('local_sdk_requests') is None:
                session.stats.local_sdk_requests_unknown += 1
            else:
                session.stats.local_sdk_requests += int(meta['local_sdk_requests'])
            region.evidence.update(content=visual['description'], confidence=visual['confidence'])
            region.evidence['provenance'].update(analysis_mode='local_structured_parse',
                vision_verified=bool(visual['verified']), vision_adapter=type(self.adapter).__name__,
                vision_metadata=meta, parser_cache_identity=self._cache_identity())
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
            session.stats.local_parse_attempts += int(getattr(exc, 'local_parse_attempts', 1))
            session.stats.local_cache_hits += int(bool(getattr(exc, 'local_cache_hit', False)))
            sdk = getattr(exc, 'local_sdk_requests', None)
            session.stats.local_sdk_requests += int(sdk or 0)
            session.stats.local_sdk_requests_unknown += int(sdk is None)
            region.evidence['provenance']['local_raw_artifact_path'] = getattr(exc, 'raw_artifact_path', None)
            session.stats.vision_errors.append(str(exc)[:1000])
            region.evidence['provenance'].update(vision_verified=False,
                analysis_mode='local_structured_failed', vision_error=str(exc)[:1000])
        finally:
            document = region.evidence['provenance'].get('document_sha256')
            session.current_documents.setdefault(document, set()).add(region.image)
            session.stats.local_duration_ms += (time.perf_counter() - started) * 1000
