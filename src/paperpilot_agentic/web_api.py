from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping

from .orchestrator import ResearchAgentRuntime

RuntimeFactory = Callable[[], ResearchAgentRuntime]
MetadataPreparer = Callable[[Mapping[str, Any]], Mapping[str, Any]]
_PRIVATE_PATH_KEYS = {"pdf_path", "pdf_paths", "repository_path", "code_path"}


class _BackgroundRunPool:
    def __init__(self, *, workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, workers),
            thread_name_prefix="paperpilot-agentic",
        )
        self._lock = threading.Lock()
        self._futures: dict[str, Future[Any]] = {}

    def active(self, run_id: str) -> bool:
        with self._lock:
            item = self._futures.get(run_id)
            return item is not None and not item.done()

    def submit(self, run_id: str, operation: Callable[[], Any]) -> bool:
        with self._lock:
            existing = self._futures.get(run_id)
            if existing is not None and not existing.done():
                return False
            future = self._executor.submit(operation)
            self._futures[run_id] = future
        # Future.add_done_callback executes synchronously if already completed.
        # Register outside the non-reentrant lock to avoid deadlocking POST /runs.
        future.add_done_callback(lambda completed: self._forget(run_id, completed))
        return True

    def _forget(self, run_id: str, completed: Future[Any]) -> None:
        with self._lock:
            # A delayed callback from an old invocation must not remove a newer
            # invocation's guard for the same run.
            if self._futures.get(run_id) is completed:
                self._futures.pop(run_id, None)


def _without_client_paths(claims: Any) -> Any:
    if not isinstance(claims, list):
        return claims
    cleaned: list[Any] = []
    for raw in claims:
        if not isinstance(raw, Mapping):
            cleaned.append(raw)
            continue
        item = {key: value for key, value in raw.items() if key not in _PRIVATE_PATH_KEYS}
        metadata = item.get("metadata")
        if isinstance(metadata, Mapping):
            item["metadata"] = {
                key: value
                for key, value in metadata.items()
                if key not in _PRIVATE_PATH_KEYS
            }
        cleaned.append(item)
    return cleaned


def create_fastapi_router(
    *,
    runtime_factory: RuntimeFactory | None = None,
    db_path: str | Path = ".paperpilot_agentic/runtime.sqlite3",
    workspace: str | Path = ".",
    prefix: str = "/agentic",
    prepare_metadata: MetadataPreparer | None = None,
    default_engine: str = 'legacy',
    default_visual_policy: str = 'explicit',
):
    """Create an optional FastAPI router for the durable runtime.

    FastAPI is imported lazily so the core package remains dependency-light. Mount this
    router behind the host application's authentication/authorization middleware; the
    approve/reject endpoints are privileged control-plane operations.
    """

    if default_engine not in {'legacy', 'answer_first_text'}:
        raise ValueError('Unknown default research engine')
    if default_visual_policy not in {'auto', 'off', 'explicit'}:
        raise ValueError('Unknown default visual policy')

    try:
        from fastapi import APIRouter, Body, HTTPException, Header
        from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    except Exception as exc:  # pragma: no cover - optional integration
        raise RuntimeError("FastAPI is required to create the agentic web router") from exc

    factory = runtime_factory or (
        lambda: ResearchAgentRuntime(db_path=db_path, workspace=workspace)
    )
    router = APIRouter(prefix=prefix, tags=["paperpilot-agentic"])
    background = _BackgroundRunPool(workers=2)

    def call(operation: Callable[[ResearchAgentRuntime], Any]) -> Any:
        try:
            return operation(factory())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError, TypeError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/runs")
    def start_run(payload: Mapping[str, Any] = Body(...)) -> Any:
        value = dict(payload)
        if not value.get("question"):
            raise HTTPException(status_code=422, detail="question is required")
        def operation(runtime: ResearchAgentRuntime) -> Any:
            metadata = dict(value.get("metadata") or {})
            if 'retrieval_profile' in value:
                metadata['retrieval_profile'] = value['retrieval_profile']
            profile = value.get('execution_profile', metadata.get('execution_profile', 'daily'))
            if profile not in {'daily', 'frozen_acceptance'}:
                raise ValueError('Unknown execution_profile')
            metadata['execution_profile'] = profile
            if 'visual_policy' in value:
                if value['visual_policy'] not in {'auto', 'off', 'explicit'}:
                    raise ValueError('Unknown visual_policy')
                metadata['visual_policy'] = value['visual_policy']
            if prepare_metadata is not None:
                metadata = dict(prepare_metadata(metadata))
            engine = value.get('engine', default_engine)
            if engine not in {'legacy', 'answer_first_text', 'research_notebook'}:
                raise ValueError('Unknown research engine')
            if engine in {'answer_first_text', 'research_notebook'}:
                if value.get('claims') or value.get('edges') or not value.get('auto_advance', True):
                    raise ValueError('Text reader accepts questions, not Claim/experiment execution plans')
                if not runtime.text_reader_protocol:
                    raise ValueError('Text reader is not configured')
                metadata['engine'] = engine
                metadata.setdefault('visual_policy', default_visual_policy)
            else:
                metadata.pop('engine', None)
            if not bool(value.get("auto_advance", True)):
                return runtime.start(
                    question=str(value["question"]),
                    claims=_without_client_paths(value.get("claims")),
                    edges=value.get("edges"),
                    budget=value.get("budget"),
                    metadata=metadata,
                    auto_advance=False,
                )
            queued = runtime.enqueue(
                question=str(value["question"]),
                claims=_without_client_paths(value.get("claims")),
                edges=value.get("edges"),
                budget=value.get("budget"),
                metadata=metadata,
            )
            run_id = str(queued["run"]["run_id"])
            background.submit(run_id, lambda: factory().resume(run_id))
            return JSONResponse(status_code=202, content=queued)

        return call(operation)

    @router.get("/runs")
    def list_runs(limit: int = 50) -> Any:
        return {"runs": call(lambda runtime: runtime.list_runs(limit=limit))}

    @router.get("/runs/{run_id}")
    def get_run(run_id: str, include_trace: bool = True) -> Any:
        return call(lambda runtime: runtime.status(run_id, include_trace=include_trace))

    @router.post('/runs/{run_id}/workset-review')
    def start_workset_review(run_id: str, payload: Mapping[str, Any] = Body(default={})) -> Any:
        """Explicit read-only review; server builds pages from the owned ledger."""
        if set(payload) - {'budget', 'synthesize'}:
            raise HTTPException(status_code=422, detail='Only budget and synthesize may be supplied')
        def operation(runtime):
            from .workset_review import enqueue_review
            queued = enqueue_review(runtime, run_id, budget=payload.get('budget'),
                                    synthesize=payload.get('synthesize', True))
            child = queued['run']['run_id']
            background.submit(child, lambda: factory().resume(child))
            return JSONResponse(status_code=202, content=queued)
        return call(operation)

    @router.post('/runs/{run_id}/research-design')
    def start_research_design(run_id: str, payload: Mapping[str, Any] = Body(...)) -> Any:
        if set(payload) - {'question', 'source_ids', 'reference_ids', 'budget', 'validation_run_id'}:
            raise HTTPException(status_code=422, detail='Only design request, source IDs, validation run ID and budget allowed')
        def operation(runtime):
            from .research_design import enqueue_design
            queued = enqueue_design(runtime, run_id, question=payload.get('question'),
                source_ids=payload.get('source_ids'), reference_ids=payload.get('reference_ids'),
                budget=payload.get('budget'), validation_run_id=payload.get('validation_run_id'))
            child = queued['run']['run_id']
            background.submit(child, lambda: factory().resume(child))
            return JSONResponse(status_code=202, content=queued)
        return call(operation)

    @router.post('/runs/{run_id}/method-validation')
    def start_method_validation(run_id: str, payload: Mapping[str, Any] = Body(...)) -> Any:
        if set(payload) != {'binding_id'}:
            raise HTTPException(status_code=422, detail='Only a host-prepared binding_id is allowed')
        def operation(runtime):
            from .reference_methods import enqueue_validation
            queued = enqueue_validation(runtime, payload['binding_id'], parent_run_id=run_id)
            child = queued['run']['run_id']
            background.submit(child, lambda: factory().resume(child))
            return JSONResponse(status_code=202, content=queued)
        return call(operation)

    @router.get('/runs/{run_id}/notebook')
    def notebook_notes(run_id: str, query: str = '', kind: str | None = None, offset: int = 0, limit: int = 20):
        from .research_notebook import ResearchNotebook
        def operation(runtime):
            book = ResearchNotebook(runtime.db, run_id)
            return {'overview':book.overview(), 'notes':book.notes(query=query,kind=kind,offset=offset,limit=limit)}
        return call(operation)

    @router.post('/runs/{run_id}/method-comparison')
    def start_method_comparison(run_id: str, payload: Mapping[str, Any] = Body(...)) -> Any:
        if set(payload) != {'binding_id'}:
            raise HTTPException(status_code=422, detail='Only a host-reviewed comparison binding_id is allowed')
        def operation(runtime):
            from .paired_methods import describe_comparison, enqueue_comparison
            if describe_comparison(runtime, payload['binding_id'])['design_run_id'] != run_id:
                raise ValueError('Comparison binding belongs to another design run')
            queued = enqueue_comparison(runtime, payload['binding_id'])
            child = queued['run']['run_id']
            background.submit(child, lambda: factory().resume(child))
            return JSONResponse(status_code=202, content=queued)
        return call(operation)

    @router.get('/runs/{run_id}/notebook/sources/{batch_id}')
    def notebook_source(run_id: str, batch_id: str):
        from .research_notebook import ResearchNotebook
        return call(lambda runtime: ResearchNotebook(runtime.db,run_id).source(batch_id))

    @router.post('/runs/{run_id}/notebook/import-experiment')
    def notebook_import(run_id: str, payload: Mapping[str, Any] = Body(...)):
        from .research_notebook import ResearchNotebook
        return call(lambda runtime: ResearchNotebook(runtime.db,run_id).import_experiment_run(str(payload['source_run_id'])))

    @router.post('/runs/{run_id}/notebook/continue')
    def notebook_continue(run_id: str, payload: Mapping[str, Any] = Body(...)):
        from .research_notebook import ResearchNotebook
        def operation(runtime):
            if runtime.db.get_run(run_id)['metadata'].get('engine')!='research_notebook':
                raise ValueError('Not a notebook extraction run')
            ResearchNotebook(runtime.db,run_id).extend_budget(payload['budget'])
            background.submit(run_id,lambda: factory().resume(run_id))
            return JSONResponse(status_code=202,content=runtime.status(run_id))
        return call(operation)

    @router.post("/runs/{run_id}/follow-up")
    def start_follow_up(run_id: str, payload: Mapping[str, Any] = Body(...)) -> Any:
        value = dict(payload)
        if not str(value.get("question") or "").strip():
            raise HTTPException(status_code=422, detail="question is required")

        def operation(runtime: ResearchAgentRuntime) -> Any:
            metadata = dict(value.get("metadata") or {})
            if 'execution_profile' in value:
                if value['execution_profile'] not in {'daily', 'frozen_acceptance'}:
                    raise ValueError('Unknown execution_profile')
                metadata['execution_profile'] = value['execution_profile']
            if 'visual_policy' in value:
                if value['visual_policy'] not in {'auto', 'off', 'explicit'}:
                    raise ValueError('Unknown visual_policy')
                metadata['visual_policy'] = value['visual_policy']
            if prepare_metadata is not None:
                metadata = dict(prepare_metadata(metadata))
            queued = runtime.enqueue_follow_up(
                parent_run_id=run_id,
                question=str(value["question"]),
                claims=_without_client_paths(value.get("claims")),
                edges=value.get("edges"),
                budget=value.get("budget"),
                metadata=metadata,
            )
            child_run_id = str(queued["run"]["run_id"])
            background.submit(child_run_id, lambda: factory().resume(child_run_id))
            return JSONResponse(status_code=202, content=queued)

        return call(operation)

    @router.get("/runs/{run_id}/evidence/{evidence_id}/artifact")
    def get_evidence_artifact(run_id: str, evidence_id: str) -> Any:
        """Serve only run-owned visual evidence beneath the configured workspace."""

        runtime = factory()
        runtime.status(run_id, include_trace=False)
        evidence = next(
            (
                item
                for item in runtime.db.list_evidence(run_id)
                if str(item.get("evidence_id")) == evidence_id
            ),
            None,
        )
        if evidence is None:
            raise HTTPException(status_code=404, detail="evidence not found")
        raw_path = str(evidence.get("artifact_path") or "").strip()
        if not raw_path:
            raise HTTPException(status_code=404, detail="evidence has no artifact")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file() or not path.is_relative_to(runtime.workspace):
            raise HTTPException(status_code=403, detail="artifact is outside the run workspace")
        if path.suffix.casefold() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise HTTPException(status_code=415, detail="only visual evidence can be served")
        return FileResponse(path)

    @router.post("/runs/{run_id}/resume")
    def resume_run(run_id: str) -> Any:
        def operation(runtime: ResearchAgentRuntime) -> Any:
            current = runtime.status(run_id, include_trace=False)
            background.submit(run_id, lambda: factory().resume(run_id))
            return JSONResponse(status_code=202, content=current)

        return call(operation)

    @router.post("/runs/{run_id}/cancel")
    def cancel_run(run_id: str, payload: Mapping[str, Any] | None = Body(default=None)) -> Any:
        reason = str((payload or {}).get("reason", "Cancelled through API"))
        return call(lambda runtime: runtime.cancel(run_id, reason=reason))

    @router.post("/approvals/{approval_id}/approve")
    def approve(approval_id: str, payload: Mapping[str, Any] | None = Body(default=None)) -> Any:
        value = dict(payload or {})

        def operation(runtime: ResearchAgentRuntime) -> Any:
            approval = runtime.approve(
                approval_id,
                actor=str(value.get("actor", "api-user")),
                comment=value.get("comment"),
            )
            if value.get("resume", True):
                background.submit(
                    str(approval["run_id"]),
                    lambda: factory().resume(str(approval["run_id"])),
                )
                return JSONResponse(
                    status_code=202,
                    content=runtime.status(str(approval["run_id"]), include_trace=False),
                )
            return approval

        return call(operation)

    @router.post("/approvals/{approval_id}/reject")
    def reject(approval_id: str, payload: Mapping[str, Any] | None = Body(default=None)) -> Any:
        value = dict(payload or {})

        def operation(runtime: ResearchAgentRuntime) -> Any:
            approval = runtime.reject(
                approval_id,
                actor=str(value.get("actor", "api-user")),
                comment=value.get("comment"),
            )
            if value.get("resume", True):
                background.submit(
                    str(approval["run_id"]),
                    lambda: factory().resume(str(approval["run_id"])),
                )
                return JSONResponse(
                    status_code=202,
                    content=runtime.status(str(approval["run_id"]), include_trace=False),
                )
            return approval

        return call(operation)

    @router.get("/tools")
    def list_tools() -> Any:
        return call(lambda runtime: runtime.list_tools())

    @router.get("/runs/{run_id}/events")
    def stream_events(run_id: str, after: int = 0,
                      last_event_id: str | None = Header(default=None)) -> Any:
        runtime = factory()
        try:
            runtime.status(run_id, include_trace=False)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        def generate():
            try:
                cursor = max(0, int(after), int(last_event_id or 0))
            except ValueError:
                cursor = max(0, int(after))
            last_heartbeat = time.monotonic()
            while True:
                events = runtime.db.list_events(
                    run_id,
                    after_event_id=cursor,
                    limit=200,
                )
                for event in events:
                    cursor = int(event["event_id"])
                    yield (
                        f"id: {cursor}\n"
                        f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
                    )
                run = runtime.db.get_run(run_id)
                if not background.active(run_id) and str(run["status"]) in {
                    "WAITING_APPROVAL",
                    "COMPLETED",
                    "FAILED",
                    "CANCELLED",
                    "BUDGET_EXHAUSTED",
                }:
                    yield (
                        f"data: {json.dumps({'event_type': 'stream.end', 'run_id': run_id, 'status': run['status']})}\n\n"
                    )
                    break
                if time.monotonic() - last_heartbeat >= 5.0:
                    yield ": heartbeat\n\n"
                    last_heartbeat = time.monotonic()
                time.sleep(0.25)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    from .workbench_api import mount_workbench
    mount_workbench(router, factory(), background, call)
    return router
