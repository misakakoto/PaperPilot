from __future__ import annotations

import contextlib
import inspect
import os
import socket
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .db import RuntimeDatabase
from .models import (
    RiskLevel,
    RunStatus,
    ToolCallStatus,
    ToolInvocationOutcome,
    ToolResult,
    stable_hash,
)
from .tracing import DurableTracer

ToolHandler = Callable[["ToolContext", Mapping[str, Any]], ToolResult | Mapping[str, Any]]


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    handler: ToolHandler
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.READ_ONLY
    side_effects: bool = False
    requires_approval: bool = False
    timeout_seconds: int | None = None
    max_attempts: int = 1
    tags: tuple[str, ...] = ()


@dataclass(slots=True)
class ToolContext:
    run_id: str
    step_id: str
    tool_call_id: str
    workspace: Path
    db: RuntimeDatabase
    tracer: DurableTracer

    def cancelled(self) -> bool:
        return self.db.get_run(self.run_id)['status'] == RunStatus.CANCELLED.value


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec, *, replace: bool = False) -> None:
        if not spec.name or "." not in spec.name:
            raise ValueError("Tool names must be namespaced, for example paper.search")
        if spec.name in self._tools and not replace:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool {name!r}. Registered: {sorted(self._tools)}") from exc

    def list(self) -> list[ToolSpec]:
        return [self._tools[name] for name in sorted(self._tools)]

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
                "output_schema": spec.output_schema,
                "risk_level": spec.risk_level.value,
                "side_effects": spec.side_effects,
                "requires_approval": spec.requires_approval,
                "timeout_seconds": spec.timeout_seconds,
                "max_attempts": spec.max_attempts,
                "tags": list(spec.tags),
            }
            for spec in self.list()
        ]


def _matches_type(value: Any, expected: str) -> bool:
    return {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "object": isinstance(value, Mapping),
        "array": isinstance(value, list),
        "null": value is None,
    }.get(expected, True)


def validate_schema(value: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Validate the small JSON-Schema subset used by built-in tools."""
    if not schema:
        return
    missing = [key for key in schema.get("required", []) if key not in value]
    if missing:
        raise ValueError(f"Missing required tool arguments: {missing}")
    properties = schema.get("properties", {})
    if schema.get("additionalProperties", True) is False:
        unexpected = sorted(set(value) - set(properties))
        if unexpected:
            raise ValueError(f"Unexpected tool arguments: {unexpected}")
    for key, item in value.items():
        prop = properties.get(key)
        if not prop:
            continue
        expected = prop.get("type")
        if isinstance(expected, list):
            valid = any(_matches_type(item, kind) for kind in expected)
        elif expected:
            valid = _matches_type(item, expected)
        else:
            valid = True
        if not valid:
            raise TypeError(f"Argument {key!r} expected {expected}, got {type(item).__name__}")
        if "enum" in prop and item not in prop["enum"]:
            raise ValueError(f"Argument {key!r} must be one of {prop['enum']}")


class _LeaseHeartbeat:
    def __init__(self, db: RuntimeDatabase, tool_call_id: str, owner: str, lease_seconds: int):
        self.db = db
        self.tool_call_id = tool_call_id
        self.owner = owner
        self.lease_seconds = lease_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_LeaseHeartbeat":
        interval = max(5.0, self.lease_seconds / 3.0)

        def run() -> None:
            while not self._stop.wait(interval):
                with contextlib.suppress(Exception):
                    active = self.db.heartbeat_tool_lease(
                        self.tool_call_id,
                        owner=self.owner,
                        lease_seconds=self.lease_seconds,
                    )
                    if not active:
                        return

        self._thread = threading.Thread(
            target=run,
            name=f"paperpilot-heartbeat-{self.tool_call_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class PersistentToolRuntime:
    """Durable dispatch with schema checks, idempotency, leases, and HITL."""

    def __init__(
        self,
        db: RuntimeDatabase,
        registry: ToolRegistry,
        tracer: DurableTracer,
        workspace: str | Path,
        *,
        approval_threshold: RiskLevel = RiskLevel.HIGH,
    ) -> None:
        self.db = db
        self.registry = registry
        self.tracer = tracer
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.approval_threshold = approval_threshold

    def invoke(
        self,
        *,
        run_id: str,
        step_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        idempotency_key: str | None = None,
        approval_context: Mapping[str, Any] | None = None,
    ) -> ToolInvocationOutcome:
        spec = self.registry.get(tool_name)
        args = dict(arguments)
        validate_schema(args, spec.input_schema)
        key = idempotency_key or stable_hash("tool-call", run_id, tool_name, args)
        tool_call_id, reused = self.db.create_tool_call(
            run_id=run_id,
            step_id=step_id,
            tool_name=tool_name,
            args=args,
            risk_level=spec.risk_level.value,
            idempotency_key=key,
            max_attempts=max(1, spec.max_attempts),
        )
        call = self.db.get_tool_call(tool_call_id)
        status = ToolCallStatus(call["status"])

        if reused:
            if status == ToolCallStatus.COMPLETED:
                return ToolInvocationOutcome(
                    status=status,
                    tool_call_id=tool_call_id,
                    result=ToolResult.from_dict(call["result"] or {"ok": True}),
                    reused=True,
                )
            if status == ToolCallStatus.WAITING_APPROVAL:
                approval = self._approval_for_call(run_id, tool_call_id)
                return ToolInvocationOutcome(
                    status=status,
                    tool_call_id=tool_call_id,
                    approval_id=approval["approval_id"] if approval else None,
                    reused=True,
                )
            if status == ToolCallStatus.REJECTED:
                return ToolInvocationOutcome(
                    status=status,
                    tool_call_id=tool_call_id,
                    result=ToolResult(ok=False, error="Tool call was rejected by a human"),
                    reused=True,
                )
            if status == ToolCallStatus.FAILED:
                return ToolInvocationOutcome(
                    status=status,
                    tool_call_id=tool_call_id,
                    result=ToolResult.from_dict(
                        call["result"] or {"ok": False, "error": call.get("error")}
                    ),
                    reused=True,
                )
            if status in {ToolCallStatus.APPROVED, ToolCallStatus.RUNNING}:
                return self.execute(tool_call_id, reused=True)

        approval_required = spec.requires_approval or spec.risk_level.rank >= self.approval_threshold.rank
        if approval_required:
            request = {
                "tool": spec.name,
                "description": spec.description,
                "arguments": args,
                "risk_level": spec.risk_level.value,
                "side_effects": spec.side_effects,
                "context": dict(approval_context or {}),
            }
            approval_id = self.db.create_approval(run_id, tool_call_id, request)
            self.db.update_tool_call(tool_call_id, ToolCallStatus.WAITING_APPROVAL)
            self.db.update_run_status(run_id, RunStatus.WAITING_APPROVAL, current_step_id=step_id)
            return ToolInvocationOutcome(
                status=ToolCallStatus.WAITING_APPROVAL,
                tool_call_id=tool_call_id,
                approval_id=approval_id,
                reused=reused,
            )
        return self.execute(tool_call_id, reused=reused)

    def execute(self, tool_call_id: str, *, reused: bool = False) -> ToolInvocationOutcome:
        call = self.db.get_tool_call(tool_call_id)
        status = ToolCallStatus(call["status"])
        if self.db.get_run(call['run_id'])['status'] == RunStatus.CANCELLED.value:
            return ToolInvocationOutcome(status=ToolCallStatus.REJECTED,
                tool_call_id=tool_call_id, result=ToolResult(ok=False, error='Run cancelled'))
        if status == ToolCallStatus.COMPLETED:
            return ToolInvocationOutcome(
                status=status,
                tool_call_id=tool_call_id,
                result=ToolResult.from_dict(call["result"] or {"ok": True}),
                reused=True,
            )
        if status == ToolCallStatus.WAITING_APPROVAL:
            approval = self._approval_for_call(call["run_id"], tool_call_id)
            return ToolInvocationOutcome(
                status=status,
                tool_call_id=tool_call_id,
                approval_id=approval["approval_id"] if approval else None,
                reused=reused,
            )
        if status == ToolCallStatus.REJECTED:
            return ToolInvocationOutcome(
                status=status,
                tool_call_id=tool_call_id,
                result=ToolResult(ok=False, error="Tool call rejected"),
                reused=reused,
            )
        if status == ToolCallStatus.FAILED:
            return ToolInvocationOutcome(
                status=status,
                tool_call_id=tool_call_id,
                result=ToolResult.from_dict(call["result"] or {"ok": False, "error": call.get("error")}),
                reused=reused,
            )

        spec = self.registry.get(call["tool_name"])
        owner = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
        lease_seconds = max(60, int(spec.timeout_seconds or 300) + 30)
        acquired = self.db.acquire_tool_lease(
            tool_call_id,
            owner=owner,
            lease_seconds=lease_seconds,
        )
        if not acquired:
            latest = self.db.get_tool_call(tool_call_id)
            latest_status = ToolCallStatus(latest["status"])
            if latest_status == ToolCallStatus.COMPLETED:
                return ToolInvocationOutcome(
                    status=latest_status,
                    tool_call_id=tool_call_id,
                    result=ToolResult.from_dict(latest["result"] or {"ok": True}),
                    reused=True,
                )
            return ToolInvocationOutcome(status=latest_status, tool_call_id=tool_call_id, reused=True)

        context = ToolContext(
            run_id=call["run_id"],
            step_id=call["step_id"],
            tool_call_id=tool_call_id,
            workspace=self.workspace,
            db=self.db,
            tracer=self.tracer,
        )
        try:
            with _LeaseHeartbeat(self.db, tool_call_id, owner, lease_seconds):
                with self.tracer.span(
                    f"tool.{spec.name}",
                    run_id=call["run_id"],
                    kind="tool",
                    attributes={
                        "gen_ai.tool.name": spec.name,
                        "paperpilot.tool_call_id": tool_call_id,
                        "paperpilot.risk_level": spec.risk_level.value,
                        "paperpilot.idempotency_key": call["idempotency_key"],
                    },
                ) as span:
                    if context.cancelled():
                        raise RuntimeError('Run cancelled before tool execution')
                    raw = spec.handler(context, call["args"])
                    if inspect.isawaitable(raw):
                        raise TypeError(
                            "Async tool handlers are not supported by this synchronous runtime; wrap them with an adapter"
                        )
                    result = raw if isinstance(raw, ToolResult) else ToolResult.from_dict(raw)
                    span.set_attribute("paperpilot.tool.ok", result.ok)
                    span.set_attribute("paperpilot.tool.evidence_count", len(result.evidence))
                    span.set_attribute("paperpilot.tool.artifact_count", len(result.artifacts))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result = ToolResult(
                ok=False,
                error=error,
                retryable=False,
                output={"traceback": traceback.format_exc(limit=20)},
            )
            self.db.update_tool_call(
                tool_call_id,
                ToolCallStatus.FAILED,
                result=result.to_dict(),
                error=error,
            )
            return ToolInvocationOutcome(
                status=ToolCallStatus.FAILED,
                tool_call_id=tool_call_id,
                result=result,
                reused=reused,
            )

        final_status = ToolCallStatus.COMPLETED if result.ok else ToolCallStatus.FAILED
        self.db.update_tool_call(
            tool_call_id,
            final_status,
            result=result.to_dict(),
            error=result.error,
        )
        return ToolInvocationOutcome(
            status=final_status,
            tool_call_id=tool_call_id,
            result=result,
            reused=reused,
        )

    def execute_approved_for_run(self, run_id: str) -> list[ToolInvocationOutcome]:
        outcomes: list[ToolInvocationOutcome] = []
        for call in self.db.list_tool_calls(run_id):
            if ToolCallStatus(call["status"]) == ToolCallStatus.APPROVED:
                outcomes.append(self.execute(call["tool_call_id"], reused=True))
        return outcomes

    def _approval_for_call(self, run_id: str, tool_call_id: str) -> dict[str, Any] | None:
        for approval in self.db.list_approvals(run_id):
            if approval["tool_call_id"] == tool_call_id:
                return approval
        return None
