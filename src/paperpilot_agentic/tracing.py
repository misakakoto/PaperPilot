from __future__ import annotations

import contextlib
import contextvars
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from .db import RuntimeDatabase
from .models import new_id, utc_now

_current_span: contextvars.ContextVar[str | None] = contextvars.ContextVar("paperpilot_current_span", default=None)
_current_trace: contextvars.ContextVar[str | None] = contextvars.ContextVar("paperpilot_current_trace", default=None)

try:  # Optional: exports automatically when the host configured an OTel provider.
    from opentelemetry import trace as otel_trace  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    otel_trace = None


def _safe_attributes(value: Mapping[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in dict(value or {}).items():
        if item is None or isinstance(item, (str, bool, int, float)):
            result[str(key)] = item
        elif isinstance(item, (list, tuple)) and all(isinstance(x, (str, bool, int, float)) for x in item):
            result[str(key)] = list(item)
        else:
            result[str(key)] = str(item)
    return result


@dataclass
class SpanHandle:
    db: RuntimeDatabase
    span_id: str
    trace_id: str
    started_monotonic: float
    otel_span: Any = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value
        if self.otel_span is not None:
            with contextlib.suppress(Exception):
                self.otel_span.set_attribute(key, value)


class DurableTracer:
    """Dual tracing: durable local spans plus optional OpenTelemetry export."""

    def __init__(self, db: RuntimeDatabase, instrumentation_name: str = "paperpilot.agentic"):
        self.db = db
        self._otel_tracer = otel_trace.get_tracer(instrumentation_name) if otel_trace is not None else None

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        *,
        run_id: str | None = None,
        kind: str = "internal",
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[SpanHandle]:
        parent_span_id = _current_span.get()
        trace_id = _current_trace.get() or new_id("trace")
        span_id = new_id("span")
        started_at = utc_now()
        attrs = _safe_attributes(attributes)
        self.db.create_span(
            {
                "span_id": span_id,
                "trace_id": trace_id,
                "parent_span_id": parent_span_id,
                "run_id": run_id,
                "name": name,
                "kind": kind,
                "status": "RUNNING",
                "attributes": attrs,
                "started_at": started_at,
            }
        )
        span_token = _current_span.set(span_id)
        trace_token = _current_trace.set(trace_id)
        otel_cm = self._otel_tracer.start_as_current_span(name, attributes=attrs) if self._otel_tracer else contextlib.nullcontext(None)
        started = time.perf_counter()
        with otel_cm as otel_span:
            handle = SpanHandle(self.db, span_id, trace_id, started, otel_span, dict(attrs))
            try:
                yield handle
            except BaseException as exc:
                duration = (time.perf_counter() - started) * 1000
                if otel_span is not None:
                    with contextlib.suppress(Exception):
                        otel_span.record_exception(exc)
                self.db.finish_span(
                    span_id,
                    status="ERROR",
                    ended_at=utc_now(),
                    duration_ms=duration,
                    error=f"{type(exc).__name__}: {exc}",
                    attributes=handle.attributes,
                )
                raise
            else:
                duration = (time.perf_counter() - started) * 1000
                self.db.finish_span(
                    span_id,
                    status="OK",
                    ended_at=utc_now(),
                    duration_ms=duration,
                    attributes=handle.attributes,
                )
            finally:
                _current_span.reset(span_token)
                _current_trace.reset(trace_token)
