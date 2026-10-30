from __future__ import annotations

import contextlib
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterator, Mapping

from .models import (
    ApprovalStatus,
    Budget,
    ClaimRecord,
    ClaimStatus,
    EvidencePolarity,
    EvidenceRecord,
    EvidenceType,
    ResearchOutcome,
    RunStatus,
    StepStatus,
    ToolCallStatus,
    json_dumps,
    json_loads,
    new_id,
    stable_hash,
    utc_now,
)

_SCHEMA_VERSION = 5


class _ClosingConnection(sqlite3.Connection):
    """A sqlite connection whose context manager also releases the file handle.

    ``sqlite3.Connection.__exit__`` only commits or rolls back; it does not close
    the connection.  The runtime opens a connection per operation, so leaving the
    default behavior leaked handles and locked temporary databases on Windows.
    """

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        try:
            return super().__exit__(exc_type, exc, traceback)
        finally:
            self.close()


class RuntimeDatabase:
    """SQLite-backed durable state store.

    Each method opens its own connection, so separate processes can approve and resume
    the same run. WAL mode and short IMMEDIATE transactions keep transitions atomic.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.migrate()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.path),
            timeout=30.0,
            isolation_level=None,
            factory=_ClosingConnection,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @contextlib.contextmanager
    def transaction(self, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            with contextlib.suppress(Exception):
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def migrate(self) -> None:
        schema = r"""
        CREATE TABLE IF NOT EXISTS runtime_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            status TEXT NOT NULL,
            budget_json TEXT NOT NULL,
            used_steps INTEGER NOT NULL DEFAULT 0,
            used_tool_calls INTEGER NOT NULL DEFAULT 0,
            used_llm_calls INTEGER NOT NULL DEFAULT 0,
            used_input_tokens INTEGER NOT NULL DEFAULT 0,
            used_cached_tokens INTEGER NOT NULL DEFAULT 0,
            used_reasoning_tokens INTEGER NOT NULL DEFAULT 0,
            used_output_tokens INTEGER NOT NULL DEFAULT 0,
            used_tokens INTEGER NOT NULL DEFAULT 0,
            estimated_cost_usd REAL NOT NULL DEFAULT 0,
            estimated_cost_cny REAL NOT NULL DEFAULT 0,
            research_outcome TEXT,
            current_step_id TEXT,
            error TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS steps (
            step_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            input_json TEXT NOT NULL DEFAULT '{}',
            output_json TEXT,
            error TEXT,
            lease_owner TEXT,
            lease_expires_at TEXT,
            started_at TEXT,
            completed_at TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(run_id, ordinal)
        );
        CREATE INDEX IF NOT EXISTS idx_steps_run_status ON steps(run_id, status, ordinal);
        CREATE TABLE IF NOT EXISTS tool_calls (
            tool_call_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            step_id TEXT REFERENCES steps(step_id) ON DELETE SET NULL,
            tool_name TEXT NOT NULL,
            args_json TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            status TEXT NOT NULL,
            idempotency_key TEXT NOT NULL UNIQUE,
            attempt INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 1,
            result_json TEXT,
            error TEXT,
            lease_owner TEXT,
            lease_expires_at TEXT,
            heartbeat_at TEXT,
            started_at TEXT,
            completed_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tool_calls_run_status ON tool_calls(run_id, status);
        CREATE TABLE IF NOT EXISTS approvals (
            approval_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            tool_call_id TEXT NOT NULL UNIQUE REFERENCES tool_calls(tool_call_id) ON DELETE CASCADE,
            status TEXT NOT NULL,
            request_json TEXT NOT NULL,
            decision_json TEXT,
            requested_at TEXT NOT NULL,
            decided_at TEXT,
            expires_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_approvals_run_status ON approvals(run_id, status);
        CREATE TABLE IF NOT EXISTS claims (
            claim_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            statement TEXT NOT NULL,
            status TEXT NOT NULL,
            importance REAL NOT NULL,
            confidence REAL NOT NULL,
            gaps_json TEXT NOT NULL,
            required_types_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_claims_run_status ON claims(run_id, status);
        CREATE TABLE IF NOT EXISTS claim_edges (
            edge_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            source_claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
            target_claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
            relation TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            UNIQUE(source_claim_id, target_claim_id, relation)
        );
        CREATE TABLE IF NOT EXISTS evidence (
            evidence_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            evidence_type TEXT NOT NULL,
            content TEXT NOT NULL,
            polarity TEXT NOT NULL,
            source_uri TEXT,
            page INTEGER,
            bbox_json TEXT,
            object_type TEXT,
            artifact_path TEXT,
            confidence REAL NOT NULL,
            provenance_json TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(run_id, content_hash)
        );
        CREATE INDEX IF NOT EXISTS idx_evidence_run_type ON evidence(run_id, evidence_type);
        CREATE TABLE IF NOT EXISTS claim_evidence (
            claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
            evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id) ON DELETE CASCADE,
            polarity TEXT NOT NULL,
            PRIMARY KEY(claim_id, evidence_id)
        );
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            step_id TEXT REFERENCES steps(step_id) ON DELETE SET NULL,
            kind TEXT NOT NULL,
            path TEXT NOT NULL,
            sha256 TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trace_spans (
            span_id TEXT PRIMARY KEY,
            trace_id TEXT NOT NULL,
            parent_span_id TEXT,
            run_id TEXT,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            attributes_json TEXT NOT NULL DEFAULT '{}',
            started_at TEXT NOT NULL,
            ended_at TEXT,
            duration_ms REAL,
            error TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_trace_run ON trace_spans(run_id, started_at);
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, event_id);
        CREATE TABLE IF NOT EXISTS llm_calls (
            llm_call_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            stage TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            status TEXT NOT NULL,
            model TEXT NOT NULL,
            thinking_enabled INTEGER NOT NULL,
            max_output_tokens INTEGER NOT NULL,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            cached_tokens INTEGER NOT NULL DEFAULT 0,
            reasoning_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            estimated_cost_usd REAL,
            estimated_cost_cny REAL,
            pricing_tier TEXT,
            pricing_source TEXT,
            pricing_version TEXT,
            finish_reason TEXT,
            request_id TEXT,
            duration_ms REAL NOT NULL DEFAULT 0,
            provider_attempts INTEGER NOT NULL DEFAULT 1,
            error TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_llm_calls_run_stage ON llm_calls(run_id, stage, created_at);
        CREATE TABLE IF NOT EXISTS provider_model_state (
            policy_sha256 TEXT NOT NULL,
            model TEXT NOT NULL,
            state TEXT NOT NULL,
            PRIMARY KEY(policy_sha256, model)
        );
        CREATE TABLE IF NOT EXISTS llm_attempts (
            attempt_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            ordinal INTEGER NOT NULL,
            stage TEXT NOT NULL,
            model TEXT NOT NULL,
            state TEXT NOT NULL,
            reserved_tokens INTEGER NOT NULL,
            usage_known INTEGER NOT NULL,
            cached_tokens_known INTEGER NOT NULL DEFAULT 0,
            request_bytes INTEGER NOT NULL,
            request_chars INTEGER NOT NULL,
            request_sha256 TEXT NOT NULL,
            max_output_tokens INTEGER NOT NULL,
            policy_sha256 TEXT,
            llm_call_id TEXT,
            failure_json TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            UNIQUE(run_id, ordinal)
        );
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            tool_call_id TEXT REFERENCES tool_calls(tool_call_id) ON DELETE SET NULL,
            protocol_hash TEXT NOT NULL,
            protocol_json TEXT NOT NULL,
            result_json TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            completed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS context_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            stage TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            policy_version TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            lineage_json TEXT NOT NULL,
            estimated_tokens_before INTEGER NOT NULL,
            estimated_tokens_after INTEGER NOT NULL,
            compressed INTEGER NOT NULL,
            compression_reason TEXT,
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(run_id, stage, ordinal)
        );
        CREATE INDEX IF NOT EXISTS idx_context_snapshots_run_stage
            ON context_snapshots(run_id, stage, ordinal);
        """
        conn = self.connect()
        try:
            # sqlite3.executescript manages its own transaction boundary; wrapping it in
            # BEGIN/COMMIT would produce "no transaction is active" on some runtimes.
            conn.executescript(schema)
            # Forward-compatible migration for databases created by the first agentic preview.
            tool_columns = {row["name"] for row in conn.execute("PRAGMA table_info(tool_calls)").fetchall()}
            for column, sql_type in (
                ("lease_owner", "TEXT"),
                ("lease_expires_at", "TEXT"),
                ("heartbeat_at", "TEXT"),
            ):
                if column not in tool_columns:
                    conn.execute(f"ALTER TABLE tool_calls ADD COLUMN {column} {sql_type}")
            run_columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
            for column, sql_type in (
                ("used_llm_calls", "INTEGER NOT NULL DEFAULT 0"),
                ("used_input_tokens", "INTEGER NOT NULL DEFAULT 0"),
                ("used_cached_tokens", "INTEGER NOT NULL DEFAULT 0"),
                ("used_reasoning_tokens", "INTEGER NOT NULL DEFAULT 0"),
                ("used_output_tokens", "INTEGER NOT NULL DEFAULT 0"),
                ("used_tokens", "INTEGER NOT NULL DEFAULT 0"),
                ("estimated_cost_usd", "REAL NOT NULL DEFAULT 0"),
                ("estimated_cost_cny", "REAL NOT NULL DEFAULT 0"),
                ("research_outcome", "TEXT"),
            ):
                if column not in run_columns:
                    conn.execute(f"ALTER TABLE runs ADD COLUMN {column} {sql_type}")
            llm_columns = {row["name"] for row in conn.execute("PRAGMA table_info(llm_calls)").fetchall()}
            for column, sql_type in (
                ("estimated_cost_cny", "REAL"),
                ("pricing_tier", "TEXT"),
                ("pricing_source", "TEXT"),
                ("pricing_version", "TEXT"),
            ):
                if column not in llm_columns:
                    conn.execute(f"ALTER TABLE llm_calls ADD COLUMN {column} {sql_type}")
            conn.execute(
                "INSERT INTO runtime_meta(key,value) VALUES('schema_version',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(_SCHEMA_VERSION),),
            )
        finally:
            conn.close()

    def emit_event(self, run_id: str | None, event_type: str, payload: Mapping[str, Any]) -> int:
        with self.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO events(run_id,event_type,payload_json,created_at) VALUES(?,?,?,?)",
                (run_id, event_type, json_dumps(payload), utc_now()),
            )
            return int(cur.lastrowid)

    def create_run(
        self,
        question: str,
        budget: Budget,
        metadata: Mapping[str, Any] | None = None,
        run_id: str | None = None,
    ) -> str:
        run_id = run_id or new_id("run")
        now = utc_now()
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO runs(run_id,question,status,budget_json,metadata_json,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    run_id,
                    question,
                    RunStatus.PENDING.value,
                    json_dumps(budget.to_dict()),
                    json_dumps(dict(metadata or {})),
                    now,
                    now,
                ),
            )
        self.emit_event(run_id, "run.created", {"question": question})
        return run_id

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        value = dict(row)
        value["budget"] = json_loads(value.pop("budget_json"), {})
        value["metadata"] = json_loads(value.pop("metadata_json"), {})
        return value

    def list_runs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        bounded = max(1, min(200, int(limit)))
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY updated_at DESC, run_id DESC LIMIT ?",
                (bounded,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            value["budget"] = json_loads(value.pop("budget_json"), {})
            value["metadata"] = json_loads(value.pop("metadata_json"), {})
            result.append(value)
        return result

    def update_run_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        current_step_id: str | None = None,
        error: str | None = None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE runs SET status=?, current_step_id=?, error=?, updated_at=? WHERE run_id=?",
                (status.value, current_step_id, error, utc_now(), run_id),
            )
        self.emit_event(run_id, "run.status", {"status": status.value, "error": error})

    def increment_usage(self, run_id: str, *, steps: int = 0, tool_calls: int = 0) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE runs SET used_steps=used_steps+?, used_tool_calls=used_tool_calls+?, updated_at=? WHERE run_id=?",
                (steps, tool_calls, utc_now(), run_id),
            )

    def reserve_llm_attempt(self, run_id: str, *, max_llm_calls: int | None) -> int:
        """Atomically reserve one provider request before any network call."""

        with self.transaction() as conn:
            row = conn.execute(
                "SELECT used_llm_calls FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown run_id: {run_id}")
            used = int(row["used_llm_calls"])
            if max_llm_calls is not None and used >= int(max_llm_calls):
                raise RuntimeError(
                    f"LLM-call budget exhausted ({used}/{int(max_llm_calls)})"
                )
            used += 1
            conn.execute(
                "UPDATE runs SET used_llm_calls=?,updated_at=? WHERE run_id=?",
                (used, utc_now(), run_id),
            )
        return used

    def record_llm_call(
        self,
        *,
        run_id: str,
        stage: str,
        attempt: int,
        status: str,
        model: str,
        thinking_enabled: bool,
        max_output_tokens: int,
        prompt_tokens: int = 0,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        completion_tokens: int = 0,
        estimated_cost_usd: float | None = None,
        estimated_cost_cny: float | None = None,
        pricing_tier: str | None = None,
        pricing_source: str | None = None,
        pricing_version: str | None = None,
        finish_reason: str | None = None,
        request_id: str | None = None,
        duration_ms: float = 0.0,
        provider_attempts: int = 1,
        error: str | None = None,
    ) -> str:
        llm_call_id = new_id("llm")
        prompt_tokens = max(0, int(prompt_tokens))
        cached_tokens = max(0, int(cached_tokens))
        reasoning_tokens = max(0, int(reasoning_tokens))
        completion_tokens = max(0, int(completion_tokens))
        total_tokens = prompt_tokens + completion_tokens
        now = utc_now()
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO llm_calls(llm_call_id,run_id,stage,attempt,status,model,"
                "thinking_enabled,max_output_tokens,prompt_tokens,cached_tokens,reasoning_tokens,"
                "completion_tokens,total_tokens,estimated_cost_usd,estimated_cost_cny,pricing_tier,"
                "pricing_source,pricing_version,finish_reason,request_id,duration_ms,provider_attempts,"
                "error,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    llm_call_id,
                    run_id,
                    stage,
                    int(attempt),
                    str(status),
                    str(model),
                    int(bool(thinking_enabled)),
                    int(max_output_tokens),
                    prompt_tokens,
                    cached_tokens,
                    reasoning_tokens,
                    completion_tokens,
                    total_tokens,
                    estimated_cost_usd,
                    estimated_cost_cny,
                    pricing_tier,
                    pricing_source,
                    pricing_version,
                    finish_reason,
                    request_id,
                    float(duration_ms),
                    max(1, int(provider_attempts)),
                    error,
                    now,
                ),
            )
            conn.execute(
                "UPDATE runs SET used_input_tokens=used_input_tokens+?,"
                "used_cached_tokens=used_cached_tokens+?,"
                "used_reasoning_tokens=used_reasoning_tokens+?,"
                "used_output_tokens=used_output_tokens+?,used_tokens=used_tokens+?,"
                "estimated_cost_usd=estimated_cost_usd+?,"
                "estimated_cost_cny=estimated_cost_cny+?,updated_at=? WHERE run_id=?",
                (
                    prompt_tokens,
                    cached_tokens,
                    reasoning_tokens,
                    completion_tokens,
                    total_tokens,
                    float(estimated_cost_usd or 0.0),
                    float(estimated_cost_cny or 0.0),
                    now,
                    run_id,
                ),
            )
        self.emit_event(
            run_id,
            "llm.call",
            {
                "llm_call_id": llm_call_id,
                "stage": stage,
                "status": status,
                "total_tokens": total_tokens,
                "finish_reason": finish_reason,
                "duration_ms": float(duration_ms),
                "estimated_cost_cny": estimated_cost_cny,
                "pricing_tier": pricing_tier,
                "error": error,
            },
        )
        return llm_call_id

    def list_llm_calls(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM llm_calls WHERE run_id=? ORDER BY created_at,llm_call_id",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_run_outcome(
        self,
        run_id: str,
        outcome: ResearchOutcome | None,
    ) -> None:
        value = outcome.value if isinstance(outcome, ResearchOutcome) else None
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT research_outcome FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown run_id: {run_id}")
            if row["research_outcome"] == value:
                return
            conn.execute(
                "UPDATE runs SET research_outcome=?,updated_at=? WHERE run_id=?",
                (value, utc_now(), run_id),
            )
        self.emit_event(run_id, "run.outcome", {"outcome": value})

    def update_run_metadata(self, run_id: str, updates: Mapping[str, Any]) -> dict[str, Any]:
        with self.transaction() as conn:
            row = conn.execute("SELECT metadata_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown run_id: {run_id}")
            metadata = json_loads(row["metadata_json"], {})
            metadata.update(dict(updates))
            conn.execute(
                "UPDATE runs SET metadata_json=?,updated_at=? WHERE run_id=?",
                (json_dumps(metadata), utc_now(), run_id),
            )
        self.emit_event(run_id, "run.metadata", {"keys": sorted(updates)})
        return metadata

    def create_step(
        self,
        run_id: str,
        kind: str,
        input_data: Mapping[str, Any],
        status: StepStatus = StepStatus.PENDING,
    ) -> str:
        with self.transaction() as conn:
            ordinal = int(
                conn.execute("SELECT COALESCE(MAX(ordinal),0)+1 FROM steps WHERE run_id=?", (run_id,)).fetchone()[0]
            )
            step_id = new_id("step")
            conn.execute(
                "INSERT INTO steps(step_id,run_id,ordinal,kind,status,input_json,created_at) VALUES(?,?,?,?,?,?,?)",
                (step_id, run_id, ordinal, kind, status.value, json_dumps(dict(input_data)), utc_now()),
            )
            conn.execute(
                "UPDATE runs SET current_step_id=?, used_steps=used_steps+1, updated_at=? WHERE run_id=?",
                (step_id, utc_now(), run_id),
            )
        self.emit_event(run_id, "step.created", {"step_id": step_id, "kind": kind, "ordinal": ordinal})
        return step_id

    def update_step(
        self,
        step_id: str,
        status: StepStatus,
        *,
        output: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        now = utc_now()
        started = now if status == StepStatus.RUNNING else None
        completed = now if status in {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED} else None
        with self.transaction() as conn:
            row = conn.execute("SELECT run_id,started_at FROM steps WHERE step_id=?", (step_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown step_id: {step_id}")
            conn.execute(
                "UPDATE steps SET status=?, output_json=COALESCE(?,output_json), error=?, "
                "started_at=COALESCE(started_at,?), completed_at=COALESCE(?,completed_at) WHERE step_id=?",
                (
                    status.value,
                    json_dumps(dict(output)) if output is not None else None,
                    error,
                    started,
                    completed,
                    step_id,
                ),
            )
        self.emit_event(str(row["run_id"]), "step.status", {"step_id": step_id, "status": status.value})

    def get_step(self, step_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM steps WHERE step_id=?", (step_id,)).fetchone()
        if row is None:
            raise KeyError(step_id)
        value = dict(row)
        value["input"] = json_loads(value.pop("input_json"), {})
        value["output"] = json_loads(value.pop("output_json"), None)
        return value

    def list_steps(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM steps WHERE run_id=? ORDER BY ordinal", (run_id,)).fetchall()
        result=[]
        for row in rows:
            item=dict(row)
            item["input"] = json_loads(item.pop("input_json"), {})
            item["output"] = json_loads(item.pop("output_json"), None)
            result.append(item)
        return result

    def create_step_tool_call(
        self, *, run_id: str, tool_name: str, args: Mapping[str, Any], risk_level: str,
        idempotency_key: str, max_attempts: int,
    ) -> tuple[str, bool]:
        """Register one logical action, its step, counters and events atomically.

        Recovery may revisit a returned model decision before its task state was
        committed. Reuse its exact registration without creating an orphan step.
        """
        arguments, now = json_dumps(dict(args)), utc_now()
        with self.transaction() as conn:
            existing = conn.execute('''SELECT c.*,s.run_id AS step_run_id,s.kind AS step_kind,
                s.input_json AS step_input FROM tool_calls c LEFT JOIN steps s ON s.step_id=c.step_id
                WHERE c.idempotency_key=?''', (idempotency_key,)).fetchone()
            if existing:
                if (existing['run_id'] != run_id or existing['tool_name'] != tool_name
                        or existing['args_json'] != arguments or existing['risk_level'] != risk_level
                        or existing['max_attempts'] != max_attempts or existing['step_run_id'] != run_id
                        or existing['step_kind'] != tool_name or existing['step_input'] != arguments):
                    raise RuntimeError('Logical action registration binding changed')
                return str(existing['tool_call_id']), True
            ordinal = int(conn.execute('SELECT COALESCE(MAX(ordinal),0)+1 FROM steps WHERE run_id=?',
                                       (run_id,)).fetchone()[0])
            step_id, call_id = new_id('step'), new_id('tool')
            conn.execute('''INSERT INTO steps(step_id,run_id,ordinal,kind,status,input_json,created_at)
                VALUES(?,?,?,?,?,?,?)''', (step_id, run_id, ordinal, tool_name, 'PENDING', arguments, now))
            conn.execute('''INSERT INTO tool_calls(tool_call_id,run_id,step_id,tool_name,args_json,risk_level,
                status,idempotency_key,max_attempts,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
                (call_id, run_id, step_id, tool_name, arguments, risk_level, 'PENDING', idempotency_key,
                 max_attempts, now, now))
            conn.execute('''UPDATE runs SET current_step_id=?,used_steps=used_steps+1,
                used_tool_calls=used_tool_calls+1,updated_at=? WHERE run_id=?''', (step_id, now, run_id))
            for kind, payload in (
                ('step.created', {'step_id': step_id, 'kind': tool_name, 'ordinal': ordinal}),
                ('tool.created', {'tool_call_id': call_id, 'tool': tool_name}),
            ):
                conn.execute('INSERT INTO events(run_id,event_type,payload_json,created_at) VALUES(?,?,?,?)',
                             (run_id, kind, json_dumps(payload), now))
        return call_id, False

    def create_tool_call(
        self,
        *,
        run_id: str,
        step_id: str,
        tool_name: str,
        args: Mapping[str, Any],
        risk_level: str,
        idempotency_key: str,
        max_attempts: int,
    ) -> tuple[str, bool]:
        now = utc_now()
        tool_call_id = new_id("tool")
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT tool_call_id FROM tool_calls WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            if existing:
                return str(existing["tool_call_id"]), True
            conn.execute(
                "INSERT INTO tool_calls(tool_call_id,run_id,step_id,tool_name,args_json,risk_level,status,"
                "idempotency_key,max_attempts,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    tool_call_id,
                    run_id,
                    step_id,
                    tool_name,
                    json_dumps(dict(args)),
                    risk_level,
                    ToolCallStatus.PENDING.value,
                    idempotency_key,
                    max_attempts,
                    now,
                    now,
                ),
            )
            conn.execute(
                "UPDATE runs SET used_tool_calls=used_tool_calls+1, updated_at=? WHERE run_id=?",
                (now, run_id),
            )
        self.emit_event(run_id, "tool.created", {"tool_call_id": tool_call_id, "tool": tool_name})
        return tool_call_id, False

    def get_tool_call(self, tool_call_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM tool_calls WHERE tool_call_id=?", (tool_call_id,)).fetchone()
        if row is None:
            raise KeyError(tool_call_id)
        value=dict(row)
        value["args"] = json_loads(value.pop("args_json"), {})
        value["result"] = json_loads(value.pop("result_json"), None)
        return value

    def list_tool_calls(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows=conn.execute("SELECT * FROM tool_calls WHERE run_id=? ORDER BY created_at",(run_id,)).fetchall()
        result=[]
        for row in rows:
            item=dict(row)
            item["args"] = json_loads(item.pop("args_json"), {})
            item["result"] = json_loads(item.pop("result_json"), None)
            result.append(item)
        return result

    def update_tool_call(
        self,
        tool_call_id: str,
        status: ToolCallStatus,
        *,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
        increment_attempt: bool = False,
    ) -> None:
        now = utc_now()
        terminal = status in {
            ToolCallStatus.COMPLETED,
            ToolCallStatus.FAILED,
            ToolCallStatus.REJECTED,
        }
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT run_id FROM tool_calls WHERE tool_call_id=?",
                (tool_call_id,),
            ).fetchone()
            if row is None:
                raise KeyError(tool_call_id)
            if terminal:
                conn.execute(
                    "UPDATE tool_calls SET status=?,result_json=COALESCE(?,result_json),error=?,"
                    "attempt=attempt+?,completed_at=?,lease_owner=NULL,lease_expires_at=NULL,"
                    "heartbeat_at=NULL,updated_at=? WHERE tool_call_id=?",
                    (
                        status.value,
                        json_dumps(dict(result)) if result is not None else None,
                        error,
                        1 if increment_attempt else 0,
                        now,
                        now,
                        tool_call_id,
                    ),
                )
            else:
                conn.execute(
                    "UPDATE tool_calls SET status=?,result_json=COALESCE(?,result_json),error=?,"
                    "attempt=attempt+?,started_at=CASE WHEN ?='RUNNING' THEN COALESCE(started_at,?) "
                    "ELSE started_at END,updated_at=? WHERE tool_call_id=?",
                    (
                        status.value,
                        json_dumps(dict(result)) if result is not None else None,
                        error,
                        1 if increment_attempt else 0,
                        status.value,
                        now,
                        now,
                        tool_call_id,
                    ),
                )
        self.emit_event(
            str(row["run_id"]),
            "tool.status",
            {"tool_call_id": tool_call_id, "status": status.value},
        )

    def acquire_tool_lease(
        self,
        tool_call_id: str,
        *,
        owner: str,
        lease_seconds: int,
    ) -> bool:
        from datetime import datetime, timedelta, timezone

        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat(timespec="milliseconds")
        expires = (now_dt + timedelta(seconds=max(15, int(lease_seconds)))).isoformat(timespec="milliseconds")
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT status,attempt,max_attempts,lease_owner,lease_expires_at FROM tool_calls WHERE tool_call_id=?",
                (tool_call_id,),
            ).fetchone()
            if row is None:
                raise KeyError(tool_call_id)
            status = ToolCallStatus(str(row["status"]))
            if status in {ToolCallStatus.COMPLETED, ToolCallStatus.REJECTED}:
                return False
            lease_expires = row["lease_expires_at"]
            active_other_lease = (
                status == ToolCallStatus.RUNNING
                and lease_expires
                and str(lease_expires) > now
                and row["lease_owner"] != owner
            )
            if active_other_lease:
                return False
            attempt = int(row["attempt"])
            max_attempts = int(row["max_attempts"])
            # A stale RUNNING call is a recovery attempt. Tools that permit recovery
            # explicitly set max_attempts > 1 and must be idempotent at the handler layer.
            if attempt >= max_attempts and status != ToolCallStatus.PENDING:
                return False
            conn.execute(
                "UPDATE tool_calls SET status=?,attempt=attempt+1,lease_owner=?,lease_expires_at=?,"
                "heartbeat_at=?,started_at=COALESCE(started_at,?),updated_at=? WHERE tool_call_id=?",
                (
                    ToolCallStatus.RUNNING.value,
                    owner,
                    expires,
                    now,
                    now,
                    now,
                    tool_call_id,
                ),
            )
        return True

    def heartbeat_tool_lease(self, tool_call_id: str, *, owner: str, lease_seconds: int) -> bool:
        from datetime import datetime, timedelta, timezone

        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat(timespec="milliseconds")
        expires = (now_dt + timedelta(seconds=max(15, int(lease_seconds)))).isoformat(timespec="milliseconds")
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE tool_calls SET heartbeat_at=?,lease_expires_at=?,updated_at=? "
                "WHERE tool_call_id=? AND lease_owner=? AND status=?",
                (now, expires, now, tool_call_id, owner, ToolCallStatus.RUNNING.value),
            )
            return cur.rowcount == 1

    def tool_lease_is_active(self, tool_call_id: str) -> bool:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        with self.connect() as conn:
            row = conn.execute(
                "SELECT status,lease_expires_at FROM tool_calls WHERE tool_call_id=?",
                (tool_call_id,),
            ).fetchone()
        return bool(
            row
            and row["status"] == ToolCallStatus.RUNNING.value
            and row["lease_expires_at"]
            and str(row["lease_expires_at"]) > now
        )

    def create_approval(
        self,
        run_id: str,
        tool_call_id: str,
        request: Mapping[str, Any],
        *,
        expires_at: str | None = None,
    ) -> str:
        with self.transaction() as conn:
            existing=conn.execute("SELECT approval_id FROM approvals WHERE tool_call_id=?",(tool_call_id,)).fetchone()
            if existing:
                return str(existing["approval_id"])
            approval_id=new_id("approval")
            conn.execute(
                "INSERT INTO approvals(approval_id,run_id,tool_call_id,status,request_json,requested_at,expires_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    approval_id,
                    run_id,
                    tool_call_id,
                    ApprovalStatus.PENDING.value,
                    json_dumps(dict(request)),
                    utc_now(),
                    expires_at,
                ),
            )
        self.emit_event(run_id, "approval.requested", {"approval_id": approval_id, "tool_call_id": tool_call_id})
        return approval_id

    def decide_approval(
        self,
        approval_id: str,
        status: ApprovalStatus,
        decision: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if status not in {ApprovalStatus.APPROVED, ApprovalStatus.REJECTED}:
            raise ValueError("Approval decision must be APPROVED or REJECTED")
        with self.transaction() as conn:
            row=conn.execute("SELECT * FROM approvals WHERE approval_id=?",(approval_id,)).fetchone()
            if row is None:
                raise KeyError(approval_id)
            current=ApprovalStatus(str(row["status"]))
            if current != ApprovalStatus.PENDING:
                if current == status:
                    pass
                else:
                    raise RuntimeError(f"Approval already decided as {current.value}")
            else:
                conn.execute(
                    "UPDATE approvals SET status=?, decision_json=?, decided_at=? WHERE approval_id=?",
                    (status.value,json_dumps(dict(decision or {})),utc_now(),approval_id),
                )
                call_status=ToolCallStatus.APPROVED if status==ApprovalStatus.APPROVED else ToolCallStatus.REJECTED
                conn.execute(
                    "UPDATE tool_calls SET status=?, updated_at=? WHERE tool_call_id=?",
                    (call_status.value,utc_now(),str(row["tool_call_id"])),
                )
        self.emit_event(str(row["run_id"]), "approval.decided", {"approval_id":approval_id,"status":status.value})
        return self.get_approval(approval_id)

    def get_approval(self, approval_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row=conn.execute("SELECT * FROM approvals WHERE approval_id=?",(approval_id,)).fetchone()
        if row is None:
            raise KeyError(approval_id)
        value=dict(row)
        value["request"] = json_loads(value.pop("request_json"), {})
        value["decision"] = json_loads(value.pop("decision_json"), None)
        return value

    def list_approvals(self, run_id: str, status: ApprovalStatus | None = None) -> list[dict[str, Any]]:
        sql="SELECT * FROM approvals WHERE run_id=?"
        params:list[Any]=[run_id]
        if status:
            sql += " AND status=?"
            params.append(status.value)
        sql += " ORDER BY requested_at"
        with self.connect() as conn:
            rows=conn.execute(sql,params).fetchall()
        result=[]
        for row in rows:
            item=dict(row)
            item["request"] = json_loads(item.pop("request_json"), {})
            item["decision"] = json_loads(item.pop("decision_json"), None)
            result.append(item)
        return result

    def add_claim(self, claim: ClaimRecord) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO claims(claim_id,run_id,statement,status,importance,confidence,gaps_json,"
                "required_types_json,metadata_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    claim.claim_id,
                    claim.run_id,
                    claim.statement,
                    claim.status.value,
                    float(claim.importance),
                    float(claim.confidence),
                    json_dumps(claim.gaps),
                    json_dumps([x.value for x in claim.required_evidence_types]),
                    json_dumps(claim.metadata),
                    claim.created_at,
                    claim.updated_at,
                ),
            )
        self.emit_event(claim.run_id, "claim.created", {"claim_id":claim.claim_id,"statement":claim.statement})

    def update_claim(self, claim: ClaimRecord) -> None:
        claim.updated_at=utc_now()
        with self.transaction() as conn:
            conn.execute(
                "UPDATE claims SET statement=?,status=?,importance=?,confidence=?,gaps_json=?,"
                "required_types_json=?,metadata_json=?,updated_at=? WHERE claim_id=?",
                (
                    claim.statement,
                    claim.status.value,
                    float(claim.importance),
                    float(claim.confidence),
                    json_dumps(claim.gaps),
                    json_dumps([x.value for x in claim.required_evidence_types]),
                    json_dumps(claim.metadata),
                    claim.updated_at,
                    claim.claim_id,
                ),
            )
        self.emit_event(claim.run_id,"claim.updated",{"claim_id":claim.claim_id,"status":claim.status.value})

    def _claim_from_row(self,row:sqlite3.Row) -> ClaimRecord:
        return ClaimRecord(
            claim_id=str(row["claim_id"]),
            run_id=str(row["run_id"]),
            statement=str(row["statement"]),
            status=ClaimStatus(str(row["status"])),
            importance=float(row["importance"]),
            confidence=float(row["confidence"]),
            gaps=list(json_loads(row["gaps_json"],[])),
            required_evidence_types=[EvidenceType(x) for x in json_loads(row["required_types_json"],[])],
            metadata=dict(json_loads(row["metadata_json"],{})),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def get_claim(self, claim_id: str) -> ClaimRecord:
        with self.connect() as conn:
            row=conn.execute("SELECT * FROM claims WHERE claim_id=?",(claim_id,)).fetchone()
        if row is None:
            raise KeyError(claim_id)
        return self._claim_from_row(row)

    def list_claims(self, run_id: str) -> list[ClaimRecord]:
        with self.connect() as conn:
            rows=conn.execute("SELECT * FROM claims WHERE run_id=? ORDER BY created_at",(run_id,)).fetchall()
        return [self._claim_from_row(row) for row in rows]

    def add_claim_edge(self, run_id: str, source: str, target: str, relation: str, metadata: Mapping[str,Any]|None=None) -> str:
        edge_id=new_id("edge")
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO claim_edges(edge_id,run_id,source_claim_id,target_claim_id,relation,metadata_json) "
                "VALUES(?,?,?,?,?,?)",
                (edge_id,run_id,source,target,relation,json_dumps(dict(metadata or {}))),
            )
        return edge_id

    def list_claim_edges(self, run_id: str) -> list[dict[str,Any]]:
        with self.connect() as conn:
            rows=conn.execute("SELECT * FROM claim_edges WHERE run_id=? ORDER BY rowid",(run_id,)).fetchall()
        result=[]
        for row in rows:
            item=dict(row)
            item["metadata"] = json_loads(item.pop("metadata_json"),{})
            result.append(item)
        return result

    def add_evidence(self, evidence: EvidenceRecord) -> tuple[str,bool]:
        with self.transaction() as conn:
            existing=conn.execute(
                "SELECT evidence_id FROM evidence WHERE run_id=? AND content_hash=?",
                (evidence.run_id,evidence.content_hash),
            ).fetchone()
            if existing:
                evidence_id=str(existing["evidence_id"])
                reused=True
            else:
                conn.execute(
                    "INSERT INTO evidence(evidence_id,run_id,evidence_type,content,polarity,source_uri,page,bbox_json,"
                    "object_type,artifact_path,confidence,provenance_json,content_hash,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        evidence.evidence_id,evidence.run_id,evidence.evidence_type.value,evidence.content,
                        evidence.polarity.value,evidence.source_uri,evidence.page,
                        json_dumps(evidence.bbox) if evidence.bbox is not None else None,
                        evidence.object_type,evidence.artifact_path,float(evidence.confidence),
                        json_dumps(evidence.provenance),evidence.content_hash,evidence.created_at,
                    ),
                )
                evidence_id=evidence.evidence_id
                reused=False
            if evidence.claim_id:
                conn.execute(
                    "INSERT OR IGNORE INTO claim_evidence(claim_id,evidence_id,polarity) VALUES(?,?,?)",
                    (evidence.claim_id,evidence_id,evidence.polarity.value),
                )
        self.emit_event(evidence.run_id,"evidence.added",{"evidence_id":evidence_id,"type":evidence.evidence_type.value,"reused":reused})
        return evidence_id,reused

    def link_evidence(self, claim_id: str, evidence_id: str, polarity: EvidencePolarity) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO claim_evidence(claim_id,evidence_id,polarity) VALUES(?,?,?) "
                "ON CONFLICT(claim_id,evidence_id) DO UPDATE SET polarity=excluded.polarity",
                (claim_id,evidence_id,polarity.value),
            )

    def list_evidence(self, run_id: str, claim_id: str | None = None) -> list[dict[str,Any]]:
        if claim_id:
            sql=("SELECT e.*,ce.polarity AS link_polarity,ce.claim_id AS linked_claim_ids,"
                 "ce.claim_id || '=' || ce.polarity AS linked_claim_polarities "
                 "FROM evidence e JOIN claim_evidence ce ON e.evidence_id=ce.evidence_id "
                 "JOIN claims c ON c.claim_id=ce.claim_id "
                 "WHERE c.run_id=? AND ce.claim_id=? ORDER BY e.created_at")
            params=(run_id,claim_id)
        else:
            sql=("SELECT e.*,NULL AS link_polarity,(SELECT GROUP_CONCAT(ce.claim_id) "
                 "FROM claim_evidence ce JOIN claims c ON c.claim_id=ce.claim_id "
                 "WHERE ce.evidence_id=e.evidence_id AND c.run_id=?) AS linked_claim_ids,"
                 "(SELECT GROUP_CONCAT(ce.claim_id || '=' || ce.polarity, '|') "
                 "FROM claim_evidence ce JOIN claims c ON c.claim_id=ce.claim_id "
                 "WHERE ce.evidence_id=e.evidence_id AND c.run_id=?) AS linked_claim_polarities "
                 "FROM evidence e WHERE e.run_id=? OR EXISTS (SELECT 1 FROM claim_evidence ce "
                 "JOIN claims c ON c.claim_id=ce.claim_id WHERE ce.evidence_id=e.evidence_id "
                 "AND c.run_id=?) ORDER BY e.created_at")
            params=(run_id,run_id,run_id,run_id)
        with self.connect() as conn:
            rows=conn.execute(sql,params).fetchall()
        result=[]
        for row in rows:
            item=dict(row)
            item["bbox"] = json_loads(item.pop("bbox_json"),None)
            item["provenance"] = json_loads(item.pop("provenance_json"),{})
            item["claim_ids"] = [
                value for value in str(item.pop("linked_claim_ids") or "").split(",") if value
            ]
            claim_polarities: dict[str, str] = {}
            for pair in str(item.pop("linked_claim_polarities") or "").split("|"):
                if "=" not in pair:
                    continue
                linked_claim_id, linked_polarity = pair.split("=", 1)
                if linked_claim_id:
                    claim_polarities[linked_claim_id] = linked_polarity
            item["claim_polarities"] = claim_polarities
            item["claim_id"] = item["claim_ids"][0] if len(item["claim_ids"]) == 1 else None
            result.append(item)
        return result

    def find_visual_evidence_by_image_sha(
        self,
        image_sha256: str,
        *,
        document_sha256: str | None = None,
        exclude_run_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Find persisted provider reads of the exact same immutable crop.

        The caller must still re-audit the stored response against its current
        prompt and gates. This lookup is only a bounded cache candidate search;
        it never treats a prior verdict as current truth.
        """

        digest = str(image_sha256 or "").strip().casefold()
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            return []
        sql = (
            "SELECT * FROM evidence WHERE provenance_json LIKE ? "
            "ORDER BY created_at DESC LIMIT ?"
        )
        with self.connect() as conn:
            rows = conn.execute(sql, (f"%{digest}%", max(1, min(int(limit), 100)))).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            provenance = json_loads(item.pop("provenance_json"), {})
            vision = dict(provenance.get("vision_metadata") or {})
            if str(vision.get("image_sha256") or "").casefold() != digest:
                continue
            if exclude_run_id and str(item.get("run_id")) == str(exclude_run_id):
                continue
            source_sha = str(
                provenance.get("document_sha256")
                or provenance.get("paper_sha256")
                or ""
            )
            if document_sha256 and source_sha != str(document_sha256):
                continue
            item["bbox"] = json_loads(item.pop("bbox_json"), None)
            item["provenance"] = provenance
            result.append(item)
        return result

    def add_artifact(self, run_id: str, step_id: str | None, artifact: Mapping[str,Any]) -> str:
        artifact_id=str(artifact.get("artifact_id") or new_id("artifact"))
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO artifacts(artifact_id,run_id,step_id,kind,path,sha256,metadata_json,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    artifact_id,run_id,step_id,str(artifact.get("kind","file")),str(artifact["path"]),
                    artifact.get("sha256"),json_dumps(dict(artifact.get("metadata") or {})),utc_now(),
                ),
            )
        return artifact_id

    def list_artifacts(self, run_id: str) -> list[dict[str,Any]]:
        with self.connect() as conn:
            rows=conn.execute("SELECT * FROM artifacts WHERE run_id=? ORDER BY created_at",(run_id,)).fetchall()
        result=[]
        for row in rows:
            item=dict(row)
            item["metadata"] = json_loads(item.pop("metadata_json"),{})
            result.append(item)
        return result

    def add_context_snapshot(
        self,
        *,
        run_id: str,
        stage: str,
        policy_version: str,
        payload: Mapping[str, Any],
        lineage: Mapping[str, Any],
        estimated_tokens_before: int,
        estimated_tokens_after: int,
        compressed: bool,
        compression_reason: str | None = None,
    ) -> str:
        """Persist the exact model view and its durable-source lineage.

        The compact payload is replay state, not a truth store: evidence IDs in
        ``lineage`` always point back to the immutable evidence ledger.
        """

        snapshot_id = new_id("context")
        created_at = utc_now()
        payload_dict = dict(payload)
        lineage_dict = dict(lineage)
        content_hash = stable_hash(
            run_id,
            stage,
            policy_version,
            payload_dict,
            lineage_dict,
        )
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(ordinal), 0) AS current_ordinal "
                "FROM context_snapshots WHERE run_id=? AND stage=?",
                (run_id, str(stage)),
            ).fetchone()
            ordinal = int(row["current_ordinal"] if row is not None else 0) + 1
            conn.execute(
                "INSERT INTO context_snapshots(snapshot_id,run_id,stage,ordinal,policy_version,"
                "payload_json,lineage_json,estimated_tokens_before,estimated_tokens_after,compressed,"
                "compression_reason,content_hash,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    snapshot_id,
                    run_id,
                    str(stage),
                    ordinal,
                    str(policy_version),
                    json_dumps(payload_dict),
                    json_dumps(lineage_dict),
                    max(0, int(estimated_tokens_before)),
                    max(0, int(estimated_tokens_after)),
                    int(bool(compressed)),
                    compression_reason,
                    content_hash,
                    created_at,
                ),
            )
        self.emit_event(
            run_id,
            "context.snapshot",
            {
                "snapshot_id": snapshot_id,
                "stage": str(stage),
                "ordinal": ordinal,
                "policy_version": str(policy_version),
                "estimated_tokens_before": max(0, int(estimated_tokens_before)),
                "estimated_tokens_after": max(0, int(estimated_tokens_after)),
                "compressed": bool(compressed),
                "compression_reason": compression_reason,
            },
        )
        return snapshot_id

    @staticmethod
    def _context_snapshot_from_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["payload"] = json_loads(item.pop("payload_json"), {})
        item["lineage"] = json_loads(item.pop("lineage_json"), {})
        item["compressed"] = bool(item["compressed"])
        return item

    def get_context_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM context_snapshots WHERE snapshot_id=?",
                (snapshot_id,),
            ).fetchone()
        if row is None:
            raise KeyError(snapshot_id)
        return self._context_snapshot_from_row(row)

    def list_context_snapshots(
        self,
        run_id: str,
        *,
        stage: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        bounded = max(1, min(500, int(limit)))
        params: list[Any] = [run_id]
        sql = "SELECT * FROM context_snapshots WHERE run_id=?"
        if stage is not None:
            sql += " AND stage=?"
            params.append(str(stage))
        sql += " ORDER BY created_at DESC, ordinal DESC LIMIT ?"
        params.append(bounded)
        with self.connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        # Present snapshots in execution order while applying the limit to the
        # newest records, so UI replay remains intuitive.
        return [self._context_snapshot_from_row(row) for row in reversed(rows)]

    def create_span(self, data: Mapping[str,Any]) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO trace_spans(span_id,trace_id,parent_span_id,run_id,name,kind,status,attributes_json,started_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    data["span_id"],data["trace_id"],data.get("parent_span_id"),data.get("run_id"),
                    data["name"],data.get("kind","internal"),data.get("status","RUNNING"),
                    json_dumps(dict(data.get("attributes") or {})),data["started_at"],
                ),
            )

    def finish_span(self, span_id: str, *, status: str, ended_at: str, duration_ms: float, error: str|None=None, attributes: Mapping[str,Any]|None=None) -> None:
        with self.transaction() as conn:
            row=conn.execute("SELECT attributes_json FROM trace_spans WHERE span_id=?",(span_id,)).fetchone()
            current=json_loads(row["attributes_json"],{}) if row else {}
            current.update(dict(attributes or {}))
            conn.execute(
                "UPDATE trace_spans SET status=?,ended_at=?,duration_ms=?,error=?,attributes_json=? WHERE span_id=?",
                (status,ended_at,float(duration_ms),error,json_dumps(current),span_id),
            )

    def list_spans(self, run_id: str) -> list[dict[str,Any]]:
        with self.connect() as conn:
            rows=conn.execute("SELECT * FROM trace_spans WHERE run_id=? ORDER BY started_at",(run_id,)).fetchall()
        result=[]
        for row in rows:
            item=dict(row)
            item["attributes"] = json_loads(item.pop("attributes_json"),{})
            result.append(item)
        return result

    def list_events(
        self,
        run_id: str,
        *,
        after_event_id: int = 0,
        limit: int = 500,
    ) -> list[dict[str,Any]]:
        bounded = max(1, min(2_000, int(limit)))
        with self.connect() as conn:
            rows=conn.execute(
                "SELECT * FROM events WHERE run_id=? AND event_id>? ORDER BY event_id LIMIT ?",
                (run_id, int(after_event_id), bounded),
            ).fetchall()
        result=[]
        for row in rows:
            item=dict(row)
            item["payload"] = json_loads(item.pop("payload_json"),{})
            result.append(item)
        return result
