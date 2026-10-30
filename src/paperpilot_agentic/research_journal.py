"""Revisioned research state, independent of chat windows and model summaries.

The journal is a navigation ledger, NEVER a replacement for original evidence.
Only goals/constraints enter a model window. Results remain linked to owned runs.
"""

import json

from .models import utc_now


class ResearchJournal:
    def __init__(self, db):
        self.db = db
        with db.transaction() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS research_state_revisions(
                conversation_id TEXT NOT NULL, revision INTEGER NOT NULL,
                value_json TEXT NOT NULL, source_run_id TEXT, created_at TEXT NOT NULL,
                PRIMARY KEY(conversation_id,revision))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS research_results(
                run_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, kind TEXT NOT NULL,
                result_json TEXT NOT NULL, updated_at TEXT NOT NULL)""")

    def state(self, cid):
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM research_state_revisions WHERE conversation_id=? "
                "ORDER BY revision DESC LIMIT 1",
                (cid,),
            ).fetchone()
        return (
            {**json.loads(row["value_json"]), "revision": row["revision"]}
            if row
            else {"revision": 0, "goal": "", "constraints": [], "open_questions": []}
        )

    def revise(self, cid, value, *, expected_revision, source_run_id):
        if set(value) != {"goal", "constraints", "open_questions"}:
            raise ValueError("Research state requires goal, constraints and open_questions")
        if not isinstance(value["goal"], str) or not value["goal"].strip():
            raise ValueError("A research goal is required")
        for field in ("constraints", "open_questions"):
            if (
                not isinstance(value[field], list)
                or len(value[field]) > 30
                or any(not isinstance(v, str) or not v.strip() for v in value[field])
            ):
                raise ValueError("State entries must be bounded nonempty strings")
        encoded = json.dumps(value, ensure_ascii=False)
        if len(encoded.encode()) > 8000:
            raise ValueError(
                "Research state exceeds its work window; keep original detail in materials"
            )
        with self.db.transaction() as conn:
            revision = conn.execute(
                "SELECT COALESCE(MAX(revision),0) FROM research_state_revisions "
                "WHERE conversation_id=?",
                (cid,),
            ).fetchone()[0]
            if revision != expected_revision:
                raise ValueError("Research state changed; reload before editing")
            conn.execute(
                "INSERT INTO research_state_revisions VALUES(?,?,?,?,?)",
                (cid, revision + 1, encoded, source_run_id, utc_now()),
            )
        return self.state(cid)

    def record(self, run_id):
        run = self.db.get_run(run_id)
        if run["status"] not in {"COMPLETED", "FAILED", "CANCELLED", "BUDGET_EXHAUSTED"}:
            return
        meta = run["metadata"]
        cid = meta.get("conversation_id") or run_id
        final = meta.get("final_answer") or {}
        design = meta.get("research_design") or {}
        experiment = meta.get("method_comparison") or meta.get("method_validation") or {}
        value = {
            "run_id": run_id,
            "question": run["question"],
            "status": run["status"],
            "outcome": run.get("research_outcome"),
            "review_required": final.get("review_required", True),
            "hypothesis": (design.get("plan") or {}).get("hypothesis"),
            "scientific_status": "NOT_ASSESSED" if experiment else "UNVERIFIED",
            "limitations": final.get("limitations") or [],
            "has_experiment_receipt": bool(experiment),
            "has_artifacts": bool(self.db.list_artifacts(run_id)),
        }
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO research_results VALUES(?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE "
                "SET result_json=excluded.result_json,updated_at=excluded.updated_at",
                (
                    run_id,
                    cid,
                    meta.get("engine", "legacy"),
                    json.dumps(value, ensure_ascii=False),
                    utc_now(),
                ),
            )

    def view(self, cid, *, offset=0, limit=20):
        offset = max(0, int(offset))
        limit = max(1, min(50, int(limit)))
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM research_results WHERE conversation_id=? "
                "ORDER BY updated_at DESC,run_id DESC LIMIT ? OFFSET ?",
                (cid, limit + 1, offset),
            ).fetchall()
            count = conn.execute(
                "SELECT COUNT(*) FROM research_results WHERE conversation_id=?", (cid,)
            ).fetchone()[0]
            revisions = conn.execute(
                "SELECT revision,source_run_id,created_at,value_json FROM research_state_revisions "
                "WHERE conversation_id=? ORDER BY revision DESC LIMIT 20",
                (cid,),
            ).fetchall()
        return {
            "state": self.state(cid),
            "items": [{"kind": r["kind"], **json.loads(r["result_json"])} for r in rows[:limit]],
            "total": count,
            "next_offset": offset + limit if len(rows) > limit else None,
            "revisions": [
                {
                    "revision": r["revision"],
                    "source_run_id": r["source_run_id"],
                    "created_at": r["created_at"],
                    "state": json.loads(r["value_json"]),
                }
                for r in revisions
            ],
        }
