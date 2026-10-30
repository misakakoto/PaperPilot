"""Physical dispatch receipts and conservative holds for unreported usage."""
from __future__ import annotations

import hashlib
import json

from .db import new_id, utc_now


class UnknownRequestResult(RuntimeError):
    request_dispatched = False


class AttemptLedger:
    def __init__(self, db):
        self.db = db

    def snapshot(self, run_id):
        with self.db.connect() as conn:
            rows = conn.execute('SELECT * FROM llm_attempts WHERE run_id=? ORDER BY ordinal',
                                (run_id,)).fetchall()
        attempts = [dict(row) for row in rows]
        return {'attempts': attempts,
                'reserved_unknown_tokens': sum(r['reserved_tokens'] for r in attempts if not r['usage_known']),
                'unknown_results': sum(r['state'] in {'STARTED', 'UNKNOWN'} for r in attempts),
                'unknown_usage_attempts': sum(not r['usage_known'] for r in attempts)}

    def begin(self, run_id, *, stage, model, messages, output_tokens, required_tokens,
              policy_sha256, max_calls, max_tokens, protected_calls=0, protected_tokens=0):
        serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode('utf-8')
        attempt_id = new_id('dispatch')
        with self.db.transaction() as conn:
            run = conn.execute('SELECT used_tokens,used_llm_calls FROM runs WHERE run_id=?', (run_id,)).fetchone()
            if run is None:
                raise KeyError(run_id)
            rows = conn.execute('SELECT state,reserved_tokens,usage_known FROM llm_attempts WHERE run_id=?',
                                (run_id,)).fetchall()
            if any(r['state'] in {'STARTED', 'UNKNOWN'} for r in rows):
                raise UnknownRequestResult('A prior dispatch is active or unknown; reconcile before sending')
            held = sum(r['reserved_tokens'] for r in rows if not r['usage_known'])
            if ((max_calls is not None and run['used_llm_calls'] + protected_calls >= max_calls) or
                    (max_tokens is not None and
                     run['used_tokens'] + held + required_tokens + protected_tokens > max_tokens)):
                from .llm_runtime import LLMBudgetExhausted
                blocked = LLMBudgetExhausted('Actual dispatch cannot fit the remaining cumulative allowance')
                blocked.request_dispatched = False
                raise blocked
            ordinal = run['used_llm_calls'] + 1
            conn.execute('UPDATE runs SET used_llm_calls=?,updated_at=? WHERE run_id=?',
                         (ordinal, utc_now(), run_id))
            conn.execute('INSERT INTO llm_attempts(attempt_id,run_id,ordinal,stage,model,state,'
                         'reserved_tokens,usage_known,request_bytes,request_chars,request_sha256,'
                         'max_output_tokens,policy_sha256,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                         (attempt_id, run_id, ordinal, stage, model, 'STARTED', required_tokens, 0,
                          len(serialized), len(serialized.decode('utf-8')), hashlib.sha256(serialized).hexdigest(),
                          output_tokens, policy_sha256, utc_now()))
        return attempt_id

    def settle(self, attempt_id, *, state, usage_known, cached_tokens_known=False,
               llm_call_id=None, failure=None):
        with self.db.transaction() as conn:
            conn.execute('UPDATE llm_attempts SET state=?,usage_known=?,cached_tokens_known=?,llm_call_id=?,'
                         'failure_json=?,completed_at=? WHERE attempt_id=? AND state=?',
                         (state, int(usage_known), int(cached_tokens_known), llm_call_id,
                          json.dumps(failure or {}), utc_now(), attempt_id, 'STARTED'))

    def record_failure(self, attempt_id, *, model, error, stage, output_tokens, thinking_enabled, duration_ms=0):
        from .provider_diagnostics import safe_error_text, safe_provider_failure
        failure = safe_provider_failure(error)
        with self.db.connect() as conn:
            row = conn.execute('SELECT run_id FROM llm_attempts WHERE attempt_id=?', (attempt_id,)).fetchone()
        if row is None:
            raise KeyError(attempt_id)
        call_id = self.db.record_llm_call(run_id=row['run_id'], stage=stage, attempt=1, status='FAILED',
                         model=model, thinking_enabled=thinking_enabled, max_output_tokens=output_tokens,
                         duration_ms=duration_ms, provider_attempts=1, error=safe_error_text(error))
        from openai import APIStatusError
        state = 'FAILED' if isinstance(error, APIStatusError) else 'UNKNOWN'
        self.settle(attempt_id, state=state, usage_known=False, llm_call_id=call_id, failure=failure)
        return failure
