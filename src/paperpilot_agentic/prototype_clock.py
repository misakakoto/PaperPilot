"""Active wall time for new prototypes; the caller holds the episode lock.

Clean returns stop the clock while approval/environment input is awaited. A
persisted active timestamp after a crash is conservatively charged on recovery;
we cannot prove when execution stopped. Legacy budgets are never reset/migrated.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import time

from .research_task import elapsed_seconds


def ensure_time_budget(db, run_id):
    from .llm_runtime import LLMBudgetExhausted
    from .models import Budget
    run = db.get_run(run_id)
    elapsed = elapsed_seconds(run) if run['metadata'].get('prototype_clock_policy') == 'active-v1' else (
        datetime.now(timezone.utc) - datetime.fromisoformat(run['created_at'].replace('Z', '+00:00'))).total_seconds()
    if elapsed >= Budget.from_dict(run['budget']).max_seconds:
        raise LLMBudgetExhausted('Time budget exhausted before prototype execution')


@contextmanager
def active_time(db, run_id):
    run = db.get_run(run_id)
    if run['metadata'].get('prototype_clock_policy') != 'active-v1':
        yield
        return
    db.update_run_metadata(run_id, {'research_active_seconds': elapsed_seconds(run),
                                    'research_active_since': time.time()})
    try:
        yield
    finally:
        current = db.get_run(run_id)
        db.update_run_metadata(run_id, {'research_active_seconds': elapsed_seconds(current),
                                        'research_active_since': None})
