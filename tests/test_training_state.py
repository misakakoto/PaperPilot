import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from paperpilot_agentic.training_state import GpuBudget, TrainingBusy, process_identity


def test_budget_resume_never_resets_and_rejects_new_cap(tmp_path):
    ledger = GpuBudget(tmp_path/'budget.sqlite3', 80)
    token, seconds = ledger.reserve('preflight', 20)
    assert seconds == 20
    ledger.finish(token, 'COMPLETED', 12)
    ledger = GpuBudget(ledger.path, 80)
    assert ledger.snapshot()['remaining_seconds'] == 68
    with pytest.raises(ValueError, match='reset'):
        GpuBudget(ledger.path, 90)


def test_reservations_are_serialized_across_instances(tmp_path):
    path = tmp_path/'budget.sqlite3'
    GpuBudget(path, 80)
    def reserve(_):
        try:
            return GpuBudget(path, 80).reserve('formal', 30)[0]
        except TrainingBusy:
            return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(x is not None for x in pool.map(reserve, range(4))) == 1


def test_dead_process_charged_full_reservation_and_pid_reuse(tmp_path):
    ledger = GpuBudget(tmp_path/'budget.sqlite3', 80)
    token, _ = ledger.reserve('recovery', 30)
    with ledger.connection() as db:
        db.execute("UPDATE launches SET status='RUNNING',pid=123,identity='old' WHERE token=?", (token,))
    ledger.recover(lambda _: ('RUNNING', 'new'))
    assert ledger.snapshot()['charged_seconds'] == 30
    assert ledger.snapshot()['remaining_seconds'] == 50
    ledger.recover(lambda _: ('DEAD', None))
    assert ledger.snapshot()['charged_seconds'] == 30


@pytest.mark.parametrize('status,identity', [('RUNNING', 'same'), ('UNKNOWN', None)])
def test_live_or_unknown_process_does_not_start_another(tmp_path, status, identity):
    ledger = GpuBudget(tmp_path/'budget.sqlite3', 80)
    token, _ = ledger.reserve('formal', 30)
    with ledger.connection() as db:
        db.execute("UPDATE launches SET status='RUNNING',pid=123,identity='same' WHERE token=?", (token,))
    with pytest.raises(TrainingBusy):
        ledger.recover(lambda _: (status, identity))
    with pytest.raises(TrainingBusy):
        ledger.reserve('duplicate')
    assert ledger.snapshot()['reserved_seconds'] == 30


def test_unattached_launch_stays_unknown_even_when_resolver_says_dead(tmp_path):
    ledger = GpuBudget(tmp_path/'budget.sqlite3', 80)
    ledger.reserve('formal', 30)
    with pytest.raises(TrainingBusy, match='UNKNOWN'):
        ledger.recover(lambda _: ('DEAD', None))


def test_budget_exhaustion_and_double_completion(tmp_path):
    ledger = GpuBudget(tmp_path/'budget.sqlite3', 80)
    token, seconds = ledger.reserve('formal', 100)
    assert seconds == 80
    ledger.finish(token, 'PAUSED', 80)
    with pytest.raises(ValueError):
        ledger.finish(token, 'COMPLETED', 0)
    with pytest.raises(TimeoutError):
        ledger.reserve('next')


def test_process_identity_and_repeat_submission(tmp_path):
    status, identity = process_identity(os.getpid())
    assert status == 'RUNNING' and identity
    ledger = GpuBudget(tmp_path/'budget.sqlite3')
    assert ledger.claim_run('frozen', 'first') == 'first'
    assert ledger.claim_run('frozen', 'second') == 'first'
