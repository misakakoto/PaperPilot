"""No network or model calls: exercise immediate and delayed completion races."""
from concurrent.futures import Future
from threading import Event

import pytest

from paperpilot_agentic.web_api import _BackgroundRunPool


class ImmediateExecutor:
    def submit(self, operation):
        future = Future()
        try:
            future.set_result(operation())
        except Exception as exc:
            future.set_exception(exc)
        return future


@pytest.mark.parametrize('fails', [False, True])
def test_already_completed_future_registers_callback_without_holding_lock(fails):
    pool = _BackgroundRunPool(workers=1)
    pool._executor.shutdown()
    pool._executor = ImmediateExecutor()
    original = pool._forget

    def checked(run_id, completed):
        # Fail immediately instead of letting the old implementation hang tests.
        assert pool._lock.acquire(blocking=False), 'completion callback ran under submit lock'
        pool._lock.release()
        original(run_id, completed)

    pool._forget = checked

    def operation():
        if fails:
            raise RuntimeError('instant failure')
        return 'done'

    assert pool.submit('run', operation)
    assert not pool._futures


def test_old_completion_does_not_remove_new_invocation_guard():
    pool = _BackgroundRunPool(workers=1)
    try:
        old, current = Future(), Future()
        old.set_result(None)
        pool._futures['run'] = current
        pool._forget('run', old)
        assert pool._futures['run'] is current
        assert not pool.submit('run', lambda: None)
        current.set_result(None)
        pool._forget('run', current)
        assert not pool._futures
    finally:
        pool._executor.shutdown()


def test_live_worker_is_deduplicated_and_completion_allows_next_submission():
    pool = _BackgroundRunPool(workers=1)
    started, release, cleaned = Event(), Event(), Event()
    calls = []
    original = pool._forget

    def forget(run_id, completed):
        original(run_id, completed)
        cleaned.set()

    pool._forget = forget

    def operation():
        calls.append('called')
        started.set()
        assert release.wait(timeout=5)

    try:
        assert pool.submit('run', operation)
        assert started.wait(timeout=5)
        assert not pool.submit('run', operation)
        release.set()
        assert cleaned.wait(timeout=5)
        assert calls == ['called']
        assert pool.submit('run', lambda: None)
    finally:
        release.set()
        pool._executor.shutdown()
