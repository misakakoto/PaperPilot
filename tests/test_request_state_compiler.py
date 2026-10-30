"""Protocol/runtime regressions; scripted output is not a semantic quality eval."""
from copy import deepcopy
import json

import pytest

from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.durable_reader import DurableResearchModel
from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
from paperpilot_agentic.models import Budget
from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.request_state import empty_request_state, request_compiler_view
from paperpilot_agentic.request_state_compiler import compile_request_batch


SOURCES = [{'id': 'r1', 'text': 'Study building change. Do not train.'}]
PROPOSAL = {'base_revision': 'initial', 'frames': [{'source_id': 'r1', 'operations': [
    {'kind': 'create_task', 'task_id': 'buildings', 'label': 'Building change', 'quote': 'Study building change.'},
    {'kind': 'activate_task', 'task_id': 'buildings', 'quote': 'Study building change.'},
    {'kind': 'set_condition', 'task_id': 'buildings', 'key': 'no_training',
     'text': 'Do not train.', 'applies_to': None, 'expect': None, 'quote': 'Do not train.'},
]}]}
ACCEPT = {'verdict': 'ACCEPT', 'issues': [], 'missing_requirements': []}


class Provider:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    def complete(self, messages, *, before_attempt, **kwargs):
        before_attempt()
        self.calls += 1
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return LLMResponse(text=output if isinstance(output, str) else json.dumps(output),
                           prompt_tokens=100, completion_tokens=50, finish_reason='stop')


def setup(tmp_path, outputs, *, max_calls=10):
    db = RuntimeDatabase(tmp_path/'compiler.sqlite3')
    rid = db.create_run('Compile requests', Budget(max_tokens=100000, max_llm_calls=max_calls))
    provider = Provider(outputs)
    model = DurableResearchModel(PersistentLLMRuntime(db, provider), rid, stage='request_memory')
    return db, rid, provider, model


def test_compiler_uses_shared_persistent_budget_and_does_not_commit(tmp_path):
    db, rid, provider, model = setup(tmp_path, [PROPOSAL, ACCEPT])
    state = empty_request_state()
    sources = deepcopy(SOURCES)
    reports = []
    report = compile_request_batch(state, sources, model, on_progress=reports.append)
    assert report['status'] == 'MODEL_ACCEPTED_UNVERIFIED'
    assert state == empty_request_state() and sources == SOURCES
    assert provider.calls == 2
    saved = RuntimeDatabase(db.path).get_run(rid)
    assert saved['used_llm_calls'] == 2 and saved['used_tokens'] == 300
    assert len(db.list_llm_calls(rid)) == 2
    assert len(reports) == 3 and reports[-1] == report
    assert all(c['response']['text'] for c in report['calls'])
    assert saved['metadata'] == {}


@pytest.mark.parametrize('review', [
    {'verdict': 'ACCEPT'},
    {'verdict': 'ACCEPT', 'issues': ['Wrong scope'], 'missing_requirements': []},
    {'verdict': 'UNCERTAIN', 'issues': [], 'missing_requirements': []},
    {'verdict': 'ACCEPT', 'issues': [], 'missing_requirements': ['Input modality']},
    {'verdict': 'ACCEPT', 'issues': [42], 'missing_requirements': []},
])
def test_audit_cannot_publish_candidate_on_missing_fields_or_objections(tmp_path, review):
    _, _, _, model = setup(tmp_path, [PROPOSAL, review])
    result = compile_request_batch(empty_request_state(), SOURCES, model)
    assert result['status'] != 'MODEL_ACCEPTED_UNVERIFIED'
    assert 'candidate_state' not in result and 'work_view' not in result


def test_input_overflow_spends_nothing_and_preserves_sources(tmp_path):
    db, rid, provider, model = setup(tmp_path, [])
    result = compile_request_batch(empty_request_state(), SOURCES, model, max_input_bytes=100)
    assert result['status'] == 'NEEDS_SMALLER_BATCH' and result['calls'] == []
    assert provider.calls == db.get_run(rid)['used_llm_calls'] == 0


def test_work_view_overflow_does_not_spend_audit(tmp_path):
    _, _, provider, model = setup(tmp_path, [PROPOSAL])
    result = compile_request_batch(empty_request_state(), SOURCES, model, max_work_bytes=1)
    assert result['status'] == 'NEEDS_SMALLER_WORK_VIEW' and provider.calls == 1
    assert 'candidate_state' not in result


def test_missing_active_task_does_not_spend_audit(tmp_path):
    proposal = deepcopy(PROPOSAL)
    del proposal['frames'][0]['operations'][1]
    _, _, provider, model = setup(tmp_path, [proposal])
    result = compile_request_batch(empty_request_state(), SOURCES, model)
    assert result['status'] == 'NEEDS_TASK_SELECTION' and provider.calls == 1


@pytest.mark.parametrize('output', ['broken json', {'base_revision': 'stale', 'frames': []}, OSError('offline')])
def test_failures_count_and_never_mutate_state(tmp_path, output):
    db, rid, provider, model = setup(tmp_path, [output])
    state = empty_request_state()
    result = compile_request_batch(state, SOURCES, model)
    assert result['status'] == 'FAILED' and 'error_type' in result
    assert provider.calls == db.get_run(rid)['used_llm_calls'] == 1
    assert state == empty_request_state() and 'candidate_state' not in result


def test_shared_budget_preserves_answer_reserve_and_blocks_audit(tmp_path):
    db, rid, provider, model = setup(tmp_path, [PROPOSAL], max_calls=3)
    result = compile_request_batch(empty_request_state(), SOURCES, model)
    assert result['error_type'] == 'LLMBudgetExhausted'
    assert result['status'] == 'FAILED' and 'candidate_state' not in result
    assert provider.calls == db.get_run(rid)['used_llm_calls'] == 1


def test_duplicate_sources_rejected_before_spending(tmp_path):
    _, _, provider, model = setup(tmp_path, [])
    with pytest.raises(ValueError, match='Duplicate'):
        compile_request_batch(empty_request_state(), SOURCES * 2, model)
    assert provider.calls == 0


def test_broken_progress_sink_does_not_start_provider(tmp_path):
    _, _, provider, model = setup(tmp_path, [])
    def fail(report):
        raise OSError('cannot persist')
    with pytest.raises(OSError, match='cannot persist'):
        compile_request_batch(empty_request_state(), SOURCES, model, on_progress=fail)
    assert provider.calls == 0


def test_audit_payload_overflow_returns_proposal_but_no_candidate(tmp_path):
    _, _, provider, model = setup(tmp_path, [PROPOSAL])
    from paperpilot_agentic.request_state_compiler import PROPOSAL_PROMPT
    state = empty_request_state()
    messages = [{'role': 'system', 'content': PROPOSAL_PROMPT},
                {'role': 'user', 'content': json.dumps({'state': request_compiler_view(state), 'sources': SOURCES}, ensure_ascii=False)}]
    cap = len(json.dumps(messages, ensure_ascii=False).encode('utf-8'))
    # Inflate valid stored label content so the audit, not the proposal request,
    # crosses the same cap. Structural validity is not useful compression.
    expanded = deepcopy(PROPOSAL)
    expanded['frames'][0]['operations'][0]['label'] = 'Building change ' * 500
    provider.outputs = iter([expanded])
    result = compile_request_batch(state, SOURCES, model, max_input_bytes=cap, max_work_bytes=20000)
    assert result['status'] == 'NEEDS_SMALLER_BATCH'
    assert provider.calls == 1 and 'proposal' in result
    assert 'candidate_state' not in result


def test_cancelled_run_never_starts_compression(tmp_path):
    from paperpilot_agentic.models import RunStatus
    db, rid, provider, model = setup(tmp_path, [])
    db.update_run_status(rid, RunStatus.CANCELLED)
    result = compile_request_batch(empty_request_state(), SOURCES, model)
    assert result['error_type'] == 'InterruptedError'
    assert provider.calls == db.get_run(rid)['used_llm_calls'] == 0
