import copy
import json

import pytest

from paperpilot_agentic.prototype_workspace import complete, plan, put
from test_prototype_completion import finished


def inspect(harness, state, artifact_id, **kwargs):
    return harness._apply('owned-run', state, {'action': 'inspect_artifact', 'reason': 'read original',
                         'arguments': {'artifact_id': artifact_id, **kwargs}}, 7)


@pytest.mark.parametrize('mode', ['adaptive', 'simple_loop'])
def test_model_can_page_exact_historical_artifact_without_certifying_it(mode):
    harness, state, rows = finished()
    state['mode'] = mode
    old = state['work']['tasks']['code']['output']
    rows[2]['goal'] = 'revised implementation'
    plan(state['work'], rows)
    complete(state['work'], 'code', 'implement', {'source': '新版本' * 5000, 'config': {}})
    current = state['work']['tasks']['code']['output']
    put(state['work'], 'unused_history', 'read', {'text': 'UNUSED_HISTORY_TEXT' * 5000})
    before = copy.deepcopy(state)
    _, messages = harness._context('owned-run', state)
    payload = json.loads(messages[-1]['content'])
    assert 'inspect_artifact' in payload['available_actions']
    assert payload['artifact_index'][old]['current'] is False
    assert payload['artifact_index'][current]['current'] is True
    assert 'UNUSED_HISTORY_TEXT' not in messages[-1]['content']
    offset, pages = 0, []
    while True:
        page = inspect(harness, state, old, offset=offset, limit=17)
        assert page['current'] is False
        assert len(page['text']) <= 17
        pages.append(page['text'])
        if page['next_offset'] is None:
            break
        assert page['next_offset'] > offset
        offset = page['next_offset']
    assert json.loads(''.join(pages)) == state['work']['artifacts'][old]
    assert state == before


@pytest.mark.parametrize('args', [
    {'artifact_id': 'foreign-run-artifact'}, {'artifact_id': '../state.json'},
    {'offset': -1}, {'offset': True}, {'limit': 0}, {'limit': 4001},
    {'limit': False}, {'file': 'model.py'}, {'offset': 10**9},
])
def test_memory_rejects_foreign_ids_and_unbounded_or_file_reads(args):
    harness, state, _ = finished()
    identity = state['work']['tasks']['read']['output']
    value = {'artifact_id': identity, **args}
    before = copy.deepcopy(state)
    with pytest.raises(ValueError):
        inspect(harness, state, **value)
    assert state == before


def test_memory_detects_tampering_and_does_not_unlock_replanning():
    harness, state, _ = finished()
    identity = state['work']['tasks']['read']['output']
    state['plan_requires_work'] = True
    assert inspect(harness, state, identity)['current'] is True
    assert 'plan' not in harness.allowed_actions(state)
    state['work']['artifacts'][identity]['content']['content'] = 'changed evidence'
    with pytest.raises(ValueError, match='identity'):
        inspect(harness, state, identity)


def test_unicode_pages_reconstruct_original_without_splitting_surrogates():
    harness, state, _ = finished()
    identity = put(state['work'], 'historical_notes', 'read', {'text': '原始证据🧪' * 2000})
    text, offset = '', 0
    while True:
        page = inspect(harness, state, identity, offset=offset)
        assert len(page['text']) <= 4000
        page['text'].encode('utf-8')
        text += page['text']
        if page['next_offset'] is None:
            break
        offset = page['next_offset']
    assert json.loads(text) == state['work']['artifacts'][identity]


def test_inspection_receipt_survives_resume_and_is_seen_by_next_decision(tmp_path, monkeypatch):
    from paperpilot_agentic.prototype_harness import PrototypeHarness
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from paperpilot_agentic.providers import LLMResponse
    from test_prototype_workspace import Reader

    class Inspector(Reader):
        requested = False
        saw_original = False

        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]['content'])
            if payload['tasks'].get('read', {}).get('status') == 'DONE' and not self.requested:
                self.requested = True
                identity = payload['tasks']['read']['output']
                return LLMResponse(text=json.dumps({'action': 'inspect_artifact', 'reason': 'original evidence',
                    'arguments': {'artifact_id': identity}}), model=self.model, finish_reason='stop',
                    prompt_tokens=20, completion_tokens=20)
            feedback = payload.get('feedback') or {}
            if feedback.get('action') == 'inspect_artifact':
                assert json.loads(feedback['text'])['content']['content'] == 'original owned evidence'
                self.saw_original = True
            return super().complete(messages, **kwargs)

    reader = Inspector()
    runtime = ResearchAgentRuntime(workspace=tmp_path/'workspace', db_path=tmp_path/'runtime.db', llm=reader)
    monkeypatch.setattr(runtime.db, 'list_evidence', lambda _: [
        {'evidence_id': 'source', 'content': 'original owned evidence'}])
    harness = PrototypeHarness(runtime)
    rid = harness.start('build', 'memory-resume', 'parent', ['source'])['run']['run_id']
    harness.advance(rid, max_turns=3)
    state = harness._load(rid)
    assert state['last_observation']['action'] == 'inspect_artifact'
    assert state['work']['tasks']['design']['status'] == 'PENDING'
    assert runtime.db.get_run(rid)['used_llm_calls'] == 3
    before = copy.deepcopy(state['work']['artifacts'])
    PrototypeHarness(runtime).advance(rid, max_turns=1)
    assert reader.saw_original
    assert runtime.db.get_run(rid)['used_llm_calls'] == 4
    assert all(harness._load(rid)['work']['artifacts'][key] == artifact for key, artifact in before.items())
