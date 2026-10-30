"""Normal Reader follow-up entry exercises pressure compaction and publication."""
import json

from paperpilot_agentic.models import RunStatus
from paperpilot_agentic.providers import LLMResponse
from test_reader_pipeline import factory
from test_durable_reader import AnswerProvider


class MemoryProvider(AnswerProvider):
    def __init__(self, reject=False):
        super().__init__()
        self.reject = reject
        self.memory_inputs, self.answer_inputs = [], []

    def complete(self, messages, *, before_attempt, **kwargs):
        payload = json.loads(messages[-1]['content'])
        if 'state' in payload and 'sources' in payload:
            before_attempt()
            self.calls += 1
            self.memory_inputs.append(payload)
            frames = []
            exists = bool(payload['state']['tasks'])
            for source in payload['sources']:
                ops = []
                if not exists:
                    ops = [{'kind': 'create_task', 'task_id': 'A', 'label': 'Source analysis', 'quote': 'What fact'},
                           {'kind': 'activate_task', 'task_id': 'A', 'quote': 'What fact'},
                           {'kind': 'set_condition', 'task_id': 'A', 'key': 'format', 'text': 'Answer briefly',
                            'expect': None, 'quote': 'Answer briefly'}]
                    exists = True
                frames.append({'source_id': source['id'], 'operations': ops})
            value = {'base_revision': payload['state']['revision'], 'frames': frames}
        elif 'before' in payload and 'proposal' in payload:
            before_attempt()
            self.calls += 1
            value = {'verdict': 'REJECT' if self.reject else 'ACCEPT',
                     'issues': ['Missing requirement'] if self.reject else [], 'missing_requirements': []}
        elif 'current_question' in payload:
            before_attempt()
            self.calls += 1
            value = {'question': 'What fact?', 'use_prior_sources': True,
                     'needs_clarification': False, 'reason': 'Same subject'}
            assert payload['history'].get('task_state') is not None or not self.memory_inputs
        else:
            self.answer_inputs.append(payload)
            return super().complete(messages, before_attempt=before_attempt, **kwargs)
        return LLMResponse(text=json.dumps(value), prompt_tokens=100, completion_tokens=50, finish_reason='stop')


def parent(runtime, large=True):
    question = 'What fact? Answer briefly. ' + ('Background note. ' * 800 if large else '')
    rid = runtime.enqueue(question=question, metadata={'engine': 'answer_first_text'})['run']['run_id']
    runtime.db.update_run_status(rid, RunStatus.COMPLETED)
    return rid


def test_normal_follow_up_automatically_compacts_and_writer_receives_current_state(tmp_path):
    provider = MemoryProvider()
    runtime = factory(tmp_path, provider)
    root = parent(runtime)
    rid = runtime.enqueue_follow_up(parent_run_id=root, question='Continue.')['run']['run_id']
    result = runtime.resume(rid)
    assert result['run']['status'] == 'COMPLETED', result['run']['metadata']
    assert result['run']['metadata']['request_memory_status'] == 'READY'
    assert result['final_answer']['answer'].startswith('A source fact.')
    assert provider.calls == 5
    assert len(provider.memory_inputs[0]['sources']) == 2
    history = provider.answer_inputs[0]['request_history']
    assert history[0]['role'] == 'derived_request_state'
    assert 'Background note' not in history[0]['user_request']
    assert 'Answer briefly' in history[0]['user_request']
    runtime.resume(rid)
    assert provider.calls == 5
    child = runtime.enqueue_follow_up(parent_run_id=rid, question='Continue.')['run']['run_id']
    continued = runtime.resume(child)
    assert continued['run']['metadata']['request_state_root'] == root
    assert len(provider.memory_inputs[-1]['sources']) == 1
    assert provider.memory_inputs[-1]['sources'][0]['id'] == child


def test_short_follow_up_keeps_original_history_without_compression_calls(tmp_path):
    provider = MemoryProvider()
    runtime = factory(tmp_path, provider)
    root = parent(runtime, large=False)
    rid = runtime.enqueue_follow_up(parent_run_id=root, question='Continue.')['run']['run_id']
    result = runtime.resume(rid)
    assert result['run']['status'] == 'COMPLETED'
    assert provider.calls == 3 and provider.memory_inputs == []
    assert 'request_state_root' not in result['run']['metadata']


def test_rejected_compression_does_not_generate_from_incomplete_state(tmp_path):
    provider = MemoryProvider(reject=True)
    runtime = factory(tmp_path, provider)
    root = parent(runtime)
    rid = runtime.enqueue_follow_up(parent_run_id=root, question='Continue.')['run']['run_id']
    result = runtime.resume(rid)
    assert result['run']['metadata']['request_memory_status'] == 'REJECTED_OR_UNCERTAIN'
    assert provider.calls == 2 and not provider.answer_inputs
    assert result['final_answer']['answer'] == ''
