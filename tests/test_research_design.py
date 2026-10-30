import json
from pathlib import Path
import time

import pytest

from paperpilot_agentic.durable_reader import persist_reader_evidence
from paperpilot_agentic.models import Budget, RunStatus
from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.research_design import enqueue_design, register_code_reference
from test_reader_pipeline import factory


class Designer:
    model = 'scripted-designer'
    def __init__(self):
        self.calls = 0
        self.messages = []
        self.mutate = lambda value: value

    def complete(self, messages, before_attempt=None, **kwargs):
        if before_attempt:
            before_attempt()
        self.calls += 1
        self.messages.append(messages)
        task = json.loads(messages[-1]['content'])
        if 'hypothesis' in task['output_schema']:
            value = {'literature_basis': [{'evidence_id': task['papers'][0]['evidence_id'],
                'quote': 'The encoder is shared.', 'interpretation': 'Weight sharing is reported.'}],
                'code_basis': [{'reference_id': task['references'][0]['evidence_id'],
                                'mechanism': 'Shared encoding'}],
                'hypothesis': 'A smaller decoder may preserve validation F1.',
                'experiment': {'baseline': 'shared encoder', 'candidate': 'smaller decoder',
                    'controls': ['split', 'seed'], 'metrics': ['F1', 'IoU'],
                    'acceptance_rule': 'Compare validation F1 under fixed settings',
                    'failure_analysis': 'Inspect false positive and false negative counts'},
                'risks': ['Not established by literature; requires testing']}
        else:
            value = {'files': [{'path': 'model.py', 'content': 'class ChangeModel:\n    pass\n'},
                               {'path': 'train.py', 'content': "raise RuntimeError('MUST_NOT_EXECUTE')\n"}],
                     'review_notes': ['Fixture skeleton; not executable training']}
        return LLMResponse(text=json.dumps(self.mutate(value)), finish_reason='stop')


def prepared(tmp_path):
    provider = Designer()
    runtime = factory(tmp_path, provider)
    parent = runtime.enqueue(question='Read paper; do not run experiments',
                             metadata={'engine': 'answer_first_text'})['run']['run_id']
    sources = persist_reader_evidence(runtime.db, parent, [{'evidence_id': 'source',
        'content': 'The encoder is shared.', 'source_uri': 'synthetic://paper'}])
    runtime.db.update_run_metadata(parent, {'final_answer': {'answer': 'POISON_OLD_ANSWER'}})
    runtime.db.update_run_status(parent, RunStatus.COMPLETED)
    reference = register_code_reference(runtime, parent, source_url=URL+'model.py',
        content='class Reference: pass', license_url=URL+'LICENSE', license_text='MIT fixture')
    return runtime, provider, parent, [sources[0]['evidence_id']], [reference]


URL = 'https://raw.githubusercontent.com/fixture/repo/'+'a'*40+'/'


@pytest.mark.parametrize('quote,accepted', [
    ('The encoder\nis\tshared.', True),
    ('The encoder ... shared.', False),
    ('The encoder is not shared.', False),
    ('Theencoder is shared.', False),
])
def test_quote_whitespace_reloads_exact_source_without_accepting_changes(tmp_path, quote, accepted):
    runtime, provider, parent, sources, refs = prepared(tmp_path)
    def mutate(value):
        if 'literature_basis' in value:
            value['literature_basis'][0]['quote'] = quote
        return value
    provider.mutate = mutate
    result = runtime.resume(enqueue(runtime, parent, sources, refs))
    assert result['run']['status'] == ('COMPLETED' if accepted else 'FAILED')
    if accepted:
        basis = result['run']['metadata']['research_design']['plan']['literature_basis'][0]
        assert basis['quote'] == 'The encoder is shared.'
        assert json.loads(result['run']['metadata']['research_design_plan_raw']['text'])['literature_basis'][0]['quote'] == quote
    assert not result['tool_calls']


def test_design_schema_repair_is_bounded_recorded_and_budgeted(tmp_path):
    runtime, provider, parent, sources, refs = prepared(tmp_path)
    def mutate(value):
        if provider.calls == 1:
            value['experiment']['acceptance_rule'] = ['wrong schema shape']
        return value
    provider.mutate = mutate
    result = runtime.resume(enqueue(runtime, parent, sources, refs))
    assert result['run']['status'] == 'COMPLETED'
    assert provider.calls == result['run']['used_llm_calls'] == 3
    assert result['run']['budget']['max_llm_calls'] == 4
    assert result['run']['budget']['max_tokens'] == 80000
    attempts = result['run']['metadata']['research_design_plan_raw_attempts']
    assert len(attempts) == 2
    assert attempts[0]['validation_error']
    assert 'validation_error' not in attempts[1]
    runtime.resume(result['run']['run_id'])
    assert provider.calls == 3


def enqueue(runtime, parent, sources, refs, **kwargs):
    return enqueue_design(runtime, parent, question='Design a bounded optical CD experiment',
        source_ids=sources, reference_ids=refs, **kwargs)['run']['run_id']


def test_design_reads_originals_emits_unexecuted_drafts_and_replays(tmp_path):
    runtime, provider, parent, sources, refs = prepared(tmp_path)
    rid = enqueue(runtime, parent, sources, refs)
    result = runtime.resume(rid)
    assert result['run']['status'] == 'COMPLETED', result['run'].get('error')
    draft = result['run']['metadata']['research_design']
    assert draft['status'] == 'DRAFT_REQUIRES_REVIEW' and draft['executed'] is False
    assert draft['constraints']['training_environment'] == 'npy3.11'
    assert draft['constraints']['execution_authorized'] is False
    assert len(draft['artifacts']) == 2
    assert len(runtime.db.list_artifacts(rid)) == 2
    assert all(Path(a['path']).is_file() for a in draft['artifacts'])
    assert result['tool_calls'] == []  # No experiment tool or generated code execution.
    assert provider.calls == 2 and result['run']['used_llm_calls'] == 2
    assert 'POISON_OLD_ANSWER' not in json.dumps(provider.messages)
    runtime.db.update_run_status(rid, RunStatus.RUNNING)
    replay = runtime.resume(rid)
    assert replay['run']['status'] == 'COMPLETED', replay['run'].get('error')
    assert provider.calls == 2
    assert len(runtime.db.list_artifacts(rid)) == 2
    child = runtime.enqueue_follow_up(parent_run_id=rid, question='Explain the idea')['run']
    assert child['metadata']['engine'] == 'answer_first_text'


@pytest.mark.parametrize('kind', ['quote', 'reference', 'path', 'syntax', 'dependency'])
def test_invalid_grounding_or_code_is_rejected_without_execution(tmp_path, kind):
    runtime, provider, parent, sources, refs = prepared(tmp_path)
    def mutate(value):
        if kind == 'quote' and 'literature_basis' in value:
            value['literature_basis'][0]['quote'] = 'Invented evidence'
        if kind == 'reference' and 'code_basis' in value:
            value['code_basis'][0]['reference_id'] = 'foreign'
        if 'files' in value:
            if kind == 'path':
                value['files'][0]['path'] = '../escape.py'
            if kind == 'syntax':
                value['files'][0]['content'] = 'def incomplete('
            if kind == 'dependency':
                value['files'][1]['content'] = 'from dataset import SYSUDataset\n'
        return value
    provider.mutate = mutate
    result = runtime.resume(enqueue(runtime, parent, sources, refs))
    assert result['run']['status'] == 'FAILED'
    assert not (tmp_path/'research-drafts').exists()
    assert result['tool_calls'] == []


def test_budget_cannot_be_reset_between_design_and_code(tmp_path):
    runtime, provider, parent, sources, refs = prepared(tmp_path)
    rid = enqueue(runtime, parent, sources, refs, budget=Budget(max_llm_calls=1,
        synthesis_reserve_calls=0, repair_reserve_calls=0))
    result = runtime.resume(rid)
    assert result['run']['status'] == 'BUDGET_EXHAUSTED'
    assert provider.calls == 1
    assert not (tmp_path/'research-drafts').exists()


def test_source_changes_and_model_changes_fail_before_spend(tmp_path):
    runtime, provider, parent, sources, refs = prepared(tmp_path)
    rid = enqueue(runtime, parent, sources, refs)
    runtime.text_reader_protocol = 'different-model'
    assert runtime.resume(rid)['run']['status'] == 'FAILED'
    assert provider.calls == 0
    with pytest.raises(ValueError, match='foreign'):
        enqueue(runtime, parent, ['foreign'], refs)
    with pytest.raises(ValueError, match='immutable'):
        register_code_reference(runtime, parent, source_url=URL.replace('a'*40, 'main')+'m.py',
            content='x', license_url=URL+'LICENSE', license_text='MIT')


def test_started_checkpoint_and_cancel_do_not_rebill(tmp_path):
    runtime, provider, parent, sources, refs = prepared(tmp_path)
    rid = enqueue(runtime, parent, sources, refs)
    class Crash(BaseException):
        pass
    complete = provider.complete
    provider.complete = lambda *a, **k: (_ for _ in ()).throw(Crash())
    with pytest.raises(Crash):
        runtime.resume(rid)
    provider.complete = complete
    assert runtime.resume(rid)['run']['status'] == 'RUNNING'
    assert provider.calls == 0
    runtime.cancel(rid)
    assert runtime.resume(rid)['run']['status'] == 'CANCELLED'


def test_original_mutation_and_draft_tampering_are_not_silently_accepted(tmp_path):
    runtime, provider, parent, sources, refs = prepared(tmp_path)
    rid = enqueue(runtime, parent, sources, refs)
    with runtime.db.transaction() as conn:
        conn.execute('UPDATE evidence SET content=? WHERE evidence_id=?', ('Changed original', sources[0]))
    assert runtime.resume(rid)['run']['status'] == 'FAILED'
    assert provider.calls == 0
    with runtime.db.transaction() as conn:
        conn.execute('UPDATE evidence SET content=? WHERE evidence_id=?', ('The encoder is shared.', sources[0]))
    rid = enqueue(runtime, parent, sources, refs)
    result = runtime.resume(rid)
    artifact = Path(result['run']['metadata']['research_design']['artifacts'][0]['path'])
    artifact.write_text('user edited draft', encoding='utf-8')
    runtime.db.update_run_status(rid, RunStatus.RUNNING)
    assert runtime.resume(rid)['run']['status'] == 'FAILED'
    assert artifact.read_text() == 'user edited draft'
    assert provider.calls == 2


def test_reference_identity_is_owned_by_each_reading_run(tmp_path):
    runtime, _, parent, _, refs = prepared(tmp_path)
    other = runtime.enqueue(question='Other reading')['run']['run_id']
    reference = register_code_reference(runtime, other, source_url=URL+'model.py',
        content='class Reference: pass', license_url=URL+'LICENSE', license_text='MIT fixture')
    assert reference != refs[0]
    assert runtime.db.list_evidence(other)[0]['run_id'] == other


def test_web_design_endpoint_returns_draft_and_rejects_client_execution_fields(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from paperpilot_agentic.web_api import create_fastapi_router
    runtime, provider, parent, sources, refs = prepared(tmp_path)
    app = FastAPI()
    app.include_router(create_fastapi_router(runtime_factory=lambda: runtime))
    payload = {'question': 'Design experiment only', 'source_ids': sources, 'reference_ids': refs}
    with TestClient(app) as client:
        rejected = client.post(f'/agentic/runs/{parent}/research-design',
                               json={**payload, 'command': ['python', 'anything.py']})
        assert rejected.status_code == 422 and provider.calls == 0
        response = client.post(f'/agentic/runs/{parent}/research-design', json=payload)
        assert response.status_code == 202, response.text
        rid = response.json()['run']['run_id']
        for _ in range(100):
            result = client.get(f'/agentic/runs/{rid}').json()
            if result['run']['status'] in {'COMPLETED', 'FAILED'}:
                break
            time.sleep(.02)
        assert result['run']['status'] == 'COMPLETED', result['run'].get('error')
        assert result['run']['metadata']['research_design']['executed'] is False
