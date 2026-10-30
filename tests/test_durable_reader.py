import pytest
import json

from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.durable_reader import (
    DurableResearchModel, reader_completion_state, persist_reader_evidence, publish_reader_answer,
    run_durable_reader, ReaderAlreadyStarted)
from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
from paperpilot_agentic.models import Budget, RunStatus, ResearchOutcome
from paperpilot_agentic.providers import LLMResponse


class Provider:
    calls = 0

    def complete(self, messages, *, before_attempt, **kwargs):
        before_attempt()
        self.calls += 1
        return LLMResponse(text='{}', prompt_tokens=10, completion_tokens=5, finish_reason='stop')


def test_withheld_model_notes_are_persisted_separately_without_promoting_truth(tmp_path):
    db = RuntimeDatabase(tmp_path/'notes.sqlite3')
    rid = db.create_run('Are all samples sterile?', Budget())
    note = 'The passages do not explicitly state that all samples are sterile; the treatment is guaranteed safe.'
    unsafe_markup = '<img src=x onerror=alert(1)>'
    report = {'status': 'REVIEW_REQUIRED', 'answer': '', 'answer_scope': 'NO_SUPPORTED_ANSWER',
              'limitations': ['Evidence is insufficient.'], 'direct_answer_supported': False,
              'unverified_limitations': [note, note, unsafe_markup]}
    result = publish_reader_answer(db, rid, report)
    assert result['answer'] == '' and result['citation_ids'] == []
    assert result['limitations'] == report['limitations']
    assert result['research_outcome'] == 'UNRESOLVED'
    assert result['review_required'] is True
    assert result['review_notes'] == [
        {'text': text, 'status': 'UNVERIFIED_MODEL_NOTE', 'origin': 'withheld_limitation'}
        for text in [note, unsafe_markup]]
    assert RuntimeDatabase(tmp_path/'notes.sqlite3').get_run(rid)['metadata']['final_answer'] == result
    assert report['unverified_limitations'] == [note, note, unsafe_markup]


def test_old_report_without_withheld_notes_has_no_review_notes(tmp_path):
    db = RuntimeDatabase(tmp_path/'old.sqlite3')
    rid = db.create_run('Question?', Budget())
    result = publish_reader_answer(db, rid, {'status': 'NO_EVIDENCE', 'answer': ''})
    assert result['review_notes'] == []


def test_bridge_uses_one_shared_budget_and_persists_usage(tmp_path):
    db = RuntimeDatabase(tmp_path/'runtime.sqlite3')
    run_id = db.create_run('question', Budget())
    provider = Provider()
    model = DurableResearchModel(PersistentLLMRuntime(db, provider), run_id, stage='answer_audit')
    guarded = []
    model.complete([{'role': 'user', 'content': 'question'}], max_tokens=100,
                   before_attempt=lambda: guarded.append(True))
    assert guarded == [True]
    assert provider.calls == 1
    assert db.get_run(run_id)['used_llm_calls'] == 1
    assert db.get_run(run_id)['used_tokens'] == 15


def test_repair_bridge_can_spend_protected_repair_tokens(tmp_path):
    db = RuntimeDatabase(tmp_path/'runtime.sqlite3')
    run_id = db.create_run('question', Budget(max_tokens=10000, repair_reserve_tokens=8000))
    provider = Provider()
    model = DurableResearchModel(PersistentLLMRuntime(db, provider), run_id,
                                 stage='answer_first', reserve_class='synthesis')
    from paperpilot_agentic.llm_runtime import LLMBudgetExhausted
    messages = [{'role': 'user', 'content': 'Evidence ' * 400}]
    with pytest.raises(LLMBudgetExhausted):
        model.complete(messages, max_tokens=100)
    model.for_repair().complete(messages, max_tokens=100)
    assert provider.calls == 1
    assert db.list_llm_calls(run_id)[0]['stage'] == 'json_repair'


def test_local_guard_stops_provider_before_global_reservation(tmp_path):
    db = RuntimeDatabase(tmp_path/'runtime.sqlite3')
    run_id = db.create_run('question', Budget())
    provider = Provider()
    model = DurableResearchModel(PersistentLLMRuntime(db, provider), run_id, stage='answer_audit')

    def stop():
        raise RuntimeError('local attempts exhausted')

    with pytest.raises(RuntimeError):
        model.complete([{'role': 'user', 'content': 'question'}], max_tokens=100, before_attempt=stop)
    assert provider.calls == 0
    assert db.get_run(run_id)['used_llm_calls'] == 0


def test_budget_exhaustion_and_no_answer_are_not_success():
    assert reader_completion_state({'status': 'FAILED', 'error_type': 'LLMBudgetExhausted'}) == (
        RunStatus.BUDGET_EXHAUSTED, ResearchOutcome.UNRESOLVED)
    assert reader_completion_state({'status': 'NO_EVIDENCE'})[1] == ResearchOutcome.UNRESOLVED
    assert reader_completion_state({'status': 'REVIEW_REQUIRED', 'answer': 'Model answer'})[1] == ResearchOutcome.PARTIAL


def test_reader_evidence_and_answer_survive_reopen_without_cross_run_collision(tmp_path):
    path = tmp_path/'runtime.sqlite3'
    db = RuntimeDatabase(path)
    first, second = db.create_run('one', Budget()), db.create_run('two', Budget())
    source = [{'evidence_id': 'old_id', 'content': 'Source fact.', 'source_uri': 'paper://sha/chunk', 'page': 2}]
    a = persist_reader_evidence(db, first, source)[0]['evidence_id']
    b = persist_reader_evidence(db, second, source)[0]['evidence_id']
    assert a != b
    assert persist_reader_evidence(db, first, source)[0]['evidence_id'] == a
    answer = f'Source fact. [{a}]'
    publish_reader_answer(db, first, {'status': 'REVIEW_REQUIRED', 'answer': answer})
    reopened = RuntimeDatabase(path)
    assert reopened.get_run(first)['metadata']['final_answer']['answer'] == answer
    assert reopened.list_evidence(first)[0]['content'] == 'Source fact.'
    assert reopened.list_evidence(first)[0]['polarity'] == 'NEUTRAL'
    assert len(reopened.list_evidence(first)) == 1
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    restored = ResearchAgentRuntime(db_path=path, workspace=tmp_path).status(first, include_trace=False)
    assert restored['final_answer']['answer'] == answer
    assert restored['evidence'][0]['evidence_id'] == a
    with pytest.raises(ValueError, match='outside'):
        publish_reader_answer(reopened, second, {'status': 'REVIEW_REQUIRED', 'answer': answer})


class AnswerProvider:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, *, before_attempt, **kwargs):
        before_attempt()
        self.calls += 1
        payload = json.loads(messages[-1]['content'])
        if 'evidence' in payload:
            value = {'facts': [{'id': 'f1', 'text': 'A source fact.',
                               'evidence_ids': [payload['evidence'][0]['evidence_id']]}],
                     'direct_answer_id': 'f1', 'limitations': []}
        else:
            value = {'assessments': [{'fact_id': 'f1', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                     'span_ids': [payload['facts'][0]['allowed_span_ids'][0]]}],
                     'coverage': 'COMPLETE', 'missing_aspects': []}
        return LLMResponse(text=json.dumps(value), prompt_tokens=10, completion_tokens=5,
                           finish_reason='stop')


def test_visual_evidence_keeps_region_and_provenance_without_promoting_confidence(tmp_path):
    path = tmp_path/'visual.sqlite3'
    db = RuntimeDatabase(path)
    rid = db.create_run('Table?', Budget())
    item = {'evidence_id': 'upstream', 'evidence_type': 'TABLE', 'content': 'A: 12',
            'source_uri': 'paper://sha/layout/table', 'page': 3, 'bbox': [1, 2, 30, 40],
            'object_type': 'table', 'artifact_path': str(tmp_path/'region.png'),
            'confidence': .99, 'provenance': {'vision_verified': True,
                'analysis_mode': 'multimodal_adapter', 'layout_object_id': 'table1',
                'vision_metadata': {'model': 'fixture', 'uncertain_fields': []}}}
    first = persist_reader_evidence(db, rid, [item])[0]
    duplicate = persist_reader_evidence(db, rid, [item])[0]
    assert first['evidence_id'] == duplicate['evidence_id']
    other = persist_reader_evidence(db, rid, [{**item, 'bbox': [2, 3, 31, 41]}])[0]
    assert other['evidence_id'] != first['evidence_id']
    rows = RuntimeDatabase(path).list_evidence(rid)
    restored = next(row for row in rows if row['evidence_id'] == first['evidence_id'])
    assert restored['evidence_type'] == 'TABLE'
    assert restored['bbox'] == item['bbox']
    assert restored['artifact_path'] == item['artifact_path']
    assert restored['provenance']['vision_metadata'] == item['provenance']['vision_metadata']
    assert restored['confidence'] == 0
    assert restored['polarity'] == 'NEUTRAL'
    assert len(rows) == 2


def test_source_identity_metadata_cannot_collide_at_the_same_reader_id(tmp_path):
    db = RuntimeDatabase(tmp_path/'identity.sqlite3')
    rid = db.create_run('Source?', Budget())
    item = {'evidence_id': 'upstream', 'content': 'Same text',
            'source_uri': 'paper://sha/chunk', 'page': 1}
    first = persist_reader_evidence(db, rid, [{**item, 'provenance': {'stable_id': 'a'}}])[0]
    second = persist_reader_evidence(db, rid, [{**item, 'provenance': {'stable_id': 'b'}}])[0]
    assert first['evidence_id'] != second['evidence_id']
    assert len(db.list_evidence(rid)) == 2


def reader_setup(tmp_path):
    db = RuntimeDatabase(tmp_path/'runtime.sqlite3')
    run_id = db.create_run('What is the fact?', Budget())
    provider = AnswerProvider()
    model = DurableResearchModel(PersistentLLMRuntime(db, provider), run_id,
                                 stage='answer_first', reserve_class='synthesis')
    evidence = [{'evidence_id': 'source1', 'content': 'A source fact.',
                 'source_uri': 'paper://sha/chunk', 'page': 1}]
    return db, run_id, provider, model, evidence


def test_durable_reader_replay_after_reopen_does_not_call_model(tmp_path):
    db, rid, provider, model, evidence = reader_setup(tmp_path)
    report = run_durable_reader(db, rid, evidence, model, protocol_id='test-model-v1')
    assert report['status'] == 'REVIEW_REQUIRED'
    assert provider.calls == 2
    reopened = RuntimeDatabase(db.path)
    replay = run_durable_reader(reopened, rid, evidence, model, protocol_id='test-model-v1')
    assert replay == report
    assert provider.calls == 2
    assert reopened.get_run(rid)['used_llm_calls'] == 2
    assert len(reopened.list_evidence(rid)) == 1
    assert reopened.get_run(rid)['research_outcome'] == 'ANSWERED'
    assert reopened.get_run(rid)['metadata']['reader_report']['semantic_correctness'] == 'MODEL_AUDIT_ONLY_NOT_INDEPENDENT_GROUND_TRUTH'
    for changed, protocol in [(evidence, 'other-model'),
                               ([{**evidence[0], 'content': 'Changed.'}], 'test-model-v1')]:
        with pytest.raises(ValueError, match='changed'):
            run_durable_reader(reopened, rid, changed, model, protocol_id=protocol)
    assert provider.calls == 2


def test_changed_request_history_cannot_reuse_finished_checkpoint(tmp_path):
    db, rid, provider, model, evidence = reader_setup(tmp_path)
    db.update_run_metadata(rid, {'answer_request_history': [{'user_request': 'Be concise.'}]})
    run_durable_reader(db, rid, evidence, model, protocol_id='v1')
    count = provider.calls
    db.update_run_metadata(rid, {'answer_request_history': [{'user_request': 'Explain in detail.'}]})
    with pytest.raises(ValueError, match='changed'):
        run_durable_reader(db, rid, evidence, model, protocol_id='v1')
    assert provider.calls == count


def test_inflight_reader_blocks_reentry_and_crash_replay(tmp_path):
    db, rid, provider, model, evidence = reader_setup(tmp_path)

    def interrupt(report):
        with pytest.raises(ReaderAlreadyStarted):
            run_durable_reader(RuntimeDatabase(db.path), rid, evidence, model, protocol_id='v1')
        raise KeyboardInterrupt('simulated process interruption')

    with pytest.raises(KeyboardInterrupt):
        run_durable_reader(db, rid, evidence, model, protocol_id='v1', on_progress=interrupt)
    with pytest.raises(ReaderAlreadyStarted):
        run_durable_reader(RuntimeDatabase(db.path), rid, evidence, model, protocol_id='v1')
    assert provider.calls == 0


def test_saved_reader_survives_failure_during_answer_publication(tmp_path, monkeypatch):
    import paperpilot_agentic.durable_reader as module
    db, rid, provider, model, evidence = reader_setup(tmp_path)
    publish = module.publish_reader_answer

    def crash(*args):
        raise OSError('simulated publication interruption')

    monkeypatch.setattr(module, 'publish_reader_answer', crash)
    with pytest.raises(OSError):
        run_durable_reader(db, rid, evidence, model, protocol_id='v1')
    assert provider.calls == 2
    monkeypatch.setattr(module, 'publish_reader_answer', publish)
    report = run_durable_reader(RuntimeDatabase(db.path), rid, evidence, model, protocol_id='v1')
    assert provider.calls == 2
    assert db.get_run(rid)['metadata']['final_answer']['answer'] == report['answer']


def test_failed_provider_is_saved_and_not_automatically_retried(tmp_path):
    db, rid, provider, _, evidence = reader_setup(tmp_path)

    class LostResponse:
        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            provider.calls += 1
            raise TimeoutError('provider outcome unknown')

    model = DurableResearchModel(PersistentLLMRuntime(db, LostResponse()), rid, stage='answer_first')
    report = run_durable_reader(db, rid, evidence, model, protocol_id='v1')
    assert report['status'] == 'FAILED'
    assert report['error_type'] == 'TimeoutError'
    assert run_durable_reader(db, rid, evidence, model, protocol_id='v1') == report
    assert provider.calls == 1
    assert db.get_run(rid)['used_llm_calls'] == 1


def test_concurrent_workers_acquire_only_one_reader_slot(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from paperpilot_agentic.durable_reader import _reader_checkpoint
    db, rid, _, _, _ = reader_setup(tmp_path)
    barrier = Barrier(2)

    def acquire():
        connection = RuntimeDatabase(db.path)
        barrier.wait(timeout=10)
        try:
            _reader_checkpoint(connection, rid, 'same-input')
            return 'acquired'
        except ReaderAlreadyStarted:
            return 'blocked'

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(acquire) for _ in range(2)]
        assert sorted(f.result(timeout=15) for f in futures) == ['acquired', 'blocked']
