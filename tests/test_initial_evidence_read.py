"""Initial source selection may reduce Writer inputs, never audit obligations."""
import json
from copy import deepcopy

import pytest

from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.answer_contracts import evidence_spans, audit_source_view
from paperpilot_agentic.direct_answer import prepare_evidence


SOURCES = [
    {'evidence_id': 'method', 'source_uri': 'paper://alpha/method', 'page': 2,
     'content': 'The encoder uses local attention with fixed normalization.'},
    {'evidence_id': 'appendix', 'source_uri': 'paper://alpha/appendix', 'page': 9,
     'content': 'Unrelated archival details. ' * 300},
]
DRAFT = {'facts': [{'id': 'f1', 'kind': 'SOURCE_FACT', 'text': SOURCES[0]['content'],
                    'evidence_ids': ['method']}], 'direct_answer_id': 'f1', 'limitations': []}
AUDIT = {'assessments': [{'fact_id': 'f1', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                           'span_ids': ['s0']}], 'coverage': 'COMPLETE', 'missing_aspects': []}


class Model:
    def __init__(self, selection=None, draft=None):
        self.outputs = [selection or {'evidence_ids': ['method']}, draft or deepcopy(DRAFT), deepcopy(AUDIT)]
        self.messages = []
        self.preflights = []

    def ensure_capacity(self, requests):
        self.preflights.append(deepcopy(requests))

    def complete(self, messages, **kwargs):
        kwargs['before_attempt']()
        self.messages.append(deepcopy(messages))
        return LLMResponse(json.dumps(self.outputs.pop(0)), finish_reason='stop')


def test_whole_source_read_only_writer_is_reduced_and_full_audit_survives():
    before = deepcopy(SOURCES)
    model = Model()
    report = answer_then_audit('Explain the encoder.', SOURCES, model,
                              evidence_read_policy='navigation-v1')
    assert report['status'] == 'REVIEW_REQUIRED', report.get('failure_detail')
    assert len(model.messages) == report['provider_attempts'] == 3
    assert 'NOT_SOURCE_EVIDENCE' in model.messages[0][0]['content']
    assert 'json' in model.messages[0][0]['content'].casefold()
    assert json.loads(model.messages[1][-1]['content'])['evidence'] == SOURCES[:1]
    assert SOURCES[1]['content'] not in model.messages[1][-1]['content']
    assert json.loads(model.messages[2][-1]['content'])['sources'] == audit_source_view(
        evidence_spans(prepare_evidence(SOURCES)))
    assert report['initial_evidence_read']['unread_evidence_ids'] == ['appendix']
    assert report['initial_evidence_read']['semantic_coverage'] == 'NOT_PROVEN'
    assert SOURCES == before
    assert report['budget_preflight']['message_utf8_bytes'][0] == len(
        json.dumps(model.messages[1], ensure_ascii=False).encode())


@pytest.mark.parametrize('selection', [
    {'evidence_ids': ['unknown']}, {'evidence_ids': ['method', 'method']},
    {'evidence_ids': []}, {'evidence_ids': ['method'], 'answer': 'unreviewed'},
])
def test_invalid_selection_has_no_writer_or_retry(selection):
    model = Model(selection)
    report = answer_then_audit('Explain the encoder.', SOURCES, model,
                              evidence_read_policy='navigation-v1')
    assert report['status'] == 'FAILED'
    assert len(model.messages) == 1
    assert not report.get('answer')


def test_unread_navigation_cannot_be_cited_as_source_even_when_audit_knows_it():
    draft = deepcopy(DRAFT)
    draft['facts'][0]['evidence_ids'] = ['appendix']
    model = Model(draft=draft)
    report = answer_then_audit('Explain the encoder.', SOURCES, model,
                              evidence_read_policy='navigation-v1', allow_repair=False)
    assert report['status'] == 'FAILED'
    assert len(model.messages) == 2
    assert not report.get('answer')


def test_original_request_and_history_reach_selection_writer_and_audit():
    history = [{'user_request': 'Use an 8GB device and do not start training.'}]
    model = Model()
    answer_then_audit('Explain the encoder.', SOURCES, model, evidence_read_policy='navigation-v1',
                     original_question='Explain only the encoder mechanism.', request_history=history)
    for messages in model.messages:
        assert '8GB' in messages[-1]['content']
        assert 'do not start training' in messages[-1]['content']
        assert 'Explain only the encoder mechanism.' in messages[-1]['content']


def test_every_paper_and_requested_table_conditions_remain_required():
    from paperpilot_agentic.initial_evidence_read import mandatory_ids, resolve_selection
    from paperpilot_agentic.answer_coverage import request_contract
    rows = [*SOURCES, {'evidence_id': 'table', 'source_uri': 'paper://alpha/table',
                      'content': 'TABLE IV\nDataset Delta, FP32, fixed encoder, F1 70.25'},
            {'evidence_id': 'second', 'source_uri': 'paper://beta/method',
             'content': 'A distinct second paper.'}]
    contract = request_contract('Report Table IV ablation results.', [],
                               [{'paper_sha256': 'alpha', 'label': 'Table IV'}])
    required = mandatory_ids(rows, [], contract)
    assert required == ['method', 'table', 'second']
    selected = resolve_selection({'evidence_ids': ['method']}, rows, required, 24000)
    assert selected == [rows[0], rows[2], rows[3]]
    assert 'FP32, fixed encoder' in selected[1]['content']


def test_requested_oversized_read_stops_instead_of_cutting_or_retrying():
    model = Model({'evidence_ids': ['appendix']})
    report = answer_then_audit('Explain the encoder.', SOURCES, model,
                              evidence_read_policy='navigation-v1', evidence_read_max_bytes=1000)
    assert report['status'] == 'FAILED'
    assert 'no passage was truncated' in report['failure_detail']
    assert len(model.messages) == 1


def test_mandatory_source_too_large_and_initial_budget_fail_send_nothing():
    from paperpilot_agentic.llm_runtime import LLMBudgetExhausted
    model = Model()
    report = answer_then_audit('Explain the encoder.', [SOURCES[1]], model,
                              evidence_read_policy='navigation-v1', evidence_read_max_bytes=1000)
    assert report['status'] == 'FAILED'
    assert not model.messages

    class NoBudget(Model):
        def ensure_capacity(self, requests):
            assert len(requests) == 3
            raise LLMBudgetExhausted('Three stage floor cannot fit')
    model = NoBudget()
    report = answer_then_audit('Explain the encoder.', SOURCES, model, evidence_read_policy='navigation-v1')
    assert report['error_type'] == 'LLMBudgetExhausted'
    assert report['provider_attempts'] == 0
    assert not model.messages


def test_selected_writer_and_full_audit_preflight_fail_before_writer_dispatch():
    from paperpilot_agentic.llm_runtime import LLMBudgetExhausted
    class LaterBudget(Model):
        def ensure_capacity(self, requests):
            super().ensure_capacity(requests)
            if len(self.preflights) == 2:
                assert len(requests) == 2
                assert json.loads(requests[1]['messages'][-1]['content'])['sources'] == audit_source_view(
                    evidence_spans(prepare_evidence(SOURCES)))
                raise LLMBudgetExhausted('Full audit cannot fit after selection')
    model = LaterBudget()
    report = answer_then_audit('Explain the encoder.', SOURCES, model, evidence_read_policy='navigation-v1')
    assert report['error_type'] == 'LLMBudgetExhausted'
    assert report['provider_attempts'] == len(model.messages) == 1
    assert not report.get('answer')


@pytest.mark.parametrize('kwargs', [{'frozen_draft': DRAFT}, {'audit_cited_only': True},
                                  {'revision_feedback': {}}])
def test_reject_conflicting_diagnostic_or_revision_modes_without_calls(kwargs):
    model = Model()
    with pytest.raises(ValueError, match='generated first draft'):
        answer_then_audit('Explain the encoder.', SOURCES, model, evidence_read_policy='navigation-v1', **kwargs)
    assert not model.messages


def test_persistent_completion_reused_and_policy_or_source_change_rejected(tmp_path):
    from paperpilot_agentic.db import RuntimeDatabase
    from paperpilot_agentic.models import Budget
    from paperpilot_agentic.durable_reader import DurableResearchModel, run_durable_reader
    from paperpilot_agentic.llm_runtime import PersistentLLMRuntime

    class Provider:
        def __init__(self):
            self.calls = 0
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            self.calls += 1
            payload = json.loads(messages[-1]['content'])
            if self.calls == 1:
                result = {'evidence_ids': [payload['source_directory'][0]['evidence_id']]}
            elif self.calls == 2:
                result = deepcopy(DRAFT)
                result['facts'][0]['evidence_ids'] = [payload['evidence'][0]['evidence_id']]
            else:
                result = deepcopy(AUDIT)
            return LLMResponse(json.dumps(result), prompt_tokens=10, completion_tokens=5, finish_reason='stop')

    db = RuntimeDatabase(tmp_path / 'runtime.sqlite3')
    rid = db.create_run('Explain the encoder.', Budget())
    db.update_run_metadata(rid, {'reader_evidence_read_policy': 'navigation-v1'})
    provider = Provider()
    model = DurableResearchModel(PersistentLLMRuntime(db, provider), rid,
                                 stage='answer_first', reserve_class='synthesis')
    first = run_durable_reader(db, rid, SOURCES, model, protocol_id='fixed', finalize=False)
    assert first['status'] == 'REVIEW_REQUIRED', first.get('failure_detail')
    assert run_durable_reader(db, rid, SOURCES, model, protocol_id='fixed', finalize=False) == first
    assert db.get_run(rid)['used_llm_calls'] == provider.calls == 3
    assert db.get_run(rid)['used_tokens'] == 45
    changed = deepcopy(SOURCES)
    changed[0]['content'] += ' Different condition.'
    with pytest.raises(ValueError, match='changed'):
        run_durable_reader(db, rid, changed, model, protocol_id='fixed', finalize=False)
    db.update_run_metadata(rid, {'reader_evidence_read_policy': 'full'})
    with pytest.raises(ValueError, match='changed'):
        run_durable_reader(db, rid, SOURCES, model, protocol_id='fixed', finalize=False)
    assert provider.calls == 3


def test_navigation_mode_preserves_existing_known_omission_recovery(tmp_path):
    from test_reader_omission_revision import InventorySearch, OmissionModel
    from test_session_entry_contract import entry_workbench
    from paperpilot_agentic.reader_pipeline import resume_text_reader

    class WithSelection(OmissionModel):
        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]['content'])
            if 'source_directory' in payload:
                kwargs['before_attempt']()
                return LLMResponse(json.dumps({'evidence_ids': [payload['source_directory'][0]['evidence_id']]}),
                                   finish_reason='stop', model=self.model)
            return super().complete(messages, **kwargs)

    wb = entry_workbench(tmp_path, WithSelection())
    wb.runtime.search_adapter = InventorySearch()
    root = wb.submit({'request_id': 'navigation-omission', 'action': 'auto',
                      'question': 'List the datasets OrionModel evaluates on.', 'paper_ids': [f'{1:064x}']})
    rid = root['run_id']
    wb.runtime.db.update_run_metadata(rid, {'reader_evidence_read_policy': 'navigation-v1'})
    wb.runtime.resume(rid)
    view = wb.turn(rid)
    assert view['status'] == 'COMPLETED', view['error']
    assert 'DeltaSet' in view['message']['answer'] and 'BetaSet' in view['message']['answer']
    assert len(view['message']['citations']) == 2
    metadata = wb.runtime.db.get_run(rid)['metadata']
    assert metadata['answer_first_checkpoint']['report']['initial_evidence_read']['status'] == 'READ'
    assert 'initial_evidence_read' not in metadata['answer_first_reload_checkpoint']['report']
    wb.runtime.db.update_run_metadata(rid, {'reader_evidence_read_policy': 'full'})
    with pytest.raises(ValueError, match='changed'):
        resume_text_reader(wb.runtime, rid, protocol_id=wb.runtime.text_reader_protocol)


def test_unknown_selection_and_refresh_keep_hold_without_sending_again(tmp_path):
    from paperpilot_agentic.db import RuntimeDatabase
    from paperpilot_agentic.models import Budget
    from paperpilot_agentic.durable_reader import DurableResearchModel, run_durable_reader
    from paperpilot_agentic.llm_runtime import PersistentLLMRuntime

    class Lost:
        calls = 0
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            self.calls += 1
            raise TimeoutError('offline unknown selection result')
    db = RuntimeDatabase(tmp_path / 'unknown.sqlite3')
    rid = db.create_run('Explain the encoder.', Budget())
    db.update_run_metadata(rid, {'reader_evidence_read_policy': 'navigation-v1'})
    provider = Lost()
    runtime = PersistentLLMRuntime(db, provider)
    model = DurableResearchModel(runtime, rid, stage='answer_first', reserve_class='synthesis')
    report = run_durable_reader(db, rid, SOURCES, model, protocol_id='fixed', finalize=False)
    assert report['error_type'] == 'TimeoutError'
    assert report['initial_evidence_read']['status'] == 'STOPPED_ERROR'
    snapshot = runtime.attempt_ledger.snapshot(rid)
    assert snapshot['unknown_results'] == 1 and snapshot['reserved_unknown_tokens'] > 0
    restored = run_durable_reader(RuntimeDatabase(db.path), rid, SOURCES, model,
                                  protocol_id='fixed', finalize=False)
    assert restored == report
    assert runtime.attempt_ledger.snapshot(rid) == snapshot
    assert provider.calls == db.get_run(rid)['used_llm_calls'] == 1
    assert not report.get('answer')
