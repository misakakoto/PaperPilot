"""Default Writer/Auditor requests must not duplicate source-owned cells."""
from copy import deepcopy
import json
import pytest

from paperpilot_agentic.answer_pipeline import (
    AnswerOptions, initialize, prepared_messages, writer_messages, audit_floor_request,
)
from paperpilot_agentic.structured_extraction import ParsedArtifact, SourceRegion
from paperpilot_agentic.request_packet import compact_payload, restore_value_references


def fixture():
    rows = [{'row_label': '', 'checked_columns': [], 'cells': {'F1': '3.25', 'Dataset': 'Delta', 'Offset': '8px'}},
            {'row_label': '', 'checked_columns': ['Gate'], 'cells': {'F1': '4.50', 'Dataset': 'Delta', 'Offset': '8px'}}]
    item = ParsedArtifact(kind='table', source=SourceRegion(paper_sha256='p', object_id='o',
        label='Table IV', page=3, bbox=[0, 0, 10, 10], crop_sha256='crop'), parser='fixture',
        parser_version='2', config_sha256='cfg', parse_status='PARSED', validation_status='VALIDATED',
        column_headers=['Gate', 'F1', 'Dataset', 'Offset'], table_rows=rows)
    text = 'Table IV\nNamed-cell transcription (each row is independent):\n' + json.dumps(
        {'column_headers': item.column_headers, 'rows': rows, 'note': 'Values in percent; fixed offset.'}, ensure_ascii=False)
    return {'evidence_id': 't', 'source_uri': 'paper://p/layout/o', 'page': 3, 'content': text,
            'provenance': {'structured_extraction': item.model_dump()} }


def context():
    q = 'Report Table IV on Delta; retain its offset, units and condition.'
    source = fixture()
    inventory = [{'paper_sha256': 'p', 'label': 'Table IV', 'status': 'PRESENT'},
                 {'paper_sha256': 'p', 'label': 'Table IX', 'status': 'PRESENT'}]
    return initialize(q, [source], None, AnswerOptions(original_question=q,
        source_inventory=inventory), None)


def test_default_writer_and_audit_floor_send_each_literal_cell_once():
    ctx = context()
    messages = [prepared_messages(ctx, writer_messages(ctx)), audit_floor_request(ctx)['messages']]
    for request in messages:
        payload = json.loads(request[-1]['content'])
        assert payload['original_question'] == ctx.question
        assert request[-1]['content'].count('3.25') == 1
        assert payload['read_packet_sha256'] == ctx.report['read_packet']['packet_sha256']
        assert 'source_texts' not in payload and 'allowed_span_sets' not in payload


@pytest.mark.parametrize('audit', [False, True])
def test_simple_source_row_references_restore_exact_values_and_originals(audit):
    ctx = context()
    request = audit_floor_request(ctx)['messages'] if audit else prepared_messages(ctx, writer_messages(ctx))
    payload = json.loads(request[-1]['content'])
    restored = restore_value_references(payload)
    for row, owned in zip(restored['table_rows'], ctx.report['table_rows']):
        assert row['cells'] == owned['cells']
        assert row['row_id'] == owned['row_id']
        assert row['checked_columns'] == owned['checked_columns']
    assert restored['original_question'] == ctx.question
    assert restored['request_contract'] == ctx.report['request_contract']
    assert restored['source_inventory_navigation_only'] == ctx.report['source_inventory'][:1]
    assert restored['source_inventory_scope']['absence_is_not_established']
    if audit:
        from paperpilot_agentic.answer_contracts import audit_source_view, evidence_spans
        assert restored['sources'] == audit_source_view(evidence_spans(ctx.rows))
    else:
        assert restored['evidence'] == ctx.rows
    # Source qualifiers, note, cells and original page/URI remain verbatim.
    assert 'Values in percent; fixed offset.' in json.dumps(restored)


@pytest.mark.parametrize('mutation', ['changed_cells', 'changed_flags', 'wrong_index', 'duplicate', 'damaged', 'cross_owner'])
def test_nonidentical_or_unowned_table_is_not_deduplicated(mutation):
    ctx = context()
    payload = restore_value_references(json.loads(prepared_messages(ctx, writer_messages(ctx))[-1]['content']))
    if mutation == 'changed_cells':
        payload['table_rows'][0]['cells']['F1'] = '999.9'
    elif mutation == 'changed_flags':
        payload['table_rows'][0]['checked_columns'] = ['Gate']
    elif mutation == 'wrong_index':
        payload['table_rows'][0]['row_index'] = 99
    elif mutation == 'duplicate':
        payload['evidence'][0]['content'] *= 2
    elif mutation == 'damaged':
        payload['evidence'][0]['content'] += 'Named-cell transcription (each row is independent): {bad'
    else:
        payload['table_rows'][0]['evidence_id'] = 'other'
    before = deepcopy(payload)
    compact = compact_payload(payload, direct_sources=True)
    assert compact['table_rows'][0]['cells'] == payload['table_rows'][0]['cells']
    assert payload == before
    assert restore_value_references(compact)['table_rows'] == before['table_rows']


def test_direct_source_mode_does_not_merge_identical_spans_or_per_fact_allowlists():
    text = 'A qualifier and condition. ' * 20
    original = {'question': 'Compare.', 'sources': [{'evidence_id': 'a', 'spans': {'s0': text}},
                                                  {'evidence_id': 'b', 'spans': {'s1': text}}],
                'facts': [{'id': 'f', 'evidence_ids': ['a'], 'allowed_span_ids': [f's{i}' for i in range(20)]}]}
    value = compact_payload(original, direct_sources=True)
    assert value['sources'] == original['sources'] and value['facts'] == original['facts']
    assert 'source_texts' not in value and 'allowed_span_sets' not in value


def test_request_history_keeps_order_revocation_and_changed_current_wording():
    original = {'question': 'Resolved retrieval wording', 'original_question': 'Compare mechanisms.',
                'request_history': [{'role': 'initial_request', 'user_request': 'Keep 8GB; do not train.'},
                                    {'role': 'current_request', 'user_request': 'Cancel brevity.'},
                                    {'role': 'current_request', 'user_request': 'Compare mechanisms.'}]}
    value = restore_value_references(compact_payload(original, direct_sources=True))
    assert value == original


def test_complete_rows_in_split_caption_remain_owned_by_the_same_audit_span():
    from paperpilot_agentic.answer_contracts import evidence_spans, audit_source_view
    ctx = context()
    source = deepcopy(ctx.rows[0])
    data = json.loads(source['content'].split('Named-cell transcription (each row is independent):')[1])
    data['caption'] = 'A fixed evaluation. The remaining caption is also source text.'
    source['content'] = source['content'].split('Named-cell transcription (each row is independent):')[0] + (
        'Named-cell transcription (each row is independent):\n' + json.dumps(data))
    original = {'question': ctx.question, 'sources': audit_source_view(evidence_spans([source])),
                'table_rows': [{key: row[key] for key in ('evidence_id','row_id','row_index','row_label','checked_columns','cells')}
                               for row in ctx.report['table_rows']]}
    assert len(original['sources'][0]['spans']) == 2
    compact = compact_payload(original, direct_sources=True)
    assert all('cells_source_row' in row for row in compact['table_rows'])
    restored = restore_value_references(compact)
    assert restored['sources'] == original['sources']
    assert restored['table_rows'] == original['table_rows']


def test_budget_rejection_retains_numeric_admission_diagnostic_without_dispatch():
    from paperpilot_agentic.answer_audit import answer_then_audit
    from paperpilot_agentic.llm_runtime import LLMBudgetExhausted
    class Model:
        calls = 0
        def ensure_capacity(self, requests):
            exc = LLMBudgetExhausted('Known sequence is too large')
            exc.budget_diagnostic = {'required_upper_bound': 110, 'available_tokens': 100,
                'reported_tokens': 900, 'max_tokens': 1000, 'unsettled_upper_bound': 0,
                'request_cost_policy': 'legacy-utf8-v1', 'private_value': 'must not be copied'}
            raise exc
        def complete(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError('No dispatch after failed admission')
    model = Model()
    report = answer_then_audit('Report Table IV.', [fixture()], model)
    assert report['status'] == 'FAILED' and report['provider_attempts'] == model.calls == 0
    assert report['budget_diagnostic']['required_upper_bound'] == 110
    assert report['budget_diagnostic']['available_tokens'] == 100
    assert 'private_value' not in report['budget_diagnostic']


def test_cost_replay_includes_runtime_message_without_modifying_source_database(tmp_path):
    import hashlib
    from evals.interview.request_cost_replay import measure
    from paperpilot_agentic.db import RuntimeDatabase
    from paperpilot_agentic.models import Budget
    ctx = context()
    db = RuntimeDatabase(tmp_path/'runtime.sqlite3')
    rid = db.create_run(ctx.question, Budget(max_tokens=300000))
    result = {'run': {**db.get_run(rid), 'metadata': {
        'reader_report': ctx.report, 'reader_source_inventory': ctx.options.source_inventory}},
        'evidence': [fixture()]}
    result_path = tmp_path/'result.json'
    result_path.write_text(json.dumps(result), encoding='utf-8')
    requests = [{'messages': prepared_messages(ctx, writer_messages(ctx)), 'max_tokens': 4000},
                audit_floor_request(ctx)]
    baseline = deepcopy(requests)
    for request in baseline:
        request['messages'][-1]['content'] = json.dumps(restore_value_references(
            json.loads(request['messages'][-1]['content'])), ensure_ascii=False)
    baseline_path = tmp_path/'messages.json'
    baseline_path.write_text(json.dumps(baseline), encoding='utf-8')
    before = hashlib.sha256(db.path.read_bytes()).hexdigest()
    replay = measure(result_path, baseline_path, available_tokens=300000)
    assert replay['actual_requests'] == 0 and replay['runtime_budget_advisory_included']
    assert replay['source_database_unchanged'] and replay['hypothetical_known_pair_fits']
    assert not replay['completion_guaranteed']
    assert hashlib.sha256(db.path.read_bytes()).hexdigest() == before
