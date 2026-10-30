"""Measure saved request shapes without constructing a network provider."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from paperpilot_agentic.answer_pipeline import (
    AnswerOptions, initialize, prepared_messages, writer_messages, audit_floor_request,
)
from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
from paperpilot_agentic.request_packet import restore_value_references, SHARED_VERSION
from .budget import CampaignBudget, save


class NoNetworkProvider:
    def complete(self, *args, **kwargs):
        raise AssertionError('Cost replay cannot dispatch a provider request')


def measure(result_path, baseline_path, *, available_tokens=None):
    raw = result_path.read_bytes()
    result = json.loads(raw)
    metadata = result['run']['metadata']
    saved = metadata['reader_report']
    owned = {r['evidence_id']: r for r in result['evidence']}
    rows = [owned[eid] for eid in saved['read_packet']['selected_evidence_ids']]
    ctx = initialize(result['run']['question'], rows, NoNetworkProvider(), AnswerOptions(
        original_question=result['run']['question'], request_history=metadata.get('answer_request_history'),
        requested_tables=metadata.get('requested_table_targets'), source_inventory=metadata.get('reader_source_inventory'),
        relation_policy=saved.get('relation_policy', False),
        question_scope_policy=saved.get('question_scope_policy', False)), None)
    current = [{'messages': prepared_messages(ctx, writer_messages(ctx)), 'max_tokens': 4000},
               audit_floor_request(ctx)]
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    if len(baseline) != 2:
        raise ValueError('Expected saved Writer and audit-floor requests')
    database = result_path.parent / 'runtime.sqlite3'
    database_sha = hashlib.sha256(database.read_bytes()).hexdigest()
    workspace = result_path.parent.resolve()
    with TemporaryDirectory(prefix='paperpilot-cost-replay-', dir=workspace) as tmp:
        if Path(tmp).resolve().parent != workspace:
            raise ValueError('Isolated replay database escaped its result directory')
        copy = Path(tmp)/'runtime.sqlite3'
        with closing(sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True)) as source:
            with closing(sqlite3.connect(copy)) as target:
                source.backup(target)
        runtime = PersistentLLMRuntime(RuntimeDatabase(copy), NoNetworkProvider())
        snapshot = runtime.budget_snapshot(result['run']['run_id'])
        before = [{**r, 'messages': runtime._with_budget(r['messages'], snapshot)} for r in baseline]
        after = [{**r, 'messages': runtime._with_budget(r['messages'], snapshot)} for r in current]
    comparisons = []
    for old, new, old_full, new_full in zip(baseline, current, before, after):
        a = json.loads(old['messages'][-1]['content'])
        b = restore_value_references(json.loads(new['messages'][-1]['content']))
        for key in ('question', 'original_question', 'request_contract', 'request_history',
                    'evidence', 'sources', 'source_roles'):
            if key in a and a[key] != b[key]:
                raise ValueError('Protected request field changed: '+key)
        if len(a.get('table_rows', [])) != len(b.get('table_rows', [])):
            raise ValueError('Source row count changed')
        for arow, brow in zip(a.get('table_rows', []), b.get('table_rows', [])):
            if any(brow.get(key) != value for key, value in arow.items()):
                raise ValueError('Table values, configuration or identity changed')
        comparisons.append({
            'before_message_utf8_bytes': len(json.dumps(old_full['messages'], ensure_ascii=False).encode()),
            'after_message_utf8_bytes': len(json.dumps(new_full['messages'], ensure_ascii=False).encode()),
            'before_reservation': CampaignBudget.request_upper_bound(old_full),
            'after_reservation': CampaignBudget.request_upper_bound(new_full),
            'source_and_request_equivalence': True})
    assert result_path.read_bytes() == raw
    assert hashlib.sha256(database.read_bytes()).hexdigest() == database_sha
    before_cost = sum(row['before_reservation'] for row in comparisons)
    after_cost = sum(row['after_reservation'] for row in comparisons)
    return {'version': SHARED_VERSION, 'actual_requests': 0,
        'source_result_sha256': hashlib.sha256(raw).hexdigest(),
        'baseline_sha256': hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        'source_database_unchanged': True, 'runtime_budget_advisory_included': True,
        'rows': comparisons, 'known_pair_before': before_cost, 'known_pair_after': after_cost,
        'available_tokens': available_tokens,
        'hypothetical_known_pair_fits': None if available_tokens is None else after_cost <= available_tokens,
        'completion_guaranteed': False, 'unestimated': ['candidate facts', 'repair', 'presentation'],
        'stopped_campaign_remains_stopped': True,
        'note': 'Message/reservation comparison, not actual token savings or new model acceptance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--baseline-messages', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--available-tokens', type=int)
    args = parser.parse_args()
    if args.available_tokens is not None and args.available_tokens < 0:
        parser.error('Available tokens must be nonnegative')
    data = measure(args.result, args.baseline_messages, available_tokens=args.available_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, data)
    print(json.dumps(data))


if __name__ == '__main__':
    main()
