"""Replay saved navigation responses against read-only originals; no provider."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.reader_recovery import plan_evidence_reload
from paperpilot_agentic.reader_workset import source_aware_workset
from paperpilot_agentic.section_router import read_outline_fallback, select_sections
from paperpilot_retrieval import PaperKnowledgeStore


class ReadOnlyStore(PaperKnowledgeStore):
    def __init__(self, path, namespace, expected_chunks):
        # Never run the writable store's initialization against a frozen corpus.
        self.path, self.namespace = Path(path).resolve(), namespace
        self.expected_chunks = expected_chunks

    def get_chunks(self, stable_ids):
        rows = super().get_chunks(stable_ids)
        for row in rows:
            expected = self.expected_chunks.get((row.paper_sha256, row.stable_id))
            if expected != hashlib.sha256(row.text.encode()).hexdigest():
                raise ValueError('Replay original differs from the frozen source snapshot')
        return rows

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()


class SavedResponse:
    def __init__(self, response):
        self.response, self.replays = response, 0

    def complete(self, messages, *, before_attempt, **kwargs):
        before_attempt()
        self.replays += 1
        return LLMResponse(**self.response)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(campaign, revision, case, namespace):
    folder = campaign / 'revisions' / f'{revision:04d}'
    source = folder / f'{case}-result.json'
    originals_path, protocol_path = folder / 'original-sources.json', folder / 'protocol.json'
    protected = [source, originals_path, protocol_path,
                 campaign / 'receipts/spending.sqlite3', campaign / 'receipts/stopped.json']
    before = {str(path): digest(path) for path in protected}
    result = json.loads(source.read_text(encoding='utf-8'))
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if digest(originals_path) != protocol['source_snapshot_sha256']:
        raise ValueError('Frozen source snapshot identity changed')
    frozen_case = next(row for row in protocol['cases'] if row['id'] == case)
    if result['run']['question'] != frozen_case['question']:
        raise ValueError('Saved question differs from the frozen original')
    frozen_sources = json.loads(originals_path.read_text(encoding='utf-8'))
    expected_chunks = {(row['paper']['sha256'], chunk['stable_id']): chunk['sha256']
                       for row in frozen_sources for chunk in row['chunks']}
    metadata = result['run']['metadata']
    if not set(metadata['reader_routing_by_paper']) <= set(frozen_case['paper_ids']):
        raise ValueError('Saved navigation is outside the frozen source scope')
    store = ReadOnlyStore(campaign / 'corpus.sqlite3', namespace, expected_chunks)
    fallback, routes, replays = [], {}, 0
    for sha, previous in metadata['reader_routing_by_paper'].items():
        model = SavedResponse(previous['calls'][-1])
        outline = store.paper_outline(sha)
        routing = select_sections(result['run']['question'], outline, model)
        if routing['status'] == 'EMPTY_SELECTION':
            rows, info = read_outline_fallback(store, sha, outline, result['run']['question'])
            fallback.extend(rows)
            routing['fallback'] = info
            routing['selected_pages'] = [row['page'] for row in rows]
        routes[sha] = routing
        replays += model.replays
    originals = result['evidence']
    # Mirror the Reader's table-first, fallback-before-retrieval allocation. This
    # is a workset simulation, not a new answer or factual acceptance result.
    tables = [row for row in originals if row.get('evidence_type') == 'TABLE']
    candidates = {row['source_uri']: row for row in tables + fallback + originals}
    working, context = source_aware_workset(list(candidates.values()),
        max_bytes=metadata['reader_workset']['max_bytes'])
    saved = SavedResponse(metadata['reader_recovery']['response'])
    decision = plan_evidence_reload(result['run']['question'], metadata['reader_rounds'][0],
        originals, set(metadata['reader_workset']['selected_evidence_ids']), saved, allow_search=True)
    replays += saved.replays
    assert before == {str(path): digest(path) for path in protected}, 'Frozen records changed'
    return {'mode': 'OFFLINE_SAVED_RESPONSE_REPLAY', 'actual_provider_requests_added': 0,
        'saved_response_replays': replays, 'protected_sha256': before,
        'source_snapshot_validation': 'MATCHED',
        'original_run_id': result['run']['run_id'],
        'routing': {sha: {k: value[k] for k in ('status', 'section_ids', 'selected_pages', 'fallback')
                         if k in value} for sha, value in routes.items()},
        'simulated_workset': {**context, 'selected_pages_by_source': {
            sha: sorted({row['page'] for row in working if row['source_uri'].startswith(f'paper://{sha}/')})
            for sha in routes}},
        'recovery': {key: decision[key] for key in ('action', 'proposed_action', 'revision_validation')
                     if key in decision},
        'answer_correctness': 'NOT_EVALUATED', 'live_demo_passes_changed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir', required=True, type=Path)
    parser.add_argument('--revision', type=int, default=1)
    parser.add_argument('--case', default='multi_change_flow_comparison')
    parser.add_argument('--namespace', default='all-papers-v1')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Replay outputs are immutable; choose a new output path')
    report = replay(args.campaign_dir, args.revision, args.case, args.namespace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
