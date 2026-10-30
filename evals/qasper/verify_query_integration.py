"""Verify raw/default production rankings against already frozen paired results."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from evals.qasper.agent_case import paper_records
from paperpilot_retrieval import PaperKnowledgeStore
from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter


def verify(dataset_path, frozen, output):
    raw = dataset_path.read_bytes()
    manifest = json.loads((frozen/'manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(raw).hexdigest() == manifest['dataset_sha256']
    expected = {r['question_id']: r for r in (
        json.loads(line) for line in (frozen/'rankings.jsonl').read_text(encoding='utf-8').splitlines())}
    data = json.loads(raw)
    assert set(expected) == {q['question_id'] for p in data.values() for q in p['qas']}
    checked, mismatches = 0, []
    for index, (pid, paper) in enumerate(data.items()):
        _, document, chunks = paper_records({k: v for k, v in paper.items() if k != 'qas'}, source_id=pid)
        with tempfile.TemporaryDirectory(prefix='paperpilot-query-verify-') as directory:
            path = Path(directory)/'index.sqlite3'
            default_store = PaperKnowledgeStore(path, namespace='paired')
            default_store.sync_documents([document], chunks)
            raw_store = PaperKnowledgeStore(path, namespace='paired', lexical_query_policy='raw')
            for qa in paper['qas']:
                saved = expected[qa['question_id']]
                assert saved['question'] == qa['question'] and saved['paper_id'] == pid
                for label, store in [('baseline', raw_store), ('content_query', default_store)]:
                    hits = PaperPilotKnowledgeSearchAdapter(store).search(qa['question'], top_k=8,
                                                                         paper_sha256s={document.sha256})
                    ranks = [hit['stable_id'] for hit in hits]
                    if ranks != saved[label]['ranks']:
                        mismatches.append({'question_id': qa['question_id'], 'profile': label,
                                           'expected': saved[label]['ranks'], 'actual': ranks})
                checked += 1
        if (index + 1) % 50 == 0:
            print(f'Checked {index + 1}/{len(data)} papers', flush=True)
    report = {'questions': checked, 'ranking_comparisons': checked * 2, 'mismatches': mismatches,
              'passed': not mismatches, 'provider_requests': 0,
              'scope': 'Exact replay, not another round of parameter selection'}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))
    assert not mismatches, 'Production integration differs from frozen candidate or baseline'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.dataset, args.frozen, args.output)
