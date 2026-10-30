"""Small real-index feedback loop for a selected development failure."""
import argparse
import json
from pathlib import Path
import tempfile

from evals.qasper.agent_case import paper_records
from evals.qasper.retrieval_diagnostic import text_metrics
from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
from paperpilot_retrieval import PaperKnowledgeStore


def check(dataset, paper_id, question_id, *, content_only=False):
    paper = json.loads(Path(dataset).read_text(encoding='utf-8'))[paper_id]
    qa = next(q for q in paper['qas'] if q['question_id'] == question_id)
    public = {k: v for k, v in paper.items() if k != 'qas'}
    _, document, chunks = paper_records(public, source_id=paper_id)
    with tempfile.TemporaryDirectory(prefix='paperpilot-retrieval-repro-') as directory:
        store = PaperKnowledgeStore(Path(directory)/'index.sqlite3', namespace='repro',
                                    lexical_query_policy='content_v1' if content_only else 'raw')
        store.sync_documents([document], chunks)
        hits = PaperPilotKnowledgeSearchAdapter(store).search(qa['question'], top_k=8,
                                                            paper_sha256s={document.sha256})
        originals = {c.stable_id: c.text for c in store.get_chunks([h['stable_id'] for h in hits])}
        metrics = text_metrics(qa, [originals[h['stable_id']] for h in hits], [c.text for c in chunks])
        print(json.dumps({'question': qa['question'], 'metrics': metrics,
                          'ranks': [h['stable_id'] for h in hits]}, ensure_ascii=False))
        assert metrics['best_text_recall']['8'] == 1.0, 'Known evidence is absent from top-8'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--paper-id', required=True)
    parser.add_argument('--question-id', required=True)
    parser.add_argument('--content-query', action='store_true')
    args = parser.parse_args()
    check(args.dataset, args.paper_id, args.question_id, content_only=args.content_query)
