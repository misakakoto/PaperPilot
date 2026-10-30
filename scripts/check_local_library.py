"""Probe the user's local corpus using an isolated SQLite snapshot; no model calls."""
import argparse
import dataclasses
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
from paperpilot_retrieval import PaperKnowledgeStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--namespace', default='all-papers-v1')
    parser.add_argument('--configured-retrieval', action='store_true', help='Use configured local embedding/reranker models; never call LLMs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with sqlite3.connect(args.database.resolve().as_uri()+'?mode=ro', uri=True) as source:
        with sqlite3.connect(args.output/'knowledge-snapshot.sqlite3') as target:
            source.backup(target)
        records = source.execute('SELECT sha256,file_name,title FROM papers WHERE namespace=? AND active=1',
                                 (args.namespace,)).fetchall()
    files = list(args.root.rglob('*.pdf'))
    hashes = {}
    for path in files:
        with path.open('rb') as handle:
            identity = hashlib.file_digest(handle, 'sha256').hexdigest()
        hashes.setdefault(identity, []).append(str(path))
    indexed = {row[0] for row in records}
    adapter = PaperPilotKnowledgeSearchAdapter(
        PaperKnowledgeStore(args.output/'knowledge-snapshot.sqlite3', namespace=args.namespace),
        paper_roots=[args.root])
    retrieval_status = None
    if args.configured_retrieval:
        from paperpilot_agentic.bootstrap import build_runtime
        from paperpilot_agentic.config import RuntimeSettings
        settings = dataclasses.replace(RuntimeSettings.from_env(),
            memory_db=args.output/'knowledge-snapshot.sqlite3', memory_namespace=args.namespace,
            paper_roots=(args.root.resolve(),), workspace=args.output.resolve(),
            runtime_db=(args.output/'probe-runtime.sqlite3').resolve())
        runtime = build_runtime(settings, llm_factory=lambda _: None, vision_factory=lambda **_: object())
        adapter = runtime.search_adapter
        retrieval_status = adapter.retrieval_status
    searches = []
    for query in ['building change detection', 'SNUNet ECAM', 'ChangeMamba', '建筑变化检测']:
        hits = [dict(item) for item in adapter.search(query, top_k=6)]
        searches.append({'query': query, 'hits': hits,
                         'text_chunks': sum(bool(item.get('content')) for item in hits),
                         'unique_papers': len({item.get('paper_sha256') for item in hits})})
    report = {'root': str(args.root.resolve()), 'database': str(args.database.resolve()),
              'pdf_count': len(files), 'unique_pdf_hashes': len(hashes), 'indexed_papers': len(records),
              'indexed_hashes_present_in_root': len(indexed & set(hashes)),
              'unindexed_paths': [p for sha, paths in hashes.items() if sha not in indexed for p in paths],
              'indexed_missing_in_root': [dict(zip(['sha256','file_name','title'], row)) for row in records if row[0] not in hashes],
              'duplicates': [paths for paths in hashes.values() if len(paths)>1],
              'retrieval_mode': 'CONFIGURED_LOCAL_RETRIEVAL_NO_LLM' if args.configured_retrieval else 'LEXICAL_ONLY_NO_MODEL_OR_NETWORK',
              'retrieval_status': retrieval_status, 'searches': searches}
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['searches','duplicates','unindexed_paths','indexed_missing_in_root']}, ensure_ascii=True))
    print(json.dumps([{'query':s['query'], 'text_chunks':s['text_chunks'], 'unique_papers':s['unique_papers'],
                       'titles':[i.get('title') for i in s['hits']]} for s in searches], ensure_ascii=True))


if __name__ == '__main__':
    main()
