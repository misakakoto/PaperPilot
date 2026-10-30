"""Offline paired MMR ablation with identical real retrieval candidates."""
import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dotenv import load_dotenv  # noqa: E402
from paperpilot_agentic.bootstrap import build_runtime  # noqa: E402
from paperpilot_agentic.config import RuntimeSettings  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    load_dotenv(args.env_file, override=False)
    settings = RuntimeSettings.from_env(require_api_key=False)
    os.environ.setdefault('PAPERPILOT_EMBEDDING_CACHE', str(settings.workspace / 'model-cache/embedding'))
    os.environ.setdefault('PAPERPILOT_RERANKER_CACHE', str(settings.workspace / 'model-cache/reranker'))
    settings = dataclasses.replace(settings, workspace=args.workspace.resolve(),
                                   runtime_db=(args.workspace/'runtime.sqlite3').resolve(),
                                   memory_db=(args.workspace/'knowledge.sqlite3').resolve(),
                                   memory_namespace='qasper-case',
                                   paper_roots=((args.workspace/'corpus').resolve(),))
    runtime = build_runtime(settings)
    snapshot = json.loads((args.workspace/'result.json').read_text(encoding='utf-8'))
    search = next(c for c in snapshot['tool_calls'] if c['tool_name'] == 'paper.search')['args']
    report = {'paid_requests': 0, 'question': snapshot['run']['question'], 'queries': search['queries']}
    original = runtime.search_adapter._diversify

    def paired(items, **kwargs):
        old = original(items, **kwargs, normalize_relevance=False)
        new = original(items, **kwargs, normalize_relevance=True)
        report.update(candidates=items, legacy=old, normalized=new)
        return new

    runtime.search_adapter._diversify = paired
    runtime.search_adapter.search_many(search['queries'], top_k=search['top_k'],
                                       paper_sha256s=set(search.get('paper_sha256s') or []) or None)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'candidates': len(report['candidates']),
                      'old_pages': [x['page'] for x in report['legacy']],
                      'new_pages': [x['page'] for x in report['normalized']]}))


if __name__ == '__main__':
    main()
