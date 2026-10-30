"""Resume a proven pre-dispatch context stop without resetting its task or budget."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from paperpilot_agentic.config import RuntimeSettings
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.prototype_harness import PrototypeHarness
from paperpilot_agentic.prototype_context_recovery import recover
from paperpilot_agentic.paper_structure import save_json, digest
from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
from paperpilot_agentic.providers import OpenAICompatibleLLM
from paperpilot_retrieval import FastEmbedder, PaperKnowledgeStore
from paperpilot_retrieval.reranking import OnnxCrossEncoderReranker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.directory.resolve()
    rid = json.loads((root / 'host-seed.json').read_bytes())['run_id']
    settings = replace(RuntimeSettings.from_env(require_api_key=True), workspace=root / 'workspace',
        runtime_db=root / 'runtime.sqlite3', fallback_models=(), thinking_enabled=False)
    if settings.provider != 'qwen' or settings.base_url != 'https://dashscope.aliyuncs.com/compatible-mode/v1':
        raise ValueError('Wrong excerpt endpoint')
    base = PaperPilotKnowledgeSearchAdapter(PaperKnowledgeStore(root / 'empty-base.sqlite3'),
        embedder=FastEmbedder(cache_dir=Path(os.environ['PAPERPILOT_EMBEDDING_CACHE']), allow_download=False),
        reranker=OnnxCrossEncoderReranker(cache_dir=Path(os.environ['PAPERPILOT_RERANKER_CACHE']), allow_download=False), rerank_stage='final')
    rt = ResearchAgentRuntime(workspace=settings.workspace, db_path=settings.runtime_db, search_adapter=base,
        llm=OpenAICompatibleLLM(settings, max_retries=0))
    h = PrototypeHarness(rt)
    before = rt.db.get_run(rid)
    freeze = {str(f): digest(f.read_bytes()) for f in Path('src').rglob('*.py')}
    save_json(args.output / 'before.json', before)
    save_json(args.output / 'source-freeze.json', freeze)
    if not args.execute:
        return
    recover(h, rid)
    save_json(args.output / 'recovered.json', h._load(rid))
    # Remaining allowance is derived from the original run, never a new Budget.
    for i in range(before['budget']['max_llm_calls']-before['used_llm_calls']):
        h.advance(rid, max_turns=1)
        state = h._load(rid)
        save_json(args.output / f'state-{i+1}.json', state)
        run = rt.db.get_run(rid)
        print(json.dumps({'calls': run['used_llm_calls'], 'tokens': run['used_tokens'], 'phase': state['phase'],
            'tasks': {k:t['status'] for k,t in state['work']['tasks'].items()}}), flush=True)
        if state['phase'] in {'FAILED', 'UNKNOWN', 'PARTIAL', 'CANCELLED', 'COMPLETED'}:
            break
    state = h._load(rid)
    if state['phase'] not in {'FAILED', 'UNKNOWN', 'PARTIAL', 'CANCELLED', 'COMPLETED'}:
        h._finish(rid, state, 'Original bounded component allowance ended; unfinished tasks retained', complete=False)
        h._save(rid, state)
        h._publish(rid, state)
    after = rt.db.get_run(rid)
    assert after['budget'] == before['budget'] and after['created_at'] == before['created_at']
    assert after['used_tokens'] >= before['used_tokens'] and after['used_llm_calls'] <= before['budget']['max_llm_calls']
    assert freeze == {str(f): digest(f.read_bytes()) for f in Path('src').rglob('*.py')}
    save_json(args.output / 'result.json', {'before': before, 'after': after,
        'budget_reset': False, 'state': state, 'scientific_correctness': 'REQUIRES_SEPARATE_REVIEW'})


if __name__ == '__main__':
    main()
