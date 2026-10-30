"""Recover the SAME known-undispatched component run; never reset its budget."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    frozen = out / 'frozen-source/src'
    shutil.copytree(root / 'src', frozen, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    freeze = {'src/' + p.relative_to(frozen).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in frozen.rglob('*.py')}
    save(out / 'source-freeze.json', freeze)
    rid = json.loads((source / 'host-seed.json').read_bytes())['run_id']
    save(out / 'scope.json', {'run_id': rid, 'original_directory': str(source), 'new_task_created': False,
        'max_total_calls': 4, 'max_total_tokens': 40000, 'budget_reset': False,
        'old_result_retained': True, 'training': False, 'candidate_execution': False,
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    sys.path.insert(0, str(frozen))
    from paperpilot_agentic.config import RuntimeSettings
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from paperpilot_agentic.prototype_harness import PrototypeHarness
    from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    from paperpilot_agentic.provider_diagnostics import safe_error_text
    from paperpilot_retrieval import FastEmbedder, PaperKnowledgeStore
    from paperpilot_retrieval.reranking import OnnxCrossEncoderReranker
    settings = replace(RuntimeSettings.from_env(require_api_key=True), workspace=source / 'workspace',
                       runtime_db=source / 'runtime.sqlite3', fallback_models=(), thinking_enabled=False)
    if settings.provider != 'qwen' or settings.base_url != 'https://dashscope.aliyuncs.com/compatible-mode/v1':
        raise ValueError('Consent is scoped to the configured Qwen endpoint')
    with sqlite3.connect(settings.runtime_db.as_uri() + '?mode=ro', uri=True) as original:
        with sqlite3.connect(out / 'before.sqlite3') as backup:
            original.backup(backup)
    embedder = FastEmbedder(cache_dir=Path(os.environ['PAPERPILOT_EMBEDDING_CACHE']), allow_download=False)
    reranker = OnnxCrossEncoderReranker(cache_dir=Path(os.environ['PAPERPILOT_RERANKER_CACHE']), allow_download=False)
    base = PaperPilotKnowledgeSearchAdapter(PaperKnowledgeStore(source / 'empty-base.sqlite3'),
                                            embedder=embedder, reranker=reranker, rerank_stage='final')
    rt = ResearchAgentRuntime(workspace=settings.workspace, db_path=settings.runtime_db, search_adapter=base,
                              llm=OpenAICompatibleLLM(settings, max_retries=0))
    h = PrototypeHarness(rt)
    before = rt.db.get_run(rid)
    save(out / 'before.json', {'run': before, 'state': h._load(rid)})
    try:
        h.recover_undispatched_context_stop(rid)
        recovered = rt.db.get_run(rid)
        for key in ('budget', 'created_at', 'used_tokens', 'used_llm_calls'):
            assert recovered[key] == before[key]
        state = h._load(rid)
        save(out / 'recovered.json', {'run': recovered, 'state': state})
        print(json.dumps({'same_run_id': rid, 'recovered_without_dispatch': True,
            'input_upper_bound': state['context_manifest']['input_upper_bound'],
            'remaining_tokens': state['context_manifest']['remaining_tokens'],
            'used_calls_before': before['used_llm_calls']}), flush=True)
        while rt.db.get_run(rid)['used_llm_calls'] < 4:
            h.advance(rid, max_turns=1)
            state = h._load(rid)
            save(out / f'state-{state["turn"]}.json', state)
            print(json.dumps({'turn': state['turn'], 'phase': state['phase'],
                'tasks': {k: t['status'] for k, t in state['work']['tasks'].items()},
                'used_calls': rt.db.get_run(rid)['used_llm_calls'], 'used_tokens': rt.db.get_run(rid)['used_tokens']}), flush=True)
            if state['work']['tasks']['analysis']['status'] == 'DONE' or state['phase'] in {'FAILED', 'UNKNOWN', 'PARTIAL', 'CANCELLED', 'COMPLETED'}:
                break
        def result(key):
            return state['work']['artifacts'][state['work']['tasks'][key]['output']]['content'] if state['work']['tasks'][key]['status'] == 'DONE' else None
        read, audit, analysis = result('read'), result('audit'), result('analysis')
        matching = bool(read and audit and all(read['local_read_request'][k] == audit['local_read_request'][k]
                                              for k in ('paper_sha256', 'child_ids')))
        checks = {'parent_read_used': bool(read and read.get('output', {}).get('view_policy') == 'source-handles-v1'),
            'full_locator_audit_same_inputs': bool(matching and audit.get('evidence') and all(
                e.get('provenance', {}).get('source_segments') for e in audit['evidence'])),
            'source_bound_analysis': bool(analysis and analysis.get('findings'))}
        if state['phase'] not in {'FAILED', 'UNKNOWN', 'PARTIAL', 'CANCELLED', 'COMPLETED'}:
            h._finish(rid, state, 'Component chain completed; no full research prototype or experimental result claimed', complete=False)
            h._save(rid, state)
            h._publish(rid, state)  # Delivery completed; prototype phase stays PARTIAL.
        after = rt.db.get_run(rid)
        assert after['used_llm_calls'] <= 4 and after['used_tokens'] <= 40000
        save(out / 'result.json', {'checks': checks, 'component_protocol_passed': all(checks.values()),
            'run_id_unchanged': rid == before['run_id'], 'budget_unchanged': after['budget'] == before['budget'],
            'run': after, 'scientific_correctness': 'REQUIRES_SEPARATE_REVIEW',
            'autonomous_planning': False, 'full_research_task_completed': False})
        print(json.dumps({'checks': checks, 'component_protocol_passed': all(checks.values())}), flush=True)
    except Exception as exc:
        save(out / 'failure.json', {'error': safe_error_text(exc), 'run': rt.db.get_run(rid)})
        raise
    finally:
        assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == sha for name, sha in freeze.items())


if __name__ == '__main__':
    main()
