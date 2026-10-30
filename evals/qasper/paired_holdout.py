"""Immutable paired workbench evaluation of a train-derived internal holdout.

The same runner imports either frozen source tree. No gold evidence or answers
enter the agent. Production prose is scored unchanged (plus citation removal),
not converted into a favorable short answer by a second model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(root):
    h = hashlib.sha256()
    for name in ('paperpilot_agentic', 'paperpilot_retrieval'):
        for path in sorted((root / 'src' / name).rglob('*.py')):
            h.update(path.relative_to(root).as_posix().encode())
            h.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def retrieval_configuration(cache, *, embedding_cache=None, reranker_cache=None, reranker_revision='main'):
    return {
        'PAPERPILOT_ENABLE_DENSE': 'true', 'PAPERPILOT_ENABLE_RERANKER': 'true',
        'PAPERPILOT_EMBEDDING_MODEL': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'PAPERPILOT_RERANKER_MODEL': 'cross-encoder/ms-marco-MiniLM-L6-v2',
        'PAPERPILOT_RERANKER_FILE': 'onnx/model_quint8_avx2.onnx',
        'PAPERPILOT_RERANKER_REVISION': reranker_revision, 'PAPERPILOT_RERANK_STAGE': 'final',
        'PAPERPILOT_EMBEDDING_CACHE': str((embedding_cache or cache/'embedding').resolve()),
        'PAPERPILOT_RERANKER_CACHE': str((reranker_cache or cache/'reranker').resolve()),
        'PAPERPILOT_ALLOW_MODEL_DOWNLOAD': 'false',
        'PAPERPILOT_ALLOW_EXTERNAL_SCHOLARLY': 'false',
    }


def frozen_settings(protocol, folder):
    """Never discover a checkout .env; both arms use the same explicit protocol."""
    from paperpilot_agentic.config import RuntimeSettings, resolve_provider_api_key
    os.environ['PYTHON_DOTENV_DISABLED'] = '1'
    os.environ.update(protocol['retrieval_config'])
    return RuntimeSettings(
        provider=protocol['provider'], api_key=resolve_provider_api_key(protocol['provider']),
        base_url=('https://api.deepseek.com' if protocol['provider']=='deepseek'
                  else 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
        model=protocol['model'], thinking_enabled=False, temperature=protocol['temperature'],
        max_tokens=protocol['max_output_tokens'], timeout_seconds=protocol['timeout_seconds'],
        workspace=folder, runtime_db=folder/'runtime.sqlite3', memory_db=folder/'knowledge.sqlite3',
        memory_namespace='qasper-holdout', paper_roots=(folder,))


def retrieval_identity(runtime):
    adapter = runtime.search_adapter
    if adapter.embedder is None or adapter.reranker is None:
        raise ValueError('Frozen dense or reranker unavailable; no provider request allowed')
    adapter.reranker.warmup()
    value = {'embedding_identity': adapter.embedder.cache_identity,
             'reranker_sha256': adapter.reranker.model_sha256,
             'reranker_tokenizer_sha256': adapter.reranker.tokenizer_sha256}
    if not all(value.values()):
        raise ValueError('Retrieval identity incomplete; no provider request allowed')
    return value


def prepare_runtime(protocol, code_root, folder):
    sys.path[:0] = [str(code_root/'src'), str(code_root)]
    from paperpilot_agentic.bootstrap import build_runtime
    from paperpilot_retrieval import PaperKnowledgeStore
    settings = frozen_settings(protocol, folder)
    if not settings.api_key:
        raise ValueError('Provider key unavailable; no requests dispatched')
    store = PaperKnowledgeStore(settings.memory_db, namespace=settings.memory_namespace)
    return store, build_runtime(settings)


def freeze(args):
    selection = json.loads(args.selection.read_bytes())
    if selection['status'] != 'FROZEN' or len(selection['cases']) != 40:
        raise ValueError('The full independent 40-case selection must pass eligibility')
    if digest(args.dataset) != selection['dataset_sha256']:
        raise ValueError('Selection dataset changed')
    save(args.protocol, {
        'kind': 'qasper-train-internal-holdout-paired-v1', 'official_test_set': False,
        'selection_sha256': digest(args.selection), 'dataset_sha256': digest(args.dataset),
        'cases': selection['cases'], 'baseline_revision': '9a9e6b8',
        'roots': {k: {'path': str(root.resolve()), 'sha256': source_digest(root)}
                  for k, root in [('baseline', args.baseline), ('candidate', args.candidate)]},
        'runner_sha256': digest(Path(__file__)), 'provider': args.provider, 'model': args.model,
        'temperature': 0.2, 'thinking_enabled': False, 'timeout_seconds': 90,
        'action': 'auto', 'visual_policy': 'off', 'global_concurrency': 4,
        'max_output_tokens': 8000,
        'retrieval_config': retrieval_configuration(args.candidate/'.paperpilot/model-cache',
            embedding_cache=args.embedding_cache, reranker_cache=args.reranker_cache,
            reranker_revision=args.reranker_revision),
        'configuration_policy': 'Explicit settings, dotenv disabled, identical non-null local model fingerprints checked before all provider requests',
        'limits_per_case': {'max_llm_calls': 20, 'max_tokens': 100000, 'max_tool_calls': 20, 'max_seconds': 3600},
        'retrieval': 'Existing production dense + reranking; frozen local model identities recorded per case',
        'question_policy': 'Original unchanged question, correct paper supplied, public text/captions only; no gold supplied',
        'score_policy': 'All 40 per arm, failures and missing outputs zero. Official token/paragraph F1 functions; raw prose and citation-stripped prose separately. No answer rewriting for scoring.',
        'evidence_policy': 'All complete official text paragraphs contained in cited original records, not gold-selected evidence; visual-only evidence is unsupported',
        'cost_policy': 'Provider attempt counters including failures/retries, actual token usage, end-to-end elapsed time; currency unavailable without verified pricing',
        'adjustment_policy': 'No tuning against this sample; implementation changes invalidate candidate fingerprint',
    })


def run_case(args, protocol):
    code_root = Path(protocol['roots'][args.arm]['path'])
    if source_digest(code_root) != protocol['roots'][args.arm]['sha256']:
        raise ValueError('Frozen source changed')
    # Resolve BOTH application code and the public-paper adapter from this arm.
    sys.path[:0] = [str(code_root / 'src'), str(code_root)]
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from paperpilot_agentic.web_api import create_fastapi_router
    from evals.qasper.agent_case import paper_records
    case = protocol['cases'][args.case_index]
    folder = args.workspace / args.arm / f'case-{args.case_index:02}'
    folder.mkdir(parents=True, exist_ok=False)  # no blind resumption of an uncertain paid run
    started, runtime, rid = time.monotonic(), None, None
    try:
        data = json.loads(args.dataset.read_bytes())
        public, doc, chunks = paper_records(data[case['paper_id']], source_id=case['paper_id'])
        save(folder / 'public-paper.json', public)
        store, runtime = prepare_runtime(protocol, code_root, folder)
        identity = retrieval_identity(runtime)
        expected = json.loads((args.workspace/'preflight.json').read_bytes())
        if expected['protocol_sha256'] != digest(args.protocol) or identity != expected['identity']:
            raise ValueError('Case retrieval identity differs from paired preflight; no provider request allowed')
        store.sync_documents([doc], chunks, embedder=runtime.search_adapter.embedder)
        app = FastAPI()
        app.include_router(create_fastapi_router(runtime_factory=lambda: runtime, prefix='/api/agentic'))
        with TestClient(app) as client:
            response = client.post('/api/agentic/turns', json={'request_id': f'holdout-{args.case_index:02}',
                'action': protocol['action'], 'question': case['question'], 'paper_ids': [doc.sha256], 'visual_policy': 'off'})
            response.raise_for_status()
            rid = response.json()['run_id']
            while True:
                result = runtime.status(rid, include_trace=False)
                view = client.get(f'/api/agentic/runs/{rid}/view').json()
                if not view['worker_active'] and result['run']['status'] in {
                        'COMPLETED', 'FAILED', 'CANCELLED', 'BUDGET_EXHAUSTED', 'WAITING_APPROVAL'}:
                    break
                if time.monotonic() - started > 3660:
                    runtime.cancel(rid, reason='Frozen evaluation observation timeout')
                    raise TimeoutError('Do not restart unknown requests')
                time.sleep(0.5)
        save(folder/'result.json', result)
        save(folder/'view.json', view)
        failures = ' '.join(str(c.get('error') or '') for c in result['llm_calls'])
        if any(marker.lower() in failures.lower() for marker in (
                'Error code: 402', 'Error code: 401', 'Error code: 403', 'Insufficient Balance',
                'insufficient_quota', 'AllocationQuota', 'Arrearage')):
            try:
                save(args.workspace/'provider-stop.json', {'reason':'Provider billing or authorization failure',
                    'arm':args.arm, 'case':args.case_index, 'no_automatic_model_switch':True})
            except FileExistsError:
                pass
        save(folder/'case.json', {**case, 'elapsed_seconds': time.monotonic()-started, 'gold_sent': False,
            'actual_models': sorted({r.get('model') for r in result['llm_calls'] if r.get('model')}),
            'source_sha256': source_digest(code_root),
            **identity})
        print(json.dumps({'arm': args.arm, 'case': args.case_index, 'status': result['run']['status'],
            'calls': result['run']['used_llm_calls'], 'tokens': result['run']['used_tokens']}), flush=True)
    except Exception as exc:
        if runtime is not None and rid is not None and not (folder/'result.json').exists():
            save(folder/'result.json', runtime.status(rid, include_trace=False))
        save(folder/'failure.json', {'type': type(exc).__name__, 'elapsed_seconds': time.monotonic()-started, 'run_id': rid})
        print(json.dumps({'arm': args.arm, 'case': args.case_index, 'error': type(exc).__name__}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['freeze', 'run', 'case', 'preflight'])
    for flag in ['dataset', 'protocol', 'selection', 'baseline', 'candidate', 'workspace']:
        p.add_argument('--'+flag, type=Path, required=flag in {'dataset', 'protocol'})
    p.add_argument('--arm', choices=['baseline', 'candidate'])
    p.add_argument('--case-index', type=int)
    p.add_argument('--provider', choices=['deepseek','qwen'], default='deepseek')
    p.add_argument('--model', default='deepseek-chat')
    p.add_argument('--embedding-cache', type=Path)
    p.add_argument('--reranker-cache', type=Path)
    p.add_argument('--reranker-revision', default='main')
    args = p.parse_args()
    if args.phase == 'freeze':
        freeze(args)
        return
    protocol = json.loads(args.protocol.read_bytes())
    if digest(Path(__file__)) != protocol['runner_sha256'] or digest(args.dataset) != protocol['dataset_sha256']:
        raise ValueError('Frozen runner or dataset changed')
    if args.phase == 'case':
        run_case(args, protocol)
        return
    if args.phase == 'preflight':
        code_root = Path(protocol['roots'][args.arm]['path'])
        if source_digest(code_root) != protocol['roots'][args.arm]['sha256']:
            raise ValueError('Frozen source changed')
        folder = args.workspace/('preflight-'+args.arm)
        folder.mkdir(parents=True, exist_ok=False)
        _, runtime = prepare_runtime(protocol, code_root, folder)
        identity = retrieval_identity(runtime)
        save(folder/'identity.json', identity)
        print(json.dumps({'arm':args.arm, 'provider_requests':0, **identity}), flush=True)
        return
    from concurrent.futures import ThreadPoolExecutor
    args.workspace.mkdir(parents=True, exist_ok=False)
    save(args.workspace/'protocol.json', protocol)
    identities = []
    for arm in ('baseline', 'candidate'):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), 'preflight',
            '--dataset', str(args.dataset.resolve()), '--protocol', str(args.protocol.resolve()),
            '--workspace', str(args.workspace.resolve()), '--arm', arm], check=True)
        identities.append(json.loads((args.workspace/('preflight-'+arm)/'identity.json').read_bytes()))
    if identities[0] != identities[1]:
        raise ValueError('Paired retrieval identity mismatch; no provider requests dispatched')
    save(args.workspace/'preflight.json', {'protocol_sha256':digest(args.protocol),
        'identity':identities[0], 'provider_requests':0})
    # Interleave arms deterministically; isolate each case in a process. Four
    # active cases total, not four workers touching the same research task.
    jobs = [(arm, i) for i in range(len(protocol['cases'])) for arm in ('baseline', 'candidate')]
    def launch(job):
        if (args.workspace/'provider-stop.json').exists():
            return 2
        arm, index = job
        return subprocess.run([sys.executable, str(Path(__file__).resolve()), 'case',
            '--dataset', str(args.dataset.resolve()), '--protocol', str(args.protocol.resolve()),
            '--workspace', str(args.workspace.resolve()), '--arm', arm, '--case-index', str(index)], check=False).returncode
    with ThreadPoolExecutor(max_workers=4) as pool:
        codes = list(pool.map(launch, jobs))
    print(json.dumps({'finished': len(codes), 'worker_failures': sum(c != 0 for c in codes)}), flush=True)


if __name__ == '__main__':
    main()
