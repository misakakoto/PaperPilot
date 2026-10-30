"""Real Qwen component chain: catalogue -> dimension reads -> source audit -> analysis.

Host seeds the plan and catalogue search. Model chooses reads/audit/analysis.
Four cumulative calls / 40k tokens, no automatic retry, execution or training.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import sys


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = out / 'frozen-source/src'
    shutil.copytree(repo / 'src', frozen, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    sys.path.insert(0, str(frozen))
    from paperpilot_agentic.paper_structure import digest
    freeze = {str(f.relative_to(frozen)): digest(f.read_bytes()) for f in frozen.rglob('*.py')}
    save(out / 'source-freeze.json', freeze)
    save(out / 'scope.json', {'max_calls': 4, 'max_tokens': 40000, 'automatic_retries': 0,
        'host_seeded': ['plan', 'catalogue search'], 'model_controlled': ['evidence read', 'source audit', 'analysis'],
        'full_research_completed': False, 'scientific_correctness': 'REQUIRES_SEPARATE_REVIEW',
        'runner_sha256': digest(Path(__file__).read_bytes()), 'execute': args.execute})
    if not args.execute:
        return
    from paperpilot_agentic.config import RuntimeSettings
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from paperpilot_agentic.prototype_harness import PrototypeHarness
    from paperpilot_agentic.prototype_workspace import plan
    from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    from paperpilot_agentic.models import Budget, RunStatus, EvidenceRecord, EvidenceType
    from paperpilot_agentic.provider_diagnostics import safe_error_text
    from paperpilot_retrieval import FastEmbedder, PaperKnowledgeStore
    from paperpilot_retrieval.reranking import OnnxCrossEncoderReranker
    settings = replace(RuntimeSettings.from_env(require_api_key=True), workspace=out / 'workspace',
                       runtime_db=out / 'runtime.sqlite3', fallback_models=(), thinking_enabled=False)
    if settings.provider != 'qwen' or settings.base_url != 'https://dashscope.aliyuncs.com/compatible-mode/v1':
        raise ValueError('Excerpt consent is scoped to the configured Qwen endpoint')
    shutil.copytree(args.index, settings.workspace / 'paper-evidence-v2')
    embedder = FastEmbedder(cache_dir=Path(os.environ['PAPERPILOT_EMBEDDING_CACHE']), allow_download=False)
    reranker = OnnxCrossEncoderReranker(cache_dir=Path(os.environ['PAPERPILOT_RERANKER_CACHE']), allow_download=False)
    base = PaperPilotKnowledgeSearchAdapter(PaperKnowledgeStore(out / 'empty-base.sqlite3'),
        embedder=embedder, reranker=reranker, rerank_stage='final')
    rt = ResearchAgentRuntime(workspace=settings.workspace, db_path=settings.runtime_db,
        search_adapter=base, llm=OpenAICompatibleLLM(settings, max_retries=0))
    goal = ('比较ChangeFormer与BIT的双时相特征建模机制和适用条件，区分作者方法、相关工作转述和未来假设。'
        '本次只做有来源的研究分析，不实现网络、不训练、不声称新颖性。缺少实验条件时不宣布性能优劣。')
    parent = rt.db.create_run('Input for bounded RAG component check', Budget(), {'engine': 'fixture_input'})
    source_id, _ = rt.db.add_evidence(EvidenceRecord(evidence_id='user_goal', run_id=parent,
        evidence_type=EvidenceType.TEXT, content=goal, source_uri='user://research-goal'))
    rt.db.update_run_status(parent, RunStatus.COMPLETED)
    h = PrototypeHarness(rt)
    rid = h.start(goal, 'paper-evidence-live-v2', parent, [source_id], literature_enabled=True,
        retrieval_profile='paper_evidence_v2', runtime_profile='research-runtime-v3',
        budget=Budget(max_llm_calls=4, max_tokens=40000, max_seconds=900,
            synthesis_reserve_calls=0, repair_reserve_calls=0, synthesis_reserve_tokens=0, repair_reserve_tokens=0).to_dict())['run']['run_id']
    state = h._load(rid)
    rows = [
        ('search', 'Find catalogue versions of ChangeFormer and BIT', 'search_literature', []),
        ('read', 'Use paper_ids of ChangeFormer and BIT from search. Read question+facets [feature mechanism, assumptions], budget_upper=9000. Use paper_evidence_contract, not legacy page-read content.', 'read_literature', ['search']),
        ('audit', 'Audit exactly two source_ids from read with view=full and budget_upper=12000; use paper_evidence_contract source audit.', 'read_literature', ['search', 'read']),
        ('analysis', 'Interpret supported mechanisms with exact quote_ids; gaps and falsifiable directions; literal numbers must occur in own cited quote.', 'analyze_literature', ['read'])]
    plan(state['work'], [{'id': i, 'goal': g, 'capability': c, 'depends_on': d,
        'acceptance': 'Durable source-bound receipt; execution is not scientific validation'} for i, g, c, d in rows], enforce_capabilities=True)
    h._apply(rid, state, {'action': 'work', 'reason': 'Host-seeded catalogue for component check',
        'arguments': {'task_id': 'search', 'content': {'scope': 'local', 'query': 'ChangeFormer BIT Transformer Siamese', 'top_k': 6}}}, 0)
    h._save(rid, state)
    save(out / 'host-seed.json', {'run_id': rid, 'model': settings.model, 'goal': goal})
    try:
        for i in range(1, 5):
            h.advance(rid, max_turns=1)
            state = h._load(rid)
            save(out / f'state-{i}.json', state)
            print(json.dumps({'decision': i, 'phase': state['phase'],
                'tasks': {k: t['status'] for k, t in state['work']['tasks'].items()},
                'calls': rt.db.get_run(rid)['used_llm_calls'], 'tokens': rt.db.get_run(rid)['used_tokens']}), flush=True)
            if state['work']['tasks']['analysis']['status'] == 'DONE' or state['phase'] in {'FAILED', 'UNKNOWN', 'PARTIAL', 'CANCELLED', 'COMPLETED'}:
                break
        def result(k):
            t = state['work']['tasks'][k]
            return state['work']['artifacts'][t['output']]['content'] if t['status'] == 'DONE' else None
        read, audit, analysis = result('read'), result('audit'), result('analysis')
        checks = {'catalogue_used': result('search')['tool'] == 'paper.search_papers',
            'evidence_matrix': bool(read and read.get('tool') == 'paper.retrieve_evidence' and read['output'].get('matrix')),
            'full_locator_read': bool(audit and audit.get('tool') == 'paper.read_sources' and audit.get('evidence') and all(e['provenance'].get('source') for e in audit['evidence'])),
            'source_bound_analysis': bool(analysis and analysis.get('findings'))}
        if state['phase'] not in {'FAILED', 'UNKNOWN', 'PARTIAL', 'CANCELLED', 'COMPLETED'}:
            h._finish(rid, state, 'Bounded paper evidence component check ended', complete=False)
            h._save(rid, state)
            h._publish(rid, state)
        save(out / 'result.json', {'checks': checks, 'component_protocol_passed': all(checks.values()),
            'run': rt.db.get_run(rid), 'scientific_correctness': 'REQUIRES_SEPARATE_REVIEW', 'full_research_completed': False})
    except Exception as exc:
        save(out / 'failure.json', {'error': safe_error_text(exc), 'run': rt.db.get_run(rid)})
        raise
    finally:
        assert freeze == {str(f.relative_to(frozen)): digest(f.read_bytes()) for f in frozen.rglob('*.py')}
        assert all(digest((repo / 'src' / name).read_bytes()) == sha for name, sha in freeze.items())


if __name__ == '__main__':
    main()
