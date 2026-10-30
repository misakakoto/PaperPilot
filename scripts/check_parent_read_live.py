"""Bounded real-model check: seeded local search -> parent read -> locator audit -> analysis.

Four cumulative model decisions / 40,000 tokens; no candidate execution or training.
This is a component chain, not an independently planned research prototype.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = out / 'frozen-source/src'
    shutil.copytree(root / 'src', frozen, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    freeze = {'src/' + p.relative_to(frozen).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in frozen.rglob('*.py')}
    save(out / 'source-freeze.json', freeze)
    pilot = args.pilot.resolve() / 'workspace/token-evidence-pilot'
    original_index = (pilot / 'index.json').read_bytes()
    save(out / 'scope.json', {'protocol': 'prototype-discovered-parent-reading-live-v1',
        'host_seeded_steps': ['task plan', 'local discovery'],
        'model_controlled_stages': ['parent reading', 'full locator audit', 'literal-citation analysis'],
        'max_llm_calls': 4, 'max_tokens': 40000, 'automatic_retries': 0,
        'fallback_models': [], 'candidate_execution': False, 'training': False,
        'source_index_sha256': hashlib.sha256(original_index).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'execute': args.execute, 'source_task_resumed': False})
    if not args.execute:
        return
    sys.path.insert(0, str(frozen))
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
        raise ValueError('Consent is scoped to the configured Qwen endpoint')
    shutil.copytree(pilot, settings.workspace / 'token-evidence-pilot')
    embedder = FastEmbedder(cache_dir=Path(os.environ['PAPERPILOT_EMBEDDING_CACHE']), allow_download=False)
    reranker = OnnxCrossEncoderReranker(cache_dir=Path(os.environ['PAPERPILOT_RERANKER_CACHE']), allow_download=False)
    base = PaperPilotKnowledgeSearchAdapter(PaperKnowledgeStore(out / 'empty-base.sqlite3'),
                                            embedder=embedder, reranker=reranker, rerank_stage='final')
    rt = ResearchAgentRuntime(workspace=settings.workspace, db_path=settings.runtime_db, search_adapter=base,
                              llm=OpenAICompatibleLLM(settings, max_retries=0))
    goal = ('围绕建筑变化检测，分析ChangeMamba原文中的方法机制、成立条件和研究缺口；区分已实现方法、'
            '已报告结果与未来假设。本次仅完成原文阅读、来源定位回查和引用分析；不实现网络、不训练、不声称新颖性。')
    parent = rt.db.create_run('User goal input for bounded local reading check', Budget(), {'engine': 'fixture_input'})
    source_id, _ = rt.db.add_evidence(EvidenceRecord(evidence_id='source_user_goal', run_id=parent,
        evidence_type=EvidenceType.TEXT, content=goal, source_uri='user://research-goal'))
    rt.db.update_run_status(parent, RunStatus.COMPLETED)
    h = PrototypeHarness(rt)
    rid = h.start(goal, 'live-parent-read-component-v1', parent, [source_id], literature_enabled=True,
                  retrieval_profile='token_evidence_pilot', runtime_profile='research-runtime-v3',
                  budget=Budget(max_llm_calls=4, max_tokens=40000, max_seconds=900,
                    synthesis_reserve_calls=0, repair_reserve_calls=0,
                    synthesis_reserve_tokens=0, repair_reserve_tokens=0).to_dict())['run']['run_id']
    state = h._load(rid)
    plan(state['work'], [
        {'id': 'search', 'goal': 'Find ChangeMamba original method passages', 'capability': 'search_literature',
         'depends_on': [], 'acceptance': 'Actual local discovery receipt, not relevance certification'},
        {'id': 'read', 'goal': 'Read complete original parent paragraphs with source_handles for one relevant discovered paper',
         'capability': 'read_literature', 'depends_on': ['search'], 'acceptance': 'Owned discovered child IDs, explicit omissions'},
        {'id': 'audit', 'goal': 'Audit FULL source locations for the exact paper and child_ids in the completed read, using view=full',
         'capability': 'read_literature', 'depends_on': ['search', 'read'], 'acceptance': 'Original PDF source segments read back'},
        {'id': 'analysis', 'goal': 'Interpret supported mechanisms and limits, with exact source quote IDs and explicit gaps',
         'capability': 'analyze_literature', 'depends_on': ['read'], 'acceptance': 'Source-bound findings, falsifiable hypotheses, no novelty claims'}],
        enforce_capabilities=True)
    search = {'scope': 'local', 'query': 'ChangeMamba cross temporal spatial state space feature fusion', 'top_k': 6}
    h._apply(rid, state, {'action': 'work', 'reason': 'Host-seeded frozen local discovery for component test',
                        'arguments': {'task_id': 'search', 'content': search}}, 0)
    h._save(rid, state)
    save(out / 'host-seed.json', {'goal': goal, 'search': search, 'model': settings.model, 'run_id': rid})
    try:
        for decision in range(1, 5):
            h.advance(rid, max_turns=1)
            state = h._load(rid)
            save(out / f'state-{decision}.json', state)
            print(json.dumps({'decision': decision, 'phase': state['phase'],
                'tasks': {k: t['status'] for k, t in state['work']['tasks'].items()},
                'used_calls': rt.db.get_run(rid)['used_llm_calls'], 'used_tokens': rt.db.get_run(rid)['used_tokens']}), flush=True)
            if state['work']['tasks']['analysis']['status'] == 'DONE' or state['phase'] in {'FAILED', 'UNKNOWN', 'PARTIAL', 'CANCELLED', 'COMPLETED'}:
                break
        artifacts = state['work']['artifacts']
        def result(key):
            return artifacts[state['work']['tasks'][key]['output']]['content'] if state['work']['tasks'][key]['status'] == 'DONE' else None
        read, audit, analysis = result('read'), result('audit'), result('analysis')
        checks = {'parent_read_used': bool(read and read.get('tool') == 'paper.read_context'
                    and read.get('output', {}).get('view_policy') == 'source-handles-v1'),
                  'full_locator_audit': bool(audit and audit.get('evidence') and all(
                      e.get('provenance', {}).get('source_segments') for e in audit['evidence'])),
                  'source_bound_analysis': bool(analysis and analysis.get('findings'))}
        if state['phase'] not in {'FAILED', 'UNKNOWN', 'PARTIAL', 'CANCELLED', 'COMPLETED'}:
            h._finish(rid, state, 'Bounded component check ended; no design, implementation or engineering result claimed', complete=False)
            h._save(rid, state)
            h._publish(rid, state)  # Delivery completed; prototype phase stays PARTIAL.
        save(out / 'result.json', {'checks': checks, 'component_protocol_passed': all(checks.values()),
            'run': rt.db.get_run(rid), 'scientific_correctness': 'REQUIRES_SEPARATE_REVIEW',
            'autonomous_planning': False, 'full_research_task_completed': False})
        print(json.dumps({'checks': checks, 'component_protocol_passed': all(checks.values())}), flush=True)
    except Exception as exc:
        save(out / 'failure.json', {'error': safe_error_text(exc), 'run': rt.db.get_run(rid)})
        raise
    finally:
        assert (pilot / 'index.json').read_bytes() == original_index
        assert freeze == {'src/' + p.relative_to(frozen).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in frozen.rglob('*.py')}
        assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == sha for name, sha in freeze.items())


if __name__ == '__main__':
    main()
