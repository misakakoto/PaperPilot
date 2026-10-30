"""One-call review of a retained real analysis, with no source-task mutation.

This regression expects the known FEA-Net future-work attribution error to be
flagged. It does not certify scientific validity or resume the original task.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-state', type=Path, required=True)
    parser.add_argument('--analysis-task', default='analyze_literature')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--with-local-pages', action='store_true', help='Read cited local pages through the registered tool')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out/'frozen-source/src'
    shutil.copytree(root/'src', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    freeze = {'src/'+p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob('*.py')}
    original = args.source_state.read_bytes()
    save(out/'source-freeze.json', freeze)
    save(out/'scope.json', {'component': 'retained real literature analysis review',
        'source_state': str(args.source_state.resolve()), 'source_sha256': hashlib.sha256(original).hexdigest(),
        'source_task_resumed': False, 'candidate_execution': False, 'execute': args.execute,
        'expected_error': 'First FEA-Net finding attributes future work as an implemented method'})
    if not args.execute:
        return
    sys.path.insert(0, str(source))
    from paperpilot_agentic.config import RuntimeSettings
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    from paperpilot_agentic.durable_reader import DurableResearchModel
    from paperpilot_agentic.models import Budget, RunStatus
    from paperpilot_agentic import prototype_literature_review as review
    from paperpilot_agentic.prototype_prompt import instructions
    from paperpilot_agentic.prototype_workspace import plan, task_rows, complete
    from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
    from paperpilot_retrieval import PaperKnowledgeStore
    settings = replace(RuntimeSettings.from_env(require_api_key=True), workspace=out/'workspace',
                       runtime_db=out/'runtime.sqlite3', fallback_models=(), thinking_enabled=False)
    if settings.provider != 'qwen' or settings.base_url != 'https://dashscope.aliyuncs.com/compatible-mode/v1':
        raise ValueError('Consent is scoped to the configured Qwen endpoint')
    adapter = (PaperPilotKnowledgeSearchAdapter(
        PaperKnowledgeStore(settings.memory_db, namespace=settings.memory_namespace), paper_roots=settings.paper_roots)
        if args.with_local_pages else None)
    rt = ResearchAgentRuntime(workspace=settings.workspace, db_path=settings.runtime_db,
                              llm=OpenAICompatibleLLM(settings, max_retries=0), search_adapter=adapter)
    rid = rt.db.create_run('Component regression: real FEA-Net analysis critique',
        Budget(max_llm_calls=1, max_tokens=30000, max_seconds=240, synthesis_reserve_calls=0,
               repair_reserve_calls=0, synthesis_reserve_tokens=0, repair_reserve_tokens=0),
        {'engine': 'prototype_component_check'})
    rt.db.update_run_status(rid, RunStatus.RUNNING)
    state = copy.deepcopy(json.loads(original))
    state['literature_review_policy'] = review.POLICY
    key = 'source_claim_review'
    rows = task_rows(state['work'])
    page_tasks = []
    if args.with_local_pages:
        from paperpilot_agentic.prototype_analysis import _original_span
        cited = state['work']['artifacts'][state['work']['tasks'][args.analysis_task]['output']]['content']['findings']
        by_sha = {}
        for finding in cited:
            span = _original_span(state['work'], finding)
            if span and str(span['source_uri']).startswith('paper://'):
                by_sha.setdefault(span['source_uri'].split('/')[2], set()).add(span['page'])
        for sha, pages in by_sha.items():
            if len(pages) > 3:
                raise ValueError('This bounded probe supports at most three pages per cited paper')
            dep = next(t['id'] for t in state['work']['tasks'].values()
                       if t['capability'] == 'search_literature' and t['status'] == 'DONE'
                       and any(e.get('provenance', {}).get('paper_sha256') == sha for e in
                               state['work']['artifacts'][t['output']]['content'].get('evidence', [])))
            name = 'context_pages_'+str(len(page_tasks))
            rows.append({'id': name, 'goal': 'Read cited registered PDF pages', 'capability': 'read_literature',
                         'depends_on': [dep], 'acceptance': 'SHA-verified bounded pages'})
            page_tasks.append((name, {'paper_sha256': sha, 'pages': sorted(pages)}))
    rows.append({'id': key, 'goal': 'Critique source interpretation and actual coverage',
                 'capability': review.CAPABILITY, 'depends_on': [args.analysis_task]+[n for n, _ in page_tasks],
                 'acceptance': 'Exact original finding review; no scientific certification'})
    plan(state['work'], rows)
    for name, request in page_tasks:
        call, _ = rt.db.create_step_tool_call(run_id=rid, tool_name='paper.read_pages', args=request,
            risk_level='READ_ONLY', idempotency_key='claim-context:'+rid+':'+name, max_attempts=1)
        outcome = rt.tools.execute(call)
        if outcome.result is None or not outcome.result.ok:
            raise ValueError('Page tool did not return successful known evidence')
        result = outcome.result.to_dict()
        result.update(tool_call_id=call, tool='paper.read_pages', local_read_request=request)
        complete(state['work'], name, 'read_literature', result)
        save(out/(name+'.json'), result)
    packet = review.packet(state['work'], state['work']['tasks'][key])
    payload = {'goal': state['work']['goal'], 'active_task_id': key, 'available_actions': ['work'],
               'literature_review_packet': packet}
    messages = [{'role': 'system', 'content': instructions(state, payload)},
                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
    save(out/'request.json', messages)
    try:
        response = DurableResearchModel(rt.llm_runtime, rid, stage='component.claim_review').complete(
            messages, max_tokens=2500, json_mode=True, thinking_enabled=False)
        save(out/'response.json', {'text': response.text, 'model': response.model,
                                   'finish_reason': response.finish_reason, 'request_id': response.request_id})
        if response.finish_reason != 'stop':
            raise ValueError('Review output truncated')
        action = json.loads(response.text)
        if action['action'] != 'work' or action['arguments']['task_id'] != key:
            raise ValueError('Expected a source review work action')
        result = review.resolve(state['work'], state['work']['tasks'][key], action['arguments']['content'])
        identity = complete(state['work'], key, review.CAPABILITY, result)
        save(out/'review.json', {'artifact_id': identity, 'content': result})
        caught = any(c['finding_index'] == 0 and c['interpretation_supported'] is False for c in result['checks'])
        assert args.source_state.read_bytes() == original
        rt.db.update_run_status(rid, RunStatus.COMPLETED if caught else RunStatus.FAILED)
        save(out/'result.json', {'protocol_valid': True, 'known_error_flagged': caught,
            'source_task_unchanged': True, 'run': rt.db.get_run(rid), 'scientific_validity': 'NOT_VERIFIED'})
        print(json.dumps({'protocol_valid': True, 'known_error_flagged': caught,
                          'review_passed': result['passed'], 'used_tokens': rt.db.get_run(rid)['used_tokens']}))
        if not caught:
            raise ValueError('Known source interpretation error was not flagged')
    except Exception as error:
        from paperpilot_agentic.provider_diagnostics import safe_error_text
        rt.db.update_run_status(rid, RunStatus.FAILED, error=safe_error_text(error))
        save(out/'failure.json', {'error': safe_error_text(error), 'run': rt.db.get_run(rid)})
        raise
    finally:
        assert freeze == {'src/'+p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source.rglob('*.py')}


if __name__ == '__main__':
    main()
