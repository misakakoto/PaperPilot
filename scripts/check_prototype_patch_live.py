"""One-call patch-protocol check; never resumes or executes the source task.

Default is dry-run. --execute tests exact source excerpt copying against the
retained CLDW failure. It does not establish a repaired working model.
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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out/'frozen-source/src'
    shutil.copytree(root/'src', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    freeze = {'src/'+p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob('*.py')}
    save(out/'source-freeze.json', freeze)
    original_bytes = args.source_state.read_bytes()
    save(out/'scope.json', {'component': 'exact source excerpt to literal patch',
        'source_state': str(args.source_state.resolve()),
        'source_state_sha256': hashlib.sha256(original_bytes).hexdigest(),
        'execute': args.execute, 'source_task_resumed': False, 'candidate_execution': False,
        'training': False, 'assurance': 'Patch applicability only; not engineering or research success'})
    if not args.execute:
        return
    sys.path.insert(0, str(source))
    from paperpilot_agentic.config import RuntimeSettings
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    from paperpilot_agentic.durable_reader import DurableResearchModel
    from paperpilot_agentic.models import Budget, RunStatus
    from paperpilot_agentic.prototype_candidate import normalize
    from paperpilot_agentic.prototype_prompt import project_patch_code, instructions
    from paperpilot_agentic.prototype_workspace import digest, check_code
    settings = replace(RuntimeSettings.from_env(require_api_key=True), workspace=out/'workspace',
                       runtime_db=out/'runtime.sqlite3', fallback_models=(), thinking_enabled=False)
    if settings.provider != 'qwen' or settings.base_url != 'https://dashscope.aliyuncs.com/compatible-mode/v1':
        raise ValueError('This component consent is scoped to the configured Qwen endpoint')
    rt = ResearchAgentRuntime(workspace=settings.workspace, db_path=settings.runtime_db,
                              llm=OpenAICompatibleLLM(settings, max_retries=0))
    budget = Budget(max_llm_calls=1, max_tokens=12000, max_seconds=180,
                    synthesis_reserve_calls=0, repair_reserve_calls=0,
                    synthesis_reserve_tokens=0, repair_reserve_tokens=0)
    rid = rt.db.create_run('Component check: source excerpt patch applicability', budget,
                          {'engine': 'prototype_component_check'})
    rt.db.update_run_status(rid, RunStatus.RUNNING)
    state = json.loads(original_bytes)
    before = copy.deepcopy(state)
    key = state['implementation_feedback']['task_id']
    identity = state['work']['heads'][key]
    artifact = state['work']['artifacts'][identity]
    if digest(artifact) != identity:
        raise ValueError('Source candidate identity changed')
    view = project_patch_code(identity, artifact)
    excerpts = [v for v in view['content']['source_excerpts'] if v['symbol'] == 'CLDW.forward']
    if len(excerpts) != 1:
        raise ValueError('Select the retained CLDW failure fixture')
    payload = {'active_task_id': key, 'available_actions': ['work'],
               'candidate_patch': {'base_artifact_id': identity}, 'config': artifact['content']['config'],
               'source_excerpts': excerpts,
               'goal': 'Component only: correct CLDW channel-wise gating and its broadcast to concatenated features. '
                       'Do not change any other method. Symbol labels are metadata, never source.',
               'failure': '64-channel cat features multiplied by 2-channel gate; design requires channel-wise similarity.'}
    messages = [{'role': 'system', 'content': instructions(state, payload)},
                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
    save(out/'request.json', messages)
    try:
        response = DurableResearchModel(rt.llm_runtime, rid, stage='component.patch').complete(
            messages, max_tokens=1800, json_mode=True, thinking_enabled=False)
        save(out/'response.json', {'text': response.text, 'model': response.model,
                                   'finish_reason': response.finish_reason, 'request_id': response.request_id})
        if response.finish_reason != 'stop':
            raise ValueError('Component output truncated')
        action = json.loads(response.text)
        if action['action'] != 'work' or action['arguments']['task_id'] != key:
            raise ValueError('Expected one literal patch work action')
        candidate, record = normalize(state, action['arguments']['content'], task_id=key)
        check_code(candidate['source'])
        assert state == before and args.source_state.read_bytes() == original_bytes
        save(out/'normalized-candidate.json', candidate)
        save(out/'normalization.json', record)
        rt.db.update_run_status(rid, RunStatus.COMPLETED)
        save(out/'result.json', {'component_passed': True, 'run': rt.db.get_run(rid),
            'source_task_unchanged': True, 'candidate_executed': False, 'scientific_validity': 'NOT_VERIFIED'})
        print(json.dumps({'component_passed': True, 'run_id': rid,
                          'used_tokens': rt.db.get_run(rid)['used_tokens'], 'candidate_executed': False}))
    except Exception as error:
        from paperpilot_agentic.provider_diagnostics import safe_error_text
        rt.db.update_run_status(rid, RunStatus.FAILED, error=safe_error_text(error))
        save(out/'result.json', {'component_passed': False, 'run': rt.db.get_run(rid), 'error': safe_error_text(error)})
        raise
    finally:
        assert freeze == {'src/'+p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source.rglob('*.py')}


if __name__ == '__main__':
    main()
