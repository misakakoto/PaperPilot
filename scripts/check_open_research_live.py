"""Open research development check; freezes source, never approves candidate execution.

Default is dry-run. --execute uses the configured API and scholarly services with
one cumulative prototype budget. Baseline and license must be the pinned public
SNUNet snapshots. This is not a heldout comparison or a training experiment.
"""
import argparse
from dataclasses import replace
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys


QUESTION = '我要做一个建筑变化检测研究，我有一个找到的开源模型SNUNet_ECAM，你能否调研最近的论文，思考合理的全新模型用于发论文。'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--license', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--configured-retrieval', action='store_true', help='Use configured local dense/retrieval models instead of the historical lexical-only probe')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out/'frozen-source/src'
    shutil.copytree(root/'src', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    freeze = {'src/'+p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob('*.py')}
    save(out/'source-freeze.json', freeze)
    save(out/'request.json', {'question': QUESTION, 'execute': args.execute,
         'candidate_execution_authorized': False, 'training_authorized': False,
         'configured_retrieval': args.configured_retrieval,
         'evaluation': 'open research development case, not a heldout benchmark'})
    if not args.execute:
        print(json.dumps({'output': str(out), 'status': 'NOT_RUN'}))
        return
    sys.path.insert(0, str(source))
    sys.path.insert(1, str(root))
    from paperpilot_agentic.bootstrap import build_runtime
    from paperpilot_agentic.config import RuntimeSettings
    from paperpilot_agentic.models import Budget, RunStatus
    from paperpilot_agentic.prototype_harness import PrototypeHarness
    from paperpilot_agentic.prototype_delivery import export_prototype
    from paperpilot_agentic.reference_methods import PIN, SOURCE_SHA, LICENSE_SHA
    from paperpilot_agentic.research_design import register_code_reference
    from paperpilot_agentic.workbench import ResearchWorkbench
    assert hashlib.sha256(args.baseline.read_bytes()).hexdigest() == SOURCE_SHA
    assert hashlib.sha256(args.license.read_bytes()).hexdigest() == LICENSE_SHA
    settings = RuntimeSettings.from_env(require_api_key=True)
    memory = None
    if settings.memory_db and settings.memory_db.is_file():
        memory = out/'knowledge.sqlite3'
        with sqlite3.connect(settings.memory_db.as_uri()+'?mode=ro', uri=True) as original:
            with sqlite3.connect(memory) as copied:
                original.backup(copied)
    settings = replace(settings, workspace=out/'workspace', runtime_db=out/'runtime.sqlite3',
                       memory_db=memory, fallback_models=(), thinking_enabled=False)
    if not args.configured_retrieval:
        os.environ['PAPERPILOT_ENABLE_DENSE'] = 'false'
        os.environ['PAPERPILOT_ENABLE_RERANKER'] = 'false'
    os.environ['PAPERPILOT_ALLOW_MODEL_DOWNLOAD'] = 'false'
    rt = build_runtime(settings)
    parent = rt.db.create_run('User-provided pinned public SNUNet model', Budget(),
                              {'engine': 'answer_first_text', 'conversation_id': 'open-research-development'})
    rt.db.update_run_status(parent, RunStatus.COMPLETED)
    eid = register_code_reference(rt, parent, source_url=PIN+'models/Models.py',
        content=args.baseline.read_text(encoding='utf-8'), license_url=PIN+'LICENSE',
        license_text=args.license.read_text(encoding='utf-8'))

    class Background:
        def submit(self, rid, callback):
            self.run_id = rid  # Explicit serial advancement below; no hidden worker.

        def active(self, rid):
            return False

    wb = ResearchWorkbench(rt, Background())
    result = wb.submit({'action': 'research_prototype', 'request_id': 'open-research-development-20261024',
        'question': QUESTION, 'parent_run_id': parent, 'conversation_id': 'open-research-development',
        'source_ids': [eid], 'prototype_literature': True})
    rid = result['run_id']
    h = PrototypeHarness(rt)
    save(out/'scope.json', {'run_id': rid, 'model': settings.model, 'source_run_id': parent,
         'budget': rt.db.get_run(rid)['budget'], 'entry': 'ResearchWorkbench.submit',
         'baseline_source_url': PIN+'models/Models.py', 'baseline_sha256': SOURCE_SHA,
         'candidate_execution_authorized': False, 'training_authorized': False})
    print(json.dumps({'run_id': rid, 'output': str(out), 'model': settings.model}), flush=True)
    for _ in range(16):
        if h._load(rid)['phase'] != 'READY':
            break
        h.advance(rid, max_turns=1)
        state = h._load(rid)
        status = rt.status(rid)
        save(out/'state.json', state)
        save(out/'status.json', status)
        print(json.dumps({'phase': state['phase'], 'turn': state['turn'],
            'calls': status['run']['used_llm_calls'], 'tokens': status['run']['used_tokens'],
            'last': state.get('last_observation'), 'stop': state.get('stop_reason')}, ensure_ascii=True), flush=True)
    state = h._load(rid)
    for identity, artifact in state['work']['artifacts'].items():
        if artifact['kind'] != 'implement':
            continue
        folder = out/identity
        folder.mkdir()
        code = artifact['content']['source']
        (folder/'model.py').write_text(code, encoding='utf-8')
        save(folder/'config.json', artifact['content']['config'])
        (folder/'candidate.diff').write_text(''.join(difflib.unified_diff(
            args.baseline.read_text(encoding='utf-8').splitlines(keepends=True),
            code.splitlines(keepends=True), fromfile='pinned-SNUNet_ECAM.py', tofile='candidate-model.py')), encoding='utf-8')
    (out/'prototype-delivery.zip').write_bytes(export_prototype(rt, rid))
    assert not any(c['tool_name'] == 'prototype.validate' and c['attempt'] for c in rt.db.list_tool_calls(rid))
    assert freeze == {'src/'+p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in source.rglob('*.py')}


if __name__ == '__main__':
    main()
