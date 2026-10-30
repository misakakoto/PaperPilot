"""Offline real-corpus tool/analysis handoff and recovery check; scripted, no LLM."""
import argparse
import copy
import dataclasses
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from paperpilot_agentic.bootstrap import build_runtime
from paperpilot_agentic.config import RuntimeSettings
from paperpilot_agentic.models import Budget, RunStatus
from paperpilot_agentic.prototype_analysis import spans
from paperpilot_agentic.prototype_harness import PrototypeHarness
from paperpilot_agentic.prototype_workspace import digest, task_rows
from paperpilot_agentic.reference_methods import PIN, SOURCE_SHA, LICENSE_SHA
from paperpilot_agentic.research_design import register_code_reference


class OfflineOnly:
    model = 'offline-script-no-generative-model'

    def complete(self, *args, **kwargs):
        raise AssertionError('Offline integration must never invoke a language model')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--license', required=True, type=Path)
    args = parser.parse_args()
    assert hashlib.sha256(args.baseline.read_bytes()).hexdigest() == SOURCE_SHA
    assert hashlib.sha256(args.license.read_bytes()).hexdigest() == LICENSE_SHA
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    settings = RuntimeSettings.from_env()
    with sqlite3.connect(settings.memory_db.as_uri()+'?mode=ro', uri=True) as original:
        with sqlite3.connect(out/'knowledge.sqlite3') as copied:
            original.backup(copied)
    settings = dataclasses.replace(settings, workspace=out, runtime_db=out/'runtime.sqlite3',
                                   memory_db=out/'knowledge.sqlite3')
    runtime = build_runtime(settings, llm_factory=lambda _: OfflineOnly(), vision_factory=lambda **_: object())
    parent = runtime.db.create_run('Offline pinned baseline import', Budget(), {'engine': 'offline_probe'})
    runtime.db.update_run_status(parent, RunStatus.COMPLETED)
    source_id = register_code_reference(runtime, parent, source_url=PIN+'models/Models.py',
        content=args.baseline.read_text(encoding='utf-8'), license_url=PIN+'LICENSE',
        license_text=args.license.read_text(encoding='utf-8'))
    harness = PrototypeHarness(runtime)
    rid = harness.start('离线检查本地文献到研究设计的工具和引用交接，不验证科研内容',
        'offline-local-handoff-v1', parent, [source_id], literature_enabled=True)['run']['run_id']
    state = harness._load(rid)

    def action(action, arguments):
        harness._apply(rid, state, {'action': action, 'reason': 'scripted offline integration check',
                                   'arguments': arguments}, 1)

    rows = [{'id': key, 'capability': kind, 'depends_on': deps, 'goal': key, 'acceptance': 'offline source binding'}
            for key, kind, deps in [('baseline', 'read', []), ('discover', 'search_literature', []),
                ('details', 'read_literature', ['discover']), ('analysis', 'analyze_literature', ['details']),
                ('design', 'design', ['baseline', 'analysis'])]]
    action('plan', {'tasks': rows})
    action('work', {'task_id': 'baseline', 'content': source_id})
    harness._save(rid, state)
    before = copy.deepcopy(state)
    discovery = {'task_id': 'discover', 'content': {'scope': 'local', 'query': 'building change detection',
                  'queries': ['building change detection', '建筑变化检测'], 'top_k': 6}}
    action('work', discovery)
    first_id = state['work']['tasks']['discover']['output']
    count = runtime.db.get_run(rid)['used_tool_calls']
    # Simulate process loss after the tool receipt, before committing task state.
    state = harness._load(rid)
    assert state == before
    action('work', discovery)
    assert runtime.db.get_run(rid)['used_tool_calls'] == count
    assert state['work']['tasks']['discover']['output'] == first_id
    evidence = state['work']['artifacts'][first_id]['content']['evidence']
    paper_sha = evidence[0]['provenance']['paper_sha256']
    action('work', {'task_id': 'details', 'content': {'paper_sha256': paper_sha,
                    'query': 'temporal attention fusion architecture method', 'top_k': 6}})
    details_id = state['work']['tasks']['details']['output']
    details = state['work']['artifacts'][details_id]
    assert details['content']['ok'] and details['content']['evidence']
    assert all(e['provenance']['paper_sha256'] == paper_sha for e in details['content']['evidence'])
    quote = next(q for q in spans(details_id, details) if q['evidence_type'] == 'TEXT')
    action('work', {'task_id': 'analysis', 'content': {
        'findings': [{'artifact_id': details_id, 'quote_id': quote['quote_id'],
                      'interpretation': 'Scripted provenance check only; scientific interpretation not reviewed.'}],
        'gaps': ['No generated research analysis, novelty review, code or training in this probe.'],
        'directions': [{'hypothesis': 'Placeholder for interface verification, not a research proposal.',
                        'mechanism': 'Exact source-span handoff.', 'falsification': 'Reject stale or changed quote IDs.'}]}})
    analysis_id = state['work']['tasks']['analysis']['output']
    action('work', {'task_id': 'design', 'content': {
        'architecture': 'Scripted source handoff check only; no model architecture generated.',
        'literature_basis': [{'artifact_id': analysis_id, 'quote_id': quote['quote_id'],
                              'interpretation': 'Reference navigation only, no scientific claim.'}]}})
    design_id = state['work']['tasks']['design']['output']
    assert state['work']['artifacts'][design_id]['content']['literature_basis'][0]['quote'] == quote['text']
    harness._save(rid, state)
    (out/'completed-handoff-state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    # A changed reading requirement must invalidate only its descendants.
    revised = task_rows(state['work'])
    next(t for t in revised if t['id'] == 'details')['acceptance'] = 'new method evidence required'
    action('plan', {'tasks': revised})
    assert [state['work']['tasks'][k]['status'] for k in ['baseline','discover','details','analysis','design']] == [
        'DONE','DONE','PENDING','PENDING','PENDING']
    assert all(identity in state['work']['artifacts'] for identity in [first_id,details_id,analysis_id,design_id])
    harness._save(rid, state)
    run = runtime.db.get_run(rid)
    assert run['used_llm_calls'] == 0 and run['used_tool_calls'] == 2
    assert all(c['tool_name'] == 'paper.search' for c in runtime.db.list_tool_calls(rid))
    report = {'run_id':rid, 'scripted_decisions':True, 'model_calls':0, 'tool_calls':2,
              'scientific_review':'NOT_RUN', 'candidate_execution':False, 'training':False,
              'receipt_recovery_no_requery':True, 'single_paper_scope_verified':True,
              'exact_quote_handoff_verified':True, 'downstream_invalidation_verified':True,
              'history_preserved':True, 'selected_paper_sha256':paper_sha,
              'selected_paper_title':evidence[0]['provenance']['title'],
              'artifacts': {'discovery':first_id,'details':details_id,'analysis':analysis_id,'design':design_id},
              'post_revision_status':{k:t['status'] for k,t in state['work']['tasks'].items()},
              'work_digest':digest(state['work'])}
    (out/'host-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True))


if __name__ == '__main__':
    main()
