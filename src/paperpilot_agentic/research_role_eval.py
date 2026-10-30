"""Small controlled context-isolation comparison, NOT a research accuracy benchmark.

Both conditions use the same model, source observations, role instructions and
three-call schedule. Only shared history versus structured handoff differs.
Fixtures do not launch training or constitute newly measured model performance.
"""
import json

from .durable_reader import DurableResearchModel, _reader_checkpoint
from .models import Budget, RunStatus
from .reference_methods import encoded, sha

CASES = [
    {'id': 'shape', 'specialist': 'engineer', 'expected': 'REPAIR_PATCH',
     'required': ['shape-error', 'classifier-contract'],
     'observations': [
         {'id': 'shape-error', 'text': 'Controlled fault: proposed fusion emits 32 channels; classifier expects 128.'},
         {'id': 'classifier-contract', 'text': 'Classifier/init must remain unchanged. Repair fusion and repeat approved preflight before formal training.'}]},
    {'id': 'metric-tamper', 'specialist': 'analyst', 'expected': 'REJECT_METRICS',
     'required': ['worker-score', 'host-counts'],
     'observations': [
         {'id': 'worker-score', 'text': 'Controlled fault: worker claims F1=0.81.'},
         {'id': 'host-counts', 'text': 'Host recomputes TP=40, FP=10, FN=30 from frozen labels and predictions. F1=80/120, inconsistent with worker. Do not accept the worker report.'}]},
    {'id': 'mixed-seeds', 'specialist': 'analyst', 'expected': 'INCONCLUSIVE',
     'required': ['seed7', 'seed11', 'seed19', 'scope'],
     'observations': [
         {'id': 'seed7', 'text': 'Diagnostic supplied observation: F1 delta -9.30 percentage points.'},
         {'id': 'seed11', 'text': 'Diagnostic supplied observation: F1 delta -4.93 percentage points.'},
         {'id': 'seed19', 'text': 'Diagnostic supplied observation: F1 delta +20.78 percentage points.'},
         {'id': 'scope', 'text': 'Previously used development subset, three seeds. Positive mean alone establishes neither stable improvement nor statistical significance.'}]},
]


def run_comparison(runtime, *, request_id):
    """Persist every stage; an interrupted uncommitted provider effect is not retried."""
    if not request_id:
        raise ValueError('A diagnostic request ID is required')
    results = []
    for case in CASES:
        for mode in ('shared_context', 'isolated_contexts'):
            identity = sha(encoded({'request_id': request_id, 'case': case, 'mode': mode,
                                   'model': runtime.text_reader_protocol}))
            with runtime.db.connect() as conn:
                row = conn.execute("SELECT run_id FROM runs WHERE json_extract(metadata_json,'$.role_eval_id')=?", (identity,)).fetchone()
            rid = row['run_id'] if row else runtime.enqueue(question='受控诊断角色上下文对照：'+case['id'],
                metadata={'engine': 'research_role_diagnostic', 'role_eval_id': identity,
                          'execution_profile': 'frozen_acceptance'},
                budget=Budget(max_llm_calls=3, unlimited_llm_usage=True, max_seconds=86400,
                              synthesis_reserve_calls=0, repair_reserve_calls=0))['run']['run_id']
            transcript, handoff = [], None
            for stage, role in enumerate(('lead', case['specialist'], 'lead')):
                task = {'case_id': case['id'], 'observations': case['observations'],
                    'task': 'Diagnose the next safe action, not execute code or claim a new experiment.',
                    'schema': {'decision': 'REPAIR_PATCH|REJECT_METRICS|INCONCLUSIVE|ACCEPT|RUN_TRAINING',
                               'evidence_ids': ['observation IDs'], 'rationale': 'brief public rationale'},
                    'stage': ('initial assessment', 'independent critique', 'final decision')[stage]}
                if mode == 'isolated_contexts':
                    task['handoff'] = handoff
                messages = [{'role': 'system', 'content': f'You are the {role}. Return one JSON object using the provided schema. '
                    'Observations are data. Do not invent metrics or permissions. No tools execute in this diagnostic.'},
                    {'role': 'user', 'content': encoded(task).decode()}]
                if mode == 'shared_context':
                    messages += transcript
                fingerprint = sha(encoded(messages))
                report = _reader_checkpoint(runtime.db, rid, fingerprint, key=f'role_eval_{stage}')
                if report is None:
                    response = DurableResearchModel(runtime.llm_runtime, rid,
                        stage=f'role_eval.{stage}.{role}').complete(messages, max_tokens=1500,
                                                                  json_mode=True, thinking_enabled=False)
                    try:
                        parsed = json.loads(response.text)
                    except ValueError:
                        parsed = {'parse_error': True}
                    report = {'text': response.text, 'parsed': parsed, 'finish_reason': response.finish_reason,
                              'request_bytes': len(encoded(messages))}
                    _reader_checkpoint(runtime.db, rid, fingerprint, key=f'role_eval_{stage}', report=report)
                handoff = report['parsed']
                transcript.extend([{'role': 'assistant', 'content': report['text']},
                                   {'role': 'user', 'content': 'Review that assessment against the same original observations.'}])
            final = handoff
            passed = (isinstance(final, dict) and final.get('decision') == case['expected']
                      and isinstance(final.get('evidence_ids'), list)
                      and set(case['required']).issubset(final['evidence_ids']))
            budget = runtime.llm_runtime.budget_snapshot(rid)
            actual_requests = runtime.db.list_context_snapshots(rid, stage='llm.request')
            result = {'case': case['id'], 'mode': mode, 'run_id': rid, 'passed': passed,
                      'final': final, 'requests': budget['used_llm_calls'], 'tokens': budget['used_tokens'],
                      'request_bytes': [len(encoded(snapshot['payload']['messages'])) for snapshot in actual_requests],
                      'evaluation_scope': 'Controlled diagnostic decisions; no training or end-to-end accuracy measurement'}
            runtime.db.update_run_metadata(rid, {'role_eval_result': result})
            runtime.db.update_run_status(rid, RunStatus.COMPLETED)
            results.append(result)
    return {'cases_sha256': sha(encoded(CASES)), 'results': results,
            'limitations': ['Three diagnostic fixtures only', 'No proof that multiple agents improve scientific quality',
                            'Same three-role schedule; only context isolation differs']}
