"""Two real read-only diagnosis conditions over a copied historical runtime DB."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3

from evals.interview.budget import CampaignBudget, RecordedProvider, save, source_fingerprint
from paperpilot_agentic.config import RuntimeSettings
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.providers import OpenAICompatibleLLM
from paperpilot_agentic.research_harness import ResearchHarness
from paperpilot_agentic.reference_methods import encoded, sha

PARENT = 'run_b05dd3008a8e4d2c856a2859366d6af5'
CANDIDATE = 'f36cedfbcc709e6a8ae14a440ecc5dd7b6f31de0fce8a5cd53a2ddf555c78fd1'


def gpu_snapshot(workspace):
    result = {}
    for path in (workspace/'research-harness/episodes').glob('*/gpu-budget.sqlite3'):
        with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True) as conn:
            result[str(path)] = conn.execute('SELECT count(*),COALESCE(sum(charged),0) FROM launches').fetchone()
    return result


def run(output, *, model, resume=False):
    root = Path(__file__).resolve().parents[2]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = RuntimeSettings.from_env(require_api_key=True)
    model = model or settings.model
    identity = {'source_fingerprint': source_fingerprint(root), 'model': model,
        'provider': settings.provider, 'endpoint_sha256': hashlib.sha256(settings.base_url.encode()).hexdigest(),
        'parent_run_id': PARENT, 'candidate_id': CANDIDATE,
        'conditions': ['summary', 'tools'], 'max_decision_turns': 12,
        'usage_limit_mode': 'unlimited', 'request_cost_policy': 'legacy-utf8-v1',
        'authority': 'User authorized unlimited cumulative model calls; NO new training'}
    protocol = output/'protocol.json'
    if protocol.exists():
        if json.loads(protocol.read_bytes()) != identity:
            raise ValueError('Frozen diagnosis protocol changed')
    else:
        save(protocol, identity)
    campaign = CampaignBudget(output/'receipts', identity, max_requests=None, max_tokens=None)
    if campaign.snapshot()['blocked']:
        print(json.dumps({'blocked': True, 'reason': 'Retained failed/unknown request; no implicit replay'}))
        return
    db = output/'runtime.sqlite3'
    if not db.exists():
        original = root/'.paperpilot/runtime.sqlite3'
        with sqlite3.connect(original.as_uri()+'?mode=ro', uri=True) as src, sqlite3.connect(db) as dst:
            src.backup(dst)
    settings = replace(settings, runtime_db=db, workspace=root/'.paperpilot', model=model,
                       fallback_models=(), thinking_enabled=False, timeout_seconds=120)
    provider = RecordedProvider(OpenAICompatibleLLM(settings, max_retries=0), campaign, phase='preflight')
    if not (output/'preflight.json').exists():
        try:
            response = provider.complete([{'role': 'user', 'content': 'Return JSON {"ready":true} only.'}],
                                         max_tokens=64, thinking_enabled=False, json_mode=True)
            save(output/'preflight.json', {'ready': json.loads(response.text) == {'ready': True},
                                          'actual_model': response.model})
        except Exception as exc:
            save(output/'preflight.json', {'ready': False, 'error_type': type(exc).__name__})
            print(json.dumps({'preflight': 'FAILED', 'error_type': type(exc).__name__}), flush=True)
            return
    if not json.loads((output/'preflight.json').read_bytes())['ready']:
        return
    runtime = ResearchAgentRuntime(workspace=settings.workspace, db_path=db,
                                   llm=provider, text_reader_protocol='diagnosis-live-v1')
    harness = ResearchHarness(runtime)
    with runtime.db.connect() as conn:
        parent_state_before = conn.execute('SELECT state_json FROM research_episodes WHERE run_id=?', (PARENT,)).fetchone()[0]
    parent_before = encoded(runtime.db.get_run(PARENT))
    gpu_before = gpu_snapshot(settings.workspace)
    for mode in ('summary', 'tools'):
        target = output/(mode+'-result.json')
        if target.exists():
            print(json.dumps({'mode': mode, 'reused': True}), flush=True)
            continue
        if campaign.snapshot()['blocked']:
            break
        provider.phase = 'diagnosis.' + mode
        submission = output/(mode+'-submission.json')
        if submission.exists():
            if not resume:
                raise RuntimeError('Interrupted diagnosis requires explicit --resume; no new submission')
            rid = json.loads(submission.read_bytes())['run_id']
        else:
            result = harness.start_diagnosis(PARENT, CANDIDATE, 'diagnosis-' + mode + '-' + sha(encoded(identity))[:20],
                                            comparison_mode=mode)
            rid = result['run']['run_id']
            save(submission, {'run_id': rid})
        harness.advance(rid)
        result = runtime.status(rid)
        result['research_episode'] = harness.view(rid)
        save(target, result)
        (output/(mode+'-answer.md')).write_text((result.get('final_answer') or {}).get('answer', ''), encoding='utf-8')
        print(json.dumps({'mode': mode, 'phase': result['research_episode']['phase'],
            'views': result['research_episode']['observed_views'],
            'calls': result['run']['used_llm_calls'], 'tokens': result['run']['used_tokens'],
            'new_gpu_starts': 0}), flush=True)
        if result['research_episode']['phase'] in {'UNKNOWN', 'FAILED'}:
            break
    with runtime.db.connect() as conn:
        parent_state_after = conn.execute('SELECT state_json FROM research_episodes WHERE run_id=?', (PARENT,)).fetchone()[0]
    preservation = {
        'parent_run_unchanged': parent_before == encoded(runtime.db.get_run(PARENT)),
        'parent_state_unchanged': parent_state_before == parent_state_after,
        'gpu_ledgers_unchanged': gpu_before == gpu_snapshot(settings.workspace),
        'gpu_ledgers_before': gpu_before, 'gpu_ledgers_after': gpu_snapshot(settings.workspace),
        'campaign': campaign.snapshot()}
    # Completed re-entry preserves the first frozen audit instead of rewriting it.
    if not (output/'parent-preservation.json').exists():
        save(output/'parent-preservation.json', preservation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.output, model=args.model, resume=args.resume)
