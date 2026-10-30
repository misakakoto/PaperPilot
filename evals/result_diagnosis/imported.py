"""Adapt existing predictions, then compare summary/tools in independent imported tasks."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sqlite3

from evals.interview.budget import CampaignBudget, RecordedProvider, save, source_fingerprint
from evals.result_diagnosis.live import PARENT, CANDIDATE, gpu_snapshot
from paperpilot_agentic.config import RuntimeSettings
from paperpilot_agentic.experiment_import import register_experiment
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.providers import OpenAICompatibleLLM
from paperpilot_agentic.research_harness import ResearchHarness
from paperpilot_agentic.research_contracts import SCIENTIFIC_REVIEW_PROTOCOLS
from scripts.export_diagnostic_bundle import export


def prepare(root, output):
    from paperpilot_agentic.reference_methods import encoded
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'source.sqlite3'
    if not target.exists():
        with sqlite3.connect((root/'.paperpilot/runtime.sqlite3').as_uri()+'?mode=ro', uri=True) as src, sqlite3.connect(target) as dst:
            src.backup(dst)
    runtime = ResearchAgentRuntime(workspace=root/'.paperpilot', db_path=target)
    before = encoded(runtime.db.get_run(PARENT))
    gpu = gpu_snapshot(runtime.workspace)
    results = {}
    for name, seeds in [('three_seeds', None), ('single_seed', [7])]:
        folder = output / name
        _, original = export(runtime, PARENT, CANDIDATE, folder, seeds=seeds)
        binding = register_experiment(runtime, folder/'manifest.json')
        from paperpilot_agentic.research_diagnostics import load_bundle
        imported = load_bundle(runtime, binding)
        for key in ('paired', 'coverage', 'samples'):
            expected = [r for r in original['views'][key] if seeds is None or r['seed'] in seeds]
            if imported['views'][key] != expected:
                raise ValueError('Imported metrics differ from verified source: ' + key)
        results[name] = binding
    if before != encoded(runtime.db.get_run(PARENT)) or gpu != gpu_snapshot(runtime.workspace):
        raise ValueError('Read-only preparation changed parent or GPU accounting')
    save(output/'adaptation.json', {'bundles': results, 'views_equal_to_verified_source': True,
        'parent_unchanged': True, 'gpu_ledgers_unchanged': True, 'new_gpu_starts': 0,
        'evaluation_scope': 'Adaptation of existing predictions, NOT independent experiment/generalization'})
    return results


def live(root, output, model, resume=False, report_protocol='legacy-v1', research_case_id=None, content_review_protocol=None):
    model = model or RuntimeSettings.from_env(require_api_key=True).model
    adaptation = json.loads((output/'adaptation.json').read_bytes())
    binding = adaptation['bundles']['three_seeds']['bundle_id']
    identity = {'version': 'imported-diagnosis-comparison-v1', 'source_sha256': source_fingerprint(root),
        'bundle_id': binding, 'model': model, 'conditions': ['summary', 'tools'], 'repeats': 3,
        'max_decision_turns': 12, 'usage_limit_mode': 'unlimited', 'request_cost_policy': 'legacy-utf8-v1',
        'data_authorization': 'Existing authorized SNUNet/SYSU-CD diagnostics; no images sent; no training'}
    if report_protocol != 'legacy-v1':
        identity['report_protocol'] = report_protocol
    if research_case_id is not None:
        identity['research_case_id'] = research_case_id
    if content_review_protocol is not None:
        identity['content_review_protocol'] = content_review_protocol
    folder = output/'live'
    folder.mkdir(exist_ok=True)
    protocol = folder/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_bytes()) != identity:
        raise ValueError('Frozen live protocol changed')
    if not protocol.exists():
        save(protocol, identity)
    ledger = CampaignBudget(folder/'receipts', identity, max_requests=None, max_tokens=None)
    if ledger.snapshot()['blocked']:
        print('Retained blocked/unknown ledger; no new dispatch', flush=True)
        return
    settings = replace(RuntimeSettings.from_env(require_api_key=True), model=model,
        fallback_models=(), thinking_enabled=False, timeout_seconds=120)
    provider = RecordedProvider(OpenAICompatibleLLM(settings, max_retries=0), ledger, phase='preflight')
    if not (folder/'preflight.json').exists():
        try:
            response = provider.complete([{'role': 'user', 'content': 'Return JSON {"ready":true} only.'}],
                max_tokens=64, thinking_enabled=False, json_mode=True)
            save(folder/'preflight.json', {'ready': json.loads(response.text) == {'ready': True}, 'actual_model': response.model})
        except Exception as exc:
            save(folder/'preflight.json', {'ready': False, 'error_type': type(exc).__name__})
            print('Preflight failed; no experiment tasks sent', flush=True)
            return
    if not json.loads((folder/'preflight.json').read_bytes())['ready']:
        return
    runtime = ResearchAgentRuntime(workspace=root/'.paperpilot', db_path=folder/'runtime.sqlite3',
        llm=provider, text_reader_protocol='imported-diagnosis-comparison-v1')
    harness = ResearchHarness(runtime)
    gpu_before = gpu_snapshot(runtime.workspace)
    for repeat in range(1, 4):
        for mode in ('summary', 'tools'):
            key = f'{mode}-{repeat}'
            result_path = folder/(key+'-result.json')
            if result_path.exists():
                print(key + ': retained completion', flush=True)
                continue
            if ledger.snapshot()['blocked']:
                break
            provider.phase = key
            submission = folder/(key+'-submission.json')
            if submission.exists():
                if not resume:
                    raise RuntimeError('Interrupted task needs explicit --resume; no new submission')
                rid = json.loads(submission.read_bytes())['run_id']
            else:
                rid = harness.start_imported_diagnosis(binding, 'imported-live-'+key,
                    comparison_mode=mode, report_protocol=report_protocol,
                    research_case_id=research_case_id, content_review_protocol=content_review_protocol)['run']['run_id']
                save(submission, {'run_id': rid})
            result = harness.advance(rid)
            save(result_path, result)
            (folder/(key+'-answer.md')).write_text((result.get('final_answer') or {}).get('answer', ''), encoding='utf-8')
            before = ledger.snapshot()['actual_requests']
            harness.advance(rid)
            assert before == ledger.snapshot()['actual_requests'], 'Completion/unknown resume sent another request'
            print(json.dumps({'case': key, 'phase': result['research_episode']['phase'],
                'views': result['research_episode']['observed_views'],
                'calls': result['run']['used_llm_calls'], 'tokens': result['run']['used_tokens'],
                'resume_additional_requests': 0}, ensure_ascii=False), flush=True)
        if ledger.snapshot()['blocked']:
            break
    if gpu_before != gpu_snapshot(runtime.workspace):
        raise ValueError('Diagnosis unexpectedly changed GPU ledger')
    save(folder/'snapshot.json', ledger.snapshot())
    save(folder/'execution-boundary.json', {'new_gpu_starts': 0, 'gpu_ledgers_unchanged': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'live'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model')
    parser.add_argument('--research-case-id')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--report-protocol',choices=('legacy-v1','metric-facts-v2','metric-facts-v3'),default='legacy-v1')
    parser.add_argument('--content-review-protocol',choices=sorted(SCIENTIFIC_REVIEW_PROTOCOLS))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.command == 'prepare':
        print(json.dumps(prepare(root, args.output.resolve()), ensure_ascii=False), flush=True)
    else:
        live(root, args.output.resolve(), args.model, args.resume, args.report_protocol, args.research_case_id, args.content_review_protocol)
