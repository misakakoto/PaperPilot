"""Two-call diagnostic on a previously inspected public QASPER workset.

This is not a benchmark, a holdout, or an answer-quality estimate. Neither arm
executes its proposed action. Store raw requests/responses and physical attempts.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

from evals.qasper.workbench_acceptance import code_identity, save
from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.durable_reader import DurableResearchModel
from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
from paperpilot_agentic.models import Budget
from paperpilot_agentic.reader_recovery import plan_evidence_reload
from paperpilot_agentic.source_references import REFERENCE_IDENTITY_POLICY


def run(prior, provider, output):
    source = json.loads((prior / 'protocol.json').read_text(encoding='utf-8'))
    case = next(row for row in source['cases'] if row['case'] == 8)
    output.mkdir(parents=True, exist_ok=False)
    fingerprint = code_identity()
    save(output / 'protocol.json', {
        'case': case, 'max_physical_requests': 2, 'code_sha256': fingerprint,
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'prior_protocol_sha256': hashlib.sha256((prior / 'protocol.json').read_bytes()).hexdigest(),
        'dataset_sha256': source['dataset_sha256'], 'requested_model': provider.model,
        'temperature': 0, 'scope': __doc__,
        'ablation': 'Remove only reference identity policy and token inventory; retain owned user intent.'})
    results, attempts = [], 0
    blocked = False
    for arm in ('baseline', 'reference_identity'):
        folder = output / arm
        folder.mkdir()
        result = {'arm': arm}
        before = attempts
        db = RuntimeDatabase(folder / 'runtime.sqlite3')
        rid = db.create_run(case['original_question'], Budget(
            max_tokens=40000, max_llm_calls=1, synthesis_reserve_calls=0, repair_reserve_calls=0))

        class Recorder:
            model = provider.model

            def complete(self, messages, *, before_attempt, **kwargs):
                request = [dict(message) for message in messages]
                if arm == 'baseline':
                    assert request[0]['content'].count(REFERENCE_IDENTITY_POLICY) == 1
                    request[0]['content'] = request[0]['content'].replace(REFERENCE_IDENTITY_POLICY, '').rstrip()
                    user_index = next(i for i, message in enumerate(request) if message['role'] == 'user')
                    payload = json.loads(request[user_index]['content'])
                    payload.pop('reference_placeholders')
                    request[user_index]['content'] = json.dumps(payload, ensure_ascii=False)
                save(folder / 'request.json', {'messages': request, 'options': kwargs})

                def guard():
                    nonlocal attempts
                    if attempts >= 2:
                        raise RuntimeError('Diagnostic physical request budget exhausted')
                    before_attempt()
                    attempts += 1

                started = time.monotonic()
                try:
                    response = provider.complete(request, before_attempt=guard, **kwargs)
                    save(folder / 'response.json', {**asdict(response), 'seconds': time.monotonic() - started})
                    return response
                except Exception as exc:
                    nonlocal blocked
                    status = getattr(exc, 'status_code', None)
                    blocked = status in {401, 402, 403, 404}
                    save(folder / 'error.json', {'error_type': type(exc).__name__, 'http_status': status})
                    raise RuntimeError(type(exc).__name__) from None

        try:
            if blocked:
                raise RuntimeError('ProviderCircuitOpen')
            if fingerprint != code_identity():
                raise ValueError('Source changed during frozen diagnostic')
            model = DurableResearchModel(PersistentLLMRuntime(db, Recorder()), rid,
                                         stage='recovery', reserve_class='synthesis')
            decision = plan_evidence_reload(case['question'], case['report'], case['evidence'],
                {row['evidence_id'] for row in case['evidence']}, model, allow_search=True,
                original_question=case['original_question'], request_history=case['request_history'])
            result.update(status='RETURNED', decision=decision)
        except Exception as exc:
            result.update(status='FAILED', error_type=type(exc).__name__)
        result.update(attempts=attempts - before, known_tokens=db.get_run(rid)['used_tokens'])
        results.append(result)
        save(folder / 'result.json', result)
        print(json.dumps({key: result[key] for key in ('arm', 'status', 'attempts', 'known_tokens')}), flush=True)
    save(output / 'results.json', results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute-paid', action='store_true')
    args = parser.parse_args()
    if not args.execute_paid:
        print('At most two paid requests; no tools or experiments. Use --execute-paid to run.')
        return
    from paperpilot_agentic.config import RuntimeSettings, resolve_provider_api_key
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    settings = replace(RuntimeSettings.from_env(), provider='deepseek', model='deepseek-chat',
        api_key=resolve_provider_api_key('deepseek'), base_url='https://api.deepseek.com',
        thinking_enabled=False, temperature=0)
    run(args.prior, OpenAICompatibleLLM(settings, max_retries=0), args.output)


if __name__ == '__main__':
    main()
