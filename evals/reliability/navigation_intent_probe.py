"""Paired navigation diagnostic with deliberately lossy retrieval rewrites.

Public QASPER material, manually authored follow-ups, NOT benchmark answers or
an accuracy estimate. Compare routing decisions, never cherry-pick final prose.
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
from paperpilot_agentic.navigation_intent import INTENT_POLICY
from paperpilot_agentic.reader_recovery import plan_evidence_reload
from paperpilot_agentic.section_router import select_sections


# Deliberately omitted from the simulated retrieval rewrite, not discovered
# naturally in a live resolver trace. No official reference answers are loaded.
FOLLOW_UPS = {
    3: 'Now restrict the analysis to XNLI, distinguish translated training data '
       'from evaluation languages, and leave out question answering.',
    7: 'Now compare their word error rates and explain the settings where the '
       'improvement does not transfer; do not only repeat the mechanism.',
    8: 'Restrict the comparison to WeiboDial dialogue generation and report '
       'grammaticality and relevance separately; leave out language-model perplexity.',
}


def freeze(prior, dataset):
    protocol = json.loads((prior / 'protocol.json').read_text(encoding='utf-8'))
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != protocol['dataset_sha256']:
        raise ValueError('Dataset fingerprint mismatch')
    data = json.loads(dataset.read_text(encoding='utf-8'))
    cases = []
    for index, latest in FOLLOW_UPS.items():
        source = protocol['cases'][index]
        paper = data[source['paper_id']]
        outline = [{'section_id': f'section_{i}', 'title': section['section_name'] or '(untitled)',
                    'chars': sum(len(text) for text in section['paragraphs'])}
                   for i, section in enumerate(paper['full_text'])]
        saved = json.loads((prior / f'{index:02}-baseline' / 'result.json').read_text(encoding='utf-8'))
        report = {key: saved['report'].get(key) for key in ('status', 'draft', 'audit', 'limitations')}
        cases.append({'case': index, 'paper_id': source['paper_id'], 'question': source['resolved_question'],
                      'original_question': latest, 'request_history': [
                          {'role': 'initial_request', 'user_request': source['question']}],
                      'outline': outline, 'evidence': source['evidence'], 'report': report})
    return {'cases': cases, 'model': 'deepseek-chat', 'provider': 'deepseek', 'temperature': 0,
            'max_requests': len(cases) * 4, 'code_sha256': code_identity(),
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'dataset_sha256': protocol['dataset_sha256'],
            'input_sha256': hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            'scope': 'Already inspected public papers with manual follow-ups and intentionally lossy '
                     'rewrites; NOT an end-to-end evaluation, holdout, or official QASPER score. '
                     'Baseline removes the new intent policy/fields; evidence and other instructions '
                     'are identical. No tools, answers, reference labels or experiments are executed.'}


def run(plan, provider, output):
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'protocol.json', plan)
    attempts = 0
    blocked = False
    results = []
    for i, case in enumerate(plan['cases']):
        for stage in ('section', 'recovery'):
            for arm in (('baseline', 'intent') if i % 2 == 0 else ('intent', 'baseline')):
                if code_identity() != plan['code_sha256']:
                    raise ValueError('Source changed during frozen diagnostic')
                folder = output / f"{case['case']:02}-{stage}-{arm}"
                folder.mkdir()
                start_attempts = attempts
                result = {'case': case['case'], 'stage': stage, 'arm': arm}
                if blocked:
                    result.update(status='FAILED', error_type='ProviderCircuitOpen', attempts=0, known_tokens=0)
                    results.append(result)
                    save(folder / 'result.json', result)
                    continue

                class Recorder:
                    model = provider.model

                    def complete(self, messages, *, before_attempt, **kwargs):
                        request = [dict(message) for message in messages]
                        if arm == 'baseline':
                            if request[0]['content'].count(INTENT_POLICY) != 1:
                                raise ValueError('Intent ablation seam changed')
                            request[0]['content'] = request[0]['content'].replace(INTENT_POLICY, '').rstrip()
                            user_index = next(i for i, message in enumerate(request) if message['role'] == 'user')
                            payload = json.loads(request[user_index]['content'])
                            payload.pop('original_question')
                            payload.pop('request_history')
                            request[user_index]['content'] = json.dumps(payload, ensure_ascii=False)

                        def guard():
                            nonlocal attempts
                            if attempts >= plan['max_requests']:
                                raise RuntimeError('Frozen physical request cap')
                            before_attempt()
                            attempts += 1
                            save(folder / 'request.json', request)

                        started = time.monotonic()
                        try:
                            response = provider.complete(request, before_attempt=guard, **kwargs)
                            save(folder / 'response.json', {**asdict(response), 'seconds': time.monotonic() - started})
                            return response
                        except Exception as exc:
                            nonlocal blocked
                            status = getattr(exc, 'status_code', None)
                            if status in {401, 402, 403, 404}:
                                blocked = True
                            save(folder / 'error.json', {'type': type(exc).__name__, 'http_status': status})
                            raise RuntimeError(type(exc).__name__) from None

                db = RuntimeDatabase(folder / 'runtime.sqlite3')
                # This diagnostic performs a single navigation decision, not a
                # full answer pipeline; it has no later synthesis/repair stage.
                rid = db.create_run(case['original_question'], Budget(max_tokens=40000, max_llm_calls=1,
                    synthesis_reserve_calls=0, repair_reserve_calls=0))
                model = DurableResearchModel(PersistentLLMRuntime(db, Recorder()), rid,
                                             stage=stage, reserve_class='synthesis')
                try:
                    kwargs = {'original_question': case['original_question'], 'request_history': case['request_history']}
                    if stage == 'section':
                        decision = select_sections(case['question'], case['outline'], model, **kwargs)
                    else:
                        decision = plan_evidence_reload(case['question'], case['report'], case['evidence'],
                            {row['evidence_id'] for row in case['evidence']}, model, allow_search=True, **kwargs)
                    result.update(status='FAILED' if decision.get('status') == 'FAILED' else 'RETURNED', decision=decision)
                except Exception as exc:
                    result.update(status='FAILED', error_type=type(exc).__name__)
                result.update(attempts=attempts - start_attempts, known_tokens=db.get_run(rid)['used_tokens'])
                save(folder / 'result.json', result)
                results.append(result)
                print(json.dumps({k: result[k] for k in ('case', 'stage', 'arm', 'status', 'attempts')}), flush=True)
    save(output / 'results.json', results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', required=True, type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute-paid', action='store_true')
    args = parser.parse_args()
    plan = freeze(args.prior, args.dataset)
    if not args.execute_paid:
        print(json.dumps({key: plan[key] for key in ('max_requests', 'input_sha256', 'scope')}))
        return
    from paperpilot_agentic.config import RuntimeSettings, resolve_provider_api_key
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    settings = replace(RuntimeSettings.from_env(), provider='deepseek', model=plan['model'],
        api_key=resolve_provider_api_key('deepseek'), base_url='https://api.deepseek.com',
        thinking_enabled=False, temperature=0)
    results = run(plan, OpenAICompatibleLLM(settings, max_retries=0), args.output)
    print(json.dumps({'attempts': sum(row['attempts'] for row in results),
                      'known_tokens': sum(row['known_tokens'] for row in results),
                      'failures': sum(row['status'] == 'FAILED' for row in results)}))


if __name__ == '__main__':
    main()
