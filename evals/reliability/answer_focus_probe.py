"""Matched Writer/Auditor diagnostic on previously inspected QASPER worksets.

No retrieval, gold input, best-of selection or success-only scoring. NOT holdout
or end-to-end evaluation. Freeze the intervention before any paid request.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
import threading
import time

from evals.qasper.workbench_acceptance import code_identity, predicted_paragraphs, save
from evals.qasper.score_oracle import load_evaluator
from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.direct_answer import prepare_evidence


BASE_FOCUS = (
    'For a single narrow question, normally use 1-3 nonredundant factual statements: '
    'the direct answer and only indispensable context. This is a default, not a hard '
    'cap: preserve all requested parts for multipart questions, detailed procedures, '
    'or exhaustive lists. Omit general motivation, training history and applications '
    'unless requested or necessary to understand the answer. '
)
FOCUSED_WRITER = (
    'Organize the answer around the requested answer type, not around a paper summary. '
    'Make the leading fact the smallest self-contained answer: name the requested '
    'entities for a names question; give the value with unit and experimental scope '
    'for a quantity; give the requested mechanism for a how question. For a yes/no '
    'question begin with Yes or No ONLY when the supplied evidence establishes that '
    'answer for the actual question scope, followed by a brief source-grounded clause; '
    'otherwise state the specific uncertainty. Add another fact only for a requested '
    'part or indispensable qualification. Do not add source statistics, motivation, '
    'applications or scores merely because they were retrieved. Concision must not '
    'remove conditions, list members, explicit subquestions or necessary procedural '
    'steps. There is no word or fact-count cap for a genuinely complex request. '
)


def freeze(workspace, manifest):
    prior = json.loads(manifest.read_text(encoding='utf-8'))
    cases = []
    for i, case in enumerate(prior['cases']):
        snapshot = json.loads((workspace / f'case-{i:02}' / 'result.json').read_text(encoding='utf-8'))
        metadata = snapshot['run']['metadata']
        report = metadata['reader_progress']
        selected = list(dict.fromkeys(s['evidence_id'] for s in report.get('source_spans', [])))
        if not selected:
            selected = metadata['reader_workset']['selected_evidence_ids']
        by_id = {row['evidence_id']: row for row in snapshot['evidence']}
        rows = prepare_evidence([by_id[eid] for eid in selected])
        cases.append({**case, 'resolved_question': report['question'], 'evidence': rows})
    return {'cases': cases, 'input_sha256': hashlib.sha256(
                json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            'dataset_sha256': prior['dataset_sha256'], 'code_sha256': code_identity(),
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'base_focus': BASE_FOCUS, 'focused_writer': FOCUSED_WRITER,
            'max_requests': len(cases) * 2 * 3, 'model': 'qwen-plus', 'temperature': 0,
            'scope': 'All previously frozen cases; debugging, NOT holdout. Identical fixed '
                     'retrieved evidence; no reference answers in model input. No retrieval/recovery '
                     'planner; one shared format/truncation repair per arm; no transport retries.',
            'adoption': 'Do not adopt on token F1 alone: inspect factual scope, missing requested '
                        'parts and unsupported yes/no assertions. All failures score zero.'}


def run(plan, provider, output):
    from paperpilot_agentic.db import RuntimeDatabase
    from paperpilot_agentic.durable_reader import DurableResearchModel
    from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
    from paperpilot_agentic.models import Budget
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'protocol.json', plan)
    lock = threading.Lock()
    provider_blocked = threading.Event()
    total_attempts = 0

    def pair(index):
        case = plan['cases'][index]
        results = []
        for arm in (['baseline', 'focused'] if index % 2 == 0 else ['focused', 'baseline']):
            if code_identity() != plan['code_sha256']:
                raise ValueError('Code changed during frozen experiment')
            folder = output / f'{index:02}-{arm}'
            folder.mkdir()
            attempts = 0
            if provider_blocked.is_set():
                result = {'case': index, 'question_id': case['question_id'], 'arm': arm,
                          'report': {'status': 'FAILED', 'error_type': 'ProviderCircuitOpen'},
                          'attempts': 0, 'known_tokens': 0}
                save(folder / 'result.json', result)
                results.append(result)
                continue

            class Recorder:
                model = provider.model

                def complete(self, messages, *, before_attempt, **kwargs):
                    nonlocal attempts, total_attempts
                    request = [dict(m) for m in messages]
                    matches = request[0]['content'].count(BASE_FOCUS)
                    if matches > 1:
                        raise ValueError('Ambiguous Writer seam')
                    if arm == 'focused' and matches and plan.get('intervention', 'writer-paragraph') == 'writer-paragraph':
                        request[0]['content'] = request[0]['content'].replace(BASE_FOCUS, FOCUSED_WRITER)

                    def counted():
                        nonlocal attempts, total_attempts
                        with lock:
                            if provider_blocked.is_set():
                                raise RuntimeError('Provider circuit open; no further requests')
                            if total_attempts >= plan['max_requests']:
                                raise RuntimeError('Frozen request cap')
                            before_attempt()
                            total_attempts += 1
                            attempts += 1
                        save(folder / f'{attempts:02}-request.json', request)

                    start = time.monotonic()
                    try:
                        response = provider.complete(request, before_attempt=counted, **kwargs)
                        save(folder / f'{attempts:02}-response.json',
                             {**asdict(response), 'seconds': time.monotonic() - start})
                        return response
                    except Exception as exc:
                        status = getattr(exc, 'status_code', None)
                        if status in {401, 402, 403, 404}:
                            provider_blocked.set()
                        save(folder / f'{attempts:02}-error.json',
                             {'type': type(exc).__name__, 'http_status': status,
                              'usage_unknown': attempts > 0})
                        raise RuntimeError(type(exc).__name__) from None

            db = RuntimeDatabase(folder / 'runtime.sqlite3')
            rid = db.create_run(case['question'], Budget(max_tokens=100000, max_llm_calls=3))
            model = DurableResearchModel(PersistentLLMRuntime(db, Recorder()), rid,
                                         stage='answer_first', reserve_class='synthesis')
            report = answer_then_audit(case['resolved_question'], case['evidence'], model,
                original_question=case['question'], relation_policy=True, question_scope_policy=True,
                draft_contract='layered-v1' if arm == 'focused' and
                plan.get('intervention') == 'layered-schema' else 'flat')
            result = {'case': index, 'question_id': case['question_id'], 'arm': arm,
                      'report': report, 'attempts': attempts, 'known_tokens': db.get_run(rid)['used_tokens']}
            save(folder / 'result.json', result)
            results.append(result)
            print(json.dumps({'case': index, 'arm': arm, 'status': report['status'],
                              'attempts': attempts}), flush=True)
        return results

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [row for pair_results in pool.map(pair, range(len(plan['cases']))) for row in pair_results]
    save(output / 'results.json', results)
    return results


def score(plan, results, dataset, evaluator):
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != plan['dataset_sha256']:
        raise ValueError('Dataset changed')
    module = load_evaluator(evaluator)
    data = json.loads(dataset.read_text(encoding='utf-8'))
    refs = module.get_answers_and_evidence(data, True)
    rows = []
    keyed = {(r['case'], r['arm']): r for r in results}
    if len(keyed) != len(results):
        raise ValueError('Duplicate results')
    complete = [keyed.get((index, arm), {'case': index, 'arm': arm,
                'question_id': case['question_id'], 'attempts': 0, 'known_tokens': 0,
                'report': {'status': 'FAILED'}})
                for index, case in enumerate(plan['cases']) for arm in ('baseline', 'focused')]
    for result in complete:
        report = result['report']
        answer = report.get('answer', '') if report['status'] != 'FAILED' else ''
        stripped = re.sub(r'\[[^\]\n]+\]', '', answer).strip()
        gold = refs[result['question_id']]
        evidence = plan['cases'][result['case']]['evidence']
        paragraphs = predicted_paragraphs(data[plan['cases'][result['case']]['paper_id']],
            {'evidence': evidence, 'final_answer': {'citation_ids': re.findall(r'\[([^\]\n]+)\]', answer)}})
        rows.append({k: result[k] for k in ('case', 'arm', 'question_id', 'attempts', 'known_tokens')} | {
            'published': bool(answer), 'status': report['status'], 'answer_chars': len(stripped),
            'raw_f1': max(module.token_f1_score(answer, g['answer']) for g in gold) if answer else 0,
            'stripped_f1': max(module.token_f1_score(stripped, g['answer']) for g in gold) if answer else 0,
            'evidence_f1': max(module.paragraph_f1_score(paragraphs, g['evidence']) for g in gold) if answer else 0})
    aggregate = {}
    for arm in ('baseline', 'focused'):
        selected = [r for r in rows if r['arm'] == arm]
        aggregate[arm] = {k: sum(r[k] for r in selected) / len(selected)
                          for k in ('raw_f1', 'stripped_f1', 'evidence_f1', 'answer_chars')}
        aggregate[arm].update({k: sum(r[k] for r in selected)
                              for k in ('published', 'attempts', 'known_tokens')})
        aggregate[arm]['failed'] = sum(r['status'] == 'FAILED' for r in selected)
    return {'summary': aggregate, 'per_case': rows, 'scope': plan['scope']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--evaluator', type=Path, required=True)
    p.add_argument('--execute-paid', action='store_true')
    p.add_argument('--intervention', choices=['writer-paragraph', 'layered-schema'], default='writer-paragraph')
    p.add_argument('--provider', choices=['qwen', 'deepseek'], default='qwen')
    p.add_argument('--model', default='qwen-plus')
    args = p.parse_args()
    plan = freeze(args.workspace, args.manifest)
    plan['intervention'] = args.intervention
    plan.update(provider=args.provider, model=args.model,
                fatal_provider_policy='stop unsent jobs on 401/402/403/404; keep in-flight attempts and failures')
    if args.intervention == 'layered-schema':
        from paperpilot_agentic.draft_schema import LAYERED_SCHEMA, LAYERED_INSTRUCTION
        plan.update(layered_schema=LAYERED_SCHEMA, layered_instruction=LAYERED_INSTRUCTION)
        plan['scope'] += (' Treatment changes Writer output schema and canonicalizes answer/context '
                          'facts without rewriting them; Auditor treats writer roles as fallible. '
                          'Primary score still includes the COMPLETE answer, not just the direct section.')
    if not args.execute_paid:
        print(json.dumps({k: plan[k] for k in ('max_requests', 'input_sha256', 'scope')}))
        return
    from paperpilot_agentic.config import RuntimeSettings, resolve_provider_api_key
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    settings = replace(RuntimeSettings.from_env(), provider=args.provider,
        api_key=resolve_provider_api_key(args.provider), model=args.model, thinking_enabled=False,
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1' if args.provider == 'qwen'
        else 'https://api.deepseek.com', temperature=0)
    results = run(plan, OpenAICompatibleLLM(settings, max_retries=0), args.output)
    scored = score(plan, results, args.dataset, args.evaluator)
    save(args.output / 'score.json', scored)
    print(json.dumps(scored['summary']))


if __name__ == '__main__':
    main()
