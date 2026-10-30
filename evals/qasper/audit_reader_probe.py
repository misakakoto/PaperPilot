"""Re-score unchanged saved responses and paired bootstrap; no API calls."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random


def audit(output, dataset_path, evaluator_path):
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    raw = dataset_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest['dataset_sha256']
    spec = importlib.util.spec_from_file_location('official', evaluator_path)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    data = json.loads(raw)
    gold = ev.get_answers_and_evidence(data, False)
    papers = {q['question_id']:p for p,paper in data.items() for q in paper['qas']}
    values, groups = {}, defaultdict(list)
    usage, errors, attempts = 0, 0, 0
    for qid in manifest['question_ids']:
        values[qid] = {}
        for arm in ('current','hybrid_final'):
            row = json.loads((output/(qid+'-'+arm+'-response.json')).read_text(encoding='utf-8'))
            assert row['question_id'] == qid and row['arm'] == arm
            response = row['response']
            attempts += row['provider_attempts']
            errors += bool(row['error'])
            answer = response['text'] if response else ''
            values[qid][arm] = max(ev.token_f1_score(answer,g['answer']) for g in gold[qid]) if not row['error'] else 0
            if response:
                usage += response['prompt_tokens'] + response['completion_tokens']
        groups[papers[qid]].append(values[qid]['hybrid_final']-values[qid]['current'])
    rng, samples = random.Random(20260913), []
    for _ in range(2000):
        vals = [v for p in rng.choices(list(groups),k=len(groups)) for v in groups[p]]
        samples.append(sum(vals)/len(vals))
    samples.sort()
    report = dict(questions=len(values), papers=len(groups), actual_attempts=attempts, errors=errors,
        provider_tokens=usage, answer_f1={a:sum(v[a] for v in values.values())/len(values) for a in ('current','hybrid_final')},
        delta_paper_bootstrap95=[samples[50],samples[1949]],
        scope='small paired short-answer reader diagnostic; not full Agent/official benchmark', scores=values)
    path = output/'rescored.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    return {k:v for k,v in report.items() if k != 'scores'}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','dataset','evaluator'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(audit(a.output,a.dataset,a.evaluator),indent=2))
