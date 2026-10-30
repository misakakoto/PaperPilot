"""Independent set-based re-scoring of saved paragraph IDs, without retrieval/model calls."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from evals.qasper.agent_case import paper_records
from paperpilot_retrieval.store import stable_chunk_id


def audit(dataset_path, directory):
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    raw = dataset_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest['dataset_sha256']
    dataset = json.loads(raw)
    assert not set(manifest['papers']) & set(manifest['excluded_papers'])
    expected_papers = sorted(set(dataset)-set(manifest['excluded_papers']),
        key=lambda p: hashlib.sha256(('retrieval-v2:'+p).encode()).hexdigest())[:len(manifest['papers'])]
    assert manifest['papers'] == expected_papers, 'cohort must match frozen hash selection'
    rows = [json.loads(line) for line in (directory/'rankings.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == len({r['question_id'] for r in rows})
    by_question = {r['question_id']:r for r in rows}
    assert set(by_question) == set(manifest['question_ids'])
    totals, count, checked = defaultdict(float), 0, 0
    candidate_totals, candidate_count = defaultdict(float), defaultdict(int)
    for pid in manifest['papers']:
        paper = dataset[pid]
        _,doc,chunks = paper_records({k:v for k,v in paper.items() if k!='qas'},source_id=pid)
        texts = {stable_chunk_id(doc.sha256,c):c.text for c in chunks}
        for qa in paper['qas']:
            row = by_question[qa['question_id']]
            assert row['question'] == qa['question'] and row['paper_id'] == pid
            references = []
            for annotation in qa['answers']:
                a = annotation['answer']
                if a.get('unanswerable'):
                    continue
                gold = {e for e in a.get('evidence',[]) if e and 'FLOAT SELECTED' not in e}
                if gold:
                    references.append(gold)
            if references:
                count += 1
            for arm in manifest['arms']:
                ranks, candidates = row[arm]['ranks'], row[arm]['candidates']
                assert len(ranks) <= 8 and len(ranks) == len(set(ranks))
                assert set(ranks) <= set(candidates) <= set(texts)
                checked += 1
                candidate_count[arm] += len(candidates)
                for k in (1,3,8):
                    score = max((len({texts[i] for i in ranks[:k]} & ref)/len(ref) for ref in references),default=None)
                    stored = row[arm]['metrics']['best_text_recall']
                    assert (stored is None and score is None) or abs(stored[str(k)]-score)<1e-12
                    if score is not None:
                        totals[(arm,k)] += score
                if references:
                    coverage = max(len({texts[i] for i in candidates}&g)/len(g) for g in references)
                    assert abs(row[arm]['candidate_coverage']-coverage)<1e-12
                    candidate_totals[arm] += coverage
    report = dict(queries=len(rows), eligible=count, verified_rankings=checked,
        no_duplicate_ranks=True, only_original_paragraphs=True, paper_scope_valid=True,
        recall={a:{str(k):totals[(a,k)]/count for k in (1,3,8)} for a in manifest['arms']},
        coverage={a:candidate_totals[a]/count for a in manifest['arms']},
        mean_candidates={a:candidate_count[a]/len(rows) for a in manifest['arms']},
        provider_requests=0)
    with (directory/'independent-audit.json').open('x',encoding='utf-8') as out:
        json.dump(report,out,indent=2)
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--directory',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(audit(args.dataset,args.directory),indent=2))
