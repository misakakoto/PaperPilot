"""Reconstruct LLM input/ID mapping and independently score saved original paragraphs."""
import argparse
import hashlib
import json
from pathlib import Path

from evals.qasper.agent_case import paper_records
from paperpilot_retrieval.store import stable_chunk_id


def audit(dataset_path, candidates_dir, directory):
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    manifest = read(directory/'manifest.json')
    raw = dataset_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest['dataset_sha256']
    assert hashlib.sha256((candidates_dir/'manifest.json').read_bytes()).hexdigest() == manifest['candidate_manifest_sha256']
    dataset = json.loads(raw)
    records = {r['question_id']: r for r in map(json.loads, (candidates_dir/'rankings.jsonl').read_text(encoding='utf-8').splitlines())}
    totals = {str(k): 0.0 for k in (1, 3, 8)}
    baseline_totals = {str(k): 0.0 for k in (1, 3, 8)}
    eligible = attempts = errors = tokens = 0
    assert len(manifest['question_ids']) == len(set(manifest['question_ids']))
    assert {p.name.removesuffix('-result.json') for p in directory.glob('*-result.json')} == set(manifest['question_ids'])
    for qid in manifest['question_ids']:
        source = records[qid]
        row = read(directory/(qid+'-result.json'))
        assert row['baseline'] == source['legacy']
        paper = dataset[source['paper_id']]
        qa = next(q for q in paper['qas'] if q['question_id'] == qid)
        assert row['question'] == qa['question'] == source['question']
        _, doc, chunks = paper_records({k:v for k,v in paper.items() if k!='qas'}, source_id=source['paper_id'])
        by_id = {stable_chunk_id(doc.sha256,c):c for c in chunks}
        ids = source['expanded']['candidates'][:24]
        ordered = sorted(ids, key=lambda i: hashlib.sha256((qa['question']+'\0'+i).encode()).hexdigest())
        mapping = {f'C{i:02d}':sid for i,sid in enumerate(ordered,1)}
        request = read(directory/(qid+'-request.json'))
        expected = {'question':qa['question'], 'passages':[
            {'id':i,'text':f'{doc.title}\n{by_id[sid].section}\n{by_id[sid].text}'.strip()[:3500]}
            for i,sid in mapping.items()]}
        assert request['messages'] == [{'role':'system','content':manifest['prompt']},
            {'role':'user','content':json.dumps(expected,ensure_ascii=False)}]
        assert request['max_tokens'] == manifest.get('max_output_tokens',512) and request['json_mode'] is True and request['thinking_enabled'] is False
        assert row['attempts'] == 1
        attempts += row['attempts']
        response_path = directory/(qid+'-response.json')
        if not row['error']:
            response = read(response_path)
            ranking = json.loads(response['text'])['ranking']
            if manifest.get('protocol') == 'select-v2':
                assert len(ranking) == min(8,len(mapping)) and len(set(ranking)) == len(ranking) and set(ranking) <= set(mapping)
            else:
                assert len(ranking) == len(mapping) and set(ranking) == set(mapping)
            assert row['ranks'] == [mapping[i] for i in ranking[:8]]
        else:
            errors += 1
            if manifest.get('protocol') == 'select-v2':
                assert row['degraded'] is True and row['ranks'] == source['legacy']['ranks']
            else:
                assert row['ranks'] == []
        if response_path.exists():
            usage = read(response_path)
            tokens += max(0, usage['prompt_tokens']) + max(0, usage['completion_tokens'])
        assert len(row['ranks']) <= 8 and len(set(row['ranks'])) == len(row['ranks'])
        refs = []
        for annotation in qa['answers']:
            answer = annotation['answer']
            if answer.get('unanswerable'):
                continue
            evidence = {e for e in answer.get('evidence',[]) if e and 'FLOAT SELECTED' not in e}
            if evidence:
                refs.append(evidence)
        if refs:
            eligible += 1
        for k in (1,3,8):
            score = max((len({by_id[i].text for i in row['ranks'][:k]} & r)/len(r) for r in refs), default=None)
            stored = row['metrics']['best_text_recall']
            assert (score is None and stored is None) or abs(score-stored[str(k)]) < 1e-12
            if score is not None:
                totals[str(k)] += score
                baseline_score = max(len({by_id[i].text for i in source['legacy']['ranks'][:k]} & r)/len(r) for r in refs)
                baseline_totals[str(k)] += baseline_score
    scores = {k:v/eligible for k,v in totals.items()}
    summary = read(directory/'summary.json')
    assert summary['provider_attempts'] == attempts and summary['errors'] == errors and summary['tokens'] == tokens
    assert all(abs(v-summary['recall']['llm'][k]) < 1e-12 for k,v in scores.items())
    assert all(abs(v/eligible-summary['recall']['baseline'][k]) < 1e-12 for k,v in baseline_totals.items())
    report = dict(queries=len(manifest['question_ids']), eligible=eligible, recall=scores,
        attempts=attempts, errors=errors, tokens=tokens,
        input_matches_question_and_original_passages_only=True, opaque_id_mapping_verified=True,
        original_top8_paragraphs_only=True, independent_set_scoring_matches=True)
    with (directory/'independent-audit.json').open('x',encoding='utf-8') as out:
        json.dump(report,out,indent=2)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','candidates','directory'):
        p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.dataset,args.candidates,args.directory),indent=2))
