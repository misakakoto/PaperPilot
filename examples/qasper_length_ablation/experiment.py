"""Fixed QASPER-dev length-normalization ablation; no training or gold-based tuning."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def rank(query, paragraphs, b):
    """No answers or relevance judgments enter this function."""
    counts = [Counter(tokenize(p)) for p in paragraphs]
    lengths = [sum(c.values()) for c in counts]
    average = sum(lengths) / max(1, len(lengths)) or 1
    df = Counter(t for c in counts for t in c)
    terms = set(tokenize(query))
    scores = []
    for count, length in zip(counts, lengths):
        score = 0.0
        for term in terms:
            tf = count[term]
            if tf:
                # IIR Eq. 86 log(N/df), identical between arms; only b differs.
                idf = math.log(len(counts) / df[term])
                score += idf * (tf * 2.2) / (tf + 1.2 * (1 - b + b * length / average))
        scores.append(score)
    return sorted(range(len(paragraphs)), key=lambda i: (-scores[i], i))


def references(qa):
    result = []
    for annotation in qa['answers']:
        answer = annotation['answer']
        if answer.get('unanswerable'):
            continue
        gold = {p for p in answer.get('evidence', []) if p and 'FLOAT SELECTED' not in p}
        if gold:
            result.append(gold)
    return result


def evaluate(dataset, seed):
    ids = sorted(dataset)
    random.Random(seed).shuffle(ids)
    selected = ids[:12]
    rows, excluded = [], 0
    for pid in selected:
        paper = dataset[pid]
        paragraphs = [p for section in paper['full_text'] for p in section['paragraphs'] if p]
        for qa in paper['qas']:
            gold = references(qa)
            if not gold:
                excluded += 1
                continue
            row = {'paper_id': pid, 'question_id': qa['question_id']}
            for label, b in [('baseline', 0.0), ('candidate', 0.75)]:
                try:
                    ranked = rank(qa['question'], paragraphs, b)[:3]
                    retrieved = {paragraphs[i] for i in ranked}
                    row[label] = max(len(retrieved & g) / len(g) for g in gold)
                    row[label + '_ranked_paragraphs'] = ranked
                except Exception as exc:
                    row[label] = 0.0
                    row[label + '_error'] = type(exc).__name__
            rows.append(row)
    if not rows:
        raise ValueError('No text-evidence questions in selected papers')
    baseline = sum(r['baseline'] for r in rows) / len(rows)
    candidate = sum(r['candidate'] for r in rows) / len(rows)
    return {'seed': seed, 'baseline_recall_at_3': baseline, 'candidate_recall_at_3': candidate,
            'configuration': {'k1': 1.2, 'baseline_b': 0.0, 'candidate_b': 0.75, 'top_k': 3,
                              'papers_per_seed': 12, 'training': False, 'tuning': False,
                              'idf': 'log(N/df)', 'tokenization': 'lowercase ASCII alphanumeric'},
            'improvement': candidate - baseline, 'question_count': len(rows),
            'excluded_no_text_reference': excluded, 'selected_papers': selected,
            'official_qasper_score': False, 'held_out': False,
            'sampling': '12 papers per seed; samples may overlap; no independence claim',
            'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = Path('qasper-dev-v0.3.json').read_bytes()
    metrics = evaluate(json.loads(raw), args.seed)
    metrics['dataset_sha256'] = hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(metrics, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
