"""Deterministic binary diagnostics shared by verified and imported experiments."""
VERSION = 'binary-diagnostic-core-v1'
GROUPS = ('0', '(0,1%]', '(1%,10%]', '(10%,100%]')


def metrics(counts):
    tn, fp, fn, tp = counts
    return dict(zip(('tn', 'fp', 'fn', 'tp'), counts)) | {
        'precision': tp / (tp + fp) if tp + fp else None,
        'recall': tp / (tp + fn) if tp + fn else None,
        'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        'iou': tp / (tp + fp + fn) if tp + fp + fn else None}


def coverage_group(positives, pixels):
    return GROUPS[0 if positives == 0 else 1 if positives * 100 <= pixels
                  else 2 if positives * 10 <= pixels else 3]


def views(samples, paired):
    """Per-sample counts in, views out. No filesystem, model or training authority."""
    coverage, selected = [], []
    for pair in paired:
        seed = pair['seed']
        rows = list(samples[str(seed)].values())
        for group in GROUPS:
            members = [r for r in rows if r['group'] == group]
            item = {'seed': seed, 'group': group, 'sample_count': len(members)}
            for variant in ('baseline', 'candidate'):
                item[variant] = metrics([sum(r[variant][k] for r in members)
                                        for k in ('tn', 'fp', 'fn', 'tp')])
            item['delta'] = {k: (item['candidate'][k] - item['baseline'][k]
                if item['candidate'][k] is not None and item['baseline'][k] is not None else None)
                for k in ('precision', 'recall', 'f1', 'iou')}
            coverage.append(item)
        for variant in ('baseline', 'candidate'):
            for key in ('tn', 'fp', 'fn', 'tp'):
                if sum(r[variant][key] for r in coverage if r['seed'] == seed) != pair[variant][key]:
                    raise ValueError('Coverage counts do not reconcile with verified totals')
        for error in ('fn', 'fp'):
            ranked = sorted(rows, key=lambda r: (-(r['candidate'][error] - r['baseline'][error]), r['sample_id']))
            selected.append({'seed': seed, 'error': error,
                'selection': 'top five strictly positive candidate-minus-baseline counts; ties sample_id',
                'rows': [dict(r, extra_errors=r['candidate'][error] - r['baseline'][error])
                         for r in ranked if r['candidate'][error] > r['baseline'][error]][:5]})
    return {'paired': paired, 'coverage': coverage, 'samples': selected}
