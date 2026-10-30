"""Score frozen internal holdout outputs; never invokes a model or rewrites answers."""
import argparse
import json
from pathlib import Path
from .workbench_acceptance import score, save, digest


def paired(dataset, protocol, workspace, evaluator):
    manifest = json.loads(protocol.read_bytes())
    reports = {}
    for arm in ('baseline', 'candidate'):
        report = score(dataset, protocol, workspace/arm, evaluator)
        report['scope'] = 'QASPER train internal holdout, given-paper text-only, four first-annotation types × 10; not official test performance'
        report['official_test_set'] = False
        models, elapsed, statuses = set(), 0, {}
        for i, row in enumerate(report['rows']):
            folder = workspace/arm/f'case-{i:02}'
            info = folder/'case.json'
            if info.exists():
                data = json.loads(info.read_bytes())
                row['elapsed_seconds'] = data['elapsed_seconds']
                row['actual_models'] = data['actual_models']
                row['embedding_identity'] = data['embedding_identity']
                row['reranker_sha256'] = data['reranker_sha256']
                if data['source_sha256'] != manifest['roots'][arm]['sha256']:
                    raise ValueError('An evaluated source fingerprint differs from its frozen arm')
                if data['question_id'] != row['question_id'] or data['question'] != row['question']:
                    raise ValueError('A case question differs from the frozen sample')
                models.update(data['actual_models'])
            else:
                failure = folder/'failure.json'
                row['elapsed_seconds'] = json.loads(failure.read_bytes())['elapsed_seconds'] if failure.exists() else 0
            elapsed += row['elapsed_seconds']
            statuses[row['run_status']] = statuses.get(row['run_status'], 0) + 1
        report.update(actual_models=sorted(models), total_case_elapsed_seconds=elapsed,
                      mean_case_elapsed_seconds=elapsed/len(report['rows']), statuses=statuses)
        report['by_type'] = {kind: {metric: sum(r[metric] for r in report['rows'] if r['answer_type']==kind)/10
            for metric in report['metrics']} for kind in ('extractive','abstractive','boolean','unanswerable')}
        reports[arm] = report
    return {'kind':manifest['kind'], 'official_test_set':False,
        'protocol_sha256':digest(protocol), 'reports':reports,
        'delta_percentage_points':{k:100*(reports['candidate']['metrics'][k]-reports['baseline']['metrics'][k])
                                   for k in reports['baseline']['metrics']},
        'limitations':[
            '40 internal holdout questions, not a complete benchmark or a proof of factual accuracy.',
            'Final production prose is unchanged; lengthy valid explanations can score poorly on token overlap.',
            'Text evidence is mapped to complete cited paragraphs; this does not measure visual table understanding.',
            'Four concurrent isolated case processes; end-to-end timings include indexing and local contention.',
            'No verified monetary prices are configured; zero reported currency is not zero cost.',
        ]}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','protocol','workspace','evaluator','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    result=paired(a.dataset,a.protocol,a.workspace,a.evaluator)
    save(a.output,result)
    print(json.dumps({arm:{k:r[k] for k in ('metrics','failures','statuses','total_calls','total_tokens','actual_models','mean_case_elapsed_seconds')}
                      for arm,r in result['reports'].items()},ensure_ascii=False))
