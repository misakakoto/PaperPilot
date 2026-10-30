import copy

import pytest

from evals.structured_pdf import quality_summary
from paperpilot_agentic.structured_extraction import ParsedArtifact, SourceRegion, config_sha


def reviewed_case():
    source = SourceRegion(paper_sha256='paper', object_id='table1', label='Table I', page=2,
                          bbox=[10, 20, 100, 90], crop_sha256='crop')
    case = {'id': 't', 'kind': 'table', 'known_failure': True, 'source': source.model_dump()}
    artifact = ParsedArtifact(kind='table', source=source, parser='test', parser_version='1',
        config_sha256='config', parse_status='PARSED', validation_status='VALIDATED',
        column_headers=['F1'], table_rows=[{'cells': {'F1': '80.14'}}])
    record = {'status': 'PARSED', 'result': {'metadata': {'structured_extraction': artifact.model_dump()}}}
    return case, record


def reviewed_summary(case, record):
    review = {'verdict': 'PASS', 'result_sha256': config_sha(record), 'crop_sha256': 'crop'}
    return quality_summary([case], {'t': record}, {'t': review})


def test_a_bound_parse_and_separate_review_are_counted_without_promoting_small_set():
    case, record = reviewed_case()
    result = reviewed_summary(case, record)
    assert result['groups']['table']['passed'] == 1
    assert not result['promote_default']


@pytest.mark.parametrize('problem', ['no_artifact', 'other_source', 'failed_parse', 'structure_conflict', 'no_rows'])
def test_review_cannot_promote_wrong_or_invalid_structured_source(problem):
    case, record = reviewed_case()
    record = copy.deepcopy(record)
    artifact = record['result']['metadata']['structured_extraction']
    if problem == 'no_artifact':
        record['result']['metadata'].clear()
    elif problem == 'other_source':
        artifact['source']['paper_sha256'] = 'other'
    elif problem == 'failed_parse':
        artifact['parse_status'] = 'FAILED'
    elif problem == 'structure_conflict':
        artifact['validation_status'] = 'CONFLICT'
    else:
        artifact['table_rows'] = []
    assert reviewed_summary(case, record)['groups']['table']['passed'] == 0


@pytest.mark.parametrize('include_known_failures', [False, True])
def test_complete_protocol_requires_known_failures_even_when_all_reviews_pass(include_known_failures):
    cases, records, reviews = [], {}, {}
    for kind, known_count in [('table', 2), ('equation', 1)]:
        for index in range(10):
            cid = f'{kind}-{index}'
            source = SourceRegion(paper_sha256='paper', object_id=cid, label=cid,
                                  page=2, bbox=[10, 20, 100, 90], crop_sha256=cid)
            case = {'id': cid, 'kind': kind, 'source': source.model_dump(),
                    'known_failure': include_known_failures and index < known_count}
            fields = ({'column_headers': ['F1'], 'table_rows': [{'cells': {'F1': '80.14'}}]}
                      if kind == 'table' else {'latex': r'h(t)=Ax(t)'})
            artifact = ParsedArtifact(kind=kind, source=source, parser='test', parser_version='1',
                config_sha256='config', parse_status='PARSED', validation_status='VALIDATED', **fields)
            record = {'status': 'PARSED', 'result': {'metadata': {'structured_extraction': artifact.model_dump()}}}
            cases.append(case)
            records[cid] = record
            reviews[cid] = {'verdict': 'PASS', 'result_sha256': config_sha(record), 'crop_sha256': cid}
    summary = quality_summary(cases, records, reviews)
    assert all(group['passed'] == 10 for group in summary['groups'].values())
    assert summary['promote_default'] is include_known_failures
