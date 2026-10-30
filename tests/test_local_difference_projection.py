"""A host-created difference endpoint is not a model-created patch addition."""
from copy import deepcopy

import pytest

from paperpilot_agentic.answer_isolation import merge_repair
from paperpilot_agentic.table_projection import project_table_facts
from test_table_fact_projection import rows


def setup_patch():
    evidence = [{'evidence_id': 't', 'content': 'Table IV source.', 'source_uri': 'paper://p/table'}]
    original = {'facts': [
        {'id': 'keep', 'kind': 'SOURCE_FACT', 'text': 'A retained source fact.',
         'evidence_ids': ['t'], 'depends_on': []},
        {'id': 'delta', 'kind': 'INFERENCE', 'text': 'A requested difference.',
         'evidence_ids': ['t'], 'depends_on': ['keep']},
        {'id': 'interpretation', 'kind': 'INFERENCE', 'text': 'A scoped comparison.',
         'evidence_ids': ['t'], 'depends_on': ['keep']}],
        'direct_answer_id': 'interpretation', 'limitations': []}
    patch = {'facts': [
        {'id': 'delta', 'kind': 'TABLE_FACT', 'operation': 'difference', 'row_refs': [
            {'evidence_id': 't', 'row_id': 'r0', 'metrics': ['F1 (%)']},
            {'evidence_id': 't', 'row_id': 'r1', 'metrics': ['F1 (%)']}]},
        {'id': 'interpretation', 'kind': 'INFERENCE', 'text': 'The modified configuration scores higher.',
         'evidence_ids': ['t'], 'depends_on': ['keep', 'delta']}], 'limitations': []}
    return evidence, original, patch


def merge(projected, trace, original, evidence):
    return merge_repair(original, projected,
        {'quarantined_fact_ids': ['delta', 'interpretation']}, evidence,
        question='Compare scores.', request_history=[], table_rows=rows(), premise_ids=['keep'],
        table_projection=trace)


def test_host_endpoints_survive_local_merge_and_keep_original_fact_unchanged():
    evidence, original, patch = setup_patch()
    before = deepcopy(original)
    projected, trace = project_table_facts(patch, rows(), chinese=True)
    merged = merge(projected, trace, original, evidence)
    assert original == before
    assert merged['facts'][0] == original['facts'][0]
    by_id = {f['id']: f for f in merged['facts']}
    assert '1.10 percentage points' in by_id['delta']['text']
    assert by_id['interpretation']['depends_on'] == ['keep', 'delta_endpoint_0', 'delta_endpoint_1']
    assert 'audit' not in merged


@pytest.mark.parametrize('mutation', ['extra_fact', 'endpoint_text', 'forged_trace', 'original_collision'])
def test_only_recomputed_host_endpoints_are_allowed(mutation):
    evidence, original, patch = setup_patch()
    projected, trace = project_table_facts(patch, rows(), chinese=True)
    if mutation == 'extra_fact':
        projected['facts'].append({'id': 'invented', 'kind': 'SOURCE_FACT', 'text': 'Unsupported addition',
                                   'evidence_ids': ['t'], 'depends_on': []})
    elif mutation == 'endpoint_text':
        projected['facts'][0]['text'] = 'Invented measurement'
    elif mutation == 'forged_trace':
        trace[0]['raw_selection']['row_refs'][0]['row_id'] = 'unknown'
    else:
        original['facts'].append(deepcopy(projected['facts'][0]))
    with pytest.raises(ValueError):
        merge(projected, trace, original, evidence)
