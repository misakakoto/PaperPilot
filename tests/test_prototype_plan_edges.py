import copy

import pytest

from paperpilot_agentic.prototype_plan_edges import elaborate, POLICY
from paperpilot_agentic.prototype_workspace import new_workspace, plan


def fixture():
    rows = [{'id': k, 'goal': k, 'capability': c, 'depends_on': d, 'acceptance': 'fixed'}
            for k, c, d in [('source', 'read', []), ('search', 'search_literature', []),
                           ('pages', 'read_literature', ['search']),
                           ('analysis', 'analyze_literature', ['source', 'pages']),
                           ('claims', 'review_literature', ['analysis']),
                           ('design', 'design', ['analysis', 'claims']),
                           ('code', 'implement', ['design', 'source']),
                           ('check', 'validate', ['code']), ('review', 'review', ['design', 'code', 'check'])]]
    return {'work': new_workspace('research'), 'plan_edge_policy': POLICY,
            'implementation_source_ids': ['baseline'], 'literature_analysis_policy': 'source-spans-analysis-v1'}, rows


def ancestors(rows, key):
    tasks = {t['id']: t for t in rows}
    seen, queue = set(), list(tasks[key]['depends_on'])
    while queue:
        p = queue.pop()
        if p not in seen:
            seen.add(p)
            queue.extend(tasks[p]['depends_on'])
    return seen


def test_elaboration_only_adds_preexisting_transitive_inputs_and_is_idempotent():
    state, rows = fixture()
    original = copy.deepcopy(rows)
    compiled, receipt = elaborate(state, rows)
    assert rows == original and len(compiled) == len(rows)
    for a, b in zip(rows, compiled):
        assert {k: v for k, v in a.items() if k != 'depends_on'} == {
            k: v for k, v in b.items() if k != 'depends_on'}
        assert ancestors(rows, a['id']) == ancestors(compiled, a['id'])
    assert 'source' in next(t for t in compiled if t['id'] == 'design')['depends_on']
    assert 'source' in next(t for t in compiled if t['id'] == 'review')['depends_on']
    assert receipt['added_edges']
    assert elaborate(state, compiled)[0] == compiled
    assert not elaborate(state, compiled)[1]['added_edges']
    plan(new_workspace('validated'), compiled, enforce_capabilities=True)


def test_compiler_can_expose_analysis_through_an_existing_review_but_invents_no_tasks():
    state, rows = fixture()
    next(t for t in rows if t['id'] == 'design')['depends_on'] = ['source', 'claims']
    compiled, _ = elaborate(state, rows)
    assert 'analysis' in next(t for t in compiled if t['id'] == 'design')['depends_on']
    # Missing real literature work remains missing and the normal validator rejects it.
    from paperpilot_agentic.prototype_analysis import check_plan
    next(t for t in rows if t['id'] == 'analysis')['depends_on'] = ['source']
    compiled, _ = elaborate(state, rows)
    with pytest.raises(ValueError, match='direct literature'):
        check_plan(compiled)


def test_legacy_missing_and_cyclic_proposals_are_not_silently_rewritten():
    state, rows = fixture()
    state.pop('plan_edge_policy')
    assert elaborate(state, rows) == (rows, None)
    state['plan_edge_policy'] = POLICY
    rows[0]['depends_on'] = ['review']
    with pytest.raises(ValueError, match='cycle'):
        elaborate(state, rows)
