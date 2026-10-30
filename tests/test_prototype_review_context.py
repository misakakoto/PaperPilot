import copy

from paperpilot_agentic.prototype_workspace import complete
from test_prototype_literature_review import reviewed, judgment
from test_prototype_literature import work


def test_review_view_references_exact_full_page_without_losing_qualifiers():
    from paperpilot_agentic.prototype_review_context import project
    text = 'Existing method. Future work will introduce reinforcement learning. Further discussion.'
    packet = {'page_contexts': {'page': {'text': text}}, 'findings': [{
        'page_context_ref': 'page', 'quote': 'reinforcement learning',
        'surrounding_text': 'Future work will introduce reinforcement learning.',
        'interpretation': 'An unverified proposal'}], 'extra': {'arbitrary_claim': 'Retain me'}}
    before = copy.deepcopy(packet)
    view = project(packet)
    assert packet == before and view['page_contexts'] == before['page_contexts']
    finding = view['findings'][0]
    ref = finding['surrounding_text_ref']
    assert text[ref['start']:ref['end']] == before['findings'][0]['surrounding_text']
    assert finding['quote'] == before['findings'][0]['quote']
    assert view['extra'] == before['extra']


def test_evidence_repair_stage_keeps_repair_budget_after_new_reading(tmp_path, monkeypatch):
    _, h, rid, state, _ = reviewed(tmp_path, monkeypatch)
    state['literature_repair_budget_policy'] = 'evidence-repair-budget-v1'
    assert h._decision_reserve_class(state) == 'general'
    content = judgment(state, supported=True)
    content['coverage'].update(adequate_for_design=False, reason='Missing methods')
    work(h, rid, state, 'claims', content)
    state['evidence_replanning_policy'] = 'evidence-gap-extension-v2'
    h._apply(rid, state, {'action': 'gather_evidence', 'reason': 'Need a new observation', 'arguments': {
        'analysis_task_id': 'analysis', 'tasks': [{'id': 'more_pages', 'goal': 'Read methods',
            'capability': 'read_literature', 'depends_on': ['search'], 'acceptance': 'Original pages'}]}}, 6)
    complete(state['work'], 'more_pages', 'read_literature', {'ok': True, 'evidence': []})
    assert h._decision_reserve_class(state) == 'repair'
    state.pop('literature_repair_budget_policy')
    assert h._decision_reserve_class(state) == 'general'
