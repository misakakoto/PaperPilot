from paperpilot_agentic.prototype_workspace import complete, plan
from test_prototype_completion import finished


def test_review_and_finish_use_synthesis_reserve_without_changing_budget():
    harness, state, rows = finished()
    assert harness._decision_reserve_class(state) == 'synthesis'
    rows[-1]['goal'] = 'new review'
    plan(state['work'], rows)
    assert harness._decision_reserve_class(state) == 'synthesis'


def test_revised_design_and_implementation_use_repair_reserve():
    harness, state, rows = finished()
    state['work']['repairs'] = 1
    rows[1]['goal'] = 'repair design after execution'
    plan(state['work'], rows)
    assert harness._decision_reserve_class(state) == 'repair'
    complete(state['work'], 'design', 'design', {'revised': True})
    assert harness._decision_reserve_class(state) == 'repair'


def test_initial_implementation_keeps_later_reserves_protected():
    harness, state, rows = finished()
    rows[2]['goal'] = 'initial candidate'
    plan(state['work'], rows)
    assert harness._decision_reserve_class(state) == 'general'


def test_repaired_candidate_validation_can_use_repair_reserve():
    harness, state, rows = finished()
    state['work']['repairs'] = 2
    rows[3]['goal'] = 'validate repaired candidate'
    plan(state['work'], rows)
    assert harness._decision_reserve_class(state) == 'repair'
