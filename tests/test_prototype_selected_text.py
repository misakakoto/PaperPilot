import copy

import pytest

from paperpilot_agentic.prototype_selected_text import project
from paperpilot_agentic.prototype_analysis import spans, resolve
from paperpilot_agentic.prototype_workspace import plan, task_rows
from test_prototype_quote_navigation import source


def test_selected_text_keeps_exact_quote_ids_order_and_source_immutable():
    w, identity = source()
    artifact = w['artifacts'][identity]
    before = copy.deepcopy(artifact)
    result = project(identity, artifact)
    pairs = [pair for page in result['content']['pages'] for pair in page['quotes']]
    assert pairs == [[s['quote_id'], s['text']] for s in spans(identity, artifact)]
    assert result['content']['context_view']['complete_span_sequence'] is True
    assert artifact == before
    rows = task_rows(w)
    rows.append({'id': 'analysis', 'goal': 'interpret', 'capability': 'analyze_literature',
                 'depends_on': ['paper'], 'acceptance': 'original spans'})
    plan(w, rows)
    output = resolve(w, w['tasks']['analysis'], {'findings': [{'artifact_id': identity,
        'quote_id': pairs[0][0], 'interpretation': 'A bounded hypothesis'}], 'gaps': [],
        'directions': [{'hypothesis': 'A hypothesis', 'mechanism': 'A mechanism', 'falsification': 'A control'}]}, analysis=True)
    assert output['findings'][0]['quote'] == pairs[0][1]


def test_truncation_marks_omissions_and_never_cuts_a_quote():
    w, identity = source()
    result = project(identity, w['artifacts'][identity], max_characters=30)['content']
    assert result['context_view']['omitted_spans'] > 0
    assert result['context_view']['complete_span_sequence'] is False
    assert result['pages'][0]['quotes'][0][1] == 'Alpha method is explicit.'
    with pytest.raises(ValueError, match='identity changed'):
        changed = copy.deepcopy(w['artifacts'][identity])
        changed['content']['evidence'][0]['content'] += ' altered'
        project(identity, changed)


def test_analysis_context_loads_selected_text_without_changing_canonical_artifacts(tmp_path, monkeypatch):
    import json
    from test_prototype_analysis import prepared
    from paperpilot_agentic.prototype_selected_text import POLICY
    _,h,rid,state,identity=prepared(tmp_path,monkeypatch)
    state['analysis_read_policy']=POLICY
    before=copy.deepcopy(state['work']['artifacts'])
    _,messages=h._context(rid,state)
    packet=json.loads(messages[-1]['content'])['context_artifacts'][identity]['content']
    assert packet['context_view']['policy']==POLICY and packet['pages'][0]['quotes']
    assert 'quote_catalog' not in packet
    assert state['work']['artifacts']==before


def test_successful_navigation_is_not_repeated_when_all_its_text_is_supplied(tmp_path,monkeypatch):
    import json
    from test_prototype_analysis import prepared
    from paperpilot_agentic.prototype_selected_text import POLICY
    _,h,rid,state,identity=prepared(tmp_path,monkeypatch)
    state['analysis_read_policy']=POLICY
    state['last_observation']={'status':'OK','action':'inspect_quotes','artifact_id':identity,
        'spans':[spans(identity,state['work']['artifacts'][identity])[0]]}
    before=copy.deepcopy(state)
    _,messages=h._context(rid,state)
    assert 'spans' not in json.loads(messages[-1]['content'])['feedback']
    assert state==before
