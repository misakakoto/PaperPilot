import copy
from paperpilot_agentic import prototype_design_context as dc
from test_prototype_prompt import source_artifact, SOURCE
import hashlib


def test_projection_preserves_citations_gaps_and_originals(monkeypatch):
    monkeypatch.setattr(dc, 'SOURCE_SHA', hashlib.sha256(SOURCE.encode()).hexdigest())
    raw = source_artifact()
    analysis = {'kind':'analyze_literature','content':{'findings':[{'quote_id':'q','interpretation':'hypothesis','quote':'large literal','provenance':{'page':2}}], 'gaps':['unknown effects'], 'directions':[{'mechanism':'x'}], 'extra':'retain'}}
    payload = {'active_task_id':'d','tasks':{'d':{'capability':'design'},'old':{}},'artifact_index':{'s':{},'a':{},'unused':{}},'context_artifacts':{'s':raw,'a':analysis}}
    before = copy.deepcopy(payload)
    view = dc.project(payload)
    assert payload == before
    assert view['tasks'] == {'d':{'capability':'design'}}
    assert 'unused' not in view['artifact_index']
    content = view['context_artifacts']['a']['content']
    assert content['findings'] == [{'quote_id':'q','interpretation':'hypothesis','artifact_id':'a'}]
    assert content['gaps'] == ['unknown effects'] and content['extra'] == 'retain'
    source = view['context_artifacts']['s']['content']
    assert 'class SNUNet_ECAM' in source['design_source']
    assert source['context_view']['raw_artifact_id'] == 's'
    assert 'not executable' in source['context_view']['assurance']


def test_unknown_source_and_failed_review_are_unabridged():
    raw = source_artifact()
    review = {'kind':'review_literature','content':{'passed':False,'checks':[{'reason':'wrong modality'}]}}
    payload = {'active_task_id':'d','tasks':{'d':{}},'artifact_index':{},'context_artifacts':{'s':raw,'r':review}}
    view = dc.project(payload)
    assert view['context_artifacts']['s'] == raw
    assert view['context_artifacts']['r'] == review
