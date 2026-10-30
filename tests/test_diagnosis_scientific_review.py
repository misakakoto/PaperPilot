import copy
import json
import pytest

from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.research_diagnosis import DiagnosisHarness
from test_diagnosis_content_review import revised_provider
from test_imported_diagnosis_harness import setup


def checks(issue=False):
    return [dict(aspect=aspect, status='NEEDS_REVISION' if issue and aspect=='source_scope' else 'SUPPORTED',
        statement='The proposed explanation contradicts identical validation IDs' if issue else 'Bounded hypothesis, not a measured cause',
        basis_refs=['scope/validation_ids', 'analysis/hypotheses'])
        for aspect in ('source_scope','causal_identification','discriminating_measurement','controlled_comparison')]


def test_known_review_issue_cannot_be_accepted_and_keeps_facts(tmp_path):
    _, facade, reader, binding=setup(tmp_path)
    normal=revised_provider(reader)
    seen=[]
    def complete(messages, **kwargs):
        response=normal(messages, **kwargs)
        value=json.loads(response.text)
        if value['action']=='finish':
            seen.append(value)
            value['arguments']['content_review']['checks']=checks(issue=True)
        return LLMResponse(text=json.dumps(value), model=reader.model, finish_reason='stop',prompt_tokens=10,completion_tokens=10)
    reader.complete=complete
    rid=facade.start_imported_diagnosis(binding['bundle_id'],'contradictory-review-001',comparison_mode='summary',report_protocol='metric-facts-v3')['run']['run_id']
    result=facade.advance(rid)
    state=result['research_episode']
    assert len(seen)==2
    assert state['analysis_revisions']==1
    assert state['phase']=='PARTIAL'
    assert '宿主生成的观测' in result['final_answer']['answer']
    assert state['content_review']['submitted_verdict']=='ACCEPT_TENTATIVE'
    assert state['content_review']['verdict']=='NEEDS_REVISION'
    calls=len(reader.messages)
    facade.advance(rid)
    assert len(reader.messages)==calls
    assert state['new_gpu_starts']==0


def test_actual_lead_packet_exposes_review_basis_and_matching_schema(tmp_path):
    _, facade, reader, binding=setup(tmp_path)
    reader.complete=revised_provider(reader)
    rid=facade.start_imported_diagnosis(binding['bundle_id'],'review-packet-001',comparison_mode='summary',report_protocol='metric-facts-v3')['run']['run_id']
    facade.advance(rid,max_turns=2)
    harness=DiagnosisHarness(facade.runtime)
    state=harness._load(rid)
    original=copy.deepcopy(state)
    _,messages=harness._context(rid,state)
    packet=json.loads(messages[1]['content'])
    context=packet['scientific_review_context']
    assert context['basis']['scope/validation_ids']['status']=='IDENTICAL_ACROSS_SEEDS'
    assert context['basis']['scope/coverage_definition']['status']=='LABEL_POSITIVE_PIXEL_FRACTION_NOT_OBJECT_SIZE'
    assert context['basis']['scope/training_assurance']==state['verification']['training_source']
    assert context['basis']['analysis/experiment_contract']==state['analysis']['experiment_contract']
    schema=packet['schema']['finish']['arguments']
    assert 'checks' in schema['$defs']['AuditedContentReview']['required']
    assert state==original


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'invented_basis'])
def test_new_review_rejects_missing_coverage_and_invented_basis(fault):
    from paperpilot_agentic.research_contracts import parse_diagnosis_action
    state={'content_review_protocol':'scientific-review-v2','analysis':{'hypotheses':[]},
           'comparison_mode':'summary','observations':{'paired':{}}}
    review={'verdict':'ACCEPT_TENTATIVE','issues':[],'rationale':'Bounded review','checks':checks()}
    if fault=='missing':
        review.pop('checks')
    elif fault=='duplicate':
        review['checks'][1]['aspect']='source_scope'
    else:
        review['checks'][0]['basis_refs']=['scope/invented']
    with pytest.raises(ValueError):
        parse_diagnosis_action('lead',{'action':'finish','reason':'Review complete','arguments':{
            'decision':'STOP','rationale':'No new authority','content_review':review}},state)


def test_legacy_review_is_readable_and_issues_override_acceptance_without_mutation():
    from paperpilot_agentic.research_contracts import parse_diagnosis_action
    from paperpilot_agentic.research_review import apply_content_review
    review={'verdict':'ACCEPT_TENTATIVE','issues':['Control condition contradicts the proposed intervention'],
            'rationale':'An issue was identified'}
    value={'action':'finish','reason':'Review done','arguments':{
        'decision':'STOP','rationale':'No execution','content_review':review}}
    assert parse_diagnosis_action('lead',value)['arguments']['content_review']==review
    state={'analysis_revisions':0}
    original=copy.deepcopy(review)
    assert apply_content_review(state,review,'STOP')=='REVISE'
    assert review==original
    assert state['content_review']['submitted_verdict']=='ACCEPT_TENTATIVE'
    assert state['content_review']['verdict']=='NEEDS_REVISION'

@pytest.mark.parametrize('ref,valid', [('pattern/paired/0/f1', True), ('paired/0/delta/f1', True), ('pattern/coverage/0/f1', False), ('pattern/paired/999/f1', False)])
def test_review_accepts_read_host_observations_but_not_unread_or_invented(tmp_path, ref, valid):
    from paperpilot_agentic.research_contracts import parse_diagnosis_action
    _,facade,reader,binding=setup(tmp_path)
    reader.complete=revised_provider(reader)
    rid=facade.start_imported_diagnosis(binding['bundle_id'],'review-read-refs-001',comparison_mode='summary',report_protocol='metric-facts-v3')['run']['run_id']
    facade.advance(rid,max_turns=2)
    state=DiagnosisHarness(facade.runtime)._load(rid)
    review={'verdict':'ACCEPT_TENTATIVE','issues':[],'rationale':'Bounded hypothesis','checks':checks()}
    review['checks'][1]['basis_refs']=[ref]
    value={'action':'finish','reason':'Review supplied source','arguments':{
        'decision':'STOP','rationale':'No execution','content_review':review}}
    if valid:
        assert parse_diagnosis_action('lead',value,state)['arguments']['content_review']['checks'][1]['basis_refs']==[ref]
    else:
        with pytest.raises(ValueError,match='basis ID'):
            parse_diagnosis_action('lead',value,state)
