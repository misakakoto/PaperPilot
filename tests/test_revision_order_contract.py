import json
import re
from copy import deepcopy
import pytest

from paperpilot_agentic.reader_recovery import plan_evidence_reload, revision_feedback
from paperpilot_agentic.relation_review import review_targets, apply_relation_review
from test_reader_recovery import Model


@pytest.mark.parametrize('legacy', [False, True])
def test_actual_recovery_json_example_contains_the_required_revision_basis(legacy):
    seen = []

    class Capture(Model):
        def complete(self, messages, **kwargs):
            seen.append(messages)
            return super().complete(messages, **kwargs)

    source={'evidence_id':'a','source_uri':'paper://p/a','content':'Grouping alternates positions.'}
    basis = ({'aspect':'Missing grouping detail','evidence_id':'a','quote':source['content']} if legacy
             else {'aspect_id':'g0','span_ids':['s0']})
    model=Capture({'action':'REVISE','reason':'Known original can support a rewrite',
        'revision_basis':[basis]})
    result=plan_evidence_reload('Explain grouping.',{'limitations':['Missing grouping detail']},[source],{'a'},model)
    assert result['action']=='REVISE'
    example=json.loads(re.search(r'Return JSON (\{.+?\})\.',seen[0][0]['content']).group(1))
    assert 'revision_basis' in example
    assert set(example['revision_basis'][0]) == {'aspect_id','span_ids'}
    assert result['revision_basis'][0]['quote'] == source['content']


def test_guard_rejection_is_explicit_in_feedback_without_changing_the_source_verdict():
    report={'draft':{'facts':[]},'audit':{'assessments':[
        {'fact_id':'i','verdict':'UNCERTAIN','reason':'Original model approved the comparison',
         'boundary_reason':'paper_absence_not_established'}]},
        'answer_boundaries':{'changed_fact_ids':['i']}}
    before=deepcopy(report)
    feedback=revision_feedback(report,[])
    target=feedback['host_rejection_targets'][0]
    assert target['fact_id']=='i' and target['guard']=='paper_absence_not_established'
    assert target['requires_independent_reaudit'] is True
    assert 'positive operations' in target['rewrite_constraint']
    assert report==before and feedback['audit']['assessments'][0]['verdict']=='UNCERTAIN'


def test_sequence_target_does_not_let_a_true_containment_clause_approve_reversed_order():
    draft={'facts':[{'id':'m','kind':'SOURCE_FACT','text':'Beta重排token，随后输入Alpha模块。',
                     'evidence_ids':['e']}]}
    targets=review_targets(draft)
    assert {'target_id':'m:processing_order','fact_id':'m','dimension':'processing_order'} in targets
    audit={'assessments':[{'fact_id':'m','verdict':'SUPPORTED','relevance':'DIRECT'}],
        'coverage':'COMPLETE','relation_checks':[{'target_id':'m:processing_order',
            'span_ids':['s0'],'verdict':'UNSUPPORTED','reason':'Beta is inside Alpha, not before Alpha.'}]}
    spans=[{'span_id':'s0','evidence_id':'e','text':'Alpha模块包含Beta重排器。'}]
    checked,_=apply_relation_review(draft,audit,spans,enabled=True)
    assert checked['assessments'][0]['verdict']=='UNSUPPORTED'
    assert checked['coverage']=='PARTIAL'
