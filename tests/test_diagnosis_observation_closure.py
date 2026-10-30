import copy

import pytest

from paperpilot_agentic.research_diagnosis import DiagnosisHarness


def case(protocol="metric-facts-v3"):
    delta={"precision":0.1,"recall":-0.05,"f1":-0.01,"iou":-0.02}
    state={"report_protocol":protocol,"observations":{"paired":{
        "rows":[{"seed":37,"delta":delta}],
        "facts":{f"paired/0/delta/{k}":v for k,v in delta.items()}}}}
    ids=["pattern/paired/0/f1","pattern/paired/0/precision_recall"]
    args={"observation_ids":ids,"hypotheses":[{
        "text":"A possible scale effect needs an activation measurement.",
        "observation_ids":["pattern/paired/0/iou"],"alternatives":["Optimization interaction"]}],
        "limitations":["No activation observation"],"next_experiment":{
            "decision":"PROPOSE_EXPERIMENT","hypothesis":"Test scale interaction",
            "support_refs":["pattern/paired/0/iou"],"counter_refs":["pattern/paired/0/f1"],
            "missing_observations":["Activations"],"falsification":"No response under a controlled scale intervention",
            "intervention":"Paired scale intervention","controls":["Identical inputs"],
            "measurements":["Activations"],"expected_patterns":["Response versus no response"],
            "alternatives_remaining":["Optimization differences"]}}
    return object.__new__(DiagnosisHarness),state,args


def test_read_hypothesis_reference_is_projected_even_if_not_repeated_in_top_selection():
    harness,state,args=case()
    original=copy.deepcopy((state,args))
    converted,extra=harness._prepare_v2_report(state,args)
    ref="pattern/paired/0/iou"
    assert extra["observations"][ref]["direction"]=="DOWN"
    assert "paired/0/delta/iou" in converted["fact_refs"]
    record=extra["projection_record"]
    assert record["submitted_observation_ids"]==args["observation_ids"]
    assert record["auto_included_hypothesis_ids"]==[ref]
    assert ref in record["selected_ids"]
    assert extra["hypotheses"]==args["hypotheses"]
    assert (state,args)==original  # Original model selection and source are not rewritten.


@pytest.mark.parametrize("ref",["pattern/paired/9/iou","pattern/coverage/0/iou"])
def test_invented_or_unread_hypothesis_reference_still_fails(ref):
    harness,state,args=case()
    args["hypotheses"][0]["observation_ids"]=[ref]
    with pytest.raises(ValueError,match="hypothesis.observation_ids"):
        harness._prepare_v2_report(state,args)


def test_required_observations_cannot_be_omitted_and_legacy_subset_contract_is_unchanged():
    harness,state,args=case()
    args["observation_ids"]=["pattern/paired/0/f1"]
    with pytest.raises(ValueError,match="every required"):
        harness._prepare_v2_report(state,args)
    harness,state,args=case("metric-facts-v2")
    with pytest.raises(ValueError,match="hypothesis.observation_ids"):
        harness._prepare_v2_report(state,args)


def test_legacy_projection_keeps_submitted_selection_record_including_duplicates():
    harness,state,args=case("metric-facts-v2")
    args["observation_ids"] += ["pattern/paired/0/iou","pattern/paired/0/iou"]
    _,extra=harness._prepare_v2_report(state,args)
    assert extra["projection_record"]["selected_ids"]==args["observation_ids"]
    assert "auto_included_hypothesis_ids" not in extra["projection_record"]


def test_runtime_projects_valid_extra_metric_without_model_repair_or_execution(tmp_path):
    import json
    from paperpilot_agentic.providers import LLMResponse
    from test_diagnosis_content_review import revised_provider
    from test_imported_diagnosis_harness import setup

    _,harness,reader,binding=setup(tmp_path)
    normal=revised_provider(reader)
    def complete(messages,**kwargs):
        response=normal(messages,**kwargs)
        value=json.loads(response.text)
        if value['action']=='report':
            extra='pattern/paired/0/iou'
            args=value['arguments']
            args['observation_ids']=[i for i in args['observation_ids'] if i!=extra]
            args['hypotheses'][0]['observation_ids'].append(extra)
            return LLMResponse(text=json.dumps(value),model=reader.model,finish_reason='stop',
                               prompt_tokens=10,completion_tokens=10)
        return response
    reader.complete=complete
    rid=harness.start_imported_diagnosis(binding['bundle_id'],'extra-read-observation',
        comparison_mode='summary',report_protocol='metric-facts-v3')['run']['run_id']
    result=harness.advance(rid)
    episode=result['research_episode']
    assert episode['phase']=='COMPLETED'
    assert episode['turn']==3
    assert episode['analysis']['projection_record']['auto_included_hypothesis_ids']==['pattern/paired/0/iou']
    assert 'iou' in result['final_answer']['answer']
    assert episode['new_gpu_starts']==0
    calls=len(reader.messages)
    harness.advance(rid)
    assert len(reader.messages)==calls
