"""Analysis continuation keeps original training/publications and rejects new authority."""
from paperpilot_agentic.research_analysis_continuation import continue_analysis
from paperpilot_agentic.research_training import episode_folder,gpu_ledger
from test_research_harness import harness,start,action,analysis


def partial(h,provider):
    # Reproduce the old published PARTIAL state; the current harness now rejects
    # this legacy premature stop. Only this fixture retains the old behavior.
    current_apply = h._apply
    def legacy_apply(run_id, state, value, index):
        if value['action'] == 'stop':
            h._finish(run_id, state, value['arguments']['summary'], complete=False)
            return {'action': 'stop', 'status': 'OK', 'fixture': 'legacy premature stop'}
        return current_apply(run_id, state, value, index)
    h._apply = legacy_apply
    provider.responses[3:]=[action('stop',summary='Synthetic premature stop')]
    rid=start(h)
    waiting=h.advance(rid)
    h.runtime.approve(waiting['pending_approvals'][0]['approval_id'])
    result=h.advance(rid)
    h._apply = current_apply
    assert result['research_episode']['phase']=='PARTIAL'
    return rid


def test_missing_analysis_continues_without_training_budget_reset_or_rewriting_report(harness):
    h,provider,launches=harness
    rid=partial(h,provider)
    original=episode_folder(h.runtime,rid)/'report.json'
    before=original.read_bytes()
    ledger=gpu_ledger(h.runtime,rid)
    token,_=ledger.reserve('fixture-spent',12)
    ledger.finish(token,'COMPLETED',12)
    budget=ledger.snapshot()
    provider.responses.extend([analysis(),action('stop',summary='Synthetic analyzed stop')])
    result=continue_analysis(h.runtime,rid)
    assert result['research_episode']['completed']
    assert original.read_bytes()==before and ledger.snapshot()==budget
    assert len(launches)==1 and len(provider.messages)==6
    assert len(list(original.parent.glob('report-analysis-continuation-*.json')))==1
    continue_analysis(h.runtime,rid)
    assert len(launches)==1 and len(provider.messages)==6


def test_analysis_continuation_rejects_new_candidate_actions(harness):
    h,provider,launches=harness
    rid=partial(h,provider)
    provider.responses.extend([action('request_experiment'),analysis(),
                               action('delegate_patch',proposal={}),action('stop',summary='Safe stop')])
    result=continue_analysis(h.runtime,rid)
    assert result['research_episode']['completed'] and len(launches)==1
    assert len(result['approvals'])==1
    assert any('Analysis-only' in (e['payload'].get('error') or '') for e in result['events'])
