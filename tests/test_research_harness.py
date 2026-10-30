"""Public harness integration; all providers/training here are labeled substitutes."""
import json

import pytest

from paperpilot_agentic import research_harness as module
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.research_assets import build_fusion_patch, fusion_region
from paperpilot_agentic.research_training import gpu_ledger
from paperpilot_agentic.workbench import ResearchWorkbench
from paperpilot_agentic.reference_methods import sha

SOURCE = '''class SNUNet_ECAM:
    def forward(self, xA, xB):
        x0_1 = self.encoder(xA)
        x0_2 = self.encoder(xB)
        x0_3 = x0_1 + x0_2
        x0_4 = x0_3
        out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)
        intra = torch.sum(torch.stack((x0_1,x0_2,x0_3,x0_4)), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1,4,1,1))
        out = self.conv_final(out)
        return (out,)
'''
FUSION = 'intra = torch.mean(torch.stack((x0_1,x0_2,x0_3,x0_4)), dim=0)\nca1 = self.ca1(intra)\nout = self.ca(out) * (out + ca1.repeat(1,4,1,1))'
PROPOSAL = {'hypothesis': 'Synthetic hypothesis; not measured research',
    'prediction': 'F1 may increase', 'falsification': 'Contrary per-seed deltas',
    'evidence_ids': ['paper'], 'code_basis': 'Fusion sum observed in fixture'}


def action(name, **args):
    if name == 'stop':
        args.setdefault('content_review',{'verdict':'ACCEPT_TENTATIVE','issues':[],
            'rationale':'Synthetic bounded report; no scientific mechanism established.'})
    return {'action': name, 'reason': 'Synthetic public decision rationale', 'arguments': args}


class Provider:
    model = 'fixture-model'
    def __init__(self, responses):
        self.responses, self.messages = list(responses), []
    def complete(self, messages, before_attempt=None, **kwargs):
        if before_attempt:
            before_attempt()
        self.messages.append(messages)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return LLMResponse(text=json.dumps(value), model=self.model, finish_reason='stop',
                           prompt_tokens=100, completion_tokens=30)


def receipt():
    rows = []
    for seed, diff in ((7, -.02), (11, .01), (19, -.03)):
        baseline = {'status': 'VERIFIED', 'initial_state_sha256': str(seed),
                    'metrics': dict.fromkeys(('precision', 'recall', 'f1', 'iou'), .5)}
        candidate = {**baseline, 'metrics': dict.fromkeys(('precision', 'recall', 'f1', 'iou'), .5+diff)}
        rows.append({'seed': seed, 'baseline': baseline, 'candidate': candidate,
                     'delta': dict.fromkeys(('precision', 'recall', 'f1', 'iou'), diff)})
    return {'status': 'TRAINING_VERIFIED', 'trials': rows,
            'fixture': 'Training substitute; not real metrics or GPU measurements'}


def analysis():
    return action('report', facts=[f'delta/{s}/f1' for s in (7, 11, 19)],
        interpretations=[{'text': 'Synthetic mixed directions; no stable improvement',
                          'fact_refs': ['delta/7/f1', 'delta/11/f1', 'delta/19/f1']}],
        limitations=['Synthetic fixture; development data; no significance'],
        next_experiment='Stop; no scientific claim from synthetic data')


@pytest.fixture
def harness(tmp_path, monkeypatch):
    root = tmp_path/'source'
    root.mkdir()
    (root/'baseline.py').write_text(SOURCE)
    recipe = {'limits': {'max_candidates': 2, 'max_patch_repairs': 2, 'max_turns': 24},
        'sources': {'papers': [{'evidence_id': 'paper', 'content': 'Synthetic source excerpt',
                              'source_uri': 'fixture://paper', 'page': None, 'provenance': {}}],
                    'references': []}}
    monkeypatch.setattr(module, 'load_recipe', lambda *_: (recipe, root, {}))
    cards = {}
    def freeze(runtime, rid, recipe_id, number, proposal, body):
        patch = build_fusion_patch(SOURCE, body)
        card = {'binding_id': sha(patch['candidate_source'].encode()), 'diff': patch['diff'], 'hypothesis': proposal,
                'static_checks': patch['static_checks'], 'requires_approval': True}
        cards[card['binding_id']] = card
        return card
    monkeypatch.setattr(module, 'freeze_candidate', freeze)
    monkeypatch.setattr(module, 'load_candidate', lambda rt, binding: (root, {}, cards[binding]))
    provider = Provider([action('delegate_patch', proposal=PROPOSAL),
        action('submit_patch', fusion_body=FUSION, review_notes=['Needs execution approval']),
        action('request_experiment'), action('delegate_analysis'), analysis(),
        action('stop', summary='Synthetic episode complete; negative/mixed results preserved')])
    runtime = ResearchAgentRuntime(workspace=tmp_path/'workspace', db_path=tmp_path/'runtime.sqlite3',
                                   llm=provider, text_reader_protocol='fixture-protocol')
    launches = []
    def training(*args):
        launches.append(args)
        return receipt()
    return module.ResearchHarness(runtime, training=training), provider, launches


def start(harness):
    return harness.start(goal='Synthetic research goal', recipe_id='a'*64,
                         request_id='fixture-request', unlimited_llm_usage=True)['run']['run_id']


def test_real_interface_waits_for_specific_candidate_and_resumes_without_replaying(harness):
    h, provider, launches = harness
    rid = start(h)
    result = h.advance(rid)
    assert result['run']['status'] == 'WAITING_APPROVAL'
    assert len(provider.messages) == 3 and launches == []
    result = h.advance(rid)
    assert len(provider.messages) == 3 and launches == []
    h.runtime.approve(result['pending_approvals'][0]['approval_id'])
    result = h.advance(rid)
    assert result['research_episode']['completed'] is True
    assert result['run']['status'] == 'COMPLETED'
    assert len(launches) == 1
    assert result['research_episode']['candidates'][0]['analysis']['facts']['delta/7/f1'] == -.02
    assert start(h) == rid
    assert h.advance(rid)['research_episode'] == result['research_episode']
    assert len(launches) == 1 and len(provider.messages) == 6
    assert h.runtime.db.get_run(rid)['used_llm_calls'] == 6


def test_verified_experiment_requires_analysis_before_stop_or_another_candidate(harness):
    h, provider, launches = harness
    provider.responses[3:3] = [
        action('stop', summary='Premature stop omitting the analyst'),
        action('delegate_patch', proposal=PROPOSAL),
    ]
    rid = start(h)
    waiting = h.advance(rid)
    h.runtime.approve(waiting['pending_approvals'][0]['approval_id'])
    result = h.advance(rid)
    assert result['research_episode']['completed'] is True
    assert len(launches) == 1 and len(result['research_episode']['candidates']) == 1
    assert len(provider.messages) == 8
    rejections = [e['payload'].get('error', '') for e in result['events']
                  if e['event_type'] == 'research.action' and e['payload'].get('status') == 'REJECTED']
    assert sum('Independent analyst report required' in error for error in rejections) == 2
    task = json.loads(next(m['content'] for m in provider.messages[3]
                           if m['role'] == 'user' and 'workflow_contract' in m['content']))
    assert 'delegate_analysis' in task['workflow_contract']


def test_lead_may_stop_before_experiment_without_claiming_completion(harness):
    h, provider, launches = harness
    provider.responses = [action('stop', summary='Insufficient source evidence; no experiment attempted')]
    result = h.advance(start(h))
    assert result['research_episode']['phase'] == 'PARTIAL'
    assert not result['research_episode']['completed'] and launches == []


def test_role_contexts_receive_handoffs_not_other_role_transcripts(harness):
    h, provider, _ = harness
    rid = start(h)
    result = h.advance(rid)
    h.runtime.approve(result['pending_approvals'][0]['approval_id'])
    h.advance(rid)
    engineer = provider.messages[1]
    analyst = provider.messages[4]
    assert not any(m['role'] == 'assistant' for m in engineer)
    assert not any(m['role'] == 'assistant' for m in analyst)
    assert 'fusion_body' in engineer[-1]['content']
    assert 'metric_catalog' in analyst[-1]['content']
    assert 'delegate_patch' not in engineer[0]['content']


def test_lead_can_request_bounded_revision_before_execution_approval(harness):
    h, provider, launches = harness
    provider.responses = [action('delegate_patch', proposal=PROPOSAL),
        action('submit_patch', fusion_body=FUSION, review_notes=['Needs review']),
        action('delegate_patch', revision_feedback='Bound residual amplitude before approval'),
        action('submit_patch', fusion_body=FUSION+' * 0.5', review_notes=['Revised, still unexecuted']),
        action('request_experiment')]
    rid = start(h)
    waiting = h.advance(rid)
    assert launches == [] and len(waiting['pending_approvals']) == 1
    candidate = waiting['research_episode']['candidates'][0]
    assert candidate['repairs'] == 1 and len(candidate['draft_history']) == 1
    assert candidate['card']['binding_id'] != candidate['draft_history'][0]['binding_id']
    assert waiting['pending_approvals'][0]['request']['context'] == candidate['card']


def test_original_sources_are_available_as_owned_preview_artifacts(harness):
    h, _, _ = harness
    rid = start(h)
    artifacts = h.runtime.db.list_artifacts(rid)
    assert len(artifacts) == 1 and artifacts[0]['metadata']['source_id'] == 'paper'
    class Background:
        def active(self, _): return False
    wb = ResearchWorkbench(h.runtime, Background())
    preview = wb.artifact(rid, artifacts[0]['artifact_id'])
    assert json.loads(preview['content'])['content'] == 'Synthetic source excerpt'


def test_lead_receives_actual_editable_code_and_short_owned_source_handles(harness):
    h, provider, _ = harness
    rid = start(h)
    h.advance(rid, max_turns=1)
    task = json.loads(provider.messages[0][-1]['content'])
    assert 'self.ca1(intra)' in task['editable_code']
    assert 'self.conv_final(out)' in task['editable_code']
    assert task['sources'][0]['source_ref'] == 'S1'
    assert 'No imports, branches, loops' in task['executable_contract']


def test_short_source_handles_are_canonicalized_without_changing_raw_response(harness):
    h, provider, _ = harness
    provider.responses = [action('read_source', source_ref='S1'),
        action('delegate_patch', proposal={**PROPOSAL, 'evidence_ids': ['S1']})]
    rid = start(h)
    result = h.advance(rid, max_turns=2)
    assert result['research_episode']['candidates'][0]['proposal']['evidence_ids'] == ['paper']
    with h.db.connect() as conn:
        raw = conn.execute('SELECT raw_json FROM research_episode_turns WHERE run_id=? AND turn=2', (rid,)).fetchone()[0]
    assert json.loads(json.loads(raw)['text'])['arguments']['proposal']['evidence_ids'] == ['S1']


def test_cancel_between_model_return_and_local_action_does_not_leave_phantom_execution(harness, monkeypatch):
    h, _, launches = harness
    rid = start(h)
    apply = h._apply
    def cancel_then_apply(*args):
        h.runtime.cancel(rid)
        return apply(*args)
    monkeypatch.setattr(h, '_apply', cancel_then_apply)
    result = h.advance(rid)
    assert result['run']['status'] == result['research_episode']['phase'] == 'CANCELLED'
    assert not any(c['status'] == 'RUNNING' for c in result['tool_calls'])
    assert launches == []


def test_cancelled_training_pause_preserves_cancellation_and_known_checkpoint(harness):
    h, _, launches = harness
    rid = start(h)
    waiting = h.advance(rid)
    h.runtime.approve(waiting['pending_approvals'][0]['approval_id'])
    def pause(*_):
        launches.append(True)
        h.runtime.cancel(rid)
        return {'status': 'PAUSED', 'checkpoint': 'substitute-fixture-only'}
    h.training = pause
    result = h.advance(rid)
    assert result['run']['status'] == result['research_episode']['phase'] == 'CANCELLED'
    assert not any(c['status'] == 'RUNNING' for c in result['tool_calls'])
    assert result['research_episode']['candidates'][0]['pause_receipt']['status'] == 'PAUSED'
    h.advance(rid)
    assert launches == [True]


def test_model_cannot_approve_or_launch_from_wrong_role(harness):
    h, provider, launches = harness
    provider.responses.insert(1, action('request_experiment', command='arbitrary'))
    rid = start(h)
    result = h.advance(rid)
    assert launches == []
    assert len(result['pending_approvals']) == 1
    assert any('permission' in (e['payload'].get('error') or '') for e in result['events'])


def test_analyst_cannot_omit_unfavorable_seed_or_supply_fake_number(harness):
    h, provider, _ = harness
    provider.responses.insert(4, action('report', facts=['delta/11/f1'], interpretations=[],
        limitations=['fixture'], next_experiment='Stop'))
    rid = start(h)
    waiting = h.advance(rid)
    h.runtime.approve(waiting['pending_approvals'][0]['approval_id'])
    result = h.advance(rid)
    assert result['research_episode']['completed']
    assert len(provider.messages) == 7
    assert any('omit' in (e['payload'].get('error') or '') for e in result['events'])


def test_turn_reuses_committed_model_response_after_crash_before_action_commit(harness, monkeypatch):
    h, provider, launches = harness
    rid = start(h)
    original = h._save
    def crash(*args, **kwargs):
        if kwargs.get('applied_turn'):
            raise KeyboardInterrupt('Simulated process death')
        return original(*args, **kwargs)
    monkeypatch.setattr(h, '_save', crash)
    with pytest.raises(KeyboardInterrupt):
        h.advance(rid, max_turns=1)
    assert len(provider.messages) == 1
    monkeypatch.setattr(h, '_save', original)
    h.advance(rid, max_turns=1)
    assert len(provider.messages) == 1 and launches == []
    assert h.status(rid)['research_episode']['turn'] == 1


def test_unknown_training_outcome_stops_refresh_without_relaunch(harness):
    h, provider, launches = harness
    def unknown(*_):
        launches.append(True)
        raise module.TrainingBusy('Unknown owned GPU process')
    h.training = unknown
    rid = start(h)
    waiting = h.advance(rid)
    h.runtime.approve(waiting['pending_approvals'][0]['approval_id'])
    assert h.advance(rid)['research_episode']['phase'] == 'UNKNOWN'
    h.advance(rid)
    assert launches == [True] and len(provider.messages) == 3


def test_budget_is_new_four_hour_campaign_and_never_resets(harness):
    h, _, _ = harness
    rid = start(h)
    ledger = gpu_ledger(h.runtime, rid)
    token, _ = ledger.reserve('fixture-preflight', 40)
    ledger.finish(token, 'COMPLETED', 12)
    assert gpu_ledger(h.runtime, rid).snapshot()['cap_seconds'] == 14400
    assert h.status(rid)['research_episode']['gpu_budget']['remaining_seconds'] == 14388


def test_changed_model_protocol_blocks_before_call(harness):
    h, provider, _ = harness
    rid = start(h)
    h.runtime.llm_runtime.model.model = 'changed'
    result = h.advance(rid)
    assert result['research_episode']['phase'] == 'FAILED'
    assert not provider.messages


def test_research_submission_id_does_not_authorize_changed_goal(harness):
    h, _, _ = harness
    start(h)
    with pytest.raises(ValueError, match='different'):
        h.start(goal='Other', recipe_id='a'*64, request_id='fixture-request', unlimited_llm_usage=True)


def test_workbench_starts_same_harness_with_owned_ids(harness):
    h, _, _ = harness
    class Background:
        def __init__(self): self.calls = []
        def active(self, rid): return False
        def submit(self, rid, operation): self.calls.append(rid)
    background = Background()
    wb = ResearchWorkbench(h.runtime, background)
    payload = {'request_id': 'workbench-request', 'question': 'Research through public workbench',
               'action': 'research_episode', 'recipe_id': 'a'*64, 'source_ids': ['paper']}
    first = wb.submit(payload)
    second = wb.submit(payload)
    assert first['run_id'] == second['run_id']
    assert len(background.calls) == 1
    assert wb.turn(first['run_id'])['research_episode']['phase'] == 'READY'
    with pytest.raises(ValueError, match='Unknown'):
        wb.submit({**payload, 'command': 'arbitrary'})


@pytest.mark.parametrize('body', [
    "import os\nout = out", "self.ca = out\nout = out", "out = self.encoder(out)",
    "out = unknown + out", "out = torch.load(out)", "out = out.__class__",
    "out = x0_1[0]", "out = out ** 2", "out = out\nreturn out",
])
def test_fusion_patch_rejects_scope_escape(body):
    with pytest.raises(ValueError):
        build_fusion_patch(SOURCE, body)


def test_fusion_patch_preserves_feature_extractor_init_classifier_and_signature():
    patch = build_fusion_patch(SOURCE, FUSION)
    old, new = SOURCE.splitlines(), patch['candidate_source'].splitlines()
    region = fusion_region(SOURCE)
    assert old[:region[0].lineno-1] == new[:region[0].lineno-1]
    assert old[region[-1].end_lineno:] == new[-len(old[region[-1].end_lineno:]):]
    assert patch['expected_unused'] == []
    assert build_fusion_patch(SOURCE, 'out = out * 0.5')['expected_unused'] == [
        'ca.fc1.weight', 'ca.fc2.weight', 'ca1.fc1.weight', 'ca1.fc2.weight']


def test_content_review_can_revise_once_then_preserve_experiment_facts(harness):
    h, provider, launches = harness
    rid = start(h)
    waiting = h.advance(rid)
    h.runtime.approve(waiting['pending_approvals'][0]['approval_id'])
    review = {'verdict':'NEEDS_REVISION','issues':['A measured score does not establish an internal cause.'],
              'rationale':'Synthetic negative content review, not a real model judgment.'}
    provider.responses = [action('delegate_analysis'),analysis(),action('stop',summary='Needs correction',content_review=review),
                          analysis(),action('stop',summary='Still needs correction',content_review=review)]
    result = h.advance(rid)
    episode = result['research_episode']
    assert episode['phase'] == 'PARTIAL', episode
    assert episode['analysis_revisions'] == 1
    assert len(episode['candidates'][0]['analysis_history']) == 1
    assert episode['assurance']['facts'] == 'HOST_RECOMPUTED'
    assert '宿主复算事实' in result['final_answer']['answer']
    assert '推断' in result['final_answer']['limitations'][1] or result['final_answer']['review_required']
    assert len(launches) == 1, 'Training substitute must not repeat after content revision'
    count = len(provider.messages)
    h.advance(rid)
    assert len(provider.messages) == count and len(launches) == 1
