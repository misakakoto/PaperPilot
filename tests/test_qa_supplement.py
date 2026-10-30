"""Supplemental authorization keeps old costs, new caps and source review distinct."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from evals.interview.budget import CampaignBudget, RecordedProvider, save
from evals.interview.campaign import historical_snapshot, report, REVIEW_DIMENSIONS
from evals.interview.context import context_probe_messages, semantic_dimensions
from evals.interview import run
from test_interview_campaign import Provider, MESSAGES


def test_report_keeps_unrun_cases_and_context_unmeasured_when_campaign_stops(tmp_path):
    budget = CampaignBudget(tmp_path/'receipts', {'model': 'fixed'})
    save(tmp_path/'protocol.json', {'identity': {'model': 'fixed'},
        'cases': [{'id': 'unrun', 'paper_ids': ['a'*64]}]})
    save(budget.folder/'stopped.json', {'reason': 'budget_exhausted'})
    result = report(tmp_path, budget)
    assert result['rows'][0]['status'] == 'NOT_RUN'
    assert result['rows'][0]['reason'] == 'BUDGET_EXHAUSTED'
    assert result['context']['status'] == 'NOT_RUN'
    assert result['budget']['actual_requests'] == 0


def history(folder):
    folder.mkdir()
    budget = CampaignBudget(folder/'receipts', {'model': 'fixed'})
    with budget.transaction() as conn:
        conn.execute("INSERT INTO attempts VALUES(1,'UNKNOWN',9000,0,'fixed','old',1,NULL)")
    budget.authorize_continuation({'model': 'fixed', 'revision': 'old'},
        unknown_bounds={1: 8000}, evidence={'basis': 'reviewed bound'})
    save(folder/'protocol.json', {'identity': {'model': 'fixed'}, 'cases': []})
    save(folder/'receipts'/'stopped.json', {'reason': 'old stopped'})
    return budget


def test_supplement_spends_its_own_limit_without_touching_unknown_parent(tmp_path):
    parent = history(tmp_path/'old')
    before = parent.path.read_bytes()
    old = historical_snapshot(tmp_path/'old')
    new = CampaignBudget(tmp_path/'new', {'model': 'fixed', 'campaign': 'supplement'},
                         max_requests=1, max_tokens=5000)
    provider = Provider()
    RecordedProvider(provider, new, phase='new').complete(MESSAGES, max_tokens=100)
    from paperpilot_agentic.llm_runtime import LLMBudgetExhausted
    with pytest.raises(LLMBudgetExhausted):
        RecordedProvider(provider, new, phase='limit').complete(MESSAGES, max_tokens=100)
    assert new.snapshot()['actual_requests'] == 1
    assert old['actual_requests'] == 1 and old['unsettled_upper_bound'] == 8000
    assert parent.path.read_bytes() == before and parent.snapshot()['blocked']


def test_third_campaign_keeps_all_ancestor_costs_and_unknown_without_double_count(tmp_path):
    oldest = history(tmp_path/'old')
    old = historical_snapshot(tmp_path/'old')
    old.pop('protocol')
    middle = tmp_path/'middle'
    middle.mkdir()
    middle_budget = CampaignBudget(middle/'receipts', {'model': 'fixed'})
    RecordedProvider(Provider(), middle_budget, phase='middle').complete(MESSAGES, max_tokens=100)
    save(middle/'protocol.json', {'identity': {'model': 'fixed'}, 'cases': [], 'historical_cost': old})
    before = [oldest.path.read_bytes(), middle_budget.path.read_bytes()]
    parent = historical_snapshot(middle)
    assert parent['actual_requests'] == 1  # immediate parent retains its own count
    parent.pop('protocol')
    newest = tmp_path/'newest'
    newest.mkdir()
    budget = CampaignBudget(newest/'receipts', {'model': 'fixed'})
    save(newest/'protocol.json', {'identity': {'model': 'fixed'}, 'cases': [], 'historical_cost': parent})
    result = report(newest, budget)
    assert result['aggregate_cost'] == {'actual_requests': 2, 'reported_tokens': 150,
                                       'unsettled_upper_bound': 8000, 'occupied_tokens': 8150}
    assert [oldest.path.read_bytes(), middle_budget.path.read_bytes()] == before


def test_new_supplement_does_not_reuse_the_old_budget_approval(monkeypatch, tmp_path):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'CAMPAIGN_FOLDER', tmp_path/'outputs/new')
    monkeypatch.setattr(run, 'FOLDER', tmp_path/'outputs/new')
    called = []
    monkeypatch.setattr(run, 'freeze', lambda **kwargs: called.append(kwargs))
    monkeypatch.setattr('sys.argv', ['run', '--freeze', '--campaign-dir', str(tmp_path/'outputs/new'),
        '--parent-campaign', str(tmp_path/'old'), '--max-requests', '30', '--max-tokens', '300000'])
    with pytest.raises(SystemExit):
        run.main()
    assert called == []


def test_original_approved_supplement_retains_legacy_cli_compatibility(monkeypatch, tmp_path):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'CAMPAIGN_FOLDER', tmp_path/'outputs/qa-acceptance-20261005')
    instruction = run.supplemental_instruction(tmp_path/'outputs/interview-acceptance-20261001')
    assert '30 requests / 300000 tokens' in instruction


def test_new_supplement_records_current_instruction_before_freeze(monkeypatch, tmp_path):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'CAMPAIGN_FOLDER', tmp_path/'outputs/new')
    monkeypatch.setattr(run, 'FOLDER', tmp_path/'outputs/new')
    called = []
    monkeypatch.setattr(run, 'freeze', lambda **kwargs: called.append(kwargs))
    note = 'Human approved a new bounded revalidation: 30 requests / 300000 tokens'
    monkeypatch.setattr('sys.argv', ['run', '--freeze', '--campaign-dir', str(tmp_path/'outputs/new'),
        '--parent-campaign', str(tmp_path/'old'), '--max-requests', '30', '--max-tokens', '300000',
        '--supplemental-note', note])
    run.main()
    assert called[0]['supplemental_note'] == note


def test_repair_revision_reuses_cost_and_unknown_prevents_revision(tmp_path):
    budget = CampaignBudget(tmp_path, {'model': 'fixed'}, max_requests=30, max_tokens=300000)
    RecordedProvider(Provider(), budget, phase='initial').complete(MESSAGES, max_tokens=100)
    identity = {'model': 'fixed', 'revision': 'repair'}
    budget.authorize_settled_revision(identity, evidence={'repair': 'generic source-role regression'})
    reopened = CampaignBudget(tmp_path, identity, max_requests=30, max_tokens=300000)
    assert reopened.snapshot()['actual_requests'] == 1 and reopened.snapshot()['reported_tokens'] == 150
    with pytest.raises(OSError):
        RecordedProvider(Provider(OSError('lost')), reopened, phase='unknown').complete(MESSAGES, max_tokens=100)
    with pytest.raises(ValueError, match='unresolved'):
        reopened.authorize_settled_revision({'model': 'fixed', 'revision': 'third'}, evidence={'repair': 'anything'})
    assert reopened.snapshot()['actual_requests'] == 2 and reopened.snapshot()['blocked']


def test_explicit_supplement_resume_carries_unknown_and_stops_on_new_unknown(tmp_path):
    old = CampaignBudget(tmp_path, {'model': 'fixed'}, max_requests=30, max_tokens=300000)
    with pytest.raises(OSError):
        RecordedProvider(Provider(OSError('connect failed')), old, phase='preflight').complete(
            MESSAGES, max_tokens=100, thinking_enabled=False)
    original_row = old.snapshot()['attempts'][0]
    original_request = (tmp_path/'request-001.json').read_bytes()
    identity = {'model': 'fixed', 'revision': 'explicit-resume'}
    with pytest.raises(ValueError):
        run.authorize_supplemental_revision(old, identity, 'continue')
    with pytest.raises(ValueError):
        run.authorize_supplemental_revision(old, identity, '', carry_unknown=True)
    approval = run.authorize_supplemental_revision(
        old, identity, '允许在原预算内显式续验', carry_unknown=True)
    resumed = CampaignBudget(tmp_path, identity, max_requests=30, max_tokens=300000)
    assert old.snapshot()['blocked'] and not resumed.snapshot()['blocked']
    assert resumed.snapshot()['attempts'][0] == original_row
    assert approval['unknown_bounds']['1'] >= original_row['reservation']
    assert resumed.snapshot()['unsettled_upper_bound'] == approval['unknown_bounds']['1']
    RecordedProvider(Provider(), resumed, phase='new-preflight').complete(MESSAGES, max_tokens=100)
    with pytest.raises(OSError):
        RecordedProvider(Provider(OSError('lost again')), resumed, phase='new').complete(MESSAGES, max_tokens=100)
    assert resumed.snapshot()['actual_requests'] == 3 and resumed.snapshot()['blocked']
    assert (tmp_path/'request-001.json').read_bytes() == original_request


def test_carry_unknown_cli_requires_explicit_continuation_and_current_note(monkeypatch, tmp_path):
    for argv in (['--carry-unknown'], ['--freeze-continuation', '--revision', '1', '--carry-unknown']):
        monkeypatch.setattr('sys.argv', ['run', *argv])
        with pytest.raises(SystemExit) as exc:
            run.main()
        assert exc.value.code == 2


def test_campaign_budget_reads_frozen_caps_not_legacy_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'CAMPAIGN_FOLDER', tmp_path)
    protocol = {'identity': {'model': 'fixed'}, 'max_requests': 30, 'max_tokens': 300000}
    assert run.campaign_budget(protocol).snapshot()['max_tokens'] == 300000
    with pytest.raises(ValueError, match='ceiling changed'):
        run.campaign_budget({**protocol, 'max_requests': 31})


def test_source_pass_requires_bound_review_and_all_dimensions(tmp_path):
    identity = {'model': 'fixed'}
    budget = CampaignBudget(tmp_path/'receipts', identity)
    cases = [{'id': str(i), 'paper_ids': ['a'*64]} for i in range(3)]
    save(tmp_path/'protocol.json', {'identity': identity, 'cases': cases})
    result = {'run': {'run_id': 'owned', 'status': 'COMPLETED', 'metadata': {}},
        'final_answer': {'answer': 'A fact [evidence_abc]'},
        'evidence': [{'evidence_id': 'evidence_abc', 'source_uri': 'paper://'+'a'*64+'/chunk'}]}
    save(tmp_path/'0-result.json', result)
    (tmp_path/'source-reviews').mkdir()
    review = {'case': '0', 'run_id': 'owned', 'result_sha256': hashlib.sha256((tmp_path/'0-result.json').read_bytes()).hexdigest(),
        'verdict': 'PASS', 'checks': [{'dimension': d, 'result': 'PASS'} for d in REVIEW_DIMENSIONS],
        'source_artifacts': [{'paper_sha256': 'a'*64, 'sha256': 'frozen source'}]}
    path = tmp_path/'source-reviews'/'0.json'
    save(path, review)
    assert report(tmp_path, budget)['original_demo_passes'] == 1
    review['result_sha256'] = 'stale'
    path.write_text(json.dumps(review))
    assert report(tmp_path, budget)['original_demo_passes'] == 0
    review['result_sha256'] = hashlib.sha256((tmp_path/'0-result.json').read_bytes()).hexdigest()
    review['checks'].pop()
    path.write_text(json.dumps(review))
    assert report(tmp_path, budget)['original_demo_passes'] == 0
    assert report(tmp_path, budget)['rows'][1]['status'] == 'NOT_RUN'


def test_original_demo_partial_outcome_cannot_be_promoted_by_source_review(tmp_path):
    budget = CampaignBudget(tmp_path/'receipts', {'model': 'fixed'})
    save(tmp_path/'protocol.json', {'identity': {'model': 'fixed'},
        'cases': [{'id': 'original', 'original_demo': True, 'paper_ids': ['a'*64]}]})
    result = {'run': {'run_id': 'owned', 'status': 'COMPLETED', 'research_outcome': 'PARTIAL', 'metadata': {}},
        'final_answer': {'answer': 'Fact [evidence_abc]'},
        'evidence': [{'evidence_id': 'evidence_abc', 'source_uri': 'paper://'+'a'*64+'/chunk'}]}
    save(tmp_path/'original-result.json', result)
    (tmp_path/'source-reviews').mkdir()
    save(tmp_path/'source-reviews'/'original.json', {'case': 'original', 'run_id': 'owned',
        'result_sha256': hashlib.sha256((tmp_path/'original-result.json').read_bytes()).hexdigest(),
        'verdict': 'PASS', 'checks': [{'dimension': d, 'result': 'PASS'} for d in REVIEW_DIMENSIONS],
        'source_artifacts': [{'sha256': 'source'}]})
    assert report(tmp_path, budget)['original_demo_passes'] == 0


def test_context_scoring_checks_retractions_and_missing_history_without_prompt_leak():
    correct = {'goal': '理解 ChangeMamba token 组织机制', 'concise_required': False,
               'papers': ['ChangeMamba'], 'gpu_gb': 8, 'training_allowed': False}
    assert semantic_dimensions(correct)['omitted_or_wrong'] == 0
    stale = {**correct, 'concise_required': True, 'papers': ['ChangeMamba', 'FlowMamba'], 'gpu_gb': None}
    assert semantic_dimensions(stale)['omitted_or_wrong'] == 3
    assert semantic_dimensions({})['omitted_or_wrong'] == 5
    messages = context_probe_messages(['neutral progress']*6)
    text = json.dumps(messages)
    assert '8GB' not in text and 'ChangeMamba' not in text and 'FlowMamba' not in text


def test_cli_cannot_implicitly_reset_campaign_or_change_supplement_ceiling(monkeypatch, tmp_path):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'CAMPAIGN_FOLDER', tmp_path/'original')
    monkeypatch.setattr(run, 'FOLDER', tmp_path/'original')
    monkeypatch.setattr('sys.argv', ['run', '--freeze', '--campaign-dir', str(tmp_path/'outputs/new')])
    with pytest.raises(SystemExit):
        run.main()
    monkeypatch.setattr('sys.argv', ['run', '--freeze', '--campaign-dir', str(tmp_path/'outputs/new'),
        '--parent-campaign', str(tmp_path/'old'), '--max-requests', '60', '--max-tokens', '600000'])
    with pytest.raises(SystemExit):
        run.main()


def test_saved_answer_reentry_does_not_build_provider_or_enqueue(tmp_path, monkeypatch):
    parent = history(tmp_path/'old')
    current = tmp_path/'new'
    current.mkdir()
    historical = historical_snapshot(tmp_path/'old')
    historical.pop('protocol')
    protocol = {'identity': {'model': 'fixed'}, 'max_requests': 30, 'max_tokens': 300000,
        'historical_cost': historical, 'cases': [{'id': 'case', 'paper_ids': ['a'*64]}]}
    save(current/'protocol.json', protocol)
    save(current/'case-result.json', {'run': {'run_id': 'owned', 'status': 'FAILED', 'metadata': {}},
                                    'final_answer': {}, 'evidence': []})
    monkeypatch.setattr(run, 'CAMPAIGN_FOLDER', current)
    monkeypatch.setattr(run, 'FOLDER', current)
    monkeypatch.setattr(run, 'local_configuration', lambda: pytest.fail('No provider setup on completed replay'))
    run.execute(case_ids=['case'], stage='answers')
    assert run.campaign_budget(protocol).snapshot()['actual_requests'] == 0
    assert parent.snapshot()['blocked']


def test_context_pair_counts_complete_requests_and_uses_identical_downstream_question(tmp_path):
    from evals.interview.context import paired_probe
    from evals.interview.budget import canonical
    from paperpilot_agentic.db import RuntimeDatabase
    from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
    from paperpilot_agentic.providers import LLMResponse
    db = RuntimeDatabase(tmp_path/'runtime.sqlite3')
    budget = CampaignBudget(tmp_path/'receipts', {'model': 'fixed'}, max_requests=30, max_tokens=300000)
    sent = []
    class ProbeProvider:
        model, max_retries = 'fixed', 0
        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            sent.append((messages, kwargs))
            value = {'goal': None, 'concise_required': None, 'papers': None,
                     'gpu_gb': None, 'training_allowed': False}
            return LLMResponse(json.dumps(value), 100, 50, finish_reason='stop')
    provider = RecordedProvider(ProbeProvider(), budget, phase='context')
    runtime = SimpleNamespace(db=db, llm_runtime=PersistentLLMRuntime(db, provider))
    arms = paired_probe(runtime, tmp_path, 'owner', ['progress']*23, {'status': 'READY'})
    assert budget.snapshot()['actual_requests'] == 2 and budget.snapshot()['reported_tokens'] == 300
    assert sent[0][1] == sent[1][1]
    assert json.loads(sent[0][0][-1]['content'])['question'] == json.loads(sent[1][0][-1]['content'])['question']
    assert arms['derived_state']['requests'][0]['complete_messages_utf8_bytes'] == len(canonical(sent[0][0]).encode())
    assert arms['last_six_turns']['review']['omitted_or_wrong'] == 4
    paired_probe(runtime, tmp_path, 'owner', ['progress']*23, {'status': 'READY'})
    assert budget.snapshot()['actual_requests'] == 2
