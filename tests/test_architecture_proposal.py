import json

import pytest

from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.research_design import enqueue_design
from test_research_design import prepared


@pytest.mark.parametrize('bad_quote', [False, True])
def test_paper_only_proposal_is_not_a_fact_or_execution_permission(tmp_path, bad_quote):
    runtime, provider, parent, sources, _ = prepared(tmp_path)
    def complete(messages, before_attempt=None, **kwargs):
        before_attempt()
        provider.calls += 1
        task = json.loads(messages[-1]['content'])
        value = {key: 'Explicit untested design choice' for key in task['output_schema']}
        value.update(literature_basis=[{'evidence_id': sources[0],
            'quote': 'Fabricated quote' if bad_quote else 'The encoder is shared.',
            'interpretation': 'Shared feature extraction is a source observation.'}],
            code_basis=[], risks=['No experiment or novelty verification'],
            experiment={'baseline': 'Shared encoder and absolute difference',
                'candidate': 'Optional alignment before difference',
                'controls': ['Same evaluation samples'], 'metrics': ['F1', 'IoU'],
                'acceptance_rule': 'Evaluate the predefined practical improvement; inconclusive is allowed',
                'failure_analysis': 'Check if alignment suppresses true changes'})
        return LLMResponse(text=json.dumps(value), finish_reason='stop')
    provider.complete = complete
    rid = enqueue_design(runtime, parent, question='Design a building change detector; no training',
        source_ids=sources, reference_ids=[], design_mode='architecture_proposal')['run']['run_id']
    result = runtime.resume(rid)
    assert result['run']['status'] == ('FAILED' if bad_quote else 'COMPLETED')
    assert result['tool_calls'] == [] and result['pending_approvals'] == []
    if not bad_quote:
        draft = result['run']['metadata']['research_design']
        assert draft['executed'] is False
        assert draft['draft_mode'] == 'architecture_proposal'
        assert draft['constraints']['execution_authorized'] is False
        assert len(draft['artifacts']) == 1
        assert draft['artifacts'][0]['path'].endswith('proposal.json')
        assert 'architecture' in draft['plan']
        assert 'POISON_OLD_ANSWER' not in json.dumps(draft)
        calls = provider.calls
        runtime.resume(rid)
        assert provider.calls == calls


def test_workbench_routes_paper_only_proposal_without_baseline_execution(tmp_path):
    from paperpilot_agentic.workbench import ResearchWorkbench
    from test_workbench import Background
    runtime, provider, parent, sources, _ = prepared(tmp_path)
    wb = ResearchWorkbench(runtime, Background())
    cid = runtime.db.get_run(parent)['metadata']['conversation_id']
    result = wb.submit({'action': 'design', 'request_id': 'paper-only-design-001',
        'parent_run_id': parent, 'conversation_id': cid, 'source_ids': sources,
        'question': 'Propose a network, no execution', 'design_mode': 'architecture_proposal',
        'budget': {'unlimited_llm_usage': True}})
    manifest = runtime.db.get_run(result['run_id'])['metadata']['design_manifest']
    assert manifest['draft_mode'] == 'architecture_proposal'
    assert manifest['reference_ids'] == []
    assert manifest['constraints']['execution_authorized'] is False
    assert provider.calls == 0
    assert wb.submit({'action': 'design', 'request_id': 'paper-only-design-001',
        'parent_run_id': parent, 'conversation_id': cid, 'source_ids': sources,
        'question': 'Propose a network, no execution', 'design_mode': 'architecture_proposal',
        'budget': {'unlimited_llm_usage': True}})['reused']


def test_proposal_rejects_foreign_sources_and_unknown_mode(tmp_path):
    runtime, _, parent, _, _ = prepared(tmp_path)
    with pytest.raises(ValueError):
        enqueue_design(runtime, parent, question='Design', source_ids=['foreign'],
            reference_ids=[], design_mode='architecture_proposal')
    with pytest.raises(ValueError):
        enqueue_design(runtime, parent, question='Design', source_ids=[],
            reference_ids=[], design_mode='invented')


@pytest.mark.parametrize('quote_id', ['s0', 'forged'])
def test_proposal_projects_owned_quotes_and_accepts_structured_design(quote_id):
    from paperpilot_agentic.research_design import _ARCHITECTURE_SCHEMA, _validate_plan
    sources = {'papers': [{'evidence_id': 'original', 'content': 'Original sentence.'}], 'references': []}
    plan = {key: 'Untested proposal' for key in _ARCHITECTURE_SCHEMA}
    plan.update(literature_basis=[{'quote_id': quote_id, 'interpretation': 'Source motivation'}],
        code_basis=[], risks=['Unverified'], architecture={'encoder': {'width': 64, 'operation': 'shared encoding'}},
        ablations=['Baseline', 'Candidate'], experiment={'baseline': 'simple', 'candidate': 'alignment',
            'controls': ['same samples'], 'metrics': ['IoU'], 'acceptance_rule': 'practical effect',
            'failure_analysis': 'measure suppression of true changes'})
    if quote_id == 'forged':
        with pytest.raises(ValueError):
            _validate_plan(plan, sources, architecture=True)
    else:
        result = _validate_plan(plan, sources, architecture=True)
        assert result['literature_basis'][0]['quote'] == 'Original sentence.'
        assert result['literature_basis'][0]['evidence_id'] == 'original'
        assert _validate_plan(result, sources, architecture=True) == result
        result['literature_basis'][0]['quote'] = 'Modified quote'
        with pytest.raises(ValueError):
            _validate_plan(result, sources, architecture=True)
