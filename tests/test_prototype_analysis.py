import copy
import json

import pytest

from paperpilot_agentic.prototype_design_contract import INTERFACE
from paperpilot_agentic.prototype_analysis import POLICY, spans, inspect_quotes, resolve, project_artifact
from paperpilot_agentic.prototype_workspace import complete, plan, task_rows, digest
from test_prototype_literature import setup, work


def prepared(tmp_path, monkeypatch, *, text='Global context is expensive. Local interaction is efficient.'):
    rt, h, rid, state, _ = setup(tmp_path, monkeypatch)
    state['literature_analysis_policy'] = POLICY
    rows = task_rows(state['work'])
    rows[-1]['depends_on'] = ['read', 'analysis']
    rows.insert(-1, {'id': 'analysis', 'goal': 'analyze original literature', 'capability': 'analyze_literature',
                     'depends_on': ['paper'], 'acceptance': 'source-grounded findings and falsifiable directions'})
    plan(state['work'], rows, enforce_capabilities=True)
    complete(state['work'], 'search', 'search_literature', {'evidence': []})
    identity = complete(state['work'], 'paper', 'read_literature', {'ok': True, 'evidence': [
        {'evidence_type': 'TEXT', 'content': text, 'page': 3, 'source_uri': 'https://arxiv.org/pdf/2501.00001'}]})
    return rt, h, rid, state, identity


def proposal(state, identity):
    quote = spans(identity, state['work']['artifacts'][identity])[0]
    return {'findings': [{'artifact_id': identity, 'quote_id': quote['quote_id'], 'interpretation': 'Motivates a hypothesis'}],
            'gaps': ['No measurements for the proposed model'],
            'directions': [{'hypothesis': 'Local interaction may reduce cost', 'mechanism': 'Bounded neighborhoods',
                            'falsification': 'Compare under identical training and resource limits'}]}


def test_host_projects_exact_reader_spans_and_design_uses_analysis_artifact(tmp_path, monkeypatch):
    _, h, rid, state, identity = prepared(tmp_path, monkeypatch)
    source = proposal(state, identity)
    before = copy.deepcopy(source)
    work(h, rid, state, 'analysis', source)
    assert source == before
    analysis_id = state['work']['tasks']['analysis']['output']
    finding = state['work']['artifacts'][analysis_id]['content']['findings'][0]
    assert finding['quote'] == 'Global context is expensive.'
    assert finding['source']['start'] == 0 and finding['source']['page'] == 3
    work(h, rid, state, 'design', {'architecture': 'Shared encoder and local temporal interaction',
        'literature_basis': [{'artifact_id': analysis_id, 'quote_id': finding['quote_id'],
                              'interpretation': 'Proposed design, not a measured gain'}]})
    code_design = state['work']['artifacts'][state['work']['tasks']['design']['output']]
    row = code_design['content']['literature_basis'][0]
    assert row['source_artifact_id'] == identity and row['quote'] == finding['quote']
    assert code_design['inputs'] == sorted([analysis_id, state['work']['tasks']['read']['output']])


@pytest.mark.parametrize('case', ['unknown_id', 'foreign_artifact', 'rewritten_quote', 'metadata_only'])
def test_source_id_protocol_rejects_foreign_modified_and_metadata_only_citations(tmp_path, monkeypatch, case):
    _, _, _, state, identity = prepared(tmp_path, monkeypatch)
    content = proposal(state, identity)
    if case == 'unknown_id':
        content['findings'][0]['quote_id'] = 'q_unknown'
    elif case == 'foreign_artifact':
        content['findings'][0]['artifact_id'] = 'foreign'
    elif case == 'rewritten_quote':
        content['findings'][0]['quote'] = 'Global ... efficient.'
    else:
        artifact = state['work']['artifacts'][identity]
        artifact['content']['evidence'][0]['evidence_type'] = 'METADATA'
        revised = digest(artifact)
        state['work']['artifacts'][revised] = artifact
        state['work']['tasks']['paper']['output'] = revised
        content = proposal(state, revised)
    before = copy.deepcopy(content)
    with pytest.raises(ValueError):
        resolve(state['work'], state['work']['tasks']['analysis'], content, analysis=True)
    assert content == before


def test_span_identity_changes_with_source_version_and_whitespace_is_exact(tmp_path, monkeypatch):
    _, _, _, state, identity = prepared(tmp_path, monkeypatch, text='  Original\ntext. Next statement.  ')
    artifact = state['work']['artifacts'][identity]
    original = spans(identity, artifact)[0]
    assert original['text'] == 'Original\ntext.'
    assert artifact['content']['evidence'][0]['content'][original['start']:original['end']] == original['text']
    updated = copy.deepcopy(artifact)
    updated['content']['evidence'][0]['content'] += ' Added source revision.'
    new_id = digest(updated)
    assert spans(new_id, updated)[0]['quote_id'] != original['quote_id']


def test_design_cannot_bypass_analysis_or_certify_a_modified_analysis(tmp_path, monkeypatch):
    _, h, rid, state, identity = prepared(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='direct dependency'):
        resolve(state['work'], state['work']['tasks']['design'], {'literature_basis': [
            {'artifact_id': identity, 'quote_id': proposal(state, identity)['findings'][0]['quote_id'], 'interpretation': 'bypass'}]})
    work(h, rid, state, 'analysis', proposal(state, identity))
    analysis_id = state['work']['tasks']['analysis']['output']
    state['work']['artifacts'][analysis_id]['content']['findings'][0]['quote'] = 'tampered text'
    with pytest.raises(ValueError, match='unchanged'):
        resolve(state['work'], state['work']['tasks']['design'], {'literature_basis': [
            {'artifact_id': analysis_id, 'quote_id': proposal(state, identity)['findings'][0]['quote_id'], 'interpretation': 'bad'}]})


def test_large_sources_and_materialized_quotes_do_not_reappear_in_design_context(tmp_path, monkeypatch):
    _, h, rid, state, identity = prepared(tmp_path, monkeypatch, text='X'*90000)
    before = copy.deepcopy(state['work']['artifacts'])
    _, messages = h._context(rid, state)
    payload = json.loads(messages[-1]['content'])
    view = payload['context_artifacts'][identity]['content']['quote_catalog']
    assert len(view['spans'][0]['text']) == 1000 and view['spans'][0]['text_complete'] is False
    assert state['work']['artifacts'] == before
    work(h, rid, state, 'analysis', proposal(state, identity))
    _, messages = h._context(rid, state)
    assert len(messages[-1]['content']) < 14000
    assert 'X'*1100 not in messages[-1]['content']
    finding = state['work']['artifacts'][state['work']['tasks']['analysis']['output']]['content']['findings'][0]
    assert len(finding['quote']) == 90000  # canonical provenance is still exact


def test_bounded_quote_navigation_stays_owned_and_reports_partial_views(tmp_path, monkeypatch):
    _, _, _, state, identity = prepared(tmp_path, monkeypatch, text='X'*6000)
    view = inspect_quotes(state['work'], {'artifact_id': identity})
    assert len(view['spans'][0]['text']) == 4000 and not view['spans'][0]['text_complete']
    for args in ({'artifact_id': 'foreign'}, {'artifact_id': identity, 'offset': -1},
                 {'artifact_id': identity, 'limit': True}, {'artifact_id': identity, 'shell': 'cmd'}):
        with pytest.raises(ValueError):
            inspect_quotes(state['work'], args)


def test_new_analysis_plan_admission_is_atomic_and_legacy_protocol_stays_available(tmp_path, monkeypatch):
    _, h, rid, state, _ = prepared(tmp_path, monkeypatch)
    rows = task_rows(state['work'])
    next(t for t in rows if t['id'] == 'design')['depends_on'] = ['read', 'paper']
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match='analyze_literature'):
        h._apply(rid, state, {'action': 'plan', 'reason': 'invalid', 'arguments': {'tasks': rows}}, 1)
    assert state == before
    state.pop('literature_analysis_policy')
    assert 'inspect_quotes' not in h.allowed_actions(state)
    with pytest.raises(ValueError, match='not registered'):
        h._apply(rid, state, {'action': 'plan', 'reason': 'invalid',
                             'arguments': {'tasks': task_rows(state['work'])}}, 1)


def test_projection_never_changes_original_analysis(tmp_path, monkeypatch):
    _, h, rid, state, identity = prepared(tmp_path, monkeypatch)
    work(h, rid, state, 'analysis', proposal(state, identity))
    key = state['work']['tasks']['analysis']['output']
    raw = copy.deepcopy(state['work']['artifacts'][key])
    view = project_artifact(key, raw)
    assert view['content']['findings'][0]['artifact_id'] == key
    assert view['content']['findings'][0]['source_artifact_id'] == identity
    assert 'quote' not in view['content']['findings'][0]
    assert 'quote' in raw['content']['findings'][0]
    assert digest(raw) == key


def test_projected_analysis_findings_can_be_copied_as_valid_design_citations(tmp_path, monkeypatch):
    _, h, rid, state, identity = prepared(tmp_path, monkeypatch)
    work(h, rid, state, 'analysis', proposal(state, identity))
    key = state['work']['tasks']['analysis']['output']
    view = project_artifact(key, state['work']['artifacts'][key])
    finding = view['content']['findings'][0]
    copied = {k: finding[k] for k in ('artifact_id', 'quote_id', 'interpretation')}
    resolved = resolve(state['work'], state['work']['tasks']['design'], {'literature_basis': [copied]})
    assert resolved['literature_basis'][0]['source_artifact_id'] == identity


def test_completion_context_keeps_large_original_quotes_out_of_model_window(tmp_path, monkeypatch):
    _, h, rid, state, identity = prepared(tmp_path, monkeypatch, text='X'*90000)
    work(h, rid, state, 'analysis', proposal(state, identity))
    key = state['work']['tasks']['analysis']['output']
    finding = project_artifact(key, state['work']['artifacts'][key])['content']['findings'][0]
    work(h, rid, state, 'design', {'literature_basis': [
        {k: finding[k] for k in ('artifact_id', 'quote_id', 'interpretation')}]})
    _, messages = h._context(rid, state)
    payload = json.loads(messages[-1]['content'])
    assert not payload['ready_contexts'] and len(messages[-1]['content']) < 14000
    assert 'X'*1100 not in messages[-1]['content']


def test_durable_workflow_connects_local_search_analysis_design_and_candidate_approval(tmp_path, monkeypatch):
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from paperpilot_agentic.prototype_harness import PrototypeHarness
    from paperpilot_agentic.providers import LLMResponse
    from test_prototype_workspace import stub_environment

    rows = [{'id': key, 'goal': key, 'capability': kind, 'depends_on': deps, 'acceptance': 'fixture'}
            for key, kind, deps in [('read', 'read', []), ('search', 'search_literature', []),
                ('analysis', 'analyze_literature', ['search']),
                ('claims', 'review_literature', ['analysis']),
                ('design', 'design', ['read', 'analysis', 'claims']),
                ('code', 'implement', ['read', 'design']), ('check', 'validate', ['code']),
                ('review', 'review', ['read', 'design', 'code', 'check'])]]

    class Corpus:
        calls = 0

        def search(self, query, top_k):
            self.calls += 1
            return [{'content': 'Temporal interaction can be bounded. This is maintained test evidence.',
                     'source_uri': 'fixture://public-paper', 'page': 2}]

    class Actor:
        model = 'maintained-analysis-workflow-fixture'

        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]['content'])
            if not payload['tasks']:
                action, args = 'plan', {'tasks': rows}
            else:
                key = payload['active_task_id']
                kind = payload['tasks'][key]['capability']
                table = payload['context_artifacts']
                if kind == 'analyze_literature':
                    identity, source = next((i, a) for i, a in table.items() if a['kind'] == 'search_literature')
                    content = {'findings': [{'artifact_id': identity,
                        'quote_id': source['content']['quote_catalog']['spans'][0]['quote_id'],
                        'interpretation': 'A falsifiable local interaction hypothesis'}],
                        'gaps': ['No measured effect'], 'directions': [{'hypothesis': 'Bounded cost',
                            'mechanism': 'Local interaction', 'falsification': 'Equal-budget controlled comparison'}]}
                elif kind == 'review_literature':
                    content = {'checks': [{'finding_index': f['finding_index'], 'quote_id': f['quote_id'],
                        'source_role': 'implemented_method', 'interpretation_supported': True,
                        'reason': 'Fixture hypothesis is explicitly unmeasured', 'required_correction': ''}
                        for f in payload['literature_review_packet']['findings']],
                        'coverage': {'adequate_for_design': True, 'reason': 'Fixture only',
                                     'limitations': ['No novelty verification']}}
                elif kind == 'design':
                    source = next(a for a in table.values() if a['kind'] == 'analyze_literature')
                    content = {'architecture': 'Fixture only', 'tensor_interfaces': INTERFACE, 'literature_basis': [
                        {k: source['content']['findings'][0][k] for k in ('artifact_id', 'quote_id', 'interpretation')}]}
                else:
                    content = {'read': 'source', 'search_literature': {'scope': 'local', 'query': 'temporal differences', 'top_k': 1},
                               'implement': {'source': 'def build_model(config):\n return None\n', 'config': {}},
                               'validate': None}[kind]
                action, args = 'work', {'task_id': key, 'content': content}
            return LLMResponse(text=json.dumps({'action': action, 'arguments': args, 'reason': 'maintained fixture'}),
                               model=self.model, finish_reason='stop', prompt_tokens=20, completion_tokens=20)

    stub_environment(monkeypatch)
    corpus = Corpus()
    rt = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path/'runtime.db', llm=Actor(), search_adapter=corpus)
    monkeypatch.setattr(rt.db, 'list_evidence', lambda _: [{'evidence_id': 'source', 'evidence_type': 'CODE',
                                                           'content': 'def build_model(config): return None'}])
    h = PrototypeHarness(rt)
    rid = h.start('Open research fixture', 'analysis-workflow-fixture', 'parent', ['source'],
                  literature_enabled=True)['run']['run_id']
    h.advance(rid)
    state = h._load(rid)
    assert state['phase'] == 'WAITING_APPROVAL' and corpus.calls == 1
    assert all(state['work']['tasks'][k]['status'] == 'DONE' for k in ('read', 'search', 'analysis', 'design', 'code'))
    assert state['work']['tasks']['check']['status'] == 'PENDING'
    assert not any(c['tool_name'] == 'prototype.validate' and c['attempt'] for c in rt.db.list_tool_calls(rid))
    calls = rt.db.get_run(rid)['used_llm_calls']
    assert calls == 6  # host reads the sole source and prepares fixed validation; model still performs claim review
    PrototypeHarness(rt).advance(rid)
    assert rt.db.get_run(rid)['used_llm_calls'] == calls  # pending approval is not a free new model turn
