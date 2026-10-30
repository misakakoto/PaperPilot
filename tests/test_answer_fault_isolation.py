"""A malformed inference must not prevent independent source-fact auditing."""
import json
from copy import deepcopy

import pytest

from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.answer_isolation import isolate_draft, merge_repair, repair_workset
from paperpilot_agentic.llm_runtime import LLMBudgetExhausted
from paperpilot_agentic.providers import LLMResponse


def test_invalid_inference_chain_keeps_source_fact_for_audit():
    draft = {'facts': [
        {'id': 'method', 'text': 'The encoder uses local attention.',
         'kind': 'SOURCE_FACT', 'evidence_ids': ['e1']},
        {'id': 'guess', 'text': 'This may reduce computation.',
         'kind': 'INFERENCE', 'evidence_ids': ['e1'], 'depends_on': ['method']},
        {'id': 'bad', 'text': 'It therefore runs faster on every device.',
         'kind': 'INFERENCE', 'evidence_ids': ['e1'], 'depends_on': ['guess']}],
        'direct_answer_id': 'method', 'limitations': []}
    seen = []

    class Model:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            seen.append(messages)
            if len(seen) == 1:
                result = draft
            else:
                assert 'Independently audit' in messages[0]['content'], (
                    'whole draft repair ran before eligible facts reached audit')
                payload = json.loads(messages[-1]['content'])
                assert 'previous_output' not in payload, 'whole draft repair ran before audit'
                assert [f['id'] for f in payload['facts']] == ['method', 'guess']
                result = {'assessments': [
                    {'fact_id': 'method', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                     'span_ids': ['s0']},
                    {'fact_id': 'guess', 'verdict': 'UNSUPPORTED', 'relevance': 'DIRECT'}],
                    'coverage': 'COMPLETE', 'missing_aspects': []}
            return LLMResponse(json.dumps(result), finish_reason='stop')

    report = answer_then_audit('Explain the encoder and its benefit.',
        [{'evidence_id': 'e1', 'content': 'The encoder uses local attention.'}], Model())
    assert report['status'] == 'REVIEW_REQUIRED', report.get('failure_detail')
    assert len(seen) == 2
    assert 'The encoder uses local attention.' in report['answer']
    assert 'every device' not in report['answer']
    assert report['raw_draft'] == draft
    assert report['answer_scope'] == 'PARTIAL'
    assert report['draft_isolation']['quarantined_fact_ids'] == ['bad']
    assert report['semantic_correctness'] == 'MODEL_AUDIT_ONLY_NOT_INDEPENDENT_GROUND_TRUTH'


SOURCE = [{'evidence_id': 'e1', 'content': 'The encoder uses local attention. '
           'The decoder uses nearest-neighbor interpolation.'},
          {'evidence_id': 'other', 'content': 'Unrelated full passage. ' * 500}]
DRAFT = {'facts': [
    {'id': 'method', 'kind': 'SOURCE_FACT', 'text': 'The encoder uses local attention.',
     'evidence_ids': ['e1']},
    {'id': 'guess', 'kind': 'INFERENCE', 'text': 'This may reduce computation.',
     'evidence_ids': ['e1'], 'depends_on': ['method']},
    {'id': 'bad', 'kind': 'INFERENCE', 'text': 'It therefore always runs faster.',
     'evidence_ids': ['e1'], 'depends_on': ['guess']}],
    'direct_answer_id': 'method', 'limitations': []}


@pytest.mark.parametrize('defect', [
    {'evidence_ids': ['unknown']}, {'kind': 'FAKE'}, {'text': ''},
    {'depends_on': ['missing']}, {'depends_on': ['bad']}, {'depends_on': 'method'},
    {'row_refs': [{'evidence_id': 'e1', 'row_id': 'invented', 'metrics': ['F1']}]},
])
def test_fact_defects_are_quarantined_without_mutating_original(defect):
    draft = deepcopy(DRAFT)
    draft['facts'][2].update(defect)
    before = deepcopy(draft)
    subset, result = isolate_draft(draft, SOURCE, question='Explain the method.')
    assert draft == before
    assert subset['facts'] == draft['facts'][:2]
    assert result['quarantined_fact_ids'] == ['bad']
    assert result['factual_review'] == 'NOT_RUN'


@pytest.mark.parametrize('defect', ['duplicate', 'missing_nomination', 'bad_limitations', 'too_many'])
def test_global_ambiguity_still_requires_original_bounded_repair(defect):
    draft = deepcopy(DRAFT)
    if defect == 'duplicate':
        draft['facts'][2]['id'] = 'method'
    elif defect == 'missing_nomination':
        draft['direct_answer_id'] = 'absent'
    elif defect == 'bad_limitations':
        draft['limitations'] = {}
    else:
        draft['facts'] *= 5
    assert isolate_draft(draft, SOURCE) is None


def test_invalid_source_cascades_to_dependents_and_all_invalid_is_not_abstention():
    draft = deepcopy(DRAFT)
    draft['facts'][0]['evidence_ids'] = ['unknown']
    assert isolate_draft(draft, SOURCE) is None
    draft['facts'].append({'id': 'independent', 'text': 'A different sourced fact.',
                           'evidence_ids': ['e1']})
    subset, result = isolate_draft(draft, SOURCE)
    assert result['quarantined_fact_ids'] == ['bad', 'guess', 'method']
    assert result['direct_answer_reassigned']
    assert [f['id'] for f in subset['facts']] == ['independent']


def test_local_workset_preserves_exact_source_and_marks_omissions_as_navigation():
    _, isolation = isolate_draft(DRAFT, SOURCE)
    targets, premises, selected, nav = repair_workset(DRAFT, isolation, SOURCE)
    assert targets == DRAFT['facts'][2:]
    assert premises == DRAFT['facts'][:1]
    assert selected == SOURCE[:1]
    assert nav[1]['status'] == 'NOT_READ_IN_REPAIR'
    assert len(nav[1]['content_sha256']) == 64


@pytest.mark.parametrize('citations', [None, 123, 'e1', [['e1']], {'e1': True}])
def test_repair_workset_tolerates_quarantined_citation_type_errors(citations):
    draft = deepcopy(DRAFT)
    draft['facts'][2]['evidence_ids'] = citations
    _, isolation = isolate_draft(draft, SOURCE)
    targets, premises, selected, navigation = repair_workset(draft, isolation, SOURCE)
    assert targets[0]['evidence_ids'] == citations  # preserve rejected original
    assert selected == SOURCE[:1]  # dependency chain still provides a source
    assert navigation[1]['status'] == 'NOT_READ_IN_REPAIR'


def test_patch_cannot_modify_eligible_fact_or_create_id():
    _, isolation = isolate_draft(DRAFT, SOURCE)
    for facts in ([DRAFT['facts'][0]], [{**DRAFT['facts'][2], 'id': 'new'}], []):
        with pytest.raises(ValueError, match='exactly the quarantined'):
            merge_repair(DRAFT, {'facts': facts, 'limitations': []}, isolation,
                         SOURCE, question='Q', request_history=[], table_rows=[])


def test_local_patch_recovers_provenance_only_from_supplied_source_premise():
    _, isolation = isolate_draft(DRAFT, SOURCE)
    patch = {'facts': [{'id': 'bad', 'kind': 'INFERENCE',
        'text': 'Local attention may limit dependencies.',
        'evidence_ids': ['method'], 'depends_on': ['method']}], 'limitations': []}
    merged = merge_repair(DRAFT, patch, isolation, SOURCE, question='Q',
        request_history=[], table_rows=[], premise_ids=['method'], repair_sources=SOURCE[:1])
    assert merged['facts'][:2] == DRAFT['facts'][:2]
    assert merged['facts'][2]['evidence_ids'] == ['e1']
    assert merged['facts'][2]['depends_on'] == ['method']
    assert patch['facts'][0]['evidence_ids'] == ['method']  # raw patch retained


@pytest.mark.parametrize('citations,deps', [(['other'], []), (['e1'], ['unshown'])])
def test_local_recovery_cannot_borrow_unshown_source_or_premise(citations, deps):
    original = deepcopy(DRAFT)
    original['facts'].append({'id': 'unshown', 'kind': 'SOURCE_FACT',
                             'text': 'An unrelated fact.', 'evidence_ids': ['other']})
    _, isolation = isolate_draft(original, SOURCE)
    patch = {'facts': [{'id': 'bad', 'kind': 'INFERENCE', 'text': 'A proposed conclusion.',
                       'evidence_ids': citations, 'depends_on': deps}], 'limitations': []}
    with pytest.raises(ValueError):
        merge_repair(original, patch, isolation, SOURCE, question='Q', request_history=[],
                     table_rows=[], premise_ids=['method'], repair_sources=SOURCE[:1])


class PatchModel:
    def __init__(self, *, blocked=False, malformed=False, audit_error=False):
        self.seen, self.preflights = [], []
        self.blocked, self.malformed, self.audit_error = blocked, malformed, audit_error

    def ensure_capacity(self, requests):
        self.preflights.append(requests)
        if self.blocked and len(self.preflights) > 1:
            raise LLMBudgetExhausted('Optional repair does not fit')

    def complete(self, messages, **kwargs):
        kwargs['before_attempt']()
        self.seen.append(messages)
        n = len(self.seen)
        if n == 1:
            value = DRAFT
        elif n == 3:
            payload = json.loads(messages[-1]['content'])
            assert payload['repair_fact_ids'] == ['bad']
            assert 'output_contract' not in payload
            assert 'json' in messages[0]['content'].casefold()
            assert 'Return JSON: {"facts"' not in messages[0]['content']
            assert 'direct_answer_id' in messages[0]['content']  # explicitly forbidden
            assert 'REQUESTED MEANING' in messages[0]['content']
            assert 'numbers-free' in messages[0]['content']
            assert 'duplicate operation=value' in messages[0]['content']
            assert [e['evidence_id'] for e in payload['evidence']] == ['e1']
            assert payload['read_packet_sha256'] == json.loads(self.seen[0][-1]['content'])['read_packet_sha256']
            assert messages == self.preflights[-1][0]['messages']
            value = {'facts': [{'id': 'bad' if not self.malformed else 'method',
                'kind': 'SOURCE_FACT', 'text': 'The decoder uses nearest-neighbor interpolation.',
                'evidence_ids': ['e1'], 'depends_on': []}], 'limitations': []}
        else:
            if n == 4 and self.audit_error:
                raise OSError('Lost audit response: stop without replay')
            payload = json.loads(messages[-1]['content'])
            # Complete audit and local repair use the exact same question packet.
            assert {r['evidence_id'] for r in payload['sources']} == {'e1'}
            assert payload['read_packet_sha256'] == json.loads(self.seen[0][-1]['content'])['read_packet_sha256']
            value = {'assessments': [{'fact_id': f['id'],
                'verdict': 'UNSUPPORTED' if f['id'] == 'guess' else 'SUPPORTED',
                'relevance': 'DIRECT',
                'span_ids': [] if f['id'] == 'guess' else ['s1' if f['id'] == 'bad' else 's0']}
                for f in payload['facts']], 'coverage': 'COMPLETE', 'missing_aspects': []}
        return LLMResponse(json.dumps(value), finish_reason='stop')


def test_local_patch_runs_after_audit_and_keeps_eligible_facts_immutable():
    model = PatchModel()
    report = answer_then_audit('Explain the encoder, decoder and benefit.', SOURCE, model)
    assert report['status'] == 'REVIEW_REQUIRED', report.get('failure_detail')
    assert len(model.seen) == report['provider_attempts'] == 4
    assert report['draft_isolation']['repair_status'] == 'PATCH_AUDITED'
    assert report['draft']['facts'][:2] == DRAFT['facts'][:2]
    assert 'nearest-neighbor interpolation' in report['answer']
    assert 'always runs faster' not in report['answer']
    assert report['raw_draft'] == DRAFT
    assert report['answer_scope'] == 'PARTIAL'  # guess was rejected by audit.


def test_local_patch_uses_host_keyed_replacement_without_model_supplied_fact_id():
    class HostBoundPatchModel(PatchModel):
        def complete(self, messages, **kwargs):
            if len(self.seen) != 2:
                return super().complete(messages, **kwargs)
            kwargs['before_attempt']()
            self.seen.append(messages)
            payload=json.loads(messages[-1]['content'])
            assert payload['replacement_contract']['target_ids']==['bad']
            assert payload['replacement_contract']['id_owner']=='host'
            assert '"id":"f4"' not in messages[0]['content']
            value={'replacements':{'bad':{'kind':'SOURCE_FACT',
                'text':'The decoder uses nearest-neighbor interpolation.',
                'evidence_ids':['e1'],'depends_on':[]}},'limitations':[]}
            return LLMResponse(json.dumps(value),finish_reason='stop')
    model=HostBoundPatchModel()
    report=answer_then_audit('Explain the encoder, decoder and benefit.',SOURCE,model)
    assert report['draft_isolation']['repair_status']=='PATCH_AUDITED'
    assert report['draft']['facts'][:2]==DRAFT['facts'][:2]
    assert 'nearest-neighbor interpolation' in report['answer']
    assert report['local_repair_bindings'][0]['fact_id']=='bad'
    assert 'id' not in report['raw_local_repair']['replacements']['bad']
    assert len(model.seen)==4


@pytest.mark.parametrize('defect',['foreign_key','content_id','ambiguous_envelope','missing_target'])
def test_host_bound_replacements_never_guess_or_reassign_identity(defect):
    from paperpilot_agentic.answer_isolation import bind_replacements
    value={'replacements':{'target':{'kind':'SOURCE_FACT','text':'A source fact.',
           'evidence_ids':['e1'],'depends_on':[]}},'limitations':[]}
    if defect=='foreign_key':value['replacements']={'other':value['replacements']['target']}
    elif defect=='content_id':value['replacements']['target']['id']='target'
    elif defect=='ambiguous_envelope':value['facts']=[]
    else:value['replacements']={}
    original=deepcopy(value)
    with pytest.raises(ValueError):bind_replacements(value,['target'])
    assert value==original


@pytest.mark.parametrize('option,expected,calls', [
    ('blocked', 'NOT_RUN_BUDGET_EXHAUSTED', 2), ('malformed', 'FAILED_VALIDATION', 3),
    ('audit_error', 'STOPPED_ERROR', 4),
])
def test_optional_failure_retains_only_previously_audited_answer(option, expected, calls):
    model = PatchModel(**{option: True})
    report = answer_then_audit('Explain the encoder, decoder and benefit.', SOURCE, model)
    assert len(model.seen) == report['provider_attempts'] == calls
    assert report['draft_isolation']['repair_status'] == expected
    assert report['draft']['facts'] == DRAFT['facts'][:2]
    assert 'local attention' in report['answer']
    assert 'nearest-neighbor interpolation' not in report['answer']
    assert report['answer_scope'] == 'PARTIAL'
    if option == 'blocked':
        assert report['error_type'] == 'LLMBudgetExhausted'


def test_normalization_type_error_cannot_hide_independent_fact():
    draft = deepcopy(DRAFT)
    draft['facts'][0]['evidence_ids'] = None
    draft['facts'][1].pop('depends_on')
    draft['facts'].append({'id': 'decoder', 'kind': 'SOURCE_FACT',
        'text': 'The decoder uses nearest-neighbor interpolation.', 'evidence_ids': ['e1']})

    class Model:
        calls = 0

        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            self.calls += 1
            if self.calls == 1:
                value = draft
            else:
                assert 'Independently audit' in messages[0]['content']
                assert [f['id'] for f in json.loads(messages[-1]['content'])['facts']] == ['decoder']
                value = {'assessments': [{'fact_id': 'decoder', 'verdict': 'SUPPORTED',
                    'relevance': 'DIRECT', 'span_ids': ['s1']}], 'coverage': 'COMPLETE'}
            return LLMResponse(json.dumps(value), finish_reason='stop')

    report = answer_then_audit('Explain the encoder and decoder.', SOURCE, Model())
    assert report['status'] == 'REVIEW_REQUIRED', report.get('failure_detail')
    assert report['provider_attempts'] == 2
    assert report['draft_isolation']['quarantined_fact_ids'] == ['bad', 'guess', 'method']
    assert 'nearest-neighbor interpolation' in report['answer']
    assert report['answer_scope'] == 'PARTIAL'


def test_frozen_draft_diagnostic_remains_strict_and_spends_nothing():
    model = PatchModel()
    with pytest.raises(ValueError, match='SOURCE_FACT IDs'):
        answer_then_audit('Q', SOURCE, model, frozen_draft=DRAFT)
    assert not model.seen


@pytest.mark.parametrize('lose_audit', [False, True])
def test_durable_patch_reuses_completed_report_and_blocks_unknown(tmp_path, lose_audit):
    from paperpilot_agentic.db import RuntimeDatabase
    from paperpilot_agentic.durable_reader import DurableResearchModel, run_durable_reader
    from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
    from paperpilot_agentic.llm_attempts import UnknownRequestResult
    from paperpilot_agentic.models import Budget

    class Provider:
        model, max_retries = 'offline-fixed', 0

        def __init__(self):
            self.calls = 0

        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            self.calls += 1
            payload = json.loads(messages[-1]['content'])
            if self.calls == 1:
                self.eid = payload['evidence'][0]['evidence_id']
                value = deepcopy(DRAFT)
                for fact in value['facts']:
                    fact['evidence_ids'] = [self.eid]
            elif self.calls == 3:
                value = {'facts': [{'id': 'bad', 'kind': 'SOURCE_FACT',
                    'text': 'The decoder uses nearest-neighbor interpolation.',
                    'evidence_ids': [self.eid]}], 'limitations': []}
            else:
                if self.calls == 4 and lose_audit:
                    raise OSError('Offline unknown response fixture')
                value = {'assessments': [{'fact_id': f['id'],
                    'verdict': 'UNSUPPORTED' if f['id'] == 'guess' else 'SUPPORTED',
                    'relevance': 'DIRECT',
                    'span_ids': [] if f['id'] == 'guess' else ['s1' if f['id'] == 'bad' else 's0']}
                    for f in payload['facts']], 'coverage': 'COMPLETE', 'missing_aspects': []}
            return LLMResponse(json.dumps(value), 10, 5, model=self.model, finish_reason='stop')

    db = RuntimeDatabase(tmp_path / 'isolated.sqlite3')
    rid = db.create_run('Explain the encoder, decoder and benefit.', Budget(max_tokens=300000))
    provider = Provider()
    runtime = PersistentLLMRuntime(db, provider)
    model = DurableResearchModel(runtime, rid, stage='answer_first', reserve_class='synthesis')
    report = run_durable_reader(db, rid, SOURCE, model, protocol_id='offline-isolation-test')
    assert provider.calls == report['provider_attempts'] == db.get_run(rid)['used_llm_calls'] == 4
    assert bool('nearest-neighbor interpolation' in report['answer']) is not lose_audit
    assert 'local attention' in report['answer']
    assert report['status'] == ('FAILED' if lose_audit else 'REVIEW_REQUIRED')
    before = runtime.budget_snapshot(rid)
    assert run_durable_reader(db, rid, SOURCE, model, protocol_id='offline-isolation-test') == report
    assert provider.calls == 4
    assert runtime.budget_snapshot(rid) == before
    if lose_audit:
        with pytest.raises(UnknownRequestResult):
            runtime.ensure_capacity(rid, [{'messages': [{'role': 'user', 'content': 'Q'}],
                                          'max_tokens': 2000}])
        assert provider.calls == 4


@pytest.mark.parametrize('bad_reaudit', [False, True])
def test_content_patch_survives_used_format_repair_and_is_independently_audited(tmp_path, bad_reaudit):
    """Real failure pattern: audit format repair consumed the formula patch slot."""
    from paperpilot_agentic.db import RuntimeDatabase
    from paperpilot_agentic.durable_reader import DurableResearchModel, run_durable_reader
    from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
    from paperpilot_agentic.models import Budget

    class Provider:
        model, max_retries = 'offline-fixed', 0
        calls = 0
        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            self.calls += 1
            payload = json.loads(next(m['content'] for m in messages if m['role'] == 'user'))
            if self.calls == 1:
                self.eid = payload['evidence'][0]['evidence_id']
                value = {'facts': [
                    {'id': 'kept', 'kind': 'SOURCE_FACT', 'text': 'The encoder uses local attention.',
                     'evidence_ids': [self.eid]},
                    {'id': 'formula', 'kind': 'SOURCE_FACT', 'text': 'The mechanism uses F = concat_c(T1,T2).',
                     'evidence_ids': [self.eid]}], 'direct_answer_id': 'kept', 'limitations': []}
            elif self.calls == 4:
                assert payload['repair_fact_ids'] == ['formula']
                value = {'facts': [{'id': 'formula', 'kind': 'SOURCE_FACT',
                    'text': 'The mechanism concatenates temporal features along the channel axis.',
                    'evidence_ids': [self.eid]}], 'limitations': []}
            else:
                value = {'assessments': [{'fact_id': f['id'], 'verdict': 'SUPPORTED',
                    'relevance': 'DIRECT', 'span_ids': (['s0'] * 13 if self.calls == 2 or
                        self.calls == 5 and bad_reaudit else
                        ['s0' if f['id'] == 'kept' else 's1'])}
                    for f in payload['facts']], 'coverage': 'COMPLETE', 'missing_aspects': []}
            return LLMResponse(json.dumps(value), 10, 5, model=self.model, finish_reason='stop')

    db = RuntimeDatabase(tmp_path / 'separate-repairs.sqlite3')
    rid = db.create_run('Explain the encoder and mechanism in words.', Budget(max_tokens=300000))
    provider = Provider()
    runtime = PersistentLLMRuntime(db, provider)
    model = DurableResearchModel(runtime, rid, stage='answer_first', reserve_class='synthesis')
    report = run_durable_reader(db, rid, [{'evidence_id': 'e1', 'content':
        'The encoder uses local attention. The mechanism concatenates temporal features along the channel axis.'}],
        model, protocol_id='independent-content-patch')
    assert provider.calls == 5, {k: report.get(k) for k in ('failure_detail', 'draft_isolation', 'error_type')}
    assert report['draft_isolation']['repair_status'] == ('STOPPED_ERROR' if bad_reaudit else 'PATCH_AUDITED')
    assert ('channel axis' in report['answer']) is not bad_reaudit
    assert 'local attention' in report['answer']
    assert 'concat_c' not in report['answer']
    assert report['raw_draft']['facts'][1]['text'] == 'The mechanism uses F = concat_c(T1,T2).'
    stages = [row['stage'] for row in db.list_llm_calls(rid)]
    assert stages.count('json_repair') == 1
    assert stages.count('answer_content_patch') == 1
    before = runtime.budget_snapshot(rid)
    assert run_durable_reader(db, rid, [{'evidence_id': 'e1', 'content':
        'The encoder uses local attention. The mechanism concatenates temporal features along the channel axis.'}],
        model, protocol_id='independent-content-patch') == report
    assert provider.calls == 5 and runtime.budget_snapshot(rid) == before
