"""Synthetic contract tests; these do not measure research/model quality."""
import ast
import json
from pathlib import Path
import sys

import pytest

from paperpilot_agentic.durable_reader import persist_reader_evidence
from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
from paperpilot_agentic.method_drafts import build_patch, record_draft_review
from paperpilot_agentic.models import RunStatus
from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic import reference_methods as methods
from paperpilot_agentic.research_design import enqueue_design, register_code_reference
from test_reference_methods import FakeRunner, profile as profile, runtime_with_runner
from test_research_design import Designer


SOURCE = 'class SNUNet_ECAM:\n    def forward(self, xA, xB):\n        return (xA + xB,)\n'
REPLACEMENT = 'def forward(self, xA, xB):\n    return (xA - xB,)\n'


class LocalDesigner(Designer):
    def complete(self, messages, before_attempt=None, **kwargs):
        task = json.loads(messages[-1]['content'])
        if 'replacement' not in task['output_schema']:
            return super().complete(messages, before_attempt=before_attempt, **kwargs)
        if before_attempt:
            before_attempt()
        self.calls += 1
        self.messages.append(messages)
        value = {'target': 'SNUNet_ECAM.forward', 'replacement': REPLACEMENT,
                 'review_notes': ['Synthetic fixture; no shape, safety or hypothesis assurance']}
        if 'target' not in task['output_schema']:
            value.pop('target')
        return LLMResponse(text=json.dumps(self.mutate(value)), finish_reason='stop')


@pytest.fixture
def bound(profile, monkeypatch):
    workspace, _, snapshot, dataset = profile
    data = json.loads(snapshot.read_text())
    data['code'] = SOURCE
    snapshot.write_text(json.dumps(data))
    monkeypatch.setattr(methods, 'SOURCE_SHA', methods.sha(SOURCE.encode()))
    binding = methods.prepare_snunet(workspace, snapshot=snapshot, dataset=dataset, python=sys.executable)
    runner, provider = FakeRunner(), LocalDesigner()
    runtime = runtime_with_runner(workspace, runner)
    runtime.llm_runtime = PersistentLLMRuntime(runtime.db, provider)
    runtime.text_reader_protocol = 'method-draft-fixture-v1'
    parent = runtime.enqueue(question='Read original excerpt', metadata={'engine': 'answer_first_text'})['run']['run_id']
    sources = persist_reader_evidence(runtime.db, parent, [{'evidence_id': 'paper',
        'content': 'The encoder is shared.', 'source_uri': 'synthetic://paper'}])
    runtime.db.update_run_status(parent, RunStatus.COMPLETED)
    reference = register_code_reference(runtime, parent, source_url=methods.PIN+'models/Models.py',
        content=SOURCE, license_url=methods.PIN+'LICENSE', license_text=data['license_text'])
    validation = methods.enqueue_validation(runtime, binding, parent_run_id=parent)['run']['run_id']
    waiting = runtime.resume(validation)
    runtime.approve(waiting['pending_approvals'][0]['approval_id'], actor='test')
    assert runtime.resume(validation)['run']['status'] == 'COMPLETED'
    args = {'question': 'One local diagnostic hypothesis only',
            'source_ids': [sources[0]['evidence_id']], 'reference_ids': [reference],
            'validation_run_id': validation}
    return runtime, provider, parent, args


def test_verified_draft_preserves_baseline_and_replays_without_executing(bound):
    runtime, provider, parent, args = bound
    run_id = enqueue_design(runtime, parent, **args)['run']['run_id']
    result = runtime.resume(run_id)
    assert result['run']['status'] == 'COMPLETED', result['run'].get('error')
    draft = result['run']['metadata']['research_design']
    assert draft['draft_mode'] == 'validated_method_patch'
    assert draft['executed'] is False and not result['tool_calls']
    assert {Path(a['path']).name for a in draft['artifacts']} == {'candidate.patch', 'candidate.json'}
    contract = draft['patch_contract']
    assert contract['validation_run_id'] == args['validation_run_id']
    assert contract['execution_authorized'] is False
    root, _ = methods.load_profile(runtime.workspace, contract['binding_id'])
    assert (root/'upstream.py').read_text() == SOURCE
    assert provider.calls == 2
    assert json.loads(provider.messages[0][-1]['content'])['verified_baseline']['binding_id'] == contract['binding_id']
    constraints = json.loads(provider.messages[0][-1]['content'])['constraints']
    assert constraints['data_scope'] == {'train_images': 2, 'validation_images': 2,
                                         'test_images': 0, 'image_size_pixels': [256, 256]}
    assert constraints['validated_steps'] == 2
    assert 'cannot establish non-inferiority' in constraints['conclusion_boundary']
    runtime.db.update_run_status(run_id, RunStatus.RUNNING)
    assert runtime.resume(run_id)['run']['status'] == 'COMPLETED'
    assert provider.calls == 2 and len(runtime.db.list_artifacts(run_id)) == 2


def test_bound_design_does_not_mix_generic_training_defaults_with_frozen_baseline(bound):
    runtime, provider, parent, args = bound
    rid = enqueue_design(runtime, parent, **args)['run']['run_id']
    result = runtime.resume(rid)
    assert result['run']['status'] == 'COMPLETED'
    constraints = json.loads(provider.messages[0][-1]['content'])['constraints']
    assert constraints['seeds'] == [7]
    assert constraints['max_steps_per_variant_per_seed'] == 2
    contract = result['run']['metadata']['research_design']['evaluation_contract']
    assert contract['scientific_hypothesis_status'] == 'NOT_TESTED_BY_THIS_PROTOCOL'
    assert contract['execution_authorized'] is False
    assert '不由本次微型实验验证' in result['final_answer']['answer']


def test_fixed_patch_target_is_host_owned_not_a_repeated_model_format_requirement(bound):
    runtime, provider, parent, args = bound
    rid = enqueue_design(runtime, parent, **args)['run']['run_id']
    result = runtime.resume(rid)
    assert result['run']['status'] == 'COMPLETED'
    assert provider.calls == 2
    task = json.loads(provider.messages[-1][-1]['content'])
    assert set(task['output_schema']) == {'replacement', 'review_notes'}
    assert result['run']['metadata']['research_design']['patch_contract']['target'] == 'SNUNet_ECAM.forward'
    assert 'target' not in json.loads(result['run']['metadata']['research_design_patch_raw']['text'])
    assert not result['tool_calls']


def test_model_cannot_override_the_host_owned_target(bound):
    runtime, provider, parent, args = bound
    def mutate(value):
        if 'replacement' in value:
            value['target'] = 'AnotherClass.forward'
        return value
    provider.mutate = mutate
    rid = enqueue_design(runtime, parent, **args)['run']['run_id']
    result = runtime.resume(rid)
    assert result['run']['status'] == 'FAILED'
    assert not result['tool_calls'] and not result['artifacts']


@pytest.mark.parametrize('problem', ['conversation', 'status', 'receipt', 'metadata', 'source', 'approval'])
def test_unowned_stale_or_unapproved_baseline_cannot_start_a_paid_design(bound, problem):
    runtime, provider, parent, args = bound
    rid = args['validation_run_id']
    if problem == 'conversation':
        runtime.db.update_run_metadata(rid, {'conversation_id': 'another-conversation'})
    elif problem == 'status':
        runtime.db.update_run_status(rid, RunStatus.FAILED)
    elif problem == 'receipt':
        receipt = next(a for a in runtime.db.list_artifacts(rid) if a['kind'] == 'method-verification')
        Path(receipt['path']).write_text('{}')
    elif problem == 'metadata':
        runtime.db.update_run_metadata(rid, {'method_validation': {'status': 'ADAPTER_SMOKE_VERIFIED'}})
    elif problem == 'source':
        binding = runtime.db.get_run(rid)['metadata']['binding_id']
        (runtime.workspace/'method-profiles'/binding/'upstream.py').write_text('changed')
    elif problem == 'approval':
        with runtime.db.transaction() as conn:
            conn.execute("UPDATE approvals SET status='REJECTED' WHERE run_id=?", (rid,))
    with pytest.raises(ValueError):
        enqueue_design(runtime, parent, **args)
    assert provider.calls == 0


def test_profile_is_checked_again_before_each_paid_stage(bound):
    runtime, provider, parent, args = bound
    rid = enqueue_design(runtime, parent, **args)['run']['run_id']
    binding = runtime.db.get_run(args['validation_run_id'])['metadata']['binding_id']
    (runtime.workspace/'method-profiles'/binding/'worker.py').write_text('changed')
    assert runtime.resume(rid)['run']['status'] == 'FAILED'
    assert provider.calls == 0


def test_receipt_changed_between_stages_stops_before_codegen_spend(bound):
    runtime, provider, parent, args = bound
    rid = enqueue_design(runtime, parent, **args)['run']['run_id']
    def mutate(value):
        if 'hypothesis' in value:
            receipt = next(a for a in runtime.db.list_artifacts(args['validation_run_id'])
                           if a['kind'] == 'method-verification')
            Path(receipt['path']).write_text('{}')
        return value
    provider.mutate = mutate
    assert runtime.resume(rid)['run']['status'] == 'FAILED'
    assert provider.calls == 1 and not runtime.db.list_artifacts(rid)


def test_completed_metadata_without_tool_result_is_not_a_validation(bound):
    runtime, provider, parent, args = bound
    with runtime.db.transaction() as conn:
        conn.execute('UPDATE tool_calls SET result_json=NULL WHERE run_id=?', (args['validation_run_id'],))
    with pytest.raises(ValueError, match='successful validation tool receipt'):
        enqueue_design(runtime, parent, **args)
    assert provider.calls == 0


@pytest.mark.parametrize('replacement', [
    'def forward(self, xA, xB):\n    return (xA + xB,)\n',  # AST-identical no-op.
    'def forward(self, xA):\n    return xA\n',
    'def forward(self, xA, xB):\n    return xA\nprint("must not run")\n',
    '@staticmethod\ndef forward(self, xA, xB):\n    return xA\n',
    'def forward(self, xA, xB):\n    return xA\ndef extra():\n    pass\n',
    'async def forward(self, xA, xB):\n    return xA\n',
    'def forward(self, xA, xB):\n    return (\n',
])
def test_out_of_scope_or_noop_patch_is_rejected(bound, replacement):
    runtime, provider, parent, args = bound
    def mutate(value):
        if 'replacement' in value:
            value['replacement'] = replacement
        return value
    provider.mutate = mutate
    rid = enqueue_design(runtime, parent, **args)['run']['run_id']
    result = runtime.resume(rid)
    assert result['run']['status'] == 'FAILED'
    assert not runtime.db.list_artifacts(rid) and not result['tool_calls']


def test_syntax_check_is_not_misrepresented_as_safety_review():
    # Valid Python with side effects is NEVER run here and must still require review.
    value = {'target': 'SNUNet_ECAM.forward',
             'replacement': 'def forward(self, xA, xB):\n    raise RuntimeError("MUST_NOT_RUN")\n',
             'review_notes': ['Deliberately invalid semantics; review must reject execution']}
    draft = build_patch(value, {'source': SOURCE})
    assert draft['contract']['executed'] is False
    assert 'CODE_AND_LICENSE_REVIEW' in draft['contract']['required_gates']
    assert 'NOT safety' in draft['contract']['assurance']
    ast.parse(draft['contract']['replacement'])


def test_logic_diff_ignores_indent_and_discloses_removed_calls():
    source = ('class SNUNet_ECAM:\n    def forward(self, xA, xB):\n'
              '        features = xA + xB\n        return (self.ca(features),)\n')
    value = {'target': 'SNUNet_ECAM.forward', 'replacement':
             'def forward(self, xA, xB):\n        features = xA + xB\n        return (features,)\n',
             'review_notes': ['Attention ablation; not validated']}
    contract = build_patch(value, {'source': source})['contract']
    assert contract['removed_direct_self_calls'] == ['ca']
    assert contract['added_direct_self_calls'] == []
    assert '-    return (self.ca(features),)' in contract['logic_diff']
    assert '+    return (features,)' in contract['logic_diff']
    assert '+        features' not in contract['logic_diff']


def test_host_review_is_bound_replayable_and_does_not_rewrite_model_answer(bound):
    runtime, provider, parent, args = bound
    rid = enqueue_design(runtime, parent, **args)['run']['run_id']
    result = runtime.resume(rid)
    raw = result['run']['metadata']['research_design_patch_raw']
    review_args = {'reviewer': 'host-test-review', 'decision': 'REVISION_REQUIRED',
                   'findings': ['Acceptance rule lacks a predeclared margin; do not train yet']}
    review = record_draft_review(runtime, rid, **review_args)
    assert review['execution_authorized'] is False
    assert record_draft_review(runtime, rid, **review_args) == review
    assert len(runtime.db.list_artifacts(rid)) == 3 and provider.calls == 2
    assert runtime.db.get_run(rid)['metadata']['research_design_patch_raw'] == raw
    assert runtime.db.list_tool_calls(rid) == []
    with pytest.raises(ValueError, match='separate execution review'):
        record_draft_review(runtime, rid, **{**review_args, 'decision': 'APPROVE_EXECUTION'})
    artifact = Path(result['run']['metadata']['research_design']['artifacts'][0]['path'])
    artifact.write_text('modified draft')
    with pytest.raises(ValueError, match='artifact changed'):
        record_draft_review(runtime, rid, **review_args)


def test_nonmatching_reference_is_rejected(bound):
    runtime, provider, parent, args = bound
    with runtime.db.transaction() as conn:
        conn.execute('UPDATE evidence SET content=? WHERE evidence_id=?', ('other-code', args['reference_ids'][0]))
    with pytest.raises(ValueError, match='exact validated source'):
        enqueue_design(runtime, parent, **args)
    assert provider.calls == 0


def test_web_can_request_bound_draft_but_cannot_supply_execution(bound):
    import time
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from paperpilot_agentic.web_api import create_fastapi_router
    runtime, provider, parent, args = bound
    app = FastAPI()
    app.include_router(create_fastapi_router(runtime_factory=lambda: runtime))
    with TestClient(app) as client:
        url = f'/agentic/runs/{parent}/research-design'
        assert client.post(url, json={**args, 'execute': True}).status_code == 422
        assert provider.calls == 0
        response = client.post(url, json=args)
        assert response.status_code == 202, response.text
        rid = response.json()['run']['run_id']
        for _ in range(150):
            result = runtime.status(rid)
            if result['run']['status'] in {'COMPLETED', 'FAILED'}:
                break
            time.sleep(.02)
        assert result['run']['status'] == 'COMPLETED', result['run'].get('error')
        assert result['run']['metadata']['research_design']['draft_mode'] == 'validated_method_patch'
