import copy
import json

import pytest

from paperpilot_agentic import runtime_profile
from paperpilot_agentic.context_compiler import ContextCompiler
from paperpilot_agentic.prototype_workspace import digest
from test_runtime_v2_context import Owner
from test_prototype_candidate_policy import prepared, work, SOURCE


def latest(state):
    runtime_profile.attach(state, getattr(runtime_profile, 'DEFAULT_PROFILE', runtime_profile.PROFILE))


def compile_feedback(raw, *, artifacts=None, remaining=50000):
    state = {'payload': {'goal': 'repair submission', 'active_task_id': 'code',
                        'implementation_feedback': {'error': 'Expected source and config',
                                                    'rejected_candidate': raw},
                        'context_artifacts': {}},
             'work': {'artifacts': artifacts or {}}}
    latest(state)
    original = copy.deepcopy(state)
    result = ContextCompiler().compile(Owner(), 'r', state,
        {'remaining_tokens': remaining, 'remaining_llm_calls': 1, 'unknown_results': 0})
    assert state == original
    return result


def test_unstored_rejected_code_remains_available_to_the_next_decision():
    raw = {'source': SOURCE}
    result = compile_feedback(raw)
    assert result.manifest['status'] == 'READY'
    payload = json.loads(result.messages[-1]['content'])
    assert payload['implementation_feedback']['rejected_candidate'] == raw


def test_unreadable_rejected_code_cannot_be_deleted_to_make_budget_fit():
    result = compile_feedback({'source': SOURCE + '# important failed code\n' * 5000}, remaining=9000)
    assert result.manifest['status'] == 'BLOCKED'
    assert result.manifest['reason'] == 'REQUIRED_CONTEXT_EXCEEDS_BUDGET'


def test_only_exact_immutable_artifact_can_replace_a_feedback_payload():
    raw = {'source': SOURCE, 'config': {}}
    artifact = {'kind': 'implement', 'name': 'code', 'inputs': [], 'content': raw}
    identity = digest(artifact)
    result = compile_feedback(raw, artifacts={identity: artifact})
    feedback = json.loads(result.messages[-1]['content'])['implementation_feedback']
    assert feedback['rejected_candidate_ref']['artifact_id'] == identity
    assert feedback['rejected_candidate_ref']['read_action'] == 'inspect_artifact'
    corrupt = {identity: {**artifact, 'content': {'source': 'changed', 'config': {}}}}
    result = compile_feedback(raw, artifacts=corrupt)
    assert json.loads(result.messages[-1]['content'])['implementation_feedback']['rejected_candidate'] == raw


def test_latest_profile_normalizes_source_object_and_still_requires_approval(tmp_path, monkeypatch):
    runtime, harness, rid, state = prepared(tmp_path, monkeypatch)
    latest(state)
    observation = work(harness, rid, state, 'code', {'source': SOURCE})
    artifact = state['work']['artifacts'][state['work']['tasks']['code']['output']]
    assert artifact['content'] == {'source': SOURCE, 'config': {}}
    normalization = state['implementation_normalizations'][-1]
    assert normalization['config_origin'] == 'HOST_DECLARED_EMPTY_DEFAULT'
    assert normalization['source_unchanged'] is True
    harness._after_action(rid, state, 'lead', observation)
    assert runtime.db.get_approval(state['pending']['approval'])['status'] == 'PENDING'
    assert runtime.db.get_tool_call(state['pending']['call'])['attempt'] == 0


@pytest.mark.parametrize('content', [{'source': SOURCE, 'extra': True},
                                    {'source': SOURCE, 'config': None}])
def test_default_config_never_repairs_unknown_fields_or_explicit_invalid_config(tmp_path, monkeypatch, content):
    _, harness, rid, state = prepared(tmp_path, monkeypatch)
    latest(state)
    with pytest.raises(ValueError):
        work(harness, rid, state, 'code', content)


def test_v2_keeps_its_original_object_submission_protocol(tmp_path, monkeypatch):
    _, harness, rid, state = prepared(tmp_path, monkeypatch)
    runtime_profile.attach(state, runtime_profile.PROFILE)
    with pytest.raises(ValueError, match='Expected source and config'):
        work(harness, rid, state, 'code', {'source': SOURCE})


def test_unknown_profile_cannot_pass_as_a_null_configuration():
    state = {'runtime_profile': {'id': 'unknown-profile', 'config': None, 'sha256': digest(None)}}
    with pytest.raises(ValueError, match='unknown'):
        runtime_profile.enabled(state)


def test_profile_mutation_is_rejected_before_feedback_projection():
    state = {}
    latest(state)
    state['runtime_profile']['config']['source_object_default_config'] = False
    with pytest.raises(ValueError, match='identity changed'):
        runtime_profile.enabled(state)
