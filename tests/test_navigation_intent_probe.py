import json

from evals.reliability.navigation_intent_probe import run
from paperpilot_agentic.providers import LLMResponse


def plan():
    return {'code_sha256': 'frozen', 'max_requests': 4, 'cases': [{
        'case': 0, 'question': 'Old query', 'original_question': 'Use validation instead.',
        'request_history': [{'user_request': 'Report test results.'}],
        'outline': [{'section_id': 'x', 'title': 'Setup', 'chars': 8}],
        'evidence': [{'evidence_id': 'e1', 'content': 'Original passage.', 'source_uri': 'paper://p/x'}],
        'report': {'limitations': ['Missing result']}}]}


def test_probe_changes_only_navigation_intent_not_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr('evals.reliability.navigation_intent_probe.code_identity', lambda: 'frozen')
    captured = []
    class Provider:
        model = 'fixture'
        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            payload = json.loads(next(message['content'] for message in messages if message['role'] == 'user'))
            captured.append(payload)
            value = {'section_ids': ['x'], 'reason': 'Fixture'} if 'outline' in payload else {
                'action': 'STOP', 'reason': 'Fixture'}
            return LLMResponse(text=json.dumps(value), finish_reason='stop', prompt_tokens=3, completion_tokens=2)
    results = run(plan(), Provider(), tmp_path / 'probe')
    assert len(results) == 4 and sum(row['attempts'] for row in results) == 4
    assert all(row['status'] == 'RETURNED' for row in results)
    for baseline, candidate in ((captured[0], captured[1]), (captured[2], captured[3])):
        assert candidate['original_question'] == 'Use validation instead.'
        assert baseline == {key: value for key, value in candidate.items()
                            if key not in {'original_question', 'request_history'}}


def test_probe_records_unattempted_jobs_when_provider_denies_access(tmp_path, monkeypatch):
    monkeypatch.setattr('evals.reliability.navigation_intent_probe.code_identity', lambda: 'frozen')
    class Denied(RuntimeError):
        status_code = 403
    class Provider:
        model = 'fixture'
        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            raise Denied('Fixture')
    results = run(plan(), Provider(), tmp_path / 'probe')
    assert sum(row['attempts'] for row in results) == 1
    assert len(results) == 4 and all(row['status'] == 'FAILED' for row in results)
    assert sum(row.get('error_type') == 'ProviderCircuitOpen' for row in results) == 3
