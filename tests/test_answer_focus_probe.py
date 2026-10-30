import json

from evals.reliability.answer_focus_probe import BASE_FOCUS, FOCUSED_WRITER, run
from paperpilot_agentic.providers import LLMResponse


def test_matched_probe_changes_only_writer_paragraph(tmp_path, monkeypatch):
    monkeypatch.setattr('evals.reliability.answer_focus_probe.code_identity', lambda: 'frozen')
    captured = []

    class Provider:
        model = 'fixture'

        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            captured.append(messages)
            payload = json.loads(next(m['content'] for m in messages if m['role'] == 'user'))
            if 'evidence' in payload:
                value = {'facts': [{'id': 'f1', 'kind': 'SOURCE_FACT', 'text': 'A source fact.',
                                   'evidence_ids': ['e1']}], 'direct_answer_id': 'f1', 'limitations': []}
            else:
                value = {'assessments': [{'fact_id': 'f1', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                                         'span_ids': ['s0']}], 'coverage': 'COMPLETE', 'missing_aspects': []}
            return LLMResponse(text=json.dumps(value), finish_reason='stop')

    plan = {'code_sha256': 'frozen', 'max_requests': 6, 'cases': [
        {'question': 'What is the fact?', 'resolved_question': 'What is the fact?', 'question_id': 'q1',
         'evidence': [{'evidence_id': 'e1', 'content': 'A source fact.'}]}]}
    results = run(plan, Provider(), tmp_path / 'probe')
    assert len(results) == 2 and len(captured) == 4
    assert all(r['report']['status'] == 'REVIEW_REQUIRED' for r in results)
    assert captured[0][0]['content'].count(BASE_FOCUS) == 1
    assert captured[2][0]['content'] == captured[0][0]['content'].replace(BASE_FOCUS, FOCUSED_WRITER)
    assert captured[1] == captured[3]
    assert all('expected' not in m['content'] for request in captured for m in request)


def test_permission_denial_stops_unsent_jobs_and_keeps_failures(tmp_path, monkeypatch):
    monkeypatch.setattr('evals.reliability.answer_focus_probe.code_identity', lambda: 'frozen')
    class Denied(RuntimeError):
        status_code = 403

    class Provider:
        model = 'fixture'
        calls = 0

        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            self.calls += 1
            raise Denied('Fixture denial, not a credential')

    plan = {'code_sha256': 'frozen', 'max_requests': 24, 'cases': [
        {'question': 'Question?', 'resolved_question': 'Question?', 'question_id': f'q{i}',
         'evidence': [{'evidence_id': 'e1', 'content': 'A source fact.'}]} for i in range(4)]}
    provider = Provider()
    results = run(plan, provider, tmp_path / 'probe')
    assert len(results) == 8
    assert 1 <= provider.calls <= 2  # At most the two concurrent jobs were in flight.
    assert sum(r['attempts'] for r in results) == provider.calls
    assert all(r['report']['status'] == 'FAILED' for r in results)
    assert any(r['report']['error_type'] == 'ProviderCircuitOpen' for r in results)
