"""Evaluate the diagnostic protocol and replay accounting with a substitute model."""
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.research_role_eval import CASES, run_comparison
from test_research_harness import Provider


def test_same_three_call_schedule_and_completed_replay_without_new_calls(tmp_path):
    responses = []
    for case in CASES:
        for _ in range(2):
            responses.extend([{'decision': case['expected'], 'evidence_ids': case['required'],
                               'rationale': 'Controlled substitute, not measured model quality'}]*3)
    provider = Provider(responses)
    runtime = ResearchAgentRuntime(db_path=tmp_path/'db.sqlite3', workspace=tmp_path/'work',
                                   llm=provider, text_reader_protocol='eval-fixture')
    first = run_comparison(runtime, request_id='controlled-diagnostic')
    assert len(first['results']) == 6
    assert all(row['passed'] and row['requests'] == 3 for row in first['results'])
    assert len(provider.messages) == 18
    assert run_comparison(runtime, request_id='controlled-diagnostic') == first
    assert len(provider.messages) == 18
    # Shared context carries prior raw assistant messages; isolated handoff does not.
    assert any(m['role'] == 'assistant' for m in provider.messages[1])
    assert not any(m['role'] == 'assistant' for m in provider.messages[4])


def test_wrong_decision_or_missing_sources_cannot_be_counted_as_pass(tmp_path):
    provider = Provider([{'decision': 'ACCEPT', 'evidence_ids': [], 'rationale': 'Invalid'}]*18)
    runtime = ResearchAgentRuntime(db_path=tmp_path/'db.sqlite3', workspace=tmp_path/'work',
                                   llm=provider, text_reader_protocol='eval-fixture')
    result = run_comparison(runtime, request_id='controlled-invalid')
    assert not any(row['passed'] for row in result['results'])
