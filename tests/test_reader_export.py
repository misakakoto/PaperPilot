import json
import sys

from evals.qasper.export_reader_cases import main


def test_export_counts_routing_and_reader_calls(tmp_path, monkeypatch):
    folder = tmp_path/'case'
    folder.mkdir()
    (folder/'reader-result.json').write_text(json.dumps({
        'question': 'Question?',
        'routing': {'provider_attempts': 1, 'calls': [{'prompt_tokens': 20, 'completion_tokens': 5}]},
        'reader': {'provider_attempts': 2, 'calls': [{'prompt_tokens': 30, 'completion_tokens': 8}],
                   'status': 'FAILED', 'draft': {'facts': []}}
    }), encoding='utf-8')
    output = tmp_path/'report.md'
    monkeypatch.setattr(sys, 'argv', ['export', str(folder), '--output', str(output)])
    main()
    assert '合计3次请求，63 tokens' in output.read_text(encoding='utf-8')


def test_export_prefers_durable_usage_over_returned_response_list(tmp_path, monkeypatch):
    source = tmp_path/'report.json'
    source.write_text(json.dumps({'question': 'Question?',
        'reader': {'status': 'FAILED', 'provider_attempts': 1, 'calls': []},
        'persisted_run': {'used_llm_calls': 2, 'used_tokens': 800}}), encoding='utf-8')
    output = tmp_path/'review.md'
    monkeypatch.setattr(sys, 'argv', ['export', str(source), '--output', str(output)])
    main()
    assert '合计2次请求，800 tokens' in output.read_text(encoding='utf-8')
