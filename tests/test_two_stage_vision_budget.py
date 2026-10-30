import json
from types import SimpleNamespace

import pytest

from paperpilot_agentic.qwen_vision import QwenVisionEvidenceAdapter
from test_table_views import fixture


def test_second_stage_guard_keeps_first_stage_usage_without_sending(tmp_path):
    sent = []
    guarded = []

    def send(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(model='fixture', choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content=json.dumps({'column_headers': ['Flag', 'Score'],
                'indicator_columns': ['Flag'], 'group_headers': [], 'uncertain_fields': []})))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))

    def guard():
        guarded.append(True)
        if len(guarded) == 2:
            raise RuntimeError('No allowance for another request')

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=send)))
    adapter = QwenVisionEvidenceAdapter(api_key='fixture', client=client)
    with pytest.raises(RuntimeError) as failure:
        adapter.analyze(str(fixture(tmp_path)), 'Read the table', {'object_type': 'table'},
                        before_attempt=guard)
    assert len(sent) == 1
    assert len(guarded) == 2
    assert failure.value.vision_usage['prompt_tokens'] == 100
    assert failure.value.vision_usage['completion_tokens'] == 20
    assert failure.value.vision_usage['provider_attempts'] == 1


def test_uncertain_header_stops_before_numeric_generation(tmp_path):
    sent = []
    def send(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(model='fixture', choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content=json.dumps({'column_headers': ['Flag', 'Score'],
                'indicator_columns': ['Flag'], 'group_headers': [], 'uncertain_fields': ['Flag unclear']})))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=send)))
    adapter = QwenVisionEvidenceAdapter(api_key='fixture', client=client)
    with pytest.raises(ValueError) as failure:
        adapter.analyze(str(fixture(tmp_path)), 'Read the table', {'object_type': 'table'})
    assert len(sent) == 1
    assert failure.value.vision_usage['provider_attempts'] == 1
    assert failure.value.vision_usage['prompt_tokens'] == 100
