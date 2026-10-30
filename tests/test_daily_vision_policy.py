import json
from types import SimpleNamespace

from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.llm_attempts import AttemptLedger
from paperpilot_agentic.models import Budget
from paperpilot_agentic.model_policy import PRIMARY_MODEL, BACKUP_MODEL
from paperpilot_agentic.qwen_vision import QwenVisionEvidenceAdapter
from paperpilot_agentic.vision_dispatch import vision_input_upper_bound
from test_daily_availability import provider_error
from test_table_views import fixture


def test_quota_during_second_visual_stage_does_not_repeat_paid_header_read(tmp_path):
    sent = []
    headers = {'column_headers': ['Flag', 'Score'], 'indicator_columns': ['Flag'],
               'group_headers': [], 'uncertain_fields': []}
    rows = {'description': 'Table scores', 'transcription': 'Score 0.00', 'visible_values': ['0.00'],
            'column_headers': ['Flag', 'Score'], 'indicator_columns': ['Flag'],
            'table_rows': [{'row_label': '', 'checked_columns': [], 'cells': {'Flag': '', 'Score': '0.00'}}],
            'uncertain_fields': [], 'image_readable': True, 'evidence_relevant': True, 'confidence': .9}
    def send(**kwargs):
        sent.append(kwargs)
        if len(sent) == 2:
            raise provider_error()
        value = headers if len(sent) == 1 else rows
        return SimpleNamespace(model=kwargs['model'], choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content=json.dumps(value)))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=send)))
    adapter = QwenVisionEvidenceAdapter(api_key='fixture', model=PRIMARY_MODEL,
                                      fallback_models=(BACKUP_MODEL,), client=client)
    db = RuntimeDatabase(tmp_path / 'runtime.sqlite3')
    rid = db.create_run('Read the table.', Budget(max_tokens=200000), {'execution_profile': 'daily'})
    bound = adapter.for_run(db, rid)
    dispatched = []
    bound.analyze(str(fixture(tmp_path)), 'Read the table', {'object_type': 'table'},
                  on_dispatch=lambda: dispatched.append(True))
    assert [r['model'] for r in sent] == [PRIMARY_MODEL, PRIMARY_MODEL, BACKUP_MODEL]
    assert sent[1]['messages'] == sent[2]['messages']
    assert len(dispatched) == db.get_run(rid)['used_llm_calls'] == 3
    assert db.get_run(rid)['used_tokens'] == 240
    assert [c['status'] for c in db.list_llm_calls(rid)] == ['COMPLETED', 'FAILED', 'COMPLETED']
    assert AttemptLedger(db).snapshot(rid)['reserved_unknown_tokens'] > 0


def test_image_bound_counts_visual_tokens_not_base64_characters():
    def messages(size):
        return [{'role': 'user', 'content': [{'type': 'text', 'text': 'Read pixels.'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + 'a' * size}}]}]
    assert vision_input_upper_bound(messages(10)) == vision_input_upper_bound(messages(100000))
    assert vision_input_upper_bound(messages(10)) >= 16384
