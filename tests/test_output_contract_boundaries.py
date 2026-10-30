"""Output contracts stay visible after specialized prompt construction."""
import json
from types import SimpleNamespace

from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.qwen_vision import QwenVisionEvidenceAdapter, _normalize_named_cells
from test_qwen_vision import _FakeClient, _payload


def test_scalar_note_normalization_is_lossless_and_never_clears_uncertainty():
    raw = {'parser_notes': 'Parent heading spans two leaves.',
           'uncertain_fields': ['Left mark cannot be located.']}
    normalized, changes = _normalize_named_cells(raw)
    assert normalized['parser_notes'] == [raw['parser_notes']]
    assert normalized['uncertain_fields'] == raw['uncertain_fields']
    assert raw['parser_notes'] == 'Parent heading spans two leaves.'
    assert changes[0]['operation'] == 'scalar_note_to_array'


def test_specialized_table_read_retains_typed_output_example(tmp_path, monkeypatch):
    import paperpilot_agentic.table_views as views
    monkeypatch.setattr(views, 'row_focus_view', lambda path: (None, {'status': 'source_row_crops'}))
    image = tmp_path / 'crop.png'
    image.write_bytes(b'offline')
    client = _FakeClient(_payload(column_headers=['Score']))
    original = client.chat.completions.create
    captured = []

    def create(**kwargs):
        captured.append(kwargs)
        if len(captured) == 1:
            return SimpleNamespace(model='offline', usage=None, choices=[SimpleNamespace(
                finish_reason='stop', message=SimpleNamespace(content=json.dumps({
                    'column_headers': ['Score'], 'indicator_columns': [], 'group_headers': [],
                    'uncertain_fields': [], 'parser_notes': []})))])
        return original(**kwargs)

    client.chat.completions.create = create
    QwenVisionEvidenceAdapter(api_key='offline', client=client).analyze(
        str(image), 'Read Table IV.', {'object_type': 'table'})
    prompt = captured[1]['messages'][0]['content'][0]['text']
    assert '"parser_notes":[]' in prompt
    assert '"cells":' in prompt
    assert 'Copy row labels literally' in prompt
    assert 'fallible navigation hints' in prompt


def test_auditor_has_explicit_output_key_boundary_with_all_sources_preserved():
    captured = []
    draft = {'facts': [{'id': 'f1', 'kind': 'SOURCE_FACT', 'text': 'Uses a local encoder.',
                        'evidence_ids': ['e1']}], 'direct_answer_id': 'f1', 'limitations': []}
    audit = {'assessments': [{'fact_id': 'f1', 'span_ids': ['s0'], 'verdict': 'SUPPORTED',
                             'relevance': 'DIRECT', 'reason': 'Explicitly stated.'}],
             'coverage': 'COMPLETE', 'missing_aspects': []}

    class Model:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            captured.append(messages)
            return LLMResponse(json.dumps(draft if len(captured) == 1 else audit), finish_reason='stop')

    answer_then_audit('Describe the encoder.', [
        {'evidence_id': 'e1', 'content': 'Uses a local encoder.'},
        {'evidence_id': 'e2', 'content': 'The uncited encoder description remains available for coverage.'}], Model())
    system = captured[1][0]['content']
    assert 'AUDIT OUTPUT BOUNDARY' in system
    assert 'Never return sources, facts, allowed_span_ids' in system
    assert 'Close the JSON object immediately' in system
    payload = json.loads(captured[1][-1]['content'])
    assert len(payload['sources']) == 2
