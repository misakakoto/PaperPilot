"""Non-evidentiary parser notes must not weaken extraction gates."""
from copy import deepcopy
import hashlib
import json

import pytest

from paperpilot_agentic.qwen_vision import QwenVisionEvidenceAdapter, reaudit_qwen_result
from paperpilot_agentic.table_header_contract import header_candidates
from test_qwen_vision import _FakeClient, _payload


def extract(tmp_path, **fields):
    image = tmp_path / 'region.png'
    image.write_bytes(b'offline-crop')
    client = _FakeClient(_payload(**fields))
    result = QwenVisionEvidenceAdapter(api_key='offline', client=client).analyze(
        str(image), 'Read the table.', {'object_type': 'table'})
    return image, client, result


def test_explicit_header_note_is_retained_without_becoming_evidence(tmp_path):
    note = 'The parent heading groups two leaves; it has no data column.'
    image, client, result = extract(tmp_path, parser_notes=[note])
    assert result['verified'] is True
    assert result['metadata']['parser_notes'] == [note]
    assert result['metadata']['parser_notes_role'] == 'NOT_SOURCE_EVIDENCE'
    assert note not in result['description']
    assert result['metadata']['raw_provider_extraction'] == client.chat.completions.payload
    before = deepcopy(result)
    replay = reaudit_qwen_result(result, image, prompt='Read the table.')
    assert replay['verified'] is True
    assert replay['metadata']['parser_notes'] == [note]
    assert result == before


@pytest.mark.parametrize('uncertainty', ['列归属无法确定', 'The meaning of F1 is not defined within the crop.'])
def test_new_protocol_never_uses_prose_regex_to_clear_uncertainty(tmp_path, uncertainty):
    _, _, result = extract(tmp_path, parser_notes=['Spanning header'], uncertain_fields=[uncertainty])
    assert result['verified'] is False
    assert result['metadata']['blocking_uncertain_fields'] == [uncertainty]


def test_notes_do_not_override_structural_conflict(tmp_path):
    _, _, result = extract(tmp_path, parser_notes=['Everything looks correct'],
        column_headers=['Score', 'Other'])
    assert result['verified'] is False
    assert result['metadata']['table_structure_valid'] is False


def test_header_only_read_keeps_notes_but_rejects_uncertainty():
    payload = {'column_headers': ['Switch', 'Score'], 'indicator_columns': ['Switch'],
               'group_headers': ['Settings'], 'uncertain_fields': [], 'parser_notes': ['Parent tier']}
    assert header_candidates(payload)['parser_notes'] == ['Parent tier']
    with pytest.raises(ValueError, match='Uncertain'):
        header_candidates({**payload, 'uncertain_fields': ['unreadable leaf']})


@pytest.mark.parametrize('notes', [[{'field': 'Score'}], [''], ''])
def test_invalid_note_contract_cannot_be_verified(tmp_path, notes):
    with pytest.raises(ValueError):
        extract(tmp_path, parser_notes=notes)


def test_scalar_note_is_retained_but_declared_ambiguity_still_blocks(tmp_path):
    _, client, result = extract(tmp_path, parser_notes='Parent tier explanation',
        uncertain_fields=['Column association uncertain'])
    assert result['verified'] is False
    assert result['metadata']['parser_notes'] == ['Parent tier explanation']
    assert result['metadata']['raw_provider_extraction'] == client.chat.completions.payload
    assert result['metadata']['blocking_uncertain_fields'] == ['Column association uncertain']


def test_additive_notes_do_not_change_historical_row_identity():
    from paperpilot_agentic.table_claims import row_catalog
    from test_structured_extraction import table_item
    item = {**table_item(), 'evidence_id': 'table'}
    payload = item['provenance']['structured_extraction']
    del payload['parser_notes']
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert row_catalog([item])[0]['row_id'] == f'row_{expected[:24]}_0'
    assert 'parser_notes' not in payload
    payload['parser_notes'] = ['Layout explanation only']
    assert row_catalog([item])[0]['row_id'] == f'row_{expected[:24]}_0'
