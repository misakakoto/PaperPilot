import json

from paperpilot_agentic.numeric_scope import table_references
from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.reader_recovery import plan_evidence_reload
from paperpilot_agentic.source_references import REFERENCE_IDENTITY_POLICY, reference_placeholders


def test_placeholders_do_not_become_printed_numbers():
    content = 'Table TABREF35; TABREF35, Figure FIGREF12, FLOATREF7; BIBREF8; Table 2.'
    rows = [{'evidence_id': 'original', 'content': content}]
    assert reference_placeholders(rows) == [{'evidence_id': 'original',
                                             'tokens': ['TABREF35', 'FIGREF12', 'FLOATREF7']}]
    assert [row['table'] for row in table_references(content)] == [2]
    assert rows[0]['content'] == content


def test_recovery_gets_literal_reference_inventory_without_a_fabricated_mapping():
    inputs = []

    class Model:
        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            inputs.extend(messages)
            return LLMResponse(text=json.dumps({'action': 'STOP', 'reason': 'Missing table mapping'}),
                               finish_reason='stop', prompt_tokens=10, completion_tokens=10)

    candidates = [
        {'evidence_id': 'read', 'source_uri': 'paper://sha/read', 'content': 'Table TABREF35 has the results.'},
        {'evidence_id': 'unread', 'source_uri': 'paper://sha/unread',
         'content': 'Table TABREF12. ' + 'a' * 600 + ' Table TABREF99.'},
    ]
    result = plan_evidence_reload('What are the scores?', {'limitations': ['No numeric table available.']},
                                  candidates, {'read'}, Model(), allow_search=True)
    payload = json.loads(inputs[-1]['content'])
    assert REFERENCE_IDENTITY_POLICY in inputs[0]['content']
    assert payload['reference_placeholders'] == [
        {'evidence_id': 'read', 'tokens': ['TABREF35']},
        {'evidence_id': 'unread', 'tokens': ['TABREF12']},
    ]
    assert payload['read_evidence'][0]['content'] == candidates[0]['content']
    assert result['provider_attempts'] == 1
