import json

from evals.reliability.reference_identity_probe import run
from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.source_references import REFERENCE_IDENTITY_POLICY


def test_probe_only_changes_reference_guidance_and_counts_each_attempt(tmp_path):
    prior = tmp_path/'prior'
    prior.mkdir()
    case = {'case': 8, 'question': 'Compare results.', 'original_question': 'Report both metrics.',
            'request_history': [], 'report': {'limitations': ['Table unavailable.']},
            'evidence': [{'evidence_id': 'e', 'source_uri': 'paper://sha/e',
                          'content': 'See Table TABREF35.'}]}
    (prior/'protocol.json').write_text(json.dumps({'cases': [case], 'dataset_sha256': 'fixture'}))
    requests = []

    class Provider:
        model = 'fixture'

        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            requests.append(messages)
            return LLMResponse(text=json.dumps({'action': 'STOP', 'reason': 'Unresolved reference.'}),
                               finish_reason='stop', prompt_tokens=10, completion_tokens=10)

    results = run(prior, Provider(), tmp_path/'output')
    assert [row['attempts'] for row in results] == [1, 1]
    assert all(row['status'] == 'RETURNED' for row in results)
    baseline = json.loads(requests[0][-1]['content'])
    treatment = json.loads(requests[1][-1]['content'])
    assert treatment.pop('reference_placeholders') == [{'evidence_id': 'e', 'tokens': ['TABREF35']}]
    assert baseline == treatment
    assert requests[0][0]['content'] == requests[1][0]['content'].replace(REFERENCE_IDENTITY_POLICY, '').rstrip()
    assert (tmp_path/'output'/'baseline'/'response.json').is_file()
    assert (tmp_path/'output'/'reference_identity'/'response.json').is_file()
