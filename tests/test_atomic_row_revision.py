import json

from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.durable_reader import run_durable_reader, DurableResearchModel, persist_reader_evidence
from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
from paperpilot_agentic.models import Budget
from paperpilot_agentic.providers import LLMResponse
from test_table_claim_rows import table


def test_owned_table_value_survives_rejection_of_separate_overclaim_and_replay(tmp_path):
    source = table()
    parsed = source['provenance']['structured_extraction']
    source['content'] = 'Table IV\nNamed-cell transcription (each row is independent):\n' + json.dumps(
        {key: parsed[key] for key in ('column_headers', 'table_rows')}, ensure_ascii=False).replace('table_rows', 'rows')

    class Model:
        model = 'fixture-atomic-row-revision'
        max_retries = 0
        calls = 0

        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            self.calls += 1
            payload = json.loads(messages[-1]['content'])
            if 'evidence' in payload:
                row = payload['table_rows'][1]
                answer = {'facts': [{'id': 'f', 'kind': 'TABLE_FACT', 'operation': 'value',
                    'row_refs': [{'evidence_id': row['evidence_id'], 'row_id': row['row_id'], 'metrics': ['F1']}]},
                    {'id': 'bad', 'kind': 'INFERENCE', 'text': '该结果保证提升精度。',
                     'evidence_ids': [row['evidence_id']], 'depends_on': ['f']}],
                    'direct_answer_id': 'f', 'limitations': []}
            else:
                answer = {'assessments': [{'fact_id': f, 'verdict': 'SUPPORTED',
                          'relevance': 'DIRECT', 'span_ids': ['s0']} for f in ('f', 'bad')],
                          'coverage': 'COMPLETE', 'missing_aspects': []}
            return LLMResponse(json.dumps(answer, ensure_ascii=False), finish_reason='stop')

    model = Model()
    db = RuntimeDatabase(tmp_path / 'runtime.sqlite3')
    rid = db.create_run('引用 Table IV 一个消融结果。', Budget(max_llm_calls=12),
                        metadata={'requested_table_targets': [{'paper_sha256': 'p', 'label': 'Table IV'}]})
    source = persist_reader_evidence(db, rid, [source])[0]
    bridge = DurableResearchModel(PersistentLLMRuntime(db, model), rid,
                                  stage='answer_first', reserve_class='synthesis')
    revised = run_durable_reader(db, rid, [source], bridge, protocol_id='atomic-v1', finalize=False)
    assert '62.3' in revised['answer']
    assert '保证提升' not in revised['answer']
    assert revised['answer_scope'] == 'PARTIAL'
    assert any(a['fact_id'] == 'f' and a['verdict'] == 'SUPPORTED' for a in revised['audit']['assessments'])
    assert any(a['fact_id'] == 'bad' and a['verdict'] != 'SUPPORTED' for a in revised['audit']['assessments'])
    assert model.calls == 2
    replay = run_durable_reader(db, rid, [source], bridge, protocol_id='atomic-v1', finalize=False)
    assert replay == revised and model.calls == 2
