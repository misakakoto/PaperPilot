import hashlib
from types import SimpleNamespace

import pytest

from paperpilot_agentic.reader_visual import collect_visual_evidence, validate_visual_targets
from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.models import Budget, ToolResult
from paperpilot_agentic.llm_runtime import LLMBudgetExhausted


def test_named_cells_preserve_column_value_binding_and_source():
    from paperpilot_agentic.reader_visual import named_table_content
    import json
    rows = [{'row_label': 'joint', 'checked_columns': ['A', 'B'],
             'cells': {'No Dmg.': '94.97', 'F1loc(%)': '87.24'}}]
    item = {'content': 'Original pipe table', 'provenance': {'vision_metadata': {
        'column_headers': ['A', 'B', 'No Dmg.', 'F1loc(%)'], 'audited_table_rows': rows}}}
    caption = {'object_id': 'caption', 'document_sha256': 'fixture', 'page': 3,
               'bbox': [10, 20, 30, 40], 'text': 'Ablation on dataset X'}
    item['provenance']['nearby_caption_text'] = caption
    text = named_table_content(item, 'Table VII')
    assert json.loads(text.split('\n', 2)[2])['rows'] == rows
    assert json.loads(text.split('\n', 2)[2])['nearby_pdf_caption_text'] == caption
    assert item['content'] == 'Original pipe table'


@pytest.mark.parametrize('targets', [
    [{'paper_sha256': 'other', 'label': 'Table XII'}],
    [{'paper_sha256': 'sha', 'label': 'Table XII', 'pdf_path': 'arbitrary.pdf'}],
    [{'paper_sha256': 'sha', 'label': 'Table XII and VII'}],
    [{'paper_sha256': 'sha', 'label': 'Table I'}] * 2,
    [{'paper_sha256': 'sha', 'label': f'Table {n}'} for n in (1, 2, 3)],
])
def test_rejects_out_of_scope_paths_ambiguous_and_duplicate_targets(targets):
    with pytest.raises(ValueError):
        validate_visual_targets(targets, {'sha'})


def test_visual_bridge_resolves_hash_and_keeps_only_verified_transcriptions(tmp_path):
    pdf = tmp_path/'fixture.pdf'
    pdf.write_bytes(b'synthetic hash fixture, never parsed as PDF')
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    db = RuntimeDatabase(tmp_path/'runtime.sqlite3')
    rid = db.create_run('Read Table XII', Budget())
    calls = []
    def invoke(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(tool_call_id='tool_fixture', result=ToolResult(ok=True, evidence=[
            {'source_uri': f'paper://{sha}/layout/table', 'content': 'Row: 12',
             'evidence_type': 'TABLE', 'provenance': {'vision_verified': True,
                 'layout_object_id': 'table', 'layout_target_binding': {
                     'label': ['table', 'XII'], 'object_id': 'table', 'document_sha256': sha,
                     'basis': 'own_caption_or_heading'}}},
            {'source_uri': f'paper://{sha}/layout/unclear', 'content': 'Unclear',
             'provenance': {'vision_verified': False}}]))
    runtime = SimpleNamespace(db=db, tools=SimpleNamespace(invoke=invoke),
                              _resolve_pdf_paths=lambda scope: [str(pdf)])
    rows = collect_visual_evidence(runtime, rid, 'Read it', [{'paper_sha256': sha, 'label': 'Table XII'}],
                                   {sha}, lambda name, operation: operation())
    assert len(rows) == 1
    assert rows[0]['content'].startswith('[Vision-model transcription;')
    assert calls[0]['arguments']['pdf_path'] == str(pdf.resolve())
    assert calls[0]['arguments']['query'].startswith('Table XII.')
    assert 'Read it' not in calls[0]['arguments']['query']
    assert 'complete table' in calls[0]['arguments']['query']
    assert db.get_run(rid)['metadata']['reader_visual_skipped']
    pdf.write_bytes(b'changed fixture')
    with pytest.raises(ValueError, match='hash'):
        collect_visual_evidence(runtime, rid, 'Read it', [{'paper_sha256': sha, 'label': 'Table XII'}],
                                {sha}, lambda name, operation: operation())
    assert len(calls) == 1


@pytest.mark.parametrize('changed', ['label', 'document_sha256', 'object_id', 'basis', 'missing'])
def test_verified_transcription_cannot_substitute_a_different_table(tmp_path, changed):
    pdf = tmp_path/'fixture.pdf'
    pdf.write_bytes(b'synthetic hash fixture, never parsed')
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    db = RuntimeDatabase(tmp_path/'runtime.sqlite3')
    rid = db.create_run('Table XII', Budget())
    binding = {'label': ['table', 'XII'], 'object_id': 'table', 'document_sha256': sha,
               'basis': 'own_caption_or_heading'}
    if changed == 'missing':
        binding = {}
    else:
        binding[changed] = ['table', 'VII'] if changed == 'label' else 'wrong'
    outcome = SimpleNamespace(tool_call_id='unbound', result=ToolResult(ok=True, evidence=[{
        'source_uri': f'paper://{sha}/layout/table', 'content': 'A confidently transcribed wrong table',
        'provenance': {'vision_verified': True, 'layout_object_id': 'table',
                       'layout_target_binding': binding}}]))
    runtime = SimpleNamespace(db=db, _resolve_pdf_paths=lambda scope: [str(pdf)],
                              tools=SimpleNamespace(invoke=lambda **kwargs: outcome))
    rows = collect_visual_evidence(runtime, rid, 'Table XII', [{'paper_sha256': sha, 'label': 'Table XII'}],
                                   {sha}, lambda name, operation: operation())
    assert rows == []
    assert db.get_run(rid)['metadata']['reader_visual_skipped'][0]['reason'] == 'unbound_visual_target'


@pytest.mark.parametrize('limit', ['tokens', 'steps'])
def test_visual_bridge_preserves_final_answer_budget(tmp_path, limit):
    pdf = tmp_path/'fixture.pdf'
    pdf.write_bytes(b'synthetic hash fixture, not a real PDF')
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    db = RuntimeDatabase(tmp_path/'runtime.sqlite3')
    budget = Budget()
    if limit == 'tokens':
        budget.max_tokens = budget.synthesis_reserve_tokens + budget.repair_reserve_tokens
    else:
        budget.max_steps = 0
    rid = db.create_run('Read Table XII', budget)
    def unexpected_call(*args, **kwargs):
        pytest.fail('Exhausted visual budget must not dispatch a tool')
    runtime = SimpleNamespace(db=db, _resolve_pdf_paths=lambda scope: [str(pdf)])
    with pytest.raises(LLMBudgetExhausted):
        collect_visual_evidence(runtime, rid, 'Read it',
            [{'paper_sha256': sha, 'label': 'Table XII'}], {sha}, unexpected_call)


def test_visual_bridge_rejects_cross_paper_tool_result(tmp_path):
    pdf = tmp_path/'fixture.pdf'
    pdf.write_bytes(b'synthetic hash fixture, not a real PDF')
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    db = RuntimeDatabase(tmp_path/'runtime.sqlite3')
    rid = db.create_run('Read Table XII', Budget())
    outcome = SimpleNamespace(tool_call_id='wrong_paper', result=ToolResult(ok=True,
        evidence=[{'source_uri': 'paper://other/layout/table', 'content': '12',
                   'provenance': {'vision_verified': True}}]))
    runtime = SimpleNamespace(db=db, _resolve_pdf_paths=lambda scope: [str(pdf)],
                              tools=SimpleNamespace(invoke=lambda **kwargs: outcome))
    with pytest.raises(ValueError, match='another paper'):
        collect_visual_evidence(runtime, rid, 'Read it',
            [{'paper_sha256': sha, 'label': 'Table XII'}], {sha},
            lambda name, operation: operation())
