import copy
import json
from types import SimpleNamespace

import pytest

from paperpilot_agentic.paper_structure import structure_from_objects, graph_identity, digest, save_json
from paperpilot_agentic.token_evidence import TokenBudget, child_chunks, build_index, TokenEvidenceAdapter, encoded
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.prototype_harness import PrototypeHarness


def graph(text):
    return structure_from_objects('a' * 64, 'Original paper', [{
        'text': text, 'page': 1, 'bbox': [0,0,100,100], 'object_id': 'source-object',
        'object_type': 'text', 'section_path': []}])


def register(root, g):
    folder = root / 'papers' / g['paper_sha256']
    folder.mkdir(parents=True)
    content = ('\n'.join(p['text'] for p in g['paragraphs']) + '\n').encode()
    (folder / 'paragraphs.txt').write_bytes(content)
    g['text_file_sha256'] = digest(content)
    g['identity'] = graph_identity(g)
    save_json(folder / 'structure.json', g)
    path = root / 'index.json'
    papers = json.loads(path.read_bytes())['papers'] if path.exists() else []
    save_json(path, {'papers': papers + [{'sha256': g['paper_sha256'], 'file_name': 'paper.pdf',
               'structure_sha256': digest((folder / 'structure.json').read_bytes())}]})


def actual_budget(folder):
    pytest.importorskip('fastembed')
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    folder.mkdir()
    t = Tokenizer(models.WordLevel({'[UNK]': 0, '[PAD]': 1, '[CLS]': 2, '[SEP]': 3}, unk_token='[UNK]'))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.post_processor = processors.TemplateProcessing(single='[CLS] $A [SEP]',
                                                     special_tokens=[('[CLS]', 2), ('[SEP]', 3)])
    t.save(str(folder / 'tokenizer.json'))
    save_json(folder / 'config.json', {'pad_token_id': 1})
    save_json(folder / 'tokenizer_config.json', {'model_max_length': 32, 'pad_token': '[PAD]'})
    save_json(folder / 'special_tokens_map.json', {'unk_token': '[UNK]', 'pad_token': '[PAD]'})
    return TokenBudget.from_directory(folder)


def fixture_index(tmp_path):
    source = tmp_path / 'source'
    register(source, graph('Alpha mechanism preserves all original constraints. ' * 30))
    foreign = graph('Beta has separate evidence.')
    foreign['paper_sha256'] = 'b' * 64
    register(source, foreign)
    budget = actual_budget(tmp_path / 'tokenizer')
    workspace = tmp_path / 'workspace'
    destination = workspace / 'token-evidence-pilot'
    original = (source / 'index.json').read_bytes()
    manifest = build_index(source, destination, budget)
    assert (source / 'index.json').read_bytes() == original
    base = SimpleNamespace(embedder=None, reranker=None, paper_roots=(), rerank_stage='final')
    adapter = TokenEvidenceAdapter(destination, base)
    return workspace, destination, manifest, adapter, base


def test_utf8_children_are_lossless_and_include_navigation_in_budget():
    g = graph('Alpha method. 中文符号αβ。 More exact source text. ' * 30)
    original = copy.deepcopy(g)
    budget = TokenBudget(64, lambda s: len(s.encode()) + 2, {'type': 'utf8-test'})
    chunks, links, parents = child_chunks(g, 'P1', budget)
    assert ''.join(c.text for c in chunks) == g['paragraphs'][0]['text']
    assert g == original and len(parents) == 1 and len(links) == len(chunks)
    assert all(budget.count(f'{c.section} {c.text}'.strip()) <= 64 for c in chunks)
    for identity, child in links.items():
        p = parents[child['parent_id']]
        assert digest(p['text'][child['start']:child['end']].encode()) == child['text_sha256']
        assert child['source_segments'][0]['object_id'] == 'source-object'


def test_whole_parent_read_is_deduplicated_measured_and_never_silently_truncated(tmp_path):
    _, _, _, adapter, _ = fixture_index(tmp_path)
    hits = adapter.search('Alpha', top_k=3)
    ids = [h['stable_id'] for h in hits]
    assert hits and all(h['provenance']['input_tokens'] <= 32 for h in hits)
    result = adapter.read_context(ids, max_tokens=4096)
    assert len(result['evidence']) == 1 and result['status'] == 'COMPLETE'
    assert result['context_tokens'] == adapter.budget.count(encoded(result['evidence']).decode())
    assert result['context_tokens'] <= 4096
    assert result['evidence'][0]['content'] == ' '.join(['Alpha mechanism preserves all original constraints.'] * 30)
    partial = adapter.read_context(ids, max_tokens=32)
    assert partial['status'] == 'PARTIAL' and not partial['evidence'] and partial['omitted_parent_ids']
    assert ids == [h['stable_id'] for h in hits]
    with pytest.raises(ValueError, match='scope'):
        adapter.read_context(ids, paper_sha256s={'b' * 64})
    with pytest.raises(ValueError, match='owned'):
        adapter.read_context(['forged-child'])


def test_cross_page_child_retains_intersecting_original_spans():
    g = graph('First page continues')
    p = g['paragraphs'][0]
    first_end = len(p['text'])
    p['text'] += ' on second page.'
    p['spans'].append({'page': 2, 'bbox': [0,0,100,100], 'object_id': 'second',
                       'text_start': first_end + 1, 'text_end': len(p['text'])})
    g['identity'] = graph_identity(g)
    chunks, children, _ = child_chunks(g, 'P1', TokenBudget(128, len, {'test': 'characters'}))
    assert chunks[0].page_start == 1 and chunks[0].page_end == 2
    assert {s['page'] for s in next(iter(children.values()))['source_segments']} == {1, 2}


def test_manifest_and_tokenizer_tampering_fail_closed(tmp_path):
    _, destination, manifest, adapter, base = fixture_index(tmp_path)
    tokenizer = manifest['tokenizer']
    from pathlib import Path
    path = Path(tokenizer['directory']) / 'tokenizer_config.json'
    original = path.read_bytes()
    path.write_bytes(original + b' ')
    with pytest.raises(ValueError, match='Tokenizer'):
        TokenEvidenceAdapter(destination, base)
    path.write_bytes(original)
    identity = next(iter(adapter.evidence['parents']))
    adapter.evidence['parents'][identity]['text'] = 'corrupted original'
    child = next(iter(adapter.evidence['children']))
    with pytest.raises(ValueError, match='parent identity'):
        adapter.read_context([child])
    path = destination / 'evidence.json'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='index changed'):
        TokenEvidenceAdapter(destination, base)


def test_real_tool_route_uses_frozen_profile_and_preserves_source_links(tmp_path):
    workspace, destination, _, _, base = fixture_index(tmp_path)
    runtime = ResearchAgentRuntime(workspace=workspace, db_path=tmp_path / 'runs.db', search_adapter=base)
    run = runtime.enqueue(question='Alpha mechanism', metadata={
        'retrieval_profile': 'token_evidence_pilot', 'paper_sha256s': ['a' * 64]})['run']
    context = SimpleNamespace(run_id=run['run_id'], workspace=workspace, db=runtime.db)
    search = runtime.registry.get('paper.search').handler(context, {'query': 'Alpha', 'top_k': 2})
    assert search.evidence and search.evidence[0]['provenance']['source_segments']
    assert search.evidence[0]['provenance']['retrieval_profile'] == 'token_evidence_pilot'
    ids = [e['provenance']['stable_id'] for e in search.evidence]
    read = runtime.registry.get('paper.read_context').handler(context, {'child_ids': ids, 'max_tokens': 4096})
    assert read.ok and read.evidence[0]['provenance']['hit_child_ids'] == list(dict.fromkeys(ids))
    with pytest.raises(ValueError, match='scope'):
        runtime.registry.get('paper.search').handler(context, {'query': 'Beta', 'paper_sha256s': ['b' * 64]})
    with pytest.raises(ValueError, match='scope'):
        runtime.registry.get('paper.read_pages').handler(context, {'paper_sha256': 'b' * 64, 'pages': [1]})
    original = (destination / 'index.json').read_bytes()
    (destination / 'index.json').write_bytes(original + b' ')
    with pytest.raises(ValueError, match='changed'):
        runtime.registry.get('paper.search').handler(context, {'query': 'Alpha'})


def test_prototype_can_opt_in_without_changing_the_default(tmp_path, monkeypatch):
    workspace, _, _, _, base = fixture_index(tmp_path)
    from test_prototype_workspace import Reader
    runtime = ResearchAgentRuntime(workspace=workspace, db_path=tmp_path / 'runs.db', search_adapter=base, llm=Reader())
    monkeypatch.setattr(runtime.db, 'list_evidence', lambda _: [{'evidence_id': 'source', 'content': 'owned'}])
    h = PrototypeHarness(runtime)
    rid = h.start('bounded model', 'pilot-task', 'parent', ['source'],
                  retrieval_profile='token_evidence_pilot')['run']['run_id']
    assert runtime.db.get_run(rid)['metadata']['retrieval_profile'] == 'token_evidence_pilot'
    assert h._load(rid)['retrieval_identity'] == runtime.db.get_run(rid)['metadata']['retrieval_identity']
    default = h.start('bounded model', 'legacy-task', 'parent', ['source'])['run']['run_id']
    assert runtime.db.get_run(default)['metadata'].get('retrieval_profile', 'legacy') == 'legacy'


def test_source_handle_view_preserves_full_originals_and_spends_budget_on_text(tmp_path):
    _, destination, _, adapter, _ = fixture_index(tmp_path)
    before = (destination / 'evidence.json').read_bytes()
    ids = [h['stable_id'] for h in adapter.search('Alpha', top_k=3)]
    full = adapter.read_context(ids, max_tokens=4096)
    assert 'view_policy' not in full
    cap = full['context_tokens'] - 1
    assert not adapter.read_context(ids, max_tokens=cap)['evidence']
    compact = adapter.read_context(ids, max_tokens=cap, view='source_handles')
    assert compact['status'] == 'COMPLETE' and compact['context_tokens'] <= cap
    assert compact['context_tokens'] == adapter.budget.count(encoded(compact['evidence']).decode())
    selected = compact['evidence'][0]
    assert selected['content'] == full['evidence'][0]['content']
    assert selected['source_uri'] == full['evidence'][0]['source_uri']
    assert selected['provenance']['page_range'] == [1, 1]
    assert 'source_segments' not in selected['provenance']
    restored = adapter.read_context(selected['provenance']['hit_child_ids'], max_tokens=4096)
    assert restored['evidence'] == full['evidence']
    assert compact['original_read']['tool'] == 'paper.read_context'
    assert (destination / 'evidence.json').read_bytes() == before
    with pytest.raises(ValueError, match='view'):
        adapter.read_context(ids, view='unregistered-summary')
    with pytest.raises(ValueError, match='scope'):
        adapter.read_context(ids, view='source_handles', paper_sha256s={'b' * 64})


def test_source_handle_view_is_explicit_on_the_real_tool_route(tmp_path):
    workspace, _, _, _, base = fixture_index(tmp_path)
    runtime = ResearchAgentRuntime(workspace=workspace, db_path=tmp_path / 'runs.db', search_adapter=base)
    run = runtime.enqueue(question='Alpha', metadata={
        'retrieval_profile': 'token_evidence_pilot', 'paper_sha256s': ['a' * 64]})['run']
    context = SimpleNamespace(run_id=run['run_id'], workspace=workspace, db=runtime.db)
    search = runtime.registry.get('paper.search').handler(context, {'query': 'Alpha'})
    ids = [e['provenance']['stable_id'] for e in search.evidence]
    tool = runtime.registry.get('paper.read_context')
    compact = tool.handler(context, {'child_ids': ids, 'view': 'source_handles', 'max_tokens': 4096})
    assert compact.output['view_policy'] == 'source-handles-v1'
    assert compact.evidence and all('source_segments' not in e['provenance'] for e in compact.evidence)
    full = tool.handler(context, {'child_ids': ids, 'max_tokens': 4096})
    assert full.evidence[0]['provenance']['source_segments']
    assert compact.output['index_identity'] == full.output['index_identity']


def test_source_handles_keep_cross_page_locations_resolvable(tmp_path):
    g = graph('First page continues onto the second page.')
    paragraph = g['paragraphs'][0]
    split = paragraph['text'].index('onto')
    paragraph['spans'][0]['text_end'] = split
    paragraph['spans'].append({'page': 2, 'bbox': [0, 0, 100, 100], 'object_id': 'second-page',
                               'text_start': split, 'text_end': len(paragraph['text'])})
    source = tmp_path / 'source'
    register(source, g)
    budget = actual_budget(tmp_path / 'tokenizer')
    destination = tmp_path / 'token-pilot'
    build_index(source, destination, budget)
    adapter = TokenEvidenceAdapter(destination, SimpleNamespace(embedder=None, reranker=None, paper_roots=()))
    ids = list(adapter.evidence['children'])
    compact = adapter.read_context(ids, view='source_handles', max_tokens=4096)
    row = compact['evidence'][0]
    assert row['provenance']['page_range'] == [1, 2]
    original = adapter.read_context(row['provenance']['hit_child_ids'], max_tokens=4096)
    assert original['evidence'][0]['source_uri'] == row['source_uri']
    assert {s['page'] for s in original['evidence'][0]['provenance']['source_segments']} == {1, 2}
