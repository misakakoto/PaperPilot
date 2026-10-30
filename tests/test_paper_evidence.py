from types import SimpleNamespace

import pytest

from paperpilot_agentic.paper_catalog import observation, record, frontmatter_observations, compare_reports, crossref_observations
from paperpilot_agentic.paper_evidence_index import build_index
from paperpilot_agentic.paper_evidence import PaperEvidence, upper, route_question
from paperpilot_agentic.paper_evidence_ranking import bounded_rerank
from paperpilot_agentic.prototype_paper_evidence import POLICY, prepare, validate_analysis
from paperpilot_agentic.prototype_workspace import digest
from test_token_evidence import actual_budget

A, B = 'a' * 64, 'b' * 64


def observed(field, value):
    return observation(field, value, source={'kind': 'user', 'uri': 'annotation://reviewed'})


def unit(uid, text, sha=A, **kwargs):
    return {'id': uid, 'paper_sha256': sha, 'section': 'Methods', 'text': text,
            'source': {'page_range': [2, 2], 'source_uri': f'paper://{sha}/page/2'}, **kwargs}


class Embedder:
    cache_identity = 'test-embedding-v1'

    def embed_documents(self, texts):
        return [[1., 0.] if 'NEEDLE' in t else [0., 1.] for t in texts]

    def embed_query(self, text):
        return [1., 0.]


def fixture(tmp_path, units=None, *, dense=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    budget = actual_budget(tmp_path / 'tokenizer')
    root = tmp_path / 'paper-evidence-v2'
    papers = [{'sha256': A, 'title': 'Alpha fusion', 'authors': ['Alice'], 'file_name': 'Alpha2026.pdf',
               'observations': [observed('year', 2021), observed('venue', 'Journal A')]},
              {'sha256': B, 'title': 'Beta mechanism', 'file_name': 'Beta2020.pdf'}]
    embedder = Embedder() if dense else None
    build_index(papers, units if units is not None else [unit('u1', 'Alpha uses fusion with a NEEDLE.'),
                unit('u2', 'Beta describes a separate mechanism.', B)], root, budget, embedder=embedder)
    return root, PaperEvidence(root, embedder=embedder)


def test_catalog_fields_never_come_from_filename_or_references():
    rows = frontmatter_observations('References\nPublished in 2024 DOI:10.1234/not-this-paper', A)
    rec = record(A, 'published-2026.pdf', rows)
    assert rec['fields']['year']['status'] == 'MISSING'
    assert rec['fields']['doi']['status'] == 'MISSING'
    with pytest.raises(ValueError, match='filenames'):
        observation('year', 2026, source={'kind': 'filename', 'uri': 'file.pdf'})
    rec = record(A, 'paper.pdf', [observed('year', 2021), observed('year', 2022)])
    assert rec['fields']['year']['status'] == 'CONFLICT' and rec['fields']['year']['value'] is None
    p = crossref_observations({'DOI': '10.1234/same', 'published': {'date-parts': [[2021]]}}, '10.1234/same')
    one, two = record(A, 'preprint.pdf', p), record(B, 'published.pdf', p)
    assert one['work_id'] == two['work_id'] and one['document_version'] != two['document_version']
    assert next(o for o in p if o['field'] == 'year')['value'] == 2021
    with pytest.raises(ValueError, match='match'):
        crossref_observations({'DOI': '10.1234/wrong'}, '10.1234/same')
    rows = frontmatter_observations('Title\nABSTRACT\nArchitecture (abbreviated by ChangeFormer).\nIndex Terms: change detection', A)
    abstract = next(r for r in rows if r['field'] == 'abstract')
    assert abstract['value'] == 'Architecture (abbreviated by ChangeFormer).'
    assert abstract['source']['kind'] == 'frontmatter'


def test_catalog_filters_explicit_unknowns_and_empty_scope(tmp_path):
    _, lib = fixture(tmp_path)
    assert [r['paper_id'] for r in lib.search_papers('Alpha', filters={'year_from': 2021})['papers']] == [A]
    result = lib.search_papers('Beta', filters={'year_to': 2025})
    assert not result['papers'] and result['unresolved_filter_fields'] == [{'paper_id': B, 'field': 'year'}]
    assert not lib.search_papers('Alpha', paper_ids=[])['papers']
    assert not lib.rank_units('Alpha', paper_ids=[])
    with pytest.raises(ValueError, match='scope'):
        lib.search_papers('Alpha', paper_ids=['c' * 64])
    with pytest.raises(ValueError):
        lib.search_papers('Alpha', filters={'doi': 'part-of-doi'})


def test_scope_is_applied_before_global_candidate_limit(tmp_path):
    units = [unit(f'foreign-{i}', 'NEEDLE exact evidence', B) for i in range(105)]
    units += [unit('owned', 'NEEDLE exact evidence plus many irrelevant words ' * 10)]
    _, lib = fixture(tmp_path, units)
    assert [r['id'] for r in lib.rank_units('NEEDLE', paper_ids=[A])] == ['owned']


def test_encoding_windows_aggregate_to_whole_original_units_without_duplicate_slots(tmp_path):
    text = 'irrelevant words. ' * 100 + 'NEEDLE at the tail. ' * 30
    root, lib = fixture(tmp_path, [unit('long', text), unit('other', 'NEEDLE short')], dense=True)
    rows = lib.rank_units('NEEDLE', limit=8)
    assert len(rows) == 2 and len({r['id'] for r in rows}) == 2
    assert next(r for r in rows if r['id'] == 'long')['text'] == text
    with lib.connect() as c:
        windows = list(c.execute('SELECT * FROM windows WHERE unit_id=? ORDER BY start', ('long',)))
    assert len(windows) > 2 and ''.join(w['text'] for w in windows) == text
    assert all(w['tokens'] <= lib.budget.limit for w in windows)
    assert all(a['end'] == b['start'] for a, b in zip(windows, windows[1:]))
    assert lib.read_sources(['long'])['evidence'][0]['content'] == text
    with pytest.raises(ValueError, match='embedding identity'):
        PaperEvidence(root, embedder=SimpleNamespace(cache_identity='changed'))


@pytest.mark.parametrize('cap', [1024, 1800, 3000, 6000, 16000])
def test_full_packet_budget_includes_navigation_matrices_and_conditions(tmp_path, cap):
    _, lib = fixture(tmp_path, [unit('u1', 'Alpha fusion ' * 150), unit('u2', 'Beta mechanism ' * 60, B)])
    result = lib.retrieve_evidence('Compare Alpha and Beta', paper_ids=[A, B],
                                   facets=['fusion', 'conditions', 'experiment'], budget_upper=cap)
    if result['status'] == 'BLOCKED':
        assert not result['evidence'] and result['required_upper'] > cap
    else:
        assert upper(result) <= result['estimated_upper'] <= cap
        assert all(e['content'] in ['Alpha fusion ' * 150, 'Beta mechanism ' * 60] for e in result['evidence'])
        assert len(result['matrix']) == 2 and result['diagnostics']['coverage_is_candidate_retrieval_not_entailment']
    with pytest.raises(ValueError, match='Source ID'):
        lib.read_sources(['u1'], paper_ids=[B])
    with pytest.raises(ValueError):
        lib.retrieve_evidence('Alpha', facets=[])


def table_unit():
    text = 'Dataset LEVIR-CD test split. Method SNUNet. F1 (%) 93.5'
    return unit('table', text, kind='TABLE', object={'column_headers': ['Method', 'F1 (%)'],
        'source': {'page': 2}, 'parse_status': 'PARSED',
        'num_rows': 2, 'num_cols': 2, 'cells': [
            {'row': 0, 'column': 0, 'text': 'Method', 'column_header': True},
            {'row': 0, 'column': 1, 'text': 'F1 (%)', 'column_header': True},
            {'row': 1, 'column': 0, 'text': 'SNUNet'},
            {'row': 1, 'column': 1, 'text': '93.5'}]}, reports=[{
            'id': 'r1', 'method': 'SNUNet', 'dataset': 'LEVIR-CD', 'split': 'test', 'metric': 'F1',
            'unit': '%', 'value': '93.5', 'source_cell': {'row': 1, 'column': 1},
            'context_quotes': {k: text for k in ['method', 'dataset', 'split', 'metric', 'unit']}}])


def test_numeric_report_keeps_cell_and_conditions_and_refuses_fabricated_context(tmp_path):
    _, lib = fixture(tmp_path, [table_unit()])
    result = lib.retrieve_evidence('SNUNet F1', paper_ids=[A], facets=['F1'], budget_upper=16000)
    report = result['reported_results'][0]
    assert report['value'] == '93.5' and report['source_unit_id'] == 'table' and report['paper_sha256'] == A
    assert result['comparison']['status'] == 'NOT_COMPARABLE'
    assert {'pretraining', 'input_size', 'training_protocol'} <= set(result['comparison']['missing_conditions']['r1'])
    bad = table_unit()
    bad['reports'][0]['dataset'] = 'invented-dataset'
    with pytest.raises(ValueError, match='literal quote'):
        fixture(tmp_path / 'bad', [bad])


def test_conditions_can_bind_original_text_in_same_paper_but_not_another_pdf_version(tmp_path):
    table = table_unit()
    report = table['reports'][0]
    report['pretraining'] = 'none'
    report['context_sources'] = {'pretraining': {'source_unit_id': 'settings', 'quote': 'Pretraining: none'}}
    _, lib = fixture(tmp_path, [table, unit('settings', 'Pretraining: none')])
    assert lib.retrieve_evidence('SNUNet F1', paper_ids=[A], facets=['F1'])['reported_results'][0]['pretraining'] == 'none'
    with pytest.raises(ValueError, match='escaped this paper version'):
        fixture(tmp_path / 'foreign', [table, unit('settings', 'Pretraining: none', B)])


def test_report_selection_recognizes_model_name_without_ours_or_reference_decoration(tmp_path):
    from test_paper_document import raw_document
    from paperpilot_agentic.paper_document import document_units
    from paperpilot_agentic.paper_tables import literal_reports
    raw = raw_document()
    cells = raw['document']['tables'][0]['data']['table_cells']
    next(c for c in cells if c['text'] == 'A')['text'] = 'ChangeFormer (ours)'
    table = next(u for u in document_units(raw, A) if u['kind'] == 'TABLE')
    table['reports'] = literal_reports(table)
    for report in table['reports']:
        report['id'] = 'z_requested' if report['method'].startswith('ChangeFormer') else 'baseline_'+report['method']
    _, lib = fixture(tmp_path, [table])
    packet = lib.retrieve_evidence('ChangeFormer F1', paper_ids=[A], facets=['F1'])
    assert packet['reported_results'][0]['method'] == 'ChangeFormer (ours)'


def test_comparison_requires_matching_reported_protocols():
    one = {'id': 'r1', 'dataset': 'LEVIR', 'split': 'test', 'metric': 'F1', 'unit': '%',
           'pretraining': 'none', 'input_size': '256x256', 'training_protocol': 'Adam 100 epochs'}
    two = {**one, 'id': 'r2'}
    assert compare_reports([one, two])['status'] == 'REPORTED_PROTOCOLS_MATCH'
    two['split'] = 'val'
    assert compare_reports([one, two])['different_conditions'] == ['split']
    two['split'] = None
    assert compare_reports([one, two])['status'] == 'NOT_COMPARABLE'


def test_ranker_uses_actual_pair_budget_without_changing_originals():
    ranker = SimpleNamespace(max_length=64, pair_count=lambda q, p: len(q) + len(p) + 3,
                             score=lambda q, ps: [float(len(p)) for p in ps])
    passages = ['first ' * 100, 'second short']
    before = list(passages)
    scores, view = bounded_rerank(ranker, 'query ' * 100, passages)
    assert len(scores) == 2 and passages == before and view['truncated']
    assert all(n <= 64 for n in view['pair_tokens'])


def test_immutable_index_scope_and_real_registered_tools(tmp_path):
    root, lib = fixture(tmp_path)
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    base = SimpleNamespace(embedder=None, reranker=None, paper_roots=())
    runtime = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path / 'runs.db', search_adapter=base)
    run = runtime.enqueue(question='Alpha fusion', metadata={'retrieval_profile': 'paper_evidence_v2',
        'retrieval_identity': lib.index_identity, 'paper_sha256s': [A]})['run']
    context = SimpleNamespace(run_id=run['run_id'], workspace=tmp_path, db=runtime.db)
    found = runtime.registry.get('paper.search_papers').handler(context, {'query': 'Alpha', 'limit': 6})
    assert found.ok and found.evidence[0]['evidence_type'] == 'METADATA'
    read = runtime.registry.get('paper.retrieve_evidence').handler(context, {'question': 'Alpha', 'paper_ids': [A]})
    assert read.ok and read.evidence[0]['provenance']['unit_id'] == 'u1'
    with pytest.raises(ValueError, match='scope'):
        runtime.registry.get('paper.retrieve_evidence').handler(context, {'question': 'Beta', 'paper_ids': [B]})
    (root / 'evidence.sqlite3').write_bytes((root / 'evidence.sqlite3').read_bytes() + b' ')
    with pytest.raises(ValueError, match='changed'):
        runtime.registry.get('paper.search_papers').handler(context, {'query': 'Alpha'})


def test_prototype_profile_pins_unified_evidence_policy_without_affecting_legacy(tmp_path, monkeypatch):
    _, lib = fixture(tmp_path)
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from paperpilot_agentic.prototype_harness import PrototypeHarness
    from test_prototype_workspace import Reader
    runtime = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path / 'runs.db', llm=Reader(),
                                  search_adapter=SimpleNamespace(embedder=None, reranker=None, paper_roots=()))
    monkeypatch.setattr(runtime.db, 'list_evidence', lambda _: [{'evidence_id': 'source', 'content': 'owned'}])
    h = PrototypeHarness(runtime)
    result = h.start('research model', 'v2-start', 'parent', ['source'], literature_enabled=True,
                     retrieval_profile='paper_evidence_v2')
    state = h._load(result['run']['run_id'])
    assert state['literature_evidence_policy'] == POLICY
    assert state['retrieval_identity'] == lib.index_identity
    from paperpilot_agentic.prototype_workspace import plan
    plan(state['work'], [{'id': 'search', 'goal': 'Discover sources', 'capability': 'search_literature',
                         'depends_on': [], 'acceptance': 'Actual source catalogue'}], enforce_capabilities=True)
    assert h._decision_output_tokens(state) == 2000
    state['literature_evidence_policy'] = 'task-bound-paper-evidence-v2.1'
    assert h._decision_output_tokens(state) == 5000
    legacy = h.start('research model', 'legacy-start', 'parent', ['source'], literature_enabled=True)
    assert 'literature_evidence_policy' not in h._load(legacy['run']['run_id'])


def task_state():
    content = {'ok': True, 'tool': 'paper.search_papers', 'output': {'index_identity': 'index', 'papers': [{'paper_id': A}]}}
    artifact = {'kind': 'search_literature', 'content': content}
    aid = digest(artifact)
    return {'literature_evidence_policy': POLICY, 'retrieval_identity': 'index', 'work': {
        'tasks': {'search': {'capability': 'search_literature', 'status': 'DONE', 'output': aid}}, 'artifacts': {aid: artifact}}}


def test_task_reading_requires_discovery_and_supplement_requires_prior_read():
    state = task_state()
    task = {'capability': 'read_literature', 'depends_on': ['search']}
    content = {'paper_ids': [A], 'question': 'fusion', 'facets': ['mechanism']}
    assert prepare(state, task, content)[0] == 'paper.retrieve_evidence'
    with pytest.raises(ValueError, match='direct catalogue'):
        prepare(state, task, {**content, 'paper_ids': [B]})
    with pytest.raises(ValueError, match='prior reading'):
        prepare(state, task, {**content, 'supplement': True})
    with pytest.raises(ValueError, match='direct reading'):
        prepare(state, task, {'source_ids': ['unseen-but-valid-unit-id']})
    assert state == task_state()
    with pytest.raises(ValueError, match='legacy paper_sha256/pages'):
        prepare(state, task, {'paper_sha256': A, 'pages': [1]})


def test_task_focused_prompt_and_navigation_use_only_the_selected_reading_protocol():
    from paperpilot_agentic.prototype_prompt import instructions
    from paperpilot_agentic.prototype_paper_evidence import navigation_view
    state = task_state()
    state['work']['tasks']['active'] = {'capability': 'read_literature'}
    text = instructions(state, {'active_task_id': 'active'})
    assert 'paper_ids' in text and 'source_ids' in text
    assert 'Pages may be beyond page 3' not in text
    read = {'kind': 'read_literature', 'content': {'ok': True, 'tool': 'paper.retrieve_evidence',
        'output': {'index_identity': 'index', 'matrix': [], 'status': 'PARTIAL'},
        'evidence': [{'content': 'Original only in analysis/full receipt', 'provenance': {
            'unit_id': 'unit', 'paper_sha256': A, 'page_range': [2, 2]}}]}}
    before = digest(read)
    view = navigation_view(before, read)
    assert digest(read) == before and 'evidence' not in view['content']
    assert view['content']['source_catalog'][0]['source_id'] == 'unit'


def test_query_routes():
    assert route_question('作者和年份') == 'catalog'
    assert route_question('F1消融指标') == 'experiment'
    assert route_question('比较两个模型的F1和融合') == 'comparison'
    assert route_question('解释融合机制') == 'method'


def test_method_reading_retains_two_original_prose_units_instead_of_title_fragment(tmp_path):
    originals = [unit('heading', 'Alpha'),
        unit('mechanism', 'Alpha aligns feature maps from multiple scales before concatenating their representations for fusion.'),
        unit('assumption', 'Alpha assumes the two temporal images share a spatially aligned coordinate system for fusion.')]
    _, lib = fixture(tmp_path, originals)
    packet = lib.retrieve_evidence('Alpha fusion mechanism', paper_ids=[A], facets=['mechanism'])
    assert {e['provenance']['unit_id'] for e in packet['evidence']} == {'mechanism', 'assumption'}
    assert all(e['content'] in [u['text'] for u in originals] for e in packet['evidence'])
    assert all(e['title'] == 'Alpha fusion' and e['provenance']['title_status'] == 'KNOWN' for e in packet['evidence'])
    assert upper(packet) <= packet['budget_upper']
    assert lib.read_sources(['heading'])['evidence'][0]['content'] == 'Alpha'


def test_experiment_reading_pairs_original_table_with_original_conditions(tmp_path):
    _, lib = fixture(tmp_path, [unit('heading', 'Alpha F1'),
        unit('table', 'Alpha F1 91.2 validation', kind='TABLE'),
        unit('conditions', 'Alpha uses the validation split with no additional pretraining and reports F1 under fixed experimental conditions.')])
    packet = lib.retrieve_evidence('Alpha F1实验条件', paper_ids=[A],
        facets=['experimental results', 'experimental conditions'])
    assert {e['provenance']['unit_id'] for e in packet['evidence']} == {'table', 'conditions'}
    # An addressable table with an unvalidated grid is not admitted as a fact.
    assert packet['reported_results'] == [] and packet['comparison']['status'] == 'NOT_COMPARABLE'
    assert packet['matrix'][0]['facets']['experimental conditions'] == ['conditions']


def test_reading_policy_and_source_snapshot_have_distinct_identities(tmp_path):
    root, lib = fixture(tmp_path)
    assert lib.index_identity != lib.source_index_identity
    with pytest.raises(ValueError, match='version changed'):
        PaperEvidence(root, expected_identity=lib.source_index_identity)


def test_unsourced_numeric_finding_is_rejected_without_claiming_semantic_verification():
    content = {'findings': [{'interpretation': '作者报告 F1 为 99.9%。', 'quote': 'F1: 93.5%'}]}
    with pytest.raises(ValueError, match='numeric literal'):
        validate_analysis(content, {'tasks': {}, 'artifacts': {}}, {'depends_on': []})
    content['findings'][0]['interpretation'] = '作者报告 F1 为 93.5%。'
    out = validate_analysis(content, {'tasks': {}, 'artifacts': {}}, {'depends_on': []})
    assert out['experiment_comparison']['verdict'] == 'NO_AUTOMATED_WINNER'


def test_one_targeted_supplement_and_source_audit_use_direct_owned_receipts():
    state = task_state()
    read = {'kind': 'read_literature', 'content': {'ok': True, 'tool': 'paper.retrieve_evidence',
        'output': {'index_identity': 'index', 'question': 'original', 'matrix': [{'facets': {'mechanism': ['u']}}],
                   'supplement_round': 0},
        'evidence': [{'provenance': {'unit_id': 'u', 'paper_sha256': A}}]}}
    aid = digest(read)
    state['work']['artifacts'][aid] = read
    state['work']['tasks']['read'] = {'capability': 'read_literature', 'status': 'DONE', 'output': aid}
    task = {'capability': 'read_literature', 'depends_on': ['search', 'read']}
    content = {'paper_ids': [A], 'question': 'target missing condition', 'facets': ['condition'], 'supplement': True}
    assert prepare(state, task, content)[0] == 'paper.retrieve_evidence'
    assert prepare(state, task, {'source_ids': ['u'], 'view': 'full'})[1]['paper_ids'] == [A]
    read['content']['output']['supplement_round'] = 1
    # Replace the changed receipt with a new identity, exactly as the work store does.
    state['work']['artifacts'].pop(aid)
    aid = digest(read)
    state['work']['artifacts'][aid] = read
    state['work']['tasks']['read']['output'] = aid
    with pytest.raises(ValueError, match='Only one'):
        prepare(state, task, content)
