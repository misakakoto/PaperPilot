import json
from types import SimpleNamespace

from paperpilot_agentic.read_packet import build_read_packet, operation_definition_priority, purpose_definition_priority
from paperpilot_agentic.section_router import read_outline_fallback


def row(name, paper, text, section='Methods'):
    return {'evidence_id': name, 'source_uri': f'paper://{paper}/{name}',
            'page': 3, 'content': text, 'provenance': {'section_path': [section]}}


def test_chinese_mechanism_followup_ranks_definitions_without_repeating_names():
    question = '哪一种设计更直接处理空间错位？请只作机制层面的比较。'
    definition = ('We propose mechanisms. One modeling is unfolding the temporal tokens; '
                  'another works by concatenating them in the channel dimension.')
    abstract = 'We introduce a framework with three mechanisms and evaluate its results.'
    assert operation_definition_priority(question, definition) > 0
    assert operation_definition_priority(question, abstract) == 0

    chunks = [SimpleNamespace(stable_id='abstract',paper_sha256='p',title='Study',
                             section='Introduction',text=abstract,page_start=1),
              SimpleNamespace(stable_id='definition',paper_sha256='p',title='Study',
                             section='Methods',text=definition,page_start=3)]
    store = SimpleNamespace(get_chunks=lambda ids: [c for c in chunks if c.stable_id in ids])
    selected, metadata = read_outline_fallback(store,'p',
        [{'chunk_ids':['abstract','definition']}],question,max_passages=1)
    assert selected[0]['source_uri'].endswith('/definition')
    assert metadata['semantic_coverage'] == 'NOT_PROVEN'


def test_explicit_purpose_original_is_not_filtered_as_low_operation_overlap():
    sources = [row('a-method','a','Alpha is unfolding tokens. We propose Alpha to concatenate features.'),
               row('b-method','b','Beta uses warping. Features are first processed and fed into a warping layer.'),
               row('b-goal','b','To address spatial misalignment we propose Beta. The module is designed to estimate and compensate displacements.', 'Introduction')]
    packet = build_read_packet('Compare Alpha and Beta mechanisms and the problem each solves.', sources)
    assert {r['evidence_id'] for r in packet.rows} == {'a-method','b-method','b-goal'}
    assert all(r in sources for r in packet.rows)
    assert packet.metadata['semantic_coverage'] == 'NOT_PROVEN'


def test_purpose_navigation_does_not_override_requested_tables_or_truncate_sources():
    sources = [row('a-method','a','Alpha is unfolding tokens. We propose Alpha to concatenate features.'),
               row('b-method','b','Beta uses warping. Features are first processed and fed into a warping layer.'),
               row('b-goal','b','To address spatial misalignment we propose Beta.', 'Introduction'),
               row('a-extra','a','Alpha is often discussed in other experiments.', 'Introduction')]
    for paper in ('a','b'):
        table = row(paper+'-table',paper,'TABLE IV\nMeasured score: 62.30.')
        table['object_type'] = 'table'
        table['provenance']['structured_extraction'] = {'kind':'table',
            'source':{'paper_sha256':paper,'label':'Table IV'}}
        sources.append(table)
    wanted = sources[:3]+sources[-2:]
    limit = 2+sum(len(json.dumps({k:r.get(k) for k in ('evidence_id','source_uri','page','content')},ensure_ascii=False).encode())+2 for r in wanted)
    packet = build_read_packet('Compare Alpha and Beta mechanisms, what problem they solve, and Table IV.',sources,
        requested_tables=[{'paper_sha256':p,'label':'Table IV'} for p in ('a','b')],max_bytes=limit)
    assert {r['evidence_id'] for r in packet.rows} == {r['evidence_id'] for r in wanted}
    assert packet.metadata['context_upper_bytes'] <= limit
    assert packet.metadata['required_source_gaps'] == []


def test_goal_context_cannot_displace_each_papers_operation_definition():
    sources = [row('a-method','a','Alpha is unfolding tokens. We propose Alpha to concatenate features.'),
               row('b-method','b','Beta uses warping. Features are first processed and fed into a warping layer.'),
               row('b-goal','b','To address spatial misalignment we propose Beta.', 'Introduction'),
               row('b-extra','b','To address other issues we discuss Beta.', 'Introduction')]
    limit=2+sum(len(json.dumps({k:r.get(k) for k in ('evidence_id','source_uri','page','content')},ensure_ascii=False).encode())+2 for r in sources[:3])
    packet=build_read_packet('Compare Alpha and Beta mechanisms and what problem each solves.',sources,max_bytes=limit)
    assert {'a-method','b-method'} <= {r['evidence_id'] for r in packet.rows}
    assert set(packet.metadata['operation_definition_anchors']) == {'a-method','b-method'}


def test_named_goal_does_not_match_an_unrelated_hyphenated_term():
    question='What problem does Cross solve?'
    assert purpose_definition_priority(question,'To address the problem we use cross-attention.') == 0
    assert purpose_definition_priority(question,'To address the problem we propose Cross.') > 0
