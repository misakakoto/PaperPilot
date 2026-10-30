import json
from types import SimpleNamespace
from paperpilot_agentic.read_packet import build_read_packet
from paperpilot_agentic.answer_pipeline import AnswerOptions, initialize, writer_messages, prepared_messages, auditor_messages


def evidence(identifier, text, section='Methods'):
    return {'evidence_id': identifier, 'source_uri': 'paper://p/' + identifier,
            'page': 3, 'content': text, 'provenance': {'section_path': [section]}}


def test_writer_and_auditor_share_exact_read_identity_and_no_back_matter():
    sources = [evidence('method', 'Widget alternates temporal tokens along spatial positions.'),
               evidence('ref', '[1] Widget studies pp. 1-5 IEEE', 'References')]
    ctx = initialize('Explain Widget mechanism', sources, SimpleNamespace(), AnswerOptions(), None)
    w = json.loads(prepared_messages(ctx, writer_messages(ctx))[-1]['content'])
    a = json.loads(prepared_messages(ctx, auditor_messages(ctx, {'question': ctx.question,
        'facts': [], 'sources': ctx.rows}, []))[-1]['content'])
    assert w['read_packet_sha256'] == a['read_packet_sha256']
    assert ctx.rows[0]['content'] == sources[0]['content']
    assert 'ref' not in ctx.aliases.decode_map.values()
    assert 'IEEE' not in json.dumps(w) + json.dumps(a)


def test_whole_paragraph_limit_records_required_gap_not_truncation():
    source = evidence('long', 'x' * 2000)
    packet = build_read_packet('mechanism', [source], max_bytes=100)
    assert packet.rows == []
    assert packet.metadata['unread_navigation_only'][0]['reason'] == 'whole_source_does_not_fit'
    assert packet.metadata['semantic_coverage'] == 'NOT_PROVEN'


def test_explicit_bibliography_request_allows_back_matter():
    source = evidence('ref', 'References\n[1] Alpha', 'References')
    assert build_read_packet('List references', [source]).rows == [source]


def test_unknown_query_terms_never_discard_all_retrieved_originals():
    sources = [evidence('a', 'The baseline uses a linear classifier.'),
               evidence('b', 'The baseline uses a neural classifier.')]
    assert build_read_packet('Question?', sources).rows == sources
    assert build_read_packet('What classifier?', sources).rows == sources


def test_paper_title_and_across_are_not_cross_mechanism_matches():
    sources = [evidence('a', 'Widget\nCross alternates tokens.'),
               evidence('b', 'Widget\nThe paper works across different datasets.')]
    packet = build_read_packet('Explain Cross in 《Widget》', sources)
    assert packet.rows == sources[:1]


def test_query_function_words_do_not_displace_the_requested_method():
    sources = [evidence('a', 'Uses a local encoder.', section=''),
               evidence('b', 'The encoder preserves resolution.', section='')]
    assert build_read_packet('Describe the encoder.', sources).rows == sources


def test_explicit_metric_question_reads_tables_without_the_word_table():
    source = evidence('t', 'F1 62.30')
    source['object_type'] = 'table'
    assert build_read_packet('给出 F1', [source]).rows == [source]


def test_legacy_table_heading_retains_mismatching_metric_for_audit():
    sources = [evidence('m', 'Uses a local encoder.', section=''),
               evidence('t', 'TABLE IV\nAccuracy on the BRIGHT test set is 96.68%.', section='')]
    assert build_read_packet('Describe the encoder and measured F1.', sources).rows == sources
    assert build_read_packet('Describe the encoder mechanism only.', sources).rows == sources[:1]
