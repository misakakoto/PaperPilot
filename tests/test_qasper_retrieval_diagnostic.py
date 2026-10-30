import json

import pytest

from evals.qasper.retrieval_diagnostic import run, text_metrics
from evals.qasper.agent_case import paper_records


def qa(evidence, qid='q1'):
    return {'question_id': qid, 'question': 'Which zebras migrate?',
            'answers': [{'answer': {'unanswerable': False, 'evidence': evidence}}]}


def test_metrics_keep_unmatched_gold_and_remove_duplicate_hits():
    result = text_metrics(qa(['one', 'missing']), ['one', 'one'], ['one'])
    assert result['best_text_recall'] == {'1': .5, '3': .5, '8': .5}
    assert result['best_corpus_text_coverage'] == .5
    assert text_metrics(qa(['one']), [], ['one'])['best_text_recall']['8'] == 0


def test_figures_and_empty_annotations_are_not_perfect_text_retrieval():
    result = text_metrics(qa(['FLOAT SELECTED Table 1']), [], [])
    assert result['best_text_recall'] is None
    assert result['figure_annotations'] == 1
    question = qa(['one', 'two'])
    question['answers'].extend(qa(['three'])['answers'])
    assert text_metrics(question, ['three'], ['three'])['best_text_recall']['8'] == 1


def test_real_store_runs_original_questions_without_indexing_gold(tmp_path):
    data = {'p1': {'title': 'Zebras', 'abstract': 'Zebras migrate north.', 'full_text': [],
                   'figures_and_tables': [], 'qas': [qa(['SECRET_GOLD_ONLY'])]}}
    dataset = tmp_path/'qasper-dev-v0.3.json'
    dataset.write_text(json.dumps(data), encoding='utf-8')
    output = tmp_path/'run'
    summary = run(dataset, output)
    assert summary['queries'] == 1 and summary['failed_queries'] == 0
    assert summary['macro_best_text_recall']['8'] == 0
    assert summary['provider_requests'] == 0
    row = json.loads((output/'rankings.jsonl').read_text(encoding='utf-8'))
    assert row['question'] == data['p1']['qas'][0]['question']
    assert row['hits'] and all('SECRET_GOLD_ONLY' not in hit['content'] for hit in row['hits'])
    with pytest.raises(FileExistsError):
        run(dataset, output)


def test_duplicate_ids_and_test_split_rejected_before_creating_output(tmp_path):
    dataset = tmp_path/'qasper-dev-v0.3.json'
    dataset.write_text(json.dumps({'p': {'qas': [qa([]), qa([])]}}), encoding='utf-8')
    with pytest.raises(ValueError):
        run(dataset, tmp_path/'run')
    assert not (tmp_path/'run').exists()
    with pytest.raises(ValueError):
        run(tmp_path/'qasper-test-v0.3.json', tmp_path/'run')


def test_positive_hit_resolves_original_text_and_no_text_reference_stays_null(tmp_path):
    dataset = tmp_path/'qasper-dev-v0.3.json'
    dataset.write_text(json.dumps({'p': {'title': 'Zebras', 'abstract': 'Zebras migrate north.',
        'full_text': [], 'qas': [qa(['Zebras migrate north.']), qa([], 'q2')]}}), encoding='utf-8')
    summary = run(dataset, tmp_path/'run')
    assert summary['queries'] == 2 and summary['text_metric_denominator'] == 1
    assert summary['no_text_reference_queries'] == 1
    assert summary['macro_best_text_recall']['1'] == 1


def test_search_failure_remains_in_denominator(tmp_path, monkeypatch):
    from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter

    def fail(*args, **kwargs):
        raise RuntimeError('synthetic retrieval failure')

    monkeypatch.setattr(PaperPilotKnowledgeSearchAdapter, 'search', fail)
    dataset = tmp_path/'qasper-dev-v0.3.json'
    dataset.write_text(json.dumps({'p': {'title': 'Zebras', 'abstract': 'Zebras migrate north.',
        'full_text': [], 'qas': [qa(['Zebras migrate north.'])]}}), encoding='utf-8')
    summary = run(dataset, tmp_path/'run')
    assert summary['failed_queries'] == summary['text_metric_denominator'] == 1
    assert summary['macro_best_text_recall']['8'] == 0
    assert summary['macro_best_corpus_text_coverage'] == 1


def test_null_section_title_keeps_paragraph_without_gold_or_invented_title():
    paragraph = 'Original scientific paragraph.'
    public, document, chunks = paper_records({'title': 'Paper', 'abstract': 'Abstract text.',
        'full_text': [{'section_name': None, 'paragraphs': [paragraph]}],
        'qas': [qa(['DO NOT INDEX'])]})
    assert 'qas' not in public
    assert chunks[-1].section == '' and chunks[-1].text == paragraph
    assert document.pages[-1].text == paragraph


def test_ranking_probe_reuses_candidates_without_mutating_baseline():
    from copy import deepcopy
    from evals.qasper.retrieval_diagnostic import RankingProbe
    from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter

    items = [{'stable_id': str(i), 'paper_sha256': 'p', 'score': 1 - i * .01,
              'content': 'shared words' if i < 3 else 'different material'} for i in range(4)]
    before = deepcopy(items)
    probe = RankingProbe(None)
    selected = probe._diversify(items, top_k=3)
    assert selected == PaperPilotKnowledgeSearchAdapter(None)._diversify(items, top_k=3)
    assert probe.no_similarity_penalty == PaperPilotKnowledgeSearchAdapter(None)._diversify(
        items, top_k=3, diversity=0)
    assert probe.candidates == before and items == before


def test_paired_diagnostic_keeps_same_query_and_candidate_pool(tmp_path):
    dataset = tmp_path/'qasper-dev-v0.3.json'
    dataset.write_text(json.dumps({'p': {'title': 'Zebras', 'abstract': 'Zebras migrate north.',
        'full_text': [], 'qas': [qa(['Zebras migrate north.'])]}}), encoding='utf-8')
    summary = run(dataset, tmp_path/'run', compare_ranking=True)
    assert summary['ranking_comparison']['paired_top8'] == {'improved': 0, 'regressed': 0, 'unchanged': 1}
    assert summary['ranking_comparison']['macro_candidate_text_coverage'] == 1
    row = json.loads((tmp_path/'run'/'rankings.jsonl').read_text(encoding='utf-8'))
    compared = row['ranking_comparison']
    assert compared['alternative_hits'][0]['stable_id'] == row['hits'][0]['stable_id']
    assert compared['candidate_pool'][0]['original_chunk_text'] == 'Zebras migrate north.'
