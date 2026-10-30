import json
import pytest
from paperpilot_retrieval import PaperKnowledgeStore

from evals.qasper.content_query_ablation import content_query, run


def test_query_keeps_acronyms_negation_numbers_and_chinese():
    assert content_query('What does KBQA abbreviate for') == 'KBQA abbreviate'
    assert content_query('Which US model is not trained on 2 datasets?') == 'US model not trained 2 datasets?'
    assert content_query('中文论文 Table 1 的参数是什么') == '中文论文 Table 1 的参数是什么'
    assert content_query('what is it') == 'what is it'


def test_paired_protocol_retains_errors_and_unmatchable_gold(tmp_path):
    dataset = tmp_path/'qasper-dev-v0.3.json'
    dataset.write_text(json.dumps({'p': {'title': 'Retrieval', 'abstract': 'Evidence paragraph.',
        'full_text': [], 'figures_and_tables': [], 'qas': [
            {'question_id': 'q', 'question': 'Which evidence?',
             'answers': [{'answer': {'evidence': ['UNMATCHABLE_GOLD']}}]}]}}), encoding='utf-8')
    result = run(dataset, tmp_path/'result')
    assert result['queries'] == result['text_denominator'] == 1
    assert result['mean_recall']['baseline']['8'] == result['mean_recall']['content_query']['8'] == 0
    assert result['paper_bootstrap_delta_95_ci'] == [0, 0]
    rows = json.loads((tmp_path/'result/rankings.jsonl').read_text(encoding='utf-8'))
    assert all('UNMATCHABLE_GOLD' not in text for text in rows['content_query']['texts'])


def test_production_policy_default_and_legacy_switch(tmp_path):
    assert PaperKnowledgeStore(tmp_path/'default.sqlite3').lexical_query_policy == 'content_v1'
    assert PaperKnowledgeStore(tmp_path/'old.sqlite3', lexical_query_policy='raw').lexical_query_policy == 'raw'
    with pytest.raises(ValueError):
        PaperKnowledgeStore(tmp_path/'bad.sqlite3', lexical_query_policy='typo')
    assert not (tmp_path/'bad.sqlite3').exists()
