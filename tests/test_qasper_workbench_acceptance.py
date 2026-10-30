import json

import pytest

from evals.qasper.workbench_acceptance import freeze, predicted_paragraphs


def test_freeze_preserves_questions_and_never_filters_by_gold(tmp_path):
    paper={'qas':[{'question_id':str(i),'question':f'Original question {i}','answers':['POISON GOLD']} for i in range(20)]}
    dataset=tmp_path/'dev.json';dataset.write_text(json.dumps({'p':paper}),encoding='utf-8')
    a=freeze(dataset,tmp_path/'one.json',count=8)
    paper['qas'][0]['answers']=['Different annotation']
    dataset.write_text(json.dumps({'p':paper}),encoding='utf-8')
    b=freeze(dataset,tmp_path/'two.json',count=8)
    assert a['cases']==b['cases'] and len(a['cases'])==8
    assert 'POISON GOLD' not in json.dumps(a)
    with pytest.raises(FileExistsError):freeze(dataset,tmp_path/'one.json')


def test_evidence_mapping_only_uses_cited_complete_original_paragraphs():
    paper={'abstract':'abstract','full_text':[{'paragraphs':['Full original paragraph.','Another paragraph.']}]}
    result={'final_answer':{'citation_ids':['owned']},'evidence':[
        {'evidence_id':'owned','content':'Heading\nFull original paragraph.\nfragment'},
        {'evidence_id':'not-cited','content':'Another paragraph.'}]}
    assert predicted_paragraphs(paper,result)==['Full original paragraph.']
