from paperpilot_agentic.active_evidence_context import final_evidence_view
from paperpilot_agentic.active_evidence_ledger import quote_spans
from paperpilot_agentic.paper_evidence import upper


def fixture():
    reads={}
    for i in range(1,9):
        rid='R'+str(i)
        text='Background discussion. '*18+f'The decoder uses pixel queries and semantic tokens in experiment {i}.'
        reads[rid]={'source_id':rid,'paper_id':'P1' if i<5 else 'P2','page':i,'range':[0,len(text)],
            'content':text,'source_kind':'TEXT','source_quality':{},'range_identity':rid*32}
    state={'goal':'Explain semantic tokens and pixel decoder','requirements':[
        {'id':'D1','dimension':'semantic tokens and pixel decoder','kind':'method','paper_ids':['P1','P2']}],
        'ledger':{},'read_handles':reads,'source_handles':{},'paper_handles':{'P1':'a'*64,'P2':'b'*64},
        'retrieval_identity':'x'*64}
    payload={'goal':state['goal'],'requirements':state['requirements'],'papers':[
        {'paper_id':'P'+str(i),'title':'Paper '+str(i),'version':str(i)*64,
            'read_source_ids':[r for r,v in reads.items() if v['paper_id']=='P'+str(i)],
            'method_identity_hints':[{'name':'Method'+str(i),'unit_id':'u'+str(i),'quote':'Naming context '*30}]}
        for i in range(1,7)],'evidence_coverage':[]}
    return payload,state


def test_final_view_balances_actual_sources_and_never_rewrites_quotes():
    payload,state=fixture()
    view=final_evidence_view(payload,state,list(state['read_handles']))
    assert upper(view)<=6700
    assert view['allowed_verdicts']==['INSUFFICIENT','NOT_COMPARABLE']
    assert len(view['read_passages'])>=4
    assert {r['paper_id'] for r in view['read_passages']}=={'P1','P2'}
    assert view['omitted_catalogue_papers']==['P3','P4','P5','P6']
    assert view['quote_selection']['omitted_quote_count']>0
    for row in view['read_passages']:
        original={s['quote_id']:s for s in quote_spans(state['read_handles'][row['source_id']])}
        assert row['quote_spans'][0]['text'].startswith('The decoder')
        assert all(q==original[q['quote_id']] for q in row['quote_spans'])
        assert row['source_id'] in view['evidence_sources']


def test_small_view_budget_does_not_truncate_a_quote_or_claim_full_read():
    payload,state=fixture()
    view=final_evidence_view(payload,state,list(state['read_handles']),cap=2500)
    for row in view['read_passages']:
        assert row['view']=='SELECTED_EXACT_QUOTES_NOT_FULL_RANGE'
    assert view['quote_selection']['omitted_quote_count']>0
    assert set(view['omitted_read_handles']) | {r['source_id'] for r in view['read_passages']}==set(state['read_handles'])
