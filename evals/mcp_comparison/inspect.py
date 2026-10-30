"""Non-billable report projections. This does not assign factual PASS scores."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def inspect(folder):
    folder=Path(folder)
    rows=[]
    for path in sorted(folder.glob('*-result.json')):
        result=json.loads(path.read_text(encoding='utf-8'))
        metadata=result['run']['metadata']
        report=metadata.get('reader_report') or {}
        answer=(result.get('final_answer') or {}).get('answer') or ''
        rows.append({'case':path.name.removesuffix('-result.json'),
            'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'run_status':result['run']['status'],'reader_status':report.get('status'),
            'coverage':report.get('final_request_coverage',{}).get('coverage'),
            'elapsed_seconds':result.get('experiment_elapsed_seconds'),
            'calls':result['run']['used_llm_calls'],'tokens':result['run']['used_tokens'],
            'answer_chars':len(answer),'source_review':'PENDING',
            'projection_errors':report.get('table_projection_errors'),
            'workset':metadata.get('reader_workset'),
            'packet':report.get('read_packet')})
    with sqlite3.connect(folder/'receipts/spending.sqlite3') as conn:
        conn.row_factory=sqlite3.Row
        attempts=[dict(r) for r in conn.execute('select * from attempts order by id')]
    messages=[]
    for attempt in attempts:
        path=folder/'receipts'/f"request-{attempt['id']:03}.json"
        if not path.exists():
            continue
        request=json.loads(path.read_text(encoding='utf-8'))
        full=json.dumps(request.get('messages',[]),ensure_ascii=False,sort_keys=True)
        first=str((request.get('messages') or [{}])[0].get('content',''))
        messages.append({'attempt':attempt['id'],'case':attempt['phase'],
            'state':attempt['state'],'tokens':attempt['tokens'],'elapsed_seconds':attempt['elapsed'],
            'message_chars':len(full),'message_utf8_bytes':len(full.encode('utf-8')),
            'message_sha256':hashlib.sha256(full.encode('utf-8')).hexdigest(),
            'stage_hint':first[:100],
            'evidence_numbers_present':{n:n in full for n in ('82.83','82.26','0.57','84.53','94.35','89.44')},
            'has_matrix_exponential':'exp' in full,'has_alignment_mask':'y_u' in full or 'y _ { u }' in full})
    return {'rows':rows,'attempts':attempts,'actual_request_observations':messages,
        'returned_calls':sum(a['state']=='RETURNED' for a in attempts),
        'actual_attempts':len(attempts),'confirmed_tokens':sum(a['tokens'] for a in attempts),
        'unresolved':[{k:a[k] for k in ('id','state','reservation','phase')}
                      for a in attempts if a['state'] not in {'RETURNED','NOT_SENT'}]}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    opt=parser.parse_args()
    result=inspect(opt.folder)
    (opt.folder/'observations.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in {'rows','attempts','actual_request_observations'}},ensure_ascii=True))
