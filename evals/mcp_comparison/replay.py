"""Zero-provider-call counterfactuals over retained, unchanged real evidence."""
from copy import deepcopy
import json
from pathlib import Path

from paperpilot_agentic.read_packet import build_read_packet
from .run import FOLDER


def replay():
    results=[]
    for name in ['followup_alignment_inference','unseen-alignment_loss',
                 'unseen-deployment_limits','unseen-robustness_conditions']:
        result=json.loads((FOLDER/'round2/mcp'/(name+'-result.json')).read_text(encoding='utf-8'))
        rows=result['evidence']
        question=result['run']['question']
        saved=result['run']['metadata']['reader_workset']
        before=build_read_packet(question,rows,max_bytes=24000)
        after_rows=deepcopy(rows)
        for row in after_rows:
            # The real MCP export does not know exact pages. Keep its source URI
            # and content untouched; do not assert the lower endpoint is exact.
            row['page']=None
        after=build_read_packet(question,after_rows,max_bytes=24000)
        joined='\n'.join(r['content'] for r in after.rows)
        results.append({'case':name,'provider_calls':0,
            'before_matches_saved_selected_ids':before.metadata['selected_evidence_ids']==saved['selected_evidence_ids'],
            'before':before.metadata,'after_unknown_exact_page':after.metadata,
            'after_contains_equation4':r'\tag{4}' in joined,
            'after_contains_stage_definition':'feature stages' in joined,
            'after_contains_limitations':'2) Limitations' in joined or '2) Limitations:' in joined,
            'content_and_uris_unchanged':all(a['content']==b['content'] and a['source_uri']==b['source_uri']
                                          for a,b in zip(rows,after_rows)),
            'does_not_prove_final_answer_correct':True})
    (FOLDER/'page-precision-counterfactual.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([{'case':r['case'],'matches_saved':r['before_matches_saved_selected_ids'],
        'before_selected':r['before']['selected_count'],
        'after_selected':r['after_unknown_exact_page']['selected_count'],
        'after_unrepresented':r['after_unknown_exact_page']['unrepresented_sources'],
        'after_equation4':r['after_contains_equation4']} for r in results]))


if __name__=='__main__':
    replay()
