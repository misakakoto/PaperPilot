"""Isolated final-answer regression on saved reads, NOT an end-to-end rerun.

Never modifies parent episodes. Each case receives one new, durably recorded
request with a 12k input/output budget. Existing output directories are refused;
unknown results halt the batch and must be reconciled manually, never resent.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    from evaluate_current_rag_acceptance import source_freeze, save, digest
    from paperpilot_agentic.active_evidence import FINAL_SYSTEM
    from paperpilot_agentic.active_evidence_context import final_evidence_view
    from paperpilot_agentic import active_evidence_ledger as ledger
    from paperpilot_agentic.config import RuntimeSettings
    from paperpilot_agentic.db import RuntimeDatabase
    from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
    from paperpilot_agentic.models import Budget, LLMCallPolicy, RunStatus
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    settings=replace(RuntimeSettings.from_env(require_api_key=True),fallback_models=(),thinking_enabled=False)
    prior=json.loads((args.parent/'execution-spec.json').read_bytes())
    if (settings.provider!='qwen' or settings.base_url!='https://dashscope.aliyuncs.com/compatible-mode/v1'
            or settings.model!=prior['model']):
        raise ValueError('Use the same authorized Qwen model and endpoint')
    args.output.mkdir(parents=True,exist_ok=False)
    frozen=source_freeze()
    save(args.output/'spec.json',{'scope':'final-answer-only controlled replay, no end-to-end success claim',
        'source_freeze':frozen,'model':settings.model,'parent':str(args.parent.resolve()),
        'parent_results_sha256':digest((args.parent/'results.json').read_bytes()),
        'calls_max':2,'tokens_max':24000,'parent_usage':json.loads((args.parent/'results.json').read_bytes())})
    db=RuntimeDatabase(args.output/'runtime.sqlite3')
    llm=PersistentLLMRuntime(db,OpenAICompatibleLLM(settings,max_retries=0))
    requests=[json.loads(p.read_bytes()) for p in args.parent.rglob('turn-06-request.json')]
    rows=[]
    for case in ('accept-01','accept-02'):
        state=json.loads((args.parent/(case+'.json')).read_bytes())['state']
        old=next(json.loads(r['messages'][-1]['content']) for r in requests
            if json.loads(r['messages'][-1]['content'])['goal']==state['goal'])
        view=final_evidence_view(old,state,list(state['read_handles']))
        messages=[{'role':'system','content':FINAL_SYSTEM},{'role':'user','content':json.dumps(view,ensure_ascii=False)}]
        save(args.output/(case+'-request.json'),{'messages':messages,'parent_run_id':json.loads((args.parent/(case+'.json')).read_bytes())['row']['run_id']})
        rid=db.create_run(state['goal'],Budget(max_llm_calls=1,max_tokens=12000,
            synthesis_reserve_calls=0,synthesis_reserve_tokens=0,repair_reserve_calls=0,repair_reserve_tokens=0),
            metadata={'evaluation':'isolated_context_replay','case':case})
        db.update_run_status(rid,RunStatus.RUNNING)
        try:
            response=llm.complete(rid,messages,policy=LLMCallPolicy('context_replay',1800,False,True),correlation_id=case)
            save(args.output/(case+'-response.json'),{'text':response.text,'finish_reason':response.finish_reason})
            answer=ledger.canonical_answer(state,json.loads(response.text)['arguments'])
            try:
                admission=ledger.admit_final(state,answer)
            except ValueError as exc:
                admission={'admitted':False,'error':str(exc)}
            row={'case':case,'answer':answer,'admission':admission,'budget':llm.budget_snapshot(rid),
                'shown_read_handles':[r['source_id'] for r in view['read_passages']]}
            rows.append(row)
            db.update_run_status(rid,RunStatus.COMPLETED)
            save(args.output/'results.json',{'rows':rows,'content_review':'PENDING'})
            print(json.dumps(row,ensure_ascii=False),flush=True)
        except Exception as exc:
            db.update_run_status(rid,RunStatus.FAILED,error=str(exc))
            save(args.output/'stopped.json',{'case':case,'error':str(exc),'budget':llm.budget_snapshot(rid)})
            raise
        assert frozen==source_freeze()


if __name__=='__main__':
    main()
