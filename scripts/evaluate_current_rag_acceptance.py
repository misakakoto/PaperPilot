"""Evaluate one frozen current paper workflow, including paper discovery.

Six reformulated development acceptance tasks; no claim of unseen documents or
independent labels. Expected sources stay in the evaluator, never model input.
No provider retries, budget resets, automatic model switching or training.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT)]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def source_freeze():
    return {str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in (ROOT/'src').rglob('*.py')}


TASKS=[
    ('accept-01','active-03','method',
     '解释 BIT 的语义 token 是如何从双时相特征中形成、又如何回写像素空间的。它与固定网格平均池化的主要区别是什么？请引用原文。'),
    ('accept-02','active-05','method',
     'HF-MCD 面向什么影像模态和空间分辨率关系？如果手头只有两幅普通 RGB 影像，论文中的结果是否直接证明该方案适用？请区分作者设计和仍需验证的判断。'),
    ('accept-03','active-06','comparison',
     '比较 ChangerEx 与 ChangeFormer：两者分别在什么环节进行双时相交互或差分融合？只解释结构机制，不比较 F1，也不要把相关工作当成本文方法。'),
    ('accept-04','active-08','comparability',
     'ChangeFormer 与 BIT 都报告了 LEVIR-CD 实验。如果数据集名称和裁剪尺寸相同，是否就能公平比较 F1？依据两篇论文列出还要核对的条件；缺失项明确保留，不宣布胜者。'),
    ('accept-05','active-09','experiment',
     'AFDE-Net 的解码器相关消融究竟采用 EGY 还是 WHU？请区分验证集消融和测试集主结果，并给出相应原文依据。'),
    ('accept-06','active-11','insufficient',
     '只依据 BIT 和 Changer 两篇论文，能否证明它们在所有未见地区都优于其他建筑变化检测方法？请给出有范围限制的结论，不把有限实验推广成普遍保证。'),
]


def prepare(output, case_ids=None):
    from paperpilot_agentic.config import RuntimeSettings
    output.mkdir(parents=True,exist_ok=False)
    prior=ROOT/'outputs/active-evidence-12-v1-20261027/benchmark.json'
    originals={q['id']:q for q in json.loads(prior.read_bytes())['cases']}
    cases=[]
    for cid,parent,category,question in TASKS:
        if case_ids and cid not in case_ids:
            continue
        old=originals[parent]
        cases.append({'id':cid,'category':category,'question':question,
            'expected_papers':old['paper_ids'],'review_notes':old['review_notes'],
            'annotation_sources':old['annotation_sources'],'reference_case':parent})
    if case_ids and set(case_ids) != {c['id'] for c in cases}:
        raise ValueError('Unknown acceptance case ID')
    # The HF-only question does not ask for the BIT comparison from its parent.
    # Apply the already documented grading correction before freezing new runs.
    for case in cases:
        if case['id']=='accept-02':
            case['expected_papers']=[p for p in case['expected_papers']
                if p!='a81a987d5511f9be0684456a873597b3a73956ec43a91551f8e8d9efb68927df']
    save(output/'benchmark.json',{'cases':cases,'labels':'Developer-authored, previously examined papers; not independent expert or document holdout.',
        'assessment':'Review final answer against frozen notes and actual read quotes; source admission is not correctness.',
        'paper_scope':'Full registered library; expected_papers are evaluator-only, not passed to task start.'})
    settings=RuntimeSettings.from_env()
    save(output/'execution-spec.json',{'profile':'paper_active_v1','model':settings.model,
        'provider':settings.provider,'base_url':settings.base_url,'backend':'unchanged_legacy_hybrid',
        'window_aggregation_in_this_path':False,'source_freeze':source_freeze(),
        'runner_sha256':digest(Path(__file__).read_bytes()),'benchmark_sha256':digest((output/'benchmark.json').read_bytes()),
        'runs_max':len(cases),'per_run_calls':6,'per_run_tokens':60000,
        'calls_max':6*len(cases),'tokens_max':60000*len(cases),
        'prior_evaluation':{'path':'outputs/current-rag-acceptance-20261028',
            'calls':27,'tokens':112033,'status':'Preserved failed candidate; this is a new code revision evaluation'},
        'retries':0,'automatic_model_switching':False,'scope':'Current runnable candidate, no historical scores reused.'})
    print(f'Frozen {len(cases)} full-library acceptance tasks; no API calls.',flush=True)


def execute(output,resume):
    from paperpilot_agentic.config import RuntimeSettings
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
    from paperpilot_agentic.active_evidence import ActiveEvidenceHarness
    from paperpilot_retrieval import PaperKnowledgeStore
    from paperpilot_retrieval.reranking import OnnxCrossEncoderReranker
    from build_paper_evidence import local_models
    spec=json.loads((output/'execution-spec.json').read_bytes())
    settings=replace(RuntimeSettings.from_env(require_api_key=True),fallback_models=(),thinking_enabled=False)
    if (settings.provider!='qwen' or settings.base_url!='https://dashscope.aliyuncs.com/compatible-mode/v1'
            or any(spec[k]!=getattr(settings,k) for k in ['model','provider','base_url'])):
        raise ValueError('Frozen model/endpoint changed; do not switch models inside acceptance')
    if (spec['source_freeze']!=source_freeze() or spec['runner_sha256']!=digest(Path(__file__).read_bytes())
            or spec['benchmark_sha256']!=digest((output/'benchmark.json').read_bytes())):
        raise ValueError('Frozen source, runner or benchmark changed')
    if (output/'runtime.sqlite3').exists() and not resume:
        raise ValueError('Existing ledger requires --resume; never replace a failed task')
    if not (output/'source.sqlite3').exists():
        with sqlite3.connect(settings.memory_db.resolve().as_uri()+'?mode=ro',uri=True) as source:
            with sqlite3.connect(output/'source.sqlite3') as target:
                source.backup(target)
    _,embedder=local_models(settings)
    reranker=OnnxCrossEncoderReranker(cache_dir=Path(os.environ['PAPERPILOT_RERANKER_CACHE']),
        batch_size=4,allow_download=False)
    adapter=PaperPilotKnowledgeSearchAdapter(PaperKnowledgeStore(output/'source.sqlite3',namespace=settings.memory_namespace),
        embedder=embedder,reranker=reranker,rerank_stage='final')
    runtime=ResearchAgentRuntime(workspace=output,db_path=output/'runtime.sqlite3',search_adapter=adapter,
        llm=OpenAICompatibleLLM(settings,max_retries=0))
    harness=ActiveEvidenceHarness(runtime)
    save(output/'backend-receipt.json',{'source_identity':harness.sources.identity,
        'papers':len(harness.sources.papers),'embedding':embedder.cache_identity,
        'reranker_sha256':reranker.model_sha256,'reranker_tokenizer_sha256':reranker.tokenizer_sha256})
    progress=output/'results.json'
    rows=json.loads(progress.read_bytes())['rows'] if progress.exists() else []
    finished={r['case'] for r in rows}
    for case in json.loads((output/'benchmark.json').read_bytes())['cases']:
        if case['id'] in finished:
            continue
        if any(runtime.llm_runtime.budget_snapshot(r['run_id'])['unknown_results']
               for r in runtime.db.list_runs(limit=20)):
            print('Stopped: unresolved external result; no new dispatch.',flush=True)
            break
        started=time.perf_counter()
        result=harness.start(case['question'],'current-rag-'+case['id'],mode='active')
        rid=result['run']['run_id']
        initial=harness._load(rid)
        selected=set(initial['paper_handles'].values())
        result=harness.advance(rid)
        state=harness._load(rid)
        reads=list(state['read_handles'].values())
        row={'case':case['id'],'category':case['category'],'run_id':rid,'phase':state['phase'],
            'verdict':state.get('answer',{}).get('verdict'),'calls':result['run']['used_llm_calls'],
            'tokens':result['run']['used_tokens'],'tools':result['run']['used_tool_calls'],
            'initial_paper_hits':len(selected & set(case['expected_papers'])),
            'expected_papers':len(case['expected_papers']),
            'expected_papers_read':len({r['paper_sha256'] for r in reads} & set(case['expected_papers'])),
            'read_ranges':len(reads),'claims':len(state.get('answer',{}).get('claims',[])),
            'source_admitted':state.get('admission',{}).get('admitted',False),
            'unknown_results':runtime.llm_runtime.budget_snapshot(rid)['unknown_results'],
            'stop_reason':state.get('stop_reason'),'content_review':'PENDING',
            'seconds':time.perf_counter()-started}
        save(output/(case['id']+'.json'),{'row':row,'result':result,'state':state})
        rows.append(row)
        save(progress,{'rows':rows,'actual_calls':sum(r['calls'] for r in rows),
            'actual_tokens':sum(r['tokens'] for r in rows),'budget_reset':False,'content_review':'PENDING'})
        print(json.dumps(row,ensure_ascii=False),flush=True)
        runs=runtime.db.list_runs(limit=20)
        assert (len(runs)<=spec['runs_max'] and sum(r['used_llm_calls'] for r in runs)<=spec['calls_max']
            and sum(r['used_tokens'] for r in runs)<=spec['tokens_max'])
        if row['unknown_results'] or state['phase'] in {'UNKNOWN','FAILED'}:
            break
    assert spec['source_freeze']==source_freeze()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--cases',nargs='+',choices=[t[0] for t in TASKS],help='Freeze only these development cases at prepare time')
    args=parser.parse_args()
    if args.prepare:
        prepare(args.output.resolve(),args.cases)
    elif args.execute:
        execute(args.output.resolve(),args.resume)
    else:
        parser.error('Choose --prepare or --execute')


if __name__=='__main__':
    main()
