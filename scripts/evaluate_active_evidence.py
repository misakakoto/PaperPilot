"""Freeze 12 question-level cases; compare three strategies without resetting runs.

Prepare creates no model requests. Execute uses the configured Qwen endpoint,
unchanged cached hybrid backend, one shared frozen corpus, <=36 runs/216 calls/
2.16M tokens. Resume only reuses the same task IDs, receipts and spending.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from paperpilot_agentic.paper_structure import digest, save_json

CF='a9a7bd539662d494f0d2088593bfcd6ca28f8f30e645d42492407371bccbd107'
BIT='a81a987d5511f9be0684456a873597b3a73956ec43a91551f8e8d9efb68927df'
CH='c5e3104d14652705fd319e9de171e1615517c2471f838953013a9c268245aa6c'
AF='221e15fca331234bf0a1c6b0bb9d846d6996d2d2be9aa26906db6054a8f4dcf7'
HF='9e318fcc490436786db60f85c8b95ac8a76034e3b2ccc55c93a8046c96cc7a35'
DL='833794585404369e63db098e8b18045ce85d2c122d750a708b1ea1d0a99478b4'


def cases():
    # These are narrower new questions, not document-held-out research tasks.
    rows = [
        ('method','dev',[CH], '请依据 Changer 原论文解释 ChangerEx 的通道交换与空间交换分别交换什么、是否引入可学习参数；区分本文方法和相关工作 CEN。',
         'Distinguish own ChangerEx from related CEN; channel/spatial positions and parameter-free interaction.',[(CH,4,'exchange'),(CH,5,'exchange')]),
        ('method','heldout',[DL], 'DeepLabv3+ 论文中 output stride 的定义是什么？它如何通过取消步幅并使用空洞卷积得到更密集特征？',
         'Output stride is input/final feature resolution ratio; removing striding and using atrous convolution. Not input image resize.',[(DL,5,'output stride')]),
        ('method','heldout',[BIT], 'BIT 的语义分词器是固定平均池化吗？请根据其语义分组和空间注意力的原文说明哪些部分是学习得到的。',
         'Learned spatial attention maps/pointwise convolution and softmax, not fixed average pooling.',[(BIT,4,'spatial attention'),(BIT,5,'learnable')]),
        ('comparison','dev',[CH,AF], '比较 ChangerEx 与 AFDE-Net：双时相信息分别在特征提取或差分增强的哪个环节发生交互？不要用指标排名代替机制比较。',
         'Changer interaction in extraction stages versus AFDE absolute differences/FDE and attention enhancement.',[(CH,3,'interaction'),(AF,2,'differential')]),
        ('comparison','heldout',[HF,BIT], 'HF-MCD 与 BIT 所面对的输入模态和空间分辨率问题是否相同？分别引用原文，区分多模态异构性和双时相语义交互。',
         'HF multimodal heterogeneous spatial/channel resolutions versus BIT bitemporal semantic-token context; no cross-dataset winner.',[(HF,1,'multimodal'),(BIT,1,'bitemporal')]),
        ('comparison','heldout',[CF,CH], '比较 ChangeFormer 与 Changer 的孪生结构：不能只说都有两条分支，请说明二者在跨时相交互发生位置上的设计差异，并标出仍需回查的部分。',
         'CF separate hierarchical encoding/difference decoder versus Changer interleaved feature interactions; acknowledge missing detail.',[(CF,2,'difference'),(CH,3,'hierarchy')]),
        ('conditions','dev',[DL], 'DeepLabv3+ 的额外 JFT 预训练结果，能否与没有 JFT 的结果当成相同训练数据下的公平性能比较？请查清条件后作答。',
         'Separate JFT additional pretraining; do not treat as identical training data.',[(DL,11,'JFT'),(DL,12,'JFT')]),
        ('conditions','heldout',[CF,BIT], '核对 ChangeFormer 与 BIT 在 LEVIR-CD 上的裁块与划分说明：哪些设置一致？能否仅凭这些一致就认定全部训练条件相同？',
         'Both describe non-overlap 256 crops/7120-1024-2048; matching part does not certify same complete protocol.',[(CF,3,'256 × 256'),(BIT,6,'256')]),
        ('conditions','heldout',[AF], 'AFDE-Net 的模块消融是在什么数据集和划分上做的？这些消融是否能直接作为 WHU 测试集上同等模块收益的证据？',
         'Table III EGY-BCD validation; no direct transfer of numeric module gains to WHU test.',[(AF,5,'ABLATION'),(AF,5,'validation')]),
        ('insufficient','dev',[CF], '仅依据 ChangeFormer 这份论文，请给出其 INT8 量化版本在手机端实测的功耗与推理延迟；没有对应实验就说明缺口。',
         'No supplied source for INT8 mobile measured energy/latency. Bound absence to the selected paper.',[]),
        ('insufficient','heldout',[BIT,CH], '仅凭 BIT 和 Changer 这两篇原论文，能否给出特征交互在所有未见数据分布上必然有效的数学证明？请提供证明原文或明确证据边界。',
         'Empirical ablations do not establish universal mathematical guarantee.',[]),
        ('insufficient','heldout',[DL], '仅依据 DeepLabv3+ 论文，给出已实现的双时相建筑变化检测 build_model(config) 与 forward(image_t1,image_t2) 接口原文；不要自行编写接口冒充论文内容。',
         'Semantic segmentation paper is not evidence for this host-specific bitemporal code interface.',[]),
    ]
    return [{'id':f'active-{n:02d}','category':cat,'split':split,'paper_ids':papers,'question':q,
        'review_notes':notes,'annotation_locators':loc} for n,(cat,split,papers,q,notes,loc) in enumerate(rows,1)]


def prepare(args):
    args.output.mkdir(parents=True,exist_ok=False)
    rows = cases()
    with sqlite3.connect(args.source_db.resolve().as_uri()+'?mode=ro',uri=True) as c:
        for row in rows:
            row['annotation_sources'] = []
            for sha,page,term in row.pop('annotation_locators'):
                found = c.execute('''SELECT stable_id,text FROM chunks c JOIN papers p ON c.paper_row_id=p.row_id
                    WHERE p.sha256=? AND c.page_start=? AND lower(c.text) LIKE ? ORDER BY c.ordinal LIMIT 1''',
                    (sha,page,'%'+term.lower()+'%')).fetchone()
                if not found:
                    raise ValueError(f'Annotation locator absent: {row["id"]}, page {page}, {term}')
                uid,text = found
                anchor = text.lower().find(term.lower())
                row['annotation_sources'].append({'paper_sha256':sha,'unit_id':uid,'page':page,
                    'quote':text[max(0,anchor-100):anchor+900]})
    save_json(args.output/'benchmark.json',{'version':'active-evidence-12-v1','cases':rows,
        'annotation_limits':'Single implementer source review; question-level split, six previously examined papers; not independent expert adjudication or document holdout.',
        'task_scope':'Explicitly selected PDFs named by the question; this study does not measure global paper discovery.',
        'absence_labels':'Scope-limited evidence insufficiency, not proof of absence from all literature.'})
    print('Frozen 12 questions: 4 development / 8 question-held-out; no API calls.',flush=True)


def execute(args):
    from paperpilot_agentic.config import RuntimeSettings
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
    from paperpilot_agentic.active_evidence import ActiveEvidenceHarness
    from paperpilot_agentic.active_paper_sources import ActivePaperSources
    from paperpilot_agentic.provider_diagnostics import safe_error_text
    from paperpilot_retrieval import PaperKnowledgeStore
    from paperpilot_retrieval.reranking import OnnxCrossEncoderReranker
    from build_paper_evidence import local_models
    if not (args.output/'benchmark.json').is_file():
        raise ValueError('Prepare and review the frozen annotations before executing')
    benchmark = json.loads((args.output/'benchmark.json').read_bytes())
    settings = replace(RuntimeSettings.from_env(require_api_key=True),fallback_models=(),thinking_enabled=False)
    if settings.provider!='qwen' or settings.base_url!='https://dashscope.aliyuncs.com/compatible-mode/v1':
        raise ValueError('Paper excerpt consent is scoped to the configured Qwen endpoint')
    source_freeze = {str(p):digest(p.read_bytes()) for p in Path('src').rglob('*.py')}
    spec = {'benchmark_sha256':digest((args.output/'benchmark.json').read_bytes()), 'source_freeze':source_freeze,
        'runner_sha256':digest(Path(__file__).read_bytes()), 'calls_max':216,'tokens_max':2160000,
        'runs_max':36,'per_run_calls':6,'per_run_tokens':60000,'automatic_provider_retries':0,
        'model':settings.model,'arms':['static','simple_loop','active'],'backend':'identical legacy text/hybrid/reranker',
        'scope':'selected PDFs; no oracle answers passed to models','scientific_review':'REQUIRED'}
    versions = [(1,args.output/'execution-spec.json')] if (args.output/'execution-spec.json').exists() else []
    versions += sorted((int(p.stem.rsplit('v',1)[1]),p) for p in args.output.glob('execution-spec-v*.json'))
    revision,excluded = 1,[]
    spec_file = args.output/'execution-spec.json'
    if versions:
        revision,spec_file = versions[-1]
        previous = json.loads(spec_file.read_bytes())
        if args.continue_version:
            if any(previous[k]!=spec[k] for k in ('model','benchmark_sha256','arms','per_run_calls','per_run_tokens')):
                raise ValueError('Cannot change model, benchmark or budget when continuing this campaign')
            # Only previously unstarted CASES may use a revised protocol. A failed
            # arm cannot get a fresh task, quota, model identity or retry allowance.
            with sqlite3.connect(args.output/'runtime.sqlite3') as c:
                assert not c.execute("SELECT 1 FROM llm_attempts WHERE state!='RETURNED' OR usage_known!=1 LIMIT 1").fetchone()
                excluded = sorted({key.rsplit('-',1)[0] for key, in c.execute('SELECT request_key FROM research_episodes')})
            revision += 1
            spec_file = args.output/f'execution-spec-v{revision}.json'
        else:
            if not args.resume or any(previous[k]!=spec[k] for k in spec):
                raise ValueError('Frozen execution identity changed; use a documented version continuation for unstarted cases only')
            excluded = previous.get('excluded_cases',[])
    spec.update(revision=revision,excluded_cases=excluded)
    if not spec_file.exists():
        save_json(spec_file,spec)
        frozen = args.output/('frozen-source' if revision==1 else f'frozen-source-v{revision}')
        for name in source_freeze:
            target = frozen/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    baseline = args.output/'source.sqlite3'
    if not baseline.exists():
        with sqlite3.connect(args.source_db.resolve().as_uri()+'?mode=ro',uri=True) as src:
            with sqlite3.connect(baseline) as dst:
                src.backup(dst)
    _,embedder = local_models(settings)
    reranker = OnnxCrossEncoderReranker(cache_dir=Path(os.environ['PAPERPILOT_RERANKER_CACHE']),allow_download=False,batch_size=4)
    adapter = PaperPilotKnowledgeSearchAdapter(PaperKnowledgeStore(baseline,namespace=args.namespace),
        embedder=embedder,reranker=reranker,rerank_stage='final')
    runtime = ResearchAgentRuntime(workspace=args.output,db_path=args.output/'runtime.sqlite3',
        search_adapter=adapter,llm=OpenAICompatibleLLM(settings,max_retries=0))
    sources = ActivePaperSources(adapter)
    h = ActiveEvidenceHarness(runtime,sources=sources)
    results_file = args.output/'progress.json'
    rows = json.loads(results_file.read_bytes())['rows'] if results_file.exists() else []
    if args.continue_version:
        original_progress = args.output/f'progress-before-v{revision}.json'
        if not original_progress.exists():
            save_json(original_progress,{'rows':rows})
        for row in rows:
            row.setdefault('revision',1)
            row['excluded_from_current_protocol_comparison'] = row['case'] in excluded
        with runtime.db.connect() as c:
            old = [dict(r) for r in c.execute('SELECT run_id,request_key FROM research_episodes')]
        for task in old:
            case_id,arm = task['request_key'].rsplit('-',1)
            if not any(r.get('run_id')==task['run_id'] for r in rows):
                run = runtime.db.get_run(task['run_id'])
                if run['used_llm_calls']:
                    raise ValueError('Unpublished billable old task requires inspection, never a fresh replacement')
                runtime.cancel(task['run_id'],reason='Protocol revision before first model dispatch; original ledger retained')
                rows.append({'case':case_id,'arm':arm,'run_id':task['run_id'],'revision':revision-1,
                    'phase':'CANCELLED','calls':0,'tokens':0,'tools':run['used_tool_calls'],
                    'excluded_from_current_protocol_comparison':True,'scientific_correctness':'NOT_REVIEWED'})
        save_json(results_file,{'rows':rows,'budget_reset':False,'revision':revision})
    completed = {(r['case'],r['arm']) for r in rows}
    stop = False
    for case in sorted(benchmark['cases'],key=lambda c:c['split']!='dev'):
        if case['id'] in excluded or args.dev_only and case['split']!='dev':
            continue
        for mode in spec['arms']:
            if (case['id'],mode) in completed:
                continue
            for old in runtime.db.list_runs(limit=100):
                if runtime.llm_runtime.budget_snapshot(old['run_id'])['unknown_results']:
                    stop = True
                    break
            if stop:
                break
            start = time.perf_counter()
            try:
                result = h.start(case['question'],case['id']+'-'+mode,paper_ids=case['paper_ids'],mode=mode)
                rid = result['run']['run_id']
                result = h.advance(rid)
                state = h._load(rid)
                unknown = runtime.llm_runtime.budget_snapshot(rid)['unknown_results']
                row = {'case':case['id'],'category':case['category'],'split':case['split'],'arm':mode,'revision':revision,
                    'run_id':rid,'phase':state['phase'],'calls':result['run']['used_llm_calls'],
                    'tokens':result['run']['used_tokens'],'tools':result['run']['used_tool_calls'],
                    'read_ranges':len(state['read_handles']),'stop_reason':state.get('stop_reason'),
                    'source_admitted':state.get('admission',{}).get('admitted',False),
                    'verdict':state.get('answer',{}).get('verdict'),'unknown_results':unknown,
                    'scientific_correctness':'NOT_REVIEWED','seconds':time.perf_counter()-start}
                save_json(args.output/(case['id']+'-'+mode+'.json'),{'row':row,'result':result,'state':state})
                stop = bool(unknown)
            except Exception as exc:
                row = {'case':case['id'],'arm':mode,'revision':revision,'error':safe_error_text(exc),'seconds':time.perf_counter()-start}
                stop = True  # Unknown/local submission identity requires inspection, not a new run.
            rows.append(row)
            save_json(results_file,{'rows':rows,'all_case_arms_attempted':len(rows)==36,'budget_reset':False})
            print(json.dumps(row,ensure_ascii=False),flush=True)
            runs = runtime.db.list_runs(limit=100)
            assert len(runs)<=36 and sum(r['used_llm_calls'] for r in runs)<=216 and sum(r['used_tokens'] for r in runs)<=2160000
            if stop:
                break
        if stop:
            break
    assert source_freeze=={str(p):digest(p.read_bytes()) for p in Path('src').rglob('*.py')}
    save_json(args.output/'results.json',{'rows':rows,'all_case_arms_attempted':len(rows)==36,
        'stopped':stop,'budget_reset':False,'revision':revision,'excluded_cases':excluded,
        'development_only':args.dev_only,'actual_calls':sum(r.get('calls',0) for r in rows),
        'actual_tokens':sum(r.get('tokens',0) for r in rows),'scientific_review':'REQUIRED; source admission is not correctness'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-db',type=Path,default=Path('.paperpilot/paper-knowledge.sqlite3'))
    p.add_argument('--namespace',default='all-papers-v1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--continue-version',action='store_true',help='Revised protocol for unstarted cases ONLY; preserve every previous run/cost')
    p.add_argument('--dev-only',action='store_true',help='Complete only unstarted development cases before touching remaining holdout')
    args = p.parse_args()
    if args.prepare:
        prepare(args)
    elif args.execute:
        execute(args)
    else:
        p.error('Choose --prepare or --execute; resume never changes a frozen identity')


if __name__=='__main__':
    main()
