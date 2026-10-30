"""Run current production question answering against isolated source backends."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time

from evals.interview.budget import CampaignBudget, RecordedProvider, RecordedVision, save, source_fingerprint
from .cases import cases

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT/'outputs/mcp-comparison-20261017'


def markdown_blocks(text, limit=6000):
    """Keep paragraphs / HTML tables whole; no invented cells or page mapping."""
    output, current = [], ''
    for block in re.split(r'\n\s*\n', text):
        if not block.strip():
            continue
        if len(block) > 12000:
            raise ValueError('An indivisible MCP block exceeds production chunk limits')
        if current and len(current)+len(block)+2 > limit:
            output.append(current)
            current = ''
        current += ('\n\n' if current else '') + block
    if current:
        output.append(current)
    return output


def prepare_corpus(settings, folder, arm):
    target = folder/'corpus.sqlite3'
    if target.exists():
        if arm == 'mcp' and not (folder/'corpus-build.json').exists():
            raise RuntimeError('Interrupted MCP import has no completion receipt; do not mislabel the copied local corpus')
        return
    with sqlite3.connect(settings.memory_db.as_uri()+'?mode=ro',uri=True) as source:
        with sqlite3.connect(target) as dest:
            source.backup(dest)
    if arm == 'local':
        return
    from paperpilot_retrieval.models import PaperDocument, PaperChunk
    from paperpilot_retrieval import PaperKnowledgeStore, FastEmbedder
    store = PaperKnowledgeStore(target,namespace=settings.memory_namespace)
    docs, chunks = [], []
    for index, source in enumerate(json.loads((FOLDER/'manifest.json').read_text(encoding='utf-8')),1):
        text = (FOLDER/'mcp-import2'/(source['name']+'.md')).read_text(encoding='utf-8')
        # A document-wide page interval is true; a guessed precise page is not.
        page_count = {'change':20,'flow':14}[source['name']]
        docs.append(PaperDocument(id=f'P{index}',sha256=source['sha256'],
            file_name=Path(source['path']).name, title=Path(source['path']).stem,
            page_count=page_count,processed_pages=0,pages=[],
            warnings=['MinerU Markdown; exact page/bbox/cell provenance unavailable']))
        for ordinal, block in enumerate(markdown_blocks(text),1):
            section = next((s.strip('# ') for s in reversed(block.splitlines()) if s.startswith('#')), 'MCP Markdown')
            chunks.append(PaperChunk(id=f'P{index}:p1-p{page_count}:c{ordinal}',paper_id=f'P{index}',
                page_start=1,page_end=page_count,section=section[:300],text=block,
                token_count_estimate=max(1,len(block)//4)))
    embedder = FastEmbedder(cache_dir=Path(os.environ['PAPERPILOT_EMBEDDING_CACHE']),
        model_name=os.getenv('PAPERPILOT_EMBEDDING_MODEL','sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),allow_download=False)
    sync = store.sync_documents(docs,chunks,embedder=embedder)
    save(folder/'corpus-build.json',{'chunks':len(chunks),'method':'whole-paragraph-6000-char',
        'page_mapping':'document-wide interval only','sync':str(sync)})


def execute(arm, selected=None, *, output_folder=None, model=None, source_baseline=None,
            freeze_reviewed_sources=False):
    from paperpilot_agentic.config import RuntimeSettings
    from paperpilot_agentic.bootstrap import build_runtime
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    from paperpilot_agentic.qwen_vision import QwenVisionEvidenceAdapter
    from paperpilot_agentic.models import Budget
    settings = RuntimeSettings.from_env()
    if model:
        settings = replace(settings,model=model)
    os.environ['PAPERPILOT_ALLOW_MODEL_DOWNLOAD'] = 'false'
    os.environ['PAPERPILOT_ALLOW_EXTERNAL_SCHOLARLY'] = 'false'
    folder = (Path(output_folder).resolve() if output_folder else FOLDER)/arm
    folder.mkdir(parents=True,exist_ok=True)
    frozen = cases(ROOT)
    identity = {'experiment':'local-mcp-evidence-20261017-v1','arm':arm,
        'source_sha256':source_fingerprint(ROOT),'model':settings.model,'provider':settings.provider,
        'request_cost_policy':'legacy-utf8-v1','usage_limit_mode':'unlimited',
        'cases_sha256':hashlib.sha256(json.dumps(frozen,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
        'temperature':settings.temperature,'max_tokens':settings.max_tokens,
        'base_url_sha256':hashlib.sha256(settings.base_url.encode()).hexdigest()}
    if freeze_reviewed_sources:
        from paperpilot_agentic.reviewed_sources import ReviewedSources
        identity['reviewed_source_registry_sha256'] = ReviewedSources.configured().identity()
    if not (folder/'protocol.json').exists():
        save(folder/'protocol.json',{'identity':identity,'cases':frozen,
            'authorization':'User authorized unlimited model experimentation and this local versus MCP comparison',
            'source_baseline':source_baseline or '80cda5880f953a7427e31d940a7a8bb728537800'})
    elif json.loads((folder/'protocol.json').read_text(encoding='utf-8'))['identity'] != identity:
        raise ValueError('Frozen experiment changed')
    campaign = CampaignBudget(folder/'receipts',identity,max_requests=None,max_tokens=None)
    if campaign.snapshot()['blocked']:
        print(json.dumps({'blocked':True,'arm':arm}),flush=True)
        return
    prepare_corpus(settings,folder,arm)
    settings = replace(settings,workspace=folder,runtime_db=folder/'runtime.sqlite3',
                       memory_db=folder/'corpus.sqlite3',thinking_enabled=False,fallback_models=())
    provider = RecordedProvider(OpenAICompatibleLLM(settings,max_retries=0),campaign,phase='preflight')
    if not (folder/'preflight.json').exists():
        try:
            response=provider.complete([{'role':'user','content':'Reply only ready.'}],max_tokens=32,thinking_enabled=False)
            ready=response.text.strip().lower()=='ready'
            save(folder/'preflight.json',{'ready':ready})
        except Exception as exc:
            save(folder/'preflight.json',{'ready':False,'error_type':type(exc).__name__})
            print(json.dumps({'arm':arm,'preflight':'FAILED','error_type':type(exc).__name__}),flush=True)
            return
    if not json.loads((folder/'preflight.json').read_text())['ready']:
        return
    vision=RecordedVision(QwenVisionEvidenceAdapter(api_key=settings.api_key,model=settings.model,
        base_url=settings.base_url,timeout_seconds=settings.timeout_seconds),campaign,phase='vision')
    runtime=build_runtime(settings,llm_factory=lambda _:provider,vision_factory=lambda **_:vision)
    for case in frozen:
        if selected and case['id'] not in selected:
            continue
        result_path=folder/(case['id']+'-result.json')
        if result_path.exists():
            print(json.dumps({'arm':arm,'case':case['id'],'reused':True}),flush=True)
            continue
        if campaign.snapshot()['blocked']:
            break
        submission=folder/(case['id']+'-submission.json')
        if submission.exists():
            raise RuntimeError('Interrupted experiment needs receipt inspection; no repeat enqueue')
        provider.phase=case['id']
        vision.client.chat.completions.phase=case['id']+':vision'
        metadata={'engine':'answer_first_text','visual_policy':'auto','paper_sha256s':case['paper_ids'],
                  'execution_profile':'frozen_acceptance'}
        kwargs=dict(question=case['question'],budget=Budget(unlimited_llm_usage=True,max_seconds=900),metadata=metadata)
        if case.get('parent_scenario_id'):
            parent=json.loads((folder/(case['parent_scenario_id']+'-result.json')).read_text(encoding='utf-8'))
            queued=runtime.enqueue_follow_up(parent_run_id=parent['run']['run_id'],**kwargs)
        else:
            queued=runtime.enqueue(**kwargs)
        rid=queued['run']['run_id']
        save(submission,{'run_id':rid,'question':case['question'],'source_ids':case['paper_ids']})
        then=time.monotonic()
        try:
            runtime.resume(rid)
        except Exception as exc:
            save(folder/(case['id']+'-exception.json'),{'error_type':type(exc).__name__})
        result=runtime.status(rid)
        result['experiment_elapsed_seconds']=round(time.monotonic()-then,3)
        save(result_path,result)
        final=result.get('final_answer') or {}
        (folder/(case['id']+'-answer.md')).write_text(str(final.get('answer','')),encoding='utf-8')
        print(json.dumps({'arm':arm,'case':case['id'],'status':result['run']['status'],
            'elapsed_seconds':result['experiment_elapsed_seconds'],'calls':result['run']['used_llm_calls'],
            'tokens':result['run']['used_tokens'],'answer_chars':len(str(final.get('answer','')))},ensure_ascii=True),flush=True)
    save(folder/('snapshot-'+str(time.time_ns())+'.json'),campaign.snapshot())


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['local','mcp'],required=True)
    parser.add_argument('--case',action='append')
    parser.add_argument('--output-folder',type=Path)
    parser.add_argument('--model')
    parser.add_argument('--source-baseline')
    parser.add_argument('--freeze-reviewed-sources', action='store_true')
    opt=parser.parse_args()
    execute(opt.arm,opt.case,output_folder=opt.output_folder,model=opt.model,source_baseline=opt.source_baseline,
            freeze_reviewed_sources=opt.freeze_reviewed_sources)
