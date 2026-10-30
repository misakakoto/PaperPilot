"""Real-model smoke on public QASPER abstracts, not a QASPER accuracy score."""
import argparse
import json
from pathlib import Path

from paperpilot_agentic.config import RuntimeSettings, read_api_key_file
from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.durable_reader import DurableResearchModel
from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
from paperpilot_agentic.models import Budget
from paperpilot_agentic.providers import OpenAICompatibleLLM
from paperpilot_agentic.research_notebook import ResearchNotebook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--key-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    db = RuntimeDatabase(args.output/'runtime.sqlite3')
    question = '整理论文摘要中明确报告的方法特点、数据集与局限；没有分数不要补写，更不能宣称是当前SOTA。'
    rid = db.create_run(question,Budget(max_tokens=150000,max_llm_calls=6),metadata={'conversation_id':'public-abstract-smoke'})
    data = json.loads(args.dataset.read_text(encoding='utf-8'))
    book = ResearchNotebook(db,rid)
    selected = sorted(data)[:2]
    for pid in selected:
        book.add_source(f'qasper://{pid}/abstract',data[pid]['abstract'],max_bytes=12000)
    settings = RuntimeSettings('deepseek',read_api_key_file(args.key_file),'https://api.deepseek.com',
        'deepseek-v4-flash',False,0.0,3000,90,args.output,db.path,None,'smoke',())
    model = DurableResearchModel(PersistentLLMRuntime(db,OpenAICompatibleLLM(settings,max_retries=0)),rid,stage='notebook_extract')
    (args.output/'manifest.json').write_text(json.dumps({'scope':'public abstracts, engineering smoke only; no QASPER QA score',
        'paper_ids':selected,'question':question,'max_actual_requests':4,'provider_retries':0},ensure_ascii=False,indent=2),encoding='utf-8')
    for _ in range(4):
        if not book.process_next(model):
            break
    result = {'overview':book.overview(),'requests':db.get_run(rid)['used_llm_calls'],'notes':book.notes(),
              'raw_responses':'runtime.sqlite3 notebook_batches.response','raw_requests':'runtime.sqlite3 context_snapshots'}
    (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
