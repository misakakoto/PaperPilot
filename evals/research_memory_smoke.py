"""Four handcrafted live regression cases; NOT a public benchmark or accuracy score."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from paperpilot_agentic.config import RuntimeSettings,read_api_key_file
from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.models import Budget
from paperpilot_agentic.providers import OpenAICompatibleLLM
from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
from paperpilot_agentic.durable_reader import DurableResearchModel
from paperpilot_agentic.research_memory import advance_memory,parent_memory
from paperpilot_agentic.follow_up_reader import resolve_follow_up


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    db = RuntimeDatabase(args.output/'temporary-runtime.sqlite3')
    settings = RuntimeSettings('deepseek',read_api_key_file(args.key_file),'https://api.deepseek.com',
        'deepseek-v4-flash',False,0.0,1500,90,args.output,args.output/'temporary-runtime.sqlite3',None,'smoke',())
    provider = OpenAICompatibleLLM(settings,max_retries=0)
    parent = None
    for i in range(16):
        question = ('比较 Alpha 与 Beta，只讨论相同数据集上的实验，不能把作者推测当作实证。' if i==0 else
                    '请记住：后续都只比较 F1，不比较准确率。' if i==2 else f'继续讨论第{i}个实验设置。')
        rid = db.create_run(question,Budget(),metadata={'conversation_id':'regression'})
        row = db.get_run(rid)
        state = advance_memory(row,parent_memory(db,parent) if parent else None,
                               continue_task=parent is not None,scope={'alpha','beta'})
        db.update_run_metadata(rid,{'reader_task_memory':state})
        parent = db.get_run(rid)
    questions = ['按最开始的限制和我要求记住的指标，重新比较这两篇。',
                 '现在换个主题，介绍机器人路径规划，不再比较论文。',
                 '仍比较这两篇，但现在改为比较准确率，不要再沿用 F1。',
                 '请沿用已更新的指标，继续比较这两篇。']
    (args.output/'manifest.json').write_text(json.dumps({'kind':'handcrafted regression, not benchmark',
        'questions':questions,'max_provider_attempts':4,'retries':0},ensure_ascii=False),encoding='utf-8')
    total = 0
    for index,question in enumerate(questions):
        if index==3:
            for step in range(7):
                text = questions[2] if step==0 else f'继续讨论设置{step}。'
                saved_id = db.create_run(text,Budget(),metadata={'conversation_id':'regression'})
                saved = db.get_run(saved_id)
                db.update_run_metadata(saved_id,{'reader_task_memory':advance_memory(
                    saved,parent_memory(db,parent),continue_task=True,scope={'alpha','beta'})})
                parent = db.get_run(saved_id)
        rid = db.create_run(question,Budget(max_llm_calls=3),metadata={'conversation_id':'regression'})
        class Recorder:
            def complete(self,messages,*,before_attempt,**kwargs):
                def reserve():
                    nonlocal total
                    if total>=4:
                        raise RuntimeError('Live regression call cap reached')
                    before_attempt()
                    total += 1
                    (args.output/f'{index}-request.json').write_text(json.dumps(
                        {'messages':messages,**kwargs},ensure_ascii=False),encoding='utf-8')
                response = provider.complete(messages,before_attempt=reserve,**kwargs)
                (args.output/f'{index}-response.json').write_text(json.dumps(asdict(response),ensure_ascii=False),encoding='utf-8')
                return response
        model = DurableResearchModel(PersistentLLMRuntime(db,Recorder()),rid,stage='follow_up_resolution')
        try:
            result = resolve_follow_up(question,{'prior_question':parent['question'],
                'task_history':parent_memory(db,parent)},model)
        except Exception as exc:
            result = {'error_type':type(exc).__name__}
        (args.output/f'{index}-result.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
        print(json.dumps({'index':index,'result':{k:v for k,v in result.items() if k!='call'}},ensure_ascii=False),flush=True)
    print(json.dumps({'provider_attempts':total}))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--key-file',type=Path,required=True)
    run(p.parse_args())
