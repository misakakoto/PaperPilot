"""Live protocol smoke with synthetic public-to-share text, NOT a benchmark.

Run with PAPERPILOT_LLM_PROVIDER=deepseek and PYTHONPATH=src. No key is printed.
Artifacts and raw model records stay under the ignored outputs directory.
"""
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit

from paperpilot_agentic.config import RuntimeSettings
from paperpilot_agentic.providers import OpenAICompatibleLLM
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.models import Budget
from paperpilot_agentic.durable_reader import persist_reader_evidence
from paperpilot_agentic.workset_synthesis import synthesize_worksets


def main():
    settings = RuntimeSettings.from_env()
    if settings.provider != 'deepseek' or not settings.api_key:
        raise ValueError('A configured DeepSeek provider is required')
    if urlsplit(settings.base_url).hostname != 'api.deepseek.com' or not settings.model.startswith('deepseek'):
        raise ValueError('Smoke test requires an explicit DeepSeek endpoint and model')
    max_calls = int(os.getenv('WORKSET_SMOKE_MAX_CALLS', '4'))
    if not 1 <= max_calls <= 4:
        raise ValueError('Smoke test allows at most four attempts')
    Path('outputs').mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='workset-synthesis-smoke-', dir='outputs')).resolve()
    runtime = ResearchAgentRuntime(db_path=output/'runtime.sqlite3', workspace=output,
        llm=OpenAICompatibleLLM(settings, max_retries=0), text_reader_protocol='synthetic-live-synthesis-v1')
    rid = runtime.enqueue(question='比较两个合成方法的机制，说明能否直接比较精度；不要执行实验。',
        budget=Budget(max_llm_calls=max_calls, max_tokens=100000, max_seconds=180,
                      synthesis_reserve_calls=0, repair_reserve_calls=0,
                      synthesis_reserve_tokens=0, repair_reserve_tokens=0))['run']['run_id']
    evidence = persist_reader_evidence(runtime.db, rid, [
        {'evidence_id': 'synthetic_a', 'source_uri': 'synthetic://A', 'content':
         '合成材料A：方法A使用局部卷积模块。在数据集D的划分X上，F1为90%。该报告没有跨划分测试。'},
        {'evidence_id': 'synthetic_b', 'source_uri': 'synthetic://B', 'content':
         '合成材料B：方法B使用全局注意力模块。在数据集D的划分Y上，F1为92%。划分Y与划分X不同，报告没有同划分对照。'},
    ])
    pages = [{'label': name, 'conditions': {'no_train': {'text': '不执行实验', 'applies_to': None}},
              'pending_actions': {name: {'text': '比较机制；不同数据划分的分数不可直接排名'}}}
             for name in ('A', 'B')]
    manifest = {'pages': pages, 'protocol': runtime.text_reader_protocol}
    reports = [{'answer': 'Untrusted prior prose: method B is definitively superior.',
                'workset_context': {'selected_evidence_ids': [row['evidence_id']]}} for row in evidence]
    result = synthesize_worksets(runtime, rid, manifest, reports, guard=lambda: None)
    current = runtime.db.get_run(rid)
    summary = {'status': result['status'], 'calls': current['used_llm_calls'],
               'tokens': current['used_tokens'], 'model': settings.model,
               'dataset': 'synthetic protocol fixture, not public benchmark',
               'artifact_directory': str(output)}
    (output/'result.json').write_text(json.dumps({'summary': summary, 'result': result,
        'manifest': manifest, 'evidence': evidence}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))
    if result['status'] != 'REVIEWED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
