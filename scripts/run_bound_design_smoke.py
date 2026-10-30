"""Paid opt-in reader -> verified-baseline local draft. Never executes proposed code.

Uses an existing completed validation receipt in its original workspace/conversation.
All requests, failures and unchanged responses remain in an ignored attempt folder.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperpilot_agentic.config import RuntimeSettings, read_dashscope_api_key_file
from paperpilot_agentic.durable_reader import DurableResearchModel, run_durable_reader
from paperpilot_agentic.method_drafts import verified_baseline
from paperpilot_agentic.models import Budget
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.providers import OpenAICompatibleLLM
from paperpilot_agentic.reference_methods import load_profile, sha
from paperpilot_agentic.research_design import enqueue_design, register_code_reference
from scripts.run_research_loop_live import RecordedModel


QUESTION = ('基于所提供的原文片段和作者固定版本代码，提出一个针对 SNUNet_ECAM 注意力融合的单一消融假设，'
    '并生成只替换 forward 函数的局部代码草案。不修改 __init__、训练脚本、损失、数据、指标、环境；'
    '保留函数签名和单元素 tuple logits 返回接口。区分论文已报告机制、代码观察和你的新假设。'
    '已有验证只是随机初始化的两步小样本检查，并不说明基线精度合格；本次不运行任何训练。'
    '你要说明成对实验需固定哪些因素、观察什么指标、失败时怎么分析；不要编造结果或声称创新已成立。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--validation-run', required=True)
    parser.add_argument('--source-file', type=Path, required=True,
                        help='Host-reviewed public excerpts: {sources:[{content,source_uri,page}], scope:...}')
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--execute-paid', action='store_true')
    args = parser.parse_args()
    if not args.execute_paid:
        print(json.dumps({'question': QUESTION, 'model': 'qwen-turbo', 'max_actual_requests': 6,
                          'api_requests': 0, 'training_executed': False}, ensure_ascii=True))
        return
    if not args.key_file:
        parser.error('--execute-paid requires --key-file')
    workspace = args.workspace.resolve(strict=True)
    if not (workspace/'runtime.sqlite3').is_file():
        raise ValueError('Use the original validation workspace, not a new empty database')
    if args.source_file.stat().st_size > 24000:
        raise ValueError('Source window too large; select complete relevant paragraphs')
    raw = args.source_file.read_bytes()
    data = json.loads(raw)
    rows = data.get('sources')
    scope = data.get('scope') or data.get('metadata', {}).get('source_scope')
    if (not isinstance(rows, list) or not 1 <= len(rows) <= 12 or not isinstance(scope, str) or not scope.strip()
            or any(not isinstance(r, dict) or not isinstance(r.get('content'), str)
                   or not r['content'].strip() or not str(r.get('source_uri', '')).startswith('https://') for r in rows)):
        raise ValueError('Expected bounded public original sources with an explicit scope')
    inspection = ResearchAgentRuntime(workspace=workspace, db_path=workspace/'runtime.sqlite3')
    validation = inspection.db.get_run(args.validation_run)
    root, profile = load_profile(workspace, validation['metadata']['binding_id'])
    code = (root/'upstream.py').read_bytes().decode('utf-8')
    license_text = (root/'LICENSE').read_bytes().decode('utf-8')
    verified_baseline(inspection, args.validation_run,
        conversation_id=validation['metadata']['conversation_id'], references=[{
            'evidence_id': 'preflight-only', 'content': code, 'source_uri': profile['source_url'],
            'provenance': {'license_url': profile['license_url'], 'license_text': license_text}}])
    output = Path(tempfile.mkdtemp(prefix='bound-design-', dir=workspace))
    (output/'api').mkdir()
    def save(name, value):
        with (output/name).open('x', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
    settings = RuntimeSettings('qwen', read_dashscope_api_key_file(args.key_file),
        'https://dashscope.aliyuncs.com/compatible-mode/v1', 'qwen-turbo', False, 0.0, 6000, 60,
        workspace, workspace/'runtime.sqlite3', None, 'bound-design-smoke', ())
    model = RecordedModel(OpenAICompatibleLLM(settings, max_retries=0), output/'api', maximum=6)
    runtime = ResearchAgentRuntime(workspace=workspace, db_path=settings.runtime_db, llm=model,
                                  text_reader_protocol='bound-design-qwen-turbo-v1')
    summary = {'output': str(output), 'validation_run_id': args.validation_run,
               'training_executed': False, 'source_scope': scope, 'model': settings.model}
    save('manifest.json', {**summary, 'source_sha256': sha(raw), 'question': QUESTION,
                          'max_actual_requests': 6, 'source_snapshot': data})
    try:
        parent = runtime.enqueue_follow_up(parent_run_id=args.validation_run,
            question='仅依据提供的原文片段说明 SNUNet 的连接与注意力机制；明确这些片段未能支持的结论，'
                     '不要把短训练回执当作论文结论。',
            budget=Budget(max_llm_calls=4, max_tokens=40000, max_seconds=600,
                          synthesis_reserve_calls=0, repair_reserve_calls=0))['run']['run_id']
        report = run_durable_reader(runtime.db, parent,
            [{**r, 'evidence_id': f'original-{index}'} for index, r in enumerate(rows)],
            DurableResearchModel(runtime.llm_runtime, parent, stage='bound_design_reading'),
            protocol_id=runtime.text_reader_protocol)
        save('reading.json', report)
        summary['reading_run_id'] = parent
        if runtime.db.get_run(parent)['status'] != 'COMPLETED' or not report.get('answer'):
            raise ValueError('Reading did not complete; do not bypass it')
        reference = register_code_reference(runtime, parent, source_url=profile['source_url'], content=code,
            license_url=profile['license_url'], license_text=license_text)
        source_ids = [e['evidence_id'] for e in runtime.db.list_evidence(parent)
                      if e['provenance'].get('origin') == 'answer-first-source-material']
        run_id = enqueue_design(runtime, parent, question=QUESTION, source_ids=source_ids,
            reference_ids=[reference], validation_run_id=args.validation_run)['run']['run_id']
        result = runtime.resume(run_id)
        save('result.json', result)
        before = model.attempts
        replay = runtime.resume(run_id)
        summary.update(status=result['run']['status'], error=result['run'].get('error'),
            design_run_id=run_id, replay_extra_requests=model.attempts-before,
            replay_same_status=replay['run']['status'] == result['run']['status'])
    except Exception as exc:
        summary.update(status='ERROR', error_type=type(exc).__name__)
    finally:
        summary['actual_requests'] = model.attempts
        save('summary.json', summary)
        print(json.dumps(summary, ensure_ascii=True), flush=True)
    if summary.get('status') != 'COMPLETED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
