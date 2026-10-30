"""Paid opt-in reading-to-design smoke using public sources; never executes draft code."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperpilot_agentic.config import RuntimeSettings, read_dashscope_api_key_file
from paperpilot_agentic.durable_reader import DurableResearchModel, run_durable_reader
from paperpilot_agentic.models import Budget
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.providers import OpenAICompatibleLLM
from paperpilot_agentic.research_design import enqueue_design, register_code_reference
from scripts.run_research_loop_live import RecordedModel

PIN = 'https://raw.githubusercontent.com/likyoo/Siam-NestedUNet/aed3f1833f3d9d3eccba8c894fd097f358b5329e/'
CODE_SHA = 'c14f3da927c21ea5503ac333a42afd3e5a9ed05efece962b637eb10eb15186c5'
# Short original excerpt; explicitly NOT full-paper reading or benchmark evidence.
EXCERPT = ('SNUNet-CD alleviates the loss of localization information in the deep layers of neural '
           'network through compact information transmission between encoder and decoder')
PAPER_URL = 'https://ieeexplore.ieee.org/abstract/document/9355573'
QUESTION = ('基于所给论文短摘录和作者实现，提出一个可检验的纯光学变化检测假设，'
            '独立编写简化基线与单一改动候选的代码草稿。使用 SYSU-CD 的 train/val，不打开 test。'
            '训练在 npy3.11 环境运行，但本次只生成草稿，不执行；每个种子每个变体最多20步，'
            '只验证链路，不宣称新方法有效、复现原模型或取得论文精度。不要复制完整SNUNet。'
            '给出成对固定子集、相同种子/初始化与训练设置、F1/IoU统计、超时与失效条件。')


def download(url):
    chunks = []
    with httpx.stream('GET', url, timeout=20, follow_redirects=False) as response:
        response.raise_for_status()
        size = 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > 32000:
                raise ValueError('Reference response exceeds the bounded source window')
            chunks.append(chunk)
    return b''.join(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--execute-paid', action='store_true')
    args = parser.parse_args()
    if not args.execute_paid:
        print(json.dumps({'question': QUESTION, 'max_actual_requests': 6,
                          'model': 'qwen-turbo', 'api_requests': 0, 'training': False}, ensure_ascii=True))
        return
    if not args.key_file:
        parser.error('--execute-paid requires --key-file')
    root = Path(__file__).resolve().parents[1] / 'outputs'
    root.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='research-design-smoke-', dir=root))
    def save(name, value):
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    code = download(PIN+'models/Models.py')
    if hashlib.sha256(code).hexdigest() != CODE_SHA:
        raise ValueError('Pinned author code digest mismatch')
    license_text = download(PIN+'LICENSE').decode('utf-8')
    save('sources.json', {'paper_uri': PAPER_URL, 'excerpt': EXCERPT,
        'scope': 'short abstract excerpt only', 'code_url': PIN+'models/Models.py',
        'code_sha256': CODE_SHA, 'code': code.decode('utf-8'),
        'license_url': PIN+'LICENSE', 'license_text': license_text})
    (output/'api').mkdir()
    settings = RuntimeSettings('qwen', read_dashscope_api_key_file(args.key_file),
        'https://dashscope.aliyuncs.com/compatible-mode/v1', 'qwen-turbo', False, 0.0, 6000, 60,
        output, output/'runtime.sqlite3', None, 'design-smoke', ())
    model = RecordedModel(OpenAICompatibleLLM(settings, max_retries=0), output/'api', maximum=6)
    runtime = ResearchAgentRuntime(db_path=settings.runtime_db, workspace=output,
                                  llm=model, text_reader_protocol='design-smoke-qwen-turbo-v1')
    summary = {'output': str(output), 'training_executed': False, 'source_scope': 'short excerpt + pinned code'}
    save('manifest.json', {**summary, 'question': QUESTION, 'model': settings.model,
                          'max_actual_requests': 6})
    try:
        parent = runtime.enqueue(question='Explain only what this SNUNet abstract excerpt says; '
            'identify what cannot be established without full paper and experiments.',
            budget=Budget(max_llm_calls=4, max_tokens=40000, synthesis_reserve_calls=0,
                          repair_reserve_calls=0), metadata={'engine': 'answer_first_text'})['run']['run_id']
        report = run_durable_reader(runtime.db, parent,
            [{'evidence_id': 'abstract', 'content': EXCERPT, 'source_uri': PAPER_URL}],
            DurableResearchModel(runtime.llm_runtime, parent, stage='design_smoke_reading'),
            protocol_id=runtime.text_reader_protocol)
        save('reading.json', report)
        if runtime.db.get_run(parent)['status'] != 'COMPLETED' or not report.get('answer'):
            raise ValueError('Reading stage did not complete; do not bypass it')
        reference = register_code_reference(runtime, parent, source_url=PIN+'models/Models.py',
            content=code.decode('utf-8'), license_url=PIN+'LICENSE', license_text=license_text)
        source_ids = [e['evidence_id'] for e in runtime.db.list_evidence(parent)
                      if e['provenance'].get('origin') == 'answer-first-source-material']
        queued = enqueue_design(runtime, parent, question=QUESTION, source_ids=source_ids,
                                reference_ids=[reference])
        result = runtime.resume(queued['run']['run_id'])
        save('result.json', result)
        summary.update(status=result['run']['status'], error=result['run'].get('error'),
                       design_run_id=result['run']['run_id'])
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
