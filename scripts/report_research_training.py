"""Deterministic fact report and offline demo; never calls an LLM."""
import argparse
import html
import json
from pathlib import Path
import copy
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from paperpilot_agentic.orchestrator import ResearchAgentRuntime  # noqa: E402
from paperpilot_agentic.paired_training import campaign  # noqa: E402
from paperpilot_agentic.training_metrics import paired_summary, verify_trial  # noqa: E402
from paperpilot_agentic.training_profiles import load_training  # noqa: E402
from paperpilot_agentic.training_state import atomic_json, file_sha  # noqa: E402
from paperpilot_agentic.sandbox import _sanitized_environment  # noqa: E402
from paperpilot_agentic.models import SandboxPolicy  # noqa: E402


def facts(runtime, record, *, inspect_payload=False):
    binding, rid = record['binding_id'], record['run_id']
    root, manifest = load_training(runtime, binding)
    folder = runtime.workspace/'method-training'/'runs'/rid
    state = json.loads((folder/'state.json').read_bytes()) if (folder/'state.json').exists() else {'checks': {}, 'trials': {}}
    rejection_checks, entries = [], []
    for name, trial in state['trials'].items():
        if trial['status'] != 'VERIFIED':
            continue
        ref = trial['receipt']
        path = Path(ref['path'])
        if not path.resolve().is_relative_to(folder.resolve()) or file_sha(path) != ref['sha256']:
            raise ValueError('Verified report source changed')
        report = json.loads(path.read_bytes())
        seed, variant = name.split('-', 1)
        recomputed = verify_trial(report, root, manifest, seed=int(seed), variant=variant)
        if trial['metrics'] != recomputed:
            raise ValueError('Stored host metrics differ from newly recomputed counts')
        cp = report['checkpoint']
        payload = path.parent/'checkpoints'/cp['file']
        if payload.is_symlink() or payload.resolve().parent != (path.parent/'checkpoints').resolve() or file_sha(payload) != cp['sha256']:
            raise ValueError('Final checkpoint changed')
        entries.append(ref)
        forged = copy.deepcopy(report)
        forged['validation']['tp'] += 1
        try:
            verify_trial(forged, root, manifest, seed=int(seed), variant=variant)
        except ValueError:
            rejection_checks.append({'trial': name, 'forged_metric_rejected': True})
        else:
            raise ValueError('Forged metric accepted')
    fault_path = folder/'fault-validation'/'verification.json'
    fault = json.loads(fault_path.read_bytes()) if fault_path.exists() else None
    rng_path = folder/'rng-cursor-probe'/'verification.json'
    rng = json.loads(rng_path.read_bytes()) if rng_path.exists() else None
    budget = campaign(runtime).snapshot()
    display_trials = copy.deepcopy(state['trials'])
    for name, trial in display_trials.items():
        trial['gpu_charge_seconds'] = sum(r['charged'] for r in budget['launches']
            if Path(r['stage']).resolve() == (folder/'trials'/name).resolve())
    for launch in budget['launches']:
        location = Path(launch['stage'])
        if location.parent.resolve() != (folder/'trials').resolve():
            continue
        name = location.name
        if display_trials.get(name, {}).get('status') == 'VERIFIED':
            continue
        if launch['status'] in ('RUNNING', 'RESERVED', 'UNKNOWN', 'COMPLETED'):
            status = 'AWAITING_VERIFICATION' if launch['status'] == 'COMPLETED' else launch['status']
            display_trials[name] = {'status': status}
            pointer = location/'checkpoints'/'latest.json'
            if pointer.exists():
                display_trials[name]['committed_step'] = json.loads(pointer.read_bytes())['step']
    pairs = paired_summary(display_trials)
    run = runtime.db.get_run(rid)
    inspection = None
    if inspect_payload:
        target = Path(runtime.db.path).parent
        job_path = target/'payload-inspection-job.json'
        atomic_json(job_path, {'profile': str(root), 'binding_id': binding, 'trials': entries})
        env = _sanitized_environment(SandboxPolicy(network='host', env_allowlist=('PATH', 'SystemRoot', 'WINDIR', 'TEMP', 'TMP')), {})
        script = Path(__file__).with_name('inspect_training_payload.py')
        result = subprocess.run([manifest['python'], '-I', '-B', str(script), str(job_path)],
                                capture_output=True, text=True, timeout=180, env=env)
        if result.returncode:
            raise ValueError('CPU payload inspection failed: '+result.stderr)
        inspection = json.loads(result.stdout)
        atomic_json(target/'payload-inspection.json', inspection)
    return {'binding_id': binding, 'run_id': rid, 'runtime_status': run['status'],
        'reported_at_utc': datetime.now(timezone.utc).isoformat(),
        'verified_trials': sum(r['status'] == 'VERIFIED' for r in state['trials'].values()),
        'protocol': manifest['protocol'], 'environment': manifest['environment'],
        'source': {'author_source_sha256': manifest['files']['baseline.py'],
            'candidate_sha256': manifest['files']['candidate.py'], 'review_sha256': manifest['review_sha256'],
            'design_run_id': manifest['design_run_id'], 'baseline_binding_id': manifest['baseline_binding_id'],
            'worker_sha256': manifest['files']['worker.py'], 'python_sha256': manifest['python_sha256'],
            'official_splits': manifest['splits'], 'excluded_diagnostic_ids': len(manifest['excluded_ids'])},
        'trials': pairs, 'checks': state['checks'], 'fault_validation': fault,
        'rng_cursor_probe': rng,
        'forgery_checks': rejection_checks, 'payload_inspection': inspection,
        'gpu_budget': budget, 'new_llm_calls': len(runtime.db.list_llm_calls(rid)),
        'claims': {'innovation_verified': False, 'test_benchmark': False, 'statistical_significance': False,
                   'new_llm_analysis': False, 'realtime_qa_passed': False}}


def markdown(data, tests=None):
    p = data['protocol']
    budget = data['gpu_budget']
    lines = ['# SNUNet / SYSU-CD 可信研发案例验收', '',
        f"本轮完成 **{data['verified_trials']}/6** 组宿主复核训练，运行状态 `{data['runtime_status']}`。",
        '本报告从保存的预测和冻结标签重新统计，不调用模型；科研推断与工程事实分开。', '',
        '## 冻结身份与范围', '',
        f"- 绑定：`{data['binding_id']}`；运行：`{data['run_id']}`。",
        f"- 真实草案：`{data['source']['design_run_id']}`；作者源 SHA：`{data['source']['author_source_sha256']}`。",
        '- 来源来自先前已保存的论文摘要与固定作者代码，分别作为摘要依据与代码观察；未补造正文页码。',
        f"- 官方 train/val 固定哈希选择 512/256 图；排除 {data['source']['excluded_diagnostic_ids']} 个旧诊断 ID；不访问 test。",
        f"- seeds 7/11/19，batch={p['batch_size']}，256×256，FP32，20 epochs，无增广；AdamW lr=.001、weight_decay=.0001。",
        '- 同种子复制完整初始参数与缓冲区；同数据顺序；固定最终 epoch，不挑最优验证轮。',
        '- 保留候选未用的四个注意力权重以保证同初始化，故不声称总参数量减少。',
        f"- 环境：`{json.dumps(data['environment'], ensure_ascii=False)}`。", '',
        '## 全部成对结果', '',
        '指标列为百分数；差值为百分点，按候选减基线。只反映本项目固定小规模配置。', '',
        '| Seed | 变体 | 状态 | Precision | Recall | F1 | IoU | GPU记账秒 | 峰值显存 GiB |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    def percent(value):
        return '未定义' if value is None else f'{value*100:.2f}'
    for pair in data['trials']:
        for variant, label in (('baseline', '基线'), ('candidate', '候选')):
            r = pair[variant]
            if r['status'] == 'VERIFIED':
                values = ' | '.join(percent(r['metrics'][k]) for k in ('precision', 'recall', 'f1', 'iou'))
                lines.append(f"| {pair['seed']} | {label} | VERIFIED | {values} | {r['gpu_charge_seconds']:.2f} | {r['peak_memory_bytes']/1024**3:.3f} |")
            else:
                lines.append(f"| {pair['seed']} | {label} | {r['status']} | — | — | — | — | — | — |")
        if pair['delta']:
            values = ' | '.join(percent(pair['delta'][k]) for k in ('precision', 'recall', 'f1', 'iou'))
            lines.append(f"| {pair['seed']} | 成对差值 | 描述性观察 | {values} | — | — |")
    lines += ['', '## 恢复与成本', '',
        f"累计记账 **{budget['charged_seconds']:.2f} 秒 / 28,800 秒**；活跃预留 {budget['reserved_seconds']:.2f} 秒。",
        '未知进程耗时按完整预留额收费，所以该数字可能高于可确认的实际 GPU 时间；恢复不清零。',
        '每 25 步及暂停保存模型、优化器、Python/NumPy/Torch/CUDA RNG、epoch 与 batch 游标。',
        '只保留当前与前一 checkpoint 的完整 payload；全部提交哈希与步数记录保留。']
    preflight = data['checks'].get('preflight')
    if preflight:
        learning = preflight['learning']
        lines += [f"预检 100 步：前十步均值 {learning['first_10_mean']:.6f}，后十步均值 {learning['last_10_mean']:.6f}；峰值 {preflight['peak_memory_bytes']/1024**3:.3f} GiB。",
                  '这是固定四图学习预检，不能外推正式数据准确率。']
    if data['checks'].get('recovery'):
        lines += ['连续六步与第三步保存后重启：模型、优化器、RNG 摘要、逐步损失与梯度范数完全一致。']
    fault = data['fault_validation']
    if fault:
        lines += [f"强制中断：PID {fault['forced_pid']} 已确认退出，未提交完成回执；记账 {fault['interruption_charge_seconds']:.0f} 秒后从第三步恢复，与连续运行完全一致。",
                  '损坏 checkpoint 的独立副本被拒绝，原文件保留；该检查仅 CPU 读取。']
    if data.get('rng_cursor_probe'):
        lines += ['另执行纯数值随机流恢复：推进三个随机步骤后保存、额外扰动 11 步，再恢复继续三个步骤；Python/NumPy/Torch CPU/Torch CUDA 的输出逐项完全相同，非零 batch 游标恢复后匹配后续 12 个冻结 train ID。',
                  '该验证没有新增模型或优化器训练；其 GPU 进程使用同一账本。随机增广、分布式或多 worker DataLoader 的训练恢复仍未验收。']
    if data.get('payload_inspection'):
        count = len(data['payload_inspection']['trials'])
        lines += [f'另在 CPU 上直接加载 {count} 组实际 checkpoint，复核模型/优化器/RNG 摘要，并独立重建初始权重，确认分类头权重确实更新。该检查未初始化 CUDA。']
        if all(set(r.get('changed_identities_rejected', [])) == {'binding_id', 'protocol_sha256'}
               for r in data['payload_inspection']['trials']):
            lines += [f'{count} 组真实 checkpoint 均拒绝不同来源绑定或协议身份的恢复请求；只改变内存中的请求身份，原始文件不改写。']
    lines += [f"对 {len(data.get('forgery_checks', []))} 组真实结果的内存副本篡改 TP 计数，全部被独立标签复算拒绝；原始回执不改写。"]
    if tests:
        lines += [f"完整离线回归：{tests['passed']} passed，{tests['skipped']} skipped，{tests['failures']} failures/errors，{tests['seconds']:.2f} 秒。",
                  '离线回归证明对应版本的工程行为，不能作为答案正确率或科研准确率。']
    lines += ['', '## 研究推断与未执行项目', '',
        '注意力旁路的成对差值是观察结果；训练规模、无增广设置和固定样本限制了外推范围。',
        '若方向随种子变化，应先检查训练充分性、数据规模与机制假设；不据此选择更有利种子。',
        '任何较好结果均不证明论文创新、统计显著性或完整 test 性能。负结果与失败同样保留。',
        f"本轮新增 LLM 调用 **{data['new_llm_calls']}**；复用既有真实草案，新增模型设计与模型结果分析未执行。",
        '旧云端 42 次请求及 UNKNOWN、60 次/60 万 token 上限与暂停状态保留。',
        '问答实时三题仍 0/3；新版九题未完成；表格 3/10、公式 2/10，解析仍为实验功能。', '',
        '## 原始依据', '',
        '原始预测、损失日志、checkpoint、身份、数据副本与 SQLite 在 Git 外。报告中的每个 VERIFIED 结果保留原始回执路径及 SHA。',
        '宿主重新加载冻结协议、校验回执哈希与最终 checkpoint，并再次从冻结标签复算所有 Precision/Recall/F1/IoU。',
        '架构与边界见 [研发链路](architecture/RESEARCH_TRAINING_20261004.md)。']
    completed_pairs = [r for r in data['trials'] if r['delta'] and r['delta']['f1'] is not None]
    if completed_pairs:
        differences = [r['delta']['f1'] for r in completed_pairs]
        direction = ('全部低于基线' if all(x < 0 for x in differences) else
                     '全部高于基线' if all(x > 0 for x in differences) else '未呈现一致的改善或下降')
        observed = f"已完成 {len(differences)} 对的候选 F1 {direction}，平均成对差值 {sum(differences)/len(differences)*100:+.2f} 个百分点。"
        lines[lines.index('## 研究推断与未执行项目')+2:lines.index('## 研究推断与未执行项目')+2] = [observed, '该描述由固定规则计算，未调用模型；不能外推为机制因果或统计结论。', '']
    lines += ['训练耗时为本机执行观察，包含 checkpoint 与恢复成本；没有控制其他主机进程，不作为吞吐基准。']
    return '\n'.join(lines)+'\n'


def page(data):
    rows = []
    for pair in data['trials']:
        for variant, label in (('baseline', '基线'), ('candidate', '候选')):
            r = pair[variant]
            values = [r.get('metrics', {}).get(k) for k in ('precision', 'recall', 'f1', 'iou')]
            fields = ''.join('<td>'+('—' if v is None else f'{v*100:.2f}%')+'</td>' for v in values)
            rows.append(f'<tr data-seed="{pair["seed"]}"><td>{pair["seed"]}</td><td>{label}</td><td>{html.escape(r["status"])}</td>{fields}</tr>')
    budget = data['gpu_budget']
    escaped = html.escape(json.dumps(data, ensure_ascii=False, indent=2))
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PaperPilot · 冻结研发案例</title><style>
body{{margin:0;background:#eef2f6;color:#182434;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1060px;margin:38px auto;padding:28px}}h1{{font-size:32px;line-height:1.25}}h2{{font-size:22px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}article,section{{background:white;padding:22px;border-radius:14px;margin:18px 0;box-shadow:0 4px 18px #1633520a}}article strong{{font-size:28px;display:block}}.muted{{color:#58697b}}table{{width:100%;border-collapse:collapse}}th,td{{padding:12px;border-bottom:1px solid #e2e8ef;text-align:left}}select{{font:inherit;padding:6px 10px}}pre{{white-space:pre-wrap;word-break:break-all;font-size:12px}}@media(max-width:750px){{main{{padding:14px}}.cards{{grid-template-columns:1fr}}section{{overflow:auto}}}}
</style><main><p class="muted">PaperPilot · 证据与冻结协议驱动的研发 Agent</p><h1>训练能恢复，结果可复核</h1>
<p>已保存真实草案 → 宿主审阅 → 冻结协议 → 成对训练 → 独立指标复算。此页只读取保存事实，不触发模型或 GPU 调用。</p>
<div class="cards"><article>宿主复核训练<strong>{data['verified_trials']} / 6</strong><span class="muted">状态 {html.escape(data['runtime_status'])}</span></article>
<article>累计 GPU 记账<strong>{budget['charged_seconds']/60:.1f} 分钟</strong><span class="muted">上限 480 分钟 · 未知时长保守计费</span></article>
<article>新增 LLM 调用<strong>{data['new_llm_calls']}</strong><span class="muted">复用既有真实草案</span></article></div>
<section><h2>固定小规模成对结果</h2><p class="muted">官方 train / val 512 / 256 图，排除旧诊断输入；20 epochs，batch 4，FP32，无增广。相同初始化和数据顺序，不使用 test。</p>
<label>种子 <select id="seed"><option value="all">全部</option><option>7</option><option>11</option><option>19</option></select></label>
<table><thead><tr><th>Seed</th><th>变体</th><th>状态</th><th>Precision</th><th>Recall</th><th>F1</th><th>IoU</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section><h2>恢复与结论边界</h2><p>每 25 步及暂停保存完整 checkpoint。连续运行、主动暂停恢复与强制进程中断分别验收；损坏 checkpoint 拒绝，未确认进程状态拦截。</p>
<p>保留全部种子、负结果和失败。这是小规模描述性对照，不证明创新、显著性或作者成绩复现。三道问答实时演示仍 0/3，解析仍为实验功能。</p></section>
<section><details><summary>查看协议、来源 SHA、回执与精确指标</summary><pre>{escaped}</pre></details></section></main>
<script>document.getElementById('seed').addEventListener('change',e=>document.querySelectorAll('tr[data-seed]').forEach(r=>r.hidden=e.target.value!=='all'&&r.dataset.seed!==e.target.value));</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--db', required=True)
    parser.add_argument('--publish-doc', action='store_true')
    parser.add_argument('--inspect-payload', action='store_true')
    args = parser.parse_args()
    target = Path(args.db).parent
    record = json.loads((target/'campaign.json').read_bytes())
    runtime = ResearchAgentRuntime(workspace=args.workspace, db_path=args.db)
    data = facts(runtime, record, inspect_payload=args.inspect_payload)
    tests = None
    xml = next((target/name for name in ('offline-regression-release.xml', 'offline-regression-final.xml',
                                        'offline-regression.xml') if (target/name).exists()),
               target/'offline-regression-release.xml')
    if xml.exists():
        suite = ET.parse(xml).getroot().find('testsuite')
        total, skipped = int(suite.get('tests')), int(suite.get('skipped'))
        failures = int(suite.get('failures'))+int(suite.get('errors'))
        tests = {'passed': total-skipped-failures, 'skipped': skipped, 'failures': failures, 'seconds': float(suite.get('time')),
                 'source': str(xml), 'sha256': file_sha(xml)}
    data['offline_tests'] = tests
    atomic_json(target/'facts.json', data)
    text = markdown(data, tests)
    (target/'report.md').write_text(text, encoding='utf-8')
    rendered = page(data)
    (target/'index.html').write_text(rendered, encoding='utf-8')
    demo = target/'demo'
    demo.mkdir(exist_ok=True)
    (demo/'index.html').write_text(rendered, encoding='utf-8')
    if args.publish_doc:
        if (data['verified_trials'] != 6 or data['runtime_status'] != 'COMPLETED' or not tests or tests['failures']
                or not data['payload_inspection'] or len(data['payload_inspection']['trials']) != 6):
            raise ValueError('Final document requires six verified trials and a completed offline regression')
        (Path(__file__).resolve().parents[1]/'docs'/'RESEARCH_TRAINING_ACCEPTANCE_20261004.md').write_text(text, encoding='utf-8')
    print(json.dumps({'verified_trials': data['verified_trials'], 'status': data['runtime_status'],
        'gpu_charged_seconds': data['gpu_budget']['charged_seconds'], 'output': str(target)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
