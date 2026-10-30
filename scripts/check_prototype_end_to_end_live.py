"""Bounded development run through the real workbench and frozen prototype engine.

init never calls the provider. advance performs at most one decision, preserving
the same ledger. Approval is separate and binds an explicitly reviewed candidate;
this script never automatically approves or executes an unreviewed candidate.
Generated source, local excerpts and credentials must stay out of Git.
"""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from dataclasses import replace


QUESTION = (
    '我要做建筑变化检测研究，已有开源SNUNet_ECAM。请先从本地论文库调研近年的相关方法，'
    '重点分析屋顶纹理和阴影造成伪变化的问题，基于实际读取的原文提出一个可证伪的改进假设，'
    '在另存的SNUNet_ECAM副本上实现一个保留双时相主干的完整模型原型。'
    '明确区分论文已报告的方法、你的推断与未验证的新假设；资料覆盖不足应说明范围。'
    '模型必须支持不同尺寸及非方形输入，输出同尺寸单通道raw logits，参数不超过1300万。'
    '使用独立CPU工程验证及真实反馈修复，最后交付代码、验证范围和后续对照实验建议。'
    '本轮不训练、不使用GPU，不宣称已证明新颖性或可以发表。'
)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


class SerialBackground:
    def submit(self, rid, callback):
        pass  # advance is an explicit command, never a hidden model worker.

    def active(self, rid):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'advance', 'inspect', 'approve'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--license', type=Path)
    parser.add_argument('--candidate-id')
    parser.add_argument('--review-file', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    if args.action == 'init':
        if not args.baseline or not args.license:
            parser.error('init requires --baseline and --license')
        out.mkdir(parents=True, exist_ok=False)
        source = out / 'frozen-source' / 'src'
        shutil.copytree(root / 'src', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        freeze = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in source.rglob('*') if p.is_file()}
        save(out / 'source-freeze.json', freeze)
        shutil.copyfile(args.baseline, out / 'baseline.py')
        shutil.copyfile(args.license, out / 'LICENSE')
    source = out / 'frozen-source' / 'src'
    sys.path.insert(0, str(source))
    freeze = json.loads((out / 'source-freeze.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((source / p).read_bytes()).hexdigest() == sha for p, sha in freeze.items())
    from paperpilot_agentic.bootstrap import build_runtime
    from paperpilot_agentic.config import RuntimeSettings
    from paperpilot_agentic.models import Budget, RunStatus
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    from paperpilot_agentic.prototype_harness import PrototypeHarness
    from paperpilot_agentic.prototype_workspace import digest
    from paperpilot_agentic.prototype_delivery import export_prototype
    from paperpilot_agentic.reference_methods import SOURCE_SHA, LICENSE_SHA, PIN
    from paperpilot_agentic.research_design import register_code_reference
    from paperpilot_agentic.workbench import ResearchWorkbench
    from paperpilot_agentic import runtime_profile

    settings = RuntimeSettings.from_env(require_api_key=True)
    if settings.base_url != 'https://dashscope.aliyuncs.com/compatible-mode/v1':
        raise ValueError('This test is scoped to the authorized Bailian compatible endpoint')
    memory = out / 'knowledge.sqlite3'
    if args.action == 'init':
        if not settings.memory_db or not settings.memory_db.is_file():
            raise ValueError('The indexed local paper library is required')
        with sqlite3.connect(settings.memory_db.as_uri() + '?mode=ro', uri=True) as original:
            with sqlite3.connect(memory) as copied:
                original.backup(copied)
    os.environ['PAPERPILOT_ENABLE_DENSE'] = 'false'
    os.environ['PAPERPILOT_ENABLE_RERANKER'] = 'false'
    os.environ['PAPERPILOT_ALLOW_MODEL_DOWNLOAD'] = 'false'
    settings = replace(settings, workspace=out / 'workspace', runtime_db=out / 'runtime.sqlite3',
                       memory_db=memory, fallback_models=(), thinking_enabled=False)
    rt = build_runtime(settings, llm_factory=lambda s: OpenAICompatibleLLM(s, max_retries=0))
    h = PrototypeHarness(rt)
    if args.action == 'init':
        assert hashlib.sha256((out / 'baseline.py').read_bytes()).hexdigest() == SOURCE_SHA
        assert hashlib.sha256((out / 'LICENSE').read_bytes()).hexdigest() == LICENSE_SHA
        parent = rt.db.create_run('Pinned public SNUNet baseline', Budget(),
                                  {'engine': 'answer_first_text', 'conversation_id': out.name})
        rt.db.update_run_status(parent, RunStatus.COMPLETED)
        eid = register_code_reference(rt, parent, source_url=PIN + 'models/Models.py',
            content=(out / 'baseline.py').read_text(encoding='utf-8'), license_url=PIN + 'LICENSE',
            license_text=(out / 'LICENSE').read_text(encoding='utf-8'))
        budget = {'max_llm_calls': 16, 'max_tokens': 180000, 'max_seconds': 3600,
                  'synthesis_reserve_tokens': 12000, 'repair_reserve_tokens': 8000}
        request = {'action': 'research_prototype', 'request_id': out.name,
                   'conversation_id': out.name, 'parent_run_id': parent,
                   'source_ids': [eid], 'question': QUESTION, 'prototype_literature': True,
                   'runtime_profile': getattr(runtime_profile, 'DEFAULT_PROFILE', runtime_profile.PROFILE),
                   'budget': budget,
                   'prototype_constraints': {'input_cases': [[1,48,64], [2,65,97], [1,80,112]],
                                             'max_parameters': 13000000}}
        rid = ResearchWorkbench(rt, SerialBackground()).submit(request)['run_id']
        assert rt.db.get_run(rid)['budget']['max_tokens'] == 180000
        save(out / 'scope.json', {'run_id': rid, 'request': request, 'model': settings.model,
             'base_url': settings.base_url, 'retrieval': 'local lexical; no dense/rerank in this development run',
             'budget': rt.db.get_run(rid)['budget'], 'training': False,
             'evaluation': 'new development question; not a heldout A/B or scientific effectiveness experiment'})
    else:
        scope = json.loads((out / 'scope.json').read_text(encoding='utf-8'))
        rid = scope['run_id']
        if settings.model != scope['model']:
            raise ValueError('Model identity changed; do not silently switch this task')
        if args.action == 'advance':
            h.advance(rid, max_turns=1)
        elif args.action == 'approve':
            state = h._load(rid)
            pending = state.get('pending') or {}
            if state['phase'] != 'WAITING_APPROVAL' or not args.candidate_id or not args.review_file:
                raise ValueError('Approve requires a pending exact candidate and a written host review')
            artifact = state['work']['artifacts'][args.candidate_id]
            if digest(artifact) != args.candidate_id or pending['binding']['code'] != args.candidate_id:
                raise ValueError('Candidate identity changed')
            review = json.loads(args.review_file.read_text(encoding='utf-8'))
            approval = rt.db.get_approval(pending['approval'])
            if (review.get('candidate_id') != args.candidate_id or review.get('approval_id') != pending['approval']
                    or review.get('approved') is not True
                    or review.get('source_sha256') != hashlib.sha256(artifact['content']['source'].encode()).hexdigest()
                    or review.get('request_sha256') != digest(approval['request'])):
                raise ValueError('Host review does not bind this exact source and execution request')
            approved = rt.approve(pending['approval'], actor='codex-delegated',
                                  comment=review['comment'])
            save(out / f"approval-{args.candidate_id[:12]}.json", approved)
    state = h._load(rid)
    save(out / 'state.json', state)
    save(out / 'status.json', rt.status(rid))
    save(out / 'events.json', rt.db.list_events(run_id=rid))
    for identity, artifact in state['work']['artifacts'].items():
        if artifact['kind'] != 'implement':
            continue
        folder = out / identity
        folder.mkdir(exist_ok=True)
        code = artifact['content']['source']
        (folder / 'model.py').write_text(code, encoding='utf-8')
        save(folder / 'config.json', artifact['content']['config'])
        (folder / 'candidate.diff').write_text(''.join(difflib.unified_diff(
            (out / 'baseline.py').read_text(encoding='utf-8').splitlines(keepends=True),
            code.splitlines(keepends=True), fromfile='pinned-SNUNet.py', tofile='candidate-model.py')),
            encoding='utf-8')
    (out / 'prototype-delivery.zip').write_bytes(export_prototype(rt, rid))
    run = rt.db.get_run(rid)
    print(json.dumps({'run_id': rid, 'phase': state['phase'], 'turn': state['turn'],
          'calls': run['used_llm_calls'], 'tokens': run['used_tokens'],
          'last': state.get('last_observation'), 'pending': state.get('pending'),
          'stop': state.get('stop_reason')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
