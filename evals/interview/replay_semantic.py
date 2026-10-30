"""Read-only saved-answer diagnostics; cannot dispatch or grant live acceptance."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .cases import acceptance_cases
from .semantic import dual_observations, VERSION


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(root, results_dir, output_dir, *, review_dir=None):
    root, results_dir, output_dir = map(lambda p: Path(p).resolve(), (root, results_dir, output_dir))
    if output_dir == results_dir or output_dir in results_dir.parents or results_dir in output_dir.parents:
        raise ValueError('Replay output must be separate from preserved inputs')
    output_dir.mkdir(parents=True, exist_ok=False)
    cases = acceptance_cases(root)
    saved, identities = {}, {}
    for case in cases:
        path = results_dir / (case['id'] + '-result.json')
        if path.exists():
            saved[case['id']] = json.loads(path.read_text(encoding='utf-8'))
            identities[str(path)] = sha(path)
    rows = []
    for case in cases:
        result = saved.get(case['id'])
        if result is None:
            rows.append({'case': case['id'], 'question': case['question'], 'status': 'NOT_RUN'})
            continue
        if result['run']['question'] != case['question']:
            raise ValueError('Saved result does not match the frozen original question')
        path = results_dir / (case['id'] + '-result.json')
        parent = saved.get(case.get('parent_scenario_id'))
        review_path = Path(review_dir) / (case['id'] + '.json') if review_dir else None
        review = json.loads(review_path.read_text(encoding='utf-8')) if review_path and review_path.exists() else None
        dual = dual_observations(case, result, result_sha256=sha(path), parent=parent,
                                 review=review, replay=True)
        legacy_path = results_dir / 'source-reviews' / (case['id'] + '.json')
        legacy = json.loads(legacy_path.read_text(encoding='utf-8')) if legacy_path.exists() else None
        row = {'case': case['id'], 'question': case['question'], 'run_id': result['run']['run_id'],
               'status': result['run']['status'], 'original_result_path': str(path),
               'original_result_sha256': sha(path),
               'original_answer': (result.get('final_answer') or {}).get('answer') or '',
               'legacy_review_unchanged': legacy, 'dual_scoring': dual}
        prior = result['run']['metadata'].get('reader_report') or {}
        if prior.get('draft') and prior.get('model_audit'):
            # Replay guards over the saved candidate; never synthesize a new
            # answer or upgrade a saved model verdict to independently true.
            from paperpilot_agentic.answer_pipeline import initialize, apply_guards, render_answer, AnswerOptions
            from paperpilot_agentic.answer_contracts import evidence_spans

            def no_provider(*args, **kwargs):
                raise AssertionError('Offline replay cannot call a provider')

            selected_ids = {span['evidence_id'] for span in prior.get('source_spans', [])}
            sources = [e for e in result.get('evidence', []) if e['evidence_id'] in selected_ids]
            context = initialize(case['question'], sources, None,
                AnswerOptions(frozen_draft=deepcopy(prior['draft']),
                              requested_tables=result['run']['metadata'].get('requested_table_targets')),
                None)
            draft = context.options.frozen_draft
            audit = apply_guards(context, draft, deepcopy(prior['model_audit']),
                                 evidence_spans(context.rows), [])
            render_answer(context, draft, audit, no_provider)
            row['current_guard_diagnostic'] = {**context.report, 'evaluation_mode': 'REPLAY',
                'note': 'Counterfactual guard/presentation result over the old draft; no new generation, audit or live acceptance.'}
        else:
            row['current_guard_diagnostic'] = {'status': 'NOT_RUN',
                'reason': 'No completed source-bound draft/audit pair; unpublished draft is not an answer.'}
        rows.append(row)
    assert all(sha(Path(p)) == digest for p, digest in identities.items())
    report = {'rubric_version': VERSION, 'evaluation_mode': 'REPLAY', 'actual_new_requests': 0,
              'actual_new_tokens': 0, 'live_passes_added': 0, 'input_sha256s': identities, 'cases': rows}
    (output_dir / 'replay.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    text = ['# 保存回答的双轨回放', '', '本报告只回放原回答；新增云端请求 0，实时通过数不变。', '']
    for row in rows:
        text.extend([f"## {row['case']}", '', row['question'], '', f"运行状态：{row['status']}", '',
                     '### 原始最终回答（未改写）', '', row.get('original_answer') or '未生成最终回答。', ''])
        if row.get('dual_scoring'):
            text.extend(['### 双轨判断', '', '```json', json.dumps(row['dual_scoring'], ensure_ascii=False, indent=2), '```', ''])
            current = row['current_guard_diagnostic']
            text.extend(['### 当前规则回放（不是新版实时回答）', '', current.get('answer') or current.get('reason', ''), '',
                         '完整草案、原审核与当前各规则诊断见 replay.json。', ''])
    (output_dir / 'questions-answers-diff.md').write_text('\n'.join(text), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--review-dir', type=Path)
    args = parser.parse_args()
    report = replay(args.root, args.results_dir, args.output_dir, review_dir=args.review_dir)
    print(json.dumps({'cases': len(report['cases']), 'actual_new_requests': 0,
                      'live_passes_added': 0, 'output_dir': str(args.output_dir)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
