"""One source-bound audit of a saved draft; no Writer, retrieval or automatic retry.

Run in separate processes against pinned --code-root directories for an ablation.
This known-failure diagnostic is not a held-out accuracy evaluation.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile

from check_session_entry_live import RecordedProvider, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--model', default='qwen3-30b-a3b-instruct-2507')
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--execute-paid', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_root.resolve() / 'src'))
    from paperpilot_agentic.answer_audit import answer_then_audit, evidence_spans
    from paperpilot_agentic.config import RuntimeSettings, read_dashscope_api_key_file
    from paperpilot_agentic.providers import OpenAICompatibleLLM

    snapshot = json.loads(args.snapshot.read_text(encoding='utf-8-sig'))
    first = snapshot['run']['metadata']['reader_rounds'][0]
    originals = {e['evidence_id']: e for e in snapshot['evidence']}
    selected = list(dict.fromkeys(s['evidence_id'] for s in first['source_spans']))
    rows = [originals[eid] for eid in selected]
    if evidence_spans(rows) != first['source_spans']:
        raise ValueError('Saved originals no longer reproduce the frozen source span map')
    question = first['question']
    draft = first['draft']
    identity = hashlib.sha256()
    for source in sorted((args.code_root / 'src').rglob('*.py')):
        identity.update(source.relative_to(args.code_root).as_posix().encode())
        identity.update(source.read_bytes())
    protocol = {'model': args.model, 'code_sha256': identity.hexdigest(),
        'input_sha256': hashlib.sha256(json.dumps({'question': question, 'draft': draft,
            'evidence': rows}, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        'max_actual_requests': 1, 'independent_benchmark': False, 'allow_repair': False,
        'question_policy': 'known-failure frozen-draft audit', 'source_span_map_verified': True}
    if not args.execute_paid:
        print(json.dumps({**protocol, 'actual_requests': 0}))
        return
    if not args.key_file:
        parser.error('Provide an external key file; never a key in command arguments')
    root = Path(__file__).resolve().parents[1] / 'outputs'
    folder = Path(tempfile.mkdtemp(prefix='frozen-answer-audit-', dir=root))
    save(folder / 'protocol.json', protocol)
    save(folder / 'input.json', {'question': question, 'draft': draft, 'evidence': rows})
    settings = replace(RuntimeSettings.from_env(), provider='qwen', model=args.model,
        thinking_enabled=False, timeout_seconds=90,
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        api_key=read_dashscope_api_key_file(args.key_file))

    class OneRequest(RecordedProvider):
        def complete(self, *a, **kw):
            if self.calls:
                raise RuntimeError('Frozen audit permits one actual request, including failures')
            return super().complete(*a, **kw)

    model = OneRequest(OpenAICompatibleLLM(settings, max_retries=0), folder)
    report = answer_then_audit(question, rows, model, frozen_draft=draft, allow_repair=False,
                               relation_policy=True, question_scope_policy=True)
    save(folder / 'report.json', report)
    print(json.dumps({'output': str(folder), 'status': report['status'],
        'actual_requests': model.calls, 'answer_scope': report.get('answer_scope'),
        'consistency': report.get('answer_consistency')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
