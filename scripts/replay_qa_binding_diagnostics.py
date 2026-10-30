"""Compare gates on identical saved QA inputs; never rewrite answers or call a model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from paperpilot_agentic import experimental_scope, numeric_scope  # noqa: E402


def historical_module(commit, name):
    # Only the two fixed, trusted repository modules are executable. Provider
    # responses remain data. Resolve the ref to an immutable commit first.
    raw = subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
        'show', f'{commit}:src/paperpilot_agentic/{name}.py'], cwd=ROOT)
    module = types.ModuleType(f'paperpilot_agentic._historical_{name}')
    module.__package__ = 'paperpilot_agentic'
    exec(compile(raw.decode('utf-8'), f'{commit}/{name}.py', 'exec'), module.__dict__)
    return module


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def provider_content(path):
    return json.loads(read(path)['choices'][0]['message']['content'])


def evaluate(draft, audit, evidence, roles, numeric, experimental):
    effective, diagnostics = numeric.apply_scope_guard(draft, audit, evidence)
    effective, scope = experimental.apply_experimental_scope(draft, effective, roles)
    return {'numeric_version': numeric.VERSION, 'numeric': diagnostics,
            'experimental': scope, 'assessments': effective['assessments'],
            'coverage': effective['coverage']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('request', 'draft-response', 'audit-response', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--baseline-ref', required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
        'rev-parse', '--verify', args.baseline_ref+'^{commit}'], cwd=ROOT, text=True).strip()
    protected = [args.request, args.draft_response, args.audit_response]
    before = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    request = read(args.request)
    payload = next(value for message in request['messages'] if message['role'] == 'user'
                   for value in [json.loads(message['content'])] if 'evidence' in value)
    draft, audit = provider_content(args.draft_response), provider_content(args.audit_response)
    roles = {r['evidence_id']: r for r in payload.get('source_roles', [])}
    result = {'kind': 'OFFLINE_IDENTICAL_INPUT_GATE_REPLAY', 'baseline_commit': commit,
        'current_module_sha256': {name: hashlib.sha256((ROOT/'src/paperpilot_agentic'/
            (name+'.py')).read_bytes()).hexdigest() for name in ('numeric_scope', 'experimental_scope')},
        'protected_inputs': before, 'additional_provider_attempts': 0, 'additional_tokens': 0,
        'raw_draft': draft, 'new_live_demo_passes': 0,
        'before': evaluate(draft, audit, payload['evidence'], roles,
                           historical_module(commit, 'numeric_scope'), historical_module(commit, 'experimental_scope')),
        'after': evaluate(draft, audit, payload['evidence'], roles, numeric_scope, experimental_scope)}
    assert before == {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'output': str(args.output), 'provider_attempts_added': 0,
                      'before': result['before']['experimental'],
                      'after': result['after']['experimental']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
