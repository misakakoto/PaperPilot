"""Bind a local model-function draft to a verified reference, without executing it.

Syntax/locality checks are NOT a security sandbox or scientific review. This module
has no import/exec/subprocess path for the proposed Python.
"""
import ast
import difflib
import json
from pathlib import Path
import textwrap

from .reference_methods import encoded, load_profile, sha


PATCH_SCHEMA = {
    'target': 'SNUNet_ECAM.forward',
    'replacement': 'complete def forward(self, xA, xB): with body; no surrounding class',
    'review_notes': ['semantic, shape, compute and safety checks still required'],
}


def verified_baseline(runtime, validation_run_id, *, conversation_id, references):
    """Resolve an owned completed receipt and exact source; reject stale/tampered assets."""
    if not isinstance(validation_run_id, str) or not validation_run_id:
        raise ValueError('A completed validation run ID is required')
    run = runtime.db.get_run(validation_run_id)
    metadata = run['metadata']
    if (run['status'] != 'COMPLETED' or metadata.get('engine') != 'method_validation'
            or not conversation_id or metadata.get('conversation_id') != conversation_id):
        raise ValueError('Baseline validation must be completed in the same conversation')
    calls = runtime.db.list_tool_calls(validation_run_id)
    if (len(calls) != 1 or calls[0]['tool_name'] != 'method.validate'
            or calls[0]['status'] != 'COMPLETED' or not isinstance(calls[0].get('result'), dict)
            or calls[0]['result'].get('ok') is not True):
        raise ValueError('Baseline has no successful validation tool receipt')
    call = calls[0]
    binding = metadata['binding_id']
    if call['args'] != {'binding_id': binding}:
        raise ValueError('Validation arguments do not match the baseline binding')
    approvals = runtime.db.list_approvals(validation_run_id)
    if not any(a['tool_call_id'] == call['tool_call_id'] and a['status'] == 'APPROVED' for a in approvals):
        raise ValueError('Baseline validation has no recorded approval')
    artifacts = [a for a in runtime.db.list_artifacts(validation_run_id) if a['kind'] == 'method-verification']
    if len(artifacts) != 1:
        raise ValueError('Baseline verification artifact is missing or ambiguous')
    artifact = artifacts[0]
    expected = runtime.workspace / 'method-runs' / call['tool_call_id'] / 'verification.json'
    path = Path(artifact['path'])
    if (path.is_symlink() or path.resolve() != expected.resolve()
            or not path.resolve().is_relative_to(runtime.workspace.resolve())):
        raise ValueError('Baseline receipt path is not owned')
    raw = path.read_bytes()
    receipt = json.loads(raw)
    if (sha(raw) != artifact['sha256'] or receipt != metadata.get('method_validation')
            or receipt != call['result'].get('output') or receipt.get('binding_id') != binding
            or receipt.get('status') != 'ADAPTER_SMOKE_VERIFIED'
            or receipt.get('paper_accuracy_reproduced') is not False
            or receipt.get('scientific_hypothesis_supported') is not False):
        raise ValueError('Baseline verification receipt changed or is not verified')
    root, profile = load_profile(runtime.workspace, binding)
    matching = [r for r in references if r['source_uri'] == profile['source_url']
                and sha(r['content'].encode()) == profile['source_sha256']
                and r['provenance'].get('license_url') == profile['license_url']
                and sha(r['provenance'].get('license_text', '').encode()) == profile['files']['LICENSE']]
    if len(matching) != 1:
        raise ValueError('Select the exact validated source and license as one code reference')
    return {
        'validation_run_id': validation_run_id, 'tool_call_id': call['tool_call_id'],
        'binding_id': binding, 'receipt_sha256': sha(raw), 'source_sha256': profile['source_sha256'],
        'reference_id': matching[0]['evidence_id'], 'method': profile['method'],
        'scope': profile['scope'], 'target': PATCH_SCHEMA['target'],
        'frozen_training_request': {k: profile['request'][k] for k in ('seed', 'steps', 'device')},
        'unchanged_by_draft': ['dataset/split/sample selection', 'optimizer/training loop',
                               'metric implementation', 'execution environment/command'],
        'execution_authorized': False, 'scientific_hypothesis_supported': False,
        'source': (root/'upstream.py').read_text(encoding='utf-8'),
    }


def build_patch(value, baseline):
    """Compile but never run: splice exactly one matching method, emit a reviewable diff."""
    if not isinstance(value, dict) or set(value) != set(PATCH_SCHEMA):
        raise ValueError('Expected target, replacement and review_notes for the local draft')
    replacement, notes = value['replacement'], value['review_notes']
    if (value['target'] != PATCH_SCHEMA['target'] or not isinstance(replacement, str)
            or not replacement.strip() or len(replacement.encode()) > 16000
            or not isinstance(notes, list) or not notes
            or any(not isinstance(n, str) or not n.strip() for n in notes)):
        raise ValueError('Invalid bounded method draft')
    parsed = ast.parse(textwrap.dedent(replacement))
    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.FunctionDef):
        raise ValueError('Draft must contain exactly one synchronous method, no top-level code')
    function = parsed.body[0]
    original = baseline['source']
    tree = ast.parse(original)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SNUNet_ECAM']
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == 'forward'] if len(classes) == 1 else []
    if len(methods) != 1:
        raise ValueError('Baseline target is not unique')
    target = methods[0]
    if (function.name != target.name or function.decorator_list or function.returns
            or ast.dump(function.args) != ast.dump(target.args)
            or getattr(function, 'type_params', []) or target.decorator_list):
        raise ValueError('Draft must preserve the original method signature')
    # Equal AST is a no-op even if the model changed comments/whitespace.
    if ast.dump(function) == ast.dump(target):
        raise ValueError('Draft is a no-op')
    lines = original.splitlines(keepends=True)
    body = textwrap.indent(textwrap.dedent(replacement).strip() + '\n', ' ' * target.col_offset)
    candidate = ''.join(lines[:target.lineno-1]) + body + ''.join(lines[target.end_lineno:])
    compile(candidate, '<unexecuted-method-draft>', 'exec')  # Compilation has no import/exec effect.
    patch = ''.join(difflib.unified_diff(original.splitlines(keepends=True), candidate.splitlines(keepends=True),
                                       fromfile='a/upstream.py', tofile='b/upstream.py'))
    def self_calls(node):
        return {n.func.attr for n in ast.walk(node) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                and n.func.value.id == 'self'}
    semantic_diff = ''.join(difflib.unified_diff(
        (ast.unparse(target)+'\n').splitlines(keepends=True),
        (ast.unparse(function)+'\n').splitlines(keepends=True),
        fromfile='original-method-AST', tofile='candidate-method-AST'))
    contract = {k: v for k, v in baseline.items() if k != 'source'} | {
        'candidate_sha256': sha(candidate.encode()), 'patch_sha256': sha(patch.encode()),
        'target_original_lines': [target.lineno, target.end_lineno],
        'replacement': textwrap.dedent(replacement).strip() + '\n',
        'logic_diff': semantic_diff,
        'removed_direct_self_calls': sorted(self_calls(target) - self_calls(function)),
        'added_direct_self_calls': sorted(self_calls(function) - self_calls(target)),
        'logic_diff_scope': 'AST display ignores comments/formatting, not a dataflow or safety proof; never applied',
        'review_notes': notes, 'executed': False,
        'assurance': 'One-method locality, signature and syntax only; NOT safety, behavior or correctness',
        'required_gates': ['PLAN_PATCH_ALIGNMENT', 'CODE_AND_LICENSE_REVIEW',
                           'TENSOR_AND_GRADIENT_CHECKS', 'PAIRED_PROTOCOL_REVIEW', 'EXECUTION_APPROVAL'],
    }
    return {'files': [{'path': 'candidate.patch', 'content': patch},
                      {'path': 'candidate.json', 'content': encoded(contract).decode()}],
            'review_notes': notes, 'contract': contract, 'candidate_source': candidate}


def load_draft(runtime, run_id):
    """Read and revalidate original source and draft artifacts; never execute the candidate."""
    run = runtime.db.get_run(run_id)
    draft = run['metadata'].get('research_design', {})
    if (run['status'] != 'COMPLETED' or draft.get('draft_mode') != 'validated_method_patch'
            or draft.get('executed') is not False):
        raise ValueError('Review requires a completed, unexecuted bound draft')
    # Reuse the design source ownership checks; code generation is never called.
    from .research_design import _sources, _digest
    manifest = run['metadata']['design_manifest']
    sources = _sources(runtime, manifest['parent'], manifest['source_ids'], manifest['reference_ids'])
    if _digest(sources) != manifest['source_hash']:
        raise ValueError('Review sources changed')
    baseline = verified_baseline(runtime, manifest['validation_run_id'],
        conversation_id=run['metadata']['conversation_id'], references=sources['references'])
    if _digest(baseline) != manifest['baseline_hash']:
        raise ValueError('Review baseline changed')
    folder = runtime.workspace/'research-drafts'/run_id
    digests = {}
    artifacts = draft.get('artifacts', [])
    if len(artifacts) != 2:
        raise ValueError('Review needs both original draft artifacts')
    for artifact in artifacts:
        path = Path(artifact['path'])
        if (path.name not in {'candidate.patch', 'candidate.json'} or path.name in digests
                or path.is_symlink() or path.resolve() != (folder/path.name).resolve()
                or not path.resolve().is_relative_to(runtime.workspace.resolve())
                or sha(path.read_bytes()) != artifact['sha256']):
            raise ValueError('Review artifact changed or is not owned')
        digests[path.name] = artifact['sha256']
    contract = json.loads((folder/'candidate.json').read_text(encoding='utf-8'))
    if (contract != draft['patch_contract'] or contract['patch_sha256'] != digests['candidate.patch']
            or contract['binding_id'] != baseline['binding_id']):
        raise ValueError('Review contract does not match original artifacts')
    return {'baseline': baseline, 'contract': contract, 'draft_artifacts': digests}


def record_draft_review(runtime, run_id, *, reviewer, decision, findings):
    """Host-only review record, hash-bound to the unchanged draft. Grants NO execution."""
    if decision not in {'REVISION_REQUIRED', 'READY_FOR_EXECUTION_REVIEW'}:
        raise ValueError('Review decision must request revision or a separate execution review')
    if (not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 200
            or not isinstance(findings, list) or not 1 <= len(findings) <= 20
            or any(not isinstance(f, str) or not f.strip() or len(f) > 4000 for f in findings)):
        raise ValueError('A bounded reviewer identity and explicit findings are required')
    material = load_draft(runtime, run_id)
    baseline, contract, digests = material['baseline'], material['contract'], material['draft_artifacts']
    folder = runtime.workspace/'research-drafts'/run_id
    review = {'design_run_id': run_id, 'reviewer': reviewer, 'decision': decision, 'findings': findings,
        'draft_artifacts': digests, 'candidate_sha256': contract['candidate_sha256'],
        'binding_id': baseline['binding_id'], 'execution_authorized': False,
        'scientific_hypothesis_supported': False}
    identifier = sha(encoded(review))
    path = folder/'reviews'/(identifier+'.json')
    if (path.is_symlink() or not path.resolve().is_relative_to(folder.resolve())
            or not path.resolve().is_relative_to(runtime.workspace.resolve())):
        raise ValueError('Review path escapes draft directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = encoded(review)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Existing review changed; refusing overwrite')
    else:
        with path.open('xb') as handle:
            handle.write(raw)
    artifact = {'artifact_id': 'artifact_draft_review_' + identifier[:24],
                'kind': 'method-draft-review', 'path': str(path), 'sha256': sha(raw)}
    runtime.db.add_artifact(run_id, None, artifact)
    runtime.db.update_run_metadata(run_id, {'research_design_review': {**review, 'artifact': artifact}})
    return review
