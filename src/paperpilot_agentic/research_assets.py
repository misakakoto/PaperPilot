"""Owned research recipes and reviewable fusion patches; never execute model code.

The v2 training recipe remains immutable. A v3 episode references that frozen data
and adds a separate, candidate-specific authority. Static checks are locality and
capability checks, not a hard Python sandbox; execution still needs user approval.
"""
import ast
import difflib
import json
from pathlib import Path
import re
import textwrap
from types import SimpleNamespace

from .reference_methods import encoded, sha
from .training_profiles import load_training, environment, protocol
from .training_state import file_sha

VERSION = 'research-loop-v1'
GPU_SECONDS = 14400
_ID = re.compile(r'[a-f0-9]{64}')


def owned_folder(runtime, kind, identity):
    if not isinstance(identity, str) or not _ID.fullmatch(identity):
        raise ValueError('Expected a host-owned content ID')
    folder = runtime.workspace/'research-harness'/kind/identity
    if folder.is_symlink() or not folder.resolve().is_relative_to(runtime.workspace.resolve()):
        raise ValueError('Research asset escapes owned workspace')
    return folder


def immutable(folder, name, raw):
    path = folder/name
    if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
        raise ValueError('Unsafe research asset path')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Refusing to overwrite research artifact: '+name)
    else:
        with path.open('xb') as stream:
            stream.write(raw)
    return path


def register_recipe(runtime, training_binding, *, destination_workspace=None):
    """Host-only: derive a new recipe from a verified v2 source/data authority."""
    # Import old code/data authority without claiming its old environment is
    # still installed. V3 freezes the current environment and trains fresh pairs.
    root, base = load_training(runtime, training_binding, check_environment=False)
    execution_environment = environment(base['python'])
    design = runtime.db.get_run(base['design_run_id'])['metadata']['design_manifest']
    from .research_design import _sources
    sources = _sources(runtime, design['parent'], design['source_ids'], design['reference_ids'])
    prior = historical_observations(runtime, training_binding, root, base)
    recipe = {'version': VERSION, 'training_binding': training_binding,
        'base_manifest_sha256': sha(encoded(base)), 'base_manifest': base, 'sources': sources,
        'prior_observations': prior,
        'execution_environment': execution_environment,
        'environment_delta': {key: {'historical': base['environment'].get(key), 'current': value}
            for key, value in execution_environment.items() if base['environment'].get(key) != value},
        'baseline_sha256': base['files']['baseline.py'], 'gpu_cap_seconds': GPU_SECONDS,
        'limits': {'max_candidates': 2, 'max_patch_repairs': 2, 'max_turns': 24},
        'source_scope': 'Registered excerpts and pinned code; abstracts are not full papers',
        'validation_scope': 'Previously used development subset, NOT an independent holdout',
        'implementation': implementation_identity()}
    identity = sha(encoded(recipe))
    owner = runtime if destination_workspace is None else SimpleNamespace(workspace=Path(destination_workspace).resolve())
    folder = owned_folder(owner, 'recipes', identity)
    for name in base['files']:
        immutable(folder/'source', name, (root/name).read_bytes())
    immutable(folder, 'recipe.json', encoded(recipe))
    return identity


def implementation_identity():
    return {name: file_sha(Path(__file__).with_name(name)) for name in
            ('research_assets.py', 'research_harness.py', 'research_training.py',
             'paired_training.py', 'training_metrics.py', 'training_profiles.py', 'durable_reader.py',
             'episode_runner.py', 'research_contracts.py', 'research_identity.py', 'research_review.py')}


def historical_observations(runtime, binding, root, manifest):
    """Import only reverified old predictions; never treat an old answer as evidence."""
    from .paired_training import _verify_reference
    from .training_metrics import verify_trial, paired_summary
    with runtime.db.connect() as conn:
        runs = conn.execute("SELECT run_id FROM runs WHERE status='COMPLETED' AND "
            "json_extract(metadata_json,'$.engine')='method_training' AND "
            "json_extract(metadata_json,'$.binding_id')=?", (binding,)).fetchall()
    observations = []
    for row in runs:
        run_id = row['run_id']
        folder = runtime.workspace/'method-training'/'runs'/run_id
        state_path = folder/'state.json'
        if not state_path.exists():
            continue
        trials = json.loads(state_path.read_bytes())['trials']
        for seed in (7, 11, 19):
            for variant in ('baseline', 'candidate'):
                trial = trials.get(f'{seed}-{variant}')
                if not trial or trial.get('status') != 'VERIFIED':
                    raise ValueError('Historical completed campaign has a missing seed')
                raw = _verify_reference(trial['receipt'], folder)
                metrics = verify_trial(raw, root, manifest, seed=seed, variant=variant)
                if metrics != trial['metrics']:
                    raise ValueError('Historical metrics changed')
        records = []
        for pair in paired_summary(trials):
            records.append({'seed': pair['seed'], 'baseline': pair['baseline']['metrics'],
                'candidate': pair['candidate']['metrics'], 'delta': pair['delta']})
        observations.append({'run_id': run_id, 'binding_id': binding,
            'candidate': 'Previously reviewed attention bypass', 'records': records,
            'origin': 'HISTORICAL_REVERIFIED; not a new model/GPU execution',
            'state_sha256': file_sha(state_path)})
    return observations


def load_recipe(runtime, identity, *, check_environment=True):
    folder = owned_folder(runtime, 'recipes', identity)
    recipe = json.loads((folder/'recipe.json').read_bytes())
    if sha(encoded(recipe)) != identity or recipe['version'] != VERSION:
        raise ValueError('Research recipe identity changed')
    if recipe['implementation'] != implementation_identity():
        raise ValueError('Research implementation changed; freeze a new recipe/episode')
    root, base = folder/'source', recipe['base_manifest']
    if (sha(encoded(base)) != recipe['base_manifest_sha256'] or base['protocol'] != protocol()
            or file_sha(base['python']) != base['python_sha256']):
        raise ValueError('Research baseline/data authority changed')
    for name, digest in base['files'].items():
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or file_sha(path) != digest:
            raise ValueError('Research source/data snapshot changed: '+name)
    if check_environment and environment(base['python']) != recipe['execution_environment']:
        raise ValueError('Research training environment changed')
    return recipe, root, base


def fusion_region(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SNUNet_ECAM')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    first = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'intra')
    last = next(i for i, n in enumerate(method.body[first:], first)
                if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'conv_final')
    return method.body[first:last]


def build_fusion_patch(source, fusion_body):
    """Splice expressions into the attention/fusion tail, preserving all other bytes."""
    if not isinstance(fusion_body, str) or not 0 < len(fusion_body.encode()) <= 6000:
        raise ValueError('A bounded fusion_body is required')
    body = textwrap.dedent(fusion_body).strip()+'\n'
    try:
        proposed = ast.parse(body)
    except SyntaxError as exc:
        raise ValueError(f'Fusion syntax error at line {exc.lineno}, column {exc.offset}: {exc.msg}') from exc
    parents = {id(child): node for node in ast.walk(proposed) for child in ast.iter_child_nodes(node)}
    if not 1 <= len(proposed.body) <= 16:
        raise ValueError('Fusion requires 1-16 assignments')
    known = {'out', 'x0_1', 'x0_2', 'x0_3', 'x0_4'}
    allowed_torch = {'sum', 'mean', 'stack', 'cat', 'sigmoid', 'softmax', 'clamp'}
    allowed_tensor = {'repeat', 'mean', 'sum', 'clamp', 'softmax'}
    nodes = (ast.Module, ast.Assign, ast.Name, ast.Load, ast.Store, ast.Constant,
             ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub,
             ast.Call, ast.Attribute, ast.keyword, ast.List, ast.Tuple)
    called = set()
    for statement in proposed.body:
        if (not isinstance(statement, ast.Assign) or len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)):
            raise ValueError('Only local assignments are permitted in fusion_body')
        target = statement.targets[0].id
        if target in {'self', 'torch', 'x0_1', 'x0_2', 'x0_3', 'x0_4'} or target.startswith('_'):
            raise ValueError('Fusion cannot overwrite an input or privileged name')
        for node in ast.walk(statement.value):
            if not isinstance(node, nodes):
                raise ValueError('Unsupported fusion syntax: '+type(node).__name__)
            if isinstance(node, ast.Name) and node.id not in known | {'self', 'torch'}:
                raise ValueError('Undefined fusion input: '+node.id)
            if isinstance(node, ast.Constant) and (type(node.value) not in (int, float)
                    or not -16 <= node.value <= 16):
                keyword = parents.get(id(node))
                call = parents.get(id(keyword))
                keepdim = (type(node.value) is bool and isinstance(keyword, ast.keyword)
                           and keyword.arg == 'keepdim' and isinstance(call, ast.Call)
                           and isinstance(call.func, ast.Attribute) and call.func.attr in {'mean', 'sum'})
                if not keepdim:
                    raise ValueError(f'Invalid fusion constant {node.value!r} at line {node.lineno}; '
                                     'numbers must be in [-16,16]; booleans only for mean/sum keepdim')
            if isinstance(node, ast.Attribute):
                if not isinstance(node.value, ast.Name):
                    raise ValueError('Nested attribute access is not permitted')
                owner = node.value.id
                valid = (node.attr in {'ca', 'ca1'} if owner == 'self' else
                         node.attr in allowed_torch if owner == 'torch' else
                         owner in known and node.attr in allowed_tensor)
                if not valid:
                    raise ValueError('Unsupported fusion attribute: '+node.attr)
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Attribute) or any(k.arg is None for k in node.keywords):
                    raise ValueError('Only declared tensor/module operations can be called')
                if isinstance(node.func.value, ast.Name) and node.func.value.id == 'self':
                    called.add(node.func.attr)
        known.add(target)
    if proposed.body[-1].targets[0].id != 'out':
        raise ValueError('Fusion must finish by assigning out; classifier and return stay host-owned')
    region = fusion_region(source)
    if ast.dump(ast.Module(body=region, type_ignores=[])) == ast.dump(proposed):
        raise ValueError('Fusion patch is a no-op')
    lines = source.splitlines(keepends=True)
    candidate = ''.join(lines[:region[0].lineno-1])+textwrap.indent(body, ' '*region[0].col_offset)+''.join(lines[region[-1].end_lineno:])
    compile(candidate, '<review-required-fusion>', 'exec')
    patch = ''.join(difflib.unified_diff(source.splitlines(True), candidate.splitlines(True),
                                       fromfile='a/baseline.py', tofile='b/candidate.py'))
    unused = sorted(f'{name}.{part}.weight' for name in ('ca', 'ca1') if name not in called
                    for part in ('fc1', 'fc2'))
    return {'fusion_body': body, 'candidate_source': candidate, 'diff': patch,
        'candidate_sha256': sha(candidate.encode()), 'expected_unused': unused,
        'static_checks': ['fusion-locality', 'unchanged-init-and-feature-extractor',
                          'unchanged-classifier-and-return', 'syntax', 'declared-operations'],
        'not_checked': ['runtime shapes', 'finite gradients', 'learning', 'scientific validity'],
        'network_isolated': False, 'filesystem_isolated': False}
