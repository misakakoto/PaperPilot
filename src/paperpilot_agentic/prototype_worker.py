"""Host-maintained CPU checks, launched ONLY in the approved Docker protocol.

This checks engineering properties, not model novelty, scientific fidelity or
training performance. Candidate code and resulting metrics are not attestations
against malicious code running in the same Python process.
"""
import importlib.util
import json
import math
try:
    import resource
except ImportError:  # Host-maintained Windows fixture tests; execution still requires Docker.
    resource = None
import time
import traceback
from pathlib import Path


def make_validation_contract(kind, constraints=None):
    if kind not in {'unbound', 'snunet_ecam'}:
        raise ValueError('Unknown host validation contract')
    if constraints is None:
        constraints = {}
    allowed = {'input_cases', 'max_parameters', 'max_peak_rss_kib', 'max_verifier_seconds', 'spatial_gate'}
    if not isinstance(constraints, dict) or set(constraints)-allowed:
        raise ValueError('Unknown prototype constraint fields')
    defaults = [[1,32,32],[2,32,48],[1,48,64]] if kind == 'snunet_ecam' else [[1,64,64],[2,65,97],[1,96,128]]
    cases = constraints.get('input_cases', defaults)
    if not isinstance(cases, list) or not 1 <= len(cases) <= 4:
        raise ValueError('Supply 1..4 input cases [batch,height,width]')
    normalized, seen = [], set()
    for case in cases:
        if (not isinstance(case, list) or len(case) != 3 or any(type(n) is not int for n in case)
                or not 1 <= case[0] <= 4 or any(not 16 <= n <= 256 for n in case[1:])):
            raise ValueError('Inputs require batch 1..4, height/width 16..256')
        if kind == 'snunet_ecam' and any(n < 32 or n % 16 for n in case[1:]):
            raise ValueError('SNUNet dimensions require multiples of 16, at least 32')
        if tuple(case) in seen:
            raise ValueError('Duplicate input case')
        seen.add(tuple(case))
        normalized.append(list(case))
    result = {'kind': kind, 'input_cases': normalized}
    if 'spatial_gate' in constraints:
        gate = constraints['spatial_gate']
        if (not isinstance(gate, dict) or set(gate) != {'feature_channels', 'kernel_size'}
                or type(gate['feature_channels']) is not int or not 1 <= gate['feature_channels'] <= 1024
                or type(gate['kernel_size']) is not int or gate['kernel_size'] not in {3, 5, 7}):
            raise ValueError('Invalid spatial gate requirement')
        result['spatial_gate'] = dict(gate)
    for field, upper in [('max_parameters', 1000000000), ('max_peak_rss_kib', 4*1024*1024),
                         ('max_verifier_seconds', 180)]:
        if field not in constraints:
            continue
        value = constraints[field]
        valid_type = type(value) in (int, float) if field == 'max_verifier_seconds' else type(value) is int
        if not valid_type or not 0 < value <= upper or not math.isfinite(value):
            raise ValueError(f'{field} must be positive and at most {upper}')
        result[field] = value
    return result


def evaluate_constraints(contract, metrics):
    violations = []
    for field, metric in [('max_parameters','parameters'), ('max_peak_rss_kib','peak_rss_kib'),
                          ('max_verifier_seconds','seconds')]:
        if field not in contract:
            continue
        value = metrics.get(metric)
        measured = (type(value) is int or type(value) is float and math.isfinite(value)) and value >= 0
        if not measured or value > contract[field]:
            violations.append({'kind': 'LIMIT_EXCEEDED' if measured else 'MEASUREMENT_UNAVAILABLE',
                               'metric': metric, 'actual': value, 'limit': contract[field]})
    return violations


def read_contract():
    value = json.loads(Path('contract.json').read_text()) if Path('contract.json').exists() else {'kind':'unbound'}
    if not isinstance(value, dict) or 'kind' not in value:
        raise ValueError('Unknown host validation contract')
    return make_validation_contract(value['kind'], {k:v for k,v in value.items() if k != 'kind'})


class SpatialGateCheck:
    """Observe declared convs, call order and loss gradients, not semantic fidelity."""

    def __init__(self, model, requirement):
        import torch
        self.events, self.handles, self.layers, self.dependencies = [], [], {}, set()
        channels, kernel = requirement['feature_channels'], requirement['kernel_size']
        for name, layer in model.named_modules():
            if not isinstance(layer, torch.nn.Conv2d) or layer.in_channels != channels:
                continue
            if layer.groups != 1 or layer.stride != (1, 1) or layer.dilation != (1, 1):
                continue
            kind = ('gate' if layer.out_channels == 1 and layer.kernel_size == (kernel, kernel)
                    and layer.padding == (kernel//2, kernel//2) else
                    'classifier' if layer.out_channels == 2 and layer.kernel_size == (1, 1)
                    and layer.padding == (0, 0) else None)
            if kind:
                self.layers[name] = (kind, layer)
        kinds = {kind for kind, _ in self.layers.values()}
        if kinds != {'gate', 'classifier'}:
            raise ValueError('Missing required spatial gate or two-channel classifier convolutions')
        for name, (kind, layer) in self.layers.items():
            self.handles.append(layer.register_forward_hook(self._hook(kind, name)))

    def _hook(self, kind, name):
        def record(module, inputs, output):
            self.events.append((kind, name, inputs[0], output))
        return record

    def observe_dependencies(self):
        import torch
        self.dependencies.clear()
        generator = torch.Generator(device='cpu').manual_seed(1729)
        for index, (kind, _, _, output) in enumerate(self.events):
            if kind != 'gate' or not output.requires_grad or output.ndim != 4:
                continue
            for later_index in range(index+1, len(self.events)):
                later_kind, _, inputs, _ = self.events[later_index]
                if (later_kind != 'classifier' or not inputs.requires_grad or inputs.ndim != 4
                        or output.shape[0] != inputs.shape[0] or output.shape[2:] != inputs.shape[2:]):
                    continue
                projection = torch.randn(inputs.shape, generator=generator, dtype=inputs.dtype, device='cpu')
                influence, = torch.autograd.grad(inputs, output, grad_outputs=projection,
                                                 allow_unused=True, retain_graph=True)
                if (influence is not None and bool(torch.isfinite(influence).all())
                        and bool(influence.abs().sum() > 0)):
                    self.dependencies.add((index, later_index))

    def check(self):
        import torch

        def gradient(name):
            value = self.layers[name][1].weight.grad
            return value is not None and bool(torch.isfinite(value).all()) and bool(value.abs().sum() > 0)

        for index, event in enumerate(self.events):
            kind, name, _, _ = event
            if kind != 'gate' or not gradient(name):
                continue
            for later_index in range(index+1, len(self.events)):
                later_kind, later_name, _, _ = self.events[later_index]
                if (later_kind == 'classifier' and (index, later_index) in self.dependencies
                        and gradient(later_name)):
                    return {'gate': name, 'classifier': later_name,
                            'call_order': 'gate_before_classifier', 'weight_gradient': 'finite_nonzero',
                            'classifier_input_dependency': 'observed_nonzero_projection'}
        raise ValueError('Required spatial gate must execute before classifier with finite nonzero weight gradients')

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.events.clear()


def verify(contract=None, *, progress=None):
    contract = read_contract() if contract is None else contract
    progress = {} if progress is None else progress
    progress.update(phase='load', completed_checks=[])
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    spec = importlib.util.spec_from_file_location('candidate_model', 'model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads(Path('config.json').read_text())
    snunet = contract['kind'] == 'snunet_ecam'
    model = module.SNUNet_ECAM(in_ch=3, out_ch=2) if snunet else module.build_model(config)
    if not isinstance(model, torch.nn.Module):
        raise ValueError('build_model must return torch.nn.Module')
    model = model.cpu().eval()
    observer = SpatialGateCheck(model, contract['spatial_gate']) if 'spatial_gate' in contract else None
    try:
        rows = progress['completed_checks']
        shapes = contract['input_cases']
        for case_index, (batch, height, width) in enumerate(shapes):
            progress.update(case_index=case_index, input_shape=[batch, 3, height, width], phase='inputs')
            x = torch.randn(batch, 3, height, width, requires_grad=True)
            y = torch.randn_like(x, requires_grad=True)
            model.zero_grad(set_to_none=True)
            if observer:
                observer.events.clear()
            progress['phase'] = 'forward'
            out = model(x, y)
            progress['phase'] = 'output'
            if snunet:
                if not isinstance(out, tuple) or len(out) != 1:
                    raise ValueError('SNUNet must retain its one-element tuple output')
                out = out[0]
            channels = 2 if snunet else 1
            if not isinstance(out, torch.Tensor) or out.shape != (batch, channels, height, width):
                raise ValueError(f'Expected Bx{channels}xHxW logits at input resolution')
            if not torch.isfinite(out).all():
                raise ValueError('Nonfinite output')
            progress['phase'] = 'loss'
            loss = (torch.nn.functional.cross_entropy(out, torch.randint(2, (batch, height, width))) if snunet else
                    torch.nn.functional.binary_cross_entropy_with_logits(out, torch.rand_like(out)))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            if observer:
                progress['phase'] = 'structure_dependency'
                observer.observe_dependencies()
            progress['phase'] = 'backward'
            loss.backward()
            progress['phase'] = 'gradients'
            for value in (x.grad, y.grad):
                if value is None or not torch.isfinite(value).all() or value.abs().sum() == 0:
                    raise ValueError('Both inputs must have finite nonzero gradients')
            grads = [p.grad for p in model.parameters() if p.requires_grad]
            if not grads or not any(g is not None and g.abs().sum() > 0 for g in grads):
                raise ValueError('No trainable parameter receives gradient')
            if any(g is not None and not torch.isfinite(g).all() for g in grads):
                raise ValueError('Nonfinite parameter gradient')
            row = {'shape': list(out.shape), 'loss': float(loss.detach())}
            if observer:
                progress['phase'] = 'structure'
                row['spatial_gate'] = observer.check()
            rows.append(row)
        progress['phase'] = 'parameters'
        return {'checks': rows, 'parameters': sum(p.numel() for p in model.parameters())}
    finally:
        if observer:
            observer.close()


def run_validation():
    started = time.monotonic()
    contract = None
    progress = {'phase': 'contract', 'completed_checks': []}
    try:
        contract = read_contract()  # Freeze requirements before importing candidate code.
        result = {'passed': True, **verify(contract, progress=progress)}
    except Exception as exc:
        result = {'passed': False, 'error': str(exc), 'traceback': traceback.format_exc()[-6000:],
                  'failure_kind': 'ENVIRONMENT' if isinstance(exc, ModuleNotFoundError) else
                  'RESOURCE_LIMIT' if isinstance(exc, MemoryError) else 'IMPLEMENTATION',
                  'checks': progress['completed_checks'],
                  'failure_context': {k: v for k, v in progress.items() if k != 'completed_checks'}}
    result.update(seconds=time.monotonic()-started,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None,
                  scientific_validity='NOT_VERIFIED', validation_contract=contract,
                  device='cpu', seed=1729, training_steps=0)
    if result['passed']:
        violations = evaluate_constraints(contract, result)
        result.update(constraint_violations=violations, constraints_satisfied=not violations)
        if violations:
            result['passed'] = False
            result['failure_kind'] = ('ENVIRONMENT' if any(v['kind']=='MEASUREMENT_UNAVAILABLE' for v in violations)
                                      else 'DESIGN_CONSTRAINT')
    return result


if __name__ == '__main__':
    Path('metrics.json').write_text(json.dumps(run_validation(), allow_nan=False), encoding='utf-8')
