"""Maintained tiny torch fixtures only; never imports generated candidates."""
import json
import os
from pathlib import Path
import subprocess
import sys
from functools import lru_cache

import pytest

from paperpilot_agentic.prototype_worker import make_validation_contract

PYTHON = Path(os.environ.get('PAPERPILOT_TEST_TORCH_PYTHON', sys.executable))
SOURCE = Path(__file__).resolve().parents[1]/'src'
MODEL = '''import torch
from torch import nn
class Model(nn.Module):
 def __init__(self):
  super().__init__()
  self.features=nn.Conv2d(6,8,1)
  self.head=nn.Conv2d(8,2,1)
  self.gate=nn.Conv2d(GATE_CHANNELS,1,KERNEL,padding=KERNEL//2)
 def forward(self,x,y):
  f=self.features(torch.cat([x,y],1))
  EXPRESSION
def build_model(config): return Model()
'''


@lru_cache(maxsize=1)
def torch_available():
    if not PYTHON.exists():
        return False
    probe = subprocess.run([str(PYTHON), '-c', "import importlib.util; print(int(importlib.util.find_spec('torch') is not None))"],
                           capture_output=True, text=True, timeout=30)
    return probe.returncode == 0 and probe.stdout.strip() == '1'


def run_fixture(tmp_path, expression, *, channels=8, kernel=3, required=True):
    if not torch_available():
        pytest.skip('Set PAPERPILOT_TEST_TORCH_PYTHON to an existing torch interpreter')
    code = MODEL.replace('GATE_CHANNELS', str(channels)).replace('KERNEL', str(kernel)).replace('EXPRESSION', expression)
    (tmp_path/'model.py').write_text(code, encoding='utf-8')
    (tmp_path/'config.json').write_text('{}', encoding='utf-8')
    contract = {'kind': 'unbound', 'input_cases': [[1, 16, 24]]}
    if required:
        contract['spatial_gate'] = {'feature_channels': 8, 'kernel_size': 3}
    (tmp_path/'contract.json').write_text(json.dumps(contract), encoding='utf-8')
    script = '''import json
from paperpilot_agentic.prototype_worker import verify
try:
 print(json.dumps({'ok':True,'result':verify()}))
except Exception as exc:
 print(json.dumps({'ok':False,'error':str(exc)}))
'''
    proc = subprocess.run([str(PYTHON), '-c', script], cwd=tmp_path,
        env=os.environ | {'PYTHONPATH': str(SOURCE), 'PYTHONUTF8': '1'}, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_old_engineering_pass_does_not_imply_requested_gate_position(tmp_path):
    expr = 'z=self.head(f); return (z*torch.sigmoid(self.gate(z))).sum(1,keepdim=True)'
    assert run_fixture(tmp_path, expr, channels=2, required=False)['ok']
    result = run_fixture(tmp_path, expr, channels=2)
    assert not result['ok'] and 'spatial gate' in result['error'].lower()


def test_matching_gate_executes_before_classifier_and_receives_gradient(tmp_path):
    result = run_fixture(tmp_path, 'z=self.head(f*torch.sigmoid(self.gate(f))); return z[:,1:2]-z[:,0:1]')
    assert result['ok'], result
    check = result['result']['checks'][0]['spatial_gate']
    assert check['gate'] == 'gate' and check['classifier'] == 'head'
    assert check['weight_gradient'] == 'finite_nonzero'


@pytest.mark.parametrize('expression,kernel', [
    ('g=self.gate(f); z=self.head(f); return z[:,1:2]-z[:,0:1]', 3),
    ('z=self.head(f); g=self.gate(f); return z[:,1:2]-z[:,0:1]+g', 3),
    ('g=self.gate(f).detach(); z=self.head(f*torch.sigmoid(g)); return z[:,1:2]-z[:,0:1]', 3),
    ('g=self.gate(f)*0; z=self.head(f+g); return z[:,1:2]-z[:,0:1]', 3),
    ('z=self.head(f*torch.sigmoid(self.gate(f))); return z[:,1:2]-z[:,0:1]', 1),
])
def test_declared_but_unused_late_detached_zero_or_wrong_kernel_gate_is_rejected(tmp_path, expression, kernel):
    result = run_fixture(tmp_path, expression, kernel=kernel)
    assert not result['ok'] and 'spatial gate' in result['error'].lower()


def test_spatial_requirement_is_frozen_from_host_constraints():
    value = {'spatial_gate': {'feature_channels': 128, 'kernel_size': 3}}
    contract = make_validation_contract('unbound', value)
    value['spatial_gate']['kernel_size'] = 5
    assert contract['spatial_gate'] == {'feature_channels': 128, 'kernel_size': 3}


def test_parallel_gate_branch_cannot_pretend_to_feed_classifier(tmp_path):
    result = run_fixture(tmp_path, 'g=self.gate(f); z=self.head(f); return z[:,1:2]-z[:,0:1]+g')
    assert not result['ok'] and 'spatial gate' in result['error'].lower()


@pytest.mark.parametrize('spec', [None, {}, {'feature_channels': True, 'kernel_size': 3},
    {'feature_channels': 0, 'kernel_size': 3}, {'feature_channels': 128, 'kernel_size': 2},
    {'feature_channels': 128, 'kernel_size': 3, 'allow_failure': True}])
def test_invalid_spatial_constraints_are_rejected(spec):
    with pytest.raises(ValueError):
        make_validation_contract('unbound', {'spatial_gate': spec})


def test_spatial_requirement_binds_new_approval_and_model_cannot_relax_it(tmp_path, monkeypatch):
    from test_prototype_workspace import waiting
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    state = harness._load(rid)
    old = state['pending']
    requirement = {'feature_channels': 128, 'kernel_size': 3}
    harness.revise_constraints(rid, {'spatial_gate': requirement}, expected_revision=state['work']['revision'])
    state = harness._load(rid)
    harness._apply(rid, state, {'action': 'work', 'reason': 'fixture',
        'arguments': {'task_id': 'check', 'content': {'spatial_gate': None}}}, 6)
    assert state['pending']['approval'] != old['approval']
    assert runtime.db.get_tool_call(old['call'])['status'] == 'REJECTED'
    assert state['pending']['binding']['validation_contract']['spatial_gate'] == requirement
    assert harness._training(rid, state) is False
