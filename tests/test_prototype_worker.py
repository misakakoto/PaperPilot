"""Trusted, fixed test models only. No generated candidate execution or training."""
import json

import pytest

pytest.importorskip('torch')
from paperpilot_agentic.prototype_worker import verify
from paperpilot_agentic.prototype_worker import run_validation


MODEL = '''
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 1, 1)
    def forward(self, x, y):
        return EXPRESSION
def build_model(config):
    return Model()
'''


def fixture_model(tmp_path, monkeypatch, expression):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'model.py').write_text(MODEL.replace('EXPRESSION', expression), encoding='utf-8')
    (tmp_path/'config.json').write_text(json.dumps({}), encoding='utf-8')


def test_cpu_forward_backward_for_odd_shapes_and_batch_sizes(tmp_path, monkeypatch):
    fixture_model(tmp_path, monkeypatch, 'self.conv(torch.cat([x, y], dim=1))')
    result = verify()
    assert result['parameters'] == 7
    assert [r['shape'] for r in result['checks']] == [[1, 1, 64, 64], [2, 1, 65, 97], [1, 1, 96, 128]]


@pytest.mark.parametrize('expression,error', [
    ('self.conv(torch.cat([x, y], dim=1))[:, :, ::2, ::2]', 'input resolution'),
    ('torch.cat([self.conv(torch.cat([x, y], dim=1))]*2, dim=1)', 'input resolution'),
    ('self.conv(torch.cat([x, x], dim=1))', 'Both inputs'),
    ('self.conv(torch.cat([x, y], dim=1))*float("nan")', 'Nonfinite output'),
])
def test_worker_rejects_real_tensor_failures(tmp_path, monkeypatch, expression, error):
    fixture_model(tmp_path, monkeypatch, expression)
    with pytest.raises(ValueError, match=error):
        verify()


@pytest.mark.parametrize('tuple_output', [True, False])
def test_native_snunet_contract_uses_two_channels_and_tuple(tmp_path, monkeypatch, tuple_output):
    monkeypatch.chdir(tmp_path)
    code = '''import torch
class SNUNet_ECAM(torch.nn.Module):
 def __init__(self,in_ch=3,out_ch=2):
  super().__init__()
  self.conv=torch.nn.Conv2d(in_ch*2,out_ch,1)
 def forward(self,x,y):
  out=self.conv(torch.cat([x,y],dim=1))
  return OUTPUT
'''.replace('OUTPUT', '(out,)' if tuple_output else 'out')
    (tmp_path/'model.py').write_text(code, encoding='utf-8')
    (tmp_path/'config.json').write_text('{}', encoding='utf-8')
    (tmp_path/'contract.json').write_text('{"kind":"snunet_ecam"}', encoding='utf-8')
    if tuple_output:
        assert [r['shape'] for r in verify()['checks']] == [[1, 2, 32, 32], [2, 2, 32, 48], [1, 2, 48, 64]]
    else:
        with pytest.raises(ValueError, match='tuple'):
            verify()


def test_real_tensors_use_requested_sizes_and_reject_parameter_limit(tmp_path, monkeypatch):
    fixture_model(tmp_path, monkeypatch, 'self.conv(torch.cat([x, y], dim=1))')
    contract = {'kind':'unbound', 'input_cases':[[1,48,80],[2,64,48]], 'max_parameters':6}
    (tmp_path/'contract.json').write_text(json.dumps(contract),encoding='utf-8')
    result = run_validation()
    assert [r['shape'] for r in result['checks']] == [[1,1,48,80],[2,1,64,48]]
    assert result['parameters'] == 7 and result['passed'] is False
    assert result['failure_kind'] == 'DESIGN_CONSTRAINT'
    assert result['constraint_violations'][0]['metric'] == 'parameters'
    assert result['training_steps'] == 0 and result['seed'] == 1729
