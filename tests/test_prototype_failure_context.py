"""Trusted tiny fixture models only; no generated code execution or training."""
import json
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from paperpilot_agentic.prototype_worker import run_validation


MODEL = '''import torch
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.conv=torch.nn.Conv2d(6,1,1)
 def forward(self,x,y):
  if x.shape[-2] % 2:
   raise RuntimeError('fixture spatial alignment failed')
  return self.conv(torch.cat([x,y],dim=1))
def build_model(config):
 return Model()
'''


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch environment required')
class FailureContextTests(unittest.TestCase):
    def setUp(self):
        previous = Path.cwd()
        folder = tempfile.TemporaryDirectory(prefix='paperpilot-trusted-worker-')
        self.addCleanup(folder.cleanup)
        self.addCleanup(os.chdir, previous)
        self.folder = Path(folder.name)
        os.chdir(self.folder)

    def test_later_case_failure_retains_completed_checks_and_exact_phase(self):
        (self.folder / 'model.py').write_text(MODEL, encoding='utf-8')
        (self.folder / 'config.json').write_text('{}', encoding='utf-8')
        (self.folder / 'contract.json').write_text(json.dumps({
            'kind': 'unbound', 'input_cases': [[1,16,20], [2,17,23]],
            'max_parameters': 100}), encoding='utf-8')
        result = run_validation()
        assert result['passed'] is False
        assert result['error'] == 'fixture spatial alignment failed'
        assert result['checks'][0]['shape'] == [1,1,16,20]
        assert len(result['checks']) == 1
        assert result['failure_context']['phase'] == 'forward'
        assert result['failure_context']['case_index'] == 1
        assert result['failure_context']['input_shape'] == [2,3,17,23]
        assert result['training_steps'] == 0

    def test_bad_constructor_reports_loading_failure_without_claiming_a_case(self):
        (self.folder / 'model.py').write_text('def build_model(config):\n raise ValueError("fixture bad constructor")\n', encoding='utf-8')
        (self.folder / 'config.json').write_text('{}', encoding='utf-8')
        result = run_validation()
        assert result['passed'] is False
        assert result['failure_context']['phase'] == 'load'
        assert 'case_index' not in result['failure_context']
        assert result['checks'] == []


if __name__ == '__main__':
    unittest.main()
