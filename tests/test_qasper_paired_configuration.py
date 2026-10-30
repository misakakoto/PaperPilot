"""The paired evaluator must not inherit different checkout-local settings."""
import json
import os
from pathlib import Path

from evals.qasper import paired_holdout


def test_frozen_configuration_overrides_checkout_and_ambient_settings(tmp_path, monkeypatch):
    # Restore every environment key written by the public configuration boundary.
    for key in (*paired_holdout.retrieval_configuration(tmp_path/'cache'), 'PYTHON_DOTENV_DISABLED'):
        monkeypatch.setenv(key, os.environ.get(key, ''))
    monkeypatch.setenv('PAPERPILOT_ENABLE_DENSE', 'false')
    monkeypatch.setenv('PAPERPILOT_ENABLE_RERANKER', 'false')
    monkeypatch.setenv('PAPERPILOT_LLM_MAX_TOKENS', '19')
    monkeypatch.setenv('PAPERPILOT_RERANK_STAGE', 'candidate')
    monkeypatch.setenv('PAPERPILOT_ALLOW_EXTERNAL_SCHOLARLY', 'true')
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'test-only-not-a-real-key')
    protocol = {
        'provider': 'qwen', 'model': 'frozen-model', 'temperature': 0.2,
        'timeout_seconds': 90, 'max_output_tokens': 8000,
        'retrieval_config': paired_holdout.retrieval_configuration(tmp_path / 'cache'),
    }
    for checkout in ('baseline', 'candidate'):
        folder = tmp_path / checkout
        folder.mkdir()
        (folder / '.env').write_text('PAPERPILOT_LLM_MODEL=wrong-model\nPAPERPILOT_ENABLE_DENSE=false\n')
        monkeypatch.chdir(folder)
        settings = paired_holdout.frozen_settings(protocol, folder)
        assert settings.model == 'frozen-model' and settings.max_tokens == 8000
        assert os.environ['PAPERPILOT_ENABLE_DENSE'] == 'true'
        assert os.environ['PAPERPILOT_ENABLE_RERANKER'] == 'true'
        assert os.environ['PAPERPILOT_RERANK_STAGE'] == 'final'
        assert os.environ['PAPERPILOT_ALLOW_EXTERNAL_SCHOLARLY'] == 'false'
        assert os.environ['PYTHON_DOTENV_DISABLED'] == '1'
        assert Path(os.environ['PAPERPILOT_EMBEDDING_CACHE']).is_absolute()
        assert 'test-only-not-a-real-key' not in json.dumps(protocol)
