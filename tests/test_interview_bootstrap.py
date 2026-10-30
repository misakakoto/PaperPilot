"""Budget wrapper must reach the registered visual tool, not an unused attribute."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.interview.budget import CampaignBudget, RecordedCompletions, RecordedProvider  # noqa: E402
from paperpilot_agentic.bootstrap import build_runtime
from paperpilot_agentic.config import RuntimeSettings
from paperpilot_agentic.models import Budget
from paperpilot_agentic.tooling import ToolContext
from test_interview_campaign import Provider


def test_composition_factory_wraps_the_adapter_captured_by_registered_tool(tmp_path, monkeypatch):
    from paperpilot_agentic import builtin_tools
    for key in ('PAPERPILOT_ENABLE_DENSE', 'PAPERPILOT_ENABLE_RERANKER'):
        monkeypatch.setenv(key, 'false')
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'test')
    settings = RuntimeSettings(provider='qwen', api_key='test', model='fixed', base_url='http://localhost',
        thinking_enabled=False, temperature=0, max_tokens=100, timeout_seconds=1,
        workspace=tmp_path, runtime_db=tmp_path/'runtime.sqlite3', memory_db=None,
        memory_namespace='test', paper_roots=(tmp_path,))
    campaign = CampaignBudget(tmp_path/'receipts', {'model': 'fixed'}, max_requests=1)
    calls = []
    def send(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model_dump=lambda **kw: {'usage': {'prompt_tokens': 10, 'completion_tokens': 10}})
    sdk = RecordedCompletions(SimpleNamespace(create=send), campaign, phase='vision')
    class Vision:
        model, max_output_tokens = 'fixed', 100
        def analyze(self, image, prompt, context, *, before_attempt=None):
            sdk.before_send = before_attempt
            sdk.create(model='fixed', messages=[{'role': 'user', 'content': prompt}], max_tokens=100)
            return {'description': 'A: 12', 'verified': False, 'confidence': .5,
                    'metadata': {'model': 'fixed', 'prompt_tokens': 10, 'completion_tokens': 10}}
    pdf = tmp_path/'fixture.pdf'
    pdf.write_bytes(b'fixture')
    image = tmp_path/'region.png'
    image.write_bytes(b'fixture image')
    item = {'object_id': 'table1', 'object_type': 'table', 'page': 1, 'text': 'TABLE I A: 12',
            'caption': 'TABLE I', 'artifact_path': str(image), 'document_sha256': 'source',
            'bbox': [0, 0, 100, 100]}
    def parse_layout(pdf_path, output, **kwargs):
        manifest = {'objects': [item]}
        (output/'layout_manifest.json').write_text(json.dumps(manifest))
        return manifest
    monkeypatch.setattr(builtin_tools, 'parse_pdf_layout', parse_layout)
    monkeypatch.setattr(builtin_tools, 'search_layout_objects', lambda *a, **kw: [item])
    provider = RecordedProvider(Provider(), campaign, phase='text')
    runtime = build_runtime(settings, llm_factory=lambda config: provider, vision_factory=lambda **kw: Vision())
    assert runtime.llm_runtime.model is provider
    rid = runtime.db.create_run('Read table I', Budget())
    context = ToolContext(rid, 'step', 'tool', tmp_path, runtime.db, runtime.tracer)
    result = runtime.registry.get('paper.inspect_layout').handler(context,
        {'pdf_path': str(pdf), 'query': 'Read TABLE I', 'object_type': 'table'})
    assert result.ok and len(calls) == 1
    assert campaign.snapshot()['actual_requests'] == 1
    from paperpilot_agentic.llm_runtime import LLMBudgetExhausted
    with pytest.raises(LLMBudgetExhausted):
        provider.complete([{'role': 'user', 'content': 'audit'}], max_tokens=100)
    assert len(calls) == 1
    second = runtime.db.create_run('Read table I again', Budget())
    second_context = ToolContext(second, 'step', 'tool2', tmp_path, runtime.db, runtime.tracer)
    blocked = runtime.registry.get('paper.inspect_layout').handler(second_context,
        {'pdf_path': str(pdf), 'query': 'Read TABLE I', 'object_type': 'table'})
    assert blocked.output['vision_api_calls'] == 0
    assert runtime.db.get_run(second)['used_llm_calls'] == 0
    assert len(calls) == 1
