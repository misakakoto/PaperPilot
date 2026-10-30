"""Real-model and real-Docker bounded acceptance through ResearchAgentRuntime."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from paperpilot_agentic.adapters import LocalCorpusSearchAdapter
from paperpilot_agentic.config import RuntimeSettings, read_api_key_file
from paperpilot_agentic.models import Budget
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.providers import OpenAICompatibleLLM
from paperpilot_agentic.docker_acceptance import resolve_local_docker_image


QUESTION = '''Investigate BM25 length normalization for QASPER evidence retrieval. Read the supplied
public-source notes, distinguish the motivation from any empirical claim, then propose ONE falsifiable
directional hypothesis and run the approved ablation. This is a bounded diagnostic, not model innovation.
The executable capability is fixed: baseline="bm25_b0", variants=["bm25_b075"], dataset="qasper-dev-v0.3",
seeds=[7,11,19], metric_files=["metrics.json"], metrics=["improvement"]. The program fixes k1=1.2,
IDF log(N/df), tokenization, candidates and top-k=3; the sole scoring change is b=0 to b=0.75.
For each seed it selects 12 papers, scores all their text-evidence questions using best-annotator
exact-paragraph Recall@3 and reports candidate minus baseline as improvement. Sampling can overlap.
There is no training or tuning; this dev data was used before, so do not call it held out or independent.
Use one primary_source target for the rationale and one independent_experiment claim for the
testable hypothesis, with experiment_spec preserving the exact executable capability above.
Choose a preregistered directional criterion on improvement (operator >, >=, < or <=; numeric value 0);
do not assert the direction is already true. Use resource_limits timeout_seconds=60 and memory_mb=512.
All registered seeds must meet the criterion. After execution report actual per-seed results,
support/refutation/inconclusive, limitations and a next experiment; do not equate execution success
with a supported hypothesis. Answer in Chinese. Cite sources and experiment evidence separately.'''


class RecordedModel:
    """Record unchanged requests/responses, with a process-wide actual-attempt cap."""
    def __init__(self, model, output, maximum=12):
        self.inner, self.output, self.maximum = model, output, maximum
        self.model = model.model
        self.attempts = 0

    def complete(self, messages, *, json_mode=False, max_tokens=None,
                 thinking_enabled=None, before_attempt=None):
        if self.attempts >= self.maximum:
            raise RuntimeError('Live acceptance actual-request cap exhausted')
        def reserve():
            if before_attempt:
                before_attempt()
            self.attempts += 1
            self._save(f'{self.attempts:02d}-request.json', {'messages': messages,
                       'json_mode': json_mode, 'max_tokens': max_tokens,
                       'thinking_enabled': thinking_enabled, 'attempt': self.attempts})
        try:
            response = self.inner.complete(messages, json_mode=json_mode, max_tokens=max_tokens,
                                           thinking_enabled=thinking_enabled, before_attempt=reserve)
        except Exception as exc:
            self._save(f'{self.attempts:02d}-error.json', {'type': type(exc).__name__})
            raise
        self._save(f'{self.attempts:02d}-response.json', asdict(response))
        print(f'API attempt {self.attempts} completed', flush=True)
        return response

    def _save(self, name, value):
        (self.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def check_approved_protocol(protocol):
    design = protocol.get('study_design', {})
    for key, expected in {'baseline': 'bm25_b0', 'variants': ['bm25_b075'],
                          'dataset': 'qasper-dev-v0.3', 'seeds': [7, 11, 19]}.items():
        if design.get(key) != expected:
            raise ValueError(f'Model design does not match approved executable: {key}')
    if protocol.get('seed_sweep') != [7, 11, 19]:
        raise ValueError('Full seed sweep not bound')
    criteria = protocol.get('acceptance_criteria', [])
    if len(criteria) != 1 or not all(
        c.get('metric') == 'improvement' and c.get('operator') in {'>', '>=', '<', '<='}
        and type(c.get('value')) in {int, float} and c['value'] == 0 for c in criteria
    ):
        raise ValueError('Unsupported preregistered criterion; do not silently rewrite it')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--model', default='deepseek-v4-flash')
    parser.add_argument('--execute-paid', action='store_true')
    args = parser.parse_args()
    if not args.execute_paid:
        print(json.dumps({'question': QUESTION, 'max_actual_requests': 12, 'api_requests': 0}))
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    image, version = resolve_local_docker_image('python:3.11-slim')
    source, corpus, calls = output / 'source', output / 'corpus', output / 'api'
    for directory in (source, corpus, calls):
        directory.mkdir()
    shutil.copy2(root / 'examples/qasper_length_ablation/experiment.py', source / 'experiment.py')
    shutil.copy2(root / '.paperpilot/datasets/qasper/v0.3/qasper-dev-v0.3.json', source / 'qasper-dev-v0.3.json')
    shutil.copy2(root / 'examples/qasper_length_ablation/README.md', corpus / 'public-source-notes.md')
    settings = RuntimeSettings('deepseek', read_api_key_file(args.key_file), 'https://api.deepseek.com',
                               args.model, False, 0.0, 6000, 90, output, output/'runtime.sqlite3',
                               None, 'isolated-research', (corpus,))
    model = RecordedModel(OpenAICompatibleLLM(settings, max_retries=0), calls)
    runtime = ResearchAgentRuntime(db_path=output/'runtime.sqlite3', workspace=output/'workspace',
                                   search_adapter=LocalCorpusSearchAdapter([corpus]), llm=model)
    manifest = {'question': QUESTION, 'model': args.model, 'max_actual_requests': 12,
                'docker_image_preflight': image, 'docker_version': version,
                'source_type': 'curated public-source notes, not original full papers',
                'dataset_sha256': hashlib.sha256((source/'qasper-dev-v0.3.json').read_bytes()).hexdigest(),
                'experiment_code_sha256': hashlib.sha256((source/'experiment.py').read_bytes()).hexdigest()}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    try:
        result = runtime.start(question=QUESTION, budget=Budget(max_llm_calls=12, max_tool_calls=16,
                               max_steps=18, max_seconds=1200), metadata={
            'experiment_source_dir': str(source),
            'research_exploration': True,
            'experiment_docker_image': image,
            'experiment_command': ['python', 'experiment.py', '--seed', '__PAPERPILOT_SEED__', '--output', 'metrics.json'],
            'allow_external_search': False})
        (output/'before-approval.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        if result['run']['status'] == 'WAITING_APPROVAL':
            protocols = [c['metadata']['experiment'] for c in result['claim_graph']['claims']
                         if c['metadata'].get('experiment')]
            if len(protocols) != 1:
                raise ValueError('Expected exactly one registered experimental claim')
            check_approved_protocol(protocols[0])
            if protocols[0]['policy']['docker_image'] != image:
                raise ValueError('Approved experiment must use the preflight image digest')
            for approval in result['pending_approvals']:
                runtime.approve(approval['approval_id'], actor='user-authorized-live-acceptance',
                                comment='Reviewed fixed local QASPER ablation, three seeds, isolated copy, no network')
            result = runtime.resume(result['run']['run_id'])
        (output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'status': result['run']['status'], 'outcome': result['run'].get('research_outcome'),
                          'api_attempts': model.attempts, 'tools': [t['tool_name'] for t in result['tool_calls']],
                          'output': str(output)}, ensure_ascii=False), flush=True)
    finally:
        (output/'attempt-count.json').write_text(json.dumps({'actual_requests': model.attempts}), encoding='utf-8')


if __name__ == '__main__':
    main()
