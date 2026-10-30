"""Execute the approved frozen three-question public-text diagnostic, cap 20."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import httpx
from openai import OpenAI

from evals.qasper.agent_case import paper_records
from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
from paperpilot_agentic.config import RuntimeSettings, read_api_key_file
from paperpilot_agentic.models import Budget
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.providers import OpenAICompatibleLLM
from paperpilot_retrieval import PaperKnowledgeStore


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def run_summary(results, actual_requests, planned_count, limit=20):
    """Recorded outcomes include failures and skipped cases, not just completion."""
    return {'actual_requests': actual_requests, 'limit': limit, 'results': results,
            'complete_case_count': sum(row.get('status') == 'COMPLETED'
                                       and not row.get('error_type') for row in results),
            'recorded_case_count': len(results), 'planned_case_count': planned_count,
            'factual_review': 'PENDING'}


class CappedTransport(httpx.BaseTransport):
    def __init__(self, output, inner=None, *, limit=20):
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError('Request limit must be an integer between 1 and 20')
        self.limit = limit
        self.output, self.count, self.stopped = output, 0, False
        self.inner = inner or httpx.HTTPTransport(retries=0, trust_env=False)

    def handle_request(self, request):
        if (self.stopped or self.count >= self.limit or request.url.scheme != 'https'
                or request.url.host != 'api.deepseek.com'
                or request.url.path != '/v1/chat/completions'):
            raise httpx.RequestError('Frozen study network guard denied dispatch', request=request)
        self.count += 1
        number = self.count
        save(self.output/f'request-{number:03}.json', {'number': number,
            'state': 'dispatch_intent', 'payload': json.loads(request.content), 'headers_saved': False})
        try:
            response = self.inner.handle_request(request)
            save(self.output/f'response-{number:03}-status.json', {'status': response.status_code})
            if response.status_code in {400, 401, 403, 404}:
                self.stopped = True
            if response.status_code >= 400:
                response.close()
                return httpx.Response(response.status_code, json={'error': 'provider_http_error'})
            payload = response.read()
            response.close()
            with (self.output/f'response-{number:03}.bin').open('xb') as stream:
                stream.write(payload)
            headers = dict(response.headers)
            # read() returns decoded bytes, so do not decode them a second time.
            headers.pop('content-encoding', None)
            headers.pop('content-length', None)
            return httpx.Response(response.status_code, headers=headers, content=payload)
        except Exception as exc:
            save(self.output/f'request-{number:03}-error.json', {'error_type': type(exc).__name__})
            raise

    def close(self):
        self.inner.close()


def execute(plan_path, dataset_path, output, key, *, regression=False, max_requests=20):
    if type(max_requests) is not int or not 1 <= max_requests <= 20:
        raise ValueError('Request limit must be an integer between 1 and 20')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    study = plan['paperpilot']
    raw = dataset_path.read_bytes()
    if (plan['protocol'] != 'agent-quality-round2-20260913-v1' or study['cap'] != 20
            or plan['model'] != 'deepseek-chat' or study['engine'] != 'answer_first_text'
            or study['visual_policy'] != 'off' or len(study['cases']) != 3
            or hashlib.sha256(raw).hexdigest() != study['dataset_sha256']):
        raise ValueError('Frozen protocol or dataset mismatch')
    dataset = json.loads(raw.decode('utf-8-sig'))
    seen = set()
    for case in study['cases']:
        matches = [qa for qa in dataset[case['paper_id']]['qas'] if qa['question_id'] == case['question_id']]
        if len(matches) != 1 or matches[0]['question'] != case['question'] or case['question_id'] in seen:
            raise ValueError('Original question mismatch or duplicate')
        seen.add(case['question_id'])
    if not key:
        raise ValueError('API key required')
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root/'src').rglob('*.py')) + [Path(__file__), root/'evals/qasper/agent_case.py']
    budget = Budget(max_llm_calls=6, max_tokens=60000, max_seconds=240, max_tool_calls=12)
    save(output/'manifest.json', {'plan': plan, 'budget_per_question': budget.to_dict(),
        'study_kind': 'development_regression_seen_questions' if regression else 'original_frozen_study',
        'scope': 'current runtime resume, single supplied public paper, no browser or visual reader',
        'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'provider_max_retries': 0, 'transport_limit': max_requests, 'gold_evidence_sent': False, 'gold_answers_sent': False})
    transport = CappedTransport(output, limit=max_requests)
    client = httpx.Client(transport=transport, timeout=60, trust_env=False)
    results = []
    try:
        for index, case in enumerate(study['cases'], 1):
            if transport.stopped or transport.count >= max_requests:
                results.append({**case, 'status': 'NOT_RUN_PROVIDER_STOP_OR_BUDGET', 'actual_requests': 0})
                continue
            public, document, chunks = paper_records(dataset[case['paper_id']], source_id=case['paper_id'])
            save(output/f'public-source-{index}.json', public)
            with tempfile.TemporaryDirectory(prefix='paperpilot-deepseek-round2-') as temporary:
                workspace = Path(temporary)
                store = PaperKnowledgeStore(workspace/'knowledge.sqlite3', namespace='round2')
                store.sync_documents([document], chunks)
                settings = RuntimeSettings(provider='deepseek', api_key=key,
                    base_url='https://api.deepseek.com/v1', model='deepseek-chat', thinking_enabled=False,
                    temperature=0, max_tokens=4096, timeout_seconds=60, workspace=workspace,
                    runtime_db=workspace/'runtime.sqlite3', memory_db=workspace/'knowledge.sqlite3',
                    memory_namespace='round2', paper_roots=())
                provider = OpenAICompatibleLLM(settings, max_retries=0)
                provider._client.close()
                provider._client = OpenAI(api_key=key, base_url=settings.base_url, http_client=client, max_retries=0)
                runtime = ResearchAgentRuntime(db_path=settings.runtime_db, workspace=workspace,
                    search_adapter=PaperPilotKnowledgeSearchAdapter(store), llm=provider,
                    text_reader_protocol='round2:deepseek-chat:temperature0:visual-off')
                queued = runtime.enqueue(question=case['question'], budget=budget, metadata={
                    'engine': 'answer_first_text', 'visual_policy': 'off', 'paper_sha256s': [document.sha256]})
                rid = queued['run']['run_id']
                save(output/f'pending-{index}.json', {'case': case, 'run': queued})
                before, error = transport.count, None
                try:
                    result = runtime.resume(rid)
                except Exception as exc:
                    error, result = type(exc).__name__, runtime.status(rid)
                save(output/f'result-{index}.json', {'case': case, 'result': result, 'error_type': error,
                    'actual_requests': transport.count - before, 'factual_review': 'PENDING'})
                row = {**case, 'run_id': rid, 'status': result['run']['status'],
                       'actual_requests': transport.count - before, 'error_type': error}
                results.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        client.close()
        save(output/'summary.json', run_summary(results, transport.count, len(study['cases']), max_requests))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--execute-paid', action='store_true')
    parser.add_argument('--regression', action='store_true', help='Label an explicitly authorized seen-question rerun')
    parser.add_argument('--max-requests', type=int, default=20)
    args = parser.parse_args()
    if not args.execute_paid:
        print('No requests made. Explicit --execute-paid required.')
        return
    execute(args.plan, args.dataset, args.output, read_api_key_file(args.key_file),
            regression=args.regression, max_requests=args.max_requests)


if __name__ == '__main__':
    main()
