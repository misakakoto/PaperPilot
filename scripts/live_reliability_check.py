"""Submit frozen questions through the product HTTP API and save unchanged results.

This performs real provider calls. Review criteria are never transmitted.
An existing output directory is refused to prevent accidental paid replays.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--base', default='http://127.0.0.1:8014')
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    cases = json.loads(Path(args.cases).read_text(encoding='utf-8'))
    if isinstance(cases, dict):
        cases = cases['cases']
    output = Path(args.output).resolve()
    private = Path('.paperpilot').resolve()
    if not output.is_relative_to(private) or output == private:
        parser.error('Output must be a new subdirectory of .paperpilot')
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    save('frozen-cases.json', cases)
    def api(path, body=None):
        request = Request(args.base + path,
            data=json.dumps(body).encode('utf-8') if body is not None else None,
            headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    def run(index_case):
        index, case = index_case
        # Persist intent BEFORE POST. Never retry a POST with uncertain receipt.
        save(f'intent-{index}.json', {'question': case['question'], 'submission': 'pending'})
        response = api('/api/agentic/runs', {'question': case['question'],
            'engine': 'answer_first_text', 'visual_policy': 'auto',
            'metadata': {'source': 'reliability-http-20260912'},
            'budget': {'max_llm_calls': 10, 'max_tokens': 100000, 'max_seconds': 600}})
        rid = response['run']['run_id']
        save(f'submitted-{index}.json', response)
        print(json.dumps({'case': index, 'run_id': rid}), flush=True)
        previous = None
        deadline = time.monotonic() + 660
        while time.monotonic() < deadline:
            result = api('/api/agentic/runs/' + rid)
            state = (result['run']['status'], result['run']['used_llm_calls'], result['run']['used_tokens'])
            if state != previous:
                print(json.dumps({'case': index, 'state': state}), flush=True)
                previous = state
            if state[0] in {'COMPLETED', 'FAILED', 'CANCELLED', 'BUDGET_EXHAUSTED', 'WAITING_APPROVAL'}:
                save(f'result-{index}.json', result)
                return {'case': index, 'run_id': rid, 'status': state[0],
                        'calls': state[1], 'tokens': state[2], 'review': 'PENDING_HUMAN_SOURCE_REVIEW'}
            time.sleep(5)
        save(f'pending-{index}.json', result)
        return {'case': index, 'run_id': rid, 'status': 'MONITOR_TIMEOUT'}
    with ThreadPoolExecutor(max_workers=2) as pool:
        summary = list(pool.map(run, enumerate(cases, 1)))
    save('summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
