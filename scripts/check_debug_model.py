"""One-request, recorded text-model probe; dry-run unless --execute-paid is set."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from paperpilot_agentic.config import RuntimeSettings, read_api_key_file, read_dashscope_api_key_file
from paperpilot_agentic.providers import OpenAICompatibleLLM
from paperpilot_agentic.provider_diagnostics import (
    failure_category as failure_category, provider_error_code as provider_error_code,
    safe_provider_failure,
)


QUESTION = ('This is a synthetic diagnostic, not a real paper result. Method A reports F1=90% '
            'on split X; method B reports F1=92% on split Y. No common-split comparison is '
            'provided. Can these scores establish that B is better? Return a JSON object '
            'with comparable (boolean) and reason (string). Do not claim experiments ran.')
MESSAGES = [{'role': 'system', 'content': 'Answer only from the supplied facts; return JSON.'},
            {'role': 'user', 'content': QUESTION}]


def probe(settings, output):
    """Save a fresh isolated receipt, with at most one dispatched request."""
    output.mkdir(parents=True, exist_ok=False)
    result = {'provider': settings.provider, 'requested_model': settings.model,
              'thinking_enabled': False, 'max_actual_requests': 1, 'actual_requests': 0,
              'question': QUESTION, 'status': 'NOT_DISPATCHED',
              'assurance': 'Interface smoke only, not Agent or factual quality evaluation'}
    def save(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    def reserve():
        if result['actual_requests'] >= 1:
            raise RuntimeError('Probe actual-request cap exhausted')
        result['actual_requests'] += 1
        save('request.json', {'messages': MESSAGES, 'model': settings.model,
                             'max_tokens': 1024, 'json_mode': True, 'thinking_enabled': False})
        save('result.json', result)  # Keep attempted dispatch if interrupted.
    try:
        model = OpenAICompatibleLLM(settings, max_retries=0)
        response = model.complete(MESSAGES, max_tokens=1024, json_mode=True,
                                  thinking_enabled=False, before_attempt=reserve)
        save('response.json', asdict(response))
        result.update(returned_model=response.model, tokens=response.total_tokens,
                      finish_reason=response.finish_reason, request_id=response.request_id)
        try:
            parsed = json.loads(response.text)
        except (ValueError, TypeError):
            parsed = None
        shape_ok = (isinstance(parsed, dict) and type(parsed.get('comparable')) is bool
                    and isinstance(parsed.get('reason'), str) and bool(parsed['reason'].strip()))
        result['status'] = ('STRUCTURED_RESPONSE' if shape_ok and response.finish_reason == 'stop'
                            else 'INVALID_OR_INCOMPLETE_RESPONSE')
    except Exception as exc:
        diagnostics = safe_provider_failure(exc)
        result.update(status='ERROR', **diagnostics, error_category=diagnostics['category'])
    finally:
        save('result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=['qwen', 'deepseek'], default='qwen')
    parser.add_argument('--model', default=None)
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--execute-paid', action='store_true')
    args = parser.parse_args()
    if args.provider == 'deepseek' and not args.model:
        parser.error('DeepSeek requires an explicit currently available --model')
    model = args.model or 'qwen-turbo'
    if not args.execute_paid:
        print(json.dumps({'provider': args.provider, 'model': model, 'api_requests': 0,
                          'max_actual_requests': 1, 'question': QUESTION}))
        return
    if not args.key_file:
        parser.error('--execute-paid requires --key-file; never paste credentials')
    reader = read_dashscope_api_key_file if args.provider == 'qwen' else read_api_key_file
    key = reader(args.key_file)
    root = Path(__file__).resolve().parents[1] / 'outputs'
    root.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='debug-model-', dir=root)) / 'receipt'
    endpoint = ('https://dashscope.aliyuncs.com/compatible-mode/v1' if args.provider == 'qwen'
                else 'https://api.deepseek.com')
    settings = RuntimeSettings(args.provider, key, endpoint, model, False, 0, 1024, 45,
                               output, output/'unused.sqlite3', None, 'debug', ())
    result = probe(settings, output)
    print(json.dumps({**result, 'output': str(output)}, ensure_ascii=True))
    if result['status'] != 'STRUCTURED_RESPONSE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
