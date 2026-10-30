"""Export an unchanged local API result and answer; never calls an LLM."""
import argparse
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--port', type=int, default=8014)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.run_id.startswith('run_') or not args.run_id.removeprefix('run_').isalnum():
        raise ValueError('Invalid run ID')
    if args.output.exists():
        raise FileExistsError('Preserve prior exports; use a new directory')
    with urlopen(f'http://127.0.0.1:{args.port}/api/agentic/runs/{args.run_id}', timeout=15) as response:
        result = json.load(response)
    run = result['run']
    if run['status'] not in {'COMPLETED', 'FAILED', 'CANCELLED', 'BUDGET_EXHAUSTED'}:
        raise ValueError('Run is not terminal; do not treat a snapshot as a final result')
    args.output.mkdir(parents=True)
    (args.output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    answer = (result.get('final_answer') or {}).get('answer') or ''
    content = (f'# 原始问答记录\n\n运行：{args.run_id}\n\n'
               f'状态：{run["status"]} / {run["research_outcome"]}\n\n'
               f'请求：{run["used_llm_calls"]}；tokens：{run["used_tokens"]}\n\n'
               f'## 原始问题\n\n{run["question"]}\n\n## Agent 原始回答（未改写）\n\n{answer}\n')
    (args.output/'answer.md').write_text(content, encoding='utf-8')
    print(args.output/'answer.md')


if __name__ == '__main__':
    main()
