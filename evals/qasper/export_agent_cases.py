"""Export completed production-runtime cases without rewriting their answers."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output file')
    lines = ['# 完整产品链路：逐次原始回答', '',
             '以下为真实运行输出，未经助手改写。运行状态不代表事实正确；各次问题如下分别列出。', '']
    calls = tokens = 0
    for folder in args.folders:
        result = json.loads((folder/'result.json').read_text(encoding='utf-8'))
        manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        run = result['run']
        answer = result.get('final_answer') or {}
        calls += run['used_llm_calls']
        tokens += run['used_tokens']
        lines += [f'## {folder.name}', '', f"问题：{manifest['question']}", '',
                  f"状态：{run['status']} / {answer.get('research_outcome')}", '',
                  f"请求：{run['used_llm_calls']}；tokens：{run['used_tokens']}", '',
                  str(answer.get('answer') or '(没有最终回答)'), '', '工具错误：', '']
        lines += [f"- {call['tool_name']}: {call['error']}" for call in result['tool_calls'] if call.get('error')]
        lines += ['']
    lines += [f'合计请求：{calls}；合计tokens：{tokens}。未估算账单费用。', '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'calls': calls, 'tokens': tokens, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
