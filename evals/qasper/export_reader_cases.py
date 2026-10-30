"""Export experimental reader outputs verbatim, separately from correctness review."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    lines = ['# 实验阅读链路原始回答', '',
             '逐例注明自动检索或冻结证据诊断；两种范围不可混作端到端成绩。不包含正式Agent全部编排功能。',
             '模型审稿与引文绑定通过不等于事实正确；以下答案未经助手改写。', '']
    attempts = tokens = 0
    for folder in args.folders:
        source = folder if folder.is_file() else folder/'reader-result.json'
        data = json.loads(source.read_text(encoding='utf-8'))
        reader = data.get('reader', data)
        routing = data.get('routing', {})
        count = reader['provider_attempts'] + routing.get('provider_attempts', 0)
        used = sum(c['prompt_tokens'] + c['completion_tokens']
                   for c in reader['calls'] + routing.get('calls', []))
        persisted = data.get('persisted_run')
        if persisted is not None:
            # Truncated/error responses can be charged before being raised to the caller.
            # The runtime database, not the successful-response list, is authoritative.
            count = persisted['used_llm_calls']
            used = persisted['used_tokens']
        attempts += count
        tokens += used
        lines.extend([f'## {folder.name}', '', data['question'], '',
                      '评测范围：' + str(data.get('evaluation_scope') or
                          ('ORIGINAL_QUERY_RETRIEVAL_READER' if data.get('run_id') else 'UNSPECIFIED')), '',
                      f"状态：{reader['status']}；请求：{count}；tokens：{used}", '',
                      str(reader.get('answer') or '(没有通过核验的答案)'), '',
                      '限制：' + json.dumps(reader.get('limitations', []), ensure_ascii=False), '',
                      '### 原始草稿（可能含被拒绝的内容）', '', '```json',
                      json.dumps(reader.get('draft'), ensure_ascii=False, indent=2), '```', ''])
    lines.extend([f'合计{attempts}次请求，{tokens} tokens。', ''])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'attempts': attempts, 'tokens': tokens, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
