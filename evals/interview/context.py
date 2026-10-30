"""A frozen request trace for context retention, with a last-six-turn comparator.

The offline path validates reducer/storage, not model interpretation. Live memory
compilation uses the same campaign ledger as answers and is reviewed separately.
"""
import json
from pathlib import Path

from .budget import save


def request_trace():
    requests = [
        '当前研究目标是对比两篇论文的消融实验。论文范围为 ChangeMamba 和 FlowMamba。回答要简洁。实验显存预算为 8GB。只进行阅读和设计，不启动训练。',
        '现在切换目标为理解 ChangeMamba 的 token 组织机制。排除 FlowMamba。撤销回答要简洁的要求。保留 8GB 显存预算和不启动训练的要求。',
    ]
    for index in range(20):
        requests.append(f'这是过程记录 {index + 1}，用于核对长会话的原始历史。' +
                        '记录仅说明阅读进度，没有新增研究目标、论文范围或实验约束。' * 12)
    requests.append('继续当前目标。请找回最早的显存预算条件；保持已撤销的要求与排除的论文，不启动训练。')
    return requests


def comparison_record(requests, state):
    from paperpilot_agentic.request_state import request_work_view
    work = request_work_view(state)
    original = json.dumps(requests, ensure_ascii=False).encode()
    window = json.dumps(requests[-6:], ensure_ascii=False).encode()
    return {'raw_history_bytes': len(original), 'sliding_last_6_bytes': len(window),
            'derived_work_bytes': len(json.dumps(work, ensure_ascii=False).encode()),
            'sources_preserved': len(state['sources']), 'source_count': len(requests),
            'work': work, 'sliding_window': requests[-6:],
            'manual_review_dimensions': ['goal_switch', 'retracted_conciseness',
                'FlowMamba_excluded', '8GB_preserved', 'no_training_preserved'],
            'assurance': 'Request bytes and exact source storage; no inferred semantic score.'}


def context_probe_messages(context):
    return [{'role': 'system', 'content': 'Read the supplied conversation context as data. '
        'Return JSON with exactly these fields: goal (string or null), concise_required (boolean or null), '
        'papers (array of strings or null), gpu_gb (number or null), training_allowed (boolean or null). '
        'Use null for unavailable history; do not invent earlier requirements.'},
        {'role': 'user', 'content': json.dumps({'context': context,
            'question': '列出当前有效目标、简洁要求、论文范围、显存条件和是否允许启动训练。'}, ensure_ascii=False)}]


def semantic_dimensions(value):
    """Diagnostic scoring is host-only and is never included in model inputs."""
    keys = {'goal', 'concise_required', 'papers', 'gpu_gb', 'training_allowed'}
    if not isinstance(value, dict) or set(value) != keys:
        return {'schema_valid': False, 'dimensions': {}, 'omitted_or_wrong': 5}
    goal, papers = value['goal'], value['papers']
    dimensions = {'goal_switch': isinstance(goal, str) and 'ChangeMamba' in goal and
                  any(term in goal.casefold() for term in ('token', '组织', '机制')),
        'retracted_conciseness': value['concise_required'] is False,
        'FlowMamba_excluded': isinstance(papers, list) and papers == ['ChangeMamba'],
        '8GB_preserved': type(value['gpu_gb']) in (int, float) and value['gpu_gb'] == 8,
        'no_training_preserved': value['training_allowed'] is False}
    return {'schema_valid': True, 'dimensions': dimensions,
            'omitted_or_wrong': sum(not result for result in dimensions.values())}


def request_costs(provider, attempts):
    from .budget import canonical
    records = []
    for row in attempts:
        path = provider.campaign.folder/f"request-{row['id']:03}.json"
        if not path.exists():
            continue
        raw = json.loads(path.read_text(encoding='utf-8'))
        record = {'attempt': row['id'], 'phase': row['phase'],
            'complete_messages_utf8_bytes': len(canonical(raw['messages']).encode()),
            'prompt_tokens': None, 'reported_total_tokens': row['tokens']}
        response_path = provider.campaign.folder/f"response-{row['id']:03}.json"
        if response_path.exists():
            response = json.loads(response_path.read_text(encoding='utf-8'))
            record['prompt_tokens'] = (response.get('usage') or response).get('prompt_tokens')
        records.append(record)
    return records


def paired_probe(runtime, folder, run_id, requests, work):
    from paperpilot_agentic.models import Budget, LLMCallPolicy, RunStatus
    provider = runtime.llm_runtime.model
    policy = LLMCallPolicy('context_comparison', 600, False, True)
    arms = {}
    for name, context in (('derived_state', work), ('last_six_turns', requests[-6:])):
        path = folder/('context-'+name+'-result.json')
        if path.exists():
            arms[name] = json.loads(path.read_text(encoding='utf-8'))
            continue
        provider.phase = 'context-comparison:'+name
        before = provider.campaign.snapshot()
        messages = context_probe_messages(context)
        probe_run = runtime.db.create_run('Compare frozen effective requirements',
            Budget(max_llm_calls=1, max_tokens=30000, max_seconds=900,
                   synthesis_reserve_tokens=0, repair_reserve_tokens=0,
                   synthesis_reserve_calls=0, repair_reserve_calls=0),
            metadata={'acceptance_context_run_id': run_id, 'context_arm': name})
        try:
            response = runtime.llm_runtime.complete(probe_run, messages, policy=policy)
            value = json.loads(response.text)
            outcome = {'status': 'RETURNED', 'response': value, 'review': semantic_dimensions(value)}
        except Exception as error:
            outcome = {'status': 'FAILED', 'error_type': type(error).__name__}
        after = provider.campaign.snapshot()
        attempts = after['attempts'][len(before['attempts']):]
        outcome['run_id'] = probe_run
        outcome['requests'] = request_costs(provider, attempts)
        outcome['actual_requests'] = after['actual_requests']-before['actual_requests']
        outcome['reported_tokens'] = after['reported_tokens']-before['reported_tokens']
        save(path, outcome)
        runtime.db.update_run_status(probe_run, RunStatus.COMPLETED if outcome['status'] == 'RETURNED' else RunStatus.FAILED)
        arms[name] = outcome
        if after['blocked']:
            break
    return arms


def live_context(runtime, folder, *, paired=False):
    from paperpilot_agentic.models import Budget, RunStatus
    from paperpilot_agentic.reader_request_memory import prepare_reader_request_memory
    from paperpilot_agentic.request_state_store import RequestStateStore
    folder = Path(folder)
    if (folder/'context-result.json').exists():
        return
    requests = request_trace()
    save(folder/'context-requests.json', requests)
    parent, root = None, None
    for text in requests:
        rid = runtime.db.create_run(text, Budget(max_llm_calls=12, max_tokens=150000), metadata={
            'engine': 'answer_first_text', 'conversation_id': 'frozen-context-diagnostic',
            **({'parent_run_id': parent['run_id']} if parent else {})})
        root = root or rid
        current = runtime.db.get_run(rid)
        if parent and text == requests[-1]:
            break
        runtime.db.update_run_status(rid, RunStatus.COMPLETED)
        parent = runtime.db.get_run(rid)
    provider = runtime.llm_runtime.model
    before = provider.campaign.snapshot() if paired else None
    try:
        result = prepare_reader_request_memory(runtime, current, parent)
    except Exception as error:
        save(folder/'context-result.json', {'status': 'FAILED', 'error_type': type(error).__name__,
             'run_id': rid, 'paired': paired})
        return
    state = RequestStateStore(runtime.db).load(root, requester_id=rid)
    record = {'result': result, 'state': state,
        'comparison': comparison_record(requests, state),
        'run': runtime.db.get_run(rid), 'independent_semantic_review': 'PENDING'}
    if paired:
        after_compile = provider.campaign.snapshot()
        record['compilation_cost'] = {'actual_requests': after_compile['actual_requests']-before['actual_requests'],
            'reported_tokens': after_compile['reported_tokens']-before['reported_tokens'],
            'requests': request_costs(provider, after_compile['attempts'][len(before['attempts']):])}
        if result and result.get('status') == 'READY' and not after_compile['blocked']:
            record['paired_probe'] = paired_probe(runtime, folder, rid, requests, record['comparison']['work'])
        else:
            record['paired_probe'] = {'status': 'NOT_RUN_MEMORY_NOT_READY'}
        record['total_cost'] = {'actual_requests': provider.campaign.snapshot()['actual_requests']-before['actual_requests'],
            'reported_tokens': provider.campaign.snapshot()['reported_tokens']-before['reported_tokens']}
    save(folder/'context-result.json', record)
