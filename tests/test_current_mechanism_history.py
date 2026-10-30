from copy import deepcopy
import json

from paperpilot_agentic.request_obligations import current_request_history
from paperpilot_agentic.answer_pipeline import initialize, AnswerOptions, writer_messages, prepared_messages


def test_old_scientific_tasks_retire_but_constraints_remain_verbatim_in_actual_request():
    history = [{'role': 'initial_request', 'user_request':
                '仅依据《Synthetic A》和《Synthetic B》，对比二者的每种机制；再分别引用两个消融结果。'
                '不得混用论文来源。请用中文。不要启动训练。显存限8GB。'}]
    original = deepcopy(history)
    question = '请只作机制层面的比较，说明哪个设计更直接处理错位及结论边界。'
    active, retired = current_request_history(question, history)
    assert len(retired) == 2
    retained = ''.join(r['user_request'] for r in active)
    for value in ['仅依据《Synthetic A》和《Synthetic B》', '不得混用论文来源', '请用中文',
                  '不要启动训练', '显存限8GB']:
        assert value in retained
    assert '对比二者的每种机制' not in retained
    assert '消融结果' not in retained
    assert history == original
    context = initialize(question, [], object(), AnswerOptions(request_history=history), None)
    messages = prepared_messages(context, writer_messages(context))
    payload = json.loads(messages[-1]['content'])
    assert payload['request_history'] == active
    assert context.report['request_history_scope']['original_history'] == original


def test_persistent_derived_and_ambiguous_mixed_constraints_are_not_rewritten():
    history = [{'role': 'derived_request_state', 'user_request': 'Use paper A, no training, 8GB.'},
               {'role': 'initial_request', 'user_request': '比较时不要引用论文B，保持中文。'}]
    active, retired = current_request_history('Only discuss mechanisms.', history)
    assert not retired
    assert '比较时不要引用论文B' in ''.join(r['user_request'] for r in active)
    assert active[0] == history[0]


def test_normal_followup_does_not_retire_previous_task():
    history = [{'role': 'initial_request', 'user_request': '比较两篇论文并引用实验。'}]
    assert current_request_history('继续回答上一问。', history) == (history, [])
