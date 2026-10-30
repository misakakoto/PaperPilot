"""Question-swap controls for scope fidelity; synthetic labels, NOT a benchmark."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from paperpilot_agentic.audit_ablation import digest
from paperpilot_agentic.answer_audit import QUESTION_SCOPE_POLICY as ROLE_POLICY
from .relation_policy_probe import run


def freeze():
    specs = [
        ('illustration',
         'This paper introduces Atlas, an image segmentation system. Its overall test IoU is '
         'reported in Table 2, whose cells are not included here. Figure 3 illustrates a '
         'single demonstration image with IoU 0.94.',
         'The demonstration image has IoU 0.94.',
         'What is their IoU score?', 'What is the demonstration image IoU score?'),
        ('dataset',
         'Cedar is evaluated on River and Forest datasets. The Forest test F1 is 0.82. '
         'River results appear in a table not included in this excerpt.',
         'The Forest test F1 is 0.82.',
         'What F1 does Cedar achieve on River?', 'What F1 does Cedar achieve on Forest?'),
        ('stage',
         'Orchid is a distilled student model. A frozen teacher achieves F1 0.88. '
         'Student results are provided in Table 4, whose cells are not in this excerpt.',
         'The frozen teacher achieves F1 0.88.',
         'What F1 does the distilled student achieve?', 'What F1 does the frozen teacher achieve?'),
        ('component',
         'Maple is an ensemble of a lexical module and a neural module. Evaluated alone, '
         'the lexical module has recall 0.67. The final ensemble recall is reported only '
         'in Table 5, not included here.',
         'The lexical module evaluated alone has recall 0.67.',
         'What recall does the final ensemble achieve?', 'What recall does the lexical module achieve alone?'),
    ]
    cases = []
    for name, source, claim, whole_question, local_question in specs:
        for direct, question in [(False, whole_question), (True, local_question)]:
            cases.append({'id': f'{name}-' + ('requested' if direct else 'different-scope'),
                'question': question, 'evidence': [{'evidence_id': 'source', 'content': source}],
                'draft': {'facts': [{'id': 'f1', 'kind': 'SOURCE_FACT', 'text': claim,
                    'evidence_ids': ['source'], 'depends_on': []}], 'direct_answer_id': 'f1', 'limitations': []},
                'expected': {'f1': 'SUPPORTED'}, 'expected_direct': {'f1': direct},
                'origin': 'assistant-authored question-swap diagnostic, not public benchmark'})
    return {'version': 1, 'cases': cases, 'policy': ROLE_POLICY, 'treatment_name': 'question_scope',
            'base_relation_policy': True, 'planned_requests': len(cases) * 2,
            'input_sha256': digest(cases),
            'scoring': 'assistant-authored entailment and directness labels; no independent ground truth',
            'scope': 'audit-only fixed input question-swap controls; no repair/retry; not end-to-end quality',
            'adoption_rule': 'fewer false-direct decisions, no additional missed-direct or support rejections; zero failures'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute-paid', action='store_true')
    args = parser.parse_args()
    plan = freeze()
    if not args.execute_paid:
        print(json.dumps({'fingerprint': plan['input_sha256'], 'planned_requests': plan['planned_requests']}))
        return
    from paperpilot_agentic.config import RuntimeSettings, resolve_provider_api_key
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    settings = replace(RuntimeSettings.from_env(), provider='qwen', api_key=resolve_provider_api_key('qwen'),
        model='qwen-plus', thinking_enabled=False, temperature=0,
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')
    print(json.dumps(run(plan, OpenAICompatibleLLM(settings, max_retries=0), args.output)))


if __name__ == '__main__':
    main()
