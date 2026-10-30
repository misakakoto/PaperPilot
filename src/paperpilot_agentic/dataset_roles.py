"""Necessary-condition checks for explicit named-dataset training assertions.

This is deliberately not general NLI. It only recognizes explicit English/Chinese
training assignments with Latin dataset names, and only downgrades model support.
Passing the check is not proof of entailment; unrecognized assertions remain with
the semantic reviewer. Dataset names and paper-specific answers are never built in.
"""
from copy import deepcopy
import re
import unicodedata


VERSION = 'named-dataset-training-role-v2-source-navigation'
NAME = (r'(?!(?:evaluat\w*|test\w*|infer\w*|train\w*|finetun\w*|pretrain\w*)\b)'
        r'[A-Za-z][A-Za-z0-9_+\-]{0,63}')
LABEL = r'(?:\s*(?:data\s*sets?|数据集))?'
NAMES = rf'{NAME}{LABEL}(?:\s*(?:,|、|和|与|及|\band\b|&)\s*{NAME}{LABEL})*'
TRAIN = r'(?:pre[-\s]?train(?:ed|ing)?|train(?:ed|ing)?|fine[-\s]?tun(?:ed|ing))'
PATTERNS = [
    re.compile(rf'\b{TRAIN}\b(?P<link>[^.;。；]{{0,70}}?)\b(?:on|using|with)\s+(?:the\s+)?(?P<names>{NAMES})', re.I),
    re.compile(rf'\b(?:training|pretraining|fine-tuning)\s+(?:data|samples?)\s+(?:from|is|are)\s+(?:the\s+)?(?P<names>{NAMES})', re.I),
    re.compile(rf'(?P<names>{NAMES})\s+(?:(?:is|are|was|were)\s+)?used\s+(?:for|in)\s+{TRAIN}\b', re.I),
    re.compile(rf'(?:在|使用|采用|用)\s*(?P<names>{NAMES})\s*(?:上|来|进行|进行了|用于|来进行|作为训练数据进行){{0,3}}(?:预训练|微调|训练)'),
    re.compile(rf'(?P<names>{NAMES})\s*(?:被)?(?:用作|用于|作为)(?:模型的?)?(?:预训练|微调|训练)(?:数据|集)?'),
]
NEGATION = re.compile(r'\b(?:not|never|without|no)\b|未|不|没有|无需', re.I)
OTHER_STAGE = re.compile(r'\b(?:evaluat\w*|test\w*|infer\w*)\b|测试|评估|推理', re.I)


def _normalise(name):
    return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKC', name).casefold())


def _training_names(text):
    text = unicodedata.normalize('NFKC', text)
    found = {}
    # A contrasting/negative clause must not supply the role of its neighbour.
    for clause in re.split(r'[.;。；!?！？]|\bbut\b|但是|而是', text):
        for pattern in PATTERNS:
            for match in pattern.finditer(clause):
                if NEGATION.search(clause[max(0, match.start() - 24):match.end()]):
                    continue
                if OTHER_STAGE.search(match.groupdict().get('link') or ''):
                    continue
                names = re.sub(r'\bdata\s*sets?\b|数据集', '', match['names'], flags=re.I)
                for name in re.split(r'\s*(?:,|、|和|与|及|\band\b|&)\s*', names):
                    name = name.strip()
                    if re.fullmatch(NAME, name):
                        found[_normalise(name)] = name
    return found


def apply_dataset_role_guard(draft, audit, spans, *, mode='enforce'):
    if mode not in {'off', 'shadow', 'enforce'}:
        raise ValueError('Unknown dataset role mode')
    effective = deepcopy(audit)
    report = {'version': VERSION, 'mode': mode, 'facts': [], 'changed_fact_ids': []}
    if mode == 'off':
        return effective, report
    facts = {fact['id']: fact for fact in draft['facts']}
    by_span = {span['span_id']: span for span in spans}
    for assessment in effective['assessments']:
        fact = facts[assessment['fact_id']]
        claimed = _training_names(fact['text'])
        if not claimed:
            continue
        selected = [by_span[sid] for sid in assessment.get('span_ids', []) if sid in by_span
                    and by_span[sid]['evidence_id'] in fact['evidence_ids']]
        supported = {name for span in selected for name in _training_names(span['text'])}
        missing = [label for name, label in claimed.items() if name not in supported]
        # Source navigation, NOT entailment: another actor or condition can use
        # the same dataset. Only the next Writer/Auditor pass can change support.
        missing_names = {_normalise(name) for name in missing}
        candidates = [span for span in spans if missing_names.intersection(_training_names(span['text']))][:4]
        report['facts'].append({'fact_id': fact['id'], 'claimed_training_datasets': list(claimed.values()),
            'missing_training_bindings': missing, 'selected_span_ids': [s['span_id'] for s in selected],
            'candidate_source_spans': candidates,
            'status': 'ROLE_NOT_ESTABLISHED' if missing else 'NECESSARY_CONDITION_MET'})
        if not missing or mode != 'enforce' or assessment['verdict'] != 'SUPPORTED':
            continue
        assessment.update(verdict='UNCERTAIN', dataset_role_reason='training_role_not_in_selected_source')
        report['changed_fact_ids'].append(fact['id'])
        chinese = bool(re.search(r'[\u4e00-\u9fff]', fact['text']))
        limitation = (f"当前所引原文尚不能确认 {'、'.join(missing)} 的训练数据用途，需要核对训练设置。"
                      if chinese else f"The selected source does not establish the training role of {', '.join(missing)}; training settings need review.")
        effective['missing_aspects'] = list(dict.fromkeys(effective.get('missing_aspects', []) + [limitation]))
    if report['changed_fact_ids']:
        effective['coverage'] = 'PARTIAL'
    return effective, report
