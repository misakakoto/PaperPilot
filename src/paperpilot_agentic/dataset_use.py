"""A narrow source-ownership check for explicit named-dataset use claims.

The source must connect a named dataset to the paper's own experiment or
results. This is a necessary condition only; it never proves entailment.
"""
from copy import deepcopy
import re

from .dataset_roles import NAME, _normalise


VERSION = 'named-dataset-own-use-v1-source-navigation'
CLAIMS = [
    re.compile(rf'(?:论文|本文).{{0,28}}?(?:使用|采用|应用|评估于)了?\s*(?P<name>{NAME})\s*数据集', re.I),
    re.compile(rf'\b(?:paper|study|work)\b.{{0,35}}?\b(?:uses?|used|evaluat\w*\s+on)\s+'
               rf'(?:the\s+)?(?P<name>{NAME})\s+dataset\b', re.I),
]
OWN_RESULT = re.compile(
    r'\b(?:our\s+(?:methods?|models?|frameworks?|results?|experiments?)|'
    r'we\s+(?:use|used|evaluate|evaluated|test|tested|report|reported)|'
    r'proposed\s+(?:method|model|framework)|results?\s+obtained\s+by\s+our)\b|'
    r'(?:我们|本文|本研究)(?:的)?(?:方法|模型|框架|实验|结果|使用|采用|评估)', re.I)
NEGATED_USE = re.compile(
    r'\b(?:did|do|does)\s+not\s+(?:use|evaluate|test|report)\b|'
    r'\b(?:never|without)\s+(?:using|evaluating|testing)\b|'
    r'未(?:使用|采用|评估)|没有(?:使用|采用|评估)', re.I)
EXPERIMENT_SECTION = re.compile(
    r'\b(?:E\s*XPERIMENTS?(?:\s+AND\s+DISCUSSION)?|EXPERIMENTAL\s+(?:RESULTS?|SETUP))\b'
    r'.{0,160}\bDATASETS?\b|实验(?:与分析|结果|设置)?.{0,160}数据集', re.I | re.S)
ENUMERATED_DATASET = re.compile(r'\b\d{1,2}\)\s*[A-Za-z][A-Za-z0-9_+\-]{1,63}')


def _claimed_names(text):
    found = {}
    for pattern in CLAIMS:
        for match in pattern.finditer(text):
            name = match['name']
            found[_normalise(name)] = name
    return found


def _mentions(text, name):
    return any(_normalise(match.group()) == name for match in re.finditer(NAME, text, re.I))


def _owned_use(span, name, by_evidence):
    if not _mentions(span['text'], name):
        return False
    if NEGATED_USE.search(span['text']):
        return False
    # Source spans may contain several sentences about different datasets.
    # Do not transfer an author's result on one dataset to a background
    # description of another dataset in the same span.
    sentences = re.split(r'(?<=[。！？;；])\s*|(?<=[.!?])\s+(?=[A-Z])', span['text'])
    if any(_mentions(sentence, name) and OWN_RESULT.search(sentence) for sentence in sentences):
        return True
    # A numbered dataset entry under this original experimental section is
    # evidence of use; a related-work definition on another page is not.
    source = by_evidence[span['evidence_id']]
    return bool(EXPERIMENT_SECTION.search(source) and ENUMERATED_DATASET.search(span['text']))


def apply_dataset_use_guard(draft, audit, spans, *, mode='enforce'):
    if mode not in {'off', 'shadow', 'enforce'}:
        raise ValueError('Unknown dataset-use mode')
    effective = deepcopy(audit)
    report = {'version': VERSION, 'mode': mode, 'facts': [], 'changed_fact_ids': []}
    if mode == 'off':
        return effective, report
    facts = {fact['id']: fact for fact in draft['facts']}
    by_span = {span['span_id']: span for span in spans}
    by_evidence = {}
    for span in spans:
        by_evidence[span['evidence_id']] = by_evidence.get(span['evidence_id'], '') + '\n' + span['text']
    for assessment in effective['assessments']:
        fact = facts[assessment['fact_id']]
        claimed = _claimed_names(fact['text'])
        if not claimed:
            continue
        selected = [by_span[sid] for sid in assessment.get('span_ids', []) if sid in by_span
                    and by_span[sid]['evidence_id'] in fact['evidence_ids']]
        supported = {name for name in claimed if any(_owned_use(span, name, by_evidence) for span in selected)}
        missing = [label for name, label in claimed.items() if name not in supported]
        missing_names = {_normalise(name) for name in missing}
        candidates = [span for span in spans if any(_owned_use(span, name, by_evidence)
                       for name in missing_names)][:4]
        report['facts'].append({'fact_id': fact['id'], 'claimed_use_datasets': list(claimed.values()),
                                'missing_own_use_bindings': missing,
                                'selected_span_ids': [s['span_id'] for s in selected],
                                'candidate_source_spans': candidates,
                                'status': 'OWN_USE_NOT_ESTABLISHED' if missing else 'NECESSARY_CONDITION_MET'})
        if not missing or mode != 'enforce' or assessment['verdict'] != 'SUPPORTED':
            continue
        assessment.update(verdict='UNCERTAIN', dataset_use_reason='paper_own_use_not_in_selected_source')
        report['changed_fact_ids'].append(fact['id'])
        chinese = bool(re.search(r'[\u4e00-\u9fff]', fact['text']))
        notice = (f"当前引用未证明本文实际使用 {'、'.join(missing)}，需要核对作者实验或结果。"
                  if chinese else
                  f"The selected source does not establish this paper's use of {', '.join(missing)}; check its own experiments or results.")
        effective['missing_aspects'] = list(dict.fromkeys(effective.get('missing_aspects', []) + [notice]))
    if report['changed_fact_ids']:
        effective['coverage'] = 'PARTIAL'
    return effective, report
