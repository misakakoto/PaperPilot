"""Experimental tables establish measured results, not unbounded mechanism/causal claims."""
from copy import deepcopy
import re

from .structured_extraction import ParsedArtifact

MECHANISM = re.compile(r'(?:将|把|通过|进行|组织为|排列为).{0,35}'
    r'(?:token.{0,20}(?:交错|排列)|通道.{0,20}拼接)|'
    r'(?:concatenates?.{0,30}channels?|channels?.{0,30}concatenated|interleaves?.{0,30}tokens?)', re.I)
UNBOUNDED = re.compile(r'最大化|全局最优|普遍优于|globally optimal|maximi[sz]\w*|universally superior', re.I)
BOUNDED = re.compile(r'当前表|该表|表内|已测|所测|已评估|本实验|'
    r'(?:in|within|among|for) (?:the |these )?(?:table|tested|evaluated|reported)\b', re.I)
CAUSAL = re.compile(r'证明.{0,10}因果|保证.{0,15}(?:提升|更优)|必然.{0,15}(?:提升|更优)|'
                   r'(?:证明|表明|说明|具有|存在).{0,15}协同(?:增益|作用|效应|优于)?|'
                   r'(?:证明|表明|说明).{0,20}显著(?:增强|提升|提高|改善)|'
                   r'显著(?:增强|提升|提高|改善)|'
                   r'\b(?:significant(?:ly)?\s+(?:improv\w*|enhanc\w*|increas\w*)|statistically\s+significant)\b|'
                   r'(?:proves?|demonstrates?|shows?|has).{0,25}synerg(?:istic|y)|'
                   r'(?:proves? causality|guarantees? (?:improvement|superiority))', re.I)
NEGATIVE = re.compile(r'不能|无法|不代表|尚未|cannot|does not|not establish', re.I)
STATISTICAL = re.compile(r'统计(?:学)?显著|statistically\s+significant', re.I)
STAT_SOURCE = re.compile(r'\bp\s*[<≤=]\s*0?\.\d+|p[- ]value|confidence interval|置信区间|显著性检验', re.I)


def source_roles(evidence):
    roles = {}
    for row in evidence:
        payload = (row.get('provenance') or {}).get('structured_extraction')
        if not payload:
            continue
        try:
            item = ParsedArtifact.model_validate(payload)
        except ValueError:
            continue
        if (item.validation_status == 'VALIDATED' and item.parse_status == 'PARSED' and
                row.get('source_uri') == f'paper://{item.source.paper_sha256}/layout/{item.source.object_id}'):
            roles[row['evidence_id']] = {'kind': item.kind, 'label': item.source.label,
                'scope': 'reported experiment configurations' if item.kind == 'table' else 'source region',
                'factual_review': item.factual_review}
    return roles


def apply_experimental_scope(draft, audit, roles, *, evidence=None, spans=None):
    effective = deepcopy(audit)
    assessments = {a['fact_id']: a for a in effective['assessments']}
    changed = []
    for fact in draft['facts']:
        row = assessments[fact['id']]
        ids, text = fact['evidence_ids'], fact['text']
        if row['verdict'] != 'SUPPORTED' or not ids:
            continue
        reason = None
        own_text = [s['text'] for s in spans or [] if s['span_id'] in row.get('span_ids', [])
                    and s['evidence_id'] in ids]
        if spans is None:
            own_text = [e['content'] for e in evidence or [] if e['evidence_id'] in ids]
        if (STATISTICAL.search(text) and not NEGATIVE.search(text) and
                not any(STAT_SOURCE.search(source) for source in own_text)):
            reason = 'statistical_assertion_needs_statistical_source'
        elif not all(roles.get(e, {}).get('kind') == 'table' for e in ids):
            continue
        elif MECHANISM.search(text):
            reason = 'mechanism_needs_method_passage'
        elif any(not NEGATIVE.search(clause) and CAUSAL.search(clause)
                 for clause in re.split(r'[。；;\n]|但是|\bbut\b', text)):
            reason = 'experimental_rows_do_not_establish_causality'
        elif any(not NEGATIVE.search(clause) and UNBOUNDED.search(clause) and not BOUNDED.search(clause)
                 for clause in re.split(r'[。；;\n]|但是|\bbut\b', text)):
            reason = 'conclusion_exceeds_tested_configurations'
        if reason:
            row.update(verdict='UNCERTAIN', experimental_scope_reason=reason)
            changed.append(fact['id'])
    if changed:
        effective['coverage'] = 'PARTIAL'
        effective['missing_aspects'] = list(effective.get('missing_aspects', [])) + [
            '部分机制解释或推断超出所引实验表的支持范围，需要方法原文或限定已测配置。']
    return effective, {'changed_fact_ids': changed, 'policy': 'experimental-source-role-v3'}
