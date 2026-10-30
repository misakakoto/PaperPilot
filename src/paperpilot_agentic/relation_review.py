"""Source-bound review of risky qualifiers, not a general semantic truth oracle.

The narrow detector nominates pipeline-position and broadened-limitation claims.
It never parses sources into a scientific verdict. A model must explicitly review
each nominated relation in the existing audit call. Missing or unbound reviews
withhold only that fact; valid source facts are not rewritten or promoted.
"""
from copy import deepcopy
import re

VERSION = 'qualified-relation-review-v2-processing-order'
_POSITION = re.compile(
    r'(?:编码|解码|融合|训练|推理|聚合)(?:器)?(?:的)?(?:各|每个|每一)?'
    r'(?:阶段|过程中|之前|之后|前|后|中|内)'
    r'|\b(?:before|after|within|inside|outside|during|prior to)\b[^.!?;\n]{0,60}'
    r'\b(?:encod\w*|decod\w*|fusion|training|inference|aggregation|stage|phase)\b', re.I)
_BROAD = re.compile(
    r'(?:传统|常规|现有|先前|既有|其他|所有|全部|一般).{0,18}'
    r'(?:方法|模型|系统|模块|架构|方案|机制)'
    r'|\b(?:conventional|traditional|existing|prior|previous|other|all|general)\s+'
    r'(?:\w+[ -]){0,3}(?:methods?|models?|systems?|modules?|architectures?|approaches?)\b', re.I)
_LIMIT = re.compile(r'缺乏|缺少|不足|局限|无法|不能|不具备|\black\w*|\blimit\w*|cannot|unable', re.I)
_EPISTEMIC = re.compile(r'不能(?:证明|断言|认定|认为)|无法(?:证明|断言|确定)|'
                        r'\b(?:cannot|does not)\s+(?:prove|establish|conclude|determine)\b', re.I)
_ORDER = re.compile(r'随后|然后|最后|先.{0,100}(?:再|后)|\b(?:then|subsequently|finally)\b', re.I)
_MODULE = re.compile(r'模块|块|层|\b(?:[A-Z][A-Z0-9]{1,}|block|module|layer)\b')

POLICY = (
    'Preserve the actor and pipeline position of every mechanism claim. A module '
    'inside each stage of a decoder is not before the decoder. Being before a '
    'particular fusion or subsequent block does not imply being before all decoding '
    'or all state-space processing. A limitation attributed to a named predecessor '
    'does not establish that limitation for conventional methods as a class. Keep '
    'the named owner and stated scope, and separate different owners into different '
    'facts. A comparison may contrast the supported operation and purpose without '
    'adding an unrequested execution stage. Do not invent a stage or class-wide '
    'limitation to make a comparison sound more specific. Check every asserted '
    'processing-order edge independently of a true placement clause: an operation '
    'inside a block is not before entry into that block. Mere co-occurrence in the '
    'same decoder does not establish the claimed sequence.'
)


def review_targets(draft):
    targets = []
    for fact in draft['facts']:
        if fact.get('row_refs'):
            continue  # Owned experimental configurations have their own contract.
        text = fact['text']
        dimensions = []
        if _POSITION.search(text):
            dimensions.append('pipeline_position')
        if _ORDER.search(text) and _MODULE.search(text):
            dimensions.append('processing_order')
        clauses = re.split(r'[。；;.!?]\s*', text)
        if any(_BROAD.search(clause) and _LIMIT.search(clause) and
               not _EPISTEMIC.search(clause) for clause in clauses):
            dimensions.append('limitation_owner')
        for dimension in dimensions:
            targets.append({'target_id': f"{fact['id']}:{dimension}",
                            'fact_id': fact['id'], 'dimension': dimension})
    return targets


def apply_relation_review(draft, audit, spans, *, enabled):
    targets = review_targets(draft) if enabled else []
    report = {'version': VERSION, 'targets': targets, 'changed_fact_ids': [], 'checks': [],
              'semantic_assurance': 'MODEL_REVIEW_NOT_INDEPENDENT_FACTUAL_PROOF'}
    if not targets:
        return audit, report
    effective = deepcopy(audit)
    by_span = {s['span_id']: s for s in spans}
    facts = {f['id']: f for f in draft['facts']}
    assessments = {a['fact_id']: a for a in effective['assessments']}
    raw = audit.get('relation_checks', [])
    checks = raw if isinstance(raw, list) else []
    for target in targets:
        assessment = assessments[target['fact_id']]
        if assessment['verdict'] != 'SUPPORTED':
            continue  # Never upgrade a rejected claim, even with a positive subcheck.
        matching = [c for c in checks if isinstance(c, dict) and
                    c.get('target_id') == target['target_id']]
        status, error = 'UNCERTAIN', 'missing_or_duplicate_relation_check'
        if len(matching) == 1:
            check = matching[0]
            ids = check.get('span_ids')
            verdict = check.get('verdict')
            if (verdict in {'SUPPORTED', 'UNSUPPORTED', 'UNCERTAIN'} and
                    isinstance(check.get('reason'), str) and check['reason'].strip() and
                    isinstance(ids, list) and 1 <= len(ids) <= 12 and
                    len(set(s for s in ids if isinstance(s, str))) == len(ids) and
                    all(isinstance(s, str) and s in by_span and
                        by_span[s]['evidence_id'] in facts[target['fact_id']]['evidence_ids'] for s in ids)):
                status, error = verdict, None
                # Source witnesses remain selected IDs, never model-written quotes.
                report['checks'].append({**target, 'verdict': status, 'span_ids': ids,
                                         'reason': check['reason']})
            else:
                error = 'relation_check_not_bound_to_own_source'
        if status != 'SUPPORTED':
            assessment.update(verdict=status, relation_review_reason=error or 'relation_not_supported')
            report['changed_fact_ids'].append(target['fact_id'])
            if error:
                report['checks'].append({**target, 'verdict': status, 'error': error})
    report['changed_fact_ids'] = list(dict.fromkeys(report['changed_fact_ids']))
    if report['changed_fact_ids']:
        effective['coverage'] = 'PARTIAL'
    return effective, report
