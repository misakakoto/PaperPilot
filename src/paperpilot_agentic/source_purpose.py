"""Project explicit writer parts into independently audited source facts.

This never splits prose, infers a purpose, or grants factual support. Historical
flat claims are unchanged; both new parts cross the ordinary source audit.
"""
from copy import deepcopy
import re

VERSION = 'source-purpose-parts-v3-no-copyable-example'
INSTRUCTION = (
    'For a SOURCE_FACT describing a mechanism, keep text limited to its input, '
    'operation, order and target. If the question also requires its author-stated '
    'purpose, put that separately in optional purpose={"text":"<source-backed purpose '
    'in requested language>","evidence_ids":["<exact purpose source ID>"]}. The host expands it '
    'to a separate SOURCE_FACT with id <parent_id>__purpose, independently audited. '
    'Do not infer achieved benefits from a process description. Omit purpose when '
    'unrequested or unsupported; record requested missing evidence in limitations. '
    'An inference about effects belongs in a separate INFERENCE with explicit '
    'premise IDs. Never attach purpose to TABLE_FACT or INFERENCE. Both expanded '
    'parts count toward the maximum of 12 facts. Every expanded part must use '
    'the requested answer language. Schema placeholders are not evidence and '
    'must never be copied as answer content. The surrounding response contract '
    'owns fact identity; this optional nested field creates no model-owned id. '
    'Do not combine a processing operation with aims/intended purpose in text: '
    'such a mixed statement needs the existing bounded repair, even if it has citations.'
)

_PURPOSE = re.compile(r'旨在|目的是|目标是|为了|\baims? to\b|\bintended to\b|\bin order to\b', re.I)
_OPERATION = re.compile(r'(?:排列|拼接|接收|处理|聚合|估计|变形|对齐)(?!器|模块|方法|网络)'
    r'|\b(?:rearrang\w*|reorder\w*|process\w*|warp\w*|estimat\w*|concatenat\w*)\b'
    r'(?!\s+(?:module|method|network)\b)', re.I)


def process_part_errors(value):
    """Find explicit mixed process/intent markers; never split or approve prose.

    Atomic author-purpose facts remain compatible. Implicit intent/effect
    wording still needs the independent semantic audit and existing guards.
    """
    errors = {}
    for fact in value.get('facts', []) if isinstance(value, dict) else []:
        if not isinstance(fact, dict) or fact.get('kind', 'SOURCE_FACT') != 'SOURCE_FACT':
            continue
        text = fact.get('text')
        if not isinstance(text, str):continue
        marker = _PURPOSE.search(text)
        if marker and _OPERATION.search(text[:marker.start()]):
            errors[fact.get('id')] = ('Mixed process and purpose: keep the cited operation in text; '
                'supply author-stated purpose and its own citations in purpose, or use a separate '
                'atomic inference. Do not delete the requested process or invent a goal.')
    return errors


def project_source_purposes(value, *, reserved_ids=()):
    """Losslessly expand only explicit parts; reject ambiguous identities."""
    if not isinstance(value, dict) or not isinstance(value.get('facts'), list):
        return value, []  # Existing envelope validator owns legacy formats.
    if not any(isinstance(f, dict) and 'purpose' in f for f in value['facts']):
        return value, []
    facts = value['facts']
    ids = [f.get('id') if isinstance(f, dict) else None for f in facts]
    if (any(not isinstance(fid, str) or not fid for fid in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('Purpose projection requires unique explicit fact IDs')
    occupied = set(ids) | set(reserved_ids)
    result = deepcopy(value)
    projected, trace = [], []
    for fact in facts:
        parent = deepcopy(fact)
        if 'purpose' not in parent:
            projected.append(parent)
            continue
        if parent.get('kind', 'SOURCE_FACT') != 'SOURCE_FACT' or parent.get('row_refs'):
            raise ValueError('Purpose is allowed only on prose SOURCE_FACTs')
        purpose = parent.pop('purpose')
        if (not isinstance(purpose, dict) or set(purpose) != {'text', 'evidence_ids'}
                or not isinstance(purpose['text'], str) or not purpose['text'].strip()
                or not isinstance(purpose['evidence_ids'], list) or not purpose['evidence_ids']
                or any(not isinstance(eid, str) or not eid for eid in purpose['evidence_ids'])):
            raise ValueError('Purpose requires explicit text and its own source IDs')
        child_id = parent['id'] + '__purpose'
        if child_id in occupied:
            raise ValueError('Purpose fact identity collides with an existing fact')
        occupied.add(child_id)
        child = {'id': child_id, 'kind': 'SOURCE_FACT', **deepcopy(purpose), 'depends_on': []}
        projected.extend([parent, child])
        trace.append({'version': VERSION, 'parent_fact_id': parent['id'],
                      'purpose_fact_id': child_id, 'original_parts': deepcopy(fact),
                      'support_status': 'REQUIRES_INDEPENDENT_AUDIT'})
    if len(projected) > 12:
        raise ValueError('Expanded source parts exceed the 12-fact limit')
    result['facts'] = projected
    return result, trace
