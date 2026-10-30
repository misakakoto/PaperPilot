"""Lossless value references and question-scoped navigation, not source summaries."""
from copy import deepcopy
from collections import Counter
import json

from .numeric_scope import table_references

VERSION = 'evidence-request-packet-v2-lossless-audit-values'
SHARED_VERSION = 'shared-request-packet-v1-source-row-index'
SHARED_INSTRUCTION = (
    '\nShared request: original_question retains the actual user request; '
    'request_contract.original_question defaults to that actual question. '
    'A history user_request_ref names the SAME verbatim top-level request, retaining its role and order. '
    'cells_source_row identifies rows[row_index].cells in the Named-cell transcription JSON '
    'of that evidence_id (and span_id for audit). Read the literal cells there, including '
    'units and conditions; row_id is still the host selection ID, not a citation span. '
    'Source text and span ownership remain verbatim. Read ALL audit source spans, including '
    'uncited sources. Omitted navigation never proves paper-wide absence. '
    'These references are read-only inputs, not output fields.'
)
INSTRUCTION = (
    '\nRequest packet v2: original_question defaults to question when absent. '
    'request_contract.original_question also defaults to that actual question. '
    'A history user_request_ref names a top-level question field containing the SAME verbatim '
    'request text; preserve its role and position. cells_source_ref points to an exact JSON cells '
    'object inside the cited evidence content (or named source span) by character offsets; read it there. '
    'A source span containing {"text_ref":"t0"} has exactly the verbatim text in '
    'source_texts.t0; its original span ID and evidence owner remain distinct. '
    'allowed_span_ids_ref names an EXACT allowed_span_ids list in allowed_span_sets; '
    'it never permits spans owned by another source. Resolve references before reading '
    'cells_source_ref offsets. Scan ALL source spans for coverage, including uncited ones. '
    'These dictionaries are read-only input: never copy them into the audit output. '
    'Omitted navigation entries do not establish paper-wide absence; no source text is omitted.'
)


def compact_payload(payload, *, paper_scope=None, direct_sources=False):
    result = deepcopy(payload)
    original = result.get('original_question', result.get('question'))
    contract = result.get('request_contract')
    if isinstance(contract, dict) and contract.get('original_question') == original:
        contract.pop('original_question')
    # Keep the public-to-model original_question field for existing readers:
    # its role differs from a retrieval rewrite, even when the strings coincide.
    # Remove the additional contract/history echoes through exact references.
    for row in result.get('request_history', []):
        if not isinstance(row, dict) or 'user_request' not in row:
            continue
        for field in ('question', 'original_question'):
            if field in result and row['user_request'] == result[field]:
                row['user_request_ref'] = field
                row.pop('user_request')
                break
    inventory = result.get('source_inventory_navigation_only')
    if isinstance(inventory, list):
        requirements = (contract or {}).get('table_requirements') or []
        refs = table_references(original or '')
        requested = {r['table'] for r in refs}
        papers = {uri.split('/')[2] for row in result.get('evidence', [])
                  if (uri := str(row.get('source_uri') or '')).startswith('paper://')}
        # Audits use source groups rather than evidence rows: explicit requested
        # source hashes still bound navigation to their actual owners.
        papers.update(t['paper_sha256'] for t in requirements if t.get('paper_sha256'))
        if paper_scope is not None:
            papers = set(paper_scope)
        selected = []
        seen = set()
        for row in inventory:
            labels = table_references(row.get('label', ''))
            if (papers and row.get('paper_sha256') not in papers or
                    requested and not any(r['table'] in requested for r in labels) or
                    (contract or {}).get('mechanism_only')):
                continue
            identity = json.dumps(row, sort_keys=True)
            if identity not in seen:
                selected.append(row)
                seen.add(identity)
        result['source_inventory_navigation_only'] = selected
        if len(selected) != len(inventory):
            result['source_inventory_scope'] = {'omitted_navigation_entries': len(inventory) - len(selected),
                                               'absence_is_not_established': True}
    evidence = {r['evidence_id']: r for r in result.get('evidence', [])}
    groups = {r['evidence_id']: r['spans'] for r in result.get('sources', [])
              if isinstance(r, dict) and isinstance(r.get('spans'), dict)}
    for row in result.get('table_rows', []):
        source = evidence.get(row['evidence_id'])
        if 'cells' not in row:
            continue
        texts = {None: source['content']} if source is not None else groups.get(row['evidence_id'], {})
        if direct_sources:
            matches = []
            for sid, text in texts.items():
                tables = _named_transcriptions(text)
                if len(tables) != 1:
                    continue
                index = row.get('row_index')
                rows = tables[0].get('rows')
                if not isinstance(rows, list) or type(index) is not int or not 0 <= index < len(rows):
                    continue
                original_row = rows[index]
                if (isinstance(original_row, dict) and original_row.get('cells') == row['cells'] and
                        original_row.get('row_label', '') == row.get('row_label', '') and
                        original_row.get('checked_columns') == row.get('checked_columns')):
                    matches.append(sid)
            if len(matches) == 1:
                row['cells_source_row'] = {'evidence_id': row['evidence_id'], 'row_index': row['row_index']}
                if matches[0] is not None:
                    row['cells_source_row']['span_id'] = matches[0]
                row.pop('cells')
            continue
        for separators in (None, (',', ':')):
            cells = json.dumps(row['cells'], ensure_ascii=False, separators=separators)
            matches = [(sid, text) for sid, text in texts.items() if text.count(cells) == 1]
            if len(matches) == 1 and sum(text.count(cells) for text in texts.values()) == 1:
                sid, text = matches[0]
                start = text.index(cells)
                row['cells_source_ref'] = {'evidence_id': row['evidence_id'],
                                           'start': start, 'end': start + len(cells)}
                if sid is not None:
                    row['cells_source_ref']['span_id'] = sid
                row.pop('cells')
                break
    if not direct_sources:
        _share_audit_values(result)
    return result


def _named_transcriptions(text):
    """Only the literal renderer's delimited JSON; ambiguous input stays inline."""
    if not isinstance(text, str):
        return []
    marker = 'Named-cell transcription (each row is independent):'
    parts = text.split(marker)[1:]
    tables = []
    for part in parts:
        try:
            value, _ = json.JSONDecoder().raw_decode(part.lstrip())
        except (ValueError, TypeError):
            # Audit sentence spans can end inside a later caption, after the
            # complete rows array. Read only complete top-level JSON values;
            # never complete a broken row or reconstruct source text.
            value = _complete_table_fields(part.lstrip())
        if not isinstance(value, dict):
            return []
        tables.append(value)
    return tables


def _complete_table_fields(text):
    if not text.startswith('{'):
        return None
    decoder, fields, position = json.JSONDecoder(), {}, 1
    while position < len(text):
        try:
            position += len(text[position:]) - len(text[position:].lstrip())
            key, end = decoder.raw_decode(text[position:])
            if not isinstance(key, str) or key in fields:
                return None
            position += end
            position += len(text[position:]) - len(text[position:].lstrip())
            if text[position:position + 1] != ':':
                return None
            position += 1
            position += len(text[position:]) - len(text[position:].lstrip())
            value, end = decoder.raw_decode(text[position:])
            fields[key] = value
            position += end
            position += len(text[position:]) - len(text[position:].lstrip())
            if text[position:position + 1] != ',':
                break
            position += 1
        except (ValueError, TypeError):
            break
    if isinstance(fields.get('column_headers'), list) and isinstance(fields.get('rows'), list):
        return fields
    return None


def _share_audit_values(payload):
    """Factor only byte-identical values; never merge span IDs or source owners."""
    groups = [row['spans'] for row in payload.get('sources', [])
              if isinstance(row, dict) and isinstance(row.get('spans'), dict)]
    counts = Counter(text for group in groups for text in group.values() if isinstance(text, str))
    shared = {text: f't{i}' for i, text in enumerate(
        text for text, count in counts.items() if count > 1 and len(text.encode('utf-8')) > 64)}
    if shared and 'source_texts' not in payload:
        payload['source_texts'] = {ref: text for text, ref in shared.items()}
        for group in groups:
            for sid, text in group.items():
                if isinstance(text, str) and text in shared:
                    group[sid] = {'text_ref': shared[text]}
    # Repeated large allowlists are host-derived. Sharing an exact list changes
    # neither its order nor membership; short lists cost less when kept inline.
    facts = payload.get('facts', [])
    lists = [tuple(f['allowed_span_ids']) for f in facts
             if isinstance(f, dict) and isinstance(f.get('allowed_span_ids'), list)]
    repeated = Counter(lists)
    sets = {ids: f'a{i}' for i, ids in enumerate(ids for ids, count in repeated.items()
                                               if count > 1 and len(ids) > 8)}
    if sets and 'allowed_span_sets' not in payload:
        payload['allowed_span_sets'] = {ref: list(ids) for ids, ref in sets.items()}
        for fact in facts:
            ids = fact.get('allowed_span_ids')
            if isinstance(ids, list) and tuple(ids) in sets:
                fact['allowed_span_ids_ref'] = sets[tuple(ids)]
                fact.pop('allowed_span_ids')


def restore_value_references(payload):
    """Deterministic equivalence check used in replay reports and regressions."""
    result = deepcopy(payload)
    texts = result.pop('source_texts', {})
    for source in result.get('sources', []):
        for sid, value in source.get('spans', {}).items():
            if isinstance(value, dict) and set(value) == {'text_ref'}:
                source['spans'][sid] = texts[value['text_ref']]
    sets = result.pop('allowed_span_sets', {})
    for fact in result.get('facts', []):
        if 'allowed_span_ids_ref' in fact:
            fact['allowed_span_ids'] = deepcopy(sets[fact.pop('allowed_span_ids_ref')])
    original = result.get('original_question', result.get('question'))
    if 'question' in result:
        result.setdefault('original_question', original)
    if isinstance(result.get('request_contract'), dict):
        result['request_contract'].setdefault('original_question', original)
    for row in result.get('request_history', []):
        if 'user_request_ref' in row:
            row['user_request'] = result[row.pop('user_request_ref')]
    sources = {r['evidence_id']: r for r in result.get('evidence', [])}
    groups = {r['evidence_id']: r['spans'] for r in result.get('sources', [])
              if isinstance(r, dict) and isinstance(r.get('spans'), dict)}
    for row in result.get('table_rows', []):
        if 'cells_source_row' in row:
            ref = row.pop('cells_source_row')
            content = (groups[ref['evidence_id']][ref['span_id']] if ref.get('span_id')
                       else sources[ref['evidence_id']]['content'])
            tables = _named_transcriptions(content)
            if len(tables) != 1:
                raise ValueError('Source row reference needs one literal transcription')
            row['cells'] = deepcopy(tables[0]['rows'][ref['row_index']]['cells'])
        if 'cells_source_ref' in row:
            ref = row.pop('cells_source_ref')
            content = (groups[ref['evidence_id']][ref['span_id']] if ref.get('span_id')
                       else sources[ref['evidence_id']]['content'])
            text = content[ref['start']:ref['end']]
            row['cells'] = json.loads(text)
    return result
