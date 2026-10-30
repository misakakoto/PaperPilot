"""Source-bound literature analysis shared with the reader's exact span contract.

Models select IDs and interpret them. Host projection supplies original text;
this establishes provenance, never entailment, novelty or experimental effect.
"""
import copy
import math
import re
from collections import Counter

from .answer_contracts import evidence_spans
from .prototype_workspace import digest


POLICY = 'source-spans-analysis-v1'
LITERATURE = {'search_literature', 'read_literature'}


def check_policy(state):
    if state.get('literature_analysis_policy') not in (None, POLICY):
        raise ValueError('Unknown literature analysis policy')


def check_plan(rows):
    tasks = {t['id']: t for t in rows}
    for task in rows:
        kinds = {tasks[d]['capability'] for d in task['depends_on']}
        if task['capability'] == 'analyze_literature' and not kinds & LITERATURE:
            raise ValueError('Analysis requires direct literature task dependencies')
        if task['capability'] == 'design' and 'analyze_literature' not in kinds:
            raise ValueError('Research design requires a direct analyze_literature dependency')


def spans(identity, artifact):
    if digest(artifact) != identity:
        raise ValueError('Literature artifact identity changed')
    if artifact['kind'] not in LITERATURE:
        raise ValueError('Quote spans require an owned literature artifact')
    result = []
    for index, row in enumerate(artifact['content'].get('evidence', [])):
        for span in evidence_spans([{'evidence_id': str(index), 'content': row.get('content', '')}]):
            location = {'chunk_index': index, 'start': span['start'], 'end': span['end']}
            result.append({'quote_id': 'q_'+digest([identity, location]),
                           'artifact_id': identity, **location, 'text': span['text'],
                           'evidence_type': row.get('evidence_type'),
                           'source_uri': row.get('source_uri'), 'page': row.get('page')})
    return result


def quote_page(identity, artifact, *, offset=0, limit=8, characters=1000, query=None):
    catalog = spans(identity, artifact)
    source_count = len(catalog)
    if query is not None:
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
            raise ValueError('Span query must be a nonempty string of at most 500 characters')
        terms = sorted(set(re.findall(r'\w+', query.casefold())))
        if not terms or len(terms) > 32:
            raise ValueError('Use 1..32 lexical query terms')
        documents = [Counter(re.findall(r'\w+', s['text'].casefold())) for s in catalog]
        average = sum(sum(d.values()) for d in documents)/max(1, len(documents))
        frequency = {t: sum(t in d for d in documents) for t in terms}
        ranked = []
        for index, (span, words) in enumerate(zip(catalog, documents)):
            length = sum(words.values())
            score = sum(math.log(1+(len(catalog)-frequency[t]+0.5)/(frequency[t]+0.5))
                        * words[t]*2.2/(words[t]+1.2*(0.25+0.75*length/max(1,average)))
                        for t in terms if words[t])
            if score > 0:
                ranked.append({**span, 'source_span_index': index, 'navigation_score': score})
        catalog = sorted(ranked, key=lambda s: (-s['navigation_score'], s['source_span_index']))
    if type(offset) is not int or not 0 <= offset <= len(catalog) or type(limit) is not int or not 1 <= limit <= 8:
        raise ValueError('Quote navigation requires a valid span offset and limit 1..8')
    rows, remaining = [], characters
    for span in catalog[offset:offset+limit]:
        if remaining <= 0:
            break
        size = min(len(span['text']), remaining)
        rows.append({**span, 'text': span['text'][:size], 'text_complete': size == len(span['text']),
                     'original_characters': len(span['text'])})
        remaining -= size
    end = offset+len(rows)
    return {'artifact_id': identity, 'spans': rows, 'span_offset': offset, 'total_spans': len(catalog),
            'next_action': {'action': 'inspect_quotes', 'reason': 'Read the next original span page',
                            'arguments': {'artifact_id': identity, 'span_offset': end, 'limit': limit,
                                          **({'query': query} if query is not None else {})}}
                           if end < len(catalog) else None,
            'next_offset': end if end < len(catalog) else None,
            **({'query': query, 'ranking': 'LEXICAL_BM25_NAVIGATION_NOT_FACT_VERIFICATION',
                'source_total_spans': source_count} if query is not None else {}),
            'assurance': 'SOURCE_ORIGIN_ONLY_NOT_ENTAILMENT; partial previews are explicitly marked'}


def inspect_quotes(work, arguments):
    if (not isinstance(arguments, dict) or 'artifact_id' not in arguments
            or set(arguments)-{'artifact_id', 'offset', 'span_offset', 'limit', 'query'}):
        raise ValueError('Expected artifact_id, span_offset (legacy alias offset), and optional limit/query')
    if 'offset' in arguments and 'span_offset' in arguments:
        raise ValueError('Do not supply both offset and span_offset')
    if 'query' in arguments and not isinstance(arguments['query'], str):
        raise ValueError('Span query must be a string')
    identity = arguments['artifact_id']
    if not isinstance(identity, str) or identity not in work['artifacts']:
        raise ValueError('Unknown literature artifact in this task')
    return {'status': 'OK', 'action': 'inspect_quotes', **quote_page(identity, work['artifacts'][identity],
        offset=arguments.get('span_offset', arguments.get('offset', 0)), limit=arguments.get('limit', 8),
        characters=4000, query=arguments.get('query'))}


def project_artifact(identity, artifact, *, max_characters=1000, include_child_ids=False):
    if artifact['kind'] in LITERATURE:
        from .prototype_literature import project_artifact as project_literature
        view = project_literature(identity, artifact, max_characters=max_characters, include_child_ids=include_child_ids)
        # Do not transmit the same source text twice. Quotes and original tool
        # evidence remain available in their respective exact navigation APIs.
        view['content']['evidence'] = [{k: v for k, v in e.items() if k != 'content'}
                                       for e in view['content']['evidence']]
        view['content']['quote_catalog'] = quote_page(identity, artifact, characters=max_characters)
        return view
    if artifact['kind'] in {'analyze_literature', 'design'}:
        view = copy.deepcopy(artifact)
        key = 'findings' if artifact['kind'] == 'analyze_literature' else 'literature_basis'
        for finding in view['content'].get(key, []):
            if artifact['kind'] == 'analyze_literature':
                # The citation navigation target is this analysis, not its raw
                # paper input. Keep origin separate so a model can copy a valid
                # direct dependency reference without silently bypassing analysis.
                finding['artifact_id'] = identity
            if 'quote' in finding:
                finding['quote_characters'] = len(finding.pop('quote'))
                finding['original_text_in_artifact'] = identity
        view['content']['context_view'] = {'policy': POLICY, 'canonical_artifact_id': identity,
                                          'citations_target_this_artifact': artifact['kind'] == 'analyze_literature'}
        return view
    return artifact


def _original_span(work, finding):
    identity = finding['source_artifact_id']
    if identity not in work['artifacts']:
        raise ValueError('Analysis source artifact is missing')
    return next((s for s in spans(identity, work['artifacts'][identity])
                 if s['quote_id'] == finding['quote_id']), None)


def resolve(work, task, content, *, analysis=False):
    field = 'findings' if analysis else 'literature_basis'
    if not isinstance(content, dict) or not isinstance(content.get(field), list) or not content[field]:
        raise ValueError(f'{field} must contain {{artifact_id, quote_id, interpretation}} entries')
    dependencies = {}
    for dep in task['depends_on']:
        current = work['tasks'][dep]
        if current['status'] != 'DONE' or current.get('output') not in work['artifacts']:
            raise ValueError('Citation requires a completed direct dependency')
        dependencies[current['output']] = work['artifacts'][current['output']]
    resolved, full_text = [], False
    for index, row in enumerate(content[field]):
        if (not isinstance(row, dict) or set(row) != {'artifact_id', 'quote_id', 'interpretation'}
                or any(not isinstance(row[k], str) or not row[k].strip() for k in row)):
            raise ValueError('Select artifact_id, quote_id and interpretation; host supplies the quote')
        artifact = dependencies.get(row['artifact_id'])
        if artifact is None or digest(artifact) != row['artifact_id']:
            raise ValueError(f'{field}[{index}]: Citation must select an unchanged direct dependency; '
                             f'allowed artifact IDs: {list(dependencies)}. Literature findings cannot cite CODE artifacts.')
        if analysis:
            span = next((s for s in spans(row['artifact_id'], artifact) if s['quote_id'] == row['quote_id']), None)
        else:
            if artifact['kind'] != 'analyze_literature':
                raise ValueError('Design citations require a completed analyze_literature dependency')
            finding = next((f for f in artifact['content']['findings'] if f['quote_id'] == row['quote_id']), None)
            span = _original_span(work, finding) if finding else None
            if finding and (finding['source_artifact_id'] not in artifact['inputs']
                            or span is None or finding.get('quote') != span['text']):
                raise ValueError('Analysis quote provenance changed')
        if span is None:
            raise ValueError(f'{field}[{index}]: Unknown quote_id for the selected source version; '
                             'copy the exact quote token from current selected pages or inspect_quotes, never invent one')
        resolved.append({**row, 'source_artifact_id': span['artifact_id'], 'quote': span['text'],
                         'source': {k: span[k] for k in ('chunk_index', 'start', 'end', 'evidence_type', 'source_uri', 'page')}})
        full_text |= span['evidence_type'] == 'TEXT'
    if not full_text:
        raise ValueError('At least one cited original TEXT span is required; metadata is discovery only')
    if analysis:
        if (not isinstance(content.get('gaps'), list) or any(not isinstance(x, str) for x in content['gaps'])
                or not isinstance(content.get('directions'), list) or not content['directions']):
            raise ValueError('Analysis requires gaps and directions with hypothesis, mechanism, falsification')
        for direction in content['directions']:
            if (not isinstance(direction, dict) or set(direction) != {'hypothesis', 'mechanism', 'falsification'}
                    or any(not isinstance(v, str) or not v.strip() for v in direction.values())):
                raise ValueError('Each direction requires hypothesis, mechanism and falsification')
    result = copy.deepcopy(content)
    result[field] = resolved
    result['citation_assurance'] = 'SOURCE_ORIGIN_ONLY_NOT_ENTAILMENT_OR_NOVELTY'
    return result
