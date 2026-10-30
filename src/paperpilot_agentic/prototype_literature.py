"""Bounded bridge to the existing paper and scholarly tools, with durable receipts."""
import copy
from datetime import date
from .episode_runner import ResearchStopped
from .prototype_workspace import digest


POLICY = 'tool-grounded-literature-v1'
CONTEXT_POLICY = 'literal-literature-excerpts-v1'
DISCOVERY_POLICY = 'dated-discovery-v1'
LOCAL_FIRST_POLICY = 'local-first-discovery-v1'
LOCAL_READ_POLICY = 'source-bound-local-reading-v1'
LOCAL_PAGE_POLICY = 'registered-local-pages-v1'
PARENT_READ_POLICY = 'discovered-parent-reading-v1'
CAPABILITIES = {'search_literature', 'read_literature'}


def check_policy(state):
    from .prototype_paper_evidence import SUPPORTED as EVIDENCE_POLICIES
    if state.get('literature_evidence_policy') is not None and state['literature_evidence_policy'] not in EVIDENCE_POLICIES:
        raise ValueError('Unknown paper evidence policy')
    if state.get('literature_parent_read_policy') not in (None, PARENT_READ_POLICY):
        raise ValueError('Unknown parent literature reading policy')
    if state.get('literature_local_page_policy') not in (None, LOCAL_PAGE_POLICY):
        raise ValueError('Unknown local page reading policy')
    if state.get('literature_local_read_policy') not in (None, LOCAL_READ_POLICY):
        raise ValueError('Unknown local literature reading policy')
    if state.get('literature_local_first_policy') not in (None, LOCAL_FIRST_POLICY):
        raise ValueError('Unknown local-first discovery policy')
    if state.get('literature_discovery_policy') not in (None, DISCOVERY_POLICY):
        raise ValueError('Unknown literature discovery policy')
    if state.get('literature_policy') not in (None, POLICY):
        raise ValueError('Unknown prototype literature policy')
    if state.get('literature_context_policy') not in (None, CONTEXT_POLICY):
        raise ValueError('Unknown literature context policy')
    maximum = state.get('literature_excerpt_characters', 4000)
    if type(maximum) is not int or not 1 <= maximum <= 4000:
        raise ValueError('Literature excerpt allowance must be 1..4000 characters')


def project_artifact(identity, artifact, *, max_characters=4000, include_child_ids=False):
    if artifact['kind'] not in CAPABILITIES:
        return artifact
    view = copy.deepcopy(artifact)
    raw = view['content']
    evidence = raw.pop('evidence', [])
    raw['discovery_catalog'] = [
        {'evidence_index': i, 'paper_sha256': row.get('provenance', {}).get('paper_sha256'),
         'title': row.get('title') or row.get('provenance', {}).get('title'),
         'page': row.get('page'), 'source_uri': row.get('source_uri')}
        for i, row in enumerate(evidence)]
    if raw.get('tool') in {'paper.search_papers', 'paper.retrieve_evidence', 'paper.read_sources'}:
        for catalog, row in zip(raw['discovery_catalog'], evidence, strict=True):
            p = row.get('provenance', {})
            if p.get('unit_id'):
                catalog['source_id'] = p['unit_id']
        output = raw.get('output', {})
        if raw.get('tool') == 'paper.search_papers':
            output['papers'] = [{'paper_id': p['paper_id'], 'work_id': p['work_id'],
                'document_version': p['document_version'], 'metadata_gaps': p['metadata_gaps'],
                'fields': {k: {'value': f['value'], 'status': f['status']} for k, f in p['fields'].items()}}
                for p in output.get('papers', [])]
        raw['context_view'] = {'policy': 'paper-evidence-navigation-v2.1', 'raw_artifact_id': identity,
            'read_more': 'inspect_artifact reads the full original receipt including field observations; paper.read_sources view=full reads original locators'}
    if include_child_ids:
        for catalog, row in zip(raw['discovery_catalog'], evidence, strict=True):
            provenance = row.get('provenance', {})
            ids = ([provenance['stable_id']] if provenance.get('retrieval_profile') == 'token_evidence_pilot'
                   and provenance.get('stable_id') else provenance.get('hit_child_ids', []))
            if ids:
                catalog['child_ids'] = list(ids)
    remaining = max_characters
    excerpts = []
    for row in evidence:
        if remaining <= 0:
            break
        excerpt = copy.deepcopy(row)
        text = excerpt.get('content', '')
        excerpt['content'] = text[:min(1200, remaining)]
        excerpt['view_original_characters'] = len(text)
        remaining -= len(excerpt['content'])
        excerpts.append(excerpt)
    raw['evidence'] = excerpts
    # Discovery records can contain long abstracts duplicated in evidence.
    for item in raw.get('output', {}).get('candidates', []):
        abstract = item.get('abstract') or ''
        item['abstract'] = abstract[:300]
        item['view_original_abstract_characters'] = len(abstract)
    raw['context_view'] = {'policy': CONTEXT_POLICY, 'raw_artifact_id': identity,
                           'evidence_character_limit': max_characters,
                           'original_evidence_chunks': len(evidence),
                           'shown_evidence_chunks': len(excerpts), 'literal_excerpts_only': True,
                           'read_more': 'inspect_artifact with artifact_id, offset and limit'}
    return view


def _parent_read_arguments(harness, rid, state, task, content):
    """Only current, directly discovered child IDs can enter the parent reader."""
    metadata = harness.db.get_run(rid)['metadata']
    if (state.get('retrieval_profile') != 'token_evidence_pilot'
            or metadata.get('retrieval_profile') != state.get('retrieval_profile')
            or not state.get('retrieval_identity')
            or metadata.get('retrieval_identity') != state['retrieval_identity']):
        raise ValueError('Parent reading requires the unchanged frozen token retrieval profile')
    if (not {'paper_sha256', 'child_ids'} <= set(content)
            or set(content) - {'paper_sha256', 'child_ids', 'max_tokens', 'view'}):
        raise ValueError('Parent read content requires paper_sha256, child_ids and optional max_tokens/view')
    ids, view, cap = content['child_ids'], content.get('view', 'source_handles'), content.get('max_tokens', 2048)
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 20
            or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids)
            or view not in ('full', 'source_handles') or type(cap) is not int or not 32 <= cap <= 8192):
        raise ValueError('Parent read requires 1..20 distinct child IDs, a registered view and 32..8192 tokens')
    discovered = {}
    for dependency in task['depends_on']:
        prior = state['work']['tasks'][dependency]
        if prior['capability'] != 'search_literature' or prior['status'] != 'DONE':
            continue
        identity = prior['output']
        artifact = state['work']['artifacts'][identity]
        if digest(artifact) != identity:
            raise ValueError('Parent reading discovery artifact identity changed')
        receipt = artifact['content']
        if receipt.get('tool') == 'paper.search' and receipt.get('ok'):
            for evidence in receipt.get('evidence', []):
                provenance = evidence.get('provenance', {})
                if provenance.get('index_identity') == state['retrieval_identity']:
                    discovered[provenance.get('stable_id')] = provenance.get('paper_sha256')
    if any(i not in discovered or discovered[i] != content['paper_sha256'] for i in ids):
        raise ValueError('Parent read must use child IDs from a successful direct local discovery of this paper')
    args = {'child_ids': ids, 'max_tokens': cap, 'view': view}
    normalized = {'paper_sha256': content['paper_sha256'], **args}
    for prior in state['work']['tasks'].values():
        if prior['capability'] == 'read_literature' and prior['status'] == 'DONE':
            receipt = state['work']['artifacts'][prior['output']]['content']
            if receipt.get('ok') and receipt.get('local_read_request') == normalized:
                raise ValueError('Identical parent read already completed; reuse its task dependency')
    return args, normalized


def parent_reading_view(identity, artifact):
    """Navigation for a read decision; source facts are loaded in analysis views."""
    if (artifact['kind'] not in CAPABILITIES
            or artifact['content'].get('tool') not in {'paper.search', 'paper.read_context'}):
        return artifact
    projected = project_artifact(identity, artifact, include_child_ids=True)
    raw = projected['content']
    content = {k: copy.deepcopy(raw[k]) for k in ('ok', 'tool', 'tool_call_id', 'coverage',
        'error', 'local_read_request', 'discovery_catalog') if k in raw}
    if 'output' in raw:
        content['reading_receipt'] = {k: copy.deepcopy(raw['output'][k]) for k in (
            'index_identity', 'view_policy', 'original_read', 'status', 'omitted_parent_ids') if k in raw['output']}
    content['context_view'] = {'policy': 'parent-reading-navigation-v1', 'raw_artifact_id': identity,
        'navigation_only': True, 'read_more': 'inspect_artifact reads the full unchanged tool receipt; analysis receives original quote spans'}
    view = copy.deepcopy(artifact)
    view['content'] = content
    return view


def execute(harness, rid, state, task, content):
    check_policy(state)
    if state.get('literature_policy') != POLICY:
        raise ValueError('Literature tools require the registered literature policy')
    if not isinstance(content, dict):
        raise ValueError('Literature work content must be a JSON object, never narration. '
                         'Read local pages with {paper_sha256,pages}; search passages with '
                         '{paper_sha256,query,top_k}. To inspect stored results use action=inspect_artifact '
                         'with arguments={artifact_id,offset?,limit?}, not a work content string.')
    effective_content = content
    from .prototype_paper_evidence import prepare
    prepared = prepare(state, task, content)
    local_first = state.get('literature_local_first_policy') == LOCAL_FIRST_POLICY
    if prepared is not None:
        tool, args, effective_content = prepared
    elif task['capability'] == 'search_literature':
        required = {'scope', 'query', 'top_k'}
        optional = {'published_from', 'published_to', 'sort', 'require_full_text'} if state.get('literature_discovery_policy') == DISCOVERY_POLICY else set()
        local_first = state.get('literature_local_first_policy') == LOCAL_FIRST_POLICY
        if local_first:
            optional |= {'local_gap', 'queries'}
        unknown = set(content) - required - optional
        if unknown:
            raise ValueError(f'Unknown search fields: {sorted(unknown)}; allowed fields: {sorted(required | optional)}. '
                             'For a basic local search use only scope, query (string), top_k (integer).')
        if (not required <= set(content) or set(content)-required-optional or content['scope'] not in {'local', 'external'}
                or not isinstance(content['query'], str) or not 1 <= len(content['query'].strip()) <= 500
                or type(content['top_k']) is not int or not 1 <= content['top_k'] <= 6):
            raise ValueError('Search content: scope local/external, query (1..500), top_k (1..6)')
        tool = 'paper.search' if content['scope'] == 'local' else 'scholar.search'
        if local_first and content['scope'] == 'external':
            if not isinstance(content.get('local_gap'), str) or not content['local_gap'].strip():
                raise ValueError('External discovery requires an explicit local_gap after local search')
            local_sources = []
            for dependency in task['depends_on']:
                prior = state['work']['tasks'][dependency]
                if prior['capability'] != 'search_literature' or prior['status'] != 'DONE':
                    continue
                artifact_id = prior['output']
                artifact = state['work']['artifacts'][artifact_id]
                if digest(artifact) != artifact_id:
                    raise ValueError('Local discovery artifact identity changed')
                receipt = artifact['content']
                if receipt.get('tool') in {'paper.search', 'paper.search_papers'} and receipt.get('ok'):
                    local_sources.append(artifact_id)
            if not local_sources:
                raise ValueError('External discovery must directly depend on a successful local search receipt')
        elif 'local_gap' in content:
            raise ValueError('local_gap is only valid for external discovery')
        args = {k: content[k] for k in ('query', 'top_k')}
        if 'queries' in content:
            queries = content['queries']
            if (content['scope'] != 'local' or not isinstance(queries, list) or not 1 <= len(queries) <= 3
                    or any(not isinstance(q, str) or not 1 <= len(q.strip()) <= 500 for q in queries)
                    or queries[0] != content['query']
                    or len({q.strip().casefold() for q in queries}) != len(queries)):
                raise ValueError('Local queries must contain 1..3 distinct terms, beginning with query')
            args['queries'] = queries
        supplied = set(content) & (optional - {'local_gap', 'queries'})
        if supplied and content['scope'] != 'external':
            raise ValueError('Publication bounds and scholarly sort require external search')
        for key in ('published_from', 'published_to'):
            if key in content:
                value = content[key]
                if (not isinstance(value, str) or len(value) != 10
                        or date.fromisoformat(value).isoformat() != value):
                    raise ValueError('Publication bounds must be ISO YYYY-MM-DD dates')
                if state.get('research_date') and value > state['research_date']:
                    raise ValueError('Publication bounds cannot exceed the recorded research_date')
        if (content.get('published_from') and content.get('published_to')
                and content['published_from'] > content['published_to']):
            raise ValueError('Publication date range is reversed')
        if 'sort' in content and content['sort'] not in {'relevance', 'newest'}:
            raise ValueError('Scholarly sort must be relevance or newest')
        if 'require_full_text' in content and type(content['require_full_text']) is not bool:
            raise ValueError('require_full_text must be a boolean')
        args.update({k: content[k] for k in supplied})
        if supplied and 'published_to' not in args and state.get('research_date'):
            args['published_to'] = state['research_date']
    elif (state.get('literature_parent_read_policy') == PARENT_READ_POLICY and 'child_ids' in content):
        args, effective_content = _parent_read_arguments(harness, rid, state, task, content)
        tool = 'paper.read_context'
    elif (state.get('literature_local_read_policy') == LOCAL_READ_POLICY
          and 'paper_sha256' in content):
        page_read = state.get('literature_local_page_policy') == LOCAL_PAGE_POLICY and set(content) == {'paper_sha256', 'pages'}
        if ((not page_read and set(content) != {'paper_sha256', 'query', 'top_k'})
                or not isinstance(content['paper_sha256'], str) or len(content['paper_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in content['paper_sha256'])
                or not page_read and (not isinstance(content['query'], str) or not 1 <= len(content['query'].strip()) <= 500
                    or type(content['top_k']) is not int or not 1 <= content['top_k'] <= 6)):
            raise ValueError('Local read requires discovered paper_sha256, query (1..500), top_k (1..6)')
        if page_read and (not isinstance(content['pages'], list) or not 1 <= len(content['pages']) <= 3
                          or any(type(p) is not int or p < 1 for p in content['pages'])
                          or len(set(content['pages'])) != len(content['pages'])):
            raise ValueError('Local pages require 1..3 distinct positive page numbers')
        discoveries = []
        for dependency in task['depends_on']:
            prior = state['work']['tasks'][dependency]
            if prior['capability'] != 'search_literature' or prior['status'] != 'DONE':
                continue
            identity = prior['output']
            artifact = state['work']['artifacts'][identity]
            if digest(artifact) != identity:
                raise ValueError('Local reading discovery artifact identity changed')
            receipt = artifact['content']
            if receipt.get('tool') == 'paper.search' and receipt.get('ok'):
                discoveries.extend(e.get('provenance', {}).get('paper_sha256')
                                   for e in receipt.get('evidence', []))
        if content['paper_sha256'] not in discoveries:
            raise ValueError('Local read must select a paper from a successful direct local literature dependency')
        for prior in state['work']['tasks'].values():
            if prior['capability'] == 'read_literature' and prior['status'] == 'DONE':
                receipt = state['work']['artifacts'][prior['output']]['content']
                if receipt.get('ok') and receipt.get('local_read_request') == content:
                    raise ValueError('Identical local paper query already read; reuse its task dependency')
        tool = 'paper.read_pages' if page_read else 'paper.search'
        args = copy.deepcopy(content) if page_read else {'query': content['query'], 'top_k': content['top_k'],
                                                       'paper_sha256s': [content['paper_sha256']]}
    else:
        if (set(content) != {'url', 'pages'} or not isinstance(content['url'], str)
                or not isinstance(content['pages'], list) or not 1 <= len(content['pages']) <= 3
                or any(type(p) is not int or not 1 <= p <= 100 for p in content['pages'])
                or len(set(content['pages'])) != len(content['pages'])):
            raise ValueError('Read content: discovered url and 1..3 distinct page numbers')
        discoveries = [state['work']['artifacts'][state['work']['tasks'][d]['output']]['content']
                       for d in task['depends_on'] if state['work']['tasks'][d]['capability'] == 'search_literature']
        urls = {item.get('full_text_url') for result in discoveries
                for item in result.get('output', {}).get('candidates', [])}
        if content['url'] not in urls:
            raise ValueError('Read URL must be a full_text_url from a direct search dependency')
        if state.get('literature_context_policy') == CONTEXT_POLICY:
            for prior in state['work']['tasks'].values():
                if prior['capability'] != 'read_literature' or prior['status'] != 'DONE':
                    continue
                receipt = state['work']['artifacts'][prior['output']]['content']
                if (receipt.get('ok') and receipt.get('output', {}).get('url') == content['url']
                        and set(receipt.get('requested_pages', [])) == set(content['pages'])):
                    raise ValueError(f'Identical paper pages already read by {prior["id"]}; reuse that task dependency or choose other pages/papers')
        tool, args = 'scholar.fetch', {'url': content['url']}
    binding = {'task': task['id'], 'revision': state['work']['revision'], 'tool': tool, 'args': args}
    if state.get('literature_local_read_policy') == LOCAL_READ_POLICY:
        binding['inputs'] = sorted(state['work']['tasks'][d]['output'] for d in task['depends_on'])
    if task['capability'] == 'search_literature' and content['scope'] == 'external' and local_first:
        binding['local_discovery_basis'] = {'artifact_ids': local_sources, 'gap': content['local_gap']}
    key = 'prototype-literature:'+rid+':'+digest(binding)
    old = next((c for c in harness.db.list_tool_calls(rid) if c['idempotency_key'] == key), None)
    if old and old['status'] == 'RUNNING':
        raise ResearchStopped('Literature request has an unknown result; inspect its original receipt')
    run = harness.db.get_run(rid)
    if old is None and run['used_tool_calls'] >= run['budget']['max_tool_calls']:
        raise ValueError('Prototype tool budget exhausted')
    call, _ = harness.db.create_step_tool_call(run_id=rid, tool_name=tool, args=args,
        risk_level='READ_ONLY', idempotency_key=key, max_attempts=1)
    outcome = harness.runtime.tools.execute(call)
    harness._guard(rid, state)
    if outcome.result is None:
        raise ResearchStopped('Literature tool returned no known receipt; inspect original request')
    result = outcome.result.to_dict()
    result.update(tool_call_id=call, tool=tool,
                  coverage='BOUNDED_SEARCH_NOT_EXHAUSTIVE; metadata is discovery, not full-text evidence')
    if tool in {'paper.retrieve_evidence', 'paper.read_sources'}:
        result['local_read_request'] = copy.deepcopy(effective_content)
        result['coverage'] = 'BOUNDED_ORIGINAL_UNITS_AND_REPORTED_RESULTS_NOT_SCIENTIFIC_VALIDATION'
        if any(e.get('provenance', {}).get('paper_sha256') not in args['paper_ids'] for e in result.get('evidence', [])):
            result.update(ok=False, evidence=[], error='Evidence escaped selected paper versions', coverage='SCOPE_MISMATCH_NOT_EVIDENCE')
    if task['capability'] == 'search_literature' and content['scope'] == 'external' and local_first:
        result['local_discovery_basis'] = {'artifact_ids': local_sources, 'gap': content['local_gap'],
                                           'gap_status': 'MODEL_ASSESSMENT_NOT_VERIFIED'}
    if task['capability'] == 'read_literature' and tool in {'paper.search', 'paper.read_pages', 'paper.read_context'}:
        result['local_read_request'] = copy.deepcopy(effective_content)
        result['coverage'] = ({'paper.read_pages': 'BOUNDED_LOCAL_PAGES_NOT_FULL_PAPER_READ',
                               'paper.read_context': 'BOUNDED_ORIGINAL_PARENT_PARAGRAPHS_NOT_FULL_PAPER'}
                              .get(tool, 'TARGETED_LOCAL_TEXT_RETRIEVAL_NOT_FULL_PAPER_READ'))
        if any(e.get('provenance', {}).get('paper_sha256') != content['paper_sha256']
               for e in result.get('evidence', [])):
            # Keep the complete original tool receipt for diagnosis, but never
            # expose an out-of-scope fragment as the requested paper's evidence.
            result.update(ok=False, evidence=[], error='Local retriever returned evidence outside the selected paper',
                          coverage='SCOPE_MISMATCH_NOT_EVIDENCE')
    if tool == 'scholar.fetch':
        # Full fetched text stays in the durable tool receipt. Load only requested
        # pages into task memory; never silently turn an abstract into full text.
        result['evidence'] = [e for e in result['evidence'] if e.get('page') in content['pages']]
        result['requested_pages'] = content['pages']
        result['available_pages'] = sorted({e.get('page') for e in outcome.result.evidence if e.get('page') is not None})
    return result


def check_design(work, task, content):
    """Check quote provenance, never scientific entailment or novelty."""
    sources = {work['tasks'][d]['output']: work['artifacts'][work['tasks'][d]['output']]
               for d in task['depends_on'] if work['tasks'][d]['capability'] in CAPABILITIES}
    if not sources:
        raise ValueError('Literature-grounded design requires direct literature task dependencies')
    basis = content.get('literature_basis') if isinstance(content, dict) else None
    if not isinstance(basis, list) or not basis:
        raise ValueError('Design requires literature_basis [{artifact_id, quote, interpretation}]')
    full_text = False
    for index, row in enumerate(basis):
        if (not isinstance(row, dict) or set(row) != {'artifact_id', 'quote', 'interpretation'}
                or not isinstance(row['artifact_id'], str) or row['artifact_id'] not in sources
                or not isinstance(row['quote'], str) or not row['quote'].strip()
                or not isinstance(row['interpretation'], str) or not row['interpretation'].strip()):
            raise ValueError('Literature citation must identify a direct dependency and literal quote')
        evidence = sources[row['artifact_id']]['content'].get('evidence', [])
        matched = [e for e in evidence if row['quote'] in e.get('content', '')]
        if not matched:
            raise ValueError(f'literature_basis[{index}]: quote is absent from original evidence in '
                             f'{row["artifact_id"]}. Use one short contiguous literal substring; '
                             'do not join fragments with ellipses or paraphrase inside quotes. '
                             'Use inspect_artifact to read omitted original text.')
        full_text |= any(e.get('evidence_type') == 'TEXT' for e in matched)
    if not full_text:
        raise ValueError('At least one cited original text excerpt is required; metadata alone is insufficient')
