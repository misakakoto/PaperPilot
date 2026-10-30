"""Three registered read-only tools over the version-pinned evidence module."""
from .models import ToolResult
from .tooling import ToolSpec
import json


def register_paper_evidence_tools(registry, base, select, scope_for_run):
    def invoke(name, context, args):
        metadata = context.db.get_run(context.run_id)['metadata']
        if metadata.get('retrieval_profile') != 'paper_evidence_v2':
            raise ValueError('Paper evidence tools require the opt-in paper_evidence_v2 profile')
        library = select(context, base).library
        artifacts = []
        def persist_trace(identity, trace):
            folder = context.workspace / '.paperpilot_agentic' / 'runs' / context.run_id / 'evidence-traces'
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (identity + '.json')
            body = json.dumps(trace, ensure_ascii=False, sort_keys=True).encode('utf-8')
            if path.exists():
                if path.read_bytes() != body:
                    raise ValueError('Immutable retrieval trace changed')
            else:
                with path.open('xb') as f:
                    f.write(body)
            artifacts.append({'kind': 'paper_retrieval_trace', 'path': str(path), 'sha256': identity})
        library.trace_sink = persist_trace
        scope = scope_for_run(context)
        requested = args.get('paper_ids')
        if requested is not None:
            requested = library._scope(requested)
            if scope is not None and not requested <= scope:
                raise ValueError('Evidence request escaped task paper scope')
        else:
            requested = scope
        kwargs = {k: v for k, v in args.items() if k != 'paper_ids'}
        if name == 'search_papers':
            result = library.search_papers(paper_ids=requested, **kwargs)
            # Metadata is discovery; it is never exposed as supporting body text.
            evidence = [{'evidence_type': 'METADATA', 'content': p['fields']['title']['value'] or p['file_name'],
                'source_uri': f"paper://{p['paper_id']}/metadata", 'confidence': .5,
                'provenance': {'paper_sha256': p['paper_id'], 'title': p['fields']['title']['value'],
                    'work_id': p['work_id'], 'document_version': p['document_version'],
                    'index_identity': library.index_identity, 'assurance': 'CATALOG_NOT_FULL_TEXT_EVIDENCE'}}
                for p in result['papers']]
        else:
            result = getattr(library, name)(paper_ids=requested, **kwargs)
            evidence = result.pop('evidence', [])
        return ToolResult(ok=result.get('status') != 'BLOCKED', output=result, evidence=evidence, artifacts=artifacts,
                          error=result.get('reason'))

    props = {'paper_ids': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 6},
             'budget_upper': {'type': 'integer', 'minimum': 1024, 'maximum': 64000}}
    schemas = {
        'search_papers': ({'query': {'type': 'string'}, 'filters': {'type': 'object'},
                           'limit': {'type': 'integer', 'minimum': 1, 'maximum': 6}, 'paper_ids': props['paper_ids']}, ['query']),
        'retrieve_evidence': ({**props, 'question': {'type': 'string', 'maxLength': 2000},
            'facets': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': 3},
            'supplement': {'type': 'boolean'}}, ['question']),
        'read_sources': ({**props, 'source_ids': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': 24},
                           'view': {'type': 'string', 'enum': ['context', 'full']}}, ['source_ids'])}
    descriptions = {
        'search_papers': 'Find up to six registered paper versions with provenance-bearing metadata and explicit missing/conflicting fields.',
        'retrieve_evidence': 'Retrieve whole original evidence units and source-bound experiment reports for up to six papers and three comparison facets within a conservative context budget. Candidate coverage is not entailment.',
        'read_sources': 'Read immutable original units/tables and full PDF locators by registered source IDs. Explicit scope and omissions; full view preserves parser receipts.'}
    for name, (properties, required) in schemas.items():
        registry.register(ToolSpec(name='paper.' + name, description=descriptions[name] + ' Requires retrieval_profile=paper_evidence_v2; unavailable to legacy runs.',
            handler=lambda context, args, name=name: invoke(name, context, args),
            input_schema={'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False}))
