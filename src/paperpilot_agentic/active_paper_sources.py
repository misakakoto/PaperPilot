"""Navigation/read separation over the existing, version-pinned retrieval backend.

No ingestion or model changes. Handles belong to a persisted run; only actual
read ranges become citable. The source snapshot is independent of mutable caches.
"""
import json
import re
import sqlite3
from pathlib import Path

from .paper_evidence import query_terms, upper
from .paper_catalog import identity
from .paper_structure import digest
from .paper_method_identity import method_hints

PROFILE = 'paper_active_v1'


def handle(mapping, prefix, value):
    for key, old in mapping.items():
        if old == value:
            return key
    key = prefix + str(len(mapping) + 1)
    mapping[key] = value
    return key


class ActivePaperSources:
    def __init__(self, adapter):
        self.adapter = adapter
        self.store = adapter.store
        self.path = Path(self.store.path).resolve()
        self.namespace = self.store.namespace
        self.papers, self.units = self._snapshot()
        self.method_hints = method_hints(self.units)
        from .paper_evidence_ranking import ranker_identity
        self.identity = identity({'sources': self.snapshot_identity(),
            'embedding': getattr(adapter.embedder, 'cache_identity', None),
            'ranking': ranker_identity(adapter.reranker),
            'backend': {k: getattr(adapter, k, None) for k in
                ('rerank_stage', 'max_chunks_per_paper', 'scoped_candidate_k', 'scoped_diversity')},
            'method_hints': self.method_hints,
            'policy': PROFILE, 'code': digest(Path(__file__).read_bytes())})

    def _snapshot(self):
        with sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True) as c:
            c.row_factory = sqlite3.Row
            papers = {r['sha256']: dict(r) for r in c.execute(
                'SELECT * FROM papers WHERE namespace=? AND active=1', (self.namespace,))}
            units = {r['stable_id']: dict(r) for r in c.execute('''SELECT c.*,p.sha256 AS paper_sha
                FROM chunks c JOIN papers p ON p.row_id=c.paper_row_id
                WHERE c.namespace=? AND p.active=1 ORDER BY p.sha256,c.page_start,c.ordinal''', (self.namespace,))}
        if any(digest(u['text'].encode()) != u['text_sha256'] for u in units.values()):
            raise ValueError('Source text no longer matches its registered hash')
        return papers, units

    def snapshot_identity(self):
        papers, units = self._snapshot()
        return identity({'namespace': self.namespace,
            'papers': [(s, p['title'], p['authors_json'], p['chunk_fingerprint']) for s, p in sorted(papers.items())],
            'units': [(s, u['text_sha256'], u['section'], u['page_start'], u['page_end'], u['ordinal'])
                      for s, u in sorted(units.items())]})

    def guard(self, expected):
        if expected != self.identity or self.snapshot_identity() != identity({
            'namespace': self.namespace,
            'papers': [(s, p['title'], p['authors_json'], p['chunk_fingerprint']) for s, p in sorted(self.papers.items())],
            'units': [(s, u['text_sha256'], u['section'], u['page_start'], u['page_end'], u['ordinal'])
                      for s, u in sorted(self.units.items())]}):
            raise ValueError('Registered source/backend version changed')

    def validate_reads(self,state):
        for r in state['read_handles'].values():
            u = self.units.get(r['unit_id'])
            start,end = r['range']
            if (u is None or not 0 <= start < end <= len(u['text']) or u['text'][start:end] != r['content']
                    or u['paper_sha'] != r['paper_sha256'] or u['paper_sha'] not in state['scope']
                    or state['paper_handles'].get(r['paper_id']) != u['paper_sha'] or r['index_identity'] != self.identity):
                raise ValueError('Read source range/version no longer matches its immutable original')

    @staticmethod
    def _scope(state, requested=None):
        allowed = set(state['scope'])
        if requested is None:
            return allowed
        if not isinstance(requested, list) or not 1 <= len(requested) <= 6:
            raise ValueError('Select 1..6 paper handles')
        resolved = {state['paper_handles'].get(p) for p in requested}
        if None in resolved or not resolved <= allowed:
            raise ValueError('Paper handle is unknown or outside this task')
        return resolved

    def search_papers(self, state, query, limit=6):
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 1000 or type(limit) is not int or not 1 <= limit <= 6:
            raise ValueError('Catalogue requires a query and limit 1..6')
        if state['catalogue_searches']:
            if state['supplement_uses']:
                raise ValueError('Only one targeted supplement is allowed across catalogue and body search')
            state['supplement_uses'] += 1
        state['catalogue_searches'] += 1
        hits = self.adapter.search(query_terms(query), top_k=24, paper_sha256s=set(state['scope']))
        scores = {}
        for n, h in enumerate(hits, 1):
            sha = h.get('paper_sha256') or h.get('provenance', {}).get('paper_sha256')
            if sha in state['scope']:
                scores[sha] = scores.get(sha, 0) + 1 / (60+n)
        # Literal first-page mentions are discovery hints, never authorship claims.
        terms = set(re.findall(r'[A-Za-z][A-Za-z0-9_-]{2,}', query.casefold()))
        for sha in state['scope']:
            title = self.papers[sha]['title']
            first = ' '.join(u['text'] for u in self.units.values() if u['paper_sha'] == sha and u['page_start'] == 1)
            exact = sum(bool(re.search(r'(?<!\w)' + re.escape(t) + r'(?!\w)', first.casefold())) for t in terms)
            scores[sha] = scores.get(sha, 0) + .05*exact + .1*sum(t in title.casefold() for t in terms)
            # A self-naming clause is a stronger discovery signal than a passing
            # mention in related work. It still needs actual reading to be cited.
            scores[sha] += sum(1 for h in self.method_hints.get(sha,[])
                if h['name'].casefold() in terms)
        rows = []
        for sha in sorted(scores, key=lambda s: (-scores[s], s))[:limit]:
            p = self.papers[sha]
            pid = handle(state['paper_handles'], 'P', sha)
            rows.append({'paper_id': pid, 'title': p['title'], 'document_version': sha,
                'method_identity_hints': self.method_hints.get(sha,[]),
                'authors': json.loads(p['authors_json']),
                'metadata_gaps': ['year', 'venue', 'doi', 'independently_reviewed_title'],
                'assurance': 'DISCOVERY_ONLY; first-page mentions do not establish the paper\'s own method'})
        return {'papers': rows, 'index_identity': self.identity, 'navigation_only': True}

    def retrieve_evidence(self, state, question, paper_ids, mode='hybrid', facets=None):
        scope = self._scope(state, paper_ids)
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1000 or mode not in {'hybrid', 'literal'}:
            raise ValueError('Use a question and hybrid/literal navigation')
        if facets is not None and (not isinstance(facets, list) or not 1 <= len(facets) <= 3
                or any(not isinstance(f, str) or not 1 <= len(f) <= 150 for f in facets)):
            raise ValueError('Use 1..3 bounded facets')
        if state['searches'] >= 2:
            raise ValueError('Only initial retrieval and one targeted supplement are allowed')
        signature = identity([question, sorted(scope), mode, facets])
        if signature in state['search_signatures']:
            raise ValueError('Identical navigation already exists; reuse handles or read the source')
        if state['searches']:
            if state['supplement_uses']:
                raise ValueError('Only one targeted supplement is allowed across catalogue and body search')
            state['supplement_uses'] += 1
        state['search_signatures'].append(signature)
        state['searches'] += 1
        if mode == 'hybrid':
            queries = [query_terms(question) + ' ' + f for f in (facets or [''])]
            hits = self.adapter.search_many(queries, top_k=12, paper_sha256s=scope)
            ids = [h.get('stable_id') or h.get('provenance', {}).get('stable_id') for h in hits]
        else:
            # Fixed literal lookup inside registered originals; no arbitrary paths or shell.
            ids = [uid for uid, u in self.units.items() if u['paper_sha'] in scope and question.casefold() in u['text'].casefold()][:12]
        cards = []
        terms = re.findall(r'[A-Za-z][A-Za-z0-9_-]{2,}', query_terms(question))
        for uid in dict.fromkeys(ids):
            if uid not in self.units or self.units[uid]['paper_sha'] not in scope:
                raise ValueError('Candidate escaped the frozen source scope')
            u = self.units[uid]
            anchor = next((u['text'].casefold().find(t.casefold()) for t in terms if t.casefold() in u['text'].casefold()), 0)
            start = max(0, anchor-100)
            sh = handle(state['source_handles'], 'S', uid)
            state['unit_papers'][uid] = u['paper_sha']
            ph = handle(state['paper_handles'], 'P', u['paper_sha'])
            cards.append({'source_id': sh, 'paper_id': ph, 'title': self.papers[u['paper_sha']]['title'],
                'section': u['section'], 'page_range': [u['page_start'], u['page_end']],
                'kind': 'TEXT', 'parse_quality': 'LEGACY_PAGE_TEXT_READING_ORDER_UNVERIFIED',
                'preview': u['text'][start:start+280], 'preview_range': [start, min(len(u['text']), start+280)],
                'navigation_only': True, 'read_original': 'paper.read_sources'})
        return {'candidates': cards, 'search_round': state['searches'], 'navigation_only': True,
            'gap': None if cards else 'NOT_LOCATED_IN_SELECTED_PAPERS_NOT_GLOBAL_ABSENCE'}

    def navigation(self, state):
        """Derive navigation from durable handles, never from transient chat history."""
        cards = []
        for sid, uid in state['source_handles'].items():
            u = self.units[uid]
            pid = next(p for p, sha in state['paper_handles'].items() if sha == u['paper_sha'])
            ranges = sorted(r['range'] for r in state['read_handles'].values() if r['unit_id']==uid)
            end = 0
            for start,stop in ranges:
                if start>end:
                    break
                end = max(end,stop)
            cards.append({'source_id': sid, 'paper_id': pid, 'title': self.papers[u['paper_sha']]['title'],
                'page_range': [u['page_start'], u['page_end']], 'navigation_only': True,
                'read_state':'FULL_RANGE_READ' if end==len(u['text']) else 'PARTIAL' if ranges else 'UNREAD',
                'next_read':{'source_ids':[sid],'cursor':end,'neighbors':0} if end<len(u['text']) else None})
        tools = []
        if not state['catalogue_searches'] or not state['supplement_uses']:
            tools.append('paper.search_papers')
        if state['paper_handles'] and (not state['searches'] or not state['supplement_uses']):
            tools.append('paper.retrieve_evidence')
        if cards:
            tools.append('paper.read_sources')
        step = ('ASSESS_READ_EVIDENCE_OR_READ_GAPS' if state['read_handles'] else
                'READ_LOCATED_S_HANDLES' if cards else
                'RETRIEVE_EVIDENCE_FOR_P_HANDLES' if state['paper_handles'] else 'SEARCH_PAPERS')
        return {'next_step': step, 'available_tools': tools, 'known_source_ids': list(state['source_handles']),
            'located_sources': cards[:12], 'omitted_source_ids': [c['source_id'] for c in cards[12:]],
            'rule': 'P identifies a paper; retrieve_evidence returns S handles; only read_sources creates citable R/Q evidence.'}

    def read_sources(self, state, source_ids, neighbors=0, cursor=0):
        if not state['source_handles']:
            raise ValueError('SOURCE_NOT_LOCATED: call paper.retrieve_evidence with existing P handles first; never invent S handles')
        if not isinstance(source_ids, list) or not 1 <= len(source_ids) <= 4 or any(not isinstance(s, str) for s in source_ids):
            raise ValueError('Read 1..4 located S handles as source_ids')
        if any(s not in state['source_handles'] for s in source_ids):
            raise ValueError('UNKNOWN_SOURCE_HANDLE: select an existing S handle from navigation.known_source_ids')
        if (type(neighbors) is not int or not 0 <= neighbors <= 1 or type(cursor) is not int
                or cursor < 0 or cursor and len(source_ids) != 1):
            raise ValueError('INVALID_READ_RANGE: omit optional fields or use integer neighbors=0..1 and cursor>=0; cursor applies to one source')
        ids = [state['source_handles'][s] for s in source_ids]
        selected = list(ids)
        for uid in ids:
            u = self.units[uid]
            if u['paper_sha'] not in state['scope']:
                raise ValueError('Read escaped task scope')
            if neighbors:
                # The frozen legacy parser has no independently validated section
                # hierarchy. Treat its labels as hints; expand only within a page.
                peers = [i for i, v in self.units.items() if v['paper_sha'] == u['paper_sha']
                    and v['page_start'] == u['page_start']]
                n = peers.index(uid)
                selected += peers[max(0,n-1):n+2]
        rows, omitted, continuations = [], [], []
        for uid in dict.fromkeys(selected):
            u = self.units[uid]
            start = cursor if uid == ids[0] else 0
            if start >= len(u['text']):
                raise ValueError('Cursor is outside the original source')
            end = min(len(u['text']), start+1800)
            # Prefer complete sentences without ever claiming a partial range is a full unit.
            if end < len(u['text']):
                boundaries = list(re.finditer(r'[.!?。！？](?:\s+|$)', u['text'][start:end]))
                if boundaries and boundaries[-1].end() > (end-start)//2:
                    end = start+boundaries[-1].end()
            sh = handle(state['source_handles'], 'S', uid)
            state['unit_papers'][uid] = u['paper_sha']
            key = identity([uid, start, end, self.identity])
            rh = next((r for r,v in state['read_handles'].items() if v['range_identity'] == key), 'R'+str(len(state['read_handles'])+1))
            ph = handle(state['paper_handles'], 'P', u['paper_sha'])
            row = {'source_id': rh, 'source_handle': sh, 'paper_id': ph, 'title': self.papers[u['paper_sha']]['title'],
                'paper_sha256': u['paper_sha'], 'unit_id': uid, 'range': [start,end], 'range_identity': key,
                'content': u['text'][start:end], 'unit_complete': start == 0 and end == len(u['text']),
                'source_uri': f"paper://{u['paper_sha']}/{uid}#chars={start}:{end}", 'page': u['page_start'],
                'source_kind': 'TEXT', 'source_quality': {'parse_quality': 'LEGACY_PAGE_TEXT_READING_ORDER_UNVERIFIED'},
                'index_identity': self.identity}
            proposed_next = continuations + ([{'source_ids':[sh], 'cursor':end, 'neighbors':0}] if end<len(u['text']) else [])
            if upper({'passages': rows+[row], 'omitted_sources': list(state['source_handles']), 'continuations': proposed_next,
                    'assurance':'READ_RANGES_ONLY; source origin is not semantic entailment'}) > 9000:
                omitted.append(sh)
                continue
            state['read_handles'][rh] = row
            rows.append(row)
            continuations = proposed_next
        return {'passages': rows, 'omitted_sources': omitted, 'continuations': continuations,
            'assurance': 'READ_RANGES_ONLY; source origin is not semantic entailment'}

    def invoke(self, state, tool, arguments):
        # Normalize only unambiguous spelling/type variants, before normal scope
        # and range validation. The caller's original request remains in receipts.
        if isinstance(arguments, dict):
            arguments = dict(arguments)
            changes = []
            if tool == 'paper.retrieve_evidence':
                for alias, canonical in (('paper_id', 'paper_ids'), ('query', 'question')):
                    if alias in arguments:
                        value = arguments.pop(alias)
                        if alias == 'paper_id':
                            value = [value]
                        if canonical in arguments and arguments[canonical] != value:
                            raise ValueError('Conflicting arguments: '+alias+' and '+canonical)
                        arguments[canonical] = value
                        changes.append(alias+'->'+canonical)
                if 'requirement_ids' in arguments:
                    ids = arguments.pop('requirement_ids')
                    needs = {r['id']:r for r in state['requirements']}
                    if (not isinstance(ids,list) or not 1 <= len(ids) <= 3
                            or any(not isinstance(i,str) or i not in needs for i in ids)):
                        raise ValueError('Select existing requirement_ids; never invent a search question')
                    selected = [needs[i] for i in ids]
                    papers = list(dict.fromkeys(p for r in selected for p in r['paper_ids']))
                    if 'paper_ids' in arguments and not set(arguments['paper_ids']) <= set(papers):
                        raise ValueError('Selected papers do not belong to these requirements')
                    arguments.setdefault('paper_ids',papers)
                    arguments.setdefault('question','; '.join(r['dimension'] for r in selected))
                    changes.append('requirement_ids->registered dimensions and paper scope')
            if tool == 'paper.read_sources':
                for key in ('neighbors','cursor'):
                    if key in arguments and arguments[key] is None:
                        arguments.pop(key)
                        changes.append(key+'=null->default')
                if type(arguments.get('neighbors')) is bool:
                    arguments['neighbors'] = int(arguments['neighbors'])
                    changes.append('neighbors:boolean->0/1')
        else:
            changes = []
        methods = {'paper.search_papers': (self.search_papers, {'query','limit'}),
            'paper.retrieve_evidence': (self.retrieve_evidence, {'question','paper_ids','mode','facets'}),
            'paper.read_sources': (self.read_sources, {'source_ids','neighbors','cursor'})}
        if tool not in methods or not isinstance(arguments, dict) or set(arguments)-methods[tool][1]:
            raise ValueError('Use a registered read-only paper tool and its documented arguments')
        result = methods[tool][0](state, **arguments)
        if changes:
            result['argument_normalization'] = {'changes':changes,'effective_arguments':arguments}
        return result
