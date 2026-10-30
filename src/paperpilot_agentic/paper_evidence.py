"""Paper discovery, unit retrieval and source reading behind one immutable interface."""
from array import array
from contextlib import contextmanager
import json
import math
from pathlib import Path
import re
import sqlite3

from paperpilot_retrieval.store import _fts_query, _tokenize
from .adapters import PaperPilotKnowledgeSearchAdapter
from .context_window import prompt_upper_bound
from .paper_catalog import compare_reports, identity, normalize_doi
from .paper_evidence_index import PROFILE, VERSION
from .paper_structure import digest
from .token_evidence import TokenBudget, encoded


def upper(value):
    return prompt_upper_bound([{'role': 'tool', 'content': encoded(value).decode()}])


def route_question(question):
    if re.search(r'比较|对比|区别|综述|调研|compare|versus|\bvs\b', question, re.I):
        return 'comparison'
    if re.search(r'指标|消融|准确率|F1|IoU|实验设置|ablation|metric|benchmark', question, re.I):
        return 'experiment'
    if re.search(r'作者|年份|期刊|会议|DOI|代码链接|发表于|题名|哪些.*论文|有哪些.*论文|author|venue|published|paper title', question, re.I):
        return 'catalog'
    return 'method'


def query_terms(query):
    # Navigation aliases only: never alter the user's question or fabricate facts.
    aliases = [('建筑变化检测', 'building change detection'), ('多尺度', 'multi scale'),
               ('融合', 'fusion'), ('消融', 'ablation'), ('时空', 'spatiotemporal'),
               ('差分', 'difference'), ('语义', 'semantic'), ('像素', 'pixel'),
               ('解码', 'decoder'), ('实验条件', 'experimental settings')]
    return query + ' ' + ' '.join(en for zh, en in aliases if zh in query)


def reading_candidates(ranked, route, facet):
    """Soft reading priorities; retain raw ranks and never certify relevance.

    Docling text fragments and captions remain addressable, but should not displace
    available prose in a method explanation. Result and condition facets use
    different views; no section label is used as a hard filter.
    """
    result_facet = bool(re.search(r'result|metric|实验结果|指标|消融', facet, re.I))
    condition_facet = bool(re.search(r'condition|setting|protocol|实验条件|设置', facet, re.I))

    def priority(row):
        text, kind = row['text'], row['kind']
        prose = kind == 'TEXT' and (len(text.split()) >= 12 or
                    len(re.findall(r'[\u4e00-\u9fff]', text)) >= 40)
        if route == 'experiment' or result_facet or condition_facet:
            if condition_facet:
                return 0 if prose else 1 if kind == 'TABLE' else 2
            if kind == 'TABLE':
                obj = json.loads(row['object_json']) or {}
                return 0 if obj.get('structure_status') == 'GRID_VALIDATED_NOT_FACT_VERIFIED' else 1
            return 2 if prose else 3
        return 0 if prose else 1 if kind == 'FORMULA' else 2 if kind == 'TEXT' else 3

    # Stable ordering preserves ranker order within each reading class.
    return sorted(ranked, key=priority)


class PaperEvidence:
    def __init__(self, root, *, embedder=None, reranker=None, expected_identity=None, budget=None):
        self.root = Path(root).resolve()
        self.source_index_identity = digest((self.root / 'index.json').read_bytes())
        from .paper_evidence_ranking import ranker_identity
        ranking = ranker_identity(reranker)
        self.index_identity = identity({'source_index': self.source_index_identity, 'ranking': ranking,
            'reading_policy': 'unit-reading-v2.2', 'reading_code_sha256': digest(Path(__file__).read_bytes())})
        if expected_identity is not None and expected_identity != self.index_identity:
            raise ValueError('Evidence index version changed')
        self.manifest = json.loads((self.root / 'index.json').read_bytes())
        self.path = self.root / 'evidence.sqlite3'
        if (self.manifest['version'] != VERSION
                or digest(self.path.read_bytes()) != self.manifest['database_sha256']):
            raise ValueError('Frozen evidence database changed')
        self._snapshot_stats = {p: (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns)
                                for p in (self.path, self.root / 'index.json')}
        self.budget = budget or TokenBudget.restore(self.manifest['tokenizer'])
        if self.budget.identity != self.manifest['tokenizer']:
            raise ValueError('Evidence tokenizer identity changed')
        self.embedder = embedder if self.manifest['embedding_identity'] else None
        if self.manifest['embedding_identity'] and (embedder is None or embedder.cache_identity != self.manifest['embedding_identity']):
            raise ValueError('Evidence embedding identity changed')
        self.reranker, self._vectors = reranker, None
        self.trace_sink = None
        with self.connect() as c:
            self.papers = {r['sha']: json.loads(r['record_json']) for r in c.execute('SELECT * FROM papers')}

    @contextmanager
    def connect(self):
        for p, before in getattr(self, '_snapshot_stats', {}).items():
            now = p.stat()
            if (now.st_size, now.st_mtime_ns, now.st_ctime_ns) != before:
                raise ValueError('Frozen evidence snapshot changed during reading')
        c = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)
        c.row_factory = sqlite3.Row
        try:
            yield c
        finally:
            c.close()

    def _scope(self, paper_ids):
        if paper_ids is None:
            return set(self.papers)
        if not isinstance(paper_ids, (list, tuple, set)) or any(not isinstance(i, str) for i in paper_ids):
            raise ValueError('Paper scope must contain registered document-version IDs')
        scope = set(paper_ids)
        if scope - self.papers.keys():
            raise ValueError('Paper scope escaped this index')
        return scope

    def paper_metadata(self, paper_sha256s):
        return [{'sha256': i, 'file_name': self.papers[i]['file_name'],
                 'title': self.papers[i]['fields']['title']['value'] or ''} for i in self._scope(paper_sha256s)]

    def _lexical(self, table, query, limit=100, scope=None):
        expression = _fts_query(query_terms(query))
        if not expression:
            return []
        key = 'sha' if table == 'paper_fts' else 'id'
        if scope is not None and not scope:
            return []
        restriction, params = '', [expression]
        if scope is not None:
            marks = ','.join('?' for _ in scope)
            restriction = (f' AND sha IN ({marks})' if table == 'paper_fts' else
                           f' AND id IN (SELECT id FROM units WHERE paper_sha IN ({marks}))')
            params.extend(sorted(scope))
        params.append(limit)
        with self.connect() as c:
            return [r[key] for r in c.execute(f'SELECT {key} FROM {table} WHERE {table} MATCH ?{restriction} ORDER BY bm25({table}),{key} LIMIT ?', params)]

    def search_papers(self, query, *, filters=None, limit=20, paper_ids=None):
        if not isinstance(query, str) or not query.strip() or type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError('Paper search requires a query and limit 1..20')
        filters = filters or {}
        if set(filters) - {'year_from', 'year_to', 'venue', 'doi', 'author'}:
            raise ValueError('Unknown bibliographic filter')
        for key in ('year_from', 'year_to'):
            if key in filters and (type(filters[key]) is not int or not 1800 <= filters[key] <= 2100):
                raise ValueError('Year filters require an observed year')
        if filters.get('year_from', 1800) > filters.get('year_to', 2100):
            raise ValueError('Invalid publication interval')
        if any(not isinstance(filters[k], str) or not filters[k].strip() for k in ('venue', 'doi', 'author') if k in filters):
            raise ValueError('Bibliographic text filters must be nonempty strings')
        if 'doi' in filters:
            if not normalize_doi(filters['doi']):
                raise ValueError('DOI filter must be a complete DOI')
            filters = {**filters, 'doi': normalize_doi(filters['doi'])}
        scope, scores = self._scope(paper_ids), {}
        catalogue = self._lexical('paper_fts', query, scope=scope)
        # Body-assisted paper discovery is labelled, not mistaken for bibliographic proof.
        with self.connect() as c:
            unit_papers = {r['id']: r['paper_sha'] for r in c.execute('SELECT id,paper_sha FROM units')}
        body, seen = [], set()
        for uid in self._lexical('unit_fts', query, scope=scope):
            sha = unit_papers[uid]
            if sha not in seen:
                body.append(sha)
                seen.add(sha)
        for channel, ids in [('catalog', catalogue), ('body_discovery', body)]:
            for rank, sha in enumerate(ids, 1):
                if sha not in scope:
                    continue
                item = scores.setdefault(sha, {'score': 0.0, 'channels': []})
                item['score'] += 1 / (60 + rank)
                item['channels'].append(channel)
        missing_filtered = []
        for sha in list(scores):
            fields = self.papers[sha]['fields']
            conditions = [('year', lambda v: filters.get('year_from', 1800) <= v <= filters.get('year_to', 2100))]
            conditions += [(k, lambda v, k=k: str(filters[k]).casefold() in str(v).casefold())
                           for k in ('venue',) if k in filters]
            if 'doi' in filters:
                conditions.append(('doi', lambda v: v == filters['doi']))
            if 'author' in filters:
                conditions.append(('authors', lambda v: any(str(filters['author']).casefold() in a.casefold() for a in v)))
            for field, check in conditions:
                if field == 'year' and not ({'year_from', 'year_to'} & filters.keys()):
                    continue
                value = fields[field]['value']
                if value is None or fields[field]['status'] != 'KNOWN':
                    missing_filtered.append({'paper_id': sha, 'field': field})
                    scores.pop(sha, None)
                    break
                if not check(value):
                    scores.pop(sha, None)
                    break
        ranked = sorted(scores, key=lambda i: (-scores[i]['score'], i))
        def abstract_view(sha):
            field = self.papers[sha]['fields']['abstract']
            text = field['value'] or ''
            for term in _tokenize(query_terms(query)):
                match = re.search(r'\b' + re.escape(term) + r'\b', text, re.I)
                if match:
                    start, end = max(0, match.start()-100), min(len(text), match.end()+300)
                    return {'text': text[start:end], 'char_range': [start, end], 'complete': start == 0 and end == len(text),
                        'source': field['observations'][0]['source'], 'kind': 'ABSTRACT_NAVIGATION_NOT_FULL_TEXT_EVIDENCE'}
            return None
        return {'papers': [{**self.papers[i], **scores[i], 'abstract_navigation': abstract_view(i)} for i in ranked[:limit]],
            'unresolved_filter_fields': missing_filtered, 'index_identity': self.index_identity,
            'coverage': 'REGISTERED_LOCAL_CATALOG_NOT_GLOBAL_LITERATURE', 'query': query}

    def _units(self, ids=None, paper_ids=None):
        scope = self._scope(paper_ids)
        with self.connect() as c:
            rows = [dict(r) for r in c.execute('SELECT * FROM units') if r['paper_sha'] in scope]
        if ids is not None:
            by_id = {r['id']: r for r in rows}
            if set(ids) - by_id.keys():
                raise ValueError('Source ID is unknown or outside the paper scope')
            rows = [by_id[i] for i in dict.fromkeys(ids)]
        return rows

    def _dense(self, query, scope):
        if self.embedder is None or not scope:
            return {}, {}
        import numpy as np
        if self._vectors is None:
            with self.connect() as c:
                rows = list(c.execute('SELECT w.id,w.unit_id,u.paper_sha,w.vector FROM windows w JOIN units u ON u.id=w.unit_id WHERE w.vector IS NOT NULL ORDER BY w.id'))
            self._vectors = (rows, np.asarray([array('f', r['vector']) for r in rows], dtype=np.float32))
        rows, matrix = self._vectors
        if not rows:
            return {}, {}
        # Original question stays in the packet. Only the encoding query has a bounded view.
        from .token_evidence import split_ranges
        end = next(split_ranges(query, '', self.budget))[1]
        vector = np.asarray(self.embedder.embed_query(query[:end]), dtype=np.float32)
        if not np.isfinite(vector).all() or vector.ndim != 1 or vector.size != matrix.shape[1] or not np.linalg.norm(vector):
            raise ValueError('Invalid query embedding')
        values = matrix @ (vector / np.linalg.norm(vector))
        best, anchors = {}, {}
        for row, value in zip(rows, values, strict=True):
            if row['paper_sha'] in scope and float(value) > best.get(row['unit_id'], -math.inf):
                best[row['unit_id']], anchors[row['unit_id']] = float(value), row['id']
        return best, anchors

    def rank_units(self, query, *, paper_ids=None, limit=24):
        if not isinstance(query, str) or not query.strip() or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('Unit search requires a query and limit 1..100')
        scope = self._scope(paper_ids)
        units = {u['id']: u for u in self._units(paper_ids=scope)}
        lexical = self._lexical('unit_fts', query, 100, scope=scope)
        dense, anchors = self._dense(query_terms(query), scope)
        from .token_evidence import split_ranges
        encoded_query = query_terms(query)
        query_end = next(split_ranges(encoded_query, '', self.budget))[1]
        encoding_view = {'query_char_range': [0, query_end], 'query_characters': len(encoded_query),
            'truncated': query_end < len(encoded_query), 'tokens': self.budget.count(encoded_query[:query_end]),
            'limit': self.budget.limit, 'used': self.embedder is not None}
        scores = {}
        for ids in (lexical, sorted(dense, key=lambda i: (-dense[i], i))[:100]):
            for rank, uid in enumerate(ids, 1):
                scores[uid] = scores.get(uid, 0) + 1 / (60 + rank)
        counts = {'lexical_units': len(lexical), 'dense_units': min(100, len(dense)),
                  'fused_candidates': len(scores), 'rerank_candidates': min(len(scores), max(limit, 24))}
        ordered = sorted(scores, key=lambda i: (-scores[i], i))[:max(limit, 24)]
        with self.connect() as c:
            windows = [dict(r) for r in c.execute('SELECT id,unit_id,text,start,end FROM windows')]
        by_unit = {}
        terms = set(_tokenize(query_terms(query)))
        for w in windows:
            if w['unit_id'] in ordered:
                by_unit.setdefault(w['unit_id'], []).append(w)
        selected_windows = {}
        for uid in ordered:
            candidates = by_unit[uid]
            selected_windows[uid] = next((w for w in candidates if w['id'] == anchors.get(uid)),
                max(candidates, key=lambda w: (len(terms & set(_tokenize(w['text']))), -w['start'])))
        if self.reranker is not None and ordered:
            passages = [selected_windows[i]['text'] for i in ordered]
            # Child windows, never a blindly expanded full parent, enter the ranker.
            from .paper_evidence_ranking import bounded_rerank
            rank_scores, rank_view = bounded_rerank(self.reranker, query, passages)
            if len(rank_scores) != len(ordered) or not all(math.isfinite(v) for v in rank_scores):
                raise ValueError('Invalid evidence reranker result')
            scores = dict(zip(ordered, rank_scores, strict=True))
            ordered.sort(key=lambda i: (-scores[i], i))
        return [{**units[i], 'score': scores[i], 'matched_window': selected_windows[i],
                 'lexical_rank': lexical.index(i)+1 if i in lexical else None,
                 'dense_score': dense.get(i),
                 'encoding_query_view': encoding_view,
                 'candidate_counts': counts,
                 'ranking_view': rank_view if self.reranker is not None else None} for i in ordered[:limit]]

    def _item(self, row):
        source = json.loads(row['source_json'])
        obj = json.loads(row['object_json'])
        title = self.papers[row['paper_sha']]['fields']['title']
        return {'evidence_type': 'EQUATION' if row['kind'] == 'FORMULA' else row['kind'], 'content': row['text'],
            'title': title['value'],
            'source_uri': source.get('source_uri') or f"paper://{row['paper_sha']}/unit/{row['id']}",
            'page': source['page_range'][0], 'provenance': {'paper_sha256': row['paper_sha'],
                'unit_id': row['id'], 'page_range': source['page_range'], 'claim_role': row['role'],
                'title': title['value'], 'title_status': title['status'],
                'parse_quality': source.get('quality'),
                'table_structure': obj.get('structure_status') if isinstance(obj, dict) else None,
                'index_identity': self.index_identity, 'assurance': 'SOURCE_ORIGIN_ONLY_NOT_ENTAILMENT'}}

    def read_sources(self, source_ids, *, paper_ids=None, budget_upper=16000, view='context'):
        if (not isinstance(source_ids, list) or not 1 <= len(source_ids) <= 24
                or any(not isinstance(i, str) for i in source_ids) or len(set(source_ids)) != len(source_ids)
                or type(budget_upper) is not int or not 1024 <= budget_upper <= 64000
                or view not in {'context', 'full'}):
            raise ValueError('Read 1..24 distinct owned IDs, a registered view and a 1024..64000 context budget')
        rows = self._units(source_ids, paper_ids)
        return self._pack(rows, source_ids, budget_upper, view)

    def _pack(self, rows, source_ids, budget_upper, view, extra=None):
        packet = {'evidence': [], 'omitted_source_ids': [], 'index_identity': self.index_identity,
            'status': 'COMPLETE', 'budget_upper': budget_upper,
            'source_index_identity': self.source_index_identity,
            'budget_unit': 'UTF8_BYTE_PROMPT_UPPER_BOUND_INCLUDING_JSON',
            'view': view, 'original_read': {'tool': 'paper.read_sources', 'view': 'full', 'source_ids': source_ids},
            **(extra or {})}
        # Reserve all omissions and the longest status before admitting an original.
        def bound(value):
            return upper({**value, 'omitted_source_ids': source_ids, 'status': 'COMPLETE', 'estimated_upper': budget_upper})
        if bound(packet) > budget_upper:
            return {'status': 'BLOCKED', 'reason': 'SOURCE_NAVIGATION_EXCEEDS_BUDGET', 'evidence': [],
                    'required_upper': bound(packet), 'index_identity': self.index_identity}
        for row in rows:
            item = self._item(row)
            if view == 'full':
                item['provenance']['source'] = json.loads(row['source_json'])
                if row['object_json'] != 'null':
                    item['provenance']['object'] = json.loads(row['object_json'])
            proposed = {**packet, 'evidence': packet['evidence'] + [item]}
            if bound(proposed) <= budget_upper:
                packet['evidence'].append(item)
            else:
                packet['omitted_source_ids'].append(row['id'])
        packet['status'] = 'PARTIAL' if packet['omitted_source_ids'] else 'COMPLETE'
        packet['estimated_upper'] = upper({**packet, 'estimated_upper': budget_upper})
        return packet

    def retrieve_evidence(self, question, *, paper_ids=None, facets=None, budget_upper=16000, supplement=False):
        facets = ['mechanism', 'experimental evidence'] if facets is None else facets
        if (not isinstance(question, str) or not question.strip() or len(question) > 2000
                or not isinstance(facets, list) or not 1 <= len(facets) <= 3
                or any(not isinstance(f, str) or not 1 <= len(f) <= 200 for f in facets)
                or type(supplement) is not bool or type(budget_upper) is not int or not 1024 <= budget_upper <= 64000):
            raise ValueError('Evidence requests need a question, 1..3 facets and an explicit supplement flag')
        selected = (sorted(self._scope(paper_ids)) if paper_ids is not None else
                    [p['paper_id'] for p in self.search_papers(question, limit=6)['papers']])
        if len(selected) > 6:
            raise ValueError('One evidence request reads at most six selected papers')
        candidates, matrix, trace, groups = [], [], [], {}
        route = route_question(question)
        for sha in selected:
            chosen, facet_map, reading = {}, {}, {}
            for facet in facets:
                ranked = self.rank_units(query_terms(question) + ' ' + facet, paper_ids=[sha], limit=8)
                reading[facet] = reading_candidates(ranked, route, facet)
                facet_map[facet] = [r['id'] for r in reading[facet][:1]]
                for r in reading[facet][:1]:
                    chosen.setdefault(r['id'], r)
                trace.append({'paper_id': sha, 'facet': facet, 'candidate_ids': [r['id'] for r in ranked],
                    'counts': ranked[0]['candidate_counts'] if ranked else {},
                    'encoding_query_view': ranked[0]['encoding_query_view'] if ranked else None,
                    'ranking_view': ranked[0]['ranking_view'] if ranked else None})
            cap = 4 if supplement else 2
            # A one-facet method query still needs more than a single fragment.
            # Fill in rounds so all dimensions can contribute before extra context.
            for n in range(8):
                for facet in facets:
                    if n < len(reading[facet]) and len(chosen) < cap:
                        r = reading[facet][n]
                        chosen.setdefault(r['id'], r)
                        if r['id'] not in facet_map[facet]:
                            facet_map[facet].append(r['id'])
            chosen_ids = list(chosen)[:cap]
            groups[sha] = chosen_ids
            candidates.extend(chosen_ids)
            matrix.append({'paper_id': sha, 'work_id': self.papers[sha]['work_id'],
                'facets': {f: [i for i in ids if i in chosen_ids] for f, ids in facet_map.items()},
                'gaps': [f for f, ids in facet_map.items() if not any(i in chosen_ids for i in ids)]})
        if not candidates:
            packet = {'status': 'NO_EVIDENCE', 'evidence': [], 'matrix': matrix, 'question': question,
                'coverage': 'NO_LOCAL_MATCH_NOT_GLOBAL_ABSENCE', 'index_identity': self.index_identity}
            if upper(packet) > budget_upper:
                return {'status': 'BLOCKED', 'reason': 'QUESTION_METADATA_EXCEEDS_BUDGET', 'evidence': [],
                        'required_upper': upper(packet), 'index_identity': self.index_identity}
            return packet
        # Round-robin admission prevents one paper's second unit consuming the
        # budget before the next paper receives its first original.
        candidates = [group[n] for n in range(4 if supplement else 2) for group in groups.values() if n < len(group)]
        extra = {'matrix': matrix, 'route': route, 'question': question,
                 'supplement_round': int(supplement), 'comparison': None, 'reported_results': []}
        with self.connect() as c:
            reports = [json.loads(r['report_json']) for r in c.execute('SELECT report_json,unit_id FROM reports') if r['unit_id'] in candidates]
        all_reports = reports
        reports = []
        q = query_terms(question).casefold()
        def report_priority(report):
            # Citation/"ours" decorations are navigation text, not part of the
            # model's identity. Keep the original method string in the report.
            method = re.sub(r'\[[^\]]+\]|\(\s*ours\s*\)', '', report.get('method') or '', flags=re.I)
            terms = [t for t in _tokenize(method) if len(t) > 1 and any(c.isalpha() for c in t)
                     and t not in {'ours', 'our', 'model', 'the', 'proposed'}]
            method_score = 4 * sum(t.casefold() in q for t in terms) / len(terms) if terms else 0
            return method_score + sum(weight for field, weight in [('metric', 4), ('dataset', 2), ('split', 1)]
                if report.get(field) and str(report[field]).casefold() in q)
        for uid in candidates:
            values = [r for r in all_reports if r['source_unit_id'] == uid]
            values.sort(key=lambda r: (-report_priority(r), r['id']))
            reports.extend(values[:2])
        extra['omitted_reports'] = [{'source_unit_id': uid,
            'count': sum(r['source_unit_id'] == uid and r not in reports for r in all_reports),
            'read_original': 'paper.read_sources view=full'} for uid in candidates
            if any(r['source_unit_id'] == uid and r not in reports for r in all_reports)]
        extra.update(reported_results=reports, comparison=compare_reports(reports))
        extra['diagnostics'] = {'paper_selection': selected, 'ranking_trace_ref': identity(trace),
            'candidate_counts': [len(t['candidate_ids']) for t in trace],
            'selected_units': len(candidates), 'packed_units': len(candidates),
            'trace_persisted': self.trace_sink is not None,
            'coverage_is_candidate_retrieval_not_entailment': True}
        if self.trace_sink is not None:
            self.trace_sink(identity(trace), trace)
        # Coverage gaps describe absent retrieval; omitted_source_ids describes budget loss.
        # No unbudgeted annotations are appended after packing.
        packet = self._pack(self._units(candidates, selected), candidates, budget_upper, 'context', extra)
        if packet['status'] == 'BLOCKED':
            return packet
        included = {e['provenance']['unit_id'] for e in packet['evidence']}
        packet['diagnostics']['packed_units'] = len(included)
        packet['reported_results'] = [r for r in reports if r['source_unit_id'] in included]
        packet['comparison'] = compare_reports(packet['reported_results'])
        packet['estimated_upper'] = upper({**packet, 'estimated_upper': budget_upper})
        # Filtering comparisons can lengthen missing-condition lists. Repack if needed.
        if packet['estimated_upper'] > budget_upper:
            return {'status': 'BLOCKED', 'reason': 'COMPARISON_METADATA_EXCEEDS_BUDGET', 'evidence': [],
                    'required_upper': packet['estimated_upper'], 'index_identity': self.index_identity}
        return packet


class PaperEvidenceAdapter(PaperPilotKnowledgeSearchAdapter):
    def __init__(self, root, legacy, *, expected_identity=None):
        self.library = PaperEvidence(root, embedder=getattr(legacy, 'embedder', None),
            reranker=getattr(legacy, 'reranker', None), expected_identity=expected_identity)
        self.index_identity = self.library.index_identity
        self.manifest = self.library.manifest
        self.retrieval_capabilities = {'lexical': True, 'dense': self.library.embedder is not None,
            'dense_status': 'AVAILABLE' if self.library.embedder is not None else 'NOT_INDEXED',
            'paper_catalog': True, 'versioned_evidence_units': True, 'experimental_reports': True}
        super().__init__(self.library, embedder=self.library.embedder, reranker=self.library.reranker,
                         paper_roots=getattr(legacy, 'paper_roots', ()), rerank_stage='final')

    def search(self, query, top_k=8, *, paper_sha256s=None):
        if route_question(query) == 'catalog':
            return self.search_many([query], top_k=top_k, paper_sha256s=paper_sha256s)
        rows = self.library.rank_units(query, paper_ids=paper_sha256s, limit=min(100, top_k))
        return [{**self.library._item(r), 'paper_sha256': r['paper_sha'], 'stable_id': r['id'],
            'title': self.library.papers[r['paper_sha']]['fields']['title']['value'],
            'file_name': self.library.papers[r['paper_sha']]['file_name'], 'page_end': json.loads(r['source_json'])['page_range'][1],
            'score': r['score'], 'rank': n,
            'provenance': {**self.library._item(r)['provenance'], 'retrieval_profile': PROFILE,
                'matched_window_id': r['matched_window']['id']}} for n, r in enumerate(rows, 1)]

    def search_many(self, queries, top_k=8, *, paper_sha256s=None):
        if not queries or len(queries) > 3:
            raise ValueError('Use 1..3 navigation queries')
        if route_question(queries[0]) == 'catalog':
            result = self.library.search_papers(queries[0], paper_ids=paper_sha256s, limit=min(6, top_k))
            return [{'content': json.dumps(p['fields'], ensure_ascii=False), 'paper_sha256': p['paper_id'],
                'stable_id': 'catalog:' + p['identity'], 'title': p['fields']['title']['value'],
                'file_name': p['file_name'], 'source_uri': f"paper://{p['paper_id']}/metadata",
                'evidence_type': 'METADATA', 'score': p['score'], 'rank': rank,
                'provenance': {'index_identity': self.index_identity, 'retrieval_profile': PROFILE,
                               'assurance': 'CATALOG_NOT_FULL_TEXT_EVIDENCE'}} for rank, p in enumerate(result['papers'], 1)]
        merged = {}
        for query in queries:
            for rank, row in enumerate(self.search(query, top_k=min(100, top_k*3), paper_sha256s=paper_sha256s), 1):
                item = merged.setdefault(row['stable_id'], {**row, 'score': 0.0})
                item['score'] += 1/(60+rank)
        return sorted(merged.values(), key=lambda r: (-r['score'], r['stable_id']))[:top_k]

    def evidence_candidates(self, query, *, paper_sha256s):
        return self.search(query, top_k=24, paper_sha256s=paper_sha256s)

    def resolve_explicit_title_scope(self, queries, *, paper_sha256s=None):
        scope = self.library._scope(paper_sha256s)
        return {sha for sha in scope for q in queries
                if self.library.papers[sha]['fields']['title']['value']
                and self.library.papers[sha]['fields']['title']['value'].casefold() in q.casefold()}
