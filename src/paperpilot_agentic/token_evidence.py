"""Token-bounded search children and immutable, budgeted original parent reads.

This opt-in index never rewrites the page index. Prefixes are navigation only;
child text and parent text remain exact substrings of the frozen parsed source.
"""
import inspect
import json
from pathlib import Path
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock

from paperpilot_retrieval import PaperKnowledgeStore
from paperpilot_retrieval.models import PaperChunk, PaperDocument
from paperpilot_retrieval.store import stable_chunk_id
from .adapters import PaperPilotKnowledgeSearchAdapter
from .paper_structure import PaperTextNavigator, digest, save_json

PROFILE = 'token_evidence_pilot'
VERSION = 'token-evidence-pilot-v1'


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')


@dataclass
class TokenBudget:
    limit: int
    count: object
    identity: dict

    @classmethod
    def from_directory(cls, directory):
        from fastembed.common.preprocessor_utils import load_tokenizer
        directory = Path(directory).resolve()
        files = ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')
        identity = {'directory': str(directory), 'files': {
            name: digest((directory / name).read_bytes()) for name in files},
            'loader_sha256': digest(Path(inspect.getfile(load_tokenizer)).read_bytes())}
        tokenizer, _ = load_tokenizer(directory)
        limit = tokenizer.truncation['max_length']
        tokenizer.no_truncation()
        tokenizer.no_padding()
        identity['limit'] = limit
        return cls(limit, lambda text: len(tokenizer.encode(text).ids), identity)

    @classmethod
    def restore(cls, identity):
        current = cls.from_directory(identity['directory'])
        if current.identity != identity:
            raise ValueError('Tokenizer configuration or loading implementation changed')
        return current


def _prefix(title, section, budget):
    text = f'{title} | {section}'.strip()
    ceiling = min(24, budget.limit // 4)
    while text and budget.count(text) > ceiling:
        text = text[:max(0, len(text) - max(1, len(text) // 8))]
    return text


def split_ranges(text, prefix, budget):
    """Exact contiguous character ranges, with sentence endings preferred."""
    start = 0
    while start < len(text):
        low, high, accepted = start + 1, min(len(text), start + 12000), start
        while low <= high:
            end = (low + high) // 2
            if budget.count(f'{prefix} {text[start:end]}'.strip()) <= budget.limit:
                accepted, low = end, end + 1
            else:
                high = end - 1
        if accepted == start:
            raise ValueError('One source character plus prefix cannot fit the tokenizer limit')
        if accepted < len(text):
            endings = list(re.finditer(r'[.!?。！？](?:\s+|$)', text[start:accepted]))
            if endings and endings[-1].end() >= (accepted - start) // 2:
                accepted = start + endings[-1].end()
        # Counts at boundaries can change with subword merges; never trust an estimate.
        if budget.count(f'{prefix} {text[start:accepted]}'.strip()) > budget.limit:
            raise ValueError('Tokenizer boundary did not satisfy the frozen limit')
        yield start, accepted
        start = accepted


def child_chunks(graph, paper_id, budget):
    if type(budget.limit) is not int or budget.limit < 16:
        raise ValueError('A finite tokenizer limit of at least 16 is required')
    sections = {s['section_id']: s['title'] for s in graph['sections']}
    chunks, children, parents = [], {}, {}
    for paragraph in graph['paragraphs']:
        source = paragraph['text']
        parent = {**paragraph, 'paper_sha256': graph['paper_sha256'],
                  'source_graph_identity': graph['identity'], 'text_sha256': digest(source.encode())}
        parent_id = digest(encoded(parent))
        parents[parent_id] = parent
        prefix = _prefix(graph['title'], sections[paragraph['section_id']], budget)
        for start, end in split_ranges(source, prefix, budget):
            spans = [{**s, 'paragraph_start': max(start, s['text_start']),
                      'paragraph_end': min(end, s['text_end'])}
                     for s in paragraph['spans'] if s['text_start'] < end and s['text_end'] > start]
            if not spans:
                raise ValueError('Child has no original PDF location')
            first, last = min(s['page'] for s in spans), max(s['page'] for s in spans)
            label = f'p{first}' if first == last else f'p{first}-p{last}'
            ordinal = len(chunks) + 1
            text = source[start:end]
            tokens = budget.count(f'{prefix} {text}'.strip())
            chunk = PaperChunk(id=f'{paper_id}:{label}:c{ordinal}', paper_id=paper_id,
                               page_start=first, page_end=last, section=prefix, text=text,
                               token_count_estimate=tokens)
            identity = stable_chunk_id(graph['paper_sha256'], chunk)
            children[identity] = {'paper_sha256': graph['paper_sha256'], 'parent_id': parent_id,
                'start': start, 'end': end, 'text_sha256': digest(text.encode()),
                'source_segments': spans, 'input_tokens': tokens}
            chunks.append(chunk)
    return chunks, children, parents


def build_index(source_root, destination, budget, *, embedder=None):
    source_root, destination = Path(source_root).resolve(), Path(destination).resolve()
    if source_root == destination or source_root in destination.parents:
        raise ValueError('Create the new index outside the frozen source index')
    destination.mkdir(parents=True, exist_ok=False)
    source_manifest = json.loads((source_root / 'index.json').read_bytes())
    source_digest = digest((source_root / 'index.json').read_bytes())
    navigator = PaperTextNavigator(source_root)
    documents, chunks, children, parents = [], [], {}, {}
    for ordinal, entry in enumerate(source_manifest['papers'], 1):
        _, graph = navigator._graph(entry['sha256'])
        document = PaperDocument(id=f'P{ordinal}', sha256=entry['sha256'],
            file_name=entry.get('file_name') or (entry['sha256'] + '.pdf'), title=graph['title'],
            page_count=max(s['page'] for p in graph['paragraphs'] for s in p['spans']),
            processed_pages=0, pages=[])
        parts, links, original_parents = child_chunks(graph, document.id, budget)
        documents.append(document)
        chunks.extend(parts)
        children.update(links)
        parents.update(original_parents)
    embedding_identity = embedder.cache_identity if embedder is not None else None
    store = PaperKnowledgeStore(destination / 'corpus.sqlite3', namespace=VERSION)
    store.sync_documents(documents, chunks, embedder=embedder)
    save_json(destination / 'evidence.json', {'children': children, 'parents': parents})
    manifest = {'version': VERSION, 'namespace': VERSION, 'tokenizer': budget.identity,
                'input_limit': budget.limit, 'prefix_policy': 'navigation-only-first-24-tokens-v1',
                'embedding_identity': embedding_identity, 'source_index_identity': source_digest,
                'papers': [{'sha256': d.sha256, 'title': d.title} for d in documents],
                'children': len(children), 'parents': len(parents),
                'database_sha256': digest((destination / 'corpus.sqlite3').read_bytes()),
                'evidence_sha256': digest((destination / 'evidence.json').read_bytes())}
    save_json(destination / 'index.json', manifest)
    assert digest((source_root / 'index.json').read_bytes()) == source_digest
    return manifest


class _ReadOnlyStore(PaperKnowledgeStore):
    def __init__(self, path):
        self.path, self.namespace = path, VERSION
        self.lexical_query_policy, self._lock = 'content_v1', RLock()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()


class TokenEvidenceAdapter(PaperPilotKnowledgeSearchAdapter):
    def __init__(self, root, legacy, *, budget=None):
        root = Path(root).resolve()
        self.index_identity = digest((root / 'index.json').read_bytes())
        self.manifest = json.loads((root / 'index.json').read_bytes())
        manifest = self.manifest
        if (manifest['version'] != VERSION
                or digest((root / 'corpus.sqlite3').read_bytes()) != manifest['database_sha256']
                or digest((root / 'evidence.json').read_bytes()) != manifest['evidence_sha256']):
            raise ValueError('Frozen token evidence index changed')
        self.budget = budget or TokenBudget.restore(manifest['tokenizer'])
        if self.budget.identity != manifest['tokenizer'] or self.budget.limit != manifest['input_limit']:
            raise ValueError('Tokenizer identity/limit changed')
        self.evidence = json.loads((root / 'evidence.json').read_bytes())
        embedder = getattr(legacy, 'embedder', None) if manifest['embedding_identity'] else None
        if manifest['embedding_identity'] and (embedder is None or embedder.cache_identity != manifest['embedding_identity']):
            raise ValueError('Embedding identity differs from the frozen evidence index')
        self.retrieval_capabilities = {'dense': embedder is not None, 'lexical': True,
                                       'token_limit': self.budget.limit, 'parent_read': True}
        super().__init__(_ReadOnlyStore(root / 'corpus.sqlite3'), embedder=embedder,
            reranker=getattr(legacy, 'reranker', None), rerank_stage=getattr(legacy, 'rerank_stage', 'final'),
            paper_roots=getattr(legacy, 'paper_roots', ()))

    def _hit_to_item(self, hit, *, rank, memory_ids):
        item = super()._hit_to_item(hit, rank=rank, memory_ids=memory_ids)
        link = self.evidence['children'][hit.stable_id]
        item['provenance'].update(index_identity=self.index_identity, retrieval_profile=PROFILE,
            parent_id=link['parent_id'], source_segments=link['source_segments'],
            input_tokens=link['input_tokens'])
        return item

    def read_context(self, child_ids, *, max_tokens=2048, paper_sha256s=None, view='full'):
        if view not in {'full', 'source_handles'}:
            raise ValueError('Unknown parent evidence view')
        if (not isinstance(child_ids, list) or not 1 <= len(child_ids) <= 20
                or any(not isinstance(i, str) or i not in self.evidence['children'] for i in child_ids)
                or type(max_tokens) is not int or not 32 <= max_tokens <= 8192):
            raise ValueError('Read 1..20 owned child IDs with a 32..8192 token budget')
        groups = {}
        for identity in dict.fromkeys(child_ids):
            child = self.evidence['children'][identity]
            if paper_sha256s is not None and child['paper_sha256'] not in paper_sha256s:
                raise ValueError('Parent read escaped requested paper scope')
            groups.setdefault(child['parent_id'], []).append(identity)
        evidence, omitted = [], []
        for identity, matches in groups.items():
            parent = self.evidence['parents'][identity]
            if digest(encoded(parent)) != identity or digest(parent['text'].encode()) != parent['text_sha256']:
                raise ValueError('Original parent identity changed')
            row = {'evidence_type': 'TEXT', 'content': parent['text'],
                   'page': min(s['page'] for s in parent['spans']),
                   'source_uri': f"paper://{parent['paper_sha256']}/parent/{identity}",
                   'provenance': {'paper_sha256': parent['paper_sha256'],
                       'source_graph_identity': parent['source_graph_identity'],
                       'source_segments': parent['spans'], 'parent_id': identity,
                       'hit_child_ids': matches, 'index_identity': self.index_identity,
                       'text_sha256': parent['text_sha256'],
                       'assurance': 'ORIGINAL_PARSED_TEXT_NOT_SCIENTIFIC_VERIFICATION'}}
            if view == 'source_handles':
                # The immutable index retains the full receipt. The same child IDs
                # resolve it through view=full; no generated summary replaces text.
                row['provenance'] = {'paper_sha256': parent['paper_sha256'],
                                     'page_range': [row['page'], max(s['page'] for s in parent['spans'])],
                                     'hit_child_ids': matches}
            if self.budget.count(encoded(evidence + [row]).decode()) > max_tokens:
                omitted.append(identity)
            else:
                evidence.append(row)
        result = {'evidence': evidence, 'context_tokens': self.budget.count(encoded(evidence).decode()),
                'max_tokens': max_tokens, 'index_identity': self.index_identity,
                'omitted_parent_ids': omitted, 'status': 'PARTIAL' if omitted else 'COMPLETE',
                'sizing_scope': 'sorted JSON evidence field including metadata; outer tool framing excluded',
                'original_hits_retained': True}
        if view == 'source_handles':
            result.update(view_policy='source-handles-v1',
                assurance='ORIGINAL_PARSED_TEXT_NOT_SCIENTIFIC_VERIFICATION',
                original_read={'tool': 'paper.read_context', 'view': 'full',
                               'id_field': 'provenance.hit_child_ids',
                               'storage': 'frozen index evidence.json; bound by index_identity'})
        return result
