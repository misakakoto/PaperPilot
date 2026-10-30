"""Build a new immutable evidence index; never migrate the active corpus in place."""
from array import array
import json
from pathlib import Path
import sqlite3

from .paper_catalog import record, observation, identity
from .paper_structure import digest, save_json
from .token_evidence import _prefix, split_ranges, encoded

PROFILE = 'paper_evidence_v2'
VERSION = 'paper-evidence-v2.1'


def build_index(papers, units, destination, budget, *, embedder=None, extra_observations=None, source_identity=None):
    """Input originals are dictionaries; dense windows are private vector carriers."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    records, unit_ids = {}, set()
    units = list(units)
    context_units = {u['id']: u for u in units}
    for paper in papers:
        sha = paper['sha256']
        if sha in records:
            raise ValueError('Duplicate PDF version in the paper catalogue')
        observations = list(paper.get('observations', []))
        source = {'kind': 'stored_metadata', 'uri': f'paper://{sha}/metadata'}
        for field in ('title', 'authors'):
            if paper.get(field):
                observations.append(observation(field, paper[field], source=source))
        observations += (extra_observations or {}).get(sha, [])
        records[sha] = record(sha, paper['file_name'], observations)
    db = destination / 'evidence.sqlite3'
    connection = sqlite3.connect(db)
    try:
        connection.executescript('''
          CREATE TABLE papers(sha TEXT PRIMARY KEY, work_id TEXT NOT NULL, title TEXT NOT NULL, record_json TEXT NOT NULL);
          CREATE VIRTUAL TABLE paper_fts USING fts5(sha UNINDEXED, text);
          CREATE TABLE units(id TEXT PRIMARY KEY, paper_sha TEXT NOT NULL REFERENCES papers(sha), section TEXT,
             kind TEXT NOT NULL, text TEXT NOT NULL, source_json TEXT NOT NULL, role TEXT NOT NULL, object_json TEXT);
          CREATE VIRTUAL TABLE unit_fts USING fts5(id UNINDEXED, text);
          CREATE TABLE windows(id TEXT PRIMARY KEY, unit_id TEXT NOT NULL REFERENCES units(id), start INTEGER,
             end INTEGER, prefix TEXT, text TEXT, tokens INTEGER, vector BLOB);
          CREATE INDEX windows_unit ON windows(unit_id);
          CREATE TABLE reports(id TEXT PRIMARY KEY, unit_id TEXT NOT NULL REFERENCES units(id), paper_sha TEXT NOT NULL, report_json TEXT NOT NULL);
        ''')
        connection.execute('PRAGMA foreign_keys=ON')
        for sha, item in records.items():
            title = item['fields']['title']['value'] or item['file_name']
            text = ' '.join(str(f['value']) for f in item['fields'].values() if f['value'] is not None)
            connection.execute('INSERT INTO papers VALUES(?,?,?,?)', (sha, item['work_id'], title, json.dumps(item, ensure_ascii=False)))
            connection.execute('INSERT INTO paper_fts VALUES(?,?)', (sha, text))
        pending, windows_count, reports_count = [], 0, 0

        def flush():
            if not pending:
                return
            vectors = embedder.embed_documents([f"{w['prefix']} {w['text']}".strip() for w in pending]) if embedder else [None] * len(pending)
            if len(vectors) != len(pending):
                raise ValueError('Embedding vector count changed')
            for window, vector in zip(pending, vectors, strict=True):
                if vector is not None:
                    import math
                    if not vector or not all(math.isfinite(v) for v in vector):
                        raise ValueError('Invalid window vector')
                    norm = math.sqrt(sum(v*v for v in vector))
                    if not norm:
                        raise ValueError('Empty window direction')
                    vector = array('f', (v/norm for v in vector)).tobytes()
                connection.execute('INSERT INTO windows VALUES(?,?,?,?,?,?,?,?)',
                    tuple(window[k] for k in ('id', 'unit_id', 'start', 'end', 'prefix', 'text', 'tokens')) + (vector,))
            pending.clear()

        for unit in units:
            uid, sha = unit['id'], unit['paper_sha256']
            if uid in unit_ids or sha not in records or not unit.get('text') or not unit.get('source'):
                raise ValueError('Units need unique IDs, an owned paper and original text/location')
            unit_ids.add(uid)
            source = unit['source']
            if (not source.get('page_range') or len(source['page_range']) != 2
                    or any(type(p) is not int or p < 1 for p in source['page_range'])
                    or source['page_range'][1] < source['page_range'][0]):
                raise ValueError('Every source needs a page range')
            source = {**source, 'text_sha256': digest(unit['text'].encode()), 'paper_sha256': sha}
            kind = unit.get('kind', 'TEXT')
            if kind not in {'TEXT', 'TABLE', 'FORMULA', 'FIGURE'}:
                raise ValueError('Unknown evidence type')
            section = unit.get('section', '')
            role = ('RELATED_WORK' if 'related' in section.casefold() or 'references' in section.casefold() else 'UNKNOWN')
            connection.execute('INSERT INTO units VALUES(?,?,?,?,?,?,?,?)', (uid, sha, section, kind, unit['text'],
                json.dumps(source, ensure_ascii=False), role, json.dumps(unit.get('object'), ensure_ascii=False)))
            connection.execute('INSERT INTO unit_fts VALUES(?,?)', (uid, section + ' ' + unit['text']))
            title = records[sha]['fields']['title']['value'] or ''
            prefix = _prefix(title, section, budget)
            for start, end in split_ranges(unit['text'], prefix, budget):
                text = unit['text'][start:end]
                count = budget.count(f'{prefix} {text}'.strip())
                if count > budget.limit:
                    raise ValueError('Encoding window exceeds actual tokenizer limit')
                pending.append({'id': identity([uid, start, end, budget.identity]), 'unit_id': uid,
                    'start': start, 'end': end, 'prefix': prefix, 'text': text, 'tokens': count})
                windows_count += 1
                if len(pending) == 64:
                    flush()
            for report in unit.get('reports', []):
                validate_report(report, unit, source, context_units=context_units)
                report = {**report, 'paper_sha256': sha, 'source_unit_id': uid,
                          'status': 'SOURCE_LOCATED_REPORTED_VALUE_NOT_INDEPENDENTLY_VERIFIED'}
                connection.execute('INSERT INTO reports VALUES(?,?,?,?)', (report['id'], uid, sha, json.dumps(report, ensure_ascii=False)))
                reports_count += 1
        flush()
        connection.commit()
    finally:
        connection.close()
    manifest = {'version': VERSION, 'source_identity': source_identity,
        'papers': len(records), 'units': len(unit_ids), 'windows': windows_count, 'reports': reports_count,
        'tokenizer': budget.identity, 'embedding_identity': embedder.cache_identity if embedder else None,
        'database_sha256': digest(db.read_bytes()), 'aggregation': 'max-window-cosine-per-unit-then-rrf60',
        'prefix_policy': 'token-evidence-pilot-v1-same-prefix', 'default_promoted': False}
    save_json(destination / 'index.json', manifest)
    return manifest


def validate_report(report, unit, source, *, context_units=None):
    required = {'id', 'method', 'dataset', 'split', 'metric', 'value', 'unit', 'source_cell'}
    if not isinstance(report, dict) or not required <= set(report) or unit.get('kind') != 'TABLE':
        raise ValueError('Reported numbers require a table and explicit experimental context fields')
    cell = report['source_cell']
    obj = unit.get('object') or {}
    if (not source.get('page_range') or not isinstance(cell, dict) or not {'row', 'column'} <= set(cell)
            or not obj.get('column_headers') or not obj.get('source') or obj.get('parse_status') != 'PARSED'):
        raise ValueError('A reported value requires original table location and headers')
    from .paper_tables import validate_grid
    validate_grid(obj)
    matching = [c for c in obj.get('cells', []) if c['row'] == cell['row'] and c['column'] == cell['column']]
    if len(matching) != 1 or str(report['value']).strip() != matching[0]['text'].strip():
        raise ValueError('Reported value does not match its literal source cell')
    from decimal import Decimal, InvalidOperation
    try:
        if not Decimal(str(report['value'])).is_finite():
            raise ValueError('Reported numeric values must be finite')
    except InvalidOperation as exc:
        raise ValueError('Reported values require an unambiguous literal number') from exc
    # Conditions are source-bearing assertions too. Unknowns stay null; a caller
    # cannot attach an invented dataset/split to an otherwise genuine table cell.
    refs = report.get('context_quotes', {})
    bindings = report.get('context_sources', {})
    fields = {'method', 'dataset', 'split', 'metric', 'unit', 'pretraining', 'input_size', 'training_protocol'}
    if not isinstance(bindings, dict) or set(bindings) - fields or any(report.get(k) is None for k in bindings):
        raise ValueError('Only observed experimental conditions can bind additional source units')
    for key in ('method', 'dataset', 'split', 'metric', 'unit', 'pretraining', 'input_size', 'training_protocol'):
        value = report.get(key)
        if value is None:
            continue
        quote = refs.get(key)
        original = unit
        binding = report.get('context_sources', {}).get(key)
        if binding is not None:
            if not isinstance(binding, dict) or set(binding) != {'source_unit_id', 'quote'}:
                raise ValueError('Experimental conditions need a source unit ID and a literal quote')
            original = (context_units or {}).get(binding['source_unit_id'])
            if original is None or original['paper_sha256'] != unit['paper_sha256']:
                raise ValueError('Experimental condition escaped this paper version')
            quote = binding['quote']
        if (not isinstance(value, str) or not value.strip() or not isinstance(quote, str)
                or value not in quote or quote not in original['text']):
            raise ValueError(f'Reported condition {key} needs a literal quote in its source table unit')


def from_store(store, destination, budget, *, embedder=None, extra_observations=None):
    # Read only. Stable unit IDs preserve the original chunk citation identity.
    with sqlite3.connect(Path(store.path).resolve().as_uri() + '?mode=ro', uri=True) as c:
        c.row_factory = sqlite3.Row
        papers = [dict(r) for r in c.execute('SELECT * FROM papers WHERE namespace=? AND active=1', (store.namespace,))]
        units = [dict(r) for r in c.execute('SELECT c.*,p.sha256 FROM chunks c JOIN papers p ON p.row_id=c.paper_row_id WHERE c.namespace=? AND p.active=1 ORDER BY c.row_id', (store.namespace,))]
    documents = [{'sha256': p['sha256'], 'file_name': p['file_name'], 'title': p['title'],
                  'authors': json.loads(p['authors_json'])} for p in papers]
    blocks = [{'id': u['stable_id'], 'paper_sha256': u['sha256'], 'text': u['text'], 'section': u['section'],
               'source': {'source_uri': f"paper://{u['sha256']}/{u['stable_id']}",
                   'page_range': [u['page_start'], u['page_end']], 'quality': 'IMPORTED_TEXT_WITH_PAGE_LOCATION'}} for u in units]
    return build_index(documents, blocks, destination, budget, embedder=embedder,
                       extra_observations=extra_observations, source_identity={'namespace': store.namespace,
                           'original_units_sha256': digest(encoded(blocks))})
