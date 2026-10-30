"""Source-bearing bibliographic observations, separate from body-text evidence."""
import hashlib
import json
import re

FIELDS = ('title', 'authors', 'abstract', 'year', 'venue', 'doi', 'arxiv_id', 'tasks', 'datasets', 'code_urls')


def identity(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def normalize_doi(value):
    value = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', str(value).strip(), flags=re.I).lower()
    value = re.sub(r'^doi:\s*', '', value).rstrip('.,;')
    return value if re.fullmatch(r'10\.\d{4,9}/\S+', value) else None


def observation(field, value, *, source):
    if field not in FIELDS or not isinstance(source, dict) or source.get('kind') not in {
        'stored_metadata', 'frontmatter', 'publisher', 'user'}:
        raise ValueError('Bibliographic fields require a registered original source; filenames/references are not metadata')
    if not source.get('uri'):
        raise ValueError('Metadata source URI is required')
    if field == 'year' and (type(value) is not int or not 1800 <= value <= 2100):
        raise ValueError('A publication year is an integer, never an invented full date')
    if field == 'doi':
        value = normalize_doi(value)
        if not value:
            raise ValueError('Invalid DOI')
    if field in {'authors', 'tasks', 'datasets', 'code_urls'}:
        if not isinstance(value, list) or not value or any(not isinstance(v, str) or not v.strip() for v in value):
            raise ValueError('Expected a nonempty list of observed strings')
        value = list(dict.fromkeys(v.strip() for v in value))
    elif field != 'year' and (not isinstance(value, str) or not value.strip()):
        raise ValueError('Expected an observed nonempty string')
    return {'field': field, 'value': value, 'source': source}


def record(sha, filename, observations):
    if not re.fullmatch('[a-f0-9]{64}', sha):
        raise ValueError('Paper versions use the original PDF hash')
    fields = {}
    for field in FIELDS:
        rows = [observation(field, r['value'], source=r['source']) for r in observations if r['field'] == field]
        values = {json.dumps(r['value'], sort_keys=True, ensure_ascii=False) for r in rows}
        fields[field] = {'status': 'MISSING' if not rows else 'KNOWN' if len(values) == 1 else 'CONFLICT',
                         'value': rows[0]['value'] if len(values) == 1 else None, 'observations': rows}
    doi, arxiv = fields['doi']['value'], fields['arxiv_id']['value']
    work = ('doi:' + doi if doi else 'arxiv:' + re.sub(r'v\d+$', '', arxiv) if arxiv else 'pdf:' + sha)
    return {'paper_id': sha, 'document_version': sha, 'work_id': work, 'file_name': filename,
            'fields': fields, 'metadata_gaps': [f for f, v in fields.items() if v['status'] != 'KNOWN'],
            'identity': identity({'sha': sha, 'fields': fields})}


def frontmatter_observations(text, sha):
    """Literal identifiers and explicitly labelled dates only; no filename year inference."""
    source = {'kind': 'frontmatter', 'uri': f'paper://{sha}/page/1', 'page': 1}
    rows = []
    # Stop before a references heading if the first page is an unusual reference page.
    text = re.split(r'(?im)^\s*(?:References|Bibliography)\s*$', text)[0]
    abstract = re.search(r'(?is)(?:^|\n)\s*abstract\s*[.：:\-—–]?\s*(.+?)(?=\n\s*(?:Index\s+Terms|Keywords|Key\s+words|(?:1|I)\.?\s*Introduction))', text)
    if abstract:
        rows.append(observation('abstract', abstract[1].strip(), source={**source, 'extraction': 'literal labelled abstract; not a generated summary'}))
    doi = re.search(r'(?:\bDOI\s*:\s*|https?://(?:dx\.)?doi\.org/)(10\.\d{4,9}/[^\s<>]+(?:\s*\n\s*_\d+)?)', text, re.I)
    if doi:
        value = re.sub(r'\s*\n\s*(?=_)', '', doi[1]).rstrip('.,;)')
        rows.append(observation('doi', value, source={**source, 'literal': doi[0]}))
    arxiv = re.search(r'arXiv\s*:\s*(\d{4}\.\d{4,5}(?:v\d+)?)', text, re.I)
    if arxiv:
        rows.append(observation('arxiv_id', arxiv[1], source={**source, 'literal': arxiv[0]}))
    published = re.search(r'(?i)\bpublished(?:\s+online)?\s*(?:on|:)?[^\n]{0,45}\b((?:19|20)\d{2})\b', text)
    if published:
        rows.append(observation('year', int(published[1]), source={**source, 'literal': published[0]}))
    codes = re.findall(r'https?://(?:www\.)?(?:github\.com|gitlab\.com)/[^\s<>]+', text)
    if codes:
        rows.append(observation('code_urls', [u.rstrip('.,;)') for u in codes], source=source))
    return rows


def crossref_observations(message, expected_doi):
    """Resolve an exact DOI; fuzzy title matches must not merge PDF versions."""
    doi = normalize_doi(message.get('DOI'))
    if not doi or doi != normalize_doi(expected_doi):
        raise ValueError('Publisher response does not match the requested DOI')
    source = {'kind': 'publisher', 'uri': 'https://doi.org/' + doi, 'provider': 'crossref'}
    rows = [observation('doi', doi, source=source)]
    for field, key in [('title', 'title'), ('venue', 'container-title')]:
        values = message.get(key) or []
        if values:
            rows.append(observation(field, values[0], source=source))
    authors = [' '.join(str(a.get(k) or '') for k in ('given', 'family')).strip() for a in message.get('author', [])]
    if any(authors):
        rows.append(observation('authors', [a for a in authors if a], source=source))
    dates = (message.get('published') or {}).get('date-parts') or []
    if dates and dates[0] and type(dates[0][0]) is int:
        rows.append(observation('year', dates[0][0], source=source))
    return rows


COMPARISON_KEYS = ('dataset', 'split', 'metric', 'unit', 'pretraining', 'input_size', 'training_protocol')


def compare_reports(reports):
    """Only declare reported protocols compatible; never certify an experiment's truth."""
    missing = {r['id']: [k for k in COMPARISON_KEYS if r.get(k) in (None, '', 'UNKNOWN')] for r in reports}
    conflicts = [k for k in COMPARISON_KEYS if len({json.dumps(r.get(k), sort_keys=True) for r in reports}) > 1]
    comparable = len(reports) >= 2 and not any(missing.values()) and not conflicts
    return {'status': 'REPORTED_PROTOCOLS_MATCH' if comparable else 'NOT_COMPARABLE',
            'missing_conditions': missing, 'different_conditions': conflicts,
            'assurance': 'REPORTED_RESULTS_ONLY_NOT_INDEPENDENT_EXPERIMENT_VALIDATION'}
