import hashlib
import json
from urllib.parse import parse_qs, urlsplit

import pytest

from paperpilot_agentic.scholarly import ScholarlySearchAdapter


def stubbed(tmp_path, monkeypatch):
    adapter = ScholarlySearchAdapter(tmp_path)
    calls = []

    def provider(name):
        def search(query, limit, **options):
            calls.append((name, query, limit, options))
            return [{'source_id': name+str(i), 'title': 'Paper '+str(i),
                     'publication_date': '2025-01-'+str(i+1).zfill(2), 'year': 2025,
                     'full_text_url': 'https://arxiv.org/pdf/2501.00001'} for i in range(10)]
        return search

    for name in ('openalex', 'semantic_scholar', 'arxiv', 'crossref'):
        monkeypatch.setattr(adapter, '_'+name, provider(name))
    return adapter, calls


def test_cache_identity_includes_count_dates_sort_and_full_text(tmp_path, monkeypatch):
    adapter, calls = stubbed(tmp_path, monkeypatch)
    requests = [{}, {'top_k': 2}, {'published_from': '2025-01-01'},
                {'published_to': '2026-10-24'}, {'sort': 'newest'}, {'require_full_text': True}]
    for kwargs in requests:
        assert adapter.search('building change detection', **kwargs)['cache_hit'] is False
    assert len(calls) == len(requests)
    for kwargs in requests:
        assert adapter.search('BUILDING change detection', **kwargs)['cache_hit'] is True
    assert len(calls) == len(requests)


def test_query_only_legacy_cache_cannot_satisfy_a_dated_request(tmp_path, monkeypatch):
    adapter, calls = stubbed(tmp_path, monkeypatch)
    old = tmp_path/('search-'+hashlib.sha256(b'building').hexdigest()+'.json')
    old.write_text(json.dumps({'items': [{'year': 2020}], 'cache_hit': False}), encoding='utf-8')
    result = adapter.search('building', published_from='2025-01-01')
    assert not result['cache_hit'] and calls and result['items'][0]['year'] == 2025


def test_native_publication_filters_and_newest_sort_are_sent_to_providers(tmp_path, monkeypatch):
    adapter = ScholarlySearchAdapter(tmp_path)
    urls = []

    def request(url):
        urls.append(parse_qs(urlsplit(url).query))
        return {'results': [], 'message': {'items': []}}

    monkeypatch.setattr(adapter, '_request_json', request)
    options = {'published_from': '2025-01-01', 'published_to': '2026-10-24', 'sort': 'newest'}
    adapter._openalex('building change detection', 6, **options)
    adapter._crossref('building change detection', 6, **options)
    assert urls[0]['filter'] == ['from_publication_date:2025-01-01,to_publication_date:2026-10-24']
    assert urls[0]['sort'] == ['publication_date:desc']
    assert urls[1]['filter'] == ['from-pub-date:2025-01-01,until-pub-date:2026-10-24']
    assert urls[1]['sort'] == ['published'] and urls[1]['order'] == ['desc']


def test_arxiv_submission_range_is_sent_and_date_basis_is_not_journal_publication(tmp_path, monkeypatch):
    adapter = ScholarlySearchAdapter(tmp_path)
    captured = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            return b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://arxiv.org/abs/2601.00001v1</id><published>2026-01-02T00:00:00Z</published><title>Paper</title></entry></feed>'

    def open_url(request, **kwargs):
        captured.append(parse_qs(urlsplit(request.full_url).query))
        return Response()

    monkeypatch.setattr('paperpilot_agentic.scholarly.urllib.request.urlopen', open_url)
    rows = adapter._arxiv('building', 3, published_from='2025-01-01', published_to='2026-10-24', sort='newest')
    assert captured[0]['search_query'] == ['(all:"building") AND submittedDate:[202501010000 TO 202610242359]']
    assert captured[0]['sortBy'] == ['submittedDate']
    assert rows[0]['publication_date'] == '2026-01-02'
    assert rows[0]['date_basis'] == 'arxiv_submission_date'
    adapter._arxiv('building "change detection"', 3)
    assert captured[1]['search_query'] == ['all:"building" AND all:"change detection"']


def test_unreadable_and_out_of_range_records_do_not_stop_fallback(tmp_path, monkeypatch):
    adapter, calls = stubbed(tmp_path, monkeypatch)
    monkeypatch.setattr(adapter, '_openalex', lambda *a, **kw: [
        {'source_id': 'old', 'title': 'Old', 'publication_date': '2020-01-01'},
        {'source_id': 'unknown', 'title': 'Year only', 'year': 2026},
        {'source_id': 'paywall', 'title': 'No supported PDF', 'publication_date': '2026-01-01'}])
    result = adapter.search('building', top_k=2, published_from='2025-01-01',
                            published_to='2026-10-24', sort='newest', require_full_text=True)
    assert result['providers'] == ['openalex', 'arxiv']
    assert all(row[0] != 'semantic_scholar' for row in calls)
    assert result['errors'][0]['provider'] == 'semantic_scholar'
    assert {r['reason'] for r in result['rejected_items']} == {
        'outside_date_range', 'missing_exact_date', 'no_supported_full_text_url'}
    assert [r['publication_date'] for r in result['items']] == ['2025-01-10', '2025-01-09']
    assert result['search_scope']['full_text_scope'] == 'arxiv_allowlist_only'


@pytest.mark.parametrize('kwargs', [
    {'published_from': '2025-02-30'}, {'published_to': '20260101'},
    {'published_from': '2026-01-01', 'published_to': '2025-01-01'},
    {'sort': 'fast'}, {'require_full_text': 'yes'}, {'published_from': 2025}])
def test_invalid_requests_are_rejected_before_network(tmp_path, monkeypatch, kwargs):
    adapter, calls = stubbed(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        adapter.search('building', **kwargs)
    assert not calls


@pytest.mark.parametrize('url,expected', [
    ('http://arxiv.org/abs/2501.00001v2', 'https://arxiv.org/pdf/2501.00001v2'),
    ('https://export.arxiv.org/pdf/2501.00001.pdf', 'https://arxiv.org/pdf/2501.00001'),
    ('https://arxiv.org/pdf/math.GT/0309136v1', 'https://arxiv.org/pdf/math.GT/0309136v1'),
    ('https://arxiv.org/html/2501.00001', None), ('https://evil.test/pdf/2501.00001', None),
    ('https://user:secret@arxiv.org/pdf/2501.00001', None), ('https://arxiv.org:invalid/pdf/2501.00001', None)])
def test_full_text_links_are_normalized_to_supported_arxiv_pdfs(url, expected):
    assert ScholarlySearchAdapter._safe_arxiv_url(url) == expected


def test_crossref_does_not_invent_a_day_for_year_only_metadata():
    assert ScholarlySearchAdapter._crossref_date({'published': {'date-parts': [[2025]]}}) is None
    assert ScholarlySearchAdapter._crossref_date({'published': {'date-parts': [[2025, 2, 3]]}}) == '2025-02-03'
