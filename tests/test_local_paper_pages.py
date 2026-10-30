import copy
import hashlib

import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from paperpilot_agentic.local_paper_pages import read_pages
from paperpilot_agentic.prototype_workspace import plan, complete
from test_prototype_discovery_dates import prepared


def corpus(tmp_path):
    path = tmp_path/'registered.pdf'
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
        NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({
        NameObject('/F1'): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(b'BT /F1 12 Tf 40 700 Td (Future work will introduce a new mechanism.) Tj ET')
    page[NameObject('/Contents')] = writer._add_object(stream)
    writer.write(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    class Adapter:
        paper_roots = [tmp_path]

        def resolve_pdf_paths(self, values):
            return [str(path)] if values == {sha} else []

    return Adapter(), path, sha


def test_registered_page_keeps_source_hash_and_complete_context(tmp_path):
    adapter, path, sha = corpus(tmp_path)
    result = read_pages(adapter, {'paper_sha256': sha, 'pages': [1]})
    row = result.evidence[0]
    assert 'Future work will introduce' in row['content']
    assert row['provenance']['complete_page'] is True and row['page'] == 1
    assert row['provenance']['paper_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('changes', [{'pages': []}, {'pages': [True]}, {'pages': [1, 1]},
    {'pages': [2]}, {'paper_sha256': 'b'*64}, {'path': 'other.pdf'}])
def test_invalid_or_unregistered_page_requests_are_rejected(tmp_path, changes):
    adapter, _, sha = corpus(tmp_path)
    with pytest.raises(ValueError):
        read_pages(adapter, {'paper_sha256': sha, 'pages': [1], **changes})


def test_cached_pdf_path_does_not_bypass_identity_or_root_checks(tmp_path):
    adapter, path, sha = corpus(tmp_path)
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='identity changed'):
        read_pages(adapter, {'paper_sha256': sha, 'pages': [1]})
    adapter.paper_roots = [tmp_path/'different-root']
    with pytest.raises(ValueError, match='inside configured'):
        read_pages(adapter, {'paper_sha256': sha, 'pages': [1]})


def test_prototype_local_pages_use_registered_tool_and_discovery_binding(tmp_path, monkeypatch):
    adapter, _, sha = corpus(tmp_path)
    rt, h, rid, state, _ = prepared(tmp_path, monkeypatch)
    state['literature_local_page_policy'] = 'registered-local-pages-v1'
    plan(state['work'], [{'id': k, 'goal': k, 'capability': c, 'depends_on': d, 'acceptance': 'source'}
        for k, c, d in [('discover', 'search_literature', []), ('pages', 'read_literature', ['discover'])]])
    complete(state['work'], 'discover', 'search_literature', {'ok': True, 'tool': 'paper.search',
        'evidence': [{'provenance': {'paper_sha256': sha}}]})
    monkeypatch.setattr(rt.registry.get('paper.read_pages'), 'handler', lambda context, args: read_pages(adapter, args))
    before = copy.deepcopy(state)
    request = {'action': 'work', 'reason': 'page context', 'arguments': {
        'task_id': 'pages', 'content': {'paper_sha256': sha, 'pages': [1]}}}
    h._apply(rid, state, request, 3)
    a = state['work']['artifacts'][state['work']['tasks']['pages']['output']]
    assert a['content']['coverage'] == 'BOUNDED_LOCAL_PAGES_NOT_FULL_PAPER_READ'
    assert a['inputs'] == [state['work']['tasks']['discover']['output']]
    assert len(rt.db.list_tool_calls(rid)) == 1
    h._apply(rid, before, request, 3)  # receipt completed before task commit
    assert len(rt.db.list_tool_calls(rid)) == 1
