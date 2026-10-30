from types import SimpleNamespace

from paperpilot_agentic.source_navigation import locate_requested_objects
from paperpilot_agentic.read_packet import build_read_packet


def test_object_and_limitations_are_read_from_scoped_originals_not_heading_previews():
    hits = [SimpleNamespace(stable_id='intro', text='We propose an encoder.', section='Introduction',
                channels=(), title='Synthetic', score=0, file_name='s.pdf', paper_sha256='p', page_start=2, page_end=2),
            SimpleNamespace(stable_id='eq', text='Discretization\nAbar = exp(Delta A)\n(2)', section='Methods',
                channels=(), title='Synthetic', score=0, file_name='s.pdf', paper_sha256='p', page_start=4, page_end=4),
            SimpleNamespace(stable_id='limit', text='Limitations: poor image quality and occlusion.', section='Discussion',
                channels=(), title='Synthetic', score=0, file_name='s.pdf', paper_sha256='p', page_start=9, page_end=9)]
    class Store:
        def get_chunks(self, ids):
            return [h for h in hits if h.stable_id in ids]
    outline = [{'chunk_ids': ['intro', 'eq', 'limit']}]
    result = locate_requested_objects(Store(), 'p', outline, '依据式(2)解释 Delta')
    assert len(result) == 1 and result[0]['page'] == 4
    assert 'exp(Delta A)' in result[0]['content']
    assert build_read_packet('依据式(2)解释 Delta', result).rows == result
    assert locate_requested_objects(Store(), 'p', outline, 'Explain limitations')[0]['page'] == 9
    assert locate_requested_objects(Store(), 'p', outline, 'Explain mechanism') == []
