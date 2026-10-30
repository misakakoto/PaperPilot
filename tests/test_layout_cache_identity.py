from types import SimpleNamespace
import hashlib

import pytest

from paperpilot_agentic.layout_extraction import LayoutExtraction, ExtractionStats, RegionRead
from paperpilot_agentic.structured_extraction import SourceRegion, ParsedArtifact


def test_same_run_cache_is_reaudited_for_the_current_question(tmp_path):
    image = tmp_path / 'crop.png'
    image.write_bytes(b'pixels')
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    class Adapter:
        model = 'model-a'
        def reaudit(self, original, path, *, prompt, metadata):
            assert prompt == 'Different task'
            return {**original, 'verified': False, 'metadata': {**original['metadata'],
                'offline_reaudit': True, 'evidence_relevant': False}}
    extractor = LayoutExtraction(Adapter(), layout_parser=None, layout_search=None,
        output_dir=None, file_hash=lambda path: hashlib.sha256(path.read_bytes()).hexdigest())
    cached = {'content': 'Prior values', 'confidence': .8, 'provenance': {
        'parser_cache_identity': extractor._cache_identity(), 'vision_verified': True,
        'vision_metadata': {'image_sha256': digest}, 'vision_adapter': 'Adapter'}}
    db = SimpleNamespace(get_run=lambda _: {'metadata': {'disable_cross_run_visual_cache': True}})
    session = SimpleNamespace(context=SimpleNamespace(db=db, run_id='run'),
        prior_by_artifact={str(image): cached}, stats=ExtractionStats())
    evidence = {'content': 'Current caption', 'provenance': {'document_sha256': 'paper'}}
    region = RegionRead({}, evidence, str(image), 'Different task', {})
    assert extractor._reuse(session, region)
    assert evidence['content'] == 'Current caption'
    assert not evidence['provenance']['vision_verified']
    assert session.stats.vision_api_calls == 0 and session.stats.vision_cache_hits == 1
    extractor.adapter.model = 'model-b'
    assert not extractor._reuse(session, region)
    extractor.adapter.model = 'model-a'
    image.write_bytes(b'changed pixels')
    assert not extractor._reuse(session, region)


@pytest.mark.parametrize('box', [[0, 0, float('inf'), 1], [0, 0, 1, float('nan')]])
def test_source_coordinates_must_be_finite(box):
    with pytest.raises(ValueError):
        SourceRegion(paper_sha256='paper', object_id='object', page=1, bbox=box, crop_sha256='crop')


def test_parser_identity_does_not_merge_identical_transcriptions(tmp_path):
    from paperpilot_agentic.db import RuntimeDatabase
    from paperpilot_agentic.durable_reader import persist_reader_evidence
    from paperpilot_agentic.models import Budget
    source = SourceRegion(paper_sha256='paper', object_id='table', page=1,
                          bbox=[0, 0, 10, 10], crop_sha256='crop')
    db = RuntimeDatabase(tmp_path / 'isolated.sqlite3')
    rid = db.create_run('Question', Budget())
    def item(config):
        artifact = ParsedArtifact(kind='table', source=source, parser='fixture', parser_version='1',
            config_sha256=config, parse_status='PARSED', validation_status='VALIDATED')
        return {'evidence_id': 'upstream', 'evidence_type': 'TABLE', 'content': 'Same values',
            'source_uri': 'paper://paper/layout/table', 'provenance': {
                'parser_cache_identity': config, 'structured_extraction': artifact.model_dump()}}
    first = persist_reader_evidence(db, rid, [item('a')])[0]
    assert persist_reader_evidence(db, rid, [item('a')])[0]['evidence_id'] == first['evidence_id']
    assert persist_reader_evidence(db, rid, [item('b')])[0]['evidence_id'] != first['evidence_id']
