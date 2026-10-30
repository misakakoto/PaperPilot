from types import SimpleNamespace

import pytest

from paperpilot_agentic.docling_worker import SourceFormulaStage


def element(box, *, page=2, origin='TOPLEFT', ref='formula1'):
    class Box:
        def model_dump(self, **kwargs):
            return dict(zip(('l', 't', 'r', 'b'), box), coord_origin=origin)
    return SimpleNamespace(self_ref=ref, prov=[SimpleNamespace(page_no=page, bbox=Box())])


def document():
    return SimpleNamespace(pages={2: SimpleNamespace(size=SimpleNamespace(height=200))})


def stage():
    prepared = []
    def prepare(conv_res, element):
        prepared.append(element.self_ref)
        return SimpleNamespace(item=element)
    delegate = SimpleNamespace(is_processable=lambda doc, element: True, prepare_element=prepare)
    wrapped = SourceFormulaStage(delegate, {'page': 2, 'bbox': [10, 20, 100, 90]}, [])
    return wrapped, prepared


def test_formula_crop_admits_target_and_skips_other_formula_before_rendering():
    wrapped, prepared = stage()
    result = SimpleNamespace(document=document())
    target, neighbor = element([15, 25, 95, 80]), element([15, 110, 95, 140], ref='neighbor')
    assert wrapped.prepare_element(result, target).item is target
    assert wrapped.prepare_element(result, neighbor) is None
    assert prepared == ['formula1']


def test_bottom_left_bbox_is_bound_to_same_top_left_source():
    wrapped, _ = stage()
    assert wrapped.is_processable(document(), element([15, 175, 95, 120], origin='BOTTOMLEFT'))
    assert not wrapped.is_processable(document(), element([15, 175, 95, 120]))


def test_other_page_or_missing_provenance_cannot_enter_formula_model():
    wrapped, _ = stage()
    assert not wrapped.is_processable(document(), element([15, 25, 95, 80], page=3))
    missing = element([15, 25, 95, 80])
    missing.prov = []
    assert not wrapped.is_processable(document(), missing)


@pytest.mark.parametrize('box,origin', [([15, float('nan'), 95, 80], 'TOPLEFT'),
    ([15, 25, 15, 80], 'TOPLEFT'), ([15, 25, 95, 80], 'unknown'),
    ([80, 25, 180, 80], 'TOPLEFT')])
def test_invalid_or_mostly_outside_formula_is_not_sent_to_enrichment(box, origin):
    wrapped, _ = stage()
    assert not wrapped.is_processable(document(), element(box, origin=origin))


def test_source_match_does_not_override_the_sdk_formula_type_filter():
    wrapped, prepared = stage()
    wrapped.delegate.is_processable = lambda doc, element: False
    assert wrapped.prepare_element(SimpleNamespace(document=document()), element([15, 25, 95, 80])) is None
    assert prepared == []


def test_only_model_outputs_are_recorded_as_enriched_refs():
    class Model:
        def __call__(self, doc, elements):
            for prepared in elements:
                prepared.item.text = 'x+y'
                yield prepared.item
    receipts = []
    wrapped = SourceFormulaStage(Model(), {'page': 2, 'bbox': [10, 20, 100, 90]}, receipts)
    target = element([15, 25, 95, 80])
    assert list(wrapped(document(), [SimpleNamespace(item=target)])) == [target]
    assert wrapped.enriched_refs == {'formula1'}
    assert receipts == [{'item_refs': ['formula1']}]
