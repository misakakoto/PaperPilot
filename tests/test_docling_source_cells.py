import copy
from types import SimpleNamespace

import pytest

from paperpilot_agentic.docling_worker import SourceOcrStage, retain_ocr_cells


def cell(text, box, *, index=0, ocr=True, origin='TOPLEFT', direction='left_to_right'):
    left, top, right, bottom = box
    return {'index': index, 'text': text, 'orig': text, 'from_ocr': ocr,
            'text_direction': direction, 'confidence': .98,
            'rect': {'r_x0': left, 'r_y0': bottom, 'r_x1': right, 'r_y1': bottom,
                     'r_x2': right, 'r_y2': top, 'r_x3': left, 'r_y3': top,
                     'coord_origin': origin}}


def test_same_ocr_label_in_nested_boxes_is_one_physical_observation():
    # Minimal form of the saved table header: equal text, independent IDs,
    # slightly different boxes. A duplicate string in a single cell is untouched.
    cells = [cell('Signal', [10, 20, 40, 32]),
             cell('Signal', [11, 21, 39, 31], index=9),
             cell('very very low', [10, 40, 90, 52], index=10)]
    before = copy.deepcopy(cells)
    retained, removed = retain_ocr_cells(cells)
    assert retained == [0, 2]
    assert removed[0]['retained_position'] == 0
    assert removed[0]['removed_position'] == 1
    assert removed[0]['removed_cell'] == cells[1]
    assert cells == before


def test_same_text_at_different_positions_is_preserved():
    cells = [cell('1', [10, 20, 20, 30]), cell('1', [20, 20, 30, 30], index=1),
             cell('1', [10, 31, 20, 41], index=2)]
    assert retain_ocr_cells(cells) == ([0, 1, 2], [])


def test_disagreeing_text_native_cells_and_coordinate_spaces_are_not_collapsed():
    cells = [cell('OA', [10, 20, 40, 32]), cell('VO', [10, 20, 40, 32], index=1),
             cell('OA', [10, 20, 40, 32], index=2, ocr=False),
             cell('OA', [10, 20, 40, 32], index=3, origin='BOTTOMLEFT'),
             cell('OA', [10, 20, 40, 32], index=4, direction='right_to_left')]
    assert retain_ocr_cells(cells) == ([0, 1, 2, 3, 4], [])


@pytest.mark.parametrize('problem', ['nonfinite', 'missing', 'rotated', 'zero_area', 'partial_overlap'])
def test_unreliable_or_partially_overlapping_geometry_does_not_remove_text(problem):
    first, second = cell('Signal', [10, 20, 40, 32]), cell('Signal', [10, 20, 40, 32], index=1)
    if problem == 'nonfinite':
        second['rect']['r_x0'] = float('nan')
    elif problem == 'missing':
        del second['rect']['r_y0']
    elif problem == 'rotated':
        second['rect']['r_x0'] += 1
    elif problem == 'zero_area':
        second = cell('Signal', [10, 20, 10, 32], index=1)
    else:
        second = cell('Signal', [30, 20, 60, 32], index=1)
    assert retain_ocr_cells([first, second]) == ([0, 1], [])


def test_ocr_stage_cleans_source_cells_before_the_next_table_stage():
    class Cell:
        def __init__(self, data):
            self.data = data
        def model_dump(self, **kwargs):
            return self.data
    class Page:
        page_no = 4
        def __init__(self, cells):
            self.parsed_page = SimpleNamespace(textline_cells=cells)
        @property
        def cells(self):
            return self.parsed_page.textline_cells
    cells = [Cell(cell('0.81', [10, 20, 40, 32])),
             Cell(cell('0.81', [11, 21, 39, 31], index=5))]
    page = Page(cells)
    receipts = []
    stage = SourceOcrStage(lambda result, pages: pages, receipts)
    assert list(stage(None, [page])) == [page]
    assert page.cells == cells[:1]
    assert receipts[0]['input_cells'] == 2
    assert receipts[0]['retained_cells'] == 1
