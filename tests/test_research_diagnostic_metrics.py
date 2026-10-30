from paperpilot_agentic.research_diagnostics import coverage_group, metrics


def test_fixed_boundaries_use_integer_comparisons_and_empty_counts_are_undefined():
    assert [coverage_group(v, 100) for v in (0, 1, 2, 10, 11, 100)] == [
        '0', '(0,1%]', '(1%,10%]', '(1%,10%]', '(10%,100%]', '(10%,100%]']
    empty = metrics([0, 0, 0, 0])
    assert all(empty[k] is None for k in ('precision', 'recall', 'f1', 'iou'))
    no_positives = metrics([100, 0, 0, 0])
    assert no_positives['f1'] is None
    false_positives = metrics([90, 10, 0, 0])
    assert false_positives['precision'] == 0 and false_positives['recall'] is None
