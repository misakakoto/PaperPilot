from copy import deepcopy

import pytest

from paperpilot_agentic.diagnostic_facts import (
    direction,
    observation_facts,
    select_observations,
    validate_experiment,
)


def row(seed, p, r, f1, group=None):
    return {
        "seed": seed,
        **({"group": group} if group is not None else {}),
        "delta": {"precision": p, "recall": r, "f1": f1, "iou": f1},
    }


def test_both_up_is_not_tradeoff_and_positive_group_f1_is_not_down():
    rows = [row(303, 0.01, 0.08, 0.0226, "(1%,10%]"), row(901, 0.02, -0.03, -0.01, "(10%,100%]")]
    observations = {"coverage": {"rows": rows}}
    before = deepcopy(observations)
    facts = observation_facts(observations)
    assert facts["pattern/coverage/0/precision_recall"]["relation"] == "BOTH_UP"
    assert "均上升" in facts["pattern/coverage/0/precision_recall"]["text"]
    assert facts["pattern/coverage/0/f1"]["direction"] == "UP"
    assert facts["pattern/coverage/1/precision_recall"]["relation"] == "OPPOSITE"
    assert facts["pattern/coverage/0/f1"]["seed"] == 303
    assert observations == before
    with pytest.raises(ValueError, match="existing host"):
        select_observations(["pattern/coverage/999/f1"], facts)


@pytest.mark.parametrize(
    "value,want", [(0, "UNCHANGED"), (None, "UNDEFINED"), (1e-13, "UP"), (-1e-13, "DOWN")]
)
def test_zero_undefined_and_small_values_are_not_conflated(value, want):
    assert direction(value) == want
    facts = observation_facts({"paired": {"rows": [row(999, value, value, value)]}})
    assert facts["pattern/paired/0/f1"]["direction"] == want
    if value is None:
        assert "不能记为零" in facts["pattern/paired/0/f1"]["text"]


def test_new_experiment_contract_requires_intervention_and_open_alternatives():
    old = {
        "hypothesis": "possible mechanism",
        "support_refs": ["x"],
        "counter_refs": ["y"],
        "missing_observations": ["activation"],
        "falsification": "controlled result",
    }
    with pytest.raises(ValueError, match="Experiment v2"):
        validate_experiment(old)
    new = old | {
        "decision": "STOP_INSUFFICIENT_EVIDENCE",
        "intervention": "Do not run until confound is measured",
        "controls": ["same data"],
        "measurements": ["activations"],
        "expected_patterns": ["specified contrast"],
        "alternatives_remaining": ["unmeasured optimizer effects"],
    }
    assert validate_experiment(new) == old
    with pytest.raises(ValueError, match="alternatives_remaining"):
        validate_experiment(new | {"alternatives_remaining": []})
