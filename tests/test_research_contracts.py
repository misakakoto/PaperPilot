import pytest
from pydantic import ValidationError

from paperpilot_agentic.research_contracts import new_state, validate_state, assurance


def state():
    return dict(
        goal="Observe before deciding",
        phase="READY",
        role="lead",
        turn=0,
        completed=False,
        limits={},
        contexts={},
        decisions=[],
        candidates=[],
    )


def test_state_is_versioned_preserves_engine_fields_and_keeps_assurances_separate():
    original = state() | {"binding": {"identity": "host-owned"}}
    saved = new_state(original)
    assert saved["state_contract"] == "research-task-state-v1"
    assert saved["binding"] == original["binding"]
    assert "state_contract" not in original
    saved.update(
        phase="COMPLETED",
        completed=True,
        verification={"prediction_label_metrics": "HOST_RECOMPUTED"},
    )
    assert assurance(saved) == {
        "execution": "COMPLETED",
        "facts": "HOST_RECOMPUTED",
        "inference": "NOT_INDEPENDENTLY_VERIFIED",
        "content_review": "NOT_REVIEWED",
        "scientific_conclusion": "NOT_ESTABLISHED_BY_WORKFLOW_COMPLETION",
    }


@pytest.mark.parametrize(
    "change", [{"turn": -1}, {"phase": "FAKE_SUCCESS"}, {"role": "approver"}, {"completed": "true"}]
)
def test_new_state_rejects_invalid_common_values(change):
    with pytest.raises(ValidationError):
        new_state(state() | change)


def test_legacy_state_is_not_rewritten_or_upgraded():
    old = state()
    assert validate_state(old) is old
    assert "state_contract" not in old
