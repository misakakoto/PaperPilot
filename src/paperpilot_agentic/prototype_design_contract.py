"""Typed declarations for the fixed CPU binary-change prototype interface.

This checks the declared contract only; execution and semantic review remain required.
"""

import json

POLICY = "binary-design-interface-v1"
SUBMISSION_POLICY = "literal-binary-design-interface-v2"
INTERFACE = {
    "task": "binary_change_detection",
    "image_t1": ["B", 3, "H", "W"],
    "image_t2": ["B", 3, "H", "W"],
    "output": ["B", 1, "H", "W"],
    "output_semantics": "raw_logits",
}


def enabled(state):
    policy = state.get("design_contract_policy")
    if policy not in (None, POLICY, SUBMISSION_POLICY):
        raise ValueError("Unknown design contract policy")
    return policy in (POLICY, SUBMISSION_POLICY)


def check(state, content):
    if not enabled(state):
        return
    supplied = content.get("tensor_interfaces") if isinstance(content, dict) else None
    if (
        not isinstance(supplied, dict)
        or set(supplied) != set(INTERFACE)
        or any(supplied[k] != v for k, v in INTERFACE.items())
        or any(
            not isinstance(supplied[k], list) or type(supplied[k][1]) is not int
            for k in ("image_t1", "image_t2", "output")
        )
    ):
        raise ValueError(
            "Design must declare the fixed binary change detection interface: "
            + str(INTERFACE)
            + "; keep the binary task and return one same-size raw-logit channel"
        )


def submission_instructions(state):
    enabled(state)
    if state.get("design_contract_policy") != SUBMISSION_POLICY:
        return ""
    return (
        " Put this EXACT JSON object in arguments.content.tensor_interfaces: "
        + json.dumps(INTERFACE, ensure_ascii=False, sort_keys=True)
        + ". Keep intermediate shapes and explanatory prose in architecture, not inside this object. "
        "A text description of the right shape is not this machine-readable declaration. "
        "After rejection, resubmit the complete corrected design and literature_basis, not only this object."
    )
