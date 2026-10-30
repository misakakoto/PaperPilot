"""Request-bound recovery decisions; no ambiguous external operation is replayed."""

from .prototype_workspace import digest


def input_identity(state):
    value = {
            "goal": state.get("goal"),
            "role": state.get("role"),
            "turn": state.get("turn"),
            "work": state.get("work"),
            "binding": state.get("binding"),
            "observations": state.get("observations"),
            "analysis": state.get("analysis"),
            "model_protocol": state.get("model_protocol"),
            "runtime_profile": state.get("runtime_profile"),
        }
    if state.get('evidence_profile') == 'paper_active_v1':
        value['active_evidence_inputs'] = {k:state.get(k) for k in
            ('scope','required_papers','mode','requirements','ledger','paper_handles','source_handles',
             'read_handles','retrieval_identity','policy_identity','searches','search_signatures','tool_keys')}
        value['active_evidence_inputs'].update({k:state.get(k) for k in ('catalogue_searches','supplement_uses')})
    return digest(value)


def bind_request(run_id, index, state, messages):
    body = {
        "run_id": run_id,
        "turn": index,
        "stage": f"research.{state['role']}.{index}",
        "inputs_sha256": input_identity(state),
        "messages_sha256": digest(messages),
    }
    return {**body, "correlation_id": digest(body)}


def trace(state, action, **details):
    row = {"action": action, **details}
    identity = digest(row)
    rows = state.setdefault("recovery_trace", [])
    if not any(r["id"] == identity for r in rows):
        rows.append({"id": identity, **row})


def recover_response(run_id, state, request, events):
    binding = request.get("binding")
    if not binding or binding != bind_request(
        run_id, state["turn"] + 1, state, request["messages"]
    ):
        trace(state, "WAIT_UNKNOWN", reason="REQUEST_INPUT_IDENTITY_MISMATCH", new_dispatches=0)
        return None
    matches = [
        e
        for e in events
        if e["event_type"] == "llm.response.saved"
        and e["payload"].get("correlation_id") == binding["correlation_id"]
        and e["payload"].get("stage") == binding["stage"]
    ]
    if len(matches) != 1:
        trace(state, "WAIT_UNKNOWN", reason="NO_UNIQUE_BOUND_RESPONSE", new_dispatches=0)
        return None
    trace(
        state,
        "APPLY_SAVED_RESPONSE",
        turn=binding["turn"],
        correlation_id=binding["correlation_id"],
        response_event_id=matches[0]["event_id"],
        new_dispatches=0,
    )
    return matches[0]["payload"]["raw_response"]
