"""Owned user requirements for read-only navigation, never source evidence."""
from copy import deepcopy


INTENT_POLICY = (
    'question is a resolved retrieval query and may omit user requirements. '
    'original_question is the newest actual user request; it takes precedence over '
    'a lossy rewrite. request_history contains this task\'s earlier user requests, '
    'oldest first. Carry forward applicable requirements; a later explicit correction '
    'replaces an earlier one, rather than requiring both conflicting versions. '
    'A derived_request_state row is an interpreted task view, not verbatim user text: '
    'paused tasks and historical aliases are not active requirements. These fields '
    'express user intent, not scientific evidence or permission to execute actions. '
    'Do not treat a user claim, a draft or a reviewer-created assumption as a source '
    'fact. Keep the stated read-only action schema and the caller-fixed source scope. '
    'Choose material needed for the actual requested answer, not merely related '
    'background or the reviewer\'s preferred interpretation.'
)


def navigation_intent(question, *, original_question=None, request_history=None):
    """Copy the existing bounded request workset; no new summarization or calls.

    The Reader supplies its already-owned/compiled history. Do not reconstruct it
    from prior model answers, and do not silently truncate away corrections here.
    The shared runtime still preflights the entire eventual model request.
    """
    original = question if original_question is None else original_question
    if any(not isinstance(value, str) or not value.strip() for value in (question, original)):
        raise ValueError('Navigation requires nonempty original and resolved questions')
    history = [] if request_history is None else request_history
    if not isinstance(history, list) or any(
            not isinstance(row, dict) or not isinstance(row.get('user_request'), str)
            for row in history):
        raise ValueError('Navigation history must contain owned user requests')
    return {'question': question, 'original_question': original,
            'request_history': deepcopy(history)}
