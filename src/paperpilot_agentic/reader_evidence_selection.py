"""A bounded, source-checked optional reader stage, never a hidden provider loop."""
from openai import APIError

from .durable_reader import DurableResearchModel
from .evidence_selection import select_evidence


class SelectionRetryBlocked(RuntimeError):
    """The selection stage has spent its one actual provider attempt."""


def select_reader_evidence(runtime, run_id, question, candidates, baseline, scope):
    if len(scope) != 1:
        raise ValueError('Evidence selection currently requires exactly one source')
    if any(hit.get('paper_sha256') not in scope for hit in [*candidates,*baseline]):
        raise ValueError('Evidence selection received an out-of-scope source')
    # The durable model accounts attempts, tokens and cancellation and protects
    # synthesis/repair reserves. Those stops are deliberately NOT caught here.
    model = DurableResearchModel(runtime.llm_runtime,run_id,stage='evidence_selection')
    attempts = 0
    def reserve_once():
        nonlocal attempts
        if attempts:
            raise SelectionRetryBlocked('Evidence selection does not retry provider requests')
        attempts += 1
    class OneAttemptModel:
        def complete(self,messages,**kwargs):
            return model.complete(messages,before_attempt=reserve_once,**kwargs)
    try:
        result = select_evidence(question,candidates,OneAttemptModel())
    except (ValueError, TypeError, APIError, SelectionRetryBlocked) as exc:
        result = {'hits':list(baseline), 'status':'DEGRADED', 'protocol':'evidence-select-v2',
                  'error_type':type(exc).__name__, 'fallback':'legacy_retrieval'}
    hits = [{**hit,'rank':i} for i,hit in enumerate(result['hits'],1)]
    trace = {k:v for k,v in result.items() if k != 'hits'}
    trace.update(provider_attempts=attempts,candidate_ids=[hit['stable_id'] for hit in candidates],
                 selected_ids=[hit['stable_id'] for hit in hits])
    runtime.db.update_run_metadata(run_id,{'reader_evidence_selection':trace})
    runtime.db.emit_event(run_id,'reader.evidence_selection',trace)
    return hits
