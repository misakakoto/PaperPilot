import importlib.util
from pathlib import Path

import pytest

from paperpilot_agentic.answer_audit import evidence_spans
from paperpilot_agentic.direct_answer import prepare_evidence

spec = importlib.util.spec_from_file_location('replay', Path(__file__).parents[1] / 'scripts/compare_audited_reader.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def snapshot():
    shown = [{'evidence_id': 'e1', 'content': 'One fact. Another fact.'}]
    return {'evidence': shown + [{'evidence_id': 'unused', 'content': 'Noise'}],
            'run': {'metadata': {'reader_progress': {
                'source_spans': evidence_spans(prepare_evidence(shown)),
                'draft': {'facts': []}}}}}


def test_replay_restores_only_original_workset():
    rows, draft = module.frozen_inputs(snapshot())
    assert [row['evidence_id'] for row in rows] == ['e1']
    assert draft == {'facts': []}


def test_replay_rejects_source_drift():
    value = snapshot()
    value['evidence'][0]['content'] += ' New content.'
    with pytest.raises(ValueError, match='exactly reproduce'):
        module.frozen_inputs(value)
