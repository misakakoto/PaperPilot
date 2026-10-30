from copy import deepcopy
import json
import pytest

from examples.qasper_length_ablation.experiment import rank, evaluate
from scripts.run_research_loop_live import check_approved_protocol
from paperpilot_agentic.builtin_tools import _experiment_plan_handler, _experiment_metric_summary
from test_experiment_design_binding import setup_plan
from test_agentic_v2 import _StageModel
from paperpilot_agentic.intelligence import LLMClaimPlanner, enforce_requested_answer_contract
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from test_agentic_v2 import _StaticAdapter
from paperpilot_agentic.models import Budget


def test_length_normalization_changes_only_length_factor():
    paragraphs = ['match ' + 'filler ' * 100, 'match short', 'unrelated paragraph']
    assert rank('match', paragraphs, 0)[:2] == [0, 1]
    assert rank('match', paragraphs, .75)[:2] == [1, 0]


def test_scoring_retains_unmatched_gold_and_exclusion_counts():
    dataset = {'p': {'full_text': [{'paragraphs': ['a', 'b']}], 'qas': [
        {'question_id': 'q1', 'question': 'a', 'answers': [{'answer': {'evidence': ['a', 'missing']}}]},
        {'question_id': 'q2', 'question': 'b', 'answers': [{'answer': {'unanswerable': True}}]},
    ]}}
    result = evaluate(dataset, 7)
    assert result['question_count'] == 1
    assert result['baseline_recall_at_3'] == .5
    assert result['excluded_no_text_reference'] == 1
    assert result['held_out'] is False


def test_server_image_digest_reaches_preregistered_protocol(tmp_path):
    context, args = setup_plan(tmp_path, [7])
    args['docker_image'] = 'sha256:' + 'a' * 64
    result = _experiment_plan_handler(context, args)
    assert result.output['protocol']['policy']['docker_image'] == args['docker_image']


def test_approval_gate_does_not_rewrite_model_experiment():
    protocol = {'study_design': {'baseline': 'bm25_b0', 'variants': ['bm25_b075'],
                'dataset': 'qasper-dev-v0.3', 'seeds': [7, 11, 19]},
                'seed_sweep': [7, 11, 19],
                'acceptance_criteria': [{'metric': 'improvement', 'operator': '>', 'value': 0}]}
    before = deepcopy(protocol)
    check_approved_protocol(protocol)
    assert protocol == before
    protocol['study_design']['variants'] = ['a different method']
    with pytest.raises(ValueError, match='variants'):
        check_approved_protocol(protocol)


def test_run_ablation_is_execution_not_a_request_for_paper_tables():
    plan = LLMClaimPlanner(_StageModel())._validate_plan(
        _StageModel(experiment=True)._plan_payload(),
        'Propose a hypothesis and run the approved ablation experiment; compare baseline variants.')
    assert 'independent_experiment' in plan.claims[0]['gaps']
    assert 'numeric_results' not in plan.claims[0]['gaps']
    plan = LLMClaimPlanner(_StageModel())._validate_plan(
        _StageModel(experiment=True)._plan_payload(), 'Run the approved ablation.')
    assert plan.claims[0]['required_evidence_types'] == ['EXPERIMENT']


def test_execution_claim_without_spec_is_repairable_not_silently_accepted():
    payload = _StageModel(experiment=True)._plan_payload()
    del payload['claims'][0]['experiment_spec']
    with pytest.raises(ValueError, match='experiment_spec'):
        LLMClaimPlanner(_StageModel())._validate_plan(payload, 'Run the approved ablation.')


@pytest.mark.parametrize('criterion', [
    {'metric': 'score', 'operator': '>', 'numeric_value': 0},
    {'metric': 'score', 'operator': '>', 'value': True},
    {'metric': 'score', 'operator': '>', 'value': float('nan')},
])
def test_bad_criterion_is_repaired_before_approval(criterion):
    payload = _StageModel(experiment=True)._plan_payload()
    payload['claims'][0]['experiment_spec']['acceptance_criteria'] = [criterion]
    with pytest.raises(ValueError, match='finite numeric value'):
        LLMClaimPlanner(_StageModel())._validate_plan(payload, 'Run the approved ablation.')


def test_research_orientation_precedes_planner_without_closing_claims(tmp_path):
    model = _StageModel()
    runtime = ResearchAgentRuntime(db_path=tmp_path/'r.sqlite3', workspace=tmp_path,
        search_adapter=_StaticAdapter([{'content': 'Source passage about residual connections.',
                                        'source_uri': 'paper://public/example'}]), llm=model)
    result = runtime.start(question='Investigate residual connections',
                           metadata={'research_exploration': True}, auto_advance=False)
    assert result['tool_calls'][0]['tool_name'] == 'paper.search'
    context = json.loads(model.calls[0]['messages'][-1]['content'])['context']
    assert context['research_orientation']['sources'][0]['status'] == 'retrieved_unverified'
    assert result['claim_graph']['claims'][0]['status'] == 'UNKNOWN'
    assert result['run']['used_tool_calls'] == 1


def test_registered_experiment_continuation_cannot_be_starved_by_search(tmp_path):
    runtime = ResearchAgentRuntime(db_path=tmp_path/'r.sqlite3', workspace=tmp_path)
    result = runtime.start(question='Run experiment', claims=[
        {'statement': 'An unresolved literature question', 'gaps': ['primary_source'], 'importance': 1.0},
        {'statement': 'A registered hypothesis', 'gaps': ['independent_experiment'], 'importance': .5,
         'metadata': {'experiment': {'name': 'registered'}}},
    ], auto_advance=False)
    action = runtime.scheduler.select_next(result['run']['run_id'], budget=Budget())
    assert action.tool_name == 'experiment.run'


def test_experiment_evidence_summary_preserves_parameters_and_all_seeds():
    metrics = {'by_seed': {str(seed): {'improvement': -.02,
                'configuration': {'k1': 1.2, 'candidate_b': .75},
                'rows': [{'details': 'x' * 10000}] * 100} for seed in (7, 11, 19)}}
    before = deepcopy(metrics)
    summary = _experiment_metric_summary(metrics)
    assert set(summary['by_seed']) == {'7', '11', '19'}
    for row in summary['by_seed'].values():
        assert row['configuration']['candidate_b'] == .75
        assert row['improvement'] == -.02
        assert 'rows' not in row
    assert metrics == before


def test_generic_limitation_does_not_deny_an_executed_experiment():
    result = enforce_requested_answer_contract('Run an experiment and report limitations.',
                                               {'answer': 'The experiment refuted the hypothesis.'})
    assert '只是基于论文描述' not in result['answer']
    assert '实际执行的实验条件' in result['answer']


def test_orientation_honors_zero_tool_budget(tmp_path):
    runtime = ResearchAgentRuntime(db_path=tmp_path/'r.sqlite3', workspace=tmp_path,
                                   llm=_StageModel(), search_adapter=_StaticAdapter([]))
    result = runtime.start(question='Research mechanism', budget=Budget(max_tool_calls=0),
                           metadata={'research_exploration': True}, auto_advance=False)
    assert result['tool_calls'] == []
    assert result['run']['metadata']['research_orientation']['error'] == 'Orientation budget exhausted'
