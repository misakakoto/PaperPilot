"""Host CLI: prepare owned recipes, run research roles, inspect/approve exact candidates.

No approval is implied by --resume. Import historic DB with a read-only SQLite
backup; new episode records and GPU ledgers never rewrite old campaign budgets.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from paperpilot_agentic.config import RuntimeSettings
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.providers import OpenAICompatibleLLM
from paperpilot_agentic.reference_methods import encoded, sha
from paperpilot_agentic.research_assets import register_recipe
from paperpilot_agentic.research_harness import ResearchHarness
from paperpilot_agentic.training_state import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'start', 'diagnose', 'register-experiment', 'register-case', 'diagnose-imported', 'resume', 'status', 'approve', 'report', 'evaluate-roles', 'continue-analysis', 'register-prototype-profile'))
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--db', required=True)
    parser.add_argument('--source-db')
    parser.add_argument('--training-binding')
    parser.add_argument('--install-workspace')
    parser.add_argument('--recipe-id')
    parser.add_argument('--request-id')
    parser.add_argument('--goal')
    parser.add_argument('--run-id')
    parser.add_argument('--parent-run-id')
    parser.add_argument('--candidate-id')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--source-file', type=Path)
    parser.add_argument('--license-file', type=Path)
    parser.add_argument('--bundle-id')
    parser.add_argument('--research-case-id')
    parser.add_argument('--report-protocol', choices=('legacy-v1','metric-facts-v2','metric-facts-v3'), default='metric-facts-v3')
    parser.add_argument('--delivery-mode', choices=('strict', 'assisted'), default='strict')
    parser.add_argument('--comparison-mode', choices=('summary', 'tools'), default='tools')
    parser.add_argument('--approval-id')
    parser.add_argument('--model')
    parser.add_argument('--output')
    parser.add_argument('--max-turns', type=int)
    parser.add_argument('--unlimited-llm-usage', action='store_true')
    args = parser.parse_args(argv)
    target = Path(args.db).resolve()
    if args.source_db:
        original = Path(args.source_db).resolve(strict=True)
        if original == target:
            parser.error('Historic source DB must remain read-only; use a separate target')
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(original.as_uri()+'?mode=ro', uri=True) as src, sqlite3.connect(target) as dst:
                src.backup(dst)
    settings = RuntimeSettings.from_env(require_api_key=args.command in {'start', 'diagnose', 'diagnose-imported', 'resume', 'evaluate-roles', 'continue-analysis'})
    settings = replace(settings, workspace=Path(args.workspace).resolve(), runtime_db=target,
        model=args.model or settings.model, fallback_models=(), thinking_enabled=False)
    model = OpenAICompatibleLLM(settings, max_retries=0) if settings.api_key else None
    protocol = 'research-v1/'+sha(encoded({'model': settings.model, 'provider': settings.provider,
        'base_url': settings.base_url, 'temperature': settings.temperature,
        'thinking': False, 'timeout': settings.timeout_seconds}))
    runtime = ResearchAgentRuntime(workspace=settings.workspace, db_path=target,
                                   llm=model, text_reader_protocol=protocol)
    harness = ResearchHarness(runtime)
    if args.command == 'register-prototype-profile':
        if not args.source_file or not args.license_file:
            parser.error('register-prototype-profile requires --source-file and --license-file')
        from paperpilot_agentic.prototype_profiles import register_snunet_profile
        result = {'profile_id': register_snunet_profile(runtime, args.source_file.read_bytes(), args.license_file.read_bytes()),
                  'api_calls': 0, 'code_execution': False}
    elif args.command == 'register-case':
        if not args.manifest:
            parser.error('register-case requires --manifest')
        from paperpilot_agentic.research_case import register_case
        result = register_case(runtime,json.loads(args.manifest.read_bytes()))
    elif args.command == 'register-experiment':
        if not args.manifest:
            parser.error('register-experiment requires --manifest')
        from paperpilot_agentic.experiment_import import register_experiment
        result = register_experiment(runtime, args.manifest)
    elif args.command == 'diagnose-imported':
        if not args.bundle_id or not args.request_id:
            parser.error('diagnose-imported requires --bundle-id and --request-id')
        result = harness.start_imported_diagnosis(args.bundle_id, args.request_id,
            comparison_mode=args.comparison_mode, report_protocol=args.report_protocol,
            research_case_id=args.research_case_id, delivery_mode=args.delivery_mode)
        result = harness.advance(result['run']['run_id'], max_turns=args.max_turns)
    elif args.command == 'prepare':
        if not args.training_binding:
            parser.error('prepare requires --training-binding')
        result = {'recipe_id': register_recipe(runtime, args.training_binding,
            destination_workspace=args.install_workspace), 'api_calls': 0,
            'workspace': args.install_workspace or args.workspace}
    elif args.command == 'start':
        if not all((args.goal, args.recipe_id, args.request_id)):
            parser.error('start requires --goal, --recipe-id and --request-id')
        result = harness.start(goal=args.goal, recipe_id=args.recipe_id, request_id=args.request_id,
                               unlimited_llm_usage=args.unlimited_llm_usage)
        result = harness.advance(result['run']['run_id'], max_turns=args.max_turns)
    elif args.command == 'diagnose':
        if not all((args.parent_run_id, args.candidate_id, args.request_id)):
            parser.error('diagnose requires --parent-run-id, --candidate-id and --request-id')
        result = harness.start_diagnosis(args.parent_run_id, args.candidate_id, args.request_id,
                                        comparison_mode=args.comparison_mode)
        result = harness.advance(result['run']['run_id'], max_turns=args.max_turns)
    elif args.command == 'evaluate-roles':
        from paperpilot_agentic.research_role_eval import run_comparison
        result = run_comparison(runtime, request_id=args.request_id)
    elif args.command == 'approve':
        if not args.approval_id:
            parser.error('approve requires the exact --approval-id after reviewing its candidate')
        result = runtime.approve(args.approval_id, actor='host-cli-user',
                                 comment='Explicit candidate-specific approval; resume is separate')
    else:
        if not args.run_id:
            parser.error('This command requires --run-id')
        if args.command=='continue-analysis':
            from paperpilot_agentic.research_analysis_continuation import continue_analysis
            result=continue_analysis(runtime,args.run_id,max_turns=args.max_turns)
        else:
            result = (harness.advance(args.run_id, max_turns=args.max_turns)
                  if args.command == 'resume' else harness.status(args.run_id))
    if args.output:
        path = Path(args.output).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, result)
    if 'research_episode' in result:
        print(json.dumps({'run_id': result['run']['run_id'], 'status': result['run']['status'],
            'episode': result['research_episode'], 'llm_budget': result['llm_budget'],
            'pending_approvals': result['pending_approvals']}, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == '__main__':
    main()
