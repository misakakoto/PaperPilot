"""Portable, read-only delivery of versioned prototype results; no execution."""
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .prototype_workspace import digest
from .prototype_outcomes import validated_candidates


def build_prototype_bundle(state, run, *, baseline=None, verifier=None):
    work = state['work']
    artifacts = work['artifacts']
    if any(digest(value) != identity for identity, value in artifacts.items()):
        raise ValueError('Prototype artifact identity changed')
    if state.get('profile_id') and baseline is None:
        raise ValueError('Bound baseline source and license are required')
    current = {t['output'] for t in work['tasks'].values() if t['status'] == 'DONE'}
    implementations = [t for t in work['tasks'].values() if t['capability'] == 'implement']
    validations = [t for t in work['tasks'].values() if t['capability'] == 'validate']
    codes = {t['output'] for t in implementations if t['status'] == 'DONE'}
    covered = set()
    for task in validations:
        if task['status'] != 'DONE':
            continue
        artifact = artifacts[task['output']]
        content = artifact['content']
        code = content.get('validated_code_id')
        contract_matches = (not state.get('validation_contract')
                            or content.get('validation_contract') == state['validation_contract'])
        if content.get('passed') is True and code in codes and code in artifact['inputs'] and contract_matches:
            covered.add(code)
    passed = (bool(codes) and codes <= covered and all(t['status'] == 'DONE' for t in implementations+validations)
              and all(artifacts[t['output']]['content'].get('passed') is True for t in validations))
    manifest = {'format': 'paperpilot-prototype-delivery-v1', 'run_id': run['run_id'],
                'goal': work['goal'], 'phase': state.get('phase'), 'completed': state.get('completed', False),
                'stop_reason': state.get('stop_reason'), 'workspace_revision': work['revision'],
                'engineering_validation': 'PASSED_FOR_CURRENT_CANDIDATES' if passed else 'INCOMPLETE_OR_FAILED',
                'scientific_validity': 'NOT_VERIFIED', 'model_review_facts': 'NOT_HOST_VERIFIED',
                'profile_id': state.get('profile_id'), 'worker_identity': state.get('worker_identity'),
                'execution_environment': state.get('execution_environment'),
                'validation_contract': state.get('validation_contract'),
                'verifier_included': verifier is not None,
                'usage': {k: run[k] for k in ('used_llm_calls', 'used_tokens')}, 'budget': run['budget']}
    manifest['validated_candidates'] = validated_candidates(state)
    files = {}
    for key in ('initial_plan_policy', 'evidence_replanning_policy', 'literature_review_context_policy', 'analysis_read_policy', 'automatic_source_read_policy', 'action_envelope_policy', 'research_source_policy', 'plan_edge_policy', 'literature_local_page_policy', 'literature_review_policy', 'candidate_patch_policy', 'task_definition_policy', 'candidate_input_policy', 'automatic_validation_policy', 'instruction_policy', 'context_policy', 'task_context_policy', 'plan_policy', 'source_policy', 'repair_policy', 'literature_policy', 'literature_context_policy', 'literature_analysis_policy', 'literature_discovery_policy', 'literature_local_first_policy', 'literature_local_read_policy', 'literature_excerpt_characters', 'research_date', 'required_evidence_ids', 'source_run_id'):
        if key in state:
            manifest[key] = state[key]
    if 'quote_handle_policy' in state:
        manifest['quote_handle_policy'] = state['quote_handle_policy']
    if 'analysis_workset_policy' in state:
        manifest['analysis_workset_policy'] = state['analysis_workset_policy']
    for key in ('review_page_sharing_policy', 'literature_repair_budget_policy', 'review_correction_policy', 'design_context_policy', 'design_contract_policy', 'code_navigation_policy'):
        if key in state:
            manifest[key] = state[key]

    def add_json(name, value):
        files[name] = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')

    if 'runtime_profile' in state:
        manifest['runtime_profile'] = state['runtime_profile']
    for key in ('context_manifest','change_impact','recovery_trace','change_impacts'):
        add_json(key+'.json',state.get(key))
    add_json('manifest.json', manifest)
    add_json('tasks.json', work['tasks'])
    if 'task_order' in work:
        add_json('task_order.json', work['task_order'])
    add_json('constraint_revisions.json', state.get('constraint_revisions', []))
    add_json('goal_revisions.json', state.get('goal_revisions', []))
    add_json('repair_history.json', state.get('repair_history', []))
    add_json('literature_repair_history.json', state.get('literature_repair_history', []))
    add_json('literature_evidence_extensions.json', state.get('literature_evidence_extensions', []))
    add_json('analysis_quote_normalizations.json', state.get('analysis_quote_normalizations', []))
    add_json('context_recoveries.json', state.get('context_recoveries', []))
    add_json('preflight_recoveries.json', state.get('preflight_recoveries', []))
    add_json('literature_review_normalizations.json', state.get('literature_review_normalizations', []))
    add_json('host_response_recoveries.json', state.get('host_response_recoveries', []))
    add_json('plan_elaborations.json', state.get('plan_elaborations', []))
    add_json('model_migrations.json', state.get('model_migrations', []))
    add_json('action_normalizations.json', state.get('action_normalizations', []))
    add_json('implementation_normalizations.json', state.get('implementation_normalizations', []))
    add_json('source_revisions.json', [
        {**{k: v for k, v in row.items() if k != 'previous_sources'},
         'previous_sources': {sid: {k: value for k, value in record.items()
                                   if k in {'evidence_id', 'run_id', 'content_hash', 'source_uri', 'evidence_type'}}
                              for sid, record in row['previous_sources'].items()}}
        for row in state.get('source_revisions', [])])
    index = {}
    for identity, artifact in artifacts.items():
        index[identity] = {k: artifact[k] for k in ('name', 'kind', 'inputs')}
        index[identity].update(current=identity in current, content_included=artifact['kind'] != 'read')
        if artifact['kind'] == 'read':
            record = artifact['content']
            if (isinstance(record, dict) and record.get('evidence_type') == 'CODE'
                    and isinstance(record.get('provenance', {}).get('license_text'), str)
                    and record['provenance']['license_text'].strip()):
                prefix = 'source_licenses/'+identity
                files[prefix+'/LICENSE'] = record['provenance']['license_text'].encode('utf-8')
                add_json(prefix+'/provenance.json', {'source_artifact_id': identity,
                    'source_uri': record.get('source_uri'), 'license_url': record['provenance'].get('license_url'),
                    'assurance': 'SUPPLIED_LICENSE_TEXT_NOT_LEGAL_REVIEW'})
            continue
        add_json(f'artifacts/{identity}.json', artifact)
        if artifact['kind'] == 'implement':
            content = artifact['content']
            prefix = f'candidates/{identity}'
            files[prefix+'/model.py'] = content['source'].encode('utf-8')
            add_json(prefix+'/config.json', content['config'])
            add_json(prefix+'/contract.json', state.get('validation_contract') or {
                'kind': 'snunet_ecam' if state.get('profile_id') else 'unbound'})
            if content.get('diff'):
                files[prefix+'/changes.diff'] = content['diff'].encode('utf-8')
    add_json('artifact_index.json', index)
    if baseline is not None:
        files['baseline/model.py'] = baseline['source']
        files['baseline/LICENSE'] = baseline['license']
        add_json('baseline/profile.json', baseline['manifest'])
    if verifier is not None:
        if digest(verifier.decode('utf-8')) != state.get('worker_identity'):
            raise ValueError('Prototype verifier identity changed')
        files['verifier/verify.py'] = verifier
    files['README.md'] = (
        '# PaperPilot prototype delivery\n\n'
        'manifest.json records the task status, measured scope and cumulative model usage.\n'
        'Engineering validation and overall task completion are separate results.\n'
        'No training accuracy, research novelty or scientific effectiveness is verified.\n\n'
        'artifact_index.json marks current versus historical artifacts. Historical passing receipts\n'
        'do not certify changed requirements. artifacts/ contains original design, validation and\n'
        'model review records; model review statements still require independent verification.\n'
        'candidates/ contains separate code copies and configs; baseline/ retains the pinned license\n'
        'when applicable. Full reading evidence is referenced by artifact ID rather than bundled.\n'
        'Raw execution receipts retain original workspace log references.\n'
        'Downloading this snapshot grants no new code execution or training approval.\n'
    ).encode('utf-8')
    if sum(map(len, files.values())) > 8*1024*1024:
        raise ValueError('Prototype delivery exceeds 8 MiB; inspect individual artifacts')
    buffer = BytesIO()
    with ZipFile(buffer, 'w', compression=ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    return buffer.getvalue()


def export_prototype(runtime, run_id):
    from .prototype_harness import PrototypeHarness
    state = PrototypeHarness(runtime)._load(run_id)
    baseline = None
    if state.get('profile_id'):
        from .prototype_profiles import load_profile
        manifest, folder = load_profile(runtime, state['profile_id'])
        baseline = {'manifest': manifest, 'source': (folder/'baseline.py').read_bytes(),
                    'license': (folder/'LICENSE').read_bytes()}
    worker = Path(__file__).with_name('prototype_worker.py').read_bytes()
    verifier = worker if digest(worker.decode('utf-8')) == state.get('worker_identity') else None
    if verifier is None:
        from .research_assets import owned_folder
        snapshots = [a['content'].get('execution_snapshot_id') for a in state['work']['artifacts'].values()
                     if a['kind']=='validate' and a['content'].get('execution_snapshot_id')]
        snapshots += [identity for identity,a in state['work']['artifacts'].items() if a['kind']=='implement']
        for identity in snapshots:
            folder = owned_folder(runtime, 'prototype-candidates', identity)
            saved = folder/'verify.py'
            if saved.is_symlink() or not saved.resolve().is_relative_to(folder.resolve()) or not saved.is_file():
                continue
            raw = saved.read_text(encoding='utf-8')
            if digest(raw) == state.get('worker_identity'):
                verifier = raw.encode('utf-8')
                break
    return build_prototype_bundle(state, runtime.db.get_run(run_id), baseline=baseline, verifier=verifier)
