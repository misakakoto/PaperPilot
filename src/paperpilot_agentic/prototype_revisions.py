"""Atomic user revisions of goals, source versions and validation requirements."""
import copy
import json

from .models import Budget, utc_now
from .prototype_worker import make_validation_contract
from .prototype_workspace import plan, digest, task_rows
from .reference_methods import encoded
from .research_task import task_lock
from .llm_attempts import AttemptLedger
from .runtime_profile import enabled as runtime_v2
from .change_impact import analyze as analyze_impact


def revise_constraints(harness, run_id, constraints, *, expected_revision, preview=False):
    return _revise(harness, run_id, constraints=constraints, expected_revision=expected_revision, preview=preview)


def revise_goal(harness, run_id, goal, *, expected_revision, preview=False):
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError('A nonempty research goal is required')
    return _revise(harness, run_id, goal=goal, expected_revision=expected_revision, preview=preview)


def revise_sources(harness, run_id, replacements, *, expected_revision, preview=False):
    if not isinstance(replacements, dict) or not replacements:
        raise ValueError('Supply source version replacements')
    return _revise(harness, run_id, replacements=replacements, expected_revision=expected_revision, preview=preview)


def _revise(harness, run_id, *, expected_revision, constraints=None, goal=None, replacements=None, preview=False):
    kind = 'sources' if replacements is not None else 'goal' if goal is not None else 'constraints'
    label = {'sources': 'Selected sources', 'goal': 'Research goal', 'constraints': 'Validation requirements'}[kind]
    if type(expected_revision) is not int:
        raise ValueError('An expected workspace revision is required')
    with task_lock(harness.runtime.workspace, run_id) as acquired:
        if not acquired:
            raise RuntimeError('Prototype execution is active')
        original = harness._load(run_id)
        if original['phase'] not in {'READY','WAITING_APPROVAL','WAITING_ENVIRONMENT','COMPLETED','PARTIAL'}:
            raise ValueError('Select a stable prototype; reconcile failed, unknown or cancelled work first')
        harness._guard(run_id, original)
        if original['work']['revision'] != expected_revision:
            raise RuntimeError('Workspace revision changed; reload the current task')
        if AttemptLedger(harness.db).snapshot(run_id)['unknown_results']:
            raise RuntimeError('A model request is active or unknown; inspect its original receipt')
        calls = harness.db.list_tool_calls(run_id)
        if any(c['status']=='RUNNING' or c['tool_name']=='prototype.validate' and c['status']=='FAILED' for c in calls):
            raise RuntimeError('Reconcile active or uncertain execution before revising requirements')
        with harness.db.connect() as connection:
            unapplied = connection.execute("SELECT 1 FROM research_episode_turns WHERE run_id=? AND state!='APPLIED' LIMIT 1",
                                           (run_id,)).fetchone()
        if unapplied:
            raise RuntimeError('Apply or reconcile the saved model turn before changing its inputs')
        contract = (original['validation_contract'] if kind != 'constraints' else
                    make_validation_contract('snunet_ecam' if original.get('profile_id') else 'unbound', constraints))
        if kind != 'sources' and (goal == original['goal'] if goal is not None else contract == original.get('validation_contract')):
            if preview:
                return {'revision_applied':False,'preview':True,'change_impact':analyze_impact(original,original,kind)}
            return harness.status(run_id) | {'revision_applied': False}
        code_tasks = [t for t in original['work']['tasks'].values() if t['capability']=='implement']
        validators = [t for t in original['work']['tasks'].values() if t['capability']=='validate']
        if kind == 'constraints' and (not code_tasks or not all(t['status']=='DONE' for t in code_tasks) or not validators):
            raise ValueError('Complete the current model candidate before revising validation constraints')
        updated = copy.deepcopy(original)
        rows = task_rows(updated['work'])
        if kind == 'sources':
            from .prototype_source_revisions import prepare
            rows = prepare(harness, original, updated, replacements)
            if rows is None:
                if preview:
                    return {'revision_applied':False,'preview':True,'change_impact':analyze_impact(original,original,kind)}
                return harness.status(run_id) | {'revision_applied': False}
        for task in rows:
            if goal is not None and task['capability'] != 'read':
                task['acceptance'] += '\nHost research goal '+digest(goal)
            elif kind == 'constraints' and task['capability']=='validate':
                task['acceptance'] = 'Host validation contract '+digest(contract)
        if rows:
            plan(updated['work'], rows, required_evidence=updated.get('required_evidence_ids', ()),
                 required_sources=updated.get('implementation_source_ids', ()))
        else:
            updated['work']['revision'] += 1
        reused = [] if kind != 'constraints' else [t['output'] for t in code_tasks]
        if goal is not None:
            updated['goal'] = updated['work']['goal'] = goal
        if kind != 'constraints':
            updated['implementation_feedback'] = None
        updated.update(validation_contract=contract, validation_contract_identity=digest(contract),
                       phase='READY', completed=False, stop_reason=None, validation_feedback=None,
                       consecutive_rejections=0, plan_requires_work=False,
                       last_observation={'status':{'sources':'SOURCES_REVISED', 'goal':'GOAL_REVISED', 'constraints':'CONTRACT_REVISED'}[kind],
                                         **({'replacements':replacements} if kind == 'sources' else {}),
                                         'previous_revision':expected_revision, 'code_reused':reused})
        updated.pop('pending', None)
        updated.pop('environment_request', None)
        impact = analyze_impact(original,updated,kind)
        if preview:
            return {'revision_applied':False,'preview':True,'change_impact':impact}
        if runtime_v2(updated):
            updated['change_impact'] = impact
            updated.setdefault('change_impacts',[]).append(impact)
        now = utc_now()
        with harness.db.transaction() as connection:
            saved = connection.execute('SELECT state_json FROM research_episodes WHERE run_id=?',(run_id,)).fetchone()
            run = connection.execute('SELECT * FROM runs WHERE run_id=?',(run_id,)).fetchone()
            metadata = json.loads(run['metadata_json'])
            if json.loads(saved['state_json']) != original:
                raise RuntimeError('Workspace revision changed during update')
            if metadata.get('cancel_requested') or run['status']=='CANCELLED':
                raise ValueError('Cancelled task cannot be reopened by a requirements update')
            if connection.execute("SELECT 1 FROM tool_calls WHERE run_id=? AND status='RUNNING' LIMIT 1",(run_id,)).fetchone():
                raise RuntimeError('Execution became active during revision')
            if connection.execute("SELECT 1 FROM llm_attempts WHERE run_id=? AND state IN ('STARTED','UNKNOWN') LIMIT 1",(run_id,)).fetchone():
                raise RuntimeError('A model request became active or unknown during revision')
            obsolete = connection.execute("""SELECT a.approval_id,a.status,a.tool_call_id FROM approvals a
                JOIN tool_calls c ON c.tool_call_id=a.tool_call_id WHERE a.run_id=? AND c.tool_name='prototype.validate'
                AND c.status IN ('PENDING','WAITING_APPROVAL','APPROVED')""",(run_id,)).fetchall()
            for approval in obsolete:
                if approval['status']=='PENDING':
                    connection.execute("UPDATE approvals SET status='REJECTED',decision_json=?,decided_at=? WHERE approval_id=?",
                        (encoded({'actor':kind+'-revision','reason':'Superseded '+label.lower()}).decode(),
                         now,approval['approval_id']))
                connection.execute("UPDATE tool_calls SET status='REJECTED',error=?,completed_at=?,updated_at=? WHERE tool_call_id=?",
                    ('Superseded '+label.lower()+'; no execution',now,now,approval['tool_call_id']))
            updated.setdefault({'sources':'source_revisions','goal':'goal_revisions','constraints':'constraint_revisions'}[kind],[]).append({
                **({'previous_goal':original['goal'],
                    'previous_implementation_feedback':original.get('implementation_feedback')}
                   if goal is not None else {}),
                **({'replacements':replacements, 'previous_sources':original['sources'],
                    'previous_implementation_feedback':original.get('implementation_feedback')} if kind == 'sources' else {}),
                'previous_revision':expected_revision,'previous_contract':original.get('validation_contract'),
                'previous_tasks':original['work']['tasks'],'previous_phase':original['phase'],
                'previous_stop_reason':original.get('stop_reason'),
                'previous_summary':(metadata.get('final_answer') or {}).get('answer'),
                'superseded_approvals':[a['approval_id'] for a in obsolete]})
            budget = Budget.from_dict(json.loads(run['budget_json']))
            exhausted = (updated['turn']>=updated['limits']['max_turns'] or
                         not budget.unlimited_llm_usage and run['used_llm_calls']>=budget.max_llm_calls)
            if exhausted:
                updated.update(phase='PARTIAL',stop_reason=label+' saved; existing decision budget exhausted')
            metadata['research_episode'] = harness.view(run_id,state=updated)
            metadata['final_answer'] = ({'answer':updated['stop_reason'],'prototype':harness.view(run_id,state=updated),
                                         'scientific_validity':'NOT_VERIFIED'} if exhausted else None)
            connection.execute('UPDATE research_episodes SET state_json=? WHERE run_id=?',(encoded(updated).decode(),run_id))
            connection.execute('UPDATE runs SET status=?,metadata_json=?,error=NULL,current_step_id=NULL,updated_at=? WHERE run_id=?',
                               ('COMPLETED' if exhausted else 'PENDING',encoded(metadata).decode(),now,run_id))
            connection.execute('INSERT INTO events(run_id,event_type,payload_json,created_at) VALUES(?,?,?,?)',
                (run_id,'prototype.'+kind+'.revised',encoded({'revision':updated['work']['revision'],
                 'contract_identity':digest(contract),'code_reused':reused,
                 'budget_reset':False}).decode(),now))
        return harness.status(run_id) | {'revision_applied': True}


def preview_change(harness, run_id, *, kind, value, expected_revision):
    functions = {'goal':revise_goal,'sources':revise_sources,'constraints':revise_constraints}
    if kind not in functions:
        raise ValueError('Unknown revision kind')
    return functions[kind](harness,run_id,value,expected_revision=expected_revision,preview=True)
