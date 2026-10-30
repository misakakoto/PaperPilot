"""Web adapter; paths and execution commands never come from the browser."""
from .workbench import ResearchWorkbench


def mount_workbench(router, runtime, background, call):
    from fastapi import Body, HTTPException
    from fastapi.responses import JSONResponse, Response
    workbench = ResearchWorkbench(runtime,background)

    @router.get('/conversations')
    def conversations(cursor: str | None = None, limit: int = 30):
        return call(lambda _: workbench.conversations(cursor=cursor,limit=limit))

    @router.get('/conversations/{conversation_id}/turns')
    def turns(conversation_id: str, cursor: str | None = None, limit: int = 20):
        return call(lambda _: workbench.runs(conversation_id,cursor=cursor,limit=limit))

    @router.get('/conversations/{conversation_id}/research-state')
    def research_state(conversation_id: str, offset: int = 0, limit: int = 20):
        return call(lambda _: workbench.journal.view(conversation_id,offset=offset,limit=limit))

    @router.post('/turns')
    def submit(payload: dict = Body(...)):
        result = call(lambda _: workbench.submit(payload))
        return JSONResponse(status_code=202 if result.get('start') else 200,content=result)

    @router.get('/runs/{run_id}/view')
    def turn(run_id: str):
        return call(lambda _: workbench.turn(run_id))

    @router.get('/runs/{run_id}/materials')
    def materials(run_id: str):
        return call(lambda _: workbench.materials(run_id))

    @router.post('/runs/{run_id}/change-impact')
    def change_impact(run_id: str, payload: dict = Body(...)):
        from .prototype_harness import PrototypeHarness
        if set(payload) != {'kind','value','expected_revision'}:
            raise HTTPException(status_code=400,detail='Expected kind, value and expected_revision')
        return call(lambda _: PrototypeHarness(runtime).preview_change(run_id,**payload))

    @router.get('/runs/{run_id}/prototype-bundle')
    def prototype_bundle(run_id: str):
        from .prototype_delivery import export_prototype
        content = call(lambda _: export_prototype(runtime, run_id))
        return Response(content, media_type='application/zip',
                        headers={'Content-Disposition': 'attachment; filename="paperpilot-prototype.zip"'})

    @router.get('/papers')
    def papers(query: str = '', offset: int = 0, limit: int = 30):
        return call(lambda _: workbench.papers(query,offset,limit))

    @router.get('/prototype-profiles')
    def prototype_profiles():
        def registered(_):
            from .prototype_profiles import load_profile
            items, stale = [], []
            for path in sorted((runtime.workspace/'research-harness'/'prototype-profiles').glob('*/profile.json')):
                try:
                    manifest, _ = load_profile(runtime, path.parent.name)
                    items.append({'profile_id': path.parent.name, **manifest})
                except (ValueError, OSError, KeyError):
                    stale.append({'profile_id': path.parent.name, 'status': 'STALE'})
            return {'items': items, 'stale_items': stale}
        return call(registered)

    @router.get('/prototype-environment')
    def prototype_environment():
        from .prototype_environment import preflight_environment
        return call(lambda _: preflight_environment())

    @router.get('/research-recipes')
    def research_recipes():
        def owned_recipes(_):
            folder = runtime.workspace/'research-harness'/'recipes'
            items, stale = [], []
            for path in sorted(folder.glob('*/recipe.json')):
                from .research_assets import load_recipe
                try:
                    recipe, _, _ = load_recipe(runtime, path.parent.name, check_environment=False)
                except (ValueError, OSError) as exc:
                    stale.append({'recipe_id': path.parent.name, 'status': 'STALE', 'reason': str(exc)})
                    continue
                items.append({'recipe_id': path.parent.name, 'version': recipe['version'],
                    'gpu_cap_seconds': recipe['gpu_cap_seconds'], 'validation_scope': recipe['validation_scope'],
                    'source_ids': [s['evidence_id'] for group in recipe['sources'].values() for s in group]})
            return {'items': items, 'stale_items': stale}
        return call(owned_recipes)

    @router.get('/imported-experiments')
    def imported_experiments():
        def registered(_):
            from .research_diagnostics import load_bundle
            items, stale = [], []
            folder = runtime.workspace / 'research-harness' / 'diagnostic-inputs'
            for path in sorted(folder.glob('*/bundle.json')):
                try:
                    bundle = load_bundle(runtime, {'identity': path.parent.name, 'bundle_sha256': path.parent.name})
                    if bundle.get('source_kind') != 'imported_predictions':
                        continue
                    cases = []
                    from .research_case import load_case, case_navigation
                    for case_path in (runtime.workspace/'research-harness/research-cases').glob('*/case.json'):
                        try:
                            case = load_case(runtime,case_path.parent.name,path.parent.name)
                            cases.append(case_navigation(case,case_path.parent.name))
                        except (ValueError,OSError,KeyError):
                            continue
                    items.append({'bundle_id': path.parent.name, 'research_cases':cases, 'dataset': bundle['dataset'],
                        'seeds': bundle['seeds'], 'verification': bundle['verification']})
                except (ValueError, OSError, KeyError):
                    stale.append({'bundle_id': path.parent.name, 'status': 'STALE'})
            return {'items': items, 'stale_items': stale}
        return call(registered)

    @router.get('/research-cases/{case_id}')
    def research_case(case_id: str):
        def registered_case(_):
            from .research_case import load_case, case_navigation
            value = load_case(runtime,case_id,None)
            return {**case_navigation(value,case_id),
                'sources':[s['snapshot'] for s in value['sources']],
                'candidate_diff':value['approved_candidate'].get('diff'),
                'experiment_bundle_id':value['experiment_bundle_id'],
                'verification':value['imported_verification'],
                'execution_authority':'NONE; historical evidence only'}
        return call(registered_case)

    @router.get('/runs/{run_id}/artifacts/{artifact_id}/preview')
    def artifact(run_id: str, artifact_id: str):
        return call(lambda _: workbench.artifact(run_id,artifact_id))

    @router.get('/submissions/{request_id}')
    def submission(request_id: str):
        with runtime.db.connect() as conn:
            row = conn.execute('SELECT state,result_json FROM workbench_submissions WHERE request_id=?',(request_id,)).fetchone()
        if row is None:
            raise HTTPException(404,'No submission with this ID')
        import json
        return {'state':row['state'],'result':json.loads(row['result_json']) if row['result_json'] else None}

    return workbench
