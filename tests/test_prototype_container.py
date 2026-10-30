"""Opt-in real Docker checks using maintained fixtures, never generated code."""
import json
import os
from pathlib import Path

import pytest

from paperpilot_agentic.models import ExperimentProtocol, SandboxPolicy
from paperpilot_agentic.sandbox import SandboxExperimentRunner


@pytest.mark.skipif(os.environ.get('PAPERPILOT_TEST_CONTAINER') != '1', reason='Explicit container test opt-in required')
@pytest.mark.parametrize('downsample', [False, True])
def test_real_cpu_container_worker(tmp_path, downsample):
    source = tmp_path/'source'
    source.mkdir()
    expression = 'self.conv(torch.cat([x,y],dim=1))' + ('[:,:,::2,::2]' if downsample else '')
    code = ('import torch\nclass Model(torch.nn.Module):\n'
            ' def __init__(self):\n  super().__init__()\n  self.conv=torch.nn.Conv2d(6,1,1)\n'
            ' def forward(self,x,y):\n  return '+expression+'\n'
            'def build_model(config):\n return Model()\n')
    (source/'model.py').write_text(code, encoding='utf-8')
    (source/'config.json').write_text(json.dumps({}), encoding='utf-8')
    worker = Path(__file__).resolve().parents[1]/'src/paperpilot_agentic/prototype_worker.py'
    (source/'verify.py').write_bytes(worker.read_bytes())
    protocol = ExperimentProtocol(name='maintained-prototype-fixture', command=['python', 'verify.py'],
        source_dir=str(source), output_dir=str(tmp_path/'results'),
        policy=SandboxPolicy(backend='docker', docker_image='paperpilot-prototype:cpu',
            timeout_seconds=180, cpu_seconds=360, memory_mb=4096,
            network='none', allow_local_fallback=False))
    result = SandboxExperimentRunner().run(protocol)
    assert result.status == 'COMPLETED', result.error
    assert result.backend == 'docker' and result.network_isolated
    assert result.metrics['passed'] is (not downsample)
    assert result.metrics['peak_rss_kib'] > 0
    if downsample:
        assert 'input resolution' in result.metrics['error']
    else:
        assert result.metrics['parameters'] == 7
        assert len(result.metrics['checks']) == 3


@pytest.mark.skipif(os.environ.get('PAPERPILOT_TEST_CONTAINER') != '1', reason='Explicit container test opt-in required')
def test_adapter_real_failure_repair_reapproval_and_completion(tmp_path, monkeypatch):
    from test_prototype_workspace import waiting, tasks
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    state = harness._load(rid)
    first_approval = state['pending']['approval']
    first_code = state['pending']['binding']['code']
    # The initial maintained fixture returns None, intentionally violating the
    # interface. Neither attempt comes from an external/generative model.
    runtime.approve(first_approval)
    assert harness._training(rid, state)
    assert state['last_observation']['passed'] is False
    assert state['last_observation']['execution']['backend'] == 'docker'
    revised = tasks() + [{'id': 'review', 'goal': 'check repaired implementation',
                          'capability': 'review', 'depends_on': ['design', 'code', 'check'],
                          'acceptance': 'consistent'}]
    revised[2]['goal'] = 'repair constructor'
    harness._apply(rid, state, {'action': 'plan', 'reason': 'repair constructor',
                              'arguments': {'tasks': revised}}, 7)
    code = ('import torch\nclass Model(torch.nn.Module):\n'
            ' def __init__(self):\n  super().__init__()\n  self.conv=torch.nn.Conv2d(6,1,1)\n'
            ' def forward(self,x,y):\n  return self.conv(torch.cat([x,y],dim=1))\n'
            'def build_model(config):\n return Model()\n')
    harness._apply(rid, state, {'action': 'work', 'reason': 'fixed fixture', 'arguments': {
        'task_id': 'code', 'content': {'source': code, 'config': {}}}}, 8)
    harness._apply(rid, state, {'action': 'work', 'reason': 'check fixture', 'arguments': {
        'task_id': 'check', 'content': None}}, 9)
    assert state['pending']['approval'] != first_approval
    assert state['pending']['binding']['code'] != first_code
    assert harness._training(rid, state) is False  # No inherited approval.
    runtime.approve(state['pending']['approval'])
    assert harness._training(rid, state)
    assert state['last_observation']['passed'] is True
    assert state['last_observation']['execution']['network_isolated']
    harness._apply(rid, state, {'action': 'work', 'reason': 'review fixed fixture', 'arguments': {
        'task_id': 'review', 'content': {'implementation_consistent': True, 'issues': [],
                                       'limitations': ['Maintained fixture only; no research result']}}}, 10)
    harness._apply(rid, state, {'action': 'finish', 'reason': 'fixture done',
                              'arguments': {'summary': 'Fixture engineering checks complete'}}, 11)
    assert state['completed']
    harness._save(rid, state)
    calls = runtime.db.get_run(rid)['used_llm_calls']
    harness.advance(rid)  # Recover deterministic publication only.
    assert runtime.db.get_run(rid)['used_llm_calls'] == calls
    assert runtime.db.get_run(rid)['metadata']['final_answer']['scientific_validity'] == 'NOT_VERIFIED'


@pytest.mark.skipif(os.environ.get('PAPERPILOT_TEST_CONTAINER') != '1'
                    or not os.environ.get('PAPERPILOT_SNUNET_BASELINE'),
                    reason='Explicit container and pinned baseline opt-in required')
def test_real_pinned_snunet_native_cpu_contract(tmp_path):
    from paperpilot_agentic.reference_methods import SOURCE_SHA, LICENSE_SHA, sha
    raw = Path(os.environ['PAPERPILOT_SNUNET_BASELINE']).read_bytes()
    license_text = Path(os.environ['PAPERPILOT_SNUNET_BASELINE']).with_name('LICENSE').read_bytes()
    assert sha(raw) == SOURCE_SHA and sha(license_text) == LICENSE_SHA
    source = tmp_path/'source'
    source.mkdir()
    (source/'model.py').write_bytes(raw)
    (source/'LICENSE').write_bytes(license_text)
    (source/'config.json').write_text('{}', encoding='utf-8')
    (source/'contract.json').write_text('{"kind":"snunet_ecam"}', encoding='utf-8')
    worker = Path(__file__).resolve().parents[1]/'src/paperpilot_agentic/prototype_worker.py'
    (source/'verify.py').write_bytes(worker.read_bytes())
    protocol = ExperimentProtocol(name='pinned-snunet-baseline', command=['python', 'verify.py'],
        source_dir=str(source), output_dir=str(tmp_path/'results'),
        policy=SandboxPolicy(backend='docker', docker_image='paperpilot-prototype:cpu',
            timeout_seconds=180, cpu_seconds=360, memory_mb=4096, network='none', allow_local_fallback=False))
    result = SandboxExperimentRunner().run(protocol)
    assert result.status == 'COMPLETED', result.error
    assert result.metrics.get('passed') is True, result.metrics
    assert [row['shape'] for row in result.metrics['checks']] == [[1, 2, 32, 32], [2, 2, 32, 48], [1, 2, 48, 64]]


@pytest.mark.skipif(os.environ.get('PAPERPILOT_TEST_CONTAINER') != '1'
                    or not os.environ.get('PAPERPILOT_SNUNET_BASELINE'),
                    reason='Explicit container and pinned baseline opt-in required')
@pytest.mark.parametrize(('max_parameters','passed'), [(13000000,True),(12000000,False)])
def test_real_snunet_input_and_resource_constraints(tmp_path, max_parameters, passed):
    from paperpilot_agentic.reference_methods import SOURCE_SHA, LICENSE_SHA, sha
    from paperpilot_agentic.prototype_environment import preflight_environment, freeze_environment
    from paperpilot_agentic.prototype_worker import make_validation_contract
    source = tmp_path/'source'
    source.mkdir()
    baseline = Path(os.environ['PAPERPILOT_SNUNET_BASELINE'])
    raw, license_text = baseline.read_bytes(), baseline.with_name('LICENSE').read_bytes()
    assert sha(raw)==SOURCE_SHA and sha(license_text)==LICENSE_SHA
    (source/'model.py').write_bytes(raw)
    (source/'LICENSE').write_bytes(license_text)
    (source/'config.json').write_text('{}',encoding='utf-8')
    contract = make_validation_contract('snunet_ecam', {'input_cases':[[1,48,80],[2,64,48]],
        'max_parameters':max_parameters,'max_peak_rss_kib':1024*1024,'max_verifier_seconds':120})
    (source/'contract.json').write_text(json.dumps(contract),encoding='utf-8')
    (source/'verify.py').write_bytes((Path(__file__).resolve().parents[1]/'src/paperpilot_agentic/prototype_worker.py').read_bytes())
    environment = freeze_environment(preflight_environment())
    protocol = ExperimentProtocol(name='pinned-snunet-user-constraints',command=environment['command'],
        source_dir=str(source),output_dir=str(tmp_path/'results'),policy=SandboxPolicy.from_dict(environment['policy']))
    result = SandboxExperimentRunner().run(protocol)
    assert result.status=='COMPLETED' and result.metrics['passed'] is passed, result.to_dict()
    assert result.metrics['validation_contract']==contract
    assert result.metrics['parameters']==12034978
    assert [r['shape'] for r in result.metrics['checks']]==[[1,2,48,80],[2,2,64,48]]
    if not passed:
        assert result.metrics['failure_kind']=='DESIGN_CONSTRAINT'
        assert [v['metric'] for v in result.metrics['constraint_violations']]==['parameters']


@pytest.mark.skipif(os.environ.get('PAPERPILOT_TEST_CONTAINER') != '1'
                    or not os.environ.get('PAPERPILOT_SNUNET_BASELINE'),
                    reason='Explicit container and pinned baseline opt-in required')
def test_real_snunet_constraint_revision_reuses_code_and_requires_reapproval(tmp_path, monkeypatch):
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from paperpilot_agentic.prototype_harness import PrototypeHarness
    from paperpilot_agentic.prototype_profiles import register_snunet_profile
    from paperpilot_agentic.prototype_workspace import plan, complete
    from paperpilot_agentic.reference_methods import SOURCE_SHA, sha
    from test_prototype_workspace import Reader, tasks

    baseline=Path(os.environ['PAPERPILOT_SNUNET_BASELINE'])
    raw=baseline.read_bytes()
    assert sha(raw)==SOURCE_SHA
    runtime=ResearchAgentRuntime(workspace=tmp_path/'workspace',db_path=tmp_path/'runtime.db',llm=Reader())
    profile=register_snunet_profile(runtime,raw,baseline.with_name('LICENSE').read_bytes())
    monkeypatch.setattr(runtime.db,'list_evidence',lambda _: [{'evidence_id':'source','content':raw.decode()}])
    harness=PrototypeHarness(runtime)
    rid=harness.start('Maintained public baseline revision fixture','revision-container-fixture','parent',['source'],
                      profile_id=profile,constraints={'max_parameters':13000000})['run']['run_id']
    state=harness._load(rid)
    plan(state['work'],tasks()+[{'id':'review','goal':'review','capability':'review',
                                'depends_on':['design','code','check'],'acceptance':'consistent'}])
    complete(state['work'],'read','read',{'content':'Pinned public baseline fixture'})
    complete(state['work'],'design','design',{'scope':'Maintained baseline; no generated proposal'})
    code=complete(state['work'],'code','implement',{'source':raw.decode(),'config':{}})
    state['work']['repairs']=1
    harness._apply(rid,state,{'action':'work','reason':'fixed fixture',
                            'arguments':{'task_id':'check','content':None}},1)
    harness._save(rid,state)
    runtime.approve(state['pending']['approval'])
    assert harness._training(rid,state)
    first=state['validation_feedback']['result']
    assert first['passed']
    complete(state['work'],'review','review',{'implementation_consistent':True,'issues':[],
                                             'limitations':['Fixed baseline fixture; no training']})
    harness._apply(rid,state,{'action':'finish','reason':'fixture',
                            'arguments':{'summary':'Baseline fixture engineering verified'}},2)
    harness._save(rid,state)
    harness._publish(rid,state)
    before=runtime.db.get_run(rid)
    old_check=state['work']['tasks']['check']['output']
    harness.revise_constraints(rid,{'input_cases':[[1,48,80],[2,64,48]],'max_parameters':12000000},
                               expected_revision=state['work']['revision'])
    state=harness._load(rid)
    assert state['work']['tasks']['code']['output']==code and state['work']['tasks']['check']['output'] is None
    harness._apply(rid,state,{'action':'work','reason':'new requirements',
                            'arguments':{'task_id':'check','content':None}},3)
    harness._save(rid,state)
    assert harness._training(rid,state) is False  # First approval does not authorize this contract.
    runtime.approve(state['pending']['approval'])
    assert harness._training(rid,state)
    second=state['validation_feedback']['result']
    assert second['passed'] is False and second['failure_kind']=='DESIGN_CONSTRAINT'
    assert first['execution_snapshot_id']!=second['execution_snapshot_id']
    assert state['work']['artifacts'][old_check]['content']['passed'] is True
    assert state['work']['tasks']['code']['output']==code and state['work']['repairs']==1
    for key in ('created_at','budget','used_llm_calls','used_tokens'):
        assert runtime.db.get_run(rid)[key]==before[key]
    assert runtime.db.get_run(rid)['used_llm_calls']==0
    for result in (first,second):
        path=runtime.workspace/'research-harness/prototype-candidates'/result['execution_snapshot_id']/'model.py'
        assert sha(path.read_bytes())==SOURCE_SHA
