"""Browser progress reads owned receipts; fixtures do not execute training."""
import json

from paperpilot_agentic.reference_methods import encoded, sha
from paperpilot_agentic.research_training import episode_folder, gpu_ledger
from paperpilot_agentic.workbench import ResearchWorkbench
from test_research_harness import harness, start
from test_workbench import Background


def setup(h):
    rid=start(h)
    h.advance(rid,max_turns=2)
    state=h._load(rid)
    card=state['candidates'][0]['card']
    protocol={'epochs':20,'train_images':512,'batch_size':4}
    card['protocol']=protocol
    h._save(rid,state)
    root=episode_folder(h.runtime,rid)/card['binding_id']
    folder=root/'trials'/'7-baseline'
    pointer=folder/'checkpoints'/'latest.json'
    pointer.parent.mkdir(parents=True)
    identity={'binding_id':card['binding_id'],'protocol_sha256':sha(encoded(protocol)),
              'stage':'formal','seed':7,'variant':'baseline'}
    pointer.write_text(json.dumps({'identity':identity,'step':25}))
    (root/'state.json').write_text(json.dumps({'trials':{'7-baseline':{'status':'VERIFIED'}}}))
    ledger=gpu_ledger(h.runtime,rid)
    ledger.reserve(str(folder),60)
    return rid,pointer,ledger,ResearchWorkbench(h.runtime,Background())


def test_browser_shows_owned_cursor_without_new_calls_or_training(harness):
    h,provider,launches=harness
    rid,pointer,ledger,wb=setup(h)
    before=ledger.snapshot()
    calls=len(provider.messages)
    progress=wb.turn(rid)['research_episode']['execution_progress']
    assert progress['recorded_trials']==1 and progress['total_trials']==6
    assert progress['current']=={'stage':'formal','seed':7,'variant':'baseline',
                                'step':25,'target':2560,'process_status':'RESERVED'}
    assert ledger.snapshot()==before and len(provider.messages)==calls and launches==[]
    assert 'not a checkpoint payload' in progress['assurance']


def test_wrong_identity_and_truncated_pointer_are_not_progress_or_failure(harness):
    h,_,launches=harness
    rid,pointer,ledger,wb=setup(h)
    value=json.loads(pointer.read_text())
    value['identity']['binding_id']='f'*64
    pointer.write_text(json.dumps(value))
    assert wb.turn(rid)['research_episode']['execution_progress']['current'] is None
    pointer.write_text('{')
    assert wb.turn(rid)['research_episode']['execution_progress']['current'] is None
    assert h._load(rid)['phase']=='READY' and launches==[]


def test_ledger_paths_cannot_redirect_browser_to_foreign_pointer(harness,tmp_path):
    h,_,_=harness
    rid,pointer,ledger,wb=setup(h)
    foreign=tmp_path/'outside'
    foreign.mkdir()
    (foreign/'latest.json').write_text(pointer.read_text())
    ledger.finish(ledger.snapshot()['launches'][0]['token'],'COMPLETED',0)
    ledger.reserve(str(foreign),60)
    assert wb.turn(rid)['research_episode']['execution_progress']['current'] is None
