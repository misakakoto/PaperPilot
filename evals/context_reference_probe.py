"""Read-only synthetic boundary probes of a local PaperStorm checkout.

Loads only its two standalone context modules, not the full app. This is not a
public benchmark, accuracy measurement, or an end-to-end comparison.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    package = types.ModuleType('context_probe_reference')
    package.__path__ = [str(args.reference/'knowledge_storm')]
    sys.modules[package.__name__] = package
    for name in ('paperstorm_context_v42','paperstorm_context_v56'):
        spec = importlib.util.spec_from_file_location(package.__name__+'.'+name,args.reference/'knowledge_storm'/f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    config = module.ContextEngineConfigV56(model_context_tokens=4096,operational_input_tokens=3000,
        output_reserve_tokens=512,recent_messages=1)
    engine = module.ContextEngineV56(config=config)
    pinned = {'id':'pinned','role':'system','content':'MUST_KEEP ' * 10000}
    oversized = engine.assemble([pinned,{'id':'q','role':'user','content':'hello'}])
    with tempfile.TemporaryDirectory() as directory:
        engine = module.ContextEngineV56(config=config,ledger=module.ContextLedgerV56(Path(directory)/'ledger.db'))
        first = engine.compact([{'id':'goal','role':'user','content':'UNIQUE_OLD_GOAL_42 must remain.'},
                                {'id':'recent','role':'user','content':'Recent question.'}],force=True)
        second = engine.compact(first['messages']+[{'id':'next','role':'user','content':'Next question.'}],force=True)
        result = {
            'scope':'synthetic context module boundary tests, not model accuracy',
            'oversized_pinned':{'validation':oversized['validation'],
                'pinned_sent':any(m.get('id')=='pinned' for m in oversized['messages'])},
            'recursive_compaction':{
                'first_has_old_goal':'UNIQUE_OLD_GOAL_42' in json.dumps(first['messages']),
                'second_has_old_goal':'UNIQUE_OLD_GOAL_42' in json.dumps(second['messages']),
                'raw_restore_has_old_goal':'UNIQUE_OLD_GOAL_42' in json.dumps(engine.restore(second['compaction']['compaction_id']))}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
