"""Append missing model analysis to a verified experiment without new GPU authority.

The original turn, partial publication, candidates and GPU ledger stay in place.
This explicitly versioned continuation is narrower than reopening a research task.
"""
import json

from .models import ResearchOutcome, RunStatus, utc_now
from .reference_methods import encoded, sha
from .research_assets import immutable
from .research_harness import ResearchHarness
from .research_task import task_lock
from .research_training import episode_folder
from .training_state import file_sha


class AnalysisContinuation(ResearchHarness):
    def _guard(self, run_id, state):
        authority = self.db.get_run(run_id)['metadata'].get('analysis_continuation')
        if not authority or authority['implementation_sha256'] != file_sha(__file__):
            raise ValueError('Analysis continuation implementation changed')
        return super()._guard(run_id,state)

    def _context(self, run_id, state):
        key,messages=super()._context(run_id,state)
        if state['role'] not in {'analyst','lead'}:
            raise ValueError('Analysis continuation has no implementation role')
        messages[0]['content'] += (' This is an explicit read-only analysis continuation. '
            'No new candidate, experiment, approval or GPU execution is authorized. '
            'The analyst may inspect_metrics/report; the lead may ONLY stop with a justified '
            'next-step recommendation after the analyst has reported. All seeds have equal standing. '
            'Metrics and deltas use fractions on 0..1, not percentage points. '
            'Performance changes alone cannot confirm or refute an unmeasured activation mechanism. '
            'Three seeds cannot establish statistical significance.')
        task=json.loads(messages[1]['content'])
        if state['role']=='lead':
            task['schema']={'action':'stop','reason':'brief rationale using the analysis',
                            'arguments':{'summary':'why stop, limitations and a future experiment if justified'}}
            if state.get('state_contract'):
                from .research_contracts import research_action_schema
                task['schema'] = {'stop':research_action_schema('lead',reviewed_result=True)['stop']}
        task['continuation_contract']='analysis-only-v1; original stop/publication retained'
        messages[1]['content']=encoded(task).decode()
        return key,messages

    def _apply(self, run_id, state, value, index):
        allowed={'analyst':{'inspect_metrics','report'},'lead':{'stop'}}
        if not isinstance(value,dict) or value.get('action') not in allowed.get(state['role'],set()):
            raise ValueError('Analysis-only continuation cannot generate or execute another candidate')
        if value['action']=='stop' and not state['candidates'][-1].get('analysis'):
            raise ValueError('Independent analyst report required before stopping')
        return super()._apply(run_id,state,value,index)

    def _training(self,*args):
        raise ValueError('Analysis-only continuation has no GPU authority')

    def _publish(self,run_id,state):
        authority=self.db.get_run(run_id)['metadata']['analysis_continuation']
        report=self.view(run_id,state=state) | {'analysis_continuation':authority}
        path=immutable(episode_folder(self.runtime,run_id),
                       f'report-analysis-continuation-{state["turn"]}.json',encoded(report))
        if not any(a['path']==str(path) for a in self.db.list_artifacts(run_id)):
            self.db.add_artifact(run_id,None,{'kind':'research-episode-report','path':str(path),
                                            'sha256':sha(path.read_bytes())})
        self.db.update_run_metadata(run_id,{'final_answer':{
            'answer':'科研任务闭环已完成（独立分析续行）。' if state['completed'] else '科研任务分析部分完成。',
            'limitations':[state['stop_reason'],'模型解释需科学复核，不证明显著性、泛化或创新。'],
            'review_required':True,'citation_ids':[]}})
        self.db.update_run_outcome(run_id,ResearchOutcome.PARTIAL)
        self.db.update_run_status(run_id,RunStatus.COMPLETED)


def continue_analysis(runtime,run_id,*,max_turns=None):
    harness=AnalysisContinuation(runtime)
    with task_lock(runtime.workspace,run_id) as acquired:
        if not acquired:
            return harness.status(run_id)
        run=runtime.db.get_run(run_id)
        state=harness._load(run_id)
        if not run['metadata'].get('analysis_continuation'):
            if (state['phase']!='PARTIAL' or len(state['candidates'])!=1
                    or state['candidates'][0].get('analysis')):
                raise ValueError('Only a single fully trained, unanalyzed partial task can continue analysis')
            candidate=state['candidates'][0]
            result=candidate.get('result',{})
            if (result.get('status')!='TRAINING_VERIFIED' or len(result.get('trials',[]))!=3
                    or any(row.get(variant,{}).get('status')!='VERIFIED'
                           for row in result['trials'] for variant in ('baseline','candidate'))):
                raise ValueError('All paired experiments must already be verified')
            # Validate the original, still-frozen experiment before adding authority.
            ResearchHarness._guard(harness,run_id,state)
            original=episode_folder(runtime,run_id)/'report.json'
            if not original.exists() or json.loads(original.read_bytes()).get('phase')!='PARTIAL':
                raise ValueError('Original partial publication must remain available')
            authority={'version':'analysis-only-v1','at':utc_now(),
                'implementation_sha256':file_sha(__file__), 'original_turn':state['turn'],
                'original_report_sha256':file_sha(original),
                'previous_final_answer':run['metadata'].get('final_answer'),
                'gpu_budget_before':harness.view(run_id)['gpu_budget'],
                'training_authority':'NONE; reuse original verified receipts only'}
            immutable(episode_folder(runtime,run_id),'analysis-continuation-authority.json',encoded(authority))
            runtime.db.update_run_metadata(run_id,{'analysis_continuation':authority})
            state.update(phase='READY',role='analyst',completed=False,stop_reason=None)
            state['last_observation']={'action':'continue_missing_independent_analysis',
                'original_stop_turn':state['turn'],'no_training_replay':True}
            harness._save(run_id,state)
            runtime.db.update_run_status(run_id,RunStatus.PENDING)
            runtime.db.emit_event(run_id,'research.analysis_continued',authority)
    return harness.advance(run_id,max_turns=max_turns)
