import test from 'node:test';
import assert from 'node:assert/strict';
import {researchPresentation, submissionNeedsOpen, researchReportNotice} from '../src/paperpilot_agentic/web_static/research.js';

test('first budget-only submission opens its conversation without launching work',()=>{
  assert.equal(submissionNeedsOpen({start:false,conversation_id:'new',run_id:'r'},null,[]),true);
  assert.equal(submissionNeedsOpen({start:false,conversation_id:'new',run_id:'r'},'new',[{run_id:'r'}]),false);
});

test('reading coverage never implies answered questions or training authorization',()=>{
  const view=researchPresentation({state:'PAUSED_BUDGET',coverage:[{state:'READ'},{state:'PENDING'}],
    pause_reason:'Budget exhausted'});
  assert.equal(view.progress,'已读取 1/2 篇');
  assert.match(view.note,/不代表/);
  assert.equal(view.action,'continue_research');
  assert.equal(researchPresentation({state:'UNKNOWN',coverage:[]}).action,null);
});

test('a saved partial answer is visibly marked as an unfinished stage result',()=>{
  assert.match(researchReportNotice({research_task:{state:'UNKNOWN'},message:{answer:'A cited fact'}}),/阶段材料/);
  assert.equal(researchReportNotice({research_task:{state:'RUNNING'},message:{answer:''}}),null);
  assert.equal(researchReportNotice({research_task:{state:'FINISHED'},message:{answer:'A cited fact'}}),null);
});
