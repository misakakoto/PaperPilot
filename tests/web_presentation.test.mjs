import test from 'node:test';
import assert from 'node:assert/strict';
import {pendingMessage,protocolSummary,pairedDiagnosticTable,answerSections,budgetSummary} from '../src/paperpilot_agentic/web_static/render.js';

test('model quota, local spending, window and unknown have actionable messages',()=>{
  for(const title of ['模型免费额度耗尽','本地预算不足','上下文过长','请求结果未知']) {
    assert.match(pendingMessage({status:'FAILED',failure:{title,action:'保留原请求'}}),new RegExp(title));
  }
  const text=budgetSummary({models:['primary','backup'],budget:{used_llm_calls:2,used_tokens:15,
    max_tokens:200000,reserved_unknown_tokens:2000,warning:true}});
  assert.match(text,/2 次实际请求/); assert.match(text,/已确认 15 \/ 200000/);
  assert.match(text,/未确认用量预留 2000/); assert.match(text,/primary → backup/);
  assert.match(text,/80%/);
  const unlimited=budgetSummary({budget:{used_llm_calls:8,used_tokens:103205,
    max_tokens:400000,unlimited_llm_usage:true}});
  assert.match(unlimited,/8 次实际请求.*已确认 103205 tokens.*不设本地累计模型预算/);
  assert.doesNotMatch(unlimited,/400000/);
});

test('answer sections preserve full text, and legacy or invalid sections fall back',()=>{
  const message={answer:'A [e1]\n\nCondition [e2]',answer_sections:[
    {kind:'direct',text:'A [e1]'},{kind:'context',text:'Condition [e2]'}]};
  assert.deepEqual(answerSections(message),message.answer_sections);
  assert.equal(answerSections({answer:message.answer}),null);
  assert.equal(answerSections({...message,answer:'A [e1]'}),null);
  assert.equal(answerSections({...message,answer_sections:[{kind:'direct',text:'new claim'}]}),null);
  assert.equal(answerSections({answer:'A\n\nB',answer_sections:[
    {kind:'direct',text:'A'},{kind:'direct',text:'B'}]}),null);
});

test('experiment activity is not falsely described as paper retrieval',()=>{
  const text=pendingMessage({status:'RUNNING',research:{engine:'method_comparison'}});
  assert.match(text,/已批准的短实验/); assert.doesNotMatch(text,/检索/);
  assert.match(pendingMessage({status:'RUNNING',research:{engine:'research_design'}}),/不会启动训练/);
});
test('waiting, stopping and missing answers preserve outcome boundaries',()=>{
  assert.match(pendingMessage({status:'WAITING_APPROVAL',research_task:{state:'AWAITING_BUDGET'}}),/研究范围与总预算/);
  assert.doesNotMatch(pendingMessage({status:'WAITING_APPROVAL',research_task:{state:'UNKNOWN'}}),/确认下方协议/);
  assert.match(pendingMessage({status:'WAITING_APPROVAL'}),/批准前不会执行/);
  assert.match(pendingMessage({status:'STOPPING'}),/正在等待/);
  assert.match(pendingMessage({status:'EXECUTION_UNKNOWN'}),/不会因刷新/);
  assert.match(pendingMessage({status:'COMPLETED'}),/不表示论文中一定没有答案/);
});
test('proposal and approval use frozen numbers, never fixed display constants',()=>{
  const rows=Object.fromEntries(protocolSummary({binding_id:'frozen',
    data_scope:{dataset:'Fixture',train_images:3,val_images:5,test_images:0},
    protocol:{seeds:[42],variants:['baseline','candidate'],steps_per_variant:4,max_total_optimizer_steps:8},
    limits:{timeout_seconds:90,max_output_mb:12}}));
  assert.match(rows['训练规模'],/42 × 2 变体 × 4 次更新；共 8/);
  assert.match(rows['数据'],/训练 3 张，验证 5 张；不使用测试集/);
  assert.match(rows['运行上限'],/90 秒/); assert.equal(rows['协议绑定'],'frozen');
  assert.doesNotMatch(Object.fromEntries(protocolSummary({data_scope:{dataset:'Unknown'}}))['数据'],/不使用测试集/);
});

test('paired presentation preserves zero and distinguishes engineering from science',()=>{
  const data={status:'PAIRED_DIAGNOSTIC_VERIFIED',scientific_acceptance:'NOT_ASSESSED',trials:[{
    seed:7,baseline:{validation:{f1:0,iou:0}},candidate:{validation:{f1:0.234567,iou:0.123456}}}]};
  const original=JSON.stringify(data),text=pairedDiagnosticTable(data);
  assert.match(text,/7 \| 基线 \| 0.0000 \| 0.0000/);
  assert.match(text,/0.2346 \| 0.1235/); assert.match(text,/不证明方法有效、无效或非劣/);
  assert.equal(JSON.stringify(data),original);
  assert.equal(pairedDiagnosticTable({...data,status:'FAILED'}),null);
  assert.equal(pairedDiagnosticTable({...data,scientific_acceptance:'PROVED'}),null);
  data.trials[0].candidate.validation.f1=NaN;
  assert.equal(pairedDiagnosticTable(data),null);
});

test('v2 preserves all seeds, negative deltas and unfinished trials',()=>{
  const rows=[7,11,19].map(seed=>({seed,baseline:{status:'VERIFIED',metrics:{precision:0,recall:0,f1:0,iou:0}},candidate:{status:'NOT_RUN'}}));
  const data={status:'PARTIAL',scientific_acceptance:'DESCRIPTIVE_ONLY',trials:rows};
  assert.match(pairedDiagnosticTable(data),/19 \| 候选 \| NOT_RUN/);
  assert.match(pairedDiagnosticTable(data),/0.0000 \| 0.0000/);
  assert.equal(pairedDiagnosticTable({...data,trials:rows.slice(0,2)}),null);
  const research={...data,status:'TRAINING_VERIFIED'};
  assert.equal(pairedDiagnosticTable(research),pairedDiagnosticTable(data));
  assert.equal(pairedDiagnosticTable({...research,scientific_acceptance:'PROVED'}),null);
  const summary=Object.fromEntries(protocolSummary({protocol:{version:2,seeds:[7,11,19],variants:['baseline','candidate'],epochs:20,batch_size:4},limits:{gpu_cap_seconds:28800,checkpoint_interval:25}}));
  assert.match(summary['训练规模'],/20 epochs/); assert.match(summary['运行上限'],/28800/);
  assert.match(pendingMessage({status:'RUNNING',research:{engine:'method_training'}}),/checkpoint/);
});
