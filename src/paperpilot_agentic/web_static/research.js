import {el,button,details,markdown} from './render.js';

export function comparisonTable(task,citations=[],onCitation=()=>{}) {
  const comparison=task.comparison,root=el('div');
  if(!comparison)return root;
  const fields={method:'本文方法',module:'方法模块',task:'实验任务',dataset:'数据集',split:'划分',
    protocol:'训练与评估条件',metric:'指标口径',result:'报告结果'};
  root.append(el('h3','证据比较表'),el('p','每行的字段及其联合实验归属经过模型审核；条件不足或结果冲突的记录不进入条件对照组。','state'));
  const wrap=el('div',undefined,'table-wrap'),table=el('table'),head=el('tr');
  ['论文',...Object.values(fields),'记录状态'].forEach(label=>head.append(el('th',label)));table.append(head);
  comparison.records.forEach(record=>{
    const row=el('tr');row.append(el('td',record.title));
    for(const key of Object.keys(fields)) {
      const cell=el('td'),value=record.cells[key];
      cell.append(value?markdown(value.value+' '+value.evidence_ids.map(id=>`[${id}]`).join(' '),citations,onCitation):el('span','待核对'));
      row.append(cell);
    }
    row.append(el('td',record.status==='CONFLICT'?'结果冲突':record.status==='PARTIAL'?'部分字段待核对':'字段与归属已审核'));
    table.append(row);
  });wrap.append(table);root.append(wrap);
  root.append(el('p',`${comparison.condition_groups.length} 组记录的已提取条件一致；不自动推断性能排名。`));
  comparison.coverage.forEach(p=>{
    if(p.gaps.length)root.append(el('p',`${p.title}：${p.gaps.join('；')}`,'state'));
  });return root;
}

export function submissionNeedsOpen(result,cid,turns) {
  return result.start || result.conversation_id!==cid || !turns.some(t=>t.run_id===result.run_id);
}

export function researchReportNotice(turn) {
  if(!turn.research_task || !turn.message?.answer || turn.research_task.state==='FINISHED')return null;
  return '阶段材料：这里只保留已审核章节，整个研究任务尚未完成；请同时查看缺口与停止原因。';
}

export function researchPresentation(task) {
  const read=task.coverage.filter(p=>p.state==='READ').length;
  const labels={AWAITING_BUDGET:'确认研究范围与总预算',READY:'已确认，等待开始',RUNNING:'正在分批研究',
    PAUSED_BUDGET:'预算不足，进度已保存',PAUSED_CANCELLED:'已停止，未开始的批次保留',
    UNKNOWN:'请求结果未知，禁止自动重试',FINISHED:'报告处理结束，结论仍需核验',FAILED:'部分处理失败'};
  return {title:labels[task.state] || task.state,progress:`已读取 ${read}/${task.coverage.length} 篇`,
    note:'读取完成不代表问题已回答；预算确认不授权实验或训练。',
    action:task.state==='AWAITING_BUDGET'?'approve_research':
      ['PAUSED_BUDGET','PAUSED_CANCELLED'].includes(task.state)?'continue_research':null};
}

export function coverageTable(task) {
  const wrap=el('div',undefined,'table-wrap'),table=el('table'),head=el('tr');
  ['论文','阅读状态','原文片段','尚缺材料'].forEach(x=>head.append(el('th',x)));table.append(head);
  const states={PENDING:'待处理',READ:'已读取（非正确性认证）',NO_EVIDENCE:'未取得证据'};
  task.coverage.forEach((p,index)=>{
    const row=el('tr');
    [p.title || `论文 ${index+1} · ${p.paper_sha256.slice(0,8)}`,states[p.state] || p.state,
      String(p.evidence_count),p.gaps.join('；') || '未标记缺口，不代表证据完整'].forEach(x=>row.append(el('td',x)));
    table.append(row);
  });wrap.append(table);return wrap;
}

export function researchCard(turn,{submit,openCoverage,editable=true}) {
  const task=turn.research_task,view=researchPresentation(task);
  const card=el('section',undefined,'approval research-task');
  card.append(el('h3',view.title),el('p',view.progress),el('p',view.note,'state'));
  if(task.pause_reason)card.append(el('p',task.pause_reason,'warning'));
  card.append(el('p',`已使用 ${task.usage.used_llm_calls} 次请求 · ${task.usage.used_tokens} tokens · ${task.usage.used_tool_calls} 次工具操作`));
  const dimensions=el('ul');task.dimensions.forEach(x=>dimensions.append(el('li',x)));
  card.append(details('本任务的研究问题与维度',dimensions));
  card.append(button('查看逐篇覆盖与章节状态',openCoverage));
  if(task.budget_basis)card.append(el('p',task.budget_basis,'state'));
  if(view.action) {
    const base=task.approved_budget || task.suggested_budget;
    const limits={...base};
    if(task.approved_budget)for(const key of ['max_llm_calls','max_tokens','max_tool_calls','max_seconds']) {
      const used=key==='max_seconds'?task.active_seconds:task.usage[
        {max_llm_calls:'used_llm_calls',max_tokens:'used_tokens',max_tool_calls:'used_tool_calls'}[key]];
      limits[key]=Math.max(base[key],task.suggested_budget[key],Math.ceil(used+task.suggested_budget[key]));
    }
    card.append(el('p','下面是可调整的累计上限，不是承诺消费量。追加额度不清零已用量；未配置可靠价格时不估算金额。','state'));
    const grid=el('div',undefined,'research-budget-grid'),inputs={};
    for(const [key,label] of [['max_llm_calls','实际请求总上限'],['max_tokens','Token 总上限'],
      ['max_tool_calls','工具操作总上限'],['max_seconds','累计运行秒数上限']]) {
      const field=el('label',label),input=el('input');input.type='number';input.min='1';input.step='1';
      input.value=String(limits[key]);input.setAttribute('aria-label',label);inputs[key]=input;
      field.append(input);grid.append(field);
    }
    card.append(grid);
    const confirm=button(view.action==='approve_research'?'确认总预算并开始':'追加总预算并继续',async()=>{
      const budget={...limits};
      for(const [key,input] of Object.entries(inputs)) {
        const value=Number(input.value);
        if(!Number.isSafeInteger(value)||value<=0)throw Error('预算必须是正整数');
        budget[key]=value;
      }
      await submit(view.action,turn,{plan_revision:task.plan_revision,budget});
    },'primary');
    confirm.disabled=!editable || turn.worker_active;card.append(confirm);
  }
  return card;
}
