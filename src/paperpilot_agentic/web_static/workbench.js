import {API,request,post,RunEvents} from './api.js';
import {el,button,markdown,details,jsonView,pendingMessage,protocolView,pairedDiagnosticTable,reviewedAnswer,budgetSummary} from './render.js';
import {journal} from './journal.js';
import {researchCard,coverageTable,comparisonTable,submissionNeedsOpen,researchReportNotice} from './research.js';

const $ = id => document.getElementById(id);
const state = {conversations:[],conversationCursor:null,cid:null,turns:[],older:null,
  stream:null,selected:new Map(),scopeEdited:false,sending:false,generation:0,bootstrap:null};
const terminal = new Set(['COMPLETED','FAILED','CANCELLED','BUDGET_EXHAUSTED','EXECUTION_UNKNOWN','INTERRUPTED','WAITING_APPROVAL']);
const settled = turn => terminal.has(turn.status) && !turn.worker_active;
const latest = () => state.turns.at(-1);
function notice(message) { $('notice').textContent=message; $('notice').hidden=!message; }
function error(exc) { notice(exc.message || String(exc)); }
function safe(fn) { return (...args) => Promise.resolve().then(() => fn(...args)).catch(error); }
function panel(title) { $('inspector').hidden=false; $('panel-title').textContent=title; $('panel-content').replaceChildren(); return $('panel-content'); }
function scopeLabel() {
  const inherited=latest()?.research.scope.length || 0;
  $('scope').textContent=state.scopeEdited ? (state.selected.size ? `已选 ${state.selected.size} 篇论文` : '全部本地论文')
    : inherited ? `沿用本研究 ${inherited} 篇论文` : '全部本地论文';
}

async function conversations(more=false) {
  const data = await request(`/conversations?limit=30${more && state.conversationCursor ? `&cursor=${encodeURIComponent(state.conversationCursor)}`:''}`);
  state.conversations=more ? [...state.conversations,...data.items] : data.items;
  state.conversationCursor=data.next_cursor; $('more-conversations').hidden=!data.next_cursor;
  const root=$('conversations'); root.replaceChildren();
  state.conversations.forEach(c => {
    const item=button('',safe(() => openConversation(c.conversation_id)),`conversation ${c.conversation_id===state.cid?'active':''}`);
    item.append(el('strong',c.title),el('small',`${c.turn_count} 条研究记录`)); root.append(item);
  });
}
async function openConversation(cid, older=false) {
  const generation=++state.generation;
  state.stream?.close(); state.stream=null;
  const data=await request(`/conversations/${encodeURIComponent(cid)}/turns?limit=20${older && state.older?`&cursor=${encodeURIComponent(state.older)}`:''}`);
  if (generation!==state.generation) return;
  state.turns=older ? [...data.items,...state.turns] : data.items;
  if (cid!==state.cid) {state.scopeEdited=false;state.selected.clear();}
  state.cid=cid; state.older=data.next_cursor;
  localStorage.setItem('paperpilot-conversation',cid);
  $('title').textContent=state.conversations.find(c=>c.conversation_id===cid)?.title || '当前研究';
  $('sidebar').classList.remove('open'); render(); watch();
  if (!older) $('scroll-area').scrollTop=$('scroll-area').scrollHeight;
  await conversations();
}
async function refreshTurn() {
  const current=latest(); if (!current) return;
  const generation=state.generation;
  const data=await request(`/runs/${encodeURIComponent(current.run_id)}/view`);
  if (generation!==state.generation || latest()?.run_id!==data.run_id) return;
  const nearBottom=$('scroll-area').scrollHeight-$('scroll-area').scrollTop-$('scroll-area').clientHeight<140;
  state.turns[state.turns.length-1]=data; render();
  if (nearBottom) $('scroll-area').scrollTop=$('scroll-area').scrollHeight;
  if (settled(data)) { state.stream?.close(); state.stream=null; $('connection').textContent='已同步'; }
}
function watch() {
  const current=latest(); if (!current || settled(current)) return;
  state.stream?.close(); const generation=state.generation;
  state.stream=new RunEvents(current.run_id,()=>{},safe(refreshTurn),text=>{
    if (generation===state.generation) $('connection').textContent=text;
  });
}
function controls() {
  const current=latest();
  $('send').disabled=state.sending || !state.bootstrap?.llm_configured || Boolean(current && !current.can_follow_up);
  $('send').textContent=state.sending ? '提交中…' : '发送 ↑';
  $('stop').hidden=!current?.can_cancel;
  $('resume').hidden=!current?.can_resume;
  $('materials').disabled=!current;
  $('older-turns').hidden=!state.older;
  $('hint').textContent=current && !current.can_follow_up ? current.state_text : 'Enter 发送 · Shift+Enter 换行';
}
function render() {
  $('welcome').hidden=Boolean(state.turns.length); controls(); scopeLabel();
  const root=$('thread');
  // Only changed turns are replaced: retain expanded materials and text selection.
  const ids=new Set(state.turns.map(t=>t.run_id));
  for (const child of [...root.children]) if (!ids.has(child.dataset.runId)) child.remove();
  state.turns.forEach((turn,index)=>{
    const signature=JSON.stringify(turn);
    let article=root.querySelector(`[data-run-id="${CSS.escape(turn.run_id)}"]`);
    if (article?.dataset.signature===signature) return;
    const fresh=el('article',undefined,'turn'); fresh.dataset.runId=turn.run_id; fresh.dataset.signature=signature;
    const user=el('div',undefined,'user-message'); user.append(el('p',turn.message.question));
    const head=el('div',undefined,'answer-head'); head.append(el('strong','PaperPilot'),el('span',turn.state_text,'state'));
    const answer=el('div',undefined,'answer');
    if (turn.research?.prototype) {
      const card=el('section',undefined,'research-task');
      card.append(el('h3','研究原型：任务与版本'),jsonView(turn.research.prototype));
      const verified=turn.research.prototype.validated_candidates||[];
      if(verified.length)card.append(el('p',`已有 ${verified.length} 份当前版本的候选验证回执通过工程检查；整体任务状态仍为 ${turn.research.prototype.phase}。结构要求与科研效果须分别核验。`));
      const download=el('a','下载代码、验证与版本快照');
      download.href=`${API}/runs/${encodeURIComponent(turn.run_id)}/prototype-bundle`;
      download.setAttribute('download','paperpilot-prototype.zip');
      card.append(download);
      if(turn.can_revise_prototype)card.append(button('修改验证输入与资源要求',safe(()=>prototypeRevision(turn))));
      answer.append(card);
    }
    if (turn.research_episode) {
      const episode=turn.research_episode,card=el('section',undefined,'research-task');
      card.append(el('h3','科研 Agent 任务'),el('p',`${episode.phase} · ${episode.turn}/${episode.limits.max_turns} 个决策轮次`));
      if(episode.assurance) {
        const a=episode.assurance;
        card.append(el('p',`执行状态：${a.execution} · 事实：${a.facts==='HOST_RECOMPUTED'?'宿主已复算':'尚未核验'} · 研究推断：仍需独立复核`,'state'));
        card.append(details('模型内容复核与核验层级',jsonView({content_review:episode.content_review,assurance:a})));
      }
      if(episode.research_case)card.append(button('查看论文依据、批准补丁与历史实验关联',safe(()=>researchCase(episode.research_case.case_id))));
      if(Array.isArray(episode.observed_views)) {
        card.append(el('p',`只读实验诊断 · 已读取 ${episode.observed_views.join('、')} · 新增训练 0 次`,'state'));
        if(episode.verification)card.append(details('核验层级',jsonView(episode.verification)));
        if(episode.analysis)card.append(details('宿主事实与待验证分析',jsonView(episode.analysis)));
      } else card.append(el('p',`GPU 已计 ${Math.round(episode.gpu_budget.charged_seconds)} 秒，累计上限 ${episode.gpu_budget.cap_seconds} 秒`,'state'));
      const progress=episode.execution_progress;
      if(progress) {
        card.append(el('p',`已保存 ${progress.recorded_trials}/${progress.total_trials} 组训练结果`,'state'));
        if(progress.current) {
          const p=progress.current;
          card.append(el('p',`${p.stage} · 种子 ${p.seed} · ${p.variant} · 已提交检查点 ${p.step}/${p.target} 步`,'state'));
        }
      }
      for (const candidate of episode.candidates) {
        card.append(el('h4',`候选 ${candidate.number} · ${candidate.status}`),el('p',candidate.proposal.hypothesis));
        if(candidate.card) {
          card.append(details('候选差异',el('pre',candidate.card.diff,'json-view')));
          card.append(details('检查、实验条件与执行成本',jsonView({
            binding_id:candidate.card.binding_id,static_checks:candidate.card.static_checks,
            not_checked:candidate.card.not_checked,protocol:candidate.card.protocol,
            data_scope:candidate.card.data_scope,cost:candidate.card.cost,
            environment:candidate.card.environment,environment_delta:candidate.card.environment_delta,
            filesystem_isolated:candidate.card.filesystem_isolated,
            network_isolated:candidate.card.network_isolated})));
        }
        if(candidate.result) {
          const table=pairedDiagnosticTable(candidate.result);
          if(table)card.append(markdown(table));
          card.append(details('宿主复算的完整实验结果',jsonView(candidate.result)));
        }
        if(candidate.analysis)card.append(details('模型分析（推断需复核）',jsonView(candidate.analysis)));
        if(candidate.result?.status==='TRAINING_VERIFIED' && !turn.worker_active && turn.run_id===latest()?.run_id) {
          card.append(el('p','诊断将复算统计和样本ID发送至当前配置的模型服务，不发送原始训练图像，也不启动训练。','state'));
          card.append(button('启动只读误差诊断',safe(()=>submitAction('research_diagnosis',turn,
            {candidate_id:candidate.card.binding_id}))));
        }
      }
      if(episode.stop_reason)card.append(el('p','负责人停止说明（模型解释需复核）：'+episode.stop_reason,'warning'));
      card.append(details('决策与角色交接',jsonView(episode.decisions)));
      answer.append(card);
    }
    if(turn.research_task)answer.append(researchCard(turn,{
      submit:async(...args)=>{try{return await submitAction(...args);}catch(exc){error(exc);throw exc;}},
      editable:turn.run_id===latest()?.run_id,
      openCoverage:()=>researchCoverage(turn)}));
    const reportNotice=researchReportNotice(turn);
    if(reportNotice)answer.append(el('p',reportNotice,'warning'));
    if (turn.research.plan) {
      const plan=turn.research.plan;
      answer.append(details('本轮研究计划',markdown(plan.steps.map(x=>`- ${x}`).join('\n'))));
    }
    if(turn.message.answer_scope==='BACKGROUND_ONLY')answer.append(el('p','以下仅是有来源的相关背景，尚未形成对本问题的直接回答。','warning'));
    const pairedReceipt=['method_comparison','method_training'].includes(turn.research.engine) && [...turn.artifacts].reverse().find(a=>['paired-method-verification','paired-training-verification'].includes(a.kind));
    if(pairedReceipt && !turn.message.answer)answer.append(button('查看成对训练结果与核验回执',safe(()=>artifactPreview(turn,pairedReceipt))));
    if (pairedReceipt && turn.message.answer) {
      answer.append(markdown(turn.message.answer.split('\n')[0]));
      answer.append(button('查看配对结果与核验回执',safe(()=>artifactPreview(turn,pairedReceipt))));
      answer.append(details('原始结果记录（完整精度）',markdown(turn.message.answer)));
    }
    else if (turn.message.answer) answer.append(reviewedAnswer(turn.message,safe(id=>evidence(turn,id))));
    else answer.append(el('p',pendingMessage(turn),'pending'));
    if(turn.message.answer && turn.failure)answer.append(el('p',pendingMessage(turn),'warning'));
    if(turn.budget?.warning)answer.append(el('p','本轮预算占用已达到 80%，后续审核或修订可能需要暂停。','warning'));
    if (turn.message.limitations.length) answer.append(details('回答边界与尚未解决的问题',markdown(turn.message.limitations.map(x=>`- ${x}`).join('\n'))));
    if (turn.error) answer.append(details('失败原因（已有记录保留，不会自动重跑）',el('p',turn.error)));
    const unverified=turn.message.review_notes.filter(n=>n.status==='UNVERIFIED_MODEL_NOTE');
    if (unverified.length) answer.append(details('未核验的模型说明（不作为事实依据）',markdown(unverified.map(n=>n.text).join('\n\n'))));
    const citations=el('div',undefined,'citations');
    turn.message.citations.forEach(c=>citations.append(button(c.label,safe(()=>evidence(turn,c.id)))));
    answer.append(citations);
    if (!settled(turn) && turn.activities.length) {
      answer.append(el('p',turn.activities.at(-1).text,'pending'));
    }
    const list=el('ol',undefined,'activity-list'); turn.activities.forEach(a=>list.append(el('li',a.text)));
    const progress=details(`研究活动 · ${budgetSummary(turn)}`,list);
    answer.append(progress);
    turn.approvals.forEach(a=>answer.append(approval(turn,a)));
    const actions=el('div',undefined,'actions');
    actions.append(button('研究材料',safe(()=>materials(turn))));
    if (turn.research.notebook_root) actions.append(button('研究笔记',safe(()=>notebook(turn.research.notebook_root))));
    if (turn.can_follow_up && turn.status==='COMPLETED') actions.append(button('实验工作流',safe(()=>workflow(turn))));
    answer.append(actions); fresh.append(user,head,answer);
    if (article) article.replaceWith(fresh); else root.insertBefore(fresh,root.children[index] || null);
  });
}
function approval(turn, value) {
  const card=el('div',undefined,'approval'); card.append(el('h3','需要你确认：实验执行协议'));
  card.append(el('p','批准仅适用于下面这份固定协议。修改代码或数据后需重新确认；诊断运行不等于科研假设成立。'));
  card.append(protocolView(value.context,value.description || value.tool_name));
  card.append(details('查看工具、数据、规模与风险',jsonView({tool:value.tool_name,...value.context})));
  const actions=el('div',undefined,'actions');
  for (const [label,action] of [['批准此协议','approve'],['拒绝','reject']]) {
    const control=button(label,safe(async()=>{
      actions.querySelectorAll('button').forEach(b=>b.disabled=true);
      try { await post(`/approvals/${encodeURIComponent(value.approval_id)}/${action}`,{actor:'local-workbench-user',resume:true}); await refreshTurn(); watch(); }
      finally { actions.querySelectorAll('button').forEach(b=>b.disabled=false); }
    }),action==='approve'?'primary':'');
    control.disabled=turn.worker_active;actions.append(control);
  }
  card.append(actions); return card;
}

async function evidence(turn,id) {
  const root=panel('论文原文与引用'); root.append(el('p','正在加载…'));
  const data=await request(`/runs/${encodeURIComponent(turn.run_id)}?include_trace=false`);
  const item=data.evidence.find(e=>e.evidence_id===id); if (!item) throw Error('该引用未绑定来源');
  root.replaceChildren(); root.append(el('h3',turn.message.citations.find(c=>c.id===id)?.label || '来源证据'));
  if (item.artifact_path && /\.(png|jpe?g|webp)$/i.test(item.artifact_path)) {
    const img=el('img'); img.alt='论文原始表格或图像裁剪'; img.src=`${API}/runs/${encodeURIComponent(turn.run_id)}/evidence/${encodeURIComponent(id)}/artifact`; root.append(img);
  }
  root.append(el('p','原文 / 转录（视觉转录仍需对照截图核验）','state'),el('pre',item.content,'json-view'));
  root.append(details('来源标识与定位',jsonView({source_uri:item.source_uri,page:item.page,id})));
}
async function materials(turn=latest()) {
  const root=panel('研究材料');
  root.append(el('h3','当前研究状态'),el('p',turn.state_text));
  if(turn.research_task)root.append(button('逐篇覆盖与报告章节',()=>researchCoverage(turn)));
  root.append(button('目标、修订历史与结果账本',safe(()=>journal(turn,{panel,submit:submitAction,safe,
    openRun:async id=>materials(await request(`/runs/${encodeURIComponent(id)}/view`))}))));
  root.append(details('来源范围与工作窗口',jsonView({papers:turn.research.scope,workset:turn.research.workset})));
  root.append(details('当前要求（派生记忆不是科学证据）',jsonView(turn.research.requirements)));
  root.append(el('p','修改目标或条件：在对话中明确说明。新要求作为新轮次保存，历史记录不会覆盖。','state'));
  if (turn.research.notebook_root) root.append(button('打开研究笔记',safe(()=>notebook(turn.research.notebook_root))));
  if (turn.research.design) root.append(details('实验方案（待验证假设）',jsonView(turn.research.design.plan || turn.research.design)));
  root.append(el('h3','代码、协议和结果'));
  if (!turn.artifacts.length) root.append(el('p','此轮没有独立产物。','empty'));
  turn.artifacts.forEach(a=>{
    const card=el('div',undefined,'card'); card.append(button(a.name,safe(()=>artifactPreview(turn,a))),el('small',a.kind,'state')); root.append(card);
  });
  root.append(button('实验工作流',safe(()=>workflow(turn))));
  root.append(button('查看原始运行记录',safe(async()=>{
    const data=await request(`/runs/${encodeURIComponent(turn.run_id)}?include_trace=false`);
    const p=panel('原始运行记录'); p.append(jsonView({status:data.run.status,error:data.run.error,events:data.events,tool_calls:data.tool_calls}));
  })));
}
async function artifactPreview(turn,artifact) {
  const data=await request(`/runs/${encodeURIComponent(turn.run_id)}/artifacts/${encodeURIComponent(artifact.artifact_id)}/preview`);
  const root=panel(data.name);
  root.append(el('p',`指纹 ${data.sha256.slice(0,16)} · 只读`,'state'));
  let table=null;
  if(['paired-method-verification','paired-training-verification'].includes(artifact.kind)) {
    try {table=pairedDiagnosticTable(JSON.parse(data.content));} catch { /* Retain raw malformed or historical artifacts. */ }
  }
  if(table)root.append(markdown(table),details('完整核验回执与未舍入指标',el('pre',data.content,'json-view')));
  else root.append(el('pre',data.content,'json-view'));
}
async function notebook(rid) {
  const root=panel('研究笔记'); root.append(el('p','笔记是导航材料，结论需核验；可按需重载原文。','state'));
  const list=el('div'); let offset=0; const more=button('加载笔记',safe(load)); root.append(list,more);
  async function load() {
    more.disabled=true;
    try {
      const data=await request(`/runs/${encodeURIComponent(rid)}/notebook?offset=${offset}&limit=20`);
      data.notes.forEach(n=>{
        const card=el('div',undefined,'card'); card.append(el('h3',n.title),el('p',n.statement || n.status || ''),el('small',n.kind,'state'));
        if (n.batch_id) card.append(button('读取原文',safe(async()=>{
          const raw=await request(`/runs/${encodeURIComponent(rid)}/notebook/sources/${encodeURIComponent(n.batch_id)}`);
          card.append(el('pre',raw.source_text,'json-view'));
        })));
        list.append(card);
      });
      offset+=data.notes.length; more.hidden=data.notes.length<20; more.textContent='加载下一页';
      if (!offset) list.append(el('p','尚无提取完成的笔记。','empty'));
    } finally { more.disabled=false; }
  }
  await load();
}
async function library() {
  const root=panel('选择论文范围');
  root.append(el('p','只读取已索引的本地论文。选择会限制下一轮的来源范围。','state'));
  const search=el('input'); search.placeholder='搜索论文标题'; search.setAttribute('aria-label','搜索论文');
  const list=el('div'); let offset=0; const more=button('加载更多',safe(()=>load(false)));
  root.append(search,button('搜索',safe(()=>load(true))),button('清除选择',()=>{state.selected.clear();state.scopeEdited=true;scopeLabel();load(true).catch(error);}),list,more);
  async function load(reset) {
    if (reset) { offset=0; list.replaceChildren(); }
    const data=await request(`/papers?query=${encodeURIComponent(search.value)}&offset=${offset}&limit=30`);
    data.items.forEach(p=>{
      const label=el('label',undefined,'paper-item'), check=el('input'); check.type='checkbox'; check.checked=state.selected.has(p.sha256);
      check.onchange=()=>{ state.scopeEdited=true;if(check.checked) state.selected.set(p.sha256,p.title || p.file_name); else state.selected.delete(p.sha256); scopeLabel(); };
      const text=el('span',p.title || p.file_name); text.append(el('small',`${p.page_count || '?'} 页`));label.append(check,text);list.append(label);
    });
    more.hidden=data.next_offset===null; offset=data.next_offset;
    if (!data.items.length) list.append(el('p',data.available?'没有匹配论文。':'尚未连接论文索引，请先完成本地索引配置。','empty'));
  }
  search.onkeydown=e=>{if(e.key==='Enter') load(true).catch(error);}; await load(true);
}

async function submitAction(action,turn,extra={}) {
  if(sessionStorage.getItem('paperpilot-pending-submission')) {
    await pendingSubmission();throw Error('上次提交结果仍待确认，请先检查原提交。');
  }
  const payload={request_id:crypto.randomUUID(),action,...(turn?{conversation_id:turn.conversation_id,parent_run_id:turn.run_id}:{}),...extra};
  sessionStorage.setItem('paperpilot-pending-submission',JSON.stringify(payload));
  let result;
  try {result=await post('/turns',payload);}
  catch(exc) {
    // A 409 may mean the reserved request has an unknown outcome after restart.
    // Keep its identity until the server confirms the original result.
    if(exc.status && exc.status<500 && exc.status!==409)sessionStorage.removeItem('paperpilot-pending-submission');
    else await pendingSubmission();
    throw exc;
  }
  sessionStorage.removeItem('paperpilot-pending-submission');
  if (submissionNeedsOpen(result,state.cid,state.turns)) { await conversations(); await openConversation(result.conversation_id); }
  else await refreshTurn();
  return result;
}

function prototypeConstraints(root,initial={}) {
  const inputCases=el('textarea');inputCases.setAttribute('aria-label','原型测试输入');
  inputCases.placeholder='每行一组 BxHxW，例如 1x64x64\n2x64x96；留空使用默认用例';
  inputCases.value=(initial.input_cases || []).map(row=>row.join('x')).join('\n');
  const inputLabel=el('label','测试输入（B×H×W）');inputLabel.append(inputCases);
  root.append(inputLabel,el('p','最多4组，batch为1–4。双时相三通道影像；SNUNet高宽为32–256之间的16倍数，其他原型为16–256。'),
    el('p','执行配额固定为CPU两核、4GiB内存、180秒；下方可填写更严格的验收上限。'));
  const fields=[['max_parameters','参数量上限（个，可选）',1],
    ['max_peak_rss_kib','进程峰值 RSS 上限（MiB，可选）',1024],
    ['max_verifier_seconds','验证器耗时上限（秒，可选）',1]].map(([key,label,multiplier])=>{
      const input=el('input');input.type='number';input.min='0';input.step=key==='max_verifier_seconds'?'any':'1';
      input.setAttribute('aria-label',label);if(key in initial)input.value=String(initial[key]/multiplier);
      const field=el('label',label);field.append(input);root.append(field);return {key,input,multiplier};
    });
  return ()=>{
    const result={};
    if(inputCases.value.trim())result.input_cases=inputCases.value.trim().split(/[;；\n]+/).map(line=>{
      const parts=line.trim().match(/^(\d+)\s*[x×]\s*(\d+)\s*[x×]\s*(\d+)$/);
      if(!parts)throw Error('输入格式为 BxHxW，例如 1x64x96');return parts.slice(1).map(Number);
    });
    fields.forEach(({key,input,multiplier})=>{if(input.value!=='')result[key]=Number(input.value)*multiplier;});
    return result;
  };
}

async function prototypeRevision(turn) {
  const current=await request(`/runs/${encodeURIComponent(turn.run_id)}/view`);
  if(!current.can_revise_prototype)throw Error('请先等待当前工作结束，或核对未知请求与执行回执');
  const prototype=current.research.prototype,root=panel('修改当前原型的验证要求');
  root.append(el('p','保留代码和累计用量；旧验证与复核将作为历史保留。下一次执行仍需重新确认。'));
  const read=prototypeConstraints(root,prototype.validation_contract);
  root.append(button('保存要求并继续',safe(async()=>{
    const result=await submitAction('revise_prototype_constraints',current,
      {expected_revision:prototype.work.revision,prototype_constraints:read()});
    $('inspector').hidden=true;
    await openConversation(result.conversation_id);
    notice(result.revision_applied ? (result.start?'验证要求已更新，已有代码保留。':'要求已保存，预算内尚未完成的验证保留为待处理项。')
      :'验证要求未变化，当前结果保留。');
  }),'primary'));
}

function researchCoverage(turn) {
  const root=panel('研究覆盖与阶段产物'),task=turn.research_task;
  root.append(el('p','已读取与已回答是不同状态。章节只覆盖各自引用的材料，原文可从回答中的引用重载。','warning'));
  root.append(coverageTable(task));
  if(task.comparison)root.append(comparisonTable(task,turn.message.citations,safe(id=>evidence(turn,id))));
  root.append(el('h3','已保存的报告章节'));
  task.sections.forEach(s=>root.append(el('p',`${s.dimension} · 材料组 ${s.group} · ${s.papers.length} 篇 · ${s.status}`)));
  if(!task.sections.length)root.append(el('p','还没有通过生成审核的章节；不会展示未经审核的草稿。'));
}

async function pendingSubmission() {
  const pending=sessionStorage.getItem('paperpilot-pending-submission');if(!pending)return;
  const payload=JSON.parse(pending),root=panel('上次提交结果待确认');
  root.append(el('p','连接中断不代表任务没开始。检查原请求后再继续，避免重复付费或训练。'));
  root.append(button('检查原提交',safe(async()=>{
    try {
      const result=await request(`/submissions/${payload.request_id}`);root.append(jsonView(result));
      if(result.state==='COMPLETED') {
        sessionStorage.removeItem('paperpilot-pending-submission');await openConversation(result.result.conversation_id);
      } else if(result.state==='FAILED') {
        sessionStorage.removeItem('paperpilot-pending-submission');notice('原提交失败；请检查已有材料后再明确发起新任务。');
      }
    } catch(exc) {
      if(exc.status!==404)throw exc;
      root.append(el('p','服务器未找到原请求。可以用同一请求标识重发；若原请求稍后到达，服务器会去重。'));
      root.append(button('重发同一请求',safe(async()=>{
        const result=await post('/turns',payload);sessionStorage.removeItem('paperpilot-pending-submission');
        await conversations();await openConversation(result.conversation_id);
      })));
    }
  })));
}

async function researchCase(id) {
    const data=await request(`/research-cases/${encodeURIComponent(id)}`),p=panel('历史科研来源链');
    p.append(el('p','以下为历史论文来源、已批准补丁和实验关联，不是本轮重新训练或已证明的机制。','state'));
    p.append(details('冻结来源',jsonView(data.sources)),details('批准的候选差异',el('pre',data.candidate_diff || '无差异文本','json-view')));
    p.append(details('实验引用与核验级别',jsonView({bundle_id:data.experiment_bundle_id,verification:data.verification,authority:data.execution_authority})));
}

async function importedExperiments(turn) {
    const data=await request('/imported-experiments'),p=panel('选择只读实验结果');
    p.append(el('p','复算预测与标签，不启动训练。训练来源是用户声明，机制解释需要复核。','state'));
    if(!data.items.length)p.append(el('p','尚未登记结果包；请先使用宿主 CLI 的 register-experiment 命令。'));
    for(const item of data.items) {
      p.append(button(`${item.dataset.name} / ${item.dataset.split} / 种子 ${item.seeds.join(', ')}`,
        safe(()=>submitAction('imported_diagnosis',turn,{bundle_id:item.bundle_id}))));
      for(const c of item.research_cases || [])p.append(button(`关联历史科研案例 ${c.case_id.slice(0,12)} 并诊断`,safe(()=>submitAction('imported_diagnosis',turn,{bundle_id:item.bundle_id,research_case_id:c.case_id}))));
    }
    if(data.stale_items.length)p.append(el('p',`${data.stale_items.length} 份资产失效，禁止继续使用。`,'warning'));
}

async function researchRecipes(turn) {

    const data=await request('/research-recipes'),p=panel('选择已登记的科研配方');
    p.append(el('p','模型提出假设和局部补丁，候选经你确认后自动执行训练、恢复和分析。累计 GPU 上限 4 小时。','state'));
    if(!data.items.length)p.append(el('p','尚未登记配方；请先用宿主 CLI 导入冻结的训练来源。'));
    if(data.stale_items?.length)p.append(el('p',`${data.stale_items.length} 个旧配方已失效；保留记录，需重新登记后使用。`,'warning'));
    const goal=el('textarea');goal.setAttribute('aria-label','科研目标');
    goal.value='分析 SNUNet 的注意力与融合机制，提出一个可证伪的局部融合改动，运行成对实验并根据真实结果决定下一步。';p.append(goal);
    for(const recipe of data.items) {
      p.append(button(`使用配方 ${recipe.recipe_id.slice(0,12)}`,safe(()=>submitAction('research_episode',turn,
        {question:goal.value,recipe_id:recipe.recipe_id,source_ids:recipe.source_ids}))));
    }
}

async function workflow(turn) {
  const root=panel('从文献到受控实验');
  root.append(button('诊断已导入的实验结果',safe(()=>importedExperiments(turn))));
  root.append(button('启动科研 Agent 任务',safe(()=>researchRecipes(turn))));
  root.append(el('p','先读原文 → 验证基线 → 生成局部改动 → 审阅 → 冻结协议 → 执行审批 → 汇总。','state'));
  root.append(el('p','历史流程保留 SNUNet 注意力消融；科研 Agent 配方支持经确认的局部融合修改。其他模型需另行登记与验证。','warning'));
  const data=await request(`/runs/${encodeURIComponent(turn.run_id)}/materials`);
  if (turn.research.design) {
    root.append(details('审阅实验方案',jsonView(turn.research.design.plan || turn.research.design)));
    const review=turn.research.review;
    root.append(el('p',review?.decision==='READY_FOR_EXECUTION_REVIEW'
      ? '审阅已记录，可以准备协议；仍未授权执行。'
      : review?.decision==='REVISION_REQUIRED' ? '这份草案需要修订，当前禁止准备执行协议。' : '草案尚未审阅，不能直接进入执行。','warning'));
    if(review)root.append(details('最近审阅意见',markdown(review.findings.join('\n\n'))));
    turn.artifacts.forEach(a=>root.append(button(`查看 ${a.name}`,safe(async()=>{
      const value=await request(`/runs/${turn.run_id}/artifacts/${a.artifact_id}/preview`); root.append(el('pre',value.content,'json-view'));
    }))));
    if (turn.research.design.draft_mode!=='validated_method_patch') {
      root.append(el('p','此草案未绑定已验证基线，只能查阅，不能执行。','warning')); return;
    }
    const findings=el('textarea'); findings.placeholder='记录审阅意见：代码改变了什么、还存在哪些限制？'; findings.setAttribute('aria-label','代码审阅意见'); root.append(findings);
    root.append(button('记录：需要修订',safe(async()=>{
      await submitAction('review',turn,{decision:'REVISION_REQUIRED',findings:[findings.value]});
      await workflow(await request(`/runs/${turn.run_id}/view`));
    })));
    root.append(button('记录：可进入执行审批',safe(async()=>{
      await submitAction('review',turn,{decision:'READY_FOR_EXECUTION_REVIEW',findings:[findings.value]}); notice('已保存审阅；尚未授权执行。');
      await workflow(await request(`/runs/${turn.run_id}/view`));
    })));
    const prepare=button('准备固定对照协议',safe(async()=>{
      const result=await submitAction('prepare_comparison',turn);
      root.append(el('h3','待确认的固定实验协议'),protocolView(result.protocol),details('完整协议与指纹',jsonView(result.protocol)));
      root.append(button('提交协议，等待执行审批',safe(()=>submitAction('compare',turn,{binding_id:result.binding_id})),'primary'));
    }));
    prepare.disabled=review?.decision!=='READY_FOR_EXECUTION_REVIEW';root.append(prepare);
    return;
  }
  const readerSelect=el('select'); readerSelect.setAttribute('aria-label','选择文献阅读记录');
  data.readers.forEach(r=>{const o=el('option',r.question.slice(0,70));o.value=r.run_id;readerSelect.append(o);});
  if (!data.readers.length) {root.append(el('p','请先完成一次有原文依据的论文问答。','empty'));return;}
  root.append(el('h3','1. 选择论文依据'),readerSelect);
  const sources=el('div'); root.append(sources); const selection=new Set();
  function selectReader() {
    selection.clear(); sources.replaceChildren();
    data.readers.find(r=>r.run_id===readerSelect.value).sources.forEach(s=>{
      const label=el('label',undefined,'paper-item'), check=el('input');check.type='checkbox';
      check.onchange=()=>check.checked?selection.add(s.id):selection.delete(s.id);
      label.append(check,el('span',`${s.page ? `第 ${s.page} 页 · ` : ''}${s.preview}`));sources.append(label);
      const original=details('展开这段原文',el('p','仅加载所选阅读记录中的原始材料。'));
      original.addEventListener('toggle',safe(async()=>{
        if(!original.open || original.dataset.loaded)return;
        const data=await request(`/runs/${encodeURIComponent(readerSelect.value)}?include_trace=false`);
        const item=data.evidence.find(e=>e.evidence_id===s.id);
        if(!item)throw Error('来源不再属于这次阅读');
        original.append(el('pre',item.content,'json-view'));original.dataset.loaded='1';
      }));sources.append(original);
    });
  }
  readerSelect.onchange=selectReader;selectReader();
  const parent=()=>({run_id:readerSelect.value,conversation_id:turn.conversation_id});
  root.append(el('h3','仅设计模型，不运行实验'));
  const proposalGoal=el('textarea'); proposalGoal.setAttribute('aria-label','模型提案目标');
  proposalGoal.placeholder='例如：基于所选论文，设计一个建筑变化检测模型，给出结构、伪代码、代价与消融。';
  root.append(proposalGoal);
  root.append(button('生成只读模型提案',safe(()=>{
    if(!selection.size)throw Error('请先勾选论文原文依据');
    if(!proposalGoal.value.trim())throw Error('请描述模型设计目标');
    return submitAction('design',parent(),{question:proposalGoal.value,source_ids:[...selection],
      design_mode:'architecture_proposal',budget:{unlimited_llm_usage:true}});
  })));
  root.append(el('p','只读提案需要审阅，不证明新颖性或效果。','warning'));
  root.append(el('h3','研究原型：生成代码并申请工程验证'));
  const environment=await request('/prototype-environment');
  root.append(el('p',environment.available
    ? `CPU 验证环境就绪 · ${environment.image_id.slice(0,19)} · ${environment.platform}`
    : 'CPU 验证环境未就绪；可以生成候选，执行将停在待环境恢复。',environment.available?'state':'warning'));
  const profileData=await request('/prototype-profiles');
  const prototypeProfile=el('select');prototypeProfile.setAttribute('aria-label','原型基线');
  for(const item of profileData.items) {
    const option=el('option',`SNUNet_ECAM 局部融合 · ${item.profile_id.slice(0,8)}`);
    option.value=item.profile_id;prototypeProfile.append(option);
  }
  const unboundProfile=el('option','未绑定基线的原型');unboundProfile.value='';prototypeProfile.append(unboundProfile);
  root.append(prototypeProfile);
  const readConstraints=prototypeConstraints(root);
  root.append(button('生成并验证研究原型',safe(()=>{
    if(!selection.size || !proposalGoal.value.trim())throw Error('请勾选依据并填写目标');
    const constraints=readConstraints();
    return submitAction('research_prototype',parent(),{question:proposalGoal.value,source_ids:[...selection],
      prototype_profile_id:prototypeProfile.value||undefined,
      prototype_constraints:Object.keys(constraints).length?constraints:undefined});
  })));
  root.append(button('调研论文并生成完整原型',safe(()=>{
    if(!selection.size || !proposalGoal.value.trim())throw Error('请勾选已登记的模型源码等起始依据，并填写研究目标');
    const constraints=readConstraints();
    return submitAction('research_prototype',parent(),{question:proposalGoal.value,source_ids:[...selection],
      prototype_mode:'adaptive',prototype_literature:true,
      prototype_constraints:Object.keys(constraints).length?constraints:undefined});
  })));
  root.append(el('p','调研入口由 Agent 选择本地论文检索、在线学术检索及原文读取，再设计完整模型；检索范围和资料缺口会保留。'));
  root.append(el('p','候选代码须逐份审批后在 Docker 中检查；不执行训练。','warning'));
  root.append(el('h3','2. 选择已配置的方法'));
  const methods=el('select');methods.setAttribute('aria-label','已配置的方法');
  data.profiles.forEach(m=>{const o=el('option',`${m.method} · ${m.steps} 步 · ${m.binding_id.slice(0,8)}`);o.value=m.binding_id;methods.append(o);});root.append(methods);
  if (!data.profiles.length) {
    root.append(el('p','当前工作区没有经过校验的方法配置。需要先由宿主绑定代码、数据和训练环境；不能通过浏览器提交任意路径或命令。','warning'));return;
  }
  root.append(button('查看基线协议',()=>root.append(jsonView(data.profiles.find(m=>m.binding_id===methods.value)))));
  root.append(button('申请验证基线',safe(()=>submitAction('validate',parent(),{binding_id:methods.value}))));
  root.append(el('h3','3. 描述要检验的改动'));
  const question=el('textarea'); question.placeholder='说明实验假设、候选改动与比较目标，不要求模型编造提升。';question.setAttribute('aria-label','实验设计问题');root.append(question);
  root.append(button('根据所选证据生成草案',safe(async()=>{
    const validation=data.validations.filter(v=>v.binding_id===methods.value).at(-1);
    if (!validation) throw Error('请先完成对应基线验证，再生成绑定草案');
    if (!selection.size) throw Error('请先勾选至少一段论文原文依据');
    const ref=await submitAction('attach_reference',parent(),{binding_id:methods.value});
    await submitAction('design',parent(),{question:question.value,source_ids:[...selection],reference_ids:[ref.reference_id],validation_run_id:validation.run_id});
  }),'primary'));
}

$('composer').onsubmit=event=>{ event.preventDefault(); safe(async()=>{
  if(state.sending) return;
  const question=$('question').value.trim();if(!question)return;
  state.sending=true;controls();notice('');
  const scope=state.scopeEdited ? {paper_ids:[...state.selected.keys()]} : {};
  try {await submitAction($('intent').value,latest(),{question,...scope});$('question').value='';}
  finally {state.sending=false;controls();}
})();};
$('question').onkeydown=event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();if(!$('send').disabled)$('composer').requestSubmit();}};
$('new-conversation').onclick=()=>{
  state.generation++;state.stream?.close();state.stream=null;state.cid=null;state.turns=[];state.older=null;
  state.selected.clear();state.scopeEdited=false;
  localStorage.removeItem('paperpilot-conversation');$('title').textContent='科研工作台';$('connection').textContent='';render();$('question').focus();
};
$('stop').onclick=safe(async()=>{await post(`/runs/${latest().run_id}/cancel`);await refreshTurn();watch();});
$('resume').onclick=safe(async()=>{await post(`/runs/${latest().run_id}/resume`);await refreshTurn();watch();});
$('refresh').onclick=safe(()=>conversations());$('more-conversations').onclick=safe(()=>conversations(true));
$('older-turns').onclick=safe(()=>openConversation(state.cid,true));
$('imported-experiments').onclick=safe(()=>importedExperiments(null));
$('research-workflow').onclick=safe(()=>researchRecipes(null));
$('materials').onclick=safe(()=>materials());$('library').onclick=safe(library);$('scope').onclick=safe(library);
$('close-panel').onclick=()=>{$('inspector').hidden=true;};$('mobile-menu').onclick=()=>{$('sidebar').classList.toggle('open');};
document.querySelectorAll('[data-example]').forEach(b=>b.onclick=()=>{$('question').value=b.dataset.example;$('intent').value=b.dataset.action || 'answer';$('question').focus();});

async function boot() {
  state.bootstrap=await request('/api/bootstrap');const b=state.bootstrap;
  $('runtime').append(el('div',`${b.indexed_paper_count ?? 0} 篇本地论文`),el('div',`${b.provider} · ${b.model}`),
    el('div',`运行版本 ${b.build?.loaded?.revision || '未知'} · ${b.build?.loaded?.source_digest?.slice(0,8) || ''}`));
  if (!b.capabilities?.workbench) throw Error('后端尚未加载工作台版本，请安全重启 PaperPilot 服务。');
  if (b.build?.restart_required) notice('磁盘源码已更新；当前后端仍是启动时版本。请在任务结束后重启服务。');
  await conversations(); const cid=localStorage.getItem('paperpilot-conversation');
  if(cid) {try{await openConversation(cid);}catch(exc){localStorage.removeItem('paperpilot-conversation');error(exc);}}
  controls();
  await pendingSubmission();
}
boot().catch(error);
