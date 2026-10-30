// Build nodes, never interpret model-provided HTML or arbitrary URL schemes.
export function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
export function button(text, action, className = '') {
  const node = el('button', text, className); node.type = 'button';
  node.addEventListener('click', event => {
    if(node.disabled)return;
    const result=action(event);
    if(result && typeof result.then==='function') {
      node.disabled=true; Promise.resolve(result).finally(()=>{node.disabled=false;}).catch(()=>{});
    }
  }); return node;
}

function inline(text, citations, onCitation) {
  const root = document.createDocumentFragment();
  const pattern = /(`[^`\n]+`|\*\*[^*\n]+\*\*|\[[^\]\n]+\]\([^\s)]+\)|\[(?:evidence_|e_|n_)[a-zA-Z0-9_:.-]+\])/g;
  let end = 0;
  for (const hit of text.matchAll(pattern)) {
    root.append(document.createTextNode(text.slice(end, hit.index)));
    const token = hit[0];
    if (token.startsWith('`')) root.append(el('code', token.slice(1,-1)));
    else if (token.startsWith('**')) root.append(el('strong', token.slice(2,-2)));
    else if (token.includes('](')) {
      const parts = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(token);
      if (/^https?:\/\//i.test(parts[2])) {
        const link = el('a',parts[1]); link.href = parts[2]; link.target = '_blank';
        link.rel = 'noopener noreferrer'; root.append(link);
      } else root.append(document.createTextNode(token));
    } else {
      const citation = citations.find(c => c.id === token.slice(1,-1));
      if (citation) {
        const cite = button(`[${citation.number}]`, () => onCitation(citation.id), 'cite');
        cite.title = citation.label; root.append(cite);
      } else root.append(document.createTextNode('[来源未绑定]'));
    }
    end = hit.index + token.length;
  }
  root.append(document.createTextNode(text.slice(end))); return root;
}

export function markdown(text, citations = [], onCitation = () => {}) {
  const root = el('div',undefined,'markdown');
  const lines = String(text || '').split('\n');
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.trim()==='$$' || line.trim()==='\\[') {
      const close=line.trim()==='$$'?'$$':'\\]', body=[line];
      while (++i<lines.length) {body.push(lines[i]);if(lines[i].trim()===close)break;}
      root.append(el('div',body.join('\n'),'math-block'));continue;
    }
    if (line.startsWith('```')) {
      const language = line.slice(3).trim(), body = [];
      while (++i < lines.length && !lines[i].startsWith('```')) body.push(lines[i]);
      const pre = el('pre'); pre.append(el('code',body.join('\n')));
      const wrap = el('div',undefined,'code-block');
      const head = el('div',undefined,'code-head'); head.append(el('span',language || '代码'));
      head.append(button('复制',async () => { await navigator.clipboard.writeText(body.join('\n')); }));
      wrap.append(head,pre); root.append(wrap); continue;
    }
    if (!line.trim()) continue;
    if (i+1 < lines.length && line.includes('|') && /^\s*\|?\s*:?-{3,}/.test(lines[i+1])) {
      const table = el('table'), wrap = el('div',undefined,'table-wrap');
      const cells = value => value.trim().replace(/^\||\|$/g,'').split('|');
      const header = el('tr'); cells(line).forEach(c => { const td = el('th'); td.append(inline(c.trim(),citations,onCitation)); header.append(td); });
      const thead = el('thead'); thead.append(header); table.append(thead); i++;
      const tbody = el('tbody');
      while (i+1 < lines.length && lines[i+1].includes('|')) {
        const row = el('tr'); cells(lines[++i]).forEach(c => { const td = el('td'); td.append(inline(c.trim(),citations,onCitation)); row.append(td); }); tbody.append(row);
      }
      table.append(tbody); wrap.append(table); root.append(wrap); continue;
    }
    const heading = /^(#{1,4})\s+(.+)$/.exec(line);
    const list = /^\s*(?:[-*]|\d+[.)])\s+(.+)$/.exec(line);
    const node = el(heading ? `h${Math.min(heading[1].length+1,5)}` : list ? 'li' : 'p');
    node.append(inline(heading ? heading[2] : list ? list[1] : line,citations,onCitation));
    if (list) {
      let ul = root.lastElementChild; if (!ul || ul.tagName !== 'UL') { ul = el('ul'); root.append(ul); }
      ul.append(node);
    } else root.append(node);
  }
  if (window.renderMathInElement) window.renderMathInElement(root, {
    delimiters:[{left:'$$',right:'$$',display:true},{left:'\\[',right:'\\]',display:true},
      {left:'\\(',right:'\\)',display:false},{left:'$',right:'$',display:false}],
    trust:false, throwOnError:false, maxExpand:500, maxSize:10,
  });
  return root;
}

export function details(title, child) {
  const node = el('details'); node.append(el('summary',title),child); return node;
}

export function jsonView(value) { return el('pre',JSON.stringify(value,null,2),'json-view'); }

export function answerSections(message) {
  const sections=message.answer_sections;
  // Legacy or mismatched projections fall back to the unchanged full answer.
  if(!Array.isArray(sections) || !sections.length || sections.length>2 ||
      sections.some(s=>!s || !['direct','context'].includes(s.kind) || typeof s.text!=='string' || !s.text.trim()) ||
      new Set(sections.map(s=>s.kind)).size!==sections.length ||
      sections.map(s=>s.text).join('\n\n')!==message.answer)return null;
  return sections;
}

export function reviewedAnswer(message, onCitation) {
  const sections=answerSections(message);
  if(!sections || sections.length<2)return markdown(message.answer,message.citations,onCitation);
  const root=el('div',undefined,'reviewed-answer');
  sections.forEach(s=>{
    root.append(el('h4',s.kind==='direct'?'回答':'必要解释与条件'));
    // Never collapse qualifiers, evidence gaps or inference premises.
    root.append(markdown(s.text,message.citations,onCitation));
  });
  return root;
}

export function pendingMessage(turn) {
  if(turn.failure)return `${turn.failure.title}。${turn.failure.action}`;
  if(turn.research_task?.state==='AWAITING_BUDGET')return '请确认研究范围与总预算；确认后分批读取论文，不授权实验或训练。';
  if(turn.research_task?.state==='UNKNOWN')return '上次模型请求结果未知，已暂停；不会因刷新或追加预算自动重试。';
  if(turn.status==='WAITING_APPROVAL')return '等待你确认下方协议；批准前不会执行实验。';
  if(turn.status==='STOPPING')return '已停止调度后续动作，正在等待本任务的执行进程退出。';
  if(turn.status==='CANCELLED')return '本轮已停止，已保存的来源、笔记和产物仍可查看。';
  if(turn.status==='FAILED')return '本轮执行失败。已有记录保留，不会自动重跑。';
  if(turn.status==='BUDGET_EXHAUSTED')return '本轮预算已用尽，未完成部分不会被标记为研究结论。';
  if(['INTERRUPTED','EXECUTION_UNKNOWN'].includes(turn.status))return '上次执行状态需要核对；不会因刷新而重发请求或重跑实验。';
  if(turn.status==='COMPLETED')return '当前材料尚未形成可发布的答案。请查看下方证据缺口与核验说明；这不表示论文中一定没有答案。';
  const engine=turn.research?.engine;
  if(engine==='method_training')return '正在执行冻结的成对训练并独立复算指标；中断时保存 checkpoint，累计 GPU 预算不清零。';
  if(['method_validation','method_comparison'].includes(engine))return '正在执行已批准的短实验并核对指标；不会据此宣称科研假设成立。';
  if(engine==='research_design')return '正在阅读依据、形成实验方案和代码草案；此步骤不会启动训练。';
  if(engine==='research_notebook')return '正在分批整理研究笔记，保留原文定位以便后续核对。';
  return '正在检索、阅读并核验材料。审核后的回答会在这里显示。';
}

export function budgetSummary(turn) {
  const b=turn.budget || {};
  const model=turn.models?.length ? ` · 模型 ${turn.models.join(' → ')}` : '';
  const hold=b.reserved_unknown_tokens ? ` · 未确认用量预留 ${b.reserved_unknown_tokens} tokens` : '';
  const warning=b.warning ? ' · 已达到预算的 80%，后续调用仍按剩余预算检查' : '';
  const usage=b.unlimited_llm_usage
    ? `已确认 ${b.used_tokens ?? 0} tokens · 不设本地累计模型预算`
    : `已确认 ${b.used_tokens ?? 0} / ${b.max_tokens ?? '?'} tokens`;
  return `${b.used_llm_calls ?? 0} 次实际请求 · ${usage}${hold}${model}${warning}`;
}

export function protocolSummary(context, description='见冻结协议') {
  const c=context || {},p=c.protocol || {},data=c.data_scope || {},limits=c.limits || {};
  return [
    ['代码改动',c.code_change || description],
    ['数据',data.dataset ? `${data.dataset}：训练 ${data.train_images ?? '?'} 张，验证 ${data.val_images ?? '?'} 张；${data.test_images===0 ? '不使用测试集' : '测试集范围见完整协议'}` : '见冻结协议'],
    ['训练规模',p.version===2 ? `seeds=${p.seeds.join(', ')} × ${p.variants.length} 变体 × ${p.epochs} epochs；batch=${p.batch_size}，FP32，无增广` : p.seeds ? `seeds=${p.seeds.join(', ')} × ${p.variants?.length ?? '?'} 变体 × ${p.steps_per_variant ?? '?'} 次更新；共 ${p.max_total_optimizer_steps ?? '?'} 次更新` : `${c.steps ?? '?'} 次更新 · seed=${c.seed ?? '见完整协议'} · ${c.device || '见协议'}`],
    ['运行上限',p.version===2 ? `累计 GPU ${limits.gpu_cap_seconds} 秒；每 ${limits.checkpoint_interval} 步及暂停保存 checkpoint` : `${limits.timeout_seconds ?? p.timeout_seconds ?? '?'} 秒；输出最多 ${limits.max_output_mb ?? '?'} MB`],
    ['输出位置',c.output_location || '见冻结协议'],
    ['隔离边界','本机执行已审阅代码；不具备强网络、文件系统或内存隔离'],
    ['结论边界','仅验证工程链路；科研假设、非劣性与论文精度均未验证'],
    ['协议绑定',c.binding_id || '见完整协议'],
  ];
}

export function protocolView(context, description) {
  const summary=el('dl',undefined,'protocol-summary');
  for(const [label,text] of protocolSummary(context,description))summary.append(el('dt',label),el('dd',text));
  return summary;
}

export function pairedDiagnosticTable(data) {
  // Presentation only; never infer verification from the presence of metrics.
  if(['PAIRED_TRAINING_VERIFIED','TRAINING_VERIFIED','PARTIAL'].includes(data?.status) && data.scientific_acceptance==='DESCRIPTIVE_ONLY' && Array.isArray(data.trials)) {
    if(data.trials.length!==3 || data.trials.some((t,i)=>t.seed!==[7,11,19][i]))return null;
    const lines=['固定小规模 train/val 对照；未使用 test。全部种子、失败与负结果均保留，指标范围为 0–1。','',
      '| 种子 | 变体 | 状态 | Precision | Recall | F1 | IoU |','| --- | --- | --- | --- | --- | --- | --- |'];
    for(const trial of data.trials)for(const [key,label] of [['baseline','基线'],['candidate','候选']]) {
      const record=trial[key],metrics=record?.metrics;
      if(record?.status!=='VERIFIED'){lines.push(`| ${trial.seed} | ${label} | ${record?.status || 'NOT_RUN'} | — | — | — | — |`);continue;}
      const values=['precision','recall','f1','iou'].map(k=>metrics?.[k]);
      if(values.some(v=>v!==null && !(typeof v==='number' && Number.isFinite(v) && v>=0 && v<=1)))return null;
      lines.push(`| ${trial.seed} | ${label} | VERIFIED | ${values.map(v=>v===null?'未定义':v.toFixed(4)).join(' | ')} |`);
    }
    return lines.join('\n');
  }
  if(data?.status!=='PAIRED_DIAGNOSTIC_VERIFIED' || data.scientific_acceptance!=='NOT_ASSESSED' ||
      !Array.isArray(data.trials) || !data.trials.length)return null;
  const lines=['仅为固定小样本的描述性结果，不证明方法有效、无效或非劣。指标范围为 0–1。',
    '', '| 种子 | 变体 | F1 | IoU |', '| --- | --- | --- | --- |'];
  for(const trial of data.trials) {
    if(!Number.isInteger(trial.seed))return null;
    for(const [key,label] of [['baseline','基线'],['candidate','候选']]) {
      const metrics=trial[key]?.validation;
      if(!metrics || ![metrics.f1,metrics.iou].every(v=>typeof v==='number' && Number.isFinite(v) && v>=0 && v<=1))return null;
      lines.push(`| ${trial.seed} | ${label} | ${metrics.f1.toFixed(4)} | ${metrics.iou.toFixed(4)} |`);
    }
  }
  return lines.join('\n');
}
