import {request} from './api.js';
import {el,button,details,jsonView} from './render.js';

export async function journal(turn, {panel,submit,openRun,safe}) {
  const root=panel('研究目标与结果账本');
  const path=`/conversations/${encodeURIComponent(turn.conversation_id)}/research-state`;
  const data=await request(path);
  root.append(el('p','与聊天窗口分开保存。这里的目标和假设不是原文证据；每项结果可回到原运行核查。','state'));
  const goal=el('textarea'), constraints=el('textarea'), gaps=el('textarea');
  goal.value=data.state.goal || turn.message.question;goal.setAttribute('aria-label','研究总目标');
  constraints.value=data.state.constraints.join('\n');constraints.setAttribute('aria-label','研究限制条件');
  gaps.value=data.state.open_questions.join('\n');gaps.setAttribute('aria-label','待解决问题');
  root.append(el('h3','研究总目标'),goal,el('h3','限制条件（每行一项）'),constraints,el('h3','待解决问题（每行一项）'),gaps);
  const save=button('保存修订，保留历史版本',safe(async()=>{
    const lines=node=>node.value.split('\n').map(x=>x.trim()).filter(Boolean);
    await submit('update_state',turn,{expected_revision:data.state.revision,
      research_state:{goal:goal.value,constraints:lines(constraints),open_questions:lines(gaps)}});
    await journal(turn,{panel,submit,openRun,safe});
  }));save.disabled=!turn.can_follow_up;root.append(save);
  root.append(details(`修订历史 · 当前版本 ${data.state.revision}`,jsonView(data.revisions)));
  root.append(el('h3',`结果账本 · ${data.total} 条`));
  let offset=0;const list=el('div');const more=button('加载更多结果',safe(load));root.append(list,more);
  function append(page){
    page.items.forEach(item=>{
      const card=el('div',undefined,'card');card.append(el('h3',item.question),
        el('p',`${item.status} · ${item.has_experiment_receipt?'有实验回执；科学有效性未判定':'研究记录，非已验证科学结论'}`,'state'));
      if(item.hypothesis)card.append(el('p',`待验证假设：${item.hypothesis}`));
      card.append(button('查看原轮次与产物',safe(()=>openRun(item.run_id))));list.append(card);
    });offset=page.next_offset;more.hidden=offset===null;
  }
  async function load(){append(await request(`${path}?offset=${offset}&limit=20`));}
  append(data);
}
