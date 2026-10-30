const state = {
  runs: [],
  current: null,
  thread: [],
  conversationId: null,
  newConversation: true,
  source: null,
  bootstrap: null,
  worksetStarting: false,
};

const $ = (id) => document.getElementById(id);
const streamTerminal = new Set(["COMPLETED", "FAILED", "CANCELLED", "BUDGET_EXHAUSTED", "WAITING_APPROVAL"]);
const followUpReady = new Set(["COMPLETED", "FAILED", "CANCELLED", "BUDGET_EXHAUSTED"]);

function conversationId(run) {
  return run?.metadata?.conversation_id || run?.run_id || "";
}

function chronological(left, right) {
  return String(left.created_at || "").localeCompare(String(right.created_at || ""));
}

function conversationGroups() {
  const groups = new Map();
  state.runs.forEach((run) => {
    const id = conversationId(run);
    if (!groups.has(id)) groups.set(id, []);
    groups.get(id).push(run);
  });
  return [...groups.entries()]
    .map(([id, runs]) => ({ id, runs: runs.sort(chronological) }))
    .sort((left, right) => {
      const a = left.runs.at(-1)?.updated_at || "";
      const b = right.runs.at(-1)?.updated_at || "";
      return String(b).localeCompare(String(a));
    });
}

async function request(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function boot() {
  state.bootstrap = await request("/api/bootstrap");
  renderRuntime();
  await refreshRuns();
  const saved = localStorage.getItem("paperpilot-run-id");
  const savedRun = state.runs.find((run) => run.run_id === saved);
  if (savedRun) await loadConversation(conversationId(savedRun));
  else renderComposerState();
}

function renderRuntime() {
  const data = state.bootstrap;
  const paperCount = Number.isInteger(data.indexed_paper_count) ? `（${data.indexed_paper_count} 篇）` : "";
  const retrieval = data.memory_database_configured
    ? ["BM25", data.dense_retrieval_configured ? "Dense/RRF" : null, data.reranker_configured ? "Cross-Encoder" : null].filter(Boolean).join(" + ")
    : "文件回退";
  $("runtime-card").innerHTML =
    `<b>${escapeHtml(data.provider)} · ${escapeHtml(data.model)}</b><br>` +
    `统一模型：${data.llm_configured ? "文本已配置" : "文本未配置"} / ${data.vision_configured ? "视觉已配置" : "视觉未配置"}<br>` +
    `论文索引：${data.memory_database_configured ? `已连接${paperCount}` : "未连接"}<br>` +
    `检索：${escapeHtml(retrieval)}`;
  renderComposerState();
}

async function refreshRuns() {
  const data = await request("/api/agentic/runs?limit=200");
  state.runs = data.runs || [];
  renderRunList();
}

function renderRunList() {
  const root = $("run-list");
  root.replaceChildren();
  if (!state.runs.length) {
    const empty = document.createElement("small");
    empty.textContent = "还没有历史对话";
    root.append(empty);
    return;
  }
  conversationGroups().forEach((conversation) => {
    const first = conversation.runs[0];
    const latest = conversation.runs.at(-1);
    const button = document.createElement("button");
    button.className = `run-item ${state.conversationId === conversation.id ? "active" : ""}`;
    button.type = "button";
    button.innerHTML = `<strong>${escapeHtml(first.question || conversation.id)}</strong><small>${conversation.runs.length} 轮 · ${escapeHtml(latest.status)} · ${escapeHtml(latest.research_outcome || "-")}</small>`;
    button.addEventListener("click", () => loadConversation(conversation.id));
    root.append(button);
  });
}

async function createRun(question) {
  if (!state.bootstrap.llm_configured) throw new Error("LLM API 尚未配置");
  const data = await request("/api/agentic/runs", {
    method: "POST",
    body: JSON.stringify({ question, engine: $("engine-select").value, visual_policy: "auto", metadata: { source: "paperpilot-answerfirst-web" } }),
  });
  const runId = data.run.run_id;
  localStorage.setItem("paperpilot-run-id", runId);
  await refreshRuns();
  await loadConversation(conversationId(data.run), true);
}

async function createFollowUp(parentRunId, question) {
  if (!state.bootstrap.llm_configured) throw new Error("LLM API 尚未配置");
  const data = await request(`/api/agentic/runs/${encodeURIComponent(parentRunId)}/follow-up`, {
    method: "POST",
    body: JSON.stringify({ question, metadata: { source: "paperpilot-slim-web-follow-up" } }),
  });
  const runId = data.run.run_id;
  localStorage.setItem("paperpilot-run-id", runId);
  await refreshRuns();
  await loadConversation(conversationId(data.run), true);
}

async function loadConversation(id, watch = true) {
  const summaries = state.runs
    .filter((run) => conversationId(run) === id)
    .sort(chronological);
  if (!summaries.length) throw new Error(`找不到对话 ${id}`);
  const details = await Promise.all(
    summaries.map((run) => request(`/api/agentic/runs/${encodeURIComponent(run.run_id)}?include_trace=false`)),
  );
  details.sort((left, right) => chronological(left.run, right.run));
  const latest = details.at(-1);
  state.thread = details;
  state.current = latest;
  state.conversationId = id;
  state.newConversation = false;
  localStorage.setItem("paperpilot-run-id", latest.run.run_id);
  renderConversation(details);
  renderRun(latest);
  renderRunList();
  if (watch && !streamTerminal.has(latest.run.status)) watchRun(latest.run.run_id);
}

async function reloadCurrentConversation(watch = false) {
  if (!state.conversationId) return;
  await refreshRuns();
  await loadConversation(state.conversationId, watch);
}

function watchRun(runId) {
  if (state.source) state.source.close();
  const source = new EventSource(`/api/agentic/runs/${encodeURIComponent(runId)}/events`);
  state.source = source;
  source.onmessage = async (message) => {
    const event = JSON.parse(message.data);
    $("progress").textContent = eventSummary(event);
    if (event.event_type === "stream.end") {
      source.close();
      state.source = null;
      await reloadCurrentConversation(false);
    }
  };
  source.onerror = () => {
    source.close();
    state.source = null;
    setTimeout(() => reloadCurrentConversation(false).catch(showError), 1000);
  };
}

function renderRun(data) {
  $("empty-state").hidden = true;
  $("run-view").hidden = false;
  $("run-status").textContent = data.run.status;
  $("outcome").textContent = data.run.research_outcome || data.final_answer?.research_outcome || "-";
  $("progress").textContent = eventSummary((data.events || []).at(-1) || {});
  const workset = data.run.metadata?.workset_progress;
  if (workset) $("progress").textContent = `文献审阅 ${workset.reviewed_pages}/${workset.total_pages} 批；未执行实验`;
  $("workset-review").hidden = !data.run.metadata?.workset_review_available || !followUpReady.has(data.run.status) || data.run.status === "CANCELLED";
  $("workset-review").disabled = state.worksetStarting;
  renderClaims(data);
  renderEvents(data);
  renderApprovals(data);
  const used = data.run.used_tokens || 0;
  const max = data.run.budget?.max_tokens || "-";
  $("budget").textContent = `${used} / ${max} tokens`;
  renderComposerState();
}

function renderConversation(details) {
  const root = $("conversation-thread");
  root.replaceChildren();
  details.forEach((data, index) => {
    const userMessage = document.createElement("article");
    userMessage.className = "message message-user";
    userMessage.innerHTML = `<div class="message-head"><span>你 · 第 ${index + 1} 轮</span><time>${escapeHtml(formatTime(data.run.created_at))}</time></div>`;
    const question = document.createElement("div");
    question.className = "message-body";
    question.textContent = data.run.question;
    userMessage.append(question);
    root.append(userMessage);

    const agentMessage = document.createElement("article");
    agentMessage.className = `message message-agent ${data.final_answer ? "" : "pending"}`;
    agentMessage.innerHTML = `<div class="message-head"><span>PaperPilot</span><time>${escapeHtml(data.run.status)} · ${escapeHtml(data.run.research_outcome || "-")}</time></div>`;
    const body = document.createElement("div");
    body.className = "message-body";
    body.textContent = data.final_answer?.answer || (
      followUpReady.has(data.run.status)
        ? "本轮没有生成可提交的证据化回答。"
        : "正在规划、检索并核验证据……"
    );
    agentMessage.append(body);
    if (data.run.metadata?.engine === "research_notebook") {
      const panel = document.createElement("details");
      const label = document.createElement("summary");
      label.textContent = "查看研究笔记（原文可重载，条目待复核）";
      const listing = document.createElement("div");
      const more = document.createElement("button");
      more.type = "button";
      more.textContent = "加载下一页";
      let offset = 0;
      more.onclick = async () => {
        more.disabled = true;
        try {
          const base = `/agentic/runs/${encodeURIComponent(data.run.run_id)}/notebook`;
          const value = await request(`${base}?offset=${offset}&limit=20`);
          for (const note of value.notes) {
            const item = document.createElement("p");
            item.textContent = `${note.kind} · ${note.title}: ${note.statement || note.status}`;
            if (note.batch_id) {
              const source = document.createElement("button");
              source.type = "button";
              source.textContent = "读取原文";
              source.onclick = async () => {
                try {
                  const raw = await request(`${base}/sources/${encodeURIComponent(note.batch_id)}`);
                  const pre = document.createElement("pre");
                  pre.style.whiteSpace = "pre-wrap";
                  pre.textContent = `${raw.source_uri} [${raw.start_char}, ${raw.end_char})\n${raw.source_text}`;
                  source.replaceWith(pre);
                } catch (error) { showError(error); }
              };
              item.append(source);
            }
            listing.append(item);
          }
          offset += value.notes.length;
          more.disabled = value.notes.length < 20;
          if (more.disabled) more.textContent = "已到当前末页";
        } catch (error) { more.disabled = false; showError(error); }
      };
      panel.append(label, listing, more);
      agentMessage.append(panel);
    }
    if (data.final_answer?.answer_scope === "BACKGROUND_ONLY") {
      const warning = document.createElement("p");
      warning.textContent = "核心问题尚未获得支持；以上仅为有来源的背景信息，不是完整答案。";
      agentMessage.append(warning);
    }
    if (data.final_answer?.limitations?.length && !data.final_answer.limitations_in_answer) {
      const limitations = document.createElement("div");
      const label = document.createElement("p");
      label.textContent = "回答边界与尚未解决的问题：";
      limitations.append(label);
      for (const text of data.final_answer.limitations) {
        const item = document.createElement("p");
        item.textContent = text;
        limitations.append(item);
      }
      agentMessage.append(limitations);
    }
    const unverifiedNotes = (data.final_answer?.review_notes || []).filter(
      (note) => note.status === "UNVERIFIED_MODEL_NOTE" && typeof note.text === "string"
    );
    if (unverifiedNotes.length) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = `待核查的模型说明（${unverifiedNotes.length} 条，不作为事实结论）`;
      details.append(summary);
      const caution = document.createElement("p");
      caution.textContent = "这些说明未通过事实确认，可能包含错误推断；仅供对照原文复核，不属于答案依据。";
      details.append(caution);
      for (const note of unverifiedNotes) {
        const item = document.createElement("p");
        item.textContent = note.text;
        details.append(item);
      }
      agentMessage.append(details);
    }
    if (data.final_answer?.review_required) {
      const review = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "查看需要核验的内容";
      review.append(summary);
      const claims = new Map((data.claim_graph?.claims || []).map((claim) => [claim.claim_id, claim]));
      for (const id of data.final_answer.review_claim_ids || []) {
        const item = document.createElement("p");
        item.textContent = claims.get(id)?.statement || "部分机制解释仍需对照原文确认";
        review.append(item);
      }
      agentMessage.append(review);
    }
    if (data.final_answer?.citation_ids?.length) {
      const citations = document.createElement("div");
      citations.className = "citations";
      data.final_answer.citation_ids.forEach((evidenceId) => {
        citations.append(evidenceButton(evidenceId, data));
      });
      agentMessage.append(citations);
    }
    root.append(agentMessage);
  });
}

function renderClaims(data) {
  const claims = data.claim_graph?.claims || [];
  $("claim-count").textContent = `${claims.length} 个`;
  const root = $("claims");
  root.replaceChildren();
  claims.forEach((claim) => {
    const card = document.createElement("article");
    const head = document.createElement("div");
    head.className = "section-head";
    head.innerHTML = `<strong>${escapeHtml(claim.statement)}</strong><span class="claim-status">${escapeHtml(claim.status)}</span>`;
    card.append(head);
    const gaps = document.createElement("p");
    gaps.className = "meta";
    gaps.textContent = claim.gaps?.length ? `剩余缺口：${claim.gaps.join("；")}` : "证据缺口已闭合或已终止";
    card.append(gaps);
    const buttons = document.createElement("div");
    buttons.className = "citations";
    (claim.evidence_ids || []).forEach((id) => buttons.append(evidenceButton(id, data)));
    card.append(buttons);
    root.append(card);
  });
}

function renderEvents(data) {
  const root = $("events");
  root.replaceChildren();
  (data.events || []).slice(-14).reverse().forEach((event) => {
    const card = document.createElement("article");
    const payload = event.payload || event.data || {};
    card.innerHTML = `<strong>${escapeHtml(event.event_type || "event")}</strong><p>${escapeHtml(eventSummary({ ...payload, event_type: event.event_type }))}</p>`;
    root.append(card);
  });
}

function renderApprovals(data) {
  const approvals = data.pending_approvals || [];
  $("approvals-section").hidden = !approvals.length;
  const root = $("approvals");
  root.replaceChildren();
  approvals.forEach((approval) => {
    const card = document.createElement("article");
    card.innerHTML = `<strong>${escapeHtml(approval.tool_name || "受控工具")}</strong><p class="meta">${escapeHtml(JSON.stringify(approval.context || approval.metadata || {}, null, 2))}</p>`;
    const actions = document.createElement("div");
    actions.className = "citations";
    const approve = document.createElement("button");
    approve.className = "primary";
    approve.textContent = "批准并恢复";
    approve.addEventListener("click", () => decideApproval(approval.approval_id, true));
    const reject = document.createElement("button");
    reject.className = "primary danger";
    reject.textContent = "拒绝";
    reject.addEventListener("click", () => decideApproval(approval.approval_id, false));
    actions.append(approve, reject);
    card.append(actions);
    root.append(card);
  });
}

async function decideApproval(id, approved) {
  const action = approved ? "approve" : "reject";
  const data = await request(`/api/agentic/approvals/${encodeURIComponent(id)}/${action}`, {
    method: "POST",
    body: JSON.stringify({ actor: "paperpilot-local-ui", resume: true }),
  });
  await refreshRuns();
  await loadConversation(conversationId(data.run), false);
  if (!streamTerminal.has(data.run.status)) watchRun(data.run.run_id);
}

function evidenceButton(id, data) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "evidence-btn";
  button.textContent = id;
  button.addEventListener("click", () => openEvidence(id, data));
  return button;
}

function openEvidence(id, data) {
  const evidence = (data.evidence || []).find((item) => item.evidence_id === id);
  if (!evidence) return showError(`找不到证据 ${id}`);
  $("evidence-title").textContent = id;
  $("evidence-meta").textContent = `${evidence.polarity || "NEUTRAL"} · ${evidence.source_uri || "local artifact"} · page ${evidence.page || "-"}`;
  $("evidence-content").textContent = evidence.content || "无文本内容";
  const image = $("evidence-image");
  if (evidence.artifact_path && /\.(png|jpe?g|webp)$/i.test(evidence.artifact_path)) {
    image.src = `/api/agentic/runs/${encodeURIComponent(data.run.run_id)}/evidence/${encodeURIComponent(id)}/artifact`;
    image.hidden = false;
  } else {
    image.hidden = true;
    image.removeAttribute("src");
  }
  $("evidence-dialog").showModal();
}

function eventSummary(event) {
  const type = event.event_type || "正在等待后台状态";
  const reason = event.reason || event.stop_reason || event.error || event.status || "";
  const query = event.query || event.queries?.join("；") || "";
  return [type, query, reason].filter(Boolean).join(" · ");
}

function formatTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function renderComposerState() {
  const submit = $("send-question");
  if (!submit) return;
  const engine = $("engine-select");
  engine.disabled = !state.newConversation && Boolean(state.current);
  if (engine.disabled) engine.value = state.current.run.metadata?.engine === "notebook_query"
    ? "research_notebook" : state.current.run.metadata?.engine === "workset_review"
      ? "answer_first_text" : state.current.run.metadata?.engine || "legacy";
  if (!state.bootstrap?.llm_configured) {
    submit.disabled = true;
    submit.textContent = "LLM 未配置";
    $("composer-hint").textContent = "尚未配置文本 LLM API；请先填写 .env。";
    return;
  }
  if (state.newConversation || !state.current) {
    submit.disabled = false;
    submit.textContent = "开始新研究";
    $("composer-hint").textContent = "这将创建一个新对话；之后的问题默认作为连续追问。";
    return;
  }
  if (followUpReady.has(state.current.run.status)) {
    submit.disabled = false;
    submit.textContent = "发送追问";
    $("composer-hint").textContent = `继续当前对话（已完成 ${state.thread.length} 轮），复用会话上下文但重新核验证据。模式沿用该对话；切换模式请新建对话。`;
    return;
  }
  submit.disabled = true;
  submit.textContent = state.current.run.status === "WAITING_APPROVAL" ? "等待审批" : "Agent 运行中";
  $("composer-hint").textContent = "当前轮次尚未结束；可以查看实时轨迹，或点击左侧“新建对话”并行研究。";
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
}

function showError(error) {
  const toast = $("toast");
  toast.textContent = error instanceof Error ? error.message : String(error);
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 3500);
}

$("ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("question-input");
  const question = input.value.trim();
  if (!question) return;
  try {
    if (!state.newConversation && state.current) {
      if (!followUpReady.has(state.current.run.status)) {
        throw new Error("当前轮次尚未结束，请等待完成或新建对话");
      }
      await createFollowUp(state.current.run.run_id, question);
    } else {
      await createRun(question);
    }
    input.value = "";
  } catch (error) { showError(error); }
});
$("new-run").addEventListener("click", () => {
  state.source?.close();
  state.source = null;
  state.current = null;
  state.thread = [];
  state.conversationId = null;
  state.newConversation = true;
  localStorage.removeItem("paperpilot-run-id");
  $("run-view").hidden = true;
  $("empty-state").hidden = false;
  $("run-status").textContent = "IDLE";
  $("conversation-thread").replaceChildren();
  renderRunList();
  renderComposerState();
  $("question-input").focus();
});
$("refresh-runs").addEventListener("click", () => refreshRuns().catch(showError));
$("workset-review").addEventListener("click", async () => {
  if (state.worksetStarting || !state.current) return;
  state.worksetStarting = true;
  $("workset-review").disabled = true;
  try {
    const id = state.current.run.run_id;
    const data = await request(`/api/agentic/runs/${encodeURIComponent(id)}/workset-review`, {
      method: "POST", body: JSON.stringify({}),
    });
    localStorage.setItem("paperpilot-run-id", data.run.run_id);
    await refreshRuns();
    await loadConversation(conversationId(data.run), true);
  } catch (error) { showError(error); }
  finally { state.worksetStarting = false; $("workset-review").disabled = false; }
});
$("close-dialog").addEventListener("click", () => $("evidence-dialog").close());
boot().catch(showError);
