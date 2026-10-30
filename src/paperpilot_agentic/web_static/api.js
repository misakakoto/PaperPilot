export const API = '/api/agentic';
export async function request(path, options = {}) {
  const response = await fetch(path.startsWith('/api/') ? path : `${API}${path}`, {
    ...options, headers:{'Content-Type':'application/json',...(options.headers || {})},
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error=new Error(typeof data.detail === 'string' ? data.detail : `请求失败（${response.status}）`);
    error.status=response.status;throw error;
  }
  return data;
}
export function post(path, body = {}) { return request(path,{method:'POST',body:JSON.stringify(body)}); }

// Cursor belongs to one run. Reconnect never submits or resumes a business action.
export class RunEvents {
  constructor(runId, onEvent, onRefresh, onConnection) {
    this.runId=runId; this.cursor=0; this.seen=new Set(); this.closed=false;
    this.onEvent=onEvent; this.onRefresh=onRefresh; this.onConnection=onConnection;
    this.retry=0; this.connect();
  }
  connect() {
    if (this.closed) return;
    this.source = new EventSource(`${API}/runs/${encodeURIComponent(this.runId)}/events?after=${this.cursor}`);
    this.source.onopen = () => { this.retry=0; this.onConnection('已连接'); };
    this.source.onmessage = event => {
      if (this.closed) return;
      const value = JSON.parse(event.data);
      if (value.event_type === 'stream.end') { this.close(); this.onRefresh(); return; }
      const id = Number(event.lastEventId || value.event_id || 0);
      if (id && this.seen.has(id)) return;
      if (id) { this.seen.add(id); this.cursor=Math.max(this.cursor,id); }
      this.onEvent(value);
      if(!this.refreshTimer) this.refreshTimer=setTimeout(() => {this.refreshTimer=null;this.onRefresh();},350);
    };
    this.source.onerror = () => {
      this.source.close();
      if (this.closed) return;
      this.onConnection('连接中断，正在恢复；不会重新执行任务');
      this.timer=setTimeout(() => { this.onRefresh(); this.connect(); },Math.min(10000,500*2**this.retry++));
    };
  }
  close() { this.closed=true; this.source?.close(); clearTimeout(this.timer); clearTimeout(this.refreshTimer); }
}
