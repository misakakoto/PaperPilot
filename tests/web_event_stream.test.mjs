import test from 'node:test';
import assert from 'node:assert/strict';
import {RunEvents} from '../src/paperpilot_agentic/web_static/api.js';

function fixture(run) {
  const old={EventSource:globalThis.EventSource,setTimeout:globalThis.setTimeout,clearTimeout:globalThis.clearTimeout};
  const instances=[],timers=new Map();let id=0;
  class Source {
    constructor(url){this.url=url;instances.push(this);}
    close(){this.closed=true;}
    send(id,type='progress'){this.onmessage({lastEventId:String(id),data:JSON.stringify({event_id:id,event_type:type})});}
  }
  globalThis.EventSource=Source;
  globalThis.setTimeout=(fn,ms)=>{timers.set(++id,{fn,ms});return id;};
  globalThis.clearTimeout=id=>timers.delete(id);
  try {run({instances,timers,tick(){const first=timers.entries().next().value;timers.delete(first[0]);first[1].fn();}});}
  finally{Object.assign(globalThis,old);}
}

test('reconnect resumes cursor, de-duplicates, and never resubmits a job',()=>fixture(({instances,timers,tick})=>{
  const events=[];let refresh=0;
  const client=new RunEvents('owned-run',e=>events.push(e),()=>refresh++,()=>{});
  instances[0].send(7);tick();assert.equal(refresh,1);
  instances[0].onerror();assert.equal(instances[0].closed,true);tick();
  assert.equal(instances.length,2);assert.match(instances[1].url,/after=7$/);
  instances[1].send(7);instances[1].send(8);assert.equal(events.length,2);
  client.close();assert.equal(timers.size,0);assert.equal(instances[1].closed,true);
}));

test('continuous events cannot starve activity refresh and final output is fetched',()=>fixture(({instances,timers,tick})=>{
  let refresh=0;new RunEvents('run',()=>{},()=>refresh++,()=>{});
  for(let i=1;i<501;i++)instances[0].send(i);
  assert.equal(timers.size,1);tick();assert.equal(refresh,1);
  instances[0].send(501,'stream.end');assert.equal(refresh,2);assert.equal(instances[0].closed,true);
}));

test('closing during a reconnect prevents a fresh subscription',()=>fixture(({instances,timers})=>{
  const client=new RunEvents('run',()=>{},()=>{},()=>{});
  instances[0].onerror();assert.equal(timers.size,1);client.close();assert.equal(timers.size,0);
  instances[0].onerror();assert.equal(instances.length,1);assert.equal(timers.size,0);
}));
