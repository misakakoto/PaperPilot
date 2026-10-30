"""Durable store for pressure-triggered request state, not tool authorization.

Constructing this store adds only its own table. Default research flows do not
construct it until text Reader memory reaches its pressure threshold. An owned root-to-request ancestry and a nonempty conversation ID
are required; source text always comes from the authoritative runs table.
"""
import json
from .request_state import empty_request_state, apply_request_delta, apply_request_batch
from .request_history_search import history_source_ids


class RequestStateStore:
    def __init__(self, db):
        self.db = db
        with db.transaction() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS research_request_states (
                root_run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
                revision TEXT NOT NULL,
                state_json TEXT NOT NULL
            )''')

    @staticmethod
    def _owned_source(conn, root_id, requester_id):
        root = conn.execute('SELECT * FROM runs WHERE run_id=?',(root_id,)).fetchone()
        if root is None:
            raise ValueError('Unknown request-state root')
        conversation = json.loads(root['metadata_json']).get('conversation_id')
        if not isinstance(conversation,str) or not conversation:
            raise ValueError('Request state requires an explicit conversation')
        cursor, seen, source = requester_id, set(), None
        while True:
            if cursor in seen:
                raise ValueError('Request ancestry cycle')
            seen.add(cursor)
            row = conn.execute('SELECT * FROM runs WHERE run_id=?',(cursor,)).fetchone()
            if row is None:
                raise ValueError('Unknown request source or ancestor')
            meta = json.loads(row['metadata_json'])
            if meta.get('conversation_id') != conversation:
                raise ValueError('Request source is outside conversation')
            if source is None:
                source = {'id':row['run_id'],'text':row['question']}
            if cursor == root_id:
                return source
            cursor = meta.get('parent_run_id')
            if not isinstance(cursor,str) or not cursor:
                raise ValueError('Request source is outside root ancestry')

    @staticmethod
    def _load(conn, root_id):
        row = conn.execute('SELECT revision,state_json FROM research_request_states WHERE root_run_id=?',
                           (root_id,)).fetchone()
        if row is None:
            return empty_request_state()
        state = json.loads(row['state_json'])
        if state['revision'] != row['revision']:
            raise ValueError('Stored request-state revision mismatch')
        return state

    def load(self, root_id, *, requester_id, require_ancestor=False):
        with self.db.transaction(immediate=False) as conn:
            self._owned_source(conn,root_id,requester_id)
            state = self._load(conn,root_id)
            if require_ancestor and state['events']:
                self._owned_source(conn,state['events'][-1]['source_id'],requester_id)
            return state

    def commit(self, root_id, *, source_run_id, delta):
        """Serialized compare-and-apply. No LLM call occurs under this lock."""
        with self.db.transaction() as conn:
            source = self._owned_source(conn,root_id,source_run_id)
            old = self._load(conn,root_id)
            state = apply_request_delta(old,source,delta)
            self._append_path(conn,root_id,old,source_run_id)
            conn.execute('''INSERT INTO research_request_states(root_run_id,revision,state_json)
                VALUES(?,?,?) ON CONFLICT(root_run_id) DO UPDATE SET
                revision=excluded.revision,state_json=excluded.state_json''',
                (root_id,state['revision'],json.dumps(state,ensure_ascii=False)))
            return state

    def _append_path(self,conn,root_id,old,source_id):
        if source_id in old['sources']:
            return
        if not old['events']:
            if source_id!=root_id:
                raise ValueError('First request-state source must be the root request')
        else:
            self._owned_source(conn,old['events'][-1]['source_id'],source_id)

    def reload_sources(self, root_id, *, requester_id, source_ids):
        """Return original requests referenced by this root's committed ledger."""
        if not isinstance(source_ids,list) or any(not isinstance(s,str) for s in source_ids):
            raise ValueError('Expected source ID list')
        with self.db.transaction(immediate=False) as conn:
            self._owned_source(conn,root_id,requester_id)
            state = self._load(conn,root_id)
            rows = []
            for sid in dict.fromkeys(source_ids):
                if sid not in state['sources']:
                    raise ValueError('Source is not in this request ledger')
                source = self._owned_source(conn,root_id,sid)
                if source['text'] != state['sources'][sid]:
                    raise ValueError('Original request changed after state commit')
                rows.append(source)
            return rows

    def history_page(self, root_id, *, requester_id, task_ids, expected_revision,
                     offset=0, max_items=32, max_bytes=12000, query=None):
        """Read task-attributed originals chronologically or by lexical query.

        This covers committed events attributed to selected tasks, not all chat
        messages. Complete exhausts this filtered history (only lexical hits when
        query is supplied), never proves a missing historical fact. Full source
        text is returned or the page stops: no fragment silently replaces a user
        request. A too-large first source is reported for explicit handling.
        """
        if (not isinstance(task_ids, list) or not task_ids
                or any(not isinstance(t, str) for t in task_ids)
                or len(task_ids) != len(set(task_ids))):
            raise ValueError('Expected unique task IDs')
        if (type(offset) is not int or offset < 0 or type(max_items) is not int
                or not 1 <= max_items <= 64 or type(max_bytes) is not int or max_bytes < 1):
            raise ValueError('Invalid history page limits')
        with self.db.transaction(immediate=False) as conn:
            self._owned_source(conn, root_id, requester_id)
            state = self._load(conn, root_id)
            if state['revision'] != expected_revision:
                raise ValueError('Stale history revision')
            if any(t not in state['tasks'] for t in task_ids):
                raise ValueError('Unknown history task')
            if state['events']:
                self._owned_source(conn, state['events'][-1]['source_id'], requester_id)
            ids = history_source_ids(state, task_ids, query)
            if offset > len(ids):
                raise ValueError('History offset outside matching events')
            rows, used, blocked = [], 0, None
            for sid in ids[offset:offset + max_items]:
                source = self._owned_source(conn, root_id, sid)
                if source['text'] != state['sources'].get(sid):
                    raise ValueError('Original request changed after state commit')
                size = len(json.dumps(source, ensure_ascii=False).encode('utf-8'))
                if used + size > max_bytes:
                    blocked = {'source_id': sid, 'required_bytes': size}
                    break
                rows.append(source)
                used += size
            next_offset = offset + len(rows)
            complete = next_offset == len(ids)
            return {'status': 'READY' if rows or complete else 'SOURCE_TOO_LARGE',
                    'revision': state['revision'], 'task_ids': list(task_ids),
                    'coverage': 'task_attributed_query_matches' if query is not None else 'task_attributed_committed_events',
                    'order': 'lexical_relevance' if query is not None else 'oldest_first',
                    **({'query': query} if query is not None else {}),
                    'starts_at_ledger_root': bool(query is None and offset == 0 and rows and rows[0]['id'] == root_id),
                    'sources': rows, 'offset': offset, 'next_offset': None if complete else next_offset,
                    'complete': complete, 'source_bytes': used, 'blocked_source': blocked}

    def commit_batch(self, root_id, *, source_run_ids, proposal):
        """All frames commit together; caller supplies intended ancestry order."""
        if (not isinstance(source_run_ids,list) or not source_run_ids or
                len(source_run_ids)>32 or any(not isinstance(s,str) for s in source_run_ids)):
            raise ValueError('Expected one to 32 source run IDs')
        with self.db.transaction() as conn:
            sources=[self._owned_source(conn,root_id,sid) for sid in source_run_ids]
            # Prevent reordering corrections or mixing sibling branches. Neutral
            # ancestors may be skipped by callers, but provided sources must be
            # on one ancestry path in oldest-to-newest order.
            for previous,current in zip(source_run_ids,source_run_ids[1:]):
                self._owned_source(conn,previous,current)
                if previous==current:
                    raise ValueError('Duplicate source request')
            old=self._load(conn,root_id)
            state=apply_request_batch(old,sources,proposal)
            self._append_path(conn,root_id,old,source_run_ids[0])
            conn.execute('''INSERT INTO research_request_states(root_run_id,revision,state_json)
                VALUES(?,?,?) ON CONFLICT(root_run_id) DO UPDATE SET
                revision=excluded.revision,state_json=excluded.state_json''',
                (root_id,state['revision'],json.dumps(state,ensure_ascii=False)))
            return state
