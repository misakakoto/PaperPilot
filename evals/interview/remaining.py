"""Unique grant of a stopped parent's remainder; no new budget, no parent writes."""
import json
from pathlib import Path
import sqlite3
from .budget import CampaignBudget, canonical
from .campaign import historical_snapshot

VERSION = 'remaining-allowance-v1'


def grant_remaining(parent_folder, child_folder, note, *, registry_path=None):
    parent_folder, child_folder = Path(parent_folder).resolve(), Path(child_folder).resolve()
    if not note or parent_folder == child_folder:
        raise ValueError('Explicit instruction and distinct child required')
    parent = historical_snapshot(parent_folder)
    if not parent['stopped'] and not any(r['state'] in {'FAILED', 'UNKNOWN'} for r in parent['states']):
        raise ValueError('Remainder recovery requires a stopped or unresolved parent')
    if any(r['state'] in {'STARTED', 'RESERVED'} for r in parent['states']):
        raise ValueError('Reconcile a potentially live parent request first')
    requests = parent['max_requests'] - parent['actual_requests']
    tokens = parent['max_tokens'] - parent['reported_tokens'] - parent['unsettled_upper_bound']
    if requests <= 0 or tokens <= 0:
        raise ValueError('No parent allowance remains')
    registry_path = Path(registry_path or parent_folder.parent/'remaining-budget-grants.sqlite3')
    with sqlite3.connect(registry_path, timeout=10) as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute('CREATE TABLE IF NOT EXISTS grants (parent TEXT PRIMARY KEY, child TEXT UNIQUE, payload TEXT)')
        prior = conn.execute('SELECT child,payload FROM grants WHERE parent=?', (str(parent_folder),)).fetchone()
        if prior:
            grant = json.loads(prior[1])
            if prior[0] != str(child_folder) or grant['parent_ledger_sha256'] != parent['ledger_sha256']:
                raise ValueError('Parent remainder already granted or parent changed')
            return grant
        grant = {'version': VERSION, 'parent': str(parent_folder), 'child': str(child_folder),
                 'parent_ledger_sha256': parent['ledger_sha256'], 'parent_protocol_sha256': parent['protocol_sha256'],
                 'max_requests': requests, 'max_tokens': tokens, 'human_instruction': note,
                 'held_parent_tokens': parent['unsettled_upper_bound'], 'parent_states': parent['states']}
        conn.execute('INSERT INTO grants VALUES(?,?,?)', (str(parent_folder), str(child_folder), canonical(grant)))
    return grant


class RemainingCampaignBudget(CampaignBudget):
    def __init__(self, folder, identity, *, grant):
        self.grant = grant
        self._check_parent()
        super().__init__(folder, identity, max_requests=grant['max_requests'], max_tokens=grant['max_tokens'])

    def _check_parent(self):
        parent = historical_snapshot(self.grant['parent'])
        if (parent['ledger_sha256'] != self.grant['parent_ledger_sha256'] or
                parent['protocol_sha256'] != self.grant['parent_protocol_sha256']):
            raise ValueError('Parent changed after remainder grant; never reset')

    def _reserve(self, model, phase, request):
        self._check_parent()
        return super()._reserve(model, phase, request)

    @staticmethod
    def _switchable(row):
        error = json.loads(row.get('error') or '{}')
        return (error.get('provider_code') == 'AllocationQuota.FreeTierOnly' or
                (error.get('http_status') == 404 and error.get('provider_code') == 'model_not_found'))

    def _unresolved(self, rows):
        from paperpilot_agentic.model_policy import PRIMARY_MODEL
        if json.loads(self.identity).get('selection_models'):
            return super()._unresolved(rows)
        return [r for r in super()._unresolved(rows) if not (r['state'] == 'FAILED'
            and r['phase'] == 'capability:selection' and r['model'] == PRIMARY_MODEL and self._switchable(r))]

    def _held_tokens(self, rows):
        return sum(self._unknown_bound(r) for r in rows if r['state'] in {'RESERVED', 'STARTED', 'UNKNOWN', 'FAILED'})

    def _validate_model(self, model, *, phase=None):
        from paperpilot_agentic.model_policy import PRIMARY_MODEL, BACKUP_MODEL
        identity = json.loads(self.identity)
        fixed = identity.get('selection_models')
        if fixed is not None:
            if fixed != [identity.get('model')] or model != identity.get('model'):
                raise ValueError('Explicit remainder recovery keeps one authorized model')
            if phase == 'capability:selection' and any(
                    r['phase'] == phase for r in self.snapshot()['attempts']):
                raise ValueError('A selection probe cannot be resent')
            return super()._validate_model(model, phase=phase)
        if phase == 'capability:selection' and model in {PRIMARY_MODEL, BACKUP_MODEL}:
            rows = self.snapshot()['attempts']
            if any(r['model'] == model for r in rows):
                raise ValueError('A selection probe cannot be resent')
            if model == BACKUP_MODEL and not (len(rows) == 1 and self._switchable(rows[0])):
                raise ValueError('Backup needs one explicit primary unavailable receipt')
            return
        selection = self.folder.parent/'selected-model.json'
        if selection.exists():
            chosen = json.loads(selection.read_text(encoding='utf-8'))
            if model != chosen['model'] or chosen['identity'] != identity:
                raise ValueError('Answers keep the single selected model')
            return
        super()._validate_model(model, phase=phase)
