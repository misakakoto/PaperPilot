"""One durable, fail-closed spending ledger for an entire acceptance campaign.

Receipts contain original requests/responses, never credentials. Store this ledger
and its receipts under ignored outputs/. A lost response stops the campaign rather
than resetting its allowance. Legacy protocols retain UTF-8 request reservations,
including image data URLs. New Qwen3-VL protocols reuse production still-image
bounds; text remains conservatively byte-bounded. Neither is measured usage.
"""
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from paperpilot_agentic.llm_runtime import LLMBudgetExhausted


class UncertainRequest(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def save(path, value):
    """Exclusive receipts cannot be silently overwritten on replay."""
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2))


class CampaignBudget:
    def __init__(self, folder, identity, *, max_requests=60, max_tokens=600000):
        unlimited = identity.get('usage_limit_mode') == 'unlimited'
        if unlimited and (max_requests is not None or max_tokens is not None):
            raise ValueError('Unlimited usage has explicit null cumulative ceilings')
        if not unlimited and (max_requests is None or max_tokens is None or
                              not 1 <= max_requests <= 60 or not 1 <= max_tokens <= 600000):
            raise ValueError('Campaign ceiling is 60 requests and 600000 tokens')
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / 'spending.sqlite3'
        self.identity = canonical(identity)
        self.cost_policy = identity.get('request_cost_policy', 'legacy-utf8-v1')
        if self.cost_policy not in {'legacy-utf8-v1', 'qwen3-vl-still-image-v1', 'qwen35-still-image-v1'}:
            raise ValueError('Unknown frozen request cost policy')
        if self.cost_policy == 'qwen3-vl-still-image-v1':
            from paperpilot_agentic.model_policy import PRIMARY_MODEL, BACKUP_MODEL
            if identity.get('provider') != 'qwen' or identity['model'] not in {PRIMARY_MODEL, BACKUP_MODEL}:
                raise ValueError('Image bound requires the confirmed Qwen3-VL model pair')
        if self.cost_policy == 'qwen35-still-image-v1':
            from paperpilot_agentic.model_policy import DAILY_MODEL
            if identity.get('provider') != 'qwen' or identity['model'] != DAILY_MODEL:
                raise ValueError('Image bound requires the confirmed Qwen3.5 snapshot')
        with self.transaction() as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS campaign '
                         '(id INTEGER PRIMARY KEY CHECK(id=1), identity TEXT, requests INTEGER, tokens INTEGER)')
            conn.execute('CREATE TABLE IF NOT EXISTS attempts '
                         '(id INTEGER PRIMARY KEY, state TEXT, reservation INTEGER, tokens INTEGER, '
                         'model TEXT, phase TEXT, elapsed REAL, error TEXT)')
            conn.execute('CREATE TABLE IF NOT EXISTS continuations '
                         '(identity TEXT PRIMARY KEY, authorization TEXT NOT NULL)')
            row = conn.execute('SELECT * FROM campaign').fetchone()
            if row is None:
                conn.execute('INSERT INTO campaign VALUES(1,?,?,?)',
                             (self.identity, max_requests, max_tokens))
            elif (row['identity'], row['requests'], row['tokens']) != (
                    self.identity, max_requests, max_tokens):
                approved = conn.execute('SELECT authorization FROM continuations WHERE identity=?',
                                        (self.identity,)).fetchone()
                if (approved is None or (row['requests'], row['tokens']) != (max_requests, max_tokens)):
                    raise ValueError('Frozen campaign identity or ceiling changed')
            self.continuation = None
            if row is not None and row['identity'] != self.identity:
                self.continuation = json.loads(approved['authorization'])

    def authorize_continuation(self, identity, *, unknown_bounds, evidence):
        """Explicit human-authorized new protocol; carry cost, never clear old UNKNOWN.

        Bounds must come from retained requests or provider hard limits, not a
        guess that a connection error was free. They are not reported usage.
        The original protocol remains blocked and cannot replay its checkpoint.
        """
        target = canonical(identity)
        self._retain_cost_policy(identity)
        if self.continuation is not None or target == self.identity or not evidence:
            raise ValueError('A new protocol and documented continuation evidence are required')
        bounds = {str(k): v for k, v in unknown_bounds.items()}
        if any(type(v) is not int or v <= 0 for v in bounds.values()):
            raise ValueError('Unresolved token bounds must be positive integers')
        with self.transaction() as conn:
            rows = [dict(r) for r in conn.execute('SELECT * FROM attempts ORDER BY id')]
            unresolved = [r for r in rows if r['state'] in {'RESERVED', 'STARTED', 'UNKNOWN', 'FAILED'}]
            if (not unresolved or any(r['state'] != 'UNKNOWN' for r in unresolved) or
                    set(bounds) != {str(r['id']) for r in unresolved}):
                raise ValueError('Every unresolved attempt must be UNKNOWN with its own bound')
            if json.loads(target)['model'] != json.loads(self.identity)['model']:
                raise ValueError('This continuation keeps the frozen model')
            limits = conn.execute('SELECT * FROM campaign').fetchone()
            if limits['tokens'] is not None and sum(r['tokens'] for r in rows) + sum(bounds.values()) >= limits['tokens']:
                raise LLMBudgetExhausted('No allowance remains after carried unknown costs')
            authorization = {'parent_identity': json.loads(self.identity),
                             'unknown_bounds': bounds, 'evidence': evidence,
                             'attempts_at_authorization': rows}
            conn.execute('INSERT INTO continuations VALUES(?,?)', (target, canonical(authorization)))
        return authorization

    def authorize_settled_revision(self, identity, *, evidence):
        """Freeze a code repair against the same allowance, never bypass UNKNOWN."""
        target = canonical(identity)
        self._retain_cost_policy(identity)
        if target == self.identity or not evidence or (self.folder/'stopped.json').exists():
            raise ValueError('A new revision requires evidence and an unstopped campaign')
        if json.loads(target)['model'] != json.loads(self.identity)['model']:
            raise ValueError('A revision retains the frozen model')
        with self.transaction() as conn:
            rows = [dict(r) for r in conn.execute('SELECT * FROM attempts ORDER BY id')]
            if any(row['state'] not in {'RETURNED', 'NOT_SENT'} for row in rows):
                raise ValueError('A supplemental revision cannot bypass an unresolved request')
            authorization = {'parent_identity': json.loads(self.identity), 'unknown_bounds': {},
                             'evidence': evidence, 'attempts_at_authorization': rows}
            conn.execute('INSERT INTO continuations VALUES(?,?)', (target, canonical(authorization)))
        return authorization

    def _unknown_bound(self, row):
        carried = (self.continuation or {}).get('unknown_bounds', {})
        return carried.get(str(row['id']), row['reservation'])

    def authorize_json_rejection_revision(self, identity, *, evidence):
        """Repair one demonstrably malformed JSON request; retain its full cost."""
        target = canonical(identity)
        self._retain_cost_policy(identity)
        if (target == self.identity or not evidence.get('human_instruction') or
                (self.folder/'stopped.json').exists() or
                json.loads(target)['model'] != json.loads(self.identity)['model']):
            raise ValueError('A rejected-request repair retains the authorized model and allowance')
        with self.transaction() as conn:
            rows = [dict(r) for r in conn.execute('SELECT * FROM attempts ORDER BY id')]
            failed = [r for r in rows if r['state'] == 'FAILED']
            if len(failed) != 1 or any(r['state'] not in {'FAILED', 'RETURNED', 'NOT_SENT'} for r in rows):
                raise ValueError('JSON repair cannot bypass UNKNOWN or other unresolved requests')
            row = failed[0]
            error = json.loads((self.folder/f"failure-{row['id']:03}.json").read_text(encoding='utf-8'))
            request = json.loads((self.folder/f"request-{row['id']:03}.json").read_text(encoding='utf-8'))
            if (error.get('http_status') != 400 or error.get('provider_code') != 'invalid_parameter_error' or
                    request.get('json_mode') is not True or request.get('model') != row['model'] or
                    any('json' in str(m.get('content', '')).casefold() for m in request['messages'])):
                raise ValueError('Only retained HTTP 400 missing-JSON-instruction requests qualify')
            authorization = {'parent_identity': json.loads(self.identity),
                'unknown_bounds': {str(row['id']): row['reservation']},
                'evidence': {**evidence, 'rejected_request_policy': 'provider400-json-keyword-v1'},
                'attempts_at_authorization': rows}
            conn.execute('INSERT INTO continuations VALUES(?,?)', (target, canonical(authorization)))
        return authorization

    def _carried_json_rejection(self, row):
        return (row['state'] == 'FAILED' and
                (self.continuation or {}).get('evidence', {}).get('rejected_request_policy') ==
                'provider400-json-keyword-v1' and
                str(row['id']) in (self.continuation or {}).get('unknown_bounds', {}))

    def _unresolved(self, rows):
        carried = (self.continuation or {}).get('unknown_bounds', {})
        return [r for r in rows if r['state'] in {'RESERVED', 'STARTED', 'UNKNOWN', 'FAILED'}
                and not (r['state'] == 'UNKNOWN' and str(r['id']) in carried)
                and not self._carried_json_rejection(r)]

    @contextmanager
    def transaction(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute('BEGIN IMMEDIATE')
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def snapshot(self):
        with self.transaction() as conn:
            rows = [dict(r) for r in conn.execute('SELECT * FROM attempts ORDER BY id')]
            limits = dict(conn.execute('SELECT * FROM campaign').fetchone())
        return {'max_requests': limits['requests'], 'max_tokens': limits['tokens'],
                'actual_requests': sum(r['state'] != 'NOT_SENT' for r in rows),
                'reported_tokens': sum(r['tokens'] for r in rows),
                'unsettled_upper_bound': self._held_tokens(rows),
                'blocked': (self.folder / 'stopped.json').exists() or
                           bool(self._unresolved(rows)),
                'attempts': rows, 'continuation': self.continuation}

    def _held_tokens(self, rows):
        return sum(self._unknown_bound(r) for r in rows
                   if r['state'] in {'RESERVED', 'STARTED', 'UNKNOWN'} or self._carried_json_rejection(r))

    @staticmethod
    def request_upper_bound(request):
        output = request.get('max_tokens')
        if type(output) is not int or output <= 0:
            raise ValueError('A bounded provider max_tokens is required')
        return len(canonical(request['messages']).encode('utf-8')) + output + 2048

    def _retain_cost_policy(self, identity):
        if identity.get('request_cost_policy', 'legacy-utf8-v1') != self.cost_policy:
            raise ValueError('A revision retains the frozen request cost policy')

    def estimated_request_bound(self, request):
        """New frozen protocols share the production still-image estimate.

        Legacy records keep their original bound, including UNKNOWN reservations.
        Text-only calls retain the historical UTF-8 safety bound in both modes.
        """
        legacy = self.request_upper_bound(request)
        if self.cost_policy == 'legacy-utf8-v1':
            return legacy
        messages = request['messages']
        if not any(isinstance(m.get('content'), list) for m in messages):
            return legacy
        from paperpilot_agentic.vision_dispatch import vision_input_upper_bound
        return vision_input_upper_bound(messages) + request['max_tokens']

    def _capacity_failure(self, message, *, upper, used, held, limit):
        error = LLMBudgetExhausted(message)
        error.budget_diagnostic = {'request_cost_policy': self.cost_policy,
            'required_upper_bound': upper, 'reported_tokens': used,
            'unsettled_upper_bound': held, 'max_tokens': limit,
            'available_tokens': max(0, limit - used - held)}
        return error

    def _validate_model(self, model, *, phase=None):
        identity = json.loads(self.identity)
        if model == identity['model']:
            return
        # Explicitly frozen capability probes are the only multi-model phase.
        # Answers, audits, context and ordinary legacy campaigns stay fixed.
        from paperpilot_agentic.model_policy import PRIMARY_MODEL, BACKUP_MODEL
        if (identity.get('provider') == 'qwen' and identity['model'] == PRIMARY_MODEL
                and identity.get('capability_preflight_models') == [PRIMARY_MODEL, BACKUP_MODEL]
                and phase in {'capability:text', 'capability:image'} and model == BACKUP_MODEL):
            return
        raise ValueError('Acceptance campaign cannot silently switch model')

    def ensure_capacity(self, model, requests):
        """Read-only admission for known inputs; dispatch still reserves each attempt."""
        if not requests:
            raise ValueError('At least one known request is required')
        self._validate_model(model)
        upper = sum(self.estimated_request_bound(request) for request in requests)
        snapshot = self.snapshot()
        if (self.folder / 'stopped.json').exists():
            raise LLMBudgetExhausted('Campaign has stopped; allowance cannot be reset')
        if self._unresolved(snapshot['attempts']):
            raise UncertainRequest('Campaign stopped by an interrupted or failed request; no retry')
        if snapshot['max_requests'] is not None and snapshot['actual_requests'] + len(requests) > snapshot['max_requests']:
            raise LLMBudgetExhausted('Campaign cannot fit the mandatory request sequence')
        if (snapshot['max_tokens'] is not None and
                snapshot['reported_tokens'] + snapshot['unsettled_upper_bound'] + upper > snapshot['max_tokens']):
            raise self._capacity_failure('Campaign cannot fit known mandatory input/output reservations',
                upper=upper, used=snapshot['reported_tokens'], held=snapshot['unsettled_upper_bound'],
                limit=snapshot['max_tokens'])

    def _reserve(self, model, phase, request):
        if (self.folder / 'stopped.json').exists():
            raise LLMBudgetExhausted('Campaign has stopped; allowance cannot be reset')
        # Refuse a changed model before any network dispatch.
        self._validate_model(model, phase=phase)
        upper = self.estimated_request_bound(request)
        with self.transaction() as conn:
            rows = [dict(r) for r in conn.execute('SELECT * FROM attempts')]
            if self._unresolved(rows):
                raise UncertainRequest('Campaign stopped by an interrupted or failed request; no retry')
            limits = conn.execute('SELECT * FROM campaign').fetchone()
            if limits['requests'] is not None and sum(r['state'] != 'NOT_SENT' for r in rows) >= limits['requests']:
                raise LLMBudgetExhausted('Campaign request ceiling reached')
            held = self._held_tokens(rows)
            if limits['tokens'] is not None and sum(r['tokens'] for r in rows) + held + upper > limits['tokens']:
                raise self._capacity_failure('Campaign cannot reserve complete input and output',
                    upper=upper, used=sum(r['tokens'] for r in rows), held=held, limit=limits['tokens'])
            cursor = conn.execute('INSERT INTO attempts VALUES(NULL,?,?,?,?,?,?,?)',
                                  ('RESERVED', upper, 0, model, phase, 0, None))
            return cursor.lastrowid

    def dispatch(self, *, model, phase, request, operation, before_send=None, receipt=None):
        try:
            number = self._reserve(model, phase, request)
        except Exception as exc:
            exc.request_dispatched = False
            if isinstance(exc, LLMBudgetExhausted):
                try:
                    save(self.folder / 'stopped.json', {'reason': 'budget_exhausted', 'phase': phase,
                        'model': model, **getattr(exc, 'budget_diagnostic', {})})
                except FileExistsError:
                    pass
            raise
        try:
            save(self.folder / f'request-{number:03}.json',
                 {'model': model, 'phase': phase, 'protocol_identity': json.loads(self.identity), **request})
            if before_send:
                before_send()
        except BaseException as exc:
            exc.request_dispatched = False
            with self.transaction() as conn:
                conn.execute("UPDATE attempts SET state='NOT_SENT' WHERE id=?", (number,))
            raise
        with self.transaction() as conn:
            conn.execute("UPDATE attempts SET state='STARTED' WHERE id=?", (number,))
        started = time.monotonic()
        try:
            response = operation()
        except Exception as exc:
            from paperpilot_agentic.provider_diagnostics import safe_provider_failure
            status = 'FAILED' if getattr(exc, 'status_code', None) is not None else 'UNKNOWN'
            error = safe_provider_failure(exc)
            if receipt and (raw_returned := receipt(None)) is not None:
                save(self.folder / f'response-{number:03}.json', raw_returned)
            save(self.folder / f'failure-{number:03}.json', error)
            with self.transaction() as conn:
                conn.execute('UPDATE attempts SET state=?, elapsed=?, error=? WHERE id=?',
                             (status, time.monotonic() - started, canonical(error), number))
            raise
        raw = receipt(response) if receipt else (
            asdict(response) if is_dataclass(response) else response.model_dump(mode='json'))
        usage = raw.get('usage') or raw
        prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
        # Missing usage cannot mean zero usage, even if an answer was returned.
        valid = all(type(v) is int and v >= 0 for v in (prompt, completion))
        save(self.folder / f'response-{number:03}.json', raw)
        with self.transaction() as conn:
            conn.execute('UPDATE attempts SET state=?,tokens=?,elapsed=? WHERE id=?',
                         ('RETURNED' if valid else 'UNKNOWN',
                          prompt + completion if valid else 0,
                          time.monotonic() - started, number))
        if not valid:
            raise UncertainRequest('Returned response lacks provider usage; campaign stopped')
        return response


class RecordedProvider:
    """Text seam: reserve the campaign before the production attempt counter."""
    def __init__(self, provider, campaign, *, phase):
        if getattr(provider, 'max_retries', 0) != 0:
            raise ValueError('Acceptance provider must have automatic retries disabled')
        self._wrapped_provider, self.campaign, self.phase = provider, campaign, phase
        self.raw_response = None
        if hasattr(provider, '_client'):
            owner = self
            original = provider._client.chat.completions
            class CaptureCompletions:
                def create(self, **kwargs):
                    response = original.create(**kwargs)
                    owner.raw_response = response.model_dump(mode='json')
                    return response
            provider._client.chat.completions = CaptureCompletions()

    def __getattr__(self, name):
        return getattr(self._wrapped_provider, name)

    def ensure_capacity(self, requests):
        return self.campaign.ensure_capacity(self._wrapped_provider.model, requests)

    def complete(self, messages, *, before_attempt=None, thinking_enabled=None, **kwargs):
        self.raw_response = None
        if thinking_enabled is not None:
            kwargs['thinking_enabled'] = thinking_enabled
        return self.campaign.dispatch(
            model=self._wrapped_provider.model, phase=self.phase,
            request={'messages': messages, **kwargs}, before_send=before_attempt,
            operation=lambda: self._wrapped_provider.complete(messages, before_attempt=lambda: None, **kwargs),
            receipt=lambda response: self.raw_response or (asdict(response) if response is not None else None))


class RecordedCompletions:
    """Vision SDK seam: use the same ledger, including raw returned responses."""
    def __init__(self, completions, campaign, *, phase):
        self.completions, self.campaign, self.phase = completions, campaign, phase
        self.before_send = None

    def create(self, **kwargs):
        return self.campaign.dispatch(model=kwargs['model'], phase=self.phase,
                                      request=kwargs, operation=lambda: self.completions.create(**kwargs),
                                      before_send=self.before_send)


class RecordedVision:
    """Bridge the local attempt hook into the global SDK reservation transaction."""
    def __init__(self, adapter, campaign, *, phase):
        self.adapter = adapter
        adapter.client.chat.completions = RecordedCompletions(
            adapter.client.chat.completions, campaign, phase=phase)

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def analyze(self, image_path, prompt, metadata=None, *, before_attempt=None):
        self.adapter.client.chat.completions.before_send = before_attempt
        return self.adapter.analyze(image_path, prompt, metadata, before_attempt=lambda: None)


def source_fingerprint(root):
    digest = hashlib.sha256()
    for base in ('src', 'evals/interview', 'demo'):
        for path in sorted((Path(root) / base).rglob('*')):
            if path.is_file() and path.suffix in {'.py', '.json'} and '__pycache__' not in path.parts:
                digest.update(path.relative_to(root).as_posix().encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()
