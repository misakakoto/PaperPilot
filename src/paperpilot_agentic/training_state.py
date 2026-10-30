"""Durable local GPU reservations and content-bound training checkpoints.

Unknown process outcomes retain their reservation. This is not a remote exactly-once
protocol or a security sandbox. This file is also copied into the isolated worker.
"""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def process_identity(pid):
    """Return an OS creation identity; permission failures remain UNKNOWN."""
    if type(pid) is not int or pid <= 0:
        return 'UNKNOWN', None
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ('DEAD' if ctypes.get_last_error() == 87 else 'UNKNOWN'), None
        try:
            values = [wintypes.FILETIME() for _ in range(4)]
            kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
            if not kernel.GetProcessTimes(handle, *(ctypes.byref(x) for x in values)):
                return 'UNKNOWN', None
            exit_code = wintypes.DWORD()
            kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return 'UNKNOWN', None
            identity = str((values[0].dwHighDateTime << 32) | values[0].dwLowDateTime)
            return ('RUNNING' if exit_code.value == 259 else 'DEAD'), identity
        finally:
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle(handle)
    try:
        row = Path(f'/proc/{pid}/stat').read_text()
        fields = row[row.rfind(')') + 2:].split()
        return ('DEAD' if fields[0] == 'Z' else 'RUNNING'), fields[19]
    except FileNotFoundError:
        return 'DEAD', None
    except OSError:
        return 'UNKNOWN', None


class TrainingBusy(RuntimeError):
    pass


class GpuBudget:
    """One campaign ledger, shared by preflight, training and recovery checks."""
    def __init__(self, path, cap_seconds=28800):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if type(cap_seconds) not in (int, float) or not 0 < cap_seconds <= 28800:
            raise ValueError('GPU budget must be positive and at most eight hours')
        with self.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS campaign(id INTEGER PRIMARY KEY CHECK(id=1), cap REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS launches(
                    token TEXT PRIMARY KEY, stage TEXT NOT NULL, status TEXT NOT NULL,
                    reserved REAL NOT NULL, charged REAL NOT NULL DEFAULT 0,
                    started REAL NOT NULL, pid INTEGER, identity TEXT, ended REAL);
                CREATE TABLE IF NOT EXISTS runs(binding TEXT PRIMARY KEY, run_id TEXT NOT NULL);
            ''')
            db.execute('INSERT OR IGNORE INTO campaign VALUES(1,?)', (cap_seconds,))
            if db.execute('SELECT cap FROM campaign').fetchone()[0] != cap_seconds:
                raise ValueError('Existing campaign budget cannot be reset or changed')

    @contextlib.contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def snapshot(self):
        with self.connection() as db:
            cap = db.execute('SELECT cap FROM campaign').fetchone()[0]
            rows = [dict(x) for x in db.execute('SELECT * FROM launches ORDER BY started,token')]
        charged = sum(x['charged'] for x in rows)
        held = sum(x['reserved'] for x in rows if x['status'] in ('RESERVED', 'RUNNING', 'UNKNOWN'))
        return {'cap_seconds': cap, 'charged_seconds': charged, 'reserved_seconds': held,
                'remaining_seconds': max(0, cap - charged - held), 'launches': rows}

    def recover(self, resolver=process_identity):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            for row in db.execute("SELECT * FROM launches WHERE status IN ('RESERVED','RUNNING','UNKNOWN')"):
                status, identity = resolver(row['pid'])
                if status == 'RUNNING' and identity == row['identity']:
                    raise TrainingBusy('Owned GPU worker is still running; do not launch another')
                if status == 'UNKNOWN' or row['pid'] is None:
                    raise TrainingBusy('GPU launch outcome UNKNOWN; inspect original process before retry')
                # Confirmed dead (or PID reused): its uncertain duration is charged
                # at the entire reservation, never silently refunded on resume.
                db.execute("UPDATE launches SET status='INTERRUPTED',charged=reserved,ended=? WHERE token=?",
                           (time.time(), row['token']))

    def reserve(self, stage, max_seconds=3600):
        if not isinstance(stage, str) or not stage or not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError('Invalid stage or reservation')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT 1 FROM launches WHERE status IN ('RESERVED','RUNNING','UNKNOWN')").fetchone():
                raise TrainingBusy('An unfinished GPU launch already owns this campaign')
            cap = db.execute('SELECT cap FROM campaign').fetchone()[0]
            spent = db.execute('SELECT COALESCE(SUM(charged),0) FROM launches').fetchone()[0]
            seconds = min(max_seconds, cap - spent)
            if seconds <= 0:
                raise TimeoutError('Cumulative GPU budget exhausted')
            token = uuid.uuid4().hex
            db.execute('INSERT INTO launches(token,stage,status,reserved,started) VALUES(?,?,?,?,?)',
                       (token, stage, 'RESERVED', seconds, time.time()))
        return token, seconds

    def attach(self, token):
        status, identity = process_identity(os.getpid())
        if status != 'RUNNING':
            raise TrainingBusy('Cannot verify worker process ownership')
        with self.connection() as db:
            changed = db.execute("UPDATE launches SET status='RUNNING',pid=?,identity=? WHERE token=? AND status='RESERVED'",
                                 (os.getpid(), identity, token)).rowcount
            if changed != 1:
                raise TrainingBusy('GPU launch token is not an unused reservation')
            row = dict(db.execute('SELECT * FROM launches WHERE token=?', (token,)).fetchone())
        return row

    def finish(self, token, status, elapsed):
        if status not in ('COMPLETED', 'FAILED', 'PAUSED') or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Invalid GPU completion receipt')
        with self.connection() as db:
            row = db.execute('SELECT * FROM launches WHERE token=?', (token,)).fetchone()
            if row is None or row['status'] not in ('RUNNING', 'RESERVED'):
                raise ValueError('GPU reservation already settled or unknown')
            db.execute('UPDATE launches SET status=?,charged=?,ended=? WHERE token=?',
                       (status, elapsed, time.time(), token))

    def claim_run(self, binding, run_id):
        with self.connection() as db:
            db.execute('INSERT OR IGNORE INTO runs VALUES(?,?)', (binding, run_id))
            return db.execute('SELECT run_id FROM runs WHERE binding=?', (binding,)).fetchone()[0]


def save_checkpoint(folder, payload, identity):
    import torch
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / (uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        torch.save({'identity': identity, 'payload': payload}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    digest = file_sha(temporary)
    destination = folder / (digest + '.pt')
    os.replace(temporary, destination)
    pointer = folder / 'latest.json'
    previous = json.loads(pointer.read_bytes()) if pointer.exists() else None
    receipt = {'file': destination.name, 'sha256': digest, 'identity': identity,
               'step': payload['step'], 'previous': previous['file'] if previous else None}
    with (folder / 'commits.jsonl').open('ab') as stream:
        stream.write(json.dumps(receipt, sort_keys=True).encode() + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    atomic_json(pointer, receipt)
    # Bounded retention applies only to newly generated checkpoint payloads.
    # Immutable commit hashes survive; no historic answers or protocols are deleted.
    if previous and previous.get('previous'):
        old = folder / previous['previous']
        if old.name.endswith('.pt') and old.resolve().parent == folder.resolve() and old != destination:
            old.unlink(missing_ok=True)
    return receipt


def load_checkpoint(folder, identity):
    import torch
    folder = Path(folder)
    pointer = folder / 'latest.json'
    if not pointer.exists():
        return None
    receipt = json.loads(pointer.read_bytes())
    path = folder / receipt['file']
    if (path.is_symlink() or path.resolve().parent != folder.resolve() or receipt['identity'] != identity
            or path.name != receipt['sha256'] + '.pt' or file_sha(path) != receipt['sha256']):
        raise ValueError('Training checkpoint source, identity or payload changed')
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value['identity'] != identity or value['payload']['step'] != receipt['step']:
        raise ValueError('Training checkpoint contents disagree with committed receipt')
    return value['payload']
