"""Durable, capture-preemptible optional-work queue (stdlib only).

Contract: JobStore(path).enqueue(stage, stage_version, input_signature,
payload=None, dependencies=()) returns a content/dependency-addressed job ID.
claim(owner=current_owner()) / next(...) atomically leases one eligible job.
The returned lease_token must accompany heartbeat/checkpoint/finish/pause/block/
fail. finish validates ownership, never silently accepts another worker's result.
Failed and externally blocked work requires retry(); dependency-blocked work is
automatically eligible when every declared dependency is complete.

A timed-out heartbeat is NOT process-death proof. recover_abandoned() only
recovers an authoritative OS-confirmed absent/exited owner, or a different
process creation identity at the same PID. Access denied/unknown/live owners
remain leased, including after lease expiry. A trusted injected process_probe
is supported for deterministic tests; never use untrusted client-supplied proof.
No worker process is started, stopped or killed by this module.
"""
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import runtime_clock as time
import uuid

from offline_jobs import check_work

STATES = ('pending', 'running', 'paused', 'blocked', 'failed', 'complete')


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _identity(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def probe_process(pid):
    """Return OS-backed state and creation identity; uncertainty fails closed."""
    if type(pid) is not int or pid <= 0:
        return {'state': 'unknown', 'authoritative': False, 'reason': 'invalid_pid'}
    if os.name == 'nt':
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        handle = kernel.OpenProcess(0x1000 | 0x100000, False, pid)  # query-limited + synchronize
        if not handle:
            error = ctypes.get_last_error()
            return {'state': 'dead' if error == 87 else 'unknown', 'authoritative': error == 87,
                    'pid': pid, 'reason': 'no_such_pid' if error == 87 else 'query_denied_or_failed'}
        try:
            times = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                return {'state': 'unknown', 'authoritative': False, 'pid': pid, 'reason': 'creation_query_failed'}
            created = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
            wait = kernel.WaitForSingleObject(handle, 0)
            if wait not in (0, 258):
                return {'state': 'unknown', 'authoritative': False, 'pid': pid, 'reason': 'liveness_query_failed'}
            return {'state': 'dead' if wait == 0 else 'alive', 'authoritative': True, 'pid': pid,
                    'start_token': 'windows-filetime:' + str(created)}
        finally:
            kernel.CloseHandle(handle)
    if Path('/proc').is_dir():
        try:
            raw = Path('/proc', str(pid), 'stat').read_text()
        except FileNotFoundError:
            return {'state': 'dead', 'authoritative': True, 'pid': pid, 'reason': 'no_such_pid'}
        except OSError:
            return {'state': 'unknown', 'authoritative': False, 'pid': pid, 'reason': 'identity_unavailable'}
        try:
            fields = raw[raw.rfind(')') + 2:].split()
            boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            return {'state': 'dead' if fields[0] in ('Z', 'X') else 'alive', 'authoritative': True,
                    'pid': pid, 'start_token': 'proc:' + boot + ':' + fields[19]}
        except (OSError, ValueError, IndexError):
            pass
    return {'state': 'unknown', 'authoritative': False, 'pid': pid, 'reason': 'identity_unavailable'}


def current_owner():
    value = probe_process(os.getpid())
    if value.get('state') != 'alive' or not value.get('authoritative') or not value.get('start_token'):
        raise RuntimeError('Cannot establish this worker process identity')
    return {'pid': os.getpid(), 'start_token': value['start_token']}


class LeaseLost(RuntimeError):
    pass


class JobStore:
    def __init__(self, path, *, clock=time.time, process_probe=probe_process):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock, self.process_probe = clock, process_probe
        with self._db() as db:
            db.executescript('''
CREATE TABLE IF NOT EXISTS jobs(
 id TEXT PRIMARY KEY, stage TEXT NOT NULL, stage_version TEXT NOT NULL,
 input_signature TEXT NOT NULL, payload TEXT NOT NULL, state TEXT NOT NULL,
 reason TEXT, checkpoint TEXT NOT NULL DEFAULT '{}', result TEXT,
 owner_pid INTEGER, owner_start TEXT, lease_token TEXT, lease_until REAL,
 created REAL NOT NULL, updated REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
 priority INTEGER NOT NULL DEFAULT 100);
CREATE INDEX IF NOT EXISTS jobs_ready ON jobs(state,created,id);
CREATE TABLE IF NOT EXISTS dependencies(
 job_id TEXT NOT NULL, dependency_id TEXT NOT NULL,
 PRIMARY KEY(job_id,dependency_id), FOREIGN KEY(job_id) REFERENCES jobs(id));
CREATE TABLE IF NOT EXISTS job_events(
 sequence INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
 at REAL NOT NULL, state TEXT NOT NULL, reason TEXT);
''')
            if 'priority' not in {row['name'] for row in db.execute('PRAGMA table_info(jobs)')}:
                db.execute('ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 100')
            db.execute('CREATE INDEX IF NOT EXISTS jobs_priority ON jobs(state,priority,created,id)')

    @contextmanager
    def _db(self, immediate=False):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=FULL')
        db.execute('PRAGMA foreign_keys=ON')
        try:
            if immediate:
                db.execute('BEGIN IMMEDIATE')
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _row(row):
        if row is None:
            return None
        value = dict(row)
        for name in ('input_signature', 'payload', 'checkpoint', 'result'):
            if value[name] is not None:
                value[name] = json.loads(value[name])
        return value

    def _event(self, db, identifier, state, reason=None):
        db.execute('INSERT INTO job_events(job_id,at,state,reason) VALUES(?,?,?,?)',
                   (identifier, self.clock(), state, reason))

    def enqueue(self, stage, stage_version, input_signature, payload=None, dependencies=(), *, priority=100):
        if not isinstance(stage, str) or not stage or not isinstance(stage_version, str) or not stage_version:
            raise ValueError('Stage and stage_version are required')
        if type(priority) is not int or not 0 <= priority <= 1000:
            raise ValueError('Job priority must be an integer from 0 to 1000')
        deps = sorted(set(dependencies))
        if any(not isinstance(v, str) or not v for v in deps):
            raise ValueError('Dependency identifiers must be nonempty strings')
        key = {'stage': stage, 'stage_version': stage_version, 'input_signature': input_signature,
               'dependencies': deps}
        identifier = _identity(key)
        if identifier in deps:
            raise ValueError('A job cannot depend on itself')
        encoded_payload = _json({} if payload is None else payload)
        with self._db(True) as db:
            old = db.execute('SELECT payload FROM jobs WHERE id=?', (identifier,)).fetchone()
            if old:
                if old['payload'] != encoded_payload:
                    raise ValueError('Identical job identity has conflicting payload; change its input signature')
                db.execute('UPDATE jobs SET priority=? WHERE id=?',(priority,identifier))
                return identifier
            now = self.clock()
            db.execute('INSERT INTO jobs(id,stage,stage_version,input_signature,payload,state,created,updated,priority) VALUES(?,?,?,?,?,?,?,?,?)',
                       (identifier, stage, stage_version, _json(input_signature), encoded_payload, 'pending', now, now,priority))
            db.executemany('INSERT INTO dependencies VALUES(?,?)', ((identifier, d) for d in deps))
            self._event(db, identifier, 'pending', 'enqueued')
        return identifier

    def get(self, identifier):
        with self._db() as db:
            return self._row(db.execute('SELECT * FROM jobs WHERE id=?', (identifier,)).fetchone())

    def counts(self):
        with self._db() as db:
            result = {s: 0 for s in STATES}
            result.update({r['state']: r['n'] for r in db.execute('SELECT state,COUNT(*) n FROM jobs GROUP BY state')})
            return result

    def runnable(self, limit=1, *, stages=None):
        """Read-only eligible work; optional stage filter applies before LIMIT."""
        if type(limit) is not int or not 1<=limit<=1000:raise ValueError('Runnable page limit must be 1..1000')
        selected=None
        if stages is not None:
            if not isinstance(stages,(tuple,list,set,frozenset)) or len(stages)>16:raise ValueError('Invalid stage allowlist')
            if any(type(x) is not str or not x.isidentifier() or len(x)>64 for x in stages):raise ValueError('Invalid allowed stage')
            selected=tuple(sorted(set(stages)))
            if not selected:return []
        query="""SELECT j.* FROM jobs j
WHERE (j.state IN ('pending','paused') OR (j.state='blocked' AND j.reason='dependencies'))
AND NOT EXISTS(SELECT 1 FROM dependencies d LEFT JOIN jobs prerequisite ON prerequisite.id=d.dependency_id
               WHERE d.job_id=j.id AND (prerequisite.id IS NULL OR prerequisite.state!='complete'))
"""
        args=()
        if selected is not None:query+='AND j.stage IN ('+','.join('?' for _ in selected)+')\n';args=selected
        query+='ORDER BY j.priority,j.created,j.id LIMIT ?'
        with self._db() as db:return [self._row(row) for row in db.execute(query,args+(limit,))]

    def recent(self, limit=20, *, include_details=True):
        """Read-only bounded GUI snapshot, newest activity first."""
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('Recent page limit must be 1..1000')
        with self._db() as db:
            if not include_details:
                result=[]
                for row in db.execute('SELECT id,stage,stage_version,state,reason,checkpoint,updated,created,attempts,priority FROM jobs ORDER BY updated DESC,id LIMIT ?',(limit,)):
                    value=dict(row);value['checkpoint']=json.loads(value['checkpoint']);result.append(value)
                return result
            return [self._row(row) for row in db.execute('SELECT * FROM jobs ORDER BY updated DESC,id LIMIT ?', (limit,))]

    def completed(self, stage, limit=20):
        if type(limit) is not int or not 1<=limit<=100:raise ValueError('Completed page limit must be 1..100')
        with self._db() as db:
            return [self._row(row) for row in db.execute("SELECT * FROM jobs WHERE state='complete' AND stage=? ORDER BY updated DESC,id LIMIT ?",(stage,limit))]

    def claim(self, owner=None, *, job_id=None, lease_seconds=30, cancel=None, exclusive=False):
        check_work(cancel)
        owner = current_owner() if owner is None else owner
        if type(owner.get('pid')) is not int or owner['pid'] <= 0 or not isinstance(owner.get('start_token'), str) or not owner['start_token']:
            raise ValueError('Owner needs PID and OS creation identity')
        proof = self.process_probe(owner['pid'])
        if (not proof.get('authoritative') or proof.get('state') != 'alive'
                or proof.get('start_token') != owner['start_token']):
            raise ValueError('Worker identity is not authoritatively live')
        if not 1 <= lease_seconds <= 3600:
            raise ValueError('Lease duration outside 1..3600 seconds')
        with self._db(True) as db:
            if exclusive and db.execute("SELECT 1 FROM jobs WHERE state='running' LIMIT 1").fetchone():
                return None
            query = "SELECT * FROM jobs WHERE (state IN ('pending','paused') OR (state='blocked' AND reason='dependencies'))"
            args = ()
            if job_id is not None:
                query += ' AND id=?'
                args = (job_id,)
            for row in db.execute(query + ' ORDER BY priority,created,id', args):
                check_work(cancel)
                missing = db.execute("SELECT COUNT(*) FROM dependencies d LEFT JOIN jobs j ON j.id=d.dependency_id WHERE d.job_id=? AND (j.id IS NULL OR j.state!='complete')", (row['id'],)).fetchone()[0]
                if missing:
                    if row['state'] != 'blocked':
                        db.execute("UPDATE jobs SET state='blocked',reason='dependencies',updated=? WHERE id=?", (self.clock(), row['id']))
                        self._event(db, row['id'], 'blocked', 'dependencies')
                    continue
                token = uuid.uuid4().hex
                now = self.clock()
                db.execute("UPDATE jobs SET state='running',reason=NULL,owner_pid=?,owner_start=?,lease_token=?,lease_until=?,updated=?,attempts=attempts+1 WHERE id=?",
                           (owner['pid'], owner['start_token'], token, now + lease_seconds, now, row['id']))
                self._event(db, row['id'], 'running', 'claimed')
                return self._row(db.execute('SELECT * FROM jobs WHERE id=?', (row['id'],)).fetchone())
        return None

    next = claim

    def _owned(self, db, identifier, token):
        row = db.execute('SELECT * FROM jobs WHERE id=?', (identifier,)).fetchone()
        if row is None or row['state'] != 'running' or not token or row['lease_token'] != token:
            raise LeaseLost('Job lease is no longer owned')
        return row

    def heartbeat(self, identifier, token, *, lease_seconds=30):
        if not 1 <= lease_seconds <= 3600:
            raise ValueError('Lease duration outside 1..3600 seconds')
        with self._db(True) as db:
            self._owned(db, identifier, token)
            db.execute('UPDATE jobs SET lease_until=?,updated=? WHERE id=?',
                       (self.clock() + lease_seconds, self.clock(), identifier))

    def checkpoint(self, identifier, token, value, *, cancel=None):
        check_work(cancel)
        encoded = _json(value)
        if len(encoded.encode('utf-8')) > 1_048_576:
            raise ValueError('Keep checkpoint metadata under 1 MiB; arrays belong in committed products')
        with self._db(True) as db:
            self._owned(db, identifier, token)
            db.execute('UPDATE jobs SET checkpoint=?,updated=? WHERE id=?', (encoded, self.clock(), identifier))

    def _end(self, identifier, token, state, reason=None, result=None):
        if state not in ('complete', 'paused', 'blocked', 'failed'):
            raise ValueError('Invalid terminal/pause state')
        encoded = None if result is None else _json(result)
        if encoded is not None and len(encoded.encode('utf-8'))>4*1024*1024:
            raise ValueError('Job result metadata must fit 4 MiB; large data belongs in committed artifacts')
        with self._db(True) as db:
            self._owned(db, identifier, token)
            db.execute('UPDATE jobs SET state=?,reason=?,result=?,owner_pid=NULL,owner_start=NULL,lease_token=NULL,lease_until=NULL,updated=? WHERE id=?',
                       (state, reason, encoded, self.clock(), identifier))
            self._event(db, identifier, state, reason)

    def finish(self, identifier, token, result=None):
        self._end(identifier, token, 'complete', result=result)

    def pause(self, identifier, token, reason='capture_preempted'):
        self._end(identifier, token, 'paused', reason)

    def block(self, identifier, token, reason):
        self._end(identifier, token, 'blocked', reason)

    def fail(self, identifier, token, reason):
        self._end(identifier, token, 'failed', reason)

    def retry(self, identifier, reason='dependency_or_operator_retry'):
        with self._db(True) as db:
            row = db.execute('SELECT state FROM jobs WHERE id=?', (identifier,)).fetchone()
            if row is None:
                raise KeyError(identifier)
            if row['state'] not in ('failed', 'blocked', 'paused'):
                raise ValueError('Only failed, blocked or paused jobs can be retried')
            db.execute("UPDATE jobs SET state='pending',reason=?,updated=? WHERE id=?", (reason, self.clock(), identifier))
            self._event(db, identifier, 'pending', reason)

    def blocked(self, reason, limit=100):
        if type(limit) is not int or not 1<=limit<=1000:raise ValueError('Blocked page limit must be 1..1000')
        with self._db() as db:
            return [self._row(row) for row in db.execute("SELECT * FROM jobs WHERE state='blocked' AND reason=? ORDER BY updated,id LIMIT ?",(reason,limit))]

    def fail_dispatch(self, identifier, reason):
        """Record a failed launch/pre-claim exit without touching a live lease."""
        with self._db(True) as db:
            row=db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone()
            if row is None:raise KeyError(identifier)
            if row['state'] not in ('pending','paused') and not (row['state']=='blocked' and row['reason']=='dependencies'):
                return False
            if row['owner_pid'] is not None:return False
            db.execute("UPDATE jobs SET state='failed',reason=?,updated=? WHERE id=?",(reason,self.clock(),identifier))
            self._event(db,identifier,'failed',reason)
            return True

    def fail_dead_owner(self, identifier, expected_lease_token, reason):
        """A terminal observed worker crash, after OS death proof and token CAS."""
        row=self.get(identifier)
        if not row or row['state']!='running' or row['lease_token']!=expected_lease_token:return False
        proof=self.process_probe(row['owner_pid'])
        dead=proof.get('authoritative') is True and (proof.get('state')=='dead' or
              proof.get('state')=='alive' and proof.get('start_token') and proof['start_token']!=row['owner_start'])
        if not dead:return False
        with self._db(True) as db:
            current=db.execute('SELECT state,lease_token FROM jobs WHERE id=?',(identifier,)).fetchone()
            if current['state']!='running' or current['lease_token']!=expected_lease_token:return False
            db.execute("UPDATE jobs SET state='failed',reason=?,owner_pid=NULL,owner_start=NULL,lease_token=NULL,lease_until=NULL,updated=? WHERE id=?",
                       (reason,self.clock(),identifier))
            self._event(db,identifier,'failed',reason)
            return True

    def recover_abandoned(self, *, cancel=None):
        """Expired/live and unqueryable owners stay running; death is required."""
        result = {'recovered': [], 'live': [], 'unknown': []}
        with self._db() as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM jobs WHERE state='running'")]
        for row in rows:
            check_work(cancel)
            proof = self.process_probe(row['owner_pid'])
            authoritative = proof.get('authoritative') is True
            dead = authoritative and (proof.get('state') == 'dead' or
                    proof.get('state') == 'alive' and proof.get('start_token') and proof['start_token'] != row['owner_start'])
            if not dead:
                result['live' if authoritative and proof.get('state') == 'alive' else 'unknown'].append(row['id'])
                continue
            with self._db(True) as db:
                # Do not overwrite a finished/reclaimed lease since the OS query.
                current = db.execute('SELECT state,lease_token FROM jobs WHERE id=?', (row['id'],)).fetchone()
                if current['state'] != 'running' or current['lease_token'] != row['lease_token']:
                    continue
                reason = 'owner_dead' if proof.get('state') == 'dead' else 'owner_pid_reused'
                db.execute("UPDATE jobs SET state='paused',reason=?,owner_pid=NULL,owner_start=NULL,lease_token=NULL,lease_until=NULL,updated=? WHERE id=?", (reason, self.clock(), row['id']))
                self._event(db, row['id'], 'paused', reason)
                result['recovered'].append(row['id'])
        return result
