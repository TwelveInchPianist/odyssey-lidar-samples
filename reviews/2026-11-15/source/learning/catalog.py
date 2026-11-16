"""Streaming drive catalog and coarse receiver-route memory; never live FCW input.

Public API:
  index_session(folder, data_root, cancel=None, *, chunk_records=500, ...)
  Catalog(data_root).get(session_id), .sessions(), .summary()
  physical_trip_id_for_route(route_id) -> 'comma:' + normalized route ID

Only completed sessions are indexed. Call index_session from one leased optional
job for a given input inventory; this module never starts a worker or capture.
Chunks commit journal cursor, rolling content digest, route references and coarse
points together. Cancellation rolls back at most the current bounded chunk and
raises offline_jobs.WorkPaused. Resume reads from the committed cursor, not byte
zero. Export is streamed from the catalog DB and atomically replaces derived
products only. Originals/configuration/calibration flags are never modified.

The cache inventory uses journal/raw-PCAP file stat identities plus SHA256 of
small metadata. Journal bytes are cryptographically hashed as consumed using a
documented chained-record SHA256 (not a whole-file SHA256); this digest survives
chunk-size changes. Raw PCAPs are STAT-ONLY: no raw-packet hashing or decoding here.
Completed input files are assumed immutable; same-size edits that preserve all
stat fields require a separate integrity scan. `input_signature` distinguishes
this inventory cache key from consumed content digests explicitly.

Physical trip IDs come from Comma route references, not laptop-session UUIDs.
Multiple laptop sessions on one route never add independent-trip evidence.
Unknown/multiple routes abstain from a single independent-trip claim. A route ID
is an evidence-grouping convention, not proof of statistical independence.
Mount epochs are separate: absent legacy epochs are explicitly unknown and are
never inferred merely from equal mount numbers or from the date of a user note.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import runtime_clock as time
import uuid

from offline_jobs import WorkPaused, check_work

VERSION = 'drive-catalog-v1'
DEFAULT_MINIMUM_FREE_BYTES = 10_000_000_000 + 16 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_CHUNK_BYTES = 2 * 1024 * 1024
HASH_ALGORITHM = 'sha256-chain(length-prefixed-record-blocks)-v1'
META_NAMES = ('session.json', 'final-state.json', 'raw-capture-index.json', 'route-journal-status.json')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def physical_trip_id_for_route(route_id):
    if not isinstance(route_id, str):
        raise ValueError('Comma route ID must be a string')
    route_id = route_id.strip()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,159}', route_id):
        raise ValueError('Unsupported Comma route ID')
    return 'comma:' + route_id


def _regular(path):
    if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
        raise ValueError('Catalog inputs/products must not be symbolic links or junctions')


def _stat(path):
    _regular(path)
    st = path.stat()
    return {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns, 'inode': st.st_ino}


def _space(root, minimum, allowance=4 * 1024 * 1024):
    if shutil.disk_usage(root).free <= minimum + allowance:
        raise OSError('Learning catalog paused above the disk reserve')


def _inventory(folder, cancel):
    _regular(folder)
    if not (folder / 'final-state.json').is_file():
        raise ValueError('A completed recording is required for catalog ingestion')
    metadata, objects, stats = {}, {}, {}
    for name in META_NAMES:
        check_work(cancel)
        p = folder / name
        if not p.exists():
            continue
        stats[name] = _stat(p)
        if stats[name]['bytes'] > MAX_METADATA_BYTES:
            raise ValueError('Metadata file exceeds bounded ingest size: ' + name)
        raw = p.read_bytes()
        metadata[name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        try:
            value = json.loads(raw.decode('utf-8-sig'))
            objects[name] = value if isinstance(value, dict) else {'catalog_parse_error': 'not_an_object'}
        except (UnicodeError, ValueError):
            objects[name] = {'catalog_parse_error': 'invalid_json'}
    cfg = objects.get('session.json')
    if not cfg or 'catalog_parse_error' in cfg:
        raise ValueError('A valid session.json is required')
    journals = []
    for path in sorted(folder.glob('route-journal-[0-9][0-9][0-9][0-9][0-9].jsonl')):
        check_work(cancel)
        value = _stat(path)
        stats[path.name] = value
        journals.append({'name': path.name, **value})
    pcaps = []
    for path in sorted(folder.glob('lidar-*.pcap')):
        check_work(cancel)
        value = _stat(path)
        stats[path.name] = value
        pcaps.append({'name': path.name, **value})
    return {'metadata': metadata, 'journals': journals, 'raw_pcaps_stat_only': pcaps}, objects, stats


def _unchanged(folder, expected):
    names = {name for name in META_NAMES if (folder / name).exists()}
    names.update(p.name for p in folder.glob('route-journal-[0-9][0-9][0-9][0-9][0-9].jsonl'))
    names.update(p.name for p in folder.glob('lidar-*.pcap'))
    return names == set(expected) and all(_stat(folder / n) == value for n, value in expected.items())


def inventory_cache_key(folder, cancel=None):
    """Discovery key: hash small metadata and stat journals/PCAPs; read neither.

    This is deliberately a cache identity, not a claim that raw/journal content
    has been verified. index_session returns separate consumed content digests.
    """
    check_work(cancel)
    folder = Path(folder)
    _regular(folder)
    inventory, _, _ = _inventory(folder.resolve(), cancel)
    return identity({'version': VERSION, 'inventory': inventory})


def _initial():
    return {'part': 0, 'offset': 0, 'line': 0, 'digest': '0' * 64, 'discarding': False,
            'last_seq': None, 'last_epoch': None, 'segment': 0, 'break_reason': 'recording_start',
            'last_fix': None, 'last_retained_ns': None, 'stats': {}, 'phase': 'journals'}


def _inc(state, name, amount=1):
    state['stats'][name] = state['stats'].get(name, 0) + amount


def _gap(state, reason):
    state['segment'] += 1
    state['break_reason'] = reason
    state['last_fix'] = None
    state['last_retained_ns'] = None
    _inc(state, 'gap_' + reason)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _distance(a, b):
    # Haversine is only a coarse jump detector, never map registration.
    p1, p2 = math.radians(a[1]), math.radians(b[1])
    dp, dl = p2 - p1, math.radians(b[0] - a[0])
    q = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371000 * 2 * math.asin(min(1, math.sqrt(q)))


class Catalog:
    def __init__(self, data_root):
        self.root = Path(data_root) / 'learning'
        self.root.mkdir(parents=True, exist_ok=True)
        _regular(self.root)
        self.path = self.root / 'catalog.sqlite'
        self.products = self.root / 'catalog-products'
        self.products.mkdir(exist_ok=True)
        _regular(self.products)
        with self.db() as db:
            db.executescript('''
CREATE TABLE IF NOT EXISTS catalog_sessions(id TEXT PRIMARY KEY,active_revision TEXT,folder TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS catalog_revisions(id TEXT PRIMARY KEY,session_id TEXT NOT NULL,inventory TEXT NOT NULL,state TEXT NOT NULL,checkpoint TEXT NOT NULL,result TEXT,updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS catalog_journals(revision TEXT NOT NULL,part TEXT NOT NULL,bytes INTEGER NOT NULL,digest TEXT NOT NULL,PRIMARY KEY(revision,part));
CREATE TABLE IF NOT EXISTS catalog_routes(revision TEXT NOT NULL,route TEXT NOT NULL,PRIMARY KEY(revision,route));
CREATE TABLE IF NOT EXISTS catalog_points(revision TEXT NOT NULL,part TEXT NOT NULL,offset INTEGER NOT NULL,segment INTEGER NOT NULL,at_ns INTEGER NOT NULL,lon REAL NOT NULL,lat REAL NOT NULL,accuracy REAL,provider TEXT NOT NULL,time_basis TEXT NOT NULL,reason TEXT NOT NULL,source TEXT NOT NULL,PRIMARY KEY(revision,part,offset));
CREATE INDEX IF NOT EXISTS catalog_point_segments ON catalog_points(revision,segment,at_ns);
''')

    @contextmanager
    def db(self, immediate=False):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=FULL')
        db.execute('PRAGMA cache_size=-4096')
        try:
            if immediate:
                db.execute('BEGIN IMMEDIATE')
            with db:
                yield db
        finally:
            db.close()

    def get(self, session_id):
        with self.db() as db:
            row = db.execute('SELECT r.result FROM catalog_sessions s JOIN catalog_revisions r ON r.id=s.active_revision WHERE s.id=?', (session_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def sessions(self, limit=1000):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('Catalog session page limit must be 1..1000')
        with self.db() as db:
            return [json.loads(row[0]) for row in db.execute('SELECT r.result FROM catalog_sessions s JOIN catalog_revisions r ON r.id=s.active_revision ORDER BY s.id LIMIT ?', (int(limit),))]

    def summary(self):
        with self.db() as db:
            sessions = db.execute('SELECT COUNT(*) FROM catalog_sessions WHERE active_revision IS NOT NULL').fetchone()[0]
            routes = db.execute('SELECT COUNT(DISTINCT t.route) FROM catalog_routes t JOIN catalog_sessions s ON s.active_revision=t.revision').fetchone()[0]
            unknown = db.execute('SELECT COUNT(*) FROM catalog_sessions s WHERE s.active_revision IS NOT NULL AND NOT EXISTS(SELECT 1 FROM catalog_routes t WHERE t.revision=s.active_revision)').fetchone()[0]
        return {'sessions': sessions, 'known_comma_route_groups': routes, 'unknown_route_sessions': unknown,
                'independent_trip_claim': 'Only one known route per sample qualifies; shared routes count once. Route groups do not prove statistical independence.'}


def _consume(db, revision, session_id, part, offset, line_no, event, state):
    if not isinstance(event, dict) or event.get('schema_version') != 1 or type(event.get('seq')) is not int or event['seq'] < 0:
        _inc(state, 'invalid_records'); _gap(state, 'invalid_record'); return
    if event.get('session_id') not in (None, session_id):
        _inc(state, 'foreign_session_records'); _gap(state, 'foreign_session'); return
    seq, previous = event['seq'], state['last_seq']
    if previous is not None and seq <= previous:
        _inc(state, 'non_increasing_sequence'); _gap(state, 'sequence_regression'); return
    if previous is not None and seq != previous + 1:
        _inc(state, 'missing_sequence_records', seq - previous - 1); _gap(state, 'missing_sequence')
    state['last_seq'] = seq
    epoch = event.get('epoch')
    if state['last_epoch'] is not None and epoch != state['last_epoch']:
        _gap(state, 'stream_epoch')
    state['last_epoch'] = epoch
    kind, data = event.get('kind'), event.get('data')
    if not isinstance(data, dict):
        _inc(state, 'invalid_payloads'); _gap(state, 'invalid_payload'); return
    _inc(state, 'valid_journal_records')
    if kind in ('journal_open', 'stream_restart', 'source_clock_discontinuity', 'invalid_source_time'):
        _gap(state, kind); return
    if kind == 'comma_recording_reference':
        try:
            route = physical_trip_id_for_route(data.get('route')).split(':', 1)[1]
            db.execute('INSERT OR IGNORE INTO catalog_routes VALUES(?,?)', (revision, route))
        except ValueError:
            _inc(state, 'invalid_route_references')
        return
    if kind != 'sample':
        return
    service = data.get('service')
    _inc(state, 'sample_records')
    if data.get('valid') is not True:
        _inc(state, 'invalid_source_samples')
    fix = data.get('location')
    if not isinstance(fix, dict):
        return
    provider = fix.get('provider') or service
    last = state['last_fix']
    at = fix.get('at_ns')
    position = fix.get('position') or {}
    lon, lat = position.get('longitude_deg'), position.get('latitude_deg')
    usable = (data.get('valid') is True and fix.get('usable_for_trace') is True and type(at) is int and at > 0
              and isinstance(provider, str) and 0 < len(provider) <= 100
              and _number(lon) and _number(lat) and -180 <= lon <= 180 and -90 <= lat <= 90)
    if not usable:
        _inc(state, 'unusable_location_records')
        if last is None or provider == last['provider']:
            _gap(state, 'invalid_location')
        return
    if last and provider != last['provider'] and 0 <= at - last['at_ns'] <= 2_000_000_000:
        _inc(state, 'correlated_alternate_provider_not_joined'); return
    accuracy = fix.get('horizontal_accuracy_m')
    accuracy = float(accuracy) if _number(accuracy) and accuracy > 0 else None
    if last:
        delta = (at - last['at_ns']) / 1e9
        reason = ('provider_change' if provider != last['provider'] else
                  'non_increasing_location_time' if delta <= 0 else
                  'location_gap' if delta > 5 else
                  'geographic_segment' if fix.get('geographic_segment') != last.get('geographic_segment') else None)
        if reason:
            _gap(state, reason)
        elif _distance(last['coordinate'], [lon, lat]) > 70 * delta + 3 * ((accuracy or 50) + (last['accuracy'] or 50)) + 10:
            _gap(state, 'location_jump')
    state['last_fix'] = {'at_ns': at, 'coordinate': [lon, lat], 'accuracy': accuracy,
                         'provider': provider, 'geographic_segment': fix.get('geographic_segment')}
    _inc(state, 'usable_coarse_location_records')
    if state['last_retained_ns'] is not None and at - state['last_retained_ns'] < 1_000_000_000:
        _inc(state, 'coarse_points_downsampled'); return
    source = {'part': part, 'byte_offset': offset, 'line': line_no, 'journal_seq': seq,
              'stream_epoch': epoch, 'source_ns': fix.get('source_ns'), 'receipt_ns': event.get('receipt_ns'),
              'reported_horizontal_accuracy_m': accuracy, 'provider': provider,
              'time_basis': str(fix.get('time_basis', 'unverified'))[:100],
              'measurement_time_verified': fix.get('measurement_time_verified') is True,
              'coordinate_reference': fix.get('position_frame', 'WGS84'), 'not_ground_truth': True}
    db.execute('INSERT OR IGNORE INTO catalog_points VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
               (revision, part, offset, state['segment'], at, lon, lat, accuracy, provider,
                source['time_basis'], state['break_reason'], canonical(source)))
    state['last_retained_ns'] = at
    _inc(state, 'coarse_points_retained')


class _Writer:
    def __init__(self, path, root, minimum, cancel):
        self.path, self.root, self.minimum, self.cancel = path, root, minimum, cancel
        self.temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        self.file = self.temporary.open('xb')
        self.hash = hashlib.sha256(); self.bytes = 0; self.checked = 0

    def write(self, text):
        check_work(self.cancel)
        raw = text.encode('utf-8')
        if self.bytes - self.checked >= 1024 * 1024 or not self.bytes:
            _space(self.root, self.minimum, max(4 * 1024 * 1024, len(raw)))
            self.checked = self.bytes
        self.file.write(raw); self.hash.update(raw); self.bytes += len(raw)

    def finish(self):
        self.file.flush(); os.fsync(self.file.fileno()); self.file.close()
        os.replace(self.temporary, self.path)
        return {'path': str(self.path), 'sha256': self.hash.hexdigest(), **_stat(self.path)}

    def abort(self):
        if not self.file.closed:
            self.file.close()
        if self.temporary.exists():
            self.temporary.unlink()  # only this writer's newly-created derived temp


def _export(store, revision, minimum, cancel):
    folder = store.products / revision
    folder.mkdir(exist_ok=True)
    _regular(folder)
    route = _Writer(folder / 'coarse-route.geojson', store.root, minimum, cancel)
    points = _Writer(folder / 'coarse-route-points.jsonl', store.root, minimum, cancel)
    feature_count = 0
    try:
        route.write('{"type":"FeatureCollection","properties":{"product":"coarse_receiver_trace","not_ground_truth":true,"warning_override":false,"not_calibrated_atlas":true},"features":[')
        with store.db() as db:
            groups = db.execute('SELECT segment,COUNT(*) n,MIN(at_ns) start,MAX(at_ns) end,MAX(accuracy) accuracy_max,SUM(accuracy IS NULL) accuracy_missing,MIN(provider) provider,MIN(time_basis) time_basis,MIN(reason) reason FROM catalog_points WHERE revision=? GROUP BY segment ORDER BY segment', (revision,))
            for group in groups:
                check_work(cancel)
                if feature_count:
                    route.write(',')
                props = {'segment': group['segment'], 'provider': group['provider'], 'start_at_ns': group['start'],
                         'end_at_ns': group['end'], 'retained_points': group['n'], 'start_reason': group['reason'],
                         'max_reported_horizontal_accuracy_m': group['accuracy_max'], 'accuracy_missing_points': group['accuracy_missing'],
                         'time_basis': group['time_basis'], 'not_ground_truth': True, 'warning_override': False,
                         'precision_note': 'Reported receiver accuracy is not surveyed accuracy; this is a coarse trace, not a calibrated map.'}
                geometry = 'LineString' if group['n'] > 1 else 'Point'
                route.write('{"type":"Feature","properties":' + canonical(props) + ',"geometry":{"type":' + canonical(geometry) + ',"coordinates":')
                if group['n'] > 1:
                    route.write('[')
                count = 0
                for row in db.execute('SELECT * FROM catalog_points WHERE revision=? AND segment=? ORDER BY at_ns,part,offset', (revision, group['segment'])):
                    check_work(cancel)
                    if count:
                        route.write(',')
                    coordinate = [row['lon'], row['lat']]
                    route.write(canonical(coordinate))
                    detail = {'segment': row['segment'], 'at_ns': row['at_ns'], 'coordinate_lon_lat': coordinate,
                              **json.loads(row['source'])}
                    points.write(canonical(detail) + '\n'); count += 1
                route.write((']' if group['n'] > 1 else '') + '}}')
                feature_count += 1
        route.write(']}')
        check_work(cancel)
        return {'route': route.finish(), 'points': points.finish(), 'segments': feature_count}
    finally:
        route.abort(); points.abort()


def _products_ok(result):
    try:
        return all(_stat(Path(result['products'][name]['path'])) == {k: result['products'][name][k] for k in ('bytes', 'mtime_ns', 'inode')} for name in ('route', 'points'))
    except (OSError, KeyError, TypeError, ValueError):
        return False


def index_session(folder, data_root, cancel=None, *, chunk_records=500, minimum_free_bytes=DEFAULT_MINIMUM_FREE_BYTES):
    """Index one completed drive; resume bounded journal chunks on subsequent calls."""
    if type(chunk_records) is not int or not 1 <= chunk_records <= 5000:
        raise ValueError('chunk_records must be 1..5000')
    if not _number(minimum_free_bytes) or minimum_free_bytes < 0:
        raise ValueError('The free-space reserve must be a finite nonnegative byte count')
    check_work(cancel)
    _regular(Path(folder))
    folder = Path(folder).resolve()
    inventory, objects, expected = _inventory(folder, cancel)
    cfg = objects['session.json']
    session_id = cfg.get('session_id')
    if not isinstance(session_id, str) or not session_id or len(session_id) > 200:
        session_id = 'legacy:' + identity({'session_metadata': inventory['metadata'].get('session.json'), 'raw_inventory': inventory['raw_pcaps_stat_only']})
    inventory_key = identity({'version': VERSION, 'inventory': inventory})
    revision = identity({'session_id': session_id, 'inventory_key': inventory_key})
    anchor = Path(data_root)
    while not anchor.exists():
        anchor = anchor.parent
    _space(anchor, minimum_free_bytes)
    store = Catalog(data_root)
    _space(store.root, minimum_free_bytes)
    with store.db(True) as db:
        db.execute('INSERT OR IGNORE INTO catalog_sessions VALUES(?,NULL,?)', (session_id, str(folder)))
        db.execute('INSERT OR IGNORE INTO catalog_revisions VALUES(?,?,?,?,?,?,?)',
                   (revision, session_id, canonical(inventory), 'pending', canonical(_initial()), None, time.time()))
        row = db.execute('SELECT * FROM catalog_revisions WHERE id=?', (revision,)).fetchone()
        if row['state'] == 'complete' and row['result']:
            result = json.loads(row['result'])
            if _products_ok(result):
                db.execute('UPDATE catalog_sessions SET active_revision=?,folder=? WHERE id=?', (revision, str(folder), session_id))
                return {**result, 'cache_hit': True}
        db.execute("UPDATE catalog_revisions SET state='running',updated=? WHERE id=?", (time.time(), revision))
    journals = inventory['journals']
    try:
        while True:
            check_work(cancel); _space(store.root, minimum_free_bytes)
            with store.db(True) as db:
                state = json.loads(db.execute('SELECT checkpoint FROM catalog_revisions WHERE id=?', (revision,)).fetchone()[0])
                if state['phase'] != 'journals':
                    break
                records, consumed = 0, 0
                while state['part'] < len(journals) and records < chunk_records and consumed < MAX_CHUNK_BYTES:
                    part = journals[state['part']]
                    with (folder / part['name']).open('rb') as stream:
                        stream.seek(state['offset'])
                        while records < chunk_records and consumed < MAX_CHUNK_BYTES:
                            check_work(cancel)
                            start = stream.tell(); raw = stream.readline(MAX_LINE_BYTES + 1)
                            if not raw:
                                db.execute('INSERT OR REPLACE INTO catalog_journals VALUES(?,?,?,?)', (revision, part['name'], state['offset'], state['digest']))
                                state.update(part=state['part'] + 1, offset=0, line=0, digest='0' * 64, discarding=False)
                                break
                            state['offset'] = stream.tell(); records += 1; consumed += len(raw)
                            state['digest'] = hashlib.sha256(bytes.fromhex(state['digest']) + len(raw).to_bytes(8, 'big') + raw).hexdigest()
                            if state['discarding']:
                                state['discarding'] = not raw.endswith(b'\n')
                                continue
                            state['line'] += 1
                            if len(raw) > MAX_LINE_BYTES:
                                _inc(state, 'oversized_records'); _gap(state, 'oversized_record')
                                state['discarding'] = not raw.endswith(b'\n'); continue
                            if not raw.endswith(b'\n'):
                                _inc(state, 'truncated_records'); _gap(state, 'truncated_record'); continue
                            try:
                                event = json.loads(raw)
                            except (UnicodeError, ValueError):
                                _inc(state, 'invalid_json_records'); _gap(state, 'invalid_record'); continue
                            _consume(db, revision, session_id, part['name'], start, state['line'], event, state)
                _inc(state, 'journal_bytes_hashed', consumed)
                if state['part'] >= len(journals):
                    state['phase'] = 'export'
                db.execute('UPDATE catalog_revisions SET checkpoint=?,updated=? WHERE id=?', (canonical(state), time.time(), revision))
        products = _export(store, revision, minimum_free_bytes, cancel)
        check_work(cancel)
        if not _unchanged(folder, expected):
            raise ValueError('Completed source changed during indexing; re-index its new inventory')
        with store.db() as db:
            state = json.loads(db.execute('SELECT checkpoint FROM catalog_revisions WHERE id=?', (revision,)).fetchone()[0])
            routes = [r[0] for r in db.execute('SELECT route FROM catalog_routes WHERE revision=? ORDER BY route', (revision,))]
            digests = {r['part']: {'bytes': r['bytes'], 'digest': r['digest'], 'algorithm': HASH_ALGORITHM} for r in db.execute('SELECT * FROM catalog_journals WHERE revision=? ORDER BY part', (revision,))}
        content = {'small_metadata_sha256': inventory['metadata'], 'consumed_journal_digests': digests,
                   'raw_pcap_content_hashes': None, 'raw_pcap_validation': 'stat inventory only; packets not read by catalog'}
        input_signature = identity({'catalog_version': VERSION, 'inventory_cache_key': inventory_key, 'content_digests': content})
        mount = cfg.get('mount') if isinstance(cfg.get('mount'), dict) else {}
        mount_epoch = cfg.get('mount_epoch')
        epoch_id = mount_epoch.get('id') if isinstance(mount_epoch, dict) else mount_epoch
        epoch_known = isinstance(epoch_id, str) and 0 < len(epoch_id) <= 200
        if not epoch_known:
            epoch_id = 'legacy-unknown:' + session_id
        blockers = [name for name, ok in [('mount_measurements_required', mount.get('measured') is True),
                    ('mount_alignment_unverified', mount.get('alignment_verified') is True),
                    ('timing_unverified', cfg.get('timing_verified') is True)] if not ok]
        final = objects.get('final-state.json', {})
        stat_keys = ('udp_packets', 'invalid_packets', 'duplicate_packets', 'incomplete_frames', 'processed_frames',
                     'processing_frames_dropped', 'comma_records_unsaved', 'comma_records_rejected', 'engine_deadline_overruns')
        capture_stats = final.get('stats') if isinstance(final.get('stats'), dict) else {}
        quality = {'capture_stats': {k: capture_stats[k] for k in stat_keys if k in capture_stats},
                   'journal_diagnostics': state['stats'], 'no_journal': not journals,
                   'metadata_parse_errors': [name for name, value in objects.items() if 'catalog_parse_error' in value],
                   'recording_status': final.get('recording'), 'comma_recording_status': final.get('comma_recording')}
        physical_source = cfg.get('capture_mode') in ('live', 'comma') and not cfg.get('synthetic_fixture')
        footprint = cfg.get('hindsight_footprint') if isinstance(cfg.get('hindsight_footprint'), dict) else {}
        result = {'schema_version': 1, 'algorithm': VERSION, 'status': 'complete', 'cache_hit': False,
                  'session_id': session_id, 'source_folder': str(folder), 'catalog_revision': revision,
                  'capture_mode': cfg.get('capture_mode'), 'synthetic_fixture': bool(cfg.get('synthetic_fixture')),
                  'inventory_cache_key': inventory_key, 'input_signature': input_signature, 'content_digests': content,
                  'comma_route_ids': routes, 'physical_trip_ids': [physical_trip_id_for_route(r) for r in routes],
                  'independent_trip_eligible': len(routes) == 1 and physical_source,
                  'trip_identity_status': 'nonphysical_demo_or_fixture' if not physical_source else 'one_known_comma_route' if len(routes) == 1 else 'multiple_routes_do_not_claim_independence' if routes else 'unknown_no_independent_trip_evidence',
                  'mount_epoch': epoch_id, 'mount_epoch_known': epoch_known, 'recorded_mount_epoch': mount_epoch,
                  'recorded_mount': mount, 'calibration_blockers': blockers, 'atlas_eligible': not blockers,
                  'footprint_confirmed': footprint.get('confirmed') is True,
                  'configuration_signature': identity({k: cfg.get(k) for k in ('mount', 'mount_epoch', 'timing_verified', 'lidar_time_adjustment_s', 'lidar_timing_bound_s', 'sensor_ip', 'sensor_serial')}),
                  'quality': quality, 'products': products, 'coarse_route_available': products['segments'] > 0,
                  'warning_override': False, 'not_ground_truth': True,
                  'limitations': ['A known route groups one physical-drive evidence source; repeated laptop sessions on it are not independent trips.',
                                 'Unknown/multiple Comma routes do not establish a single independent trip.',
                                 'Receiver coordinates and reported accuracy are coarse provenance, not surveyed mapping or independently verified timing.',
                                 'Cataloging does not change calibration, activate an atlas, train a model, or alter warning checks.']}
        _space(store.root, minimum_free_bytes)
        with store.db(True) as db:
            db.execute("UPDATE catalog_revisions SET state='complete',result=?,updated=? WHERE id=?", (canonical(result), time.time(), revision))
            db.execute('UPDATE catalog_sessions SET active_revision=?,folder=? WHERE id=?', (revision, str(folder), session_id))
        return result
    except BaseException as exc:
        with store.db(True) as db:
            db.execute('UPDATE catalog_revisions SET state=?,updated=? WHERE id=?', ('paused' if isinstance(exc, WorkPaused) else 'failed', time.time(), revision))
        raise
