"""Isolated compiled-app mapping smoke test; never measures road performance."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import runtime_clock as time
import numpy as np
from fcw_engine import Mount
from .journal import RouteJournal
from .pipeline import update_atlas
from .store import AtlasStore
from .trajectory import build_trajectory


class ControlledFrames:
    version = 'packaged-synthetic-repeat-v1'

    def __init__(self):
        x, y = np.meshgrid(np.arange(0, 100, .5), np.arange(-10, 10, .5))
        self.world = np.concatenate([
            np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)]),
            np.random.default_rng(831).uniform([10, -12, .5], [55, 12, 5], (1000, 3))])

    def signature(self, folder):
        return {'synthetic_cloud_sha256': hashlib.sha256(self.world.tobytes()).hexdigest()}

    def __call__(self, folder, cancel=None):
        cfg = json.loads((Path(folder) / 'session.json').read_text(encoding='utf-8'))
        mount = Mount(**cfg['mount'])
        for i in range(61):
            body = self.world - [i * .2, 0, 0]
            # Keep only the forward M1-sized viewing wedge.
            body = body[(body[:, 0] > 0) & (np.abs(np.arctan2(body[:, 1], body[:, 0])) < np.deg2rad(60))]
            raw = np.column_stack([body - mount.xyz, np.full(len(body), 25.)])
            at = 100_000_000_000 + i * 100_000_000
            yield raw, np.full(len(raw), at, dtype=np.int64), {
                'at_ns': at, 'motion_ready': True, 'ready': True,
                'speed_mps': 2., 'yaw_rate_rps': 0., 'path': np.array([[0, 0], [80, 0]])}, {
                'timestamp_valid': True, 'completeness': 1., 'timing_bound_s': .01,
                'first_device_ns': at, 'context_line': i + 1,
                'raw_capture_last_wall_ns': cfg['capture_started_wall_ns'] + i * 100_000_000,
                'raw_capture_completion_part': 'SYNTHETIC-NO-PCAP'}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def make_session(root, name, day):
    folder = root / 'recordings' / name
    folder.mkdir(parents=True)
    wall = time.time_ns() - int((2-day) * 86400 * 1e9)
    write_json(folder / 'session.json', {
        'capture_mode': 'live', 'synthetic_fixture': True, 'session_id': name,
        'synthetic_drive_group': 'fixture:'+name, 'mount_epoch': {'id':'fixture-mount'},
        'timing_verified': True, 'lidar_time_adjustment_s': 0., 'lidar_timing_bound_s': .01,
        'mount': asdict(Mount(measured=True, alignment_verified=True)),
        'capture_started_wall_ns': wall})
    write_json(folder / 'final-state.json', {})
    journal = RouteJournal(folder, name, minimum_free_bytes=0)
    try:
        journal.ingest({'kind': 'hello', 'services': ['gpsLocation'], 'laptop_receipt_ns': 98_000_000_000})
        for i in range(8):
            at = 100_000_000_000 + i * 1_000_000_000
            journal.ingest({'kind': 'sample', 'service': 'gpsLocation', 'valid': True,
                'logMonoTime': at, 'laptop_receipt_ns': at, 'comma_wall_ns': wall + i * 1_000_000_000,
                'data': {'hasFix': True, 'latitude': 28.5, 'longitude': -81.3 + i * .00002,
                    'altitude': 30., 'source': 'qcomdiag', 'unixTimestampMillis': wall // 1_000_000 + i * 1000}},
                sync=(0, .001))
    finally:
        journal.close()
    route = build_trajectory(folder)
    if route['gps_state'] != 'coarse_trace' or route['geographic_segments'] != 1:
        raise AssertionError(('Coarse GPS was lost or promoted to precise fusion', route))
    return folder


def check(root):
    from evaluation import summarize
    a, b = make_session(root, 'synthetic-pass-A', 0), make_session(root, 'synthetic-pass-B', 1)
    originals = {p: p.read_bytes() for folder in (a, b) for p in folder.iterdir() if p.name == 'session.json' or p.name.startswith('route-journal-')}
    provider = ControlledFrames()
    first = update_atlas(a, root / 'maps', frame_provider=provider, allow_synthetic=True)
    second = update_atlas(b, root / 'maps', frame_provider=provider, allow_synthetic=True)
    if first['status'] != 'complete' or first['promoted_static_voxels'] != 0:
        raise AssertionError(('Single visit promoted persistent structure', first))
    if second['status'] != 'complete' or second['revisits']['confirmed_keyframes'] < 3 or second['promoted_static_voxels'] < 100:
        raise AssertionError(('Repeat pass did not recognize/promote structure', second))
    store = AtlasStore(root / 'maps' / 'atlas-v2')
    revision, count = store.current_revision(), len(list(store.keyframes()))
    again = update_atlas(b, root / 'maps', frame_provider=provider, allow_synthetic=True)
    if not again.get('already_processed') or store.current_revision() != revision or len(list(store.keyframes())) != count:
        raise AssertionError('Reprocessing changed evidence or double-counted a visit')
    if any(p.read_bytes() != data for p, data in originals.items()):
        raise AssertionError('Original recording evidence was modified')
    for folder, result in ((a, first), (b, second)):
        write_json(folder / 'map-review.json', result)
        summarize(folder)
        if 'Synthetic demonstration' not in (folder / 'review.html').read_text(encoding='utf-8'):
            raise AssertionError('Review did not identify synthetic evidence')
    result = {'status': 'passed', 'scope': 'Synthetic mapping, GPS journal, report and idempotence; not physical sensor or road validation',
        'confirmed_revisit_keyframes': second['revisits']['confirmed_keyframes'],
        'persistent_voxels': second['promoted_static_voxels'], 'keyframes': count,
        'gps_state': 'coarse_trace', 'original_evidence_unchanged': True,
        'repeat_import_unchanged': True, 'review': str(b / 'review.html')}
    write_json(root / 'MAPPING-SELFTEST.json', result)
    return result


def run(output=None):
    if output:
        root = Path(output).resolve()
        root.mkdir(parents=True, exist_ok=False)
        return check(root)
    with tempfile.TemporaryDirectory(prefix='odyssey-mapping-selftest-') as temporary:
        result = check(Path(temporary))
        result.pop('review', None)
        return result
