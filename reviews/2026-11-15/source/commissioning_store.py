"""Read-only commissioning review and explicit, separately applied proposals.

Uploaded JSON is evidence data, never code or a path to execute/read. The caller
owns persistence and must recheck receipt source identities before explicit Apply.
Nothing here writes to settings, recordings or the proposed artifact directory.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

from commissioning import (FRAME, finite, mount_fingerprint, profile_template,
                           validate_profile, _mount_valid)

MAX_BUNDLE_BYTES = 2 * 1024 * 1024
MAX_ARTIFACT_BYTES = 256 * 1024
MAX_ARTIFACTS = 32
MEASUREMENT_FIELDS = frozenset(('xyz', 'rpy_deg', 'width_m', 'vehicle_height_m',
                                'bumper_x_m', 'reference_point_flu_m', 'reference_offset_error_m'))
_STAGES = ('calibration', 'motion', 'timing', 'corridor')
_PROVENANCE = frozenset(('evidence_kind', 'fit_sessions', 'holdout_sessions'))
_SENSOR_FLAGS = frozenset(('extrinsics_verified', 'timing_verified', 'uncertainty_verified',
                           'semantics_verified', 'calibration_epoch', 'evidence_id'))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _json(raw):
    return json.loads(raw, object_pairs_hook=_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON number')))


def _reject_synthetic(value):
    if isinstance(value, dict):
        if value.get('synthetic_fixture') or value.get('evidence_kind') == 'synthetic':
            raise ValueError('Synthetic fixtures cannot commission a physical installation')
        for item in value.values():
            _reject_synthetic(item)
    elif isinstance(value, list):
        for item in value:
            _reject_synthetic(item)


def _basename(value):
    if (not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,120}\.json', value) or
            value.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
                *('COM' + str(i) for i in range(1, 10)), *('LPT' + str(i) for i in range(1, 10))}):
        raise ValueError('Evidence names must be plain JSON basenames, not paths')
    return value


def _no_link(path):
    p = Path(path)
    if p.exists() or p.is_symlink():
        st = p.lstat()
        if p.is_symlink() or bool(getattr(st, 'st_file_attributes', 0) & 0x400):
            raise ValueError('Linked/reparse paths are not commissioning evidence')


def _read_source(path, files, limit=None):
    p = Path(path); _no_link(p)
    before = p.stat()
    if not p.is_file() or (limit is not None and before.st_size > limit):
        raise ValueError('Source file missing or exceeds supported size: ' + p.name)
    digest = hashlib.sha256(); data = [] if limit is not None else None
    with p.open('rb') as f:
        while block := f.read(1024 * 1024):
            digest.update(block)
            if data is not None:
                data.append(block)
    after = p.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Source changed during commissioning verification')
    record = {'path': str(p.resolve()), 'size': after.st_size, 'mtime_ns': after.st_mtime_ns,
              'sha256': digest.hexdigest()}
    files[record['path']] = record
    return b''.join(data) if data is not None else record['sha256']


def save_measurements(config, values):
    """Save geometry as unverified measurements and invalidate prior trust."""
    if not isinstance(values, dict) or set(values) != MEASUREMENT_FIELDS:
        raise ValueError('Supply exactly: ' + ', '.join(sorted(MEASUREMENT_FIELDS)))
    mount = {key: deepcopy(values[key]) for key in ('xyz', 'rpy_deg', 'width_m', 'vehicle_height_m', 'bumper_x_m')}
    if not _mount_valid(mount):
        raise ValueError('Mount/vehicle measurements are missing, nonfinite or outside supported ranges')
    ref, error = values['reference_point_flu_m'], values['reference_offset_error_m']
    if (not isinstance(ref, list) or len(ref) != 2 or not all(finite(v) for v in ref) or
            abs(ref[0]) > 6 or abs(ref[1]) > 2 or not finite(error) or not 0 < error <= .3):
        raise ValueError('Supply a measured reference point and positive error bound up to0.30m')
    result = deepcopy(config)
    result['mount_epoch'] = uuid.uuid4().hex
    result['mount'] = dict(mount, measured=False, alignment_verified=False)
    result['vehicle_reference'] = {'coordinate_frame': FRAME, 'reference_point_flu_m': deepcopy(ref),
                                   'reference_offset_error_m': error}
    result.update(timing_verified=False, lidar_timing_bound_s=None)
    result['fcw_commissioning'] = profile_template(result)
    from object_fusion import default_fusion_calibration
    result['sensor_fusion'] = default_fusion_calibration(result['mount_epoch'])
    result.pop('commissioning_import_receipt', None)
    return result


def _geometry(mount):
    mount = mount if isinstance(mount, dict) else {}
    return {k: mount.get(k) for k in ('xyz', 'rpy_deg', 'width_m', 'vehicle_height_m', 'bumper_x_m')}


def _locate_sessions(record_root, wanted):
    root = Path(record_root).resolve(); found = {}
    if not root.is_dir():
        raise ValueError('Recording folder is unavailable')
    folders = list(root.iterdir())
    if len(folders) > 10000:
        raise ValueError('Recording index is too large for this import; use a scoped recording root')
    for folder in folders:
        try:
            _no_link(folder)
            if not folder.is_dir():
                continue
            source = folder / 'session.json'; _no_link(source)
            if not source.is_file() or source.stat().st_size > 1024 * 1024:
                continue
            cfg = _json(source.read_bytes())
        except (OSError, ValueError):
            continue
        sid = cfg.get('session_id') if isinstance(cfg, dict) else None
        if isinstance(sid, str) and sid in wanted:
            if sid in found:
                raise ValueError('Duplicate copies share source session identity: ' + sid)
            found[sid] = folder
    if set(found) != set(wanted):
        raise ValueError('Missing original completed source sessions: ' + ', '.join(sorted(set(wanted) - set(found))))
    return found


def _session(folder, config, files):
    cfg = _json(_read_source(folder / 'session.json', files, 1024 * 1024))
    _reject_synthetic(cfg)
    if cfg.get('capture_mode') != 'live' or cfg.get('mount_epoch') != config.get('mount_epoch'):
        raise ValueError('Evidence must come from a live session in the current mounting epoch')
    if _geometry(cfg.get('mount', {})) != _geometry(config['mount']):
        raise ValueError('Source session mount geometry differs from current measurements')
    final = _json(_read_source(folder / 'final-state.json', files, 8 * 1024 * 1024))
    if not isinstance(final, dict) or final.get('mode', 'live') != 'live':
        raise ValueError('Source session is not a completed live recording')
    index = _json(_read_source(folder / 'raw-capture-index.json', files, 8 * 1024 * 1024))
    segments = index.get('segments') if isinstance(index, dict) else None
    if not isinstance(segments, list) or not segments:
        raise ValueError('Source session has no completed raw capture index')
    intervals, pcaps, names = [], [], set()
    from m1_protocol import read_pcap, valid_msop
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError('Malformed raw capture segment')
        name = segment.get('file'); first, last = segment.get('first_receipt_ns'), segment.get('last_receipt_ns')
        if (not isinstance(name, str) or not re.fullmatch(r'lidar-[0-9]+\.pcap', name) or name in names or
                type(first) is not int or type(last) is not int or not 0 < first <= last):
            raise ValueError('Malformed raw capture file identity or clock interval')
        names.add(name); path = folder / name
        if not path.is_file() or path.stat().st_size <= 24:
            raise ValueError('Source raw capture is empty or missing')
        if segment.get('bytes') is not None and segment['bytes'] != path.stat().st_size:
            raise ValueError('Source raw capture size does not match completed index')
        digest = _read_source(path, files)
        found_packet = False
        for number, (_, source, port, payload) in enumerate(read_pcap(path)):
            if valid_msop(payload) and source[0] == cfg.get('sensor_ip') and port == cfg.get('msop_port'):
                found_packet = True; break
            if number >= 100:
                break
        if not found_packet:
            raise ValueError('Capture does not contain an identifiable original-M1 packet')
        intervals.append((first, last)); pcaps.append(digest)
    return {'session_id': cfg['session_id'], 'intervals': intervals,
            'pcap_digests': set(pcaps), 'capture_sha256': _digest(_canonical(sorted(pcaps))),
            'folder': str(folder.resolve())}


def _number(value, message):
    if not finite(value):
        raise ValueError(message)
    return value


def _vector(value, n, message):
    if not isinstance(value, list) or len(value) != n or not all(finite(v) for v in value):
        raise ValueError(message)
    return value


def _near(a, b):
    return math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-9)


def _observations(report, stage, session, parameters):
    part = report['stages'][stage]
    if (not isinstance(part, dict) or part.get('parameters') != parameters or
            part.get('accepted') is not True):
        raise ValueError(stage + ' report parameters do not exactly support the proposed profile')
    rows = part.get('observations')
    minimum = {'calibration': 3 if report['role'] == 'fit' else 1,
               'motion': 100, 'timing': 100, 'corridor': 20}.get(stage, 20)
    if not isinstance(rows, list) or len(rows) < minimum or len(rows) > 2000:
        raise ValueError(stage + ' requires sufficient recorded reference observations')
    expected_reference = {'calibration': 'surveyed_targets', 'motion': 'independent_motion_reference',
                          'timing': 'independent_clock_reference', 'corridor': 'surveyed_trajectory'}
    if part.get('reference_source') != expected_reference.get(stage, 'surveyed_sensor_reference'):
        raise ValueError(stage + ' requires an explicit independent reference source')
    prior = None
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Malformed reference observation')
        at = row.get('at_ns')
        if (type(at) is not int or (prior is not None and at <= prior) or
                not any(start <= at <= end for start, end in session['intervals'])):
            raise ValueError('Reference timestamps must be ordered and inside the actual capture')
        prior = at
    if stage in ('motion', 'timing') and rows[-1]['at_ns'] - rows[0]['at_ns'] < 15_000_000_000:
        raise ValueError(stage + ' reference observations must cover at least15seconds')
    return rows


def _verify_stage(report, stage, session, params):
    rows = _observations(report, stage, session, params)
    if stage == 'calibration':
        for row in rows:
            observed = _vector(row.get('measured_xy_m'), 2, 'Missing measured target coordinates')
            reference = _vector(row.get('reference_xy_m'), 2, 'Missing surveyed target coordinates')
            error = math.dist(observed, reference)
            bound = .15 if report['role'] == 'fit' else params['holdout_target_max_error_m']
            if error > bound + 1e-9:
                raise ValueError('Target residual exceeds the declared calibration bound')
        if report['role'] == 'fit':
            import numpy as np
            points = np.asarray([row['reference_xy_m'] for row in rows])
            if np.linalg.svd(points - points.mean(0), compute_uv=False)[-1] < .4:
                raise ValueError('Fitting targets must span longitudinal and lateral directions')
    elif stage == 'motion':
        speed_errors, yaw_errors, speeds, yaws = [], [], [], []
        for row in rows:
            obs = [_number(row.get(key), 'Missing physical motion reference values') for key in
                   ('speed_observed_mps', 'speed_reference_mps', 'yaw_observed_rps', 'yaw_reference_rps')]
            speed_errors.append(obs[0] - obs[1]); yaw_errors.append(obs[2] - obs[3])
            speeds.append(obs[1]); yaws.append(abs(obs[3]))
        if report['role'] == 'fit' and (not _near(sum(speed_errors) / len(rows), params['speed_bias_mps']) or
                                        not _near(sum(yaw_errors) / len(rows), params['yaw_bias_rps'])):
            raise ValueError('Declared motion biases do not match independent fitting measurements')
        if (max(abs(v - params['speed_bias_mps']) for v in speed_errors) > params['speed_error_bound_mps'] or
                max(abs(v - params['yaw_bias_rps']) for v in yaw_errors) > params['yaw_error_bound_rps']):
            raise ValueError('Motion residual exceeds declared bounds')
        if report['role'] == 'holdout' and (max(speeds) + 1e-9 < params['max_verified_speed_mps'] or
                                            max(yaws) + 1e-9 < params['max_verified_yaw_rate_rps']):
            raise ValueError('Claimed motion envelope exceeds physical holdout coverage')
    elif stage == 'timing':
        offsets = []
        for row in rows:
            offset = _number(row.get('independent_offset_s'), 'Missing independent time offset')
            error = _number(row.get('reference_error_bound_s'), 'Missing independent timing uncertainty')
            if error < 0 or abs(offset - params['lidar_time_adjustment_s']) + error > params['error_bound_s'] + 1e-9:
                raise ValueError('Timing residual exceeds declared bound')
            offsets.append(offset)
        if report['role'] == 'fit' and not _near(sum(offsets) / len(rows), params['lidar_time_adjustment_s']):
            raise ValueError('Timing adjustment does not match independent fitting measurements')
    elif stage == 'corridor':
        horizons = []
        for row in rows:
            predicted = _vector(row.get('predicted_xy_m'), 2, 'Missing predicted position')
            reference = _vector(row.get('reference_xy_m'), 2, 'Missing surveyed trajectory position')
            h = _number(row.get('horizon_s'), 'Missing observed corridor horizon')
            if not 0 < h <= 5 or not any(start <= row['at_ns'] and row['at_ns'] + h * 1e9 <= end
                                          for start, end in session['intervals']):
                raise ValueError('Corridor lookahead extends beyond the recorded interval')
            if (abs(predicted[0] - reference[0]) > params['holdout_longitudinal_error_m'] + 1e-9 or
                    abs(predicted[1] - reference[1]) > params['holdout_lateral_error_m'] + 1e-9):
                raise ValueError('Corridor residual exceeds the declared bound')
            horizons.append(h)
        if report['role'] == 'holdout' and max(horizons) < params['max_verified_horizon_s']:
            raise ValueError('Claimed corridor horizon exceeds observed physical holdout')
    else:
        # Sensor reports bind transformed target observations to surveyed targets
        # and empirical timing/velocity references, never to another fused lead.
        for row in rows:
            measured = _vector(row.get('measured_xy_m'), 2, 'Missing measured sensor position')
            reference = _vector(row.get('reference_xy_m'), 2, 'Missing surveyed sensor target')
            bounds = _vector(params.get('position_error_bound_m'), 2, 'Missing sensor position error bounds')
            if any(abs(a - b) > c + 1e-9 for a, b, c in zip(measured, reference, bounds)):
                raise ValueError('Sensor position residual exceeds declared bound')
            if stage in ('radar', 'camera'):
                delay = _number(row.get('independent_delay_s'), 'Missing independently measured sensor delay')
                if abs(delay - params['delay_s']) > params['delay_error_bound_s'] + 1e-9:
                    raise ValueError('Sensor delay residual exceeds declared bound')
            if stage in ('radar', 'lidar'):
                n = 1 if stage == 'radar' else 2
                observed = _vector(row.get('observed_velocity_mps'), n, 'Missing observed sensor velocity')
                actual = _vector(row.get('reference_velocity_mps'), n, 'Missing independent velocity reference')
                bounds = [params['velocity_error_bound_mps']] if n == 1 else params['velocity_error_bound_mps']
                if any(abs(a - b) > c + 1e-9 for a, b, c in zip(observed, actual, bounds)):
                    raise ValueError('Sensor velocity residual exceeds declared bound')


def import_bundle(config, record_root, bundle):
    """Verify immutable sources and return an unapplied config plus write receipt.

    Malformed/fabricated/out-of-scope evidence raises ValueError. A well-formed but
    incomplete draft returns eligible=False with zero artifact_writes. File reads
    may take time because source PCAP bytes are hashed; run in the existing job
    system, never in the HTTP request handler.
    """
    if not isinstance(bundle, dict) or set(bundle) - {'profile', 'sensor_fusion', 'evidence'}:
        raise ValueError('Invalid commissioning bundle fields')
    if len(_canonical(bundle)) > MAX_BUNDLE_BYTES:
        raise ValueError('Commissioning bundle exceeds2MiB')
    _reject_synthetic(bundle)
    profile = deepcopy(bundle.get('profile'))
    if not isinstance(profile, dict):
        raise ValueError('Bundle requires a commissioning profile')
    receipt = {'schema_version': 1, 'eligible': False, 'applied': False, 'requires_explicit_apply': True,
               'config_before_sha256': _digest(_canonical(config)), 'artifact_writes': [], 'source_file_stats': []}
    artifacts = bundle.get('evidence', [])
    if not isinstance(artifacts, list) or len(artifacts) > MAX_ARTIFACTS:
        raise ValueError('At most32JSON evidence artifacts are supported')
    contents, reports = {}, {}
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {'path', 'content'}:
            raise ValueError('Each artifact requires path basename and JSON content string')
        name = _basename(artifact['path']); content = artifact['content']
        if name in contents or not isinstance(content, str) or len(content.encode('utf-8')) > MAX_ARTIFACT_BYTES:
            raise ValueError('Duplicate, invalid or oversized evidence artifact')
        report = _json(content)
        if (not isinstance(report, dict) or report.get('schema_version') != 1 or
                report.get('report_type') != 'fcw_physical_commissioning' or
                report.get('evidence_kind') != 'physical' or report.get('role') not in ('fit', 'holdout') or
                report.get('mount_epoch') != config.get('mount_epoch') or
                not isinstance(report.get('source_session_id'), str) or not report['source_session_id'] or
                not isinstance(report.get('stages'), dict) or not report['stages'] or
                set(report['stages']) - set(_STAGES + ('lidar', 'radar', 'camera'))):
            raise ValueError('Artifact is not a scoped physical commissioning report')
        _reject_synthetic(report)
        contents[name], reports[name] = content, report
    # Template/draft values remain non-executable. Never invent missing proof.
    tentative = deepcopy(config); tentative['fcw_commissioning'] = profile
    if not artifacts or any(not isinstance(profile.get(s), dict) or not profile[s].get('fit_sessions') or
                            not profile[s].get('holdout_sessions') for s in _STAGES):
        tentative['fcw_commissioning']['enabled'] = False
        receipt['validation'] = validate_profile(tentative)
        receipt['reasons'] = receipt['validation']['reasons']
        return tentative, receipt
    if profile.get('mount_epoch') != config.get('mount_epoch') or profile.get('mount_fingerprint') != mount_fingerprint(config.get('mount')):
        raise ValueError('Profile mounting epoch or geometry changed')
    manifest = profile.get('evidence_files')
    if not isinstance(manifest, list) or len(manifest) != len(contents):
        raise ValueError('Profile evidence manifest does not exactly match attached artifacts')
    names = set()
    for item in manifest:
        if not isinstance(item, dict):
            raise ValueError('Invalid evidence manifest')
        name = _basename(item.get('path'))
        if name not in contents or name in names:
            raise ValueError('Evidence manifest contains missing or duplicate artifacts')
        names.add(name); report = reports[name]
        if (not isinstance(item.get('sha256'), str) or item['sha256'].lower() != _digest(contents[name].encode('utf-8')) or
                item.get('session') != report['source_session_id'] or item.get('role') != report['role']):
            raise ValueError('Artifact hash or source identity does not match manifest')
    # Check the proposed capability contract before touching large recordings.
    # These flags are LOCAL CANDIDATE VALUES only; no draft may return them as
    # trusted configuration before all physical report checks pass.
    candidate = deepcopy(config)
    candidate['mount']['measured'] = True; candidate['mount']['alignment_verified'] = True
    candidate['timing_verified'] = True
    candidate['lidar_timing_bound_s'] = profile['timing'].get('error_bound_s')
    candidate_profile = deepcopy(profile)
    candidate_profile['mount_fingerprint'] = mount_fingerprint(candidate['mount'])
    candidate['fcw_commissioning'] = candidate_profile
    preliminary = validate_profile(candidate)
    if not preliminary['eligible']:
        tentative['fcw_commissioning']['enabled'] = False
        receipt.update(validation=preliminary, reasons=preliminary['reasons'])
        return tentative, receipt
    roles = {role: {r['source_session_id'] for r in reports.values() if r['role'] == role} for role in ('fit', 'holdout')}
    if roles['fit'] & roles['holdout']:
        raise ValueError('Physical source sessions cannot be both fitting and holdout')
    sources = _locate_sessions(record_root, roles['fit'] | roles['holdout'])
    files = {}; sessions = {sid: _session(folder, config, files) for sid, folder in sources.items()}
    fit_bytes = set().union(*(sessions[s]['pcap_digests'] for s in roles['fit']))
    held_bytes = set().union(*(sessions[s]['pcap_digests'] for s in roles['holdout']))
    if fit_bytes & held_bytes:
        raise ValueError('Copied or overlapping raw captures cannot provide independent holdout evidence')
    for report in reports.values():
        if report.get('source_capture_sha256') != sessions[report['source_session_id']]['capture_sha256']:
            raise ValueError('Report is not bound to the actual source capture bytes')
    # Only these proof-derived fields may make the candidate verified.
    proposed = deepcopy(config); proposed['mount']['measured'] = True; proposed['mount']['alignment_verified'] = True
    proposed['timing_verified'] = True
    proposed['lidar_timing_bound_s'] = profile['timing'].get('error_bound_s')
    adjustment = profile['timing'].get('lidar_time_adjustment_s', config.get('lidar_time_adjustment_s', 0.))
    if not finite(adjustment) or abs(adjustment) > 2:
        raise ValueError('Timing adjustment is missing or implausible')
    proposed['lidar_time_adjustment_s'] = adjustment
    for stage in _STAGES:
        params = {k: v for k, v in profile[stage].items() if k not in _PROVENANCE}
        if stage == 'timing':
            params['lidar_time_adjustment_s'] = adjustment
        for role, key in (('fit', 'fit_sessions'), ('holdout', 'holdout_sessions')):
            for sid in profile[stage][key]:
                matching = [r for r in reports.values() if r['source_session_id'] == sid and r['role'] == role and stage in r['stages']]
                if len(matching) != 1:
                    raise ValueError('Each stage/session/role requires exactly one scoped report')
                _verify_stage(matching[0], stage, sessions[sid], params)
    reference = config.get('vehicle_reference', {})
    if (reference.get('coordinate_frame') != FRAME or
            reference.get('reference_point_flu_m') != profile['motion'].get('reference_point_flu_m') or
            reference.get('reference_offset_error_m') != profile['motion'].get('reference_offset_error_m')):
        raise ValueError('Motion profile does not match independently entered vehicle reference measurements')
    profile['mount_fingerprint'] = mount_fingerprint(proposed['mount'])
    proposed['fcw_commissioning'] = profile
    validation = validate_profile(proposed)
    if not validation['eligible']:
        tentative['fcw_commissioning']['enabled'] = False
        receipt.update(validation=validation, reasons=validation['reasons'], source_file_stats=list(files.values()))
        return tentative, receipt
    if 'sensor_fusion' in bundle:
        proposed['sensor_fusion'] = _fusion_bundle(bundle['sensor_fusion'], profile, reports, sessions)
    else:
        # A newly accepted corridor report must not legitimize pre-existing,
        # manually inserted optional-sensor verification flags.
        from object_fusion import default_fusion_calibration
        proposed['sensor_fusion'] = default_fusion_calibration(profile['mount_epoch'])
    import_id = uuid.uuid4().hex
    destination = Path(record_root).resolve().parent / 'calibration' / 'imports' / import_id
    for parent in (destination.parent.parent, destination.parent, destination):
        _no_link(parent)
    for item in profile['evidence_files']:
        name = item['path']; target = destination / name
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError('Evidence destination escaped calibration import directory')
        item['path'] = str(target)
        receipt['artifact_writes'].append({'path': str(target), 'content': contents[name],
                                           'sha256': _digest(contents[name].encode('utf-8'))})
    # Detect completed sources changing while another capture is being verified.
    for item in files.values():
        stat = Path(item['path']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item['size'], item['mtime_ns']):
            raise ValueError('Source changed during commissioning verification')
    receipt.update(eligible=True, validation=validation, reasons=[], source_file_stats=list(files.values()),
                   import_id=import_id, source_sessions=sorted(sessions), physical_evidence_reviewed=True,
                   limitation='Imported independent reference measurements are checked; this is not an FCW recall or70mph qualification.')
    proposed['commissioning_import_receipt'] = {key: deepcopy(receipt[key]) for key in
        ('schema_version', 'import_id', 'source_sessions', 'source_file_stats', 'physical_evidence_reviewed', 'limitation')}
    proposed['commissioning_import_receipt']['trusted_config_sha256'] = _trusted_fingerprint(proposed)
    receipt['proposed_config_sha256'] = _digest(_canonical(proposed))
    return proposed, receipt


def _trusted_fingerprint(config):
    selected = {key: config.get(key) for key in ('mount', 'mount_epoch', 'vehicle_reference',
                 'timing_verified', 'lidar_timing_bound_s', 'lidar_time_adjustment_s',
                 'fcw_commissioning', 'sensor_fusion')}
    proof = config.get('commissioning_import_receipt', {})
    selected['proof'] = {key: proof.get(key) for key in ('schema_version', 'import_id',
                             'source_sessions', 'source_file_stats', 'physical_evidence_reviewed')}
    return _digest(_canonical(selected))


def _check_source_stats(records, record_root):
    root = Path(record_root).resolve(); reasons = []
    if not isinstance(records, list) or not records:
        return ['Imported source identity receipt is missing']
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get('path'), str):
            reasons.append('Source receipt contains invalid path'); continue
        path = Path(record['path'])
        try:
            if not path.is_absolute() or not path.resolve().is_relative_to(root) or len(path.resolve().relative_to(root).parts) != 2:
                raise ValueError('Source path is outside recording root')
            _no_link(path.parent); _no_link(path)
            if path.name not in ('session.json', 'final-state.json', 'raw-capture-index.json') and not re.fullmatch(r'lidar-[0-9]+\.pcap', path.name):
                raise ValueError('Source path is not a verified recording artifact')
            stat = path.stat()
            if stat.st_size != record.get('size') or stat.st_mtime_ns != record.get('mtime_ns'):
                raise ValueError('Source identity changed: ' + path.name)
            if path.suffix == '.json' and _digest(path.read_bytes()) != record.get('sha256'):
                raise ValueError('Source metadata bytes changed: ' + path.name)
        except (OSError, ValueError) as error:
            reasons.append(str(error))
    return reasons


def validate_import_apply(current_config, proposed_config, receipt, record_root):
    """Quick final Apply guard after asynchronous full-byte verification."""
    reasons = []
    if not isinstance(receipt, dict) or receipt.get('eligible') is not True:
        return {'valid': False, 'reasons': ['Proposal has not passed physical evidence verification']}
    if _digest(_canonical(current_config)) != receipt.get('config_before_sha256'):
        reasons.append('Settings changed since evidence verification; review a new proposal')
    if _digest(_canonical(proposed_config)) != receipt.get('proposed_config_sha256'):
        reasons.append('Proposed settings changed after verification')
    proposed_proof = proposed_config.get('commissioning_import_receipt', {})
    if receipt.get('source_file_stats') != proposed_proof.get('source_file_stats'):
        reasons.append('Source verification receipt changed after review')
    reasons.extend(_check_source_stats(receipt.get('source_file_stats'), record_root))
    expected = Path(record_root).resolve().parent / 'calibration' / 'imports' / str(receipt.get('import_id'))
    for item in receipt.get('artifact_writes', []):
        try:
            if (not isinstance(item, dict) or not isinstance(item.get('content'), str) or
                    _digest(item['content'].encode('utf-8')) != item.get('sha256')):
                raise ValueError('Proposed artifact contents changed')
            path = Path(item['path'])
            if not path.is_absolute() or path.parent != expected or _basename(path.name) != path.name:
                raise ValueError('Proposed artifact destination changed')
            for parent in (expected.parent.parent, expected.parent, expected, path):
                _no_link(parent)
            if path.exists() and (not path.is_file() or _digest(path.read_bytes()) != item['sha256']):
                raise ValueError('Proposed artifact destination contains different bytes; do not overwrite')
        except (OSError, ValueError, KeyError, TypeError) as error:
            reasons.append(str(error))
    if not receipt.get('artifact_writes'):
        reasons.append('Proposal has no verified physical artifacts')
    expected_artifacts = {(item.get('path'), item.get('sha256')) for item in
                          proposed_config.get('fcw_commissioning', {}).get('evidence_files', [])}
    actual_artifacts = {(item.get('path'), item.get('sha256')) for item in receipt.get('artifact_writes', [])
                        if isinstance(item, dict)}
    if expected_artifacts != actual_artifacts:
        reasons.append('Proposed artifact write list differs from verified profile')
    return {'valid': not reasons, 'reasons': list(dict.fromkeys(reasons))}


def validate_active_profile(config, record_root):
    """Startup guard: hash small reports; retain full-source hash/stat receipt.

    PCAP content was fully hashed at import; this intentionally checks size/mtime
    on startup, not gigabytes of capture data on every session or frame. It is a
    corruption/accidental-edit guard, not a hostile administrator trust boundary.
    """
    check = validate_profile(config)
    reasons = list(check['reasons'])
    proof = config.get('commissioning_import_receipt')
    if not isinstance(proof, dict) or proof.get('physical_evidence_reviewed') is not True:
        reasons.append('No verified import receipt; manual profile flags cannot enable the capability')
    else:
        try:
            if proof.get('trusted_config_sha256') != _trusted_fingerprint(config):
                reasons.append('Commissioned settings or provenance changed after Apply')
            import_id = proof.get('import_id')
            if not isinstance(import_id, str) or not re.fullmatch(r'[0-9a-f]{32}', import_id):
                raise ValueError('Invalid physical evidence import identity')
            expected = Path(record_root).resolve().parent / 'calibration' / 'imports' / import_id
            for parent in (expected.parent.parent, expected.parent, expected):
                _no_link(parent)
            records = config.get('fcw_commissioning', {}).get('evidence_files', [])
            if not isinstance(records, list) or not 1 <= len(records) <= MAX_ARTIFACTS:
                raise ValueError('Missing persisted physical evidence artifacts')
            for record in records:
                path = Path(record['path'])
                if not path.is_absolute() or path.parent != expected:
                    raise ValueError('Active evidence path is outside its import folder')
                _basename(path.name); _no_link(path)
                if path.stat().st_size > MAX_ARTIFACT_BYTES or _digest(path.read_bytes()) != record.get('sha256'):
                    raise ValueError('Applied physical evidence bytes changed')
            reasons.extend(_check_source_stats(proof.get('source_file_stats'), record_root))
        except (OSError, ValueError, KeyError, TypeError) as error:
            reasons.append(str(error))
    return {'trusted': not reasons, 'reasons': list(dict.fromkeys(reasons)),
            'calibration_id': _trusted_fingerprint(config) if not reasons else None,
            'source_check': 'full hashes at import; capture size/mtime and small metadata hashes at startup'}


def _fusion_bundle(value, profile, reports, sessions):
    from object_fusion import validate_fusion_calibration
    if not isinstance(value, dict) or value.get('epoch') != profile['mount_epoch'] or value.get('frame') != FRAME:
        raise ValueError('Sensor calibration epoch/frame mismatch')
    result = deepcopy(value)
    for sensor in ('lidar', 'radar', 'camera'):
        if sensor not in result:
            continue
        part = result[sensor]
        if not isinstance(part, dict):
            raise ValueError('Invalid sensor calibration profile')
        params = {k: v for k, v in part.items() if k not in _SENSOR_FLAGS}
        proof = [r for r in reports.values() if sensor in r['stages']]
        if {r['role'] for r in proof} != {'fit', 'holdout'}:
            raise ValueError('Sensor verification requires its own physical fitting and holdout reports: ' + sensor)
        for report in proof:
            _verify_stage(report, sensor, sessions[report['source_session_id']], params)
        for flag in ('extrinsics_verified', 'timing_verified', 'uncertainty_verified', 'semantics_verified'):
            if sensor != 'lidar' or flag == 'uncertainty_verified':
                part[flag] = True
        part['calibration_epoch'] = profile['mount_epoch']
        part['evidence_id'] = _digest(_canonical(sorted(_digest(_canonical(r)) for r in proof)))
    check = validate_fusion_calibration(result, profile['mount_epoch'])
    if not check['valid']:
        raise ValueError('Fusion calibration structural verification failed: ' + '; '.join(check['reasons']))
    return result
