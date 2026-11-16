"""Evidence-backed commissioning for optional independent FCW capabilities.

This module validates a recorded evidence contract, not the truth of a physical
experiment. The importing application must verify evidence file hashes. Nothing
in this module turns a synthetic run, a checkbox, or a successful fit into a
physical validation. Existing releases remain usable with this feature disabled.
"""
from copy import deepcopy
import hashlib
import json
import math
import re

SCHEMA_VERSION = 1
TARGET_WARNING_S = 4.0
TARGET_SPEED_MPS = 70.0 * .44704
TARGET_RANGE_M = TARGET_WARNING_S * TARGET_SPEED_MPS
FRAME = 'camera_ground_flu'
_MOUNT_KEYS = ('xyz', 'rpy_deg', 'width_m', 'vehicle_height_m', 'bumper_x_m',
               'measured', 'alignment_verified')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def mount_fingerprint(mount):
    """Stable mount/vehicle fingerprint; remount epoch is checked separately."""
    if not isinstance(mount, dict):
        return None
    try:
        raw = json.dumps({key: mount.get(key) for key in _MOUNT_KEYS},
                         sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (ValueError, TypeError):
        return None
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def default_profile():
    return {'schema_version': SCHEMA_VERSION, 'enabled': False,
            'coordinate_frame': FRAME, 'mount_epoch': None,
            'mount_fingerprint': None, 'evidence_files': [],
            'calibration': {}, 'motion': {}, 'timing': {}, 'corridor': {}}


def profile_template(config):
    """Complete export/import shape, with unknown physical evidence left blank."""
    config = config if isinstance(config, dict) else {}
    p = default_profile()
    p.update(mount_epoch=config.get('mount_epoch'),
             mount_fingerprint=mount_fingerprint(config.get('mount')))
    for name in ('calibration', 'motion', 'timing', 'corridor'):
        p[name] = {'evidence_kind': None, 'fit_sessions': [], 'holdout_sessions': []}
    p['calibration']['holdout_target_max_error_m'] = None
    p['motion'].update({key: None for key in (
        'speed_reference', 'speed_bias_mps', 'speed_error_bound_mps',
        'yaw_bias_rps', 'yaw_error_bound_rps', 'reference_point_flu_m',
        'reference_offset_error_m', 'max_verified_speed_mps', 'max_verified_yaw_rate_rps')})
    p['timing']['error_bound_s'] = None
    p['timing']['lidar_time_adjustment_s'] = None
    p['corridor'].update(max_verified_horizon_s=None, holdout_lateral_error_m=None,
                         holdout_longitudinal_error_m=None)
    return p


def workflow_status(config):
    """User-facing collection requirements; never marks unseen evidence passed."""
    return {'validation': validate_profile(config), 'template': profile_template(config),
            'steps': [
                {'id': 'measure', 'title': 'Measure the current mounting configuration',
                 'instruction': 'Record lidar position/orientation, vehicle footprint and kinematic reference relative to the ground below the comma camera. Create a new mount epoch after any remount.'},
                {'id': 'calibration', 'title': 'Fit alignment and check separate measured targets',
                 'instruction': 'Keep fitting and holdout sessions separate; holdout target error must be at most 0.20 m. A successful fit alone is insufficient.'},
                {'id': 'motion', 'title': 'Validate speed and calibrated gyro at the vehicle origin',
                 'instruction': 'Measure bias, absolute error bounds and tested speed/turn envelopes against independent references. Do not substitute default-zero yaw or assume vEgo is rear-axle speed.'},
                {'id': 'timing', 'title': 'Verify timing on independent maneuvers',
                 'instruction': 'Use independently registered lidar motion and comma motion, with disjoint fitting and holdout sessions and no more than 50 ms demonstrated error.'},
                {'id': 'corridor', 'title': 'Validate the predicted corridor on held-out trajectories',
                 'instruction': 'Record horizon-specific lateral and longitudinal errors. A 4 s first-warning target needs extra lookahead for persistence and processing; record only the horizon actually supported.'},
                {'id': 'import', 'title': 'Import the reviewed evidence bundle',
                 'instruction': 'Attach SHA-256 references to the completed physical session evidence. Review and apply explicitly; synthetic fixtures never qualify the installation.'}],
            'note': 'Commissioning enables a capability; FCW recall, nuisance rate and audible timing require separate held-out evaluation.'}


def _stage(stage, name, evidence, reasons):
    if not isinstance(stage, dict):
        reasons.append(name + ' evidence missing'); return {}
    if stage.get('evidence_kind') != 'physical':
        reasons.append(name + ' requires physical evidence')
    fit, holdout = stage.get('fit_sessions'), stage.get('holdout_sessions')
    valid_ids = lambda ids: isinstance(ids, list) and bool(ids) and all(
        isinstance(x, str) and bool(x.strip()) for x in ids) and len(ids) == len(set(ids))
    if not valid_ids(fit) or not valid_ids(holdout):
        reasons.append(name + ' needs fitting and independent holdout sessions')
    else:
        if set(fit) & set(holdout):
            reasons.append(name + ' holdout overlaps fitting sessions')
        for role, ids in (('fit', fit), ('holdout', holdout)):
            for session in ids:
                if (session, role) not in evidence:
                    reasons.append(name + ' missing hashed ' + role + ' evidence: ' + session)
    return stage


def _bounded(stage, field, low, high, name, reasons, allow_zero=False):
    value = stage.get(field)
    if not finite(value) or not (low <= value <= high if allow_zero else low < value <= high):
        reasons.append(name + '.' + field + ' missing or outside accepted bounds')


def _mount_valid(mount):
    xyz, rpy = mount.get('xyz'), mount.get('rpy_deg')
    if any(not isinstance(v, (list, tuple)) or len(v) != 3 or not all(finite(x) for x in v)
           for v in (xyz, rpy)):
        return False
    if not all(finite(mount.get(k)) for k in ('width_m', 'vehicle_height_m', 'bumper_x_m')):
        return False
    return (1 < mount['width_m'] < 3 and 1 < mount['vehicle_height_m'] < 4 and
            0 < mount['bumper_x_m'] < 5 and .5 < xyz[2] < 4 and max(abs(x) for x in rpy) <= 30)


def validate_profile(config):
    """Return explicit capabilities without modifying configuration or evidence.

    Evidence fields are empirical bounds. They are deliberately not advertised
    as Gaussian standard deviations or as a statistical safety guarantee.
    """
    if not isinstance(config, dict):
        config = {}
    profile = config.get('fcw_commissioning')
    profile = profile if isinstance(profile, dict) else default_profile()
    reasons = []
    if profile.get('enabled') is not True:
        reasons.append('independent corridor disabled')
    if type(profile.get('schema_version')) is not int or profile['schema_version'] != SCHEMA_VERSION:
        reasons.append('unknown commissioning schema')
    if profile.get('coordinate_frame') != FRAME:
        reasons.append('commissioning coordinate frame must be camera_ground_flu')
    epoch = config.get('mount_epoch')
    if not isinstance(epoch, str) or not epoch.strip() or profile.get('mount_epoch') != epoch:
        reasons.append('mount epoch missing or changed; recommission required')
    mount = config.get('mount') if isinstance(config.get('mount'), dict) else {}
    if mount.get('measured') is not True:
        reasons.append('mount measurements required')
    if mount.get('alignment_verified') is not True:
        reasons.append('lidar/path alignment not verified')
    if not _mount_valid(mount):
        reasons.append('mount/vehicle geometry invalid')
    fingerprint = mount_fingerprint(mount)
    if not fingerprint or profile.get('mount_fingerprint') != fingerprint:
        reasons.append('mount geometry changed; recommission required')
    if config.get('timing_verified') is not True:
        reasons.append('lidar timing not verified')
    bound = config.get('lidar_timing_bound_s')
    if not finite(bound) or not 0 <= bound <= .05:
        reasons.append('lidar timing bound missing or exceeds 50 ms')
    records = profile.get('evidence_files', [])
    evidence = set()
    digests = {'fit': set(), 'holdout': set()}
    if not isinstance(records, list):
        reasons.append('evidence file manifest invalid'); records = []
    for item in records:
        if (not isinstance(item, dict) or not isinstance(item.get('session'), str) or
                not item['session'].strip() or item.get('role') not in ('fit', 'holdout') or
                not isinstance(item.get('path'), str) or not item['path'].strip() or
                not isinstance(item.get('sha256'), str) or
                not re.fullmatch(r'[0-9a-fA-F]{64}', item['sha256'])):
            reasons.append('evidence file manifest contains invalid record')
        else:
            evidence.add((item['session'], item['role']))
            digests[item['role']].add(item['sha256'].lower())
    if digests['fit'] & digests['holdout']:
        reasons.append('holdout reuses fitting evidence bytes')
    stages = {name: _stage(profile.get(name), name, evidence, reasons)
              for name in ('calibration', 'motion', 'timing', 'corridor')}
    c, m, t, p = (stages[name] for name in ('calibration', 'motion', 'timing', 'corridor'))
    _bounded(c, 'holdout_target_max_error_m', 0., .2, 'calibration', reasons, True)
    if m.get('speed_reference') != 'vehicle_origin':
        reasons.append('speed reference must be empirically verified at the vehicle origin')
    for field, limit in (('speed_bias_mps', 3.), ('yaw_bias_rps', .1)):
        if not finite(m.get(field)) or abs(m[field]) > limit:
            reasons.append('motion.' + field + ' missing or implausible')
    ref = m.get('reference_point_flu_m')
    if (not isinstance(ref, (list, tuple)) or len(ref) != 2 or
            not all(finite(x) for x in ref) or abs(ref[0]) > 6 or abs(ref[1]) > 2):
        reasons.append('measured kinematic reference point missing or invalid')
    for field, hi in (('speed_error_bound_mps', 1.5), ('yaw_error_bound_rps', .02),
                      ('reference_offset_error_m', .3), ('max_verified_speed_mps', TARGET_SPEED_MPS),
                      ('max_verified_yaw_rate_rps', 1.5)):
        _bounded(m, field, 0., hi, 'motion', reasons)
    _bounded(t, 'error_bound_s', 0., .05, 'timing', reasons)
    _bounded(p, 'max_verified_horizon_s', 0., 5., 'corridor', reasons)
    _bounded(p, 'holdout_lateral_error_m', 0., .5, 'corridor', reasons)
    _bounded(p, 'holdout_longitudinal_error_m', 0., 2., 'corridor', reasons)
    reasons = list(dict.fromkeys(reasons))
    eligible = not reasons
    return {'schema_version': SCHEMA_VERSION, 'eligible': eligible,
            'enabled': profile.get('enabled') is True, 'reasons': reasons,
            'mount_epoch': epoch, 'mount_fingerprint': fingerprint,
            'coordinate_frame': FRAME, 'target_warning_s': TARGET_WARNING_S,
            'target_speed_mps': TARGET_SPEED_MPS, 'target_70mph_range_m': TARGET_RANGE_M,
            'max_verified_speed_mps': m.get('max_verified_speed_mps') if eligible else None,
            'max_verified_horizon_s': p.get('max_verified_horizon_s') if eligible else None,
            'evidence_scope': 'imported physical holdout evidence; not an FCW recall qualification',
            'evidence_file_hashes_require_import_verification': True}


def profile_proposal(config, evidence):
    """Bind supplied evidence to current geometry, returning an unapplied draft.

    An empty or synthetic evidence set stays ineligible. The caller must request
    explicit approval and verify every source file before saving an enabled one.
    """
    proposed = default_profile()
    if isinstance(evidence, dict):
        proposed.update(deepcopy(evidence))
    proposed.update(schema_version=SCHEMA_VERSION, coordinate_frame=FRAME,
                    mount_epoch=config.get('mount_epoch'),
                    mount_fingerprint=mount_fingerprint(config.get('mount')))
    candidate = dict(config, fcw_commissioning=proposed)
    return {'profile': proposed, 'validation': validate_profile(candidate),
            'applied': False, 'requires_explicit_apply': True}
