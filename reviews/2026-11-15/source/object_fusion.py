"""Optional, bounded sensor evidence for the existing lidar tracker.

This module never removes or changes a baseline track. The caller must evaluate
baseline warnings independently, then may add warnings from returned candidates.
Radar-only hypotheses are diagnostic; camera absence is never free space.
All numeric contributions require a verified, epoch-bound physical profile.
"""
import copy
import hashlib
import json
import math
import runtime_clock as time
import numpy as np

from radar import ERROR_FIELDS

SCHEMA_VERSION = 1
FRAME = 'camera_ground_flu'
MAX_OBJECTS = 64
MAX_AGE_S = .25
MIN_SUPPORT = 3


def _number(value):
    if type(value) not in (int, float):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, OverflowError):
        return None


def _vector(value, n, nonnegative=False):
    if not isinstance(value, (tuple, list, np.ndarray)):
        return None
    try:
        if len(value) != n:
            return None
        values = [_number(v.item() if isinstance(v, np.generic) else v) for v in value]
        if any(v is None or (nonnegative and v < 0) for v in values):
            return None
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None


def _identity(value):
    return (type(value) is int and 0 <= value < 2**63) or (isinstance(value, str) and 0 < len(value) <= 256)


def _stamp(value):
    return type(value) is int and 0 < value < 2**63


def _rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def _get(track, key, default=None):
    return track.get(key, default) if isinstance(track, dict) else getattr(track, key, default)


def _track_info(track):
    identity = _get(track, 'id')
    position = _vector(_get(track, 'position'), 2)
    velocity = _vector(_get(track, 'velocity'), 2)
    lo, hi = _get(track, 'lo'), _get(track, 'hi')
    if isinstance(track, dict):
        bounds = track.get('bounds_m', {})
        if isinstance(bounds, dict):
            lo, hi = bounds.get('lo', lo), bounds.get('hi', hi)
    lo, hi = _vector(lo, 3), _vector(hi, 3)
    if (not _identity(identity) or position is None or velocity is None or lo is None or hi is None or
            np.any(hi < lo) or np.any(np.abs(position) > 1000) or np.any(np.abs(velocity) > 150)):
        return None
    return {'id': identity, 'position': position, 'velocity': velocity,
            'half_extent': (hi[:2] - lo[:2]) / 2,
            'track': track, 'misses': _get(track, 'misses', 0),
            'ambiguous': _get(track, 'association_ambiguous', False)}


def covariance_intersection(mean, covariance, measurement, measurement_covariance, projection=None):
    """Generalized CI with an incomplete measurement and unknown correlations.

Inputs must be positive-definite conservative second-moment envelopes. No
independence or Gaussian distribution is asserted. The fixed 0.5 information
weight prevents one contribution from being counted as multiple measurements.
An incomplete scalar measurement cannot shrink unconstrained directions.
"""
    x, p = np.asarray(mean, dtype=float), np.asarray(covariance, dtype=float)
    z, r = np.atleast_1d(np.asarray(measurement, dtype=float)), np.atleast_2d(np.asarray(measurement_covariance, dtype=float))
    h = np.eye(len(x)) if projection is None else np.atleast_2d(np.asarray(projection, dtype=float))
    if (x.ndim != 1 or z.ndim != 1 or p.shape != (len(x), len(x)) or
            r.shape != (len(z), len(z)) or h.shape != (len(z), len(x)) or
            not all(np.isfinite(v).all() for v in (x, p, z, r, h)) or
            not np.allclose(p, p.T) or not np.allclose(r, r.T) or
            np.min(np.linalg.eigvalsh(p)) <= 0 or np.min(np.linalg.eigvalsh(r)) <= 0):
        raise ValueError('CI requires finite matching shapes and positive definite envelopes')
    pi, ri = np.linalg.inv(p), np.linalg.inv(r)
    weight = .5
    result_p = np.linalg.inv(weight*pi + (1-weight)*h.T@ri@h)
    result_x = result_p@(weight*pi@x + (1-weight)*h.T@ri@z)
    return result_x, result_p


def default_fusion_calibration(epoch=None):
    """No executable sensor trust is supplied by a default or estimated mount."""
    return {'epoch': epoch, 'frame': FRAME,
            'lidar': {'uncertainty_verified': False},
            'radar': {'extrinsics_verified': False, 'timing_verified': False,
                      'uncertainty_verified': False, 'semantics_verified': False},
            'camera': {'extrinsics_verified': False, 'timing_verified': False,
                       'uncertainty_verified': False, 'semantics_verified': False}}


def _profile(profile, sensor, epoch):
    if not isinstance(profile, dict):
        return None, 'sensor calibration profile missing'
    if profile.get('calibration_epoch') != epoch:
        return None, 'sensor calibration epoch mismatch'
    empirical=profile.get('evidence_basis')=='sensor_derived_empirical'
    if empirical:
        support=profile.get('empirical_support')
        if (sensor!='radar' or profile.get('surveyed_ground_truth') is not False or not isinstance(support,dict) or
                any(support.get(key) is not True for key in ('relative_geometry','relative_timing','velocity'))):
            return None,'empirical radar support is incomplete'
    else:
        for name in ('extrinsics_verified', 'timing_verified', 'uncertainty_verified', 'semantics_verified'):
            if profile.get(name) is not True:
                return None, name + ' is not verified'
    if not _identity(profile.get('evidence_id')):
        return None, 'physical validation evidence reference missing'
    origin = _vector(profile.get('origin_m'), 3)
    pos_bound = _vector(profile.get('position_error_bound_m'), 2, True)
    origin_bound = _number(profile.get('origin_error_bound_m'))
    delay = _number(profile.get('time_offset_s') if empirical else profile.get('delay_s'))
    delay_bound = _number(profile.get('time_error_bound_s') if empirical else profile.get('delay_error_bound_s'))
    if (origin is None or np.any(np.abs(origin) > 10) or pos_bound is None or np.any(pos_bound < 0 if empirical else pos_bound <= 0) or
            np.any(pos_bound > [10, 3]) or origin_bound is None or not 0 <= origin_bound <= .5 or
            delay is None or not (-1 <= delay <= 1 if empirical else 0 <= delay <= .25) or
            delay_bound is None or not 0 <= delay_bound <= (.15 if empirical else .04)):
        return None, 'sensor transform, timing or error bounds invalid'
    if empirical and (profile.get('time_basis')!='relative_to_corrected_lidar_time' or
            not _identity(profile.get('applied_profile_id')) or _number(profile.get('lidar_time_adjustment_s')) is None):
        return None,'empirical time reference is not bound to an applied profile'
    out = {**profile, 'origin': origin, 'position_bound': pos_bound,
           'origin_bound': origin_bound, 'delay': delay, 'delay_bound': delay_bound,'empirical':empirical}
    if sensor == 'radar':
        yaw = _number(profile.get('yaw_rad'))
        yaw_bound = _number(profile.get('yaw_error_bound_rad'))
        speed_bound = _number(profile.get('velocity_error_bound_mps'))
        if (profile.get('velocity_semantics') != 'longitudinal_relative' or yaw is None or abs(yaw) > math.pi or
                yaw_bound is None or not 0 <= yaw_bound <= .05 or
                speed_bound is None or not 0 < speed_bound <= 10):
            return None, 'radar longitudinal velocity semantics or bounds invalid'
        out.update(rotation=_rotation(yaw), yaw_bound=yaw_bound, speed_bound=speed_bound)
        if empirical:
            domain=profile.get('observed_domain');bias=_number(profile.get('velocity_bias_mps'))
            if not isinstance(domain,dict) or bias is None or abs(bias)>3:
                return None,'empirical signed velocity evidence/domain invalid'
            for name in ('range_m','vrel_mps','ego_speed_mps'):
                bounds=_vector(domain.get(name),2)
                if bounds is None or bounds[0]>bounds[1]:return None,'empirical observed domain invalid'
            for name,limit in (('abs_bearing_rad',1.6),('abs_yaw_rate_rps',.2)):
                bound=_number(domain.get(name))
                if bound is None or not 0<=bound<=limit:return None,'empirical observed domain invalid'
    else:
        try:
            rotation = np.asarray(profile.get('rotation_device_frd_to_vehicle_flu'), dtype=float)
        except (TypeError, ValueError, OverflowError):
            return None, 'camera transform invalid'
        # LeadDataV3 provides no Z. Require an empirically verified explicit
        # bound for that missing dimension; do not silently assume a ground point.
        z_bound = _number(profile.get('unobserved_device_z_bound_m'))
        orientation_bound = _number(profile.get('orientation_error_bound_rad'))
        std_scale = _number(profile.get('reported_std_bound_scale'))
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all() or
                not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-6) or
                not math.isclose(float(np.linalg.det(rotation)), 1., abs_tol=1e-6) or
                z_bound is None or not 0 < z_bound <= 20 or
                orientation_bound is None or not 0 <= orientation_bound <= .05 or
                std_scale is None or not 1 <= std_scale <= 10):
            return None, 'camera transform or unobserved-height/error bounds invalid'
        out.update(rotation=rotation, z_bound=z_bound, yaw_bound=orientation_bound, std_scale=std_scale)
    return out, None


def _lidar_uncertainty_supported(value):
    return isinstance(value,dict) and (value.get('uncertainty_verified') is True or
        (value.get('evidence_basis')=='sensor_derived_empirical' and value.get('empirical_uncertainty_supported') is True and
         _identity(value.get('evidence_id'))))


def validate_fusion_calibration(calibration, epoch=None):
    """Structural check only; commissioning importer must verify real evidence.

    One verified optional sensor is enough for structural validity. Each sensor
    has a separate result, allowing radar commissioning before camera evidence.
    A boolean in JSON is never proof that physical validation was performed.
    """
    out = {'valid': False, 'reasons': [], 'sensors': {}}
    if not isinstance(calibration, dict) or calibration.get('frame') != FRAME or not _identity(calibration.get('epoch')):
        out['reasons'].append('fusion calibration frame/epoch unavailable')
        return out
    current = calibration['epoch']
    if epoch is not None and current != epoch:
        out['reasons'].append('fusion calibration epoch differs from active mount')
    lidar = calibration.get('lidar')
    if not _lidar_uncertainty_supported(lidar):
        out['reasons'].append('lidar uncertainty not verified')
    else:
        position = _vector(lidar.get('position_error_bound_m'), 2, True)
        velocity = _vector(lidar.get('velocity_error_bound_mps'), 2, True)
        if (position is None or velocity is None or np.any(position <= 0) or np.any(position > [10, 3]) or
                np.any(velocity <= 0) or np.any(velocity > 20)):
            out['reasons'].append('lidar measured uncertainty bounds invalid')
    for sensor in ('radar', 'camera'):
        profile, reason = _profile(calibration.get(sensor), sensor, current)
        out['sensors'][sensor] = {'valid': profile is not None, 'reason': reason}
    if not any(v['valid'] for v in out['sensors'].values()):
        out['reasons'].append('no independently verified optional sensor profile')
    out['valid'] = not out['reasons']
    return out


class FusionEngine:
    """Temporal association plus optional radar speed / camera existence support.

    Input Track dataclasses and serialized track dictionaries are both accepted.
    Returned candidate shape matches the input and gains a ``fusion`` metadata
    field/attribute. Baseline positions, extents, identities, and inputs survive.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._pairs = {}
        self._clocks = {}
        self._last_at = None
        self._scope = None
        self._query_blocked = False

    def _clear_sensor(self, sensor):
        self._pairs = {k: v for k, v in self._pairs.items() if k[0] != sensor}

    def _source_clock(self, snapshot, sensor):
        scope = tuple(snapshot.get(k) for k in ('epoch', 'source_epoch', 'route_id'))
        if not _identity(scope[0]) or any(v is not None and not _identity(v) for v in scope[1:]):
            return 'sensor stream epoch missing or invalid'
        stamp = snapshot.get('source_ns')
        measurement = snapshot.get('measurement_source_ns', stamp)
        receipt = snapshot.get('receipt_ns')
        if not _stamp(stamp) or not _stamp(measurement) or not _stamp(receipt):
            return 'sensor timestamps missing or invalid'
        frame = snapshot.get('frame_id') if sensor == 'camera' else None
        if sensor == 'camera' and (type(frame) is not int or not 0 <= frame < 2**32 or
                                  not _stamp(snapshot.get('measurement_source_ns')) or measurement > stamp):
            return 'camera physical frame identity/time invalid'
        try:
            fingerprint = hashlib.sha256(json.dumps(snapshot.get('points'), sort_keys=True,
                                         separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        except (TypeError, ValueError, OverflowError):
            return 'sensor evidence contains invalid JSON measurements'
        prior = self._clocks.get(sensor)
        if prior is None or prior['scope'] != scope:
            self._clear_sensor(sensor)
            prior = {'scope': scope, 'stamp': stamp, 'measurement': measurement,
                     'receipt': receipt, 'fingerprint': fingerprint, 'frame': frame, 'blocked': False}
            self._clocks[sensor] = prior
        elif (stamp < prior['stamp'] or measurement < prior['measurement'] or receipt < prior['receipt'] or
              (stamp == prior['stamp'] and (fingerprint != prior['fingerprint'] or
                                           measurement != prior['measurement'] or receipt != prior['receipt']))):
            prior['blocked'] = True
        if (sensor == 'camera' and (frame < prior['frame'] or
                (frame == prior['frame'] and (measurement != prior['measurement'] or fingerprint != prior['fingerprint'])) or
                (frame > prior['frame'] and measurement <= prior['measurement']))):
            prior['blocked'] = True
        if prior['blocked']:
            return 'sensor clock/evidence discontinuity; explicit source epoch required'
        prior.update(stamp=stamp, measurement=measurement, receipt=receipt, fingerprint=fingerprint, frame=frame)
        return None

    def _observations(self, sensor, snapshot, profile, context, motion):
        if not isinstance(snapshot, dict) or snapshot.get('available') is not True or snapshot.get('valid') is not True:
            return [], 'sensor snapshot unavailable or invalid'
        expected = 'raw_oem_tracks' if sensor == 'radar' else 'camera_current_lead'
        if snapshot.get('kind') != expected or snapshot.get('service') != ('liveTracks' if sensor == 'radar' else 'modelV2'):
            return [], 'fused radarState/unsupported source cannot be counted as independent evidence'
        if snapshot.get('frame_origin') != ('front_bumper_flu' if sensor == 'radar' else 'device_frd'):
            return [], 'sensor coordinate convention unsupported'
        if snapshot.get('reasons') or (sensor == 'radar' and (not isinstance(snapshot.get('errors'), dict) or
                any(snapshot['errors'].get(k) is not False for k in ERROR_FIELDS) or
                any(v is not False for v in snapshot['errors'].values()))):
            return [], 'sensor fault or invalid evidence'
        if profile.get('source_commit') is not None and snapshot.get('source_commit') != profile['source_commit']:
            return [], 'sensor producer version differs from verified profile'
        if profile.get('empirical'):
            adjustment=_number(context.get('lidar_time_adjustment_s'))
            if (context.get('empirical_profile_id')!=profile['applied_profile_id'] or adjustment is None or
                    abs(adjustment-profile['lidar_time_adjustment_s'])>1e-9):
                return [], 'empirical radar timing does not match the active corrected lidar timeline'
        reason = self._source_clock(snapshot, sensor)
        if reason:
            return [], reason
        at = context['at_ns']
        clock_bound = _number(snapshot.get('clock_bound_s'))
        published = snapshot.get('publication_at_ns')
        stamp = published if sensor == 'radar' else snapshot.get('measurement_at_ns')
        if not _stamp(published) or not _stamp(stamp) or clock_bound is None or not 0 <= clock_bound <= .04:
            return [], 'bounded sensor clock alignment missing'
        if sensor == 'camera' and abs((published-snapshot['source_ns']) -
                                      (stamp-snapshot['measurement_source_ns'])) > 2:
            return [], 'camera publication and exposure clock transforms disagree'
        measurement_at = round(stamp + (profile['delay'] if profile.get('empirical') else -profile['delay'])*1e9)
        dt = (at-measurement_at)/1e9
        uncertainty_s = clock_bound + profile['delay_bound']
        receipt_age = (at-snapshot['receipt_ns'])/1e9
        if (dt < -uncertainty_s or dt+uncertainty_s > MAX_AGE_S or not 0 <= receipt_age <= MAX_AGE_S or
                published > at + clock_bound*1e9):
            return [], 'physical measurement stale, unreceived or in future'
        # A small future bound is not a known forward measurement: use zero age
        # and retain the whole timing interval in the position envelope.
        dt = max(0., dt)
        points = snapshot.get('points')
        if not isinstance(points, list) or len(points) > MAX_OBJECTS:
            return [], 'sensor object count invalid or excessive'
        observations, identities = [], set()
        for point in points:
            if not isinstance(point, dict):
                return [], 'malformed sensor object'
            identity = point.get('trackId') if sensor == 'radar' else point.get('lead_index')
            if not _identity(identity) or identity in identities:
                return [], 'sensor object identity missing or duplicate'
            identities.add(identity)
            if sensor == 'radar':
                if point.get('measured') is not True:
                    continue
                xy = _vector([point.get('dRel'), point.get('yRel')], 2)
                speed = _number(point.get('vRel'))
                if xy is None or speed is None or not 0 < xy[0] <= 1000 or abs(xy[1]) > 500 or abs(speed) > 150:
                    return [], 'radar position or signed longitudinal speed invalid'
                if profile.get('empirical'):
                    domain=profile['observed_domain']
                    if (not domain['range_m'][0]<=xy[0]<=domain['range_m'][1] or
                        not domain['vrel_mps'][0]<=speed<=domain['vrel_mps'][1] or
                        not domain['ego_speed_mps'][0]<=motion['velocity'][0]<=domain['ego_speed_mps'][1] or
                        abs(math.atan2(xy[1],xy[0]))>domain['abs_bearing_rad']+1e-9 or
                        abs(motion['yaw'])>domain['abs_yaw_rate_rps']+1e-9):
                        continue
                    speed-=profile['velocity_bias_mps']
                position = profile['rotation']@xy + profile['origin'][:2]
                axis = profile['rotation'][:, 0]
                sensor_velocity = motion['velocity'] + motion['yaw']*np.array([-profile['origin'][1], profile['origin'][0]])
                ground_speed_projection = speed + float(axis@sensor_velocity)
                speed_error = (profile['speed_bound'] + float(np.abs(axis)@motion['velocity_bound']) +
                               np.linalg.norm(profile['origin'][:2])*motion['yaw_bound'] +
                               abs(motion['yaw'])*profile['origin_bound'] +
                               np.linalg.norm(sensor_velocity)*profile['yaw_bound'])
                extra = np.zeros(2)
            else:
                probability = _number(point.get('probability'))
                if probability is None or not 0 <= probability <= 1:
                    return [], 'camera lead probability invalid'
                if probability < .8:
                    continue
                xy = _vector(point.get('position_device_xy_m'), 2)
                std = _vector(point.get('reported_position_std_m'), 2, True)
                if xy is None or std is None or not 0 < xy[0] <= 1000 or np.any(std <= 0):
                    return [], 'camera lead position or uncertainty invalid'
                if np.any(std > [10., 2.]):
                    continue
                rotation = profile['rotation']
                position = rotation[:2, :2]@xy + profile['origin'][:2]
                extra = np.abs(rotation[:2, 2])*profile['z_bound'] + np.abs(rotation[:2, :2])@std*profile['std_scale']
                ground_speed_projection = speed_error = axis = None
            bound = profile['position_bound'] + profile['origin_bound'] + np.linalg.norm(xy)*profile['yaw_bound'] + extra
            observations.append({'sensor': sensor, 'id': identity, 'position_then': position,
                'position_bound': bound, 'velocity_projection_mps': ground_speed_projection,
                'velocity_error_bound_mps': speed_error, 'projection_axis_then': axis,
                'measurement_at_ns': measurement_at, 'source_ns': snapshot['source_ns'],
                'measurement_source_ns': snapshot.get('measurement_source_ns', snapshot['source_ns']),
                'receipt_ns': snapshot['receipt_ns'], 'epoch': snapshot['epoch'],
                'frame_id': snapshot.get('frame_id'), 'dt': dt, 'time_error_bound_s': uncertainty_s,
                'evidence_id': profile['evidence_id']})
        return observations, None

    def update(self, lidar_tracks, context, radar_snapshot=None, camera_snapshot=None, calibration=None, phase=3,
               deadline_ns=None):
        baseline = copy.deepcopy(list(lidar_tracks)) if isinstance(lidar_tracks, (list, tuple)) else []
        audit = {'frame': FRAME, 'calibration_epoch': None, 'sensors': {}, 'associations': [],
                 'unassociated': [], 'reasons': [], 'baseline_preserved': True,
                 'radar_only_alerts_allowed': False, 'camera_clearance_allowed': False,
                 'uncertainty_kind': 'conservative_second_moment_envelopes_not_probability',
                 'sources_not_assumed_independent': ['lidar', 'liveTracks', 'modelV2', 'ego_motion'],
                 'budget_exhausted': False}
        result = {'schema_version': SCHEMA_VERSION, 'baseline_tracks': baseline,
                  'fused_track_candidates': [], 'audit': audit}
        def reject(reason):
            audit['reasons'].append(reason)
            self._pairs.clear()
            return result
        def expired():
            return deadline_ns is not None and time.monotonic_ns() >= deadline_ns
        def out_of_budget():
            audit['budget_exhausted'] = True
            result['fused_track_candidates'] = []
            return reject('optional fusion budget exhausted; baseline retained')
        if deadline_ns is not None and not _stamp(deadline_ns):
            return reject('optional fusion deadline invalid')
        if expired():
            return out_of_budget()
        if type(phase) is not int or not 1 <= phase <= 4:
            return reject('unsupported fusion milestone')
        if not isinstance(context, dict) or not _stamp(context.get('at_ns')):
            return reject('fusion query timestamp invalid')
        if not isinstance(calibration, dict) or calibration.get('frame') != FRAME or not _identity(calibration.get('epoch')):
            return reject('fusion calibration frame/epoch unavailable')
        epoch = calibration['epoch']
        audit['calibration_epoch'] = epoch
        if context.get('calibration_epoch') != epoch:
            return reject('fusion calibration does not match active mounting epoch')
        try:
            scope = hashlib.sha256(json.dumps(calibration, sort_keys=True, allow_nan=False).encode()).hexdigest()
        except (TypeError, ValueError, OverflowError):
            return reject('fusion calibration contains invalid values')
        if scope != self._scope:
            self.reset()
            self._scope = scope
        at = context['at_ns']
        if self._last_at is not None and at <= self._last_at:
            self._query_blocked = True
            return reject('fusion query clock regressed or duplicated; explicit reset required')
        if self._query_blocked:
            return reject('fusion query clock discontinuity; explicit reset required')
        self._last_at = at
        motion = {'velocity': _vector(context.get('ego_vx_vy'), 2),
                  'velocity_bound': _vector(context.get('ego_velocity_error_bound_mps'), 2, True),
                  'yaw': _number(context.get('yaw_rate_rps')),
                  'yaw_bound': _number(context.get('yaw_rate_error_bound_rps'))}
        if (context.get('motion_verified') is not True or any(v is None for v in motion.values()) or
                np.any(np.abs(motion['velocity']) > 60) or np.any(motion['velocity_bound'] > 5) or
                abs(motion['yaw']) > 2 or not 0 <= motion['yaw_bound'] <= .1):
            return reject('verified origin ego motion and error bounds required for time alignment')
        lidar_profile = calibration.get('lidar', {})
        if not _lidar_uncertainty_supported(lidar_profile):
            return reject('lidar uncertainty has not been measured and verified')
        lidar_position_bound = _vector(lidar_profile.get('position_error_bound_m'), 2, True)
        lidar_velocity_bound = _vector(lidar_profile.get('velocity_error_bound_mps'), 2, True)
        if (lidar_position_bound is None or lidar_velocity_bound is None or
                np.any(lidar_position_bound <= 0) or np.any(lidar_position_bound > [10, 3]) or
                np.any(lidar_velocity_bound <= 0) or np.any(lidar_velocity_bound > 20)):
            return reject('lidar measured uncertainty bounds invalid')
        infos = [v for track in baseline[:MAX_OBJECTS] if (v := _track_info(track)) is not None]
        counts = {}
        for track in infos:
            counts[track['id']] = counts.get(track['id'], 0)+1
        infos = [t for t in infos if counts[t['id']] == 1 and not t['misses'] and not t['ambiguous']]
        positions = np.asarray([t['position'] for t in infos], dtype=float).reshape((-1, 2))
        velocities = np.asarray([t['velocity'] for t in infos], dtype=float).reshape((-1, 2))
        extents = np.asarray([t['half_extent'] for t in infos], dtype=float).reshape((-1, 2))
        eligible = {}
        current_pairs = set()
        for sensor, snapshot in (('radar', radar_snapshot), ('camera', camera_snapshot)):
            if expired():
                return out_of_budget()
            if (sensor == 'radar' and phase < 2) or (sensor == 'camera' and phase < 3):
                audit['sensors'][sensor] = {'eligible': False, 'reason': 'milestone disabled'}
                self._clear_sensor(sensor)
                continue
            profile, reason = _profile(calibration.get(sensor), sensor, epoch)
            observations = []
            if reason is None:
                observations, reason = self._observations(sensor, snapshot, profile, context, motion)
            if reason:
                audit['sensors'][sensor] = {'eligible': False, 'reason': reason}
                self._clear_sensor(sensor)
                continue
            audit['sensors'][sensor] = {'eligible': True, 'objects': len(observations),
                                        'evidence_id': profile['evidence_id'],
                                        'evidence_basis':profile.get('evidence_basis','separately_verified_profile'),
                                        'observed_domain_restriction':profile.get('empirical',False)}
            if profile.get('empirical'):
                audit['common_reference']=copy.deepcopy(calibration.get('common_reference'))
            compatible = {}
            for oi, obs in enumerate(observations):
                if expired():
                    return out_of_budget()
                compatible[oi] = []
                dt, angle = obs['dt'], motion['yaw']*obs['dt']
                if abs(motion['yaw']) < 1e-8:
                    movement = motion['velocity']*dt
                else:
                    w = motion['yaw']
                    movement = np.array([[math.sin(angle)/w, -(1-math.cos(angle))/w],
                                         [(1-math.cos(angle))/w, math.sin(angle)/w]])@motion['velocity']
                rotate = _rotation(-angle)
                # Vectorize the bounded lidar side of association so a dense
                # scene does not spend its entire optional budget in Python.
                position_now = rotate@(obs['position_then']-movement) + velocities*dt
                # Unknown correlation uses summed worst-case bounds, not an
                # independence-based square-root sum of squares.
                bounds = (np.abs(rotate)@obs['position_bound'] + lidar_position_bound +
                          lidar_velocity_bound*dt + motion['velocity_bound']*dt +
                          (np.abs(velocities)+np.abs(motion['velocity']))*obs['time_error_bound_s'] +
                          np.linalg.norm(obs['position_then'])*motion['yaw_bound']*dt)
                residual = np.abs(position_now-positions)
                # Large uncertainty cannot make an arbitrary object match.
                gates = np.minimum([6., 2.], bounds + np.minimum(extents, [3., 1.]))
                for ti in np.flatnonzero(np.all(residual <= gates, axis=1)):
                    compatible[oi].append((int(ti), position_now[ti], bounds[ti], rotate))
            lidar_options = {}
            for oi, options in compatible.items():
                for ti, *_ in options:
                    lidar_options.setdefault(ti, []).append(oi)
            for oi, obs in enumerate(observations):
                if expired():
                    return out_of_budget()
                options = compatible[oi]
                ambiguous = len(options) != 1 or (options and len(lidar_options[options[0][0]]) != 1)
                key = (sensor, obs['id'])
                if ambiguous:
                    self._pairs.pop(key, None)
                    audit['unassociated'].append({'sensor': sensor, 'source_id': obs['id'],
                        'reason': 'ambiguous association' if options else 'no compatible lidar track',
                        'diagnostic_only': True})
                    continue
                ti, position_now, bound, rotate = options[0]
                track = infos[ti]
                previous = self._pairs.get(key)
                if (previous is None or previous['lidar_id'] != track['id'] or
                        at-previous['seen_at_ns'] > 300_000_000):
                    previous = {'lidar_id': track['id'], 'count': 0, 'observation_id': None,
                                'first_measurement_ns': obs['measurement_source_ns'], 'seen_at_ns': at}
                observation_id = ((obs['frame_id'], obs['measurement_source_ns']) if sensor == 'camera' else
                                  obs['measurement_source_ns'])
                if previous['observation_id'] != observation_id:
                    previous['count'] += 1
                    previous['observation_id'] = observation_id
                previous['seen_at_ns'] = at
                self._pairs[key] = previous
                current_pairs.add(key)
                metadata = {'sensor': sensor, 'source_id': obs['id'], 'lidar_id': track['id'],
                    'source_ns': obs['source_ns'], 'measurement_at_ns': obs['measurement_at_ns'],
                    'measurement_source_ns': obs['measurement_source_ns'],
                    'receipt_ns': obs['receipt_ns'], 'source_epoch': obs['epoch'],
                    'camera_frame_id': obs['frame_id'], 'calibration_epoch': epoch,
                    'position_at_query_m': position_now.tolist(), 'position_error_bound_m': bound.tolist(),
                    'time_error_bound_s': obs['time_error_bound_s'], 'support': previous['count'],
                    'evidence_id': obs['evidence_id'],
                    'evidence_basis':profile.get('evidence_basis','separately_verified_profile'),
                    'support_span_s': (obs['measurement_source_ns']-previous['first_measurement_ns'])/1e9,
                    'accepted': previous['count'] >= MIN_SUPPORT and
                                obs['measurement_source_ns']-previous['first_measurement_ns'] >= 100_000_000}
                audit['associations'].append(metadata)
                if not metadata['accepted']:
                    continue
                entry = eligible.setdefault(track['id'], {'track': track, 'sources': [], 'velocity': track['velocity'].copy(),
                                                          'velocity_envelope': 2*np.diag(lidar_velocity_bound**2)})
                if sensor == 'radar':
                    axis = rotate@obs['projection_axis_then']
                    speed = obs['velocity_projection_mps']
                    speed_error = obs['velocity_error_bound_mps'] + abs(motion['yaw'])*obs['dt']*lidar_velocity_bound.max()
                    innovation = abs(float(axis@track['velocity'])-speed)
                    allowed = min(20., float(np.abs(axis)@lidar_velocity_bound)+speed_error)
                    if innovation > allowed:
                        metadata['accepted'] = False
                        metadata['reason'] = 'radar/lidar velocity conflict'
                        continue
                    entry['velocity'], entry['velocity_envelope'] = covariance_intersection(
                        track['velocity'], entry['velocity_envelope'], [speed], [[speed_error**2]], axis[None, :])
                    metadata.update(velocity_semantics='longitudinal_relative_with_measured_origin_motion',
                                    ground_velocity_projection_mps=speed,
                                    velocity_error_bound_mps=speed_error,
                                    velocity_projection_axis=axis.tolist(), ci_information_weight=.5)
                else:
                    metadata.update(contribution='position/existence_support_only', velocity_used=False,
                                    negative_evidence_used=False, class_claimed=False)
                entry['sources'].append(copy.deepcopy(metadata))
        self._pairs = {k: v for k, v in self._pairs.items() if k in current_pairs}
        for entry in eligible.values():
            if expired():
                return out_of_budget()
            if not entry['sources']:
                continue
            candidate = copy.deepcopy(entry['track']['track'])
            metadata = {'schema_version': SCHEMA_VERSION, 'frame': FRAME, 'calibration_epoch': epoch,
                        'sources': entry['sources'], 'baseline_position_and_extent_preserved': True,
                        'baseline_warning_must_be_preserved': True, 'clearance_evidence': False,
                        'velocity_second_moment_envelope': entry['velocity_envelope'].tolist(),
                        'correlation_handling': 'covariance_intersection_unknown_cross_source_correlation'}
            if isinstance(candidate, dict):
                candidate['velocity'] = entry['velocity'].tolist()
                candidate['fusion'] = metadata
            else:
                candidate.velocity = entry['velocity'].copy()
                candidate.fusion = metadata
            result['fused_track_candidates'].append(candidate)
        audit['associations'] = audit['associations'][:MAX_OBJECTS]
        audit['unassociated'] = audit['unassociated'][:MAX_OBJECTS]
        return result
