"""Causal, bounded independent corridor, opt-in after physical commissioning.

The centerline uses the same origin-speed/yaw approximation as existing ego
compensation. The measured reference lever arm is NOT silently treated as zero:
its omitted displacement is included in the envelope and can shorten coverage.
Error envelopes are empirical engineering bounds, not probabilistic guarantees.
"""
import math
import numpy as np
from commissioning import finite, validate_profile

# Only these reasons belong exclusively to the OPTIONAL model/camera path.
# Unknown reasons always remain blocking. In particular CAN, liveCalibration,
# gyro, clock, mount, timing, processing and lidar health are never removed.
MODEL_PATH_REASONS = frozenset((
    'predicted path missing or stale', 'invalid path geometry',
    'path lateral uncertainty missing or invalid', 'insufficient confident path horizon',
    'insufficient path geometry', 'low model confidence',
    'model confidence missing or invalid', 'camera frame missing or stale',
    'excessive or unknown camera frame drops'))


def _fresh(stamp, at_ns, age_s=.25):
    return type(stamp) is int and type(at_ns) is int and stamp > 0 and at_ns > 0 and 0 <= at_ns - stamp <= age_s * 1e9


def prepare_context(context, config, quality=None):
    """Enrich a context copy; disabled/ineligible profiles preserve legacy gates.

    Required normalized input additions: speed_source_ns, steering_source_ns,
    steering_valid, steering_angle_deg, steering_rate_deg_s, motion_source_ns.
    Steering rate must come from actual observed angles, not a schema's default
    zero. Source receipt selection remains the responsibility of Telemetry.
    """
    out = dict(context)
    check = validate_profile(config)
    diagnostic = dict(check, active=False, coverage='unavailable', reasons=list(check['reasons']))
    out['independent_corridor'] = diagnostic
    out['motion_verified'] = False
    if not check['eligible']:
        return out
    profile = config['fcw_commissioning']
    motion, coverage = profile['motion'], profile['corridor']
    quality = quality if isinstance(quality, dict) else {}
    reasons = list(context.get('reasons', []))
    blockers = [r for r in reasons if r not in MODEL_PATH_REASONS]
    if type(context.get('ready')) is not bool or (not context['ready'] and not reasons):
        blockers.append('original context invalid without an explained model-only failure')
    at = context.get('at_ns')
    if context.get('motion_ready') is not True:
        blockers.append('independent corridor motion not ready')
    for field in ('speed_source_ns', 'motion_source_ns', 'steering_source_ns'):
        from telemetry_freshness import source_fresh
        if not source_fresh(context, field):
            blockers.append('independent corridor ' + field + ' missing, stale or future')
    for field in ('speed_mps', 'yaw_rate_rps', 'acceleration_mps2', 'steering_angle_deg', 'steering_rate_deg_s'):
        if not finite(context.get(field)):
            blockers.append('independent corridor ' + field + ' invalid or unknown')
    if context.get('steering_valid') is not True:
        blockers.append('independent corridor steering evidence unavailable')
    cb = context.get('clock_bound_s')
    if not finite(cb) or not 0 <= cb <= .04:
        blockers.append('comma clock not bounded within 40 ms')
    for key, low, high, reason in (
        ('completeness', .98, 1., 'incomplete lidar frame'),
        ('arrival_age_s', 0., .2, 'lidar frame stale'),
        ('timing_bound_s', 0., .05, 'lidar/comma time alignment unverified')):
        value = quality.get(key)
        if not finite(value) or not low <= value <= high:
            blockers.append(reason)
    if quality.get('timestamp_valid') is not True:
        blockers.append('lidar time invalid')
    if quality.get('processing_overrun'):
        blockers.append('inference exceeded 150 ms')
    if blockers:
        diagnostic['reasons'] = list(dict.fromkeys(blockers))
        return out
    speed = context['speed_mps'] - motion['speed_bias_mps']
    yaw = context['yaw_rate_rps'] - motion['yaw_bias_rps']
    if not 0 < speed <= motion['max_verified_speed_mps']:
        blockers.append('speed outside commissioned corridor envelope')
    if abs(yaw) > motion['max_verified_yaw_rate_rps']:
        blockers.append('turn rate outside commissioned corridor envelope')
    if abs(context['steering_angle_deg']) > 720 or abs(context['steering_rate_deg_s']) > 1000:
        blockers.append('steering evidence implausible')
    if abs(context['acceleration_mps2']) > 12:
        blockers.append('vehicle acceleration implausible')
    if blockers:
        diagnostic['reasons'] = blockers
        return out
    # A rapidly changing steering command shortens, never lengthens, the
    # constant-yaw assumption. This cap is additional to measured holdout bounds.
    rate = abs(context['steering_rate_deg_s'])
    horizon = min(coverage['max_verified_horizon_s'], 5. / (1. + rate / 15.))
    acceleration = min(2., max(-6., context['acceleration_mps2']))
    if acceleration < 0:
        horizon = min(horizon, speed / -acceleration)
    times = np.arange(0., horizon + 1e-9, .05)
    if len(times) == 0 or times[-1] < horizon - 1e-9:
        times = np.r_[times, horizon]
    # Same longitudinal projection as threat(): path length includes acceleration
    # so a 5s label does not secretly become a shorter horizon under acceleration.
    distance = speed * times + .5 * acceleration * times ** 2
    angle = yaw / speed * distance
    if abs(yaw) < 1e-6:
        path = np.column_stack((distance, np.zeros_like(times)))
    else:
        path = np.column_stack((speed / yaw * np.sin(angle), speed / yaw * (1. - np.cos(angle))))
    ref = np.asarray(motion['reference_point_flu_m'], dtype=float)
    ref_error = motion['reference_offset_error_m']
    speed_error = motion['speed_error_bound_mps']
    yaw_error = motion['yaw_error_bound_rps']
    time_error = max(quality['timing_bound_s'], profile['timing']['error_bound_s'])
    # Bound missing lever-arm compensation in the EXISTING tracker and this
    # centerline; use its norm so direction cannot accidentally understate it.
    lever_angle_bound = (abs(yaw) + yaw_error) / speed * distance
    lever_error = 2. * np.abs(np.sin(lever_angle_bound / 2.)) * (float(np.linalg.norm(ref)) + ref_error)
    peak_speed = np.maximum(speed, speed + acceleration * times)
    lateral = (coverage['holdout_lateral_error_m'] + ref_error + lever_error +
               .5 * peak_speed * yaw_error * times ** 2 +
               .5 * speed_error * abs(yaw) * times ** 2 +
               peak_speed * abs(yaw) * time_error * times)
    longitudinal = (coverage['holdout_longitudinal_error_m'] + ref_error + lever_error +
                    speed_error * times + peak_speed * time_error)
    valid = ((lateral <= .5) & (longitudinal <= 2.) & (np.abs(angle) <= .35) &
             (speed + acceleration * times <= motion['max_verified_speed_mps']))
    bad = np.flatnonzero(~valid)
    end = int(bad[0]) if len(bad) else len(times)
    path, times, lateral, longitudinal = path[:end], times[:end], lateral[:end], longitudinal[:end]
    if len(path) < 6 or path[-1, 0] < 10.:
        diagnostic['reasons'] = ['independent corridor uncertainty leaves less than 10 m usable path']
        return out
    supported = float(times[-1])
    legacy = {k: context.get(k) for k in ('ready', 'reasons', 'path_uncertainty', 'path_horizon_m')}
    removed = [r for r in reasons if r in MODEL_PATH_REASONS]
    diagnostic.update(active=True, eligible=True, reasons=[], supported_horizon_s=supported,
                      supported_distance_m=float(np.linalg.norm(path[-1])),
                      supported_forward_m=float(path[-1, 0]),
                      coverage='four_second_corridor' if supported >= 4.5 else 'limited',
                      removed_model_only_reasons=removed,
                      acquisition_coverage_qualified=False,
                      reference_point_compensation='bounded approximation; no rear-axle speed assumption',
                      uncertainty_kind='empirical absolute error envelope',
                      current_four_second_target_m=speed * 4.,
                      warning_target_achieved=False)
    # A usable geometric horizon is not a measured warning-distance result.
    out.update(ready=True, reasons=[], path=path, path_times_s=times,
               path_lateral_error_m=lateral, path_longitudinal_error_m=longitudinal,
               path_horizon_m=float(path[-1, 0]), corridor_horizon_s=supported,
               warning_horizon_s=min(4.5, supported), speed_mps=speed, yaw_rate_rps=yaw,
               path_uncertainty='bounded independently commissioned motion corridor',
               legacy_model_context=legacy, motion_verified=True,
               calibration_epoch=config['mount_epoch'], coordinate_frame='camera_ground_flu',
               ego_vx_vy=[speed, -yaw * float(ref[0])],
               ego_velocity_error_bound_mps=[speed_error, abs(float(ref[0])) * yaw_error + abs(yaw) * ref_error + yaw_error * ref_error],
               yaw_rate_error_bound_rps=yaw_error)
    if context.get('ready') is True and not context.get('reasons'):
        # A ready model path may anticipate a bend before current yaw changes.
        # Replacing it with a straight kinematic path could lose a baseline
        # warning. Preserve ALL baseline path/ego fields, including the existing
        # warning threshold; independent motion stays supplementary evidence.
        for key in ('ready', 'reasons', 'path', 'speed_mps', 'yaw_rate_rps',
                    'path_uncertainty', 'path_horizon_m', 'path_times_s',
                    'path_lateral_error_m', 'path_longitudinal_error_m',
                    'corridor_horizon_s', 'warning_horizon_s'):
            if key in context:
                out[key] = context[key]
            else:
                out.pop(key, None)
        diagnostic.update(active=False, coverage='legacy_path',
                          selection='legacy_ready_path_preserved', independent_path_available=True,
                          legacy_warning_threshold_s=3.5)
        # Fusion propagates using the unchanged legacy gyro value. Its error
        # bound therefore includes the independently measured correction which
        # was intentionally not applied to the baseline tracker.
        out['yaw_rate_error_bound_rps'] = yaw_error + abs(motion['yaw_bias_rps'])
    return out
