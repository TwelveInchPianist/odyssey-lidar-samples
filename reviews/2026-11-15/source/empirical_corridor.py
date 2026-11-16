"""Short, uncertainty-limited FCW paths from accepted ordinary-drive evidence.

The caller must verify the applied artifact at session startup. This module
checks its numeric contract again without disk I/O and never certifies a surveyed
vehicle frame, four-second coverage, or new detector thresholds.
"""
import math
import numpy as np
from corridor import MODEL_PATH_REASONS
from empirical_calibration import (validate, finite, MAX_HORIZON_S, MAX_TIMING_S,
                                    MAX_LATERAL_M, MAX_LONGITUDINAL_M)


def _fresh(value, at):
    return type(value) is int and type(at) is int and 0 < value <= at and at - value <= 250_000_000


def _limits(profile):
    limits = profile.get('limits')
    expected = {'max_horizon_s': MAX_HORIZON_S, 'max_timing_s': MAX_TIMING_S,
                'max_lateral_m': MAX_LATERAL_M, 'max_longitudinal_m': MAX_LONGITUDINAL_M}
    return (isinstance(limits, dict) and set(limits) == set(expected) and
            all(finite(limits[k]) and 0 < limits[k] <= expected[k] for k in expected))


def _bounded_model_path(context, times):
    """Keep the original model curve and kinematics, without extrapolating it."""
    try:
        original = np.asarray(context['path'], dtype=float)
        if (original.ndim != 2 or original.shape[1] != 2 or len(original) < 6 or
                not np.isfinite(original).all() or np.any(np.diff(original[:, 0]) <= 0)):
            return None
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(original, axis=0), axis=1))]
        speed = context['speed_mps']; acceleration = min(2., max(-6., context['acceleration_mps2']))
        elapsed = np.minimum(times, -speed / acceleration) if acceleration < -.1 else times
        distance = np.interp(0., original[:, 0], arc) + speed * elapsed + .5 * acceleration * elapsed ** 2
        if np.any(distance < arc[0]) or np.any(distance > arc[-1]): return None
        path = np.column_stack((np.interp(distance, arc, original[:, 0]), np.interp(distance, arc, original[:, 1])))
        if np.linalg.norm(path[0]) >= .5 or np.any(np.diff(path[:, 0]) <= 0): return None
        return path
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def prepare_context(context, profile, quality=None):
    if profile is None:
        return dict(context)  # No empirical authority: legacy behavior is exact.
    out = dict(context); quality = quality if isinstance(quality, dict) else {}
    try:
        check = validate(profile)
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        check = {'eligible': False, 'reasons': ['Malformed empirical calibration profile']}
    diagnostic = {'active': False, 'eligible': False, 'empirical': True,
                  'method': 'sensor_derived_empirical', 'coverage': 'unavailable',
                  'surveyed_ground_truth': False, 'physical_warning_target_validated': False,
                  'warning_target_achieved': False, 'reasons': list(check['reasons'])}
    out['independent_corridor'] = diagnostic
    out['empirical_profile_validated'] = False
    out['motion_verified'] = False
    def reject(reasons):
        diagnostic['reasons'] = list(dict.fromkeys(reasons))
        out['ready'] = False
        out['reasons'] = list(dict.fromkeys(list(context.get('reasons', [])) + diagnostic['reasons']))
        return out
    if not check['eligible']:
        return reject(check['reasons'])
    out['empirical_requires_observed_ground']=profile.get('requires_observed_ground') is True
    out['empirical_geometry_basis']=profile.get('geometry_basis')
    if not _limits(profile):
        return reject(['Unknown or unsupported empirical capability limits'])
    limits = profile['limits']; m = profile['motion']; g = profile['geometry']; t = profile['timing']; c = profile['corridor']
    blockers = [r for r in context.get('reasons', []) if r not in MODEL_PATH_REASONS]
    if type(context.get('ready')) is not bool or (not context['ready'] and not context.get('reasons')):
        blockers.append('Original context invalid without an explained model-path failure')
    if context.get('motion_ready') is not True:
        blockers.append('Calibrated vehicle motion unavailable')
    at = context.get('at_ns')
    for name in ('speed_source_ns', 'motion_source_ns', 'steering_source_ns'):
        from telemetry_freshness import source_fresh
        if not source_fresh(context, name):
            blockers.append(name + ' missing, stale, future or reversed')
    for name in ('speed_mps', 'yaw_rate_rps', 'acceleration_mps2', 'steering_angle_deg', 'steering_rate_deg_s'):
        if not finite(context.get(name)):
            blockers.append(name + ' is unknown or invalid')
    if context.get('steering_valid') is not True:
        blockers.append('Measured steering-change evidence unavailable')
    clock = context.get('clock_bound_s')
    if not finite(clock) or not 0 <= clock <= .04:
        blockers.append('Comma clock not bounded within40ms')
    for name, low, high, message in (
        ('completeness', .98, 1., 'Incomplete lidar frame'),
        ('arrival_age_s', 0., .2, 'Lidar frame stale'),
        ('timing_bound_s', 0., limits['max_timing_s'], 'Lidar/comma timing unavailable')):
        value = quality.get(name)
        if not finite(value) or not low <= value <= high:
            blockers.append(message)
    if quality.get('timestamp_valid') is not True:
        blockers.append('Lidar timestamp invalid')
    if quality.get('processing_overrun'):
        blockers.append('Inference deadline exceeded')
    for name in ('speed_error_bound_mps', 'yaw_error_bound_rps', 'lateral_velocity_error_bound_mps'):
        if not finite(m.get(name)) or m[name] <= 0:
            blockers.append('Unknown empirical ' + name)
    from empirical_kinematics import lever, path as kinematic_path
    try:forward_lever=lever(m)
    except ValueError:blockers.append('Empirical lateral motion model unavailable');forward_lever=0.
    if blockers:
        return reject(blockers)
    speed = context['speed_mps'] - m['speed_bias_mps']; yaw = context['yaw_rate_rps'] - m['yaw_bias_rps']
    minimum_speed = m.get('min_observed_speed_mps', 1.)
    if not finite(minimum_speed) or not max(1., minimum_speed) <= speed <= m['max_verified_speed_mps'] + 1e-9:
        blockers.append('Speed outside the recorded empirical coverage')
    if abs(yaw) > m['max_verified_yaw_rate_rps'] + 1e-9:
        blockers.append('Turn rate outside the recorded empirical coverage')
    if abs(context['acceleration_mps2']) > 12 or abs(context['steering_angle_deg']) > 720 or abs(context['steering_rate_deg_s']) > 1000:
        blockers.append('Vehicle dynamics implausible')
    fitting_clocks = [window.get('clock_bound_s') for window in t.get('fit_windows', []) + t.get('holdout_windows', [])
                      if isinstance(window, dict) and finite(window.get('clock_bound_s'))]
    known_clock = max(fitting_clocks) if fitting_clocks else 0.
    actual_timing = max(quality['timing_bound_s'], t['error_bound_s'] + max(0., clock - known_clock))
    if actual_timing > limits['max_timing_s']:
        blockers.append('Current clock uncertainty exceeds the limited empirical timing envelope')
    if blockers:
        return reject(blockers)
    model_ready = context.get('ready') is True and not context.get('reasons')
    dynamics_horizon = min(limits['max_horizon_s'], 2. / (1. + abs(context['steering_rate_deg_s']) / 30.))
    # Match the actual predictor tested by fit_motion. Acceleration changes
    # shorten validity; they do not silently introduce an unvalidated model.
    dynamics_horizon = min(dynamics_horizon, 2. / (1. + abs(context['acceleration_mps2']) / 3.))
    candidates = []
    for row in c['horizons']:
        if not isinstance(row, dict) or row.get('supported') is not True:
            continue
        if (not all(finite(row.get(k)) for k in ('horizon_s', 'lateral_error_m', 'longitudinal_error_m')) or
                not 0 < row['horizon_s'] <= dynamics_horizon or min(row['lateral_error_m'], row['longitudinal_error_m']) < 0 or
                type(row.get('samples')) is not int or row['samples'] < 6 or
                type(row.get('independent_holdout_windows')) is not int or row['independent_holdout_windows'] < 1):
            continue
        horizon = row['horizon_s']; times = np.arange(0., horizon + 1e-9, .05)
        if times[-1] < horizon - 1e-9: times = np.r_[times, horizon]
        distance = speed * times
        angle = yaw / speed * distance
        if np.max(abs(angle)) > .35:
            continue
        reference_path=kinematic_path(speed,yaw,times,forward_lever)
        path = _bounded_model_path(context, times) if model_ready else reference_path
        if path is None: continue
        tangent = np.gradient(path, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-9)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        model_difference = path - reference_path
        along_difference = np.abs(np.sum(model_difference * tangent, axis=1))
        cross_difference = np.abs(np.sum(model_difference * normal, axis=1))
        bx, by = g['translation_error_bound_m'][:2]; angular = g['yaw_error_bound_rad']
        # Keep common frame offsets ONCE, separated by axis. The measured
        # relative trajectory bounds already include motion residuals; adding
        # those same speed/yaw errors again per scan would double count them.
        peak_speed = (np.maximum(speed, context['speed_mps'] + min(2., max(-6., context['acceleration_mps2'])) * times)
                      if model_ready else speed)
        span = np.maximum(distance, np.linalg.norm(path - path[0], axis=1))
        # The original model curve has not been validated by the constant-motion
        # holdout table. Include its difference, and rotate the ONE static-frame
        # offset rectangle into its tangent/normal, rather than borrowing trust.
        lateral = (np.abs(normal[:, 0]) * bx + np.abs(normal[:, 1]) * by +
                   span * abs(math.sin(angular)) + row['lateral_error_m'] + cross_difference +
                   peak_speed * np.abs(tangent[:, 1]) * actual_timing)
        longitudinal = (np.abs(tangent[:, 0]) * bx + np.abs(tangent[:, 1]) * by +
                        span * (1 - math.cos(angular)) + row['longitudinal_error_m'] + along_difference +
                        peak_speed * actual_timing)
        if (np.max(lateral) > limits['max_lateral_m'] or np.max(longitudinal) > limits['max_longitudinal_m'] or
                len(path) < 6 or path[-1, 0] < 3.):
            continue
        candidates.append((horizon, path, times, lateral, longitudinal, row))
    if not candidates:
        return reject(['No short horizon fits current speed, steering, alignment and timing uncertainty; collect richer ordinary-drive evidence or wait for supported motion.'])
    horizon, path, times, lateral, longitudinal, selected = max(candidates, key=lambda item: item[0])
    diagnostic.update(active=True, eligible=True, reasons=[], coverage='limited', supported_horizon_s=horizon,
                      supported_forward_m=float(path[-1, 0]), supported_distance_m=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
                      actual_timing_bound_s=actual_timing, selected_holdout_horizon=selected,
                      alignment_offset_applied_once=True, four_second_coverage=False,
                      selection='bounded_ready_model_path' if model_ready else 'bounded_empirical_motion_path',
                      model_difference_included_in_uncertainty=model_ready,
                      removed_model_only_reasons=[r for r in context.get('reasons', []) if r in MODEL_PATH_REASONS],
                      message='Short empirical warning horizon only; sensor-derived checks are not surveyed ground truth.')
    out.update(ready=True, reasons=[], speed_mps=speed, yaw_rate_rps=yaw, acceleration_mps2=0., path=path,
               empirical_observed_acceleration_mps2=context['acceleration_mps2'],
               path_times_s=times, path_lateral_error_m=lateral, path_longitudinal_error_m=longitudinal,
               path_horizon_m=float(path[-1, 0]), corridor_horizon_s=horizon, warning_horizon_s=horizon,
               coordinate_frame='camera_ground_flu', calibration_epoch=profile['mount_epoch'],
               empirical_profile_validated=True, empirical_timing_bound_s=actual_timing,
               empirical_profile_id=profile['id'], lidar_time_adjustment_s=profile['lidar_time_adjustment_s'],
               motion_verified=True, ego_vx_vy=[speed, forward_lever*yaw],
               ego_velocity_error_bound_mps=[m['speed_error_bound_mps'], m['lateral_velocity_error_bound_mps']],
               yaw_rate_error_bound_rps=m['yaw_error_bound_rps'],
               path_uncertainty='held-out empirical relative motion plus one-time frame/timing uncertainty',
               legacy_model_context={key: context.get(key) for key in ('ready', 'reasons', 'path_uncertainty', 'path_horizon_m')})
    if model_ready:
        # Empirical geometry/timing is this session's ONLY calibration authority.
        # Keep the model's core kinematics, but never restore its unbounded path
        # or legacy warning threshold beyond the accepted empirical envelope.
        for key in ('speed_mps', 'yaw_rate_rps', 'acceleration_mps2'):
            out[key] = context[key]
        out['yaw_rate_error_bound_rps'] += abs(m['yaw_bias_rps'])
    return out
