"""Sensor-derived motion/timing evidence; never surveyed ground truth.

Reference registration uses only un-deskewed lidar geometry. Comma samples are
introduced later for fitting and disjoint-window validation, not to seed ICP.
Empirical errors describe observed coverage, not guaranteed statistical bounds.
"""
import math
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

VERSION = 'lidar-independent-motion-v2-case-diagnostics'
HORIZONS = (.5, 1., 1.5, 2.)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _rot(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def _velocity(translation, angle, dt):
    if abs(angle) < 1e-6:
        return np.asarray(translation) / dt
    a, b = math.sin(angle) / angle, (1. - math.cos(angle)) / angle
    return np.linalg.solve(np.array([[a, -b], [b, a]]), translation) / dt


def _increment(speed, yaw, dt):
    a = yaw * dt
    if abs(yaw) < 1e-7:
        return np.array([speed * dt, 0.]), a
    return np.array([speed / yaw * math.sin(a), speed / yaw * (1. - math.cos(a))]), a


class IndependentMotion:
    def __init__(self, max_points=1600):
        self.max_points = min(3000, max(200, int(max_points)))
        self.max_target_points = min(16000, self.max_points * 8)
        self.previous = None
        self.previous_ns = None

    def reset(self):
        self.previous = None; self.previous_ns = None

    def _features(self, points):
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] < 3:
            return np.empty((0, 2))
        p = p[np.isfinite(p[:, :3]).all(axis=1) & (p[:, 0] > 2) & (p[:, 0] < 95) &
              (np.abs(p[:, 1]) < 38) & (np.abs(p[:, 2]) < 12), :3]
        if len(p) < 60:
            return np.empty((0, 2))
        # Absolute height is deliberately irrelevant: roof lidar sees the road
        # near z=-1.8m. Local vertical extent isolates structure from flat road.
        cells = np.floor(p[:, :2] / 1.).astype(np.int32)
        _, inverse = np.unique(cells, axis=0, return_inverse=True)
        lo = np.full(inverse.max() + 1, np.inf); hi = -lo.copy()
        np.minimum.at(lo, inverse, p[:, 2]); np.maximum.at(hi, inverse, p[:, 2])
        p = p[(hi - lo)[inverse] >= .25]
        if not len(p):
            return np.empty((0, 2))
        _, indices = np.unique(np.floor(p[:, :2] / .2).astype(np.int32), axis=0, return_index=True)
        p = p[indices, :2]
        # Query thinning is safe, but independently thinning the target to the
        # same small count creates an artificial correspondence-distance floor
        # on real MEMS scans. Keep a bounded dense target for nearest neighbors.
        if len(p) > self.max_target_points:
            p = p[np.linspace(0, len(p) - 1, self.max_target_points).astype(int)]
        return p

    @staticmethod
    def _geometry(p):
        if len(p) < 60:
            return None
        _, counts = np.unique(np.floor(p / [8., 5.]).astype(int), axis=0, return_counts=True)
        cov = np.linalg.eigvalsh(np.cov(p.T))
        condition = float(cov[0] / max(cov[-1], 1e-9))
        if np.ptp(p[:, 0]) < 10 or np.ptp(p[:, 1]) < 4 or len(counts) < 6 or max(counts) / len(p) > .45 or condition < .01:
            return None
        # Normal-direction diversity rejects translation along a featureless
        # parallel-wall corridor, even when raw XY covariance looks healthy.
        tree = cKDTree(p); _, idx = tree.query(p, k=min(8, len(p)))
        neighborhoods = p[idx] - p[idx].mean(axis=1)[:, None, :]
        matrices = np.einsum('nki,nkj->nij', neighborhoods, neighborhoods)
        eigenvalues, eigenvectors = np.linalg.eigh(matrices)
        linear = eigenvalues[:, 0] < .3 * np.maximum(eigenvalues[:, 1], 1e-8)
        if linear.sum() >= 30:
            normals = eigenvectors[linear, :, 0]
            information = np.linalg.eigvalsh(normals.T @ normals / len(normals))
            normal_condition = float(information[0] / max(information[-1], 1e-9))
            if normal_condition < .025:
                return None
        else:
            normal_condition = condition
        return {'spatial_bins': len(counts), 'conditioning': condition,
                'normal_conditioning': normal_condition, 'largest_bin_fraction': float(max(counts) / len(p))}

    @staticmethod
    def _seeds(old, new):
        grids = []
        for p in (old, new):
            ij = np.floor((p + [0., 40.]) / .5).astype(int)
            good = (ij[:, 0] >= 0) & (ij[:, 0] < 192) & (ij[:, 1] >= 0) & (ij[:, 1] < 160)
            ij = ij[good]
            grid = np.zeros((192, 160)); grid[ij[:, 0], ij[:, 1]] = 1
            grids.append(grid)
        correlation = fftconvolve(grids[1], grids[0][::-1, ::-1], mode='full')
        center = np.array(grids[0].shape) - 1
        # Neither a vehicle speed nor comma yaw is used to select these seeds.
        region = correlation[center[0]-14:center[0]+15, center[1]-10:center[1]+11]
        maxima = region == maximum_filter(region, size=3)
        candidates = np.argwhere(maxima)
        order = sorted(candidates, key=lambda ij: -region[tuple(ij)])[:4]
        return [np.zeros(2)] + [(ij - [14, 10]) * .5 for ij in order]

    @staticmethod
    def _icp(old, new, seed):
        tree = cKDTree(new); r = np.eye(2); t = np.asarray(seed, dtype=float)
        for _ in range(22):
            moved = old @ r.T + t; distance, ids = tree.query(moved)
            cutoff = min(1.5, max(.06, float(np.quantile(distance, .7))))
            good = distance <= cutoff
            if good.sum() < 45:
                return None
            a, b = old[good], new[ids[good]]; ac, bc = a - a.mean(0), b - b.mean(0)
            u, _, vt = np.linalg.svd(ac.T @ bc); nr = vt.T @ u.T
            if np.linalg.det(nr) < 0:
                vt[-1] *= -1; nr = vt.T @ u.T
            nt = b.mean(0) - nr @ a.mean(0)
            change = np.linalg.norm(nr - r) + np.linalg.norm(nt - t)
            r, t = nr, nt
            if change < .0003:
                break
        moved = old @ r.T + t; distance, ids = tree.query(moved)
        good = distance < .35
        if good.sum() < 45:
            return None
        rms = float(np.sqrt(np.mean(distance[good] ** 2)))
        return {'r': r, 't': t, 'rms': rms, 'fraction': float(good.mean()),
                'good': good, 'residual': moved - new[ids], 'score': rms + .2 * (1 - good.mean())}

    def update(self, points, at_ns, *, provenance=None):
        base = {'version': VERSION, 'valid': False, 'reference_kind': 'lidar_scan_registration_empirical',
                'surveyed_ground_truth': False, 'independent_of_comma': True,
                'input_contract': 'un-deskewed raw lidar or a fixed lidar-only leveling transform'}
        if (provenance and (provenance.get('ego_deskewed') or provenance.get('comma_seeded') or
                            provenance.get('motion_compensated') or provenance.get('deskew_source') or
                            provenance.get('seed_source') not in (None, 'lidar_only'))):
            self.reset()
            return dict(base, independent_of_comma=False, reason='circular reference: externally seeded or ego-deskewed points')
        if type(at_ns) is not int or at_ns <= 0:
            self.reset(); return dict(base, reason='invalid lidar clock')
        try:
            current = self._features(points)
        except (TypeError, ValueError, OverflowError):
            self.reset(); return dict(base, at_ns=at_ns, reason='invalid lidar geometry')
        old, previous = self.previous, self.previous_ns
        self.previous, self.previous_ns = current, at_ns
        base.update(start_ns=previous, end_ns=at_ns, at_ns=at_ns if previous is None else (previous + at_ns) // 2)
        if previous is None:
            return dict(base, reason='waiting for a second lidar scan')
        dt = (at_ns - previous) / 1e9; base['dt_s'] = dt
        if not .035 <= dt <= .25:
            return dict(base, reason='lidar clock gap or reversal; reference segment restarted')
        geometry = self._geometry(current); previous_geometry = self._geometry(old)
        if geometry is None or previous_geometry is None:
            return dict(base, reason='insufficient static structure or poorly conditioned scene')
        query = old if len(old) <= self.max_points else old[np.linspace(0, len(old) - 1, self.max_points).astype(int)]
        fits = []
        # Translation-only starts can trade a real turn for a fictitious lateral
        # jump. Search modest rotations using lidar geometry only, then refine
        # each transform. No previous comma/vehicle prediction enters this.
        for initial_angle in np.arange(-.12, .1201, .04):
            if abs(initial_angle) > 1.2 * dt + .02:
                continue
            initial_rotation = _rot(float(initial_angle)); rotated = query @ initial_rotation.T
            seeds = self._seeds(rotated, current)
            chosen = [seeds[1]] if len(seeds) > 1 else [np.zeros(2)]
            if abs(initial_angle) < 1e-8 and np.linalg.norm(chosen[0]) > .01:
                chosen.append(np.zeros(2))
            for seed in chosen:
                fit = self._icp(rotated, current, seed)
                if fit is not None:
                    fit['r'] = fit['r'] @ initial_rotation
                    fits.append(fit)
        fits = sorted((fit for fit in fits if fit is not None), key=lambda fit: fit['score'])
        if not fits:
            return dict(base, reason='scan registration could not find a supported transform')
        best = fits[0]; rotation, translation = best['r'], best['t']
        angle = -math.atan2(rotation[1, 0], rotation[0, 0])
        ego = -rotation.T @ translation; velocity = _velocity(ego, angle, dt)
        for other in fits[1:]:
            delta_angle = abs(math.atan2(other['r'][1, 0], other['r'][0, 0]) + angle)
            if (np.linalg.norm(other['t'] - translation) > .35 or delta_angle > .008) and other['score'] - best['score'] < .025:
                return dict(base, reason='scan registration ambiguous across independent lidar-only starts')
        good = best['good']; cells = np.floor(query[good] / [8., 5.]).astype(int)
        _, inverse = np.unique(cells, axis=0, return_inverse=True)
        medians = [np.median(best['residual'][good][inverse == i], axis=0)
                   for i in range(inverse.max() + 1) if np.count_nonzero(inverse == i) >= 5]
        sector_spread = float(np.max(np.linalg.norm(medians, axis=1))) if medians else float('inf')
        if (best['rms'] > .18 or best['fraction'] < .45 or abs(angle / dt) > 1.2 or
                np.linalg.norm(velocity) > 55 or len(medians) < 6 or sector_spread > .28):
            return dict(base, reason='static-scene consensus failed; motion/dynamic dominance or insufficient overlap',
                        quality={**geometry, 'rmse_m': best['rms'], 'inlier_fraction': best['fraction'], 'sector_residual_m': sector_spread})
        effective_regions = min(12, len(medians))
        translation_error = max(.008, best['rms'] / math.sqrt(effective_regions), sector_spread * .35)
        lever = max(5., float(np.sqrt(np.mean(np.sum((query[good] - query[good].mean(0)) ** 2, axis=1)))))
        yaw_error = max(.0002, translation_error / lever)
        quality = {**geometry, 'rmse_m': best['rms'], 'inlier_fraction': best['fraction'],
                   'sector_residual_m': sector_spread, 'matched_points': int(good.sum()),
                   'query_points': len(query), 'target_points': len(current),
                   'translation_error_m': translation_error, 'yaw_error_rad': yaw_error,
                   'uncertainty_kind': 'empirical region-consistency estimate; not a guaranteed bound',
                   'static_scene_assumption': True}
        return dict(base, valid=True, reason='lidar-only relative motion available',
                    ego_increment_se2={'translation_m': ego.tolist(), 'yaw_rad': angle},
                    speed_mps=float(velocity[0]), velocity_lidar_mps=velocity.tolist(),
                    yaw_rate_rps=angle / dt, quality=quality)


def _key(row):
    return (str(row.get('source_session_id', row.get('session_id', row.get('source_id', 'stream')))),
            str(row.get('registration_window_id', row.get('clock_epoch', row.get('block_id', 'default')))))


def _body_increment(row, alignment):
    if row.get('ego_increment_se3'):
        r,t=_body_transform(row,alignment)
        from scipy.spatial.transform import Rotation
        return t[:2],float(Rotation.from_matrix(r).as_rotvec()[2])
    increment = row['ego_increment_se2']; theta = increment['yaw_rad']
    lidar = np.asarray(increment['translation_m'], dtype=float)
    rotation = _rot(alignment['yaw_rad']); origin = np.asarray(alignment['sensor_origin_camera_flu_m'][:2])
    return rotation @ lidar + origin - _rot(theta) @ origin, theta


def _body_transform(row,alignment):
    raw=row['ego_increment_se3'];level=np.eye(3);level[:2,:2]=_rot(alignment['yaw_rad'])
    rotation=level@np.asarray(raw['rotation'])@level.T;origin=np.asarray(alignment['sensor_origin_camera_flu_m'])
    return rotation,level@np.asarray(raw['translation_m'])+origin-rotation@origin


def _windows(rows):
    from selfcal_motion3d import rigid_valid
    blocks, current = [], []
    for row in rows:
        quality = row.get('quality', {}) if isinstance(row, dict) else {}
        inc = row.get('ego_increment_se2', {}) if isinstance(row, dict) else {}
        vector = inc.get('translation_m') if isinstance(inc, dict) else None
        valid = (isinstance(row, dict) and row.get('valid') is True and row.get('independent_of_comma') is True and
                 row.get('reference_kind') == 'lidar_scan_registration_empirical' and
                 type(row.get('start_ns')) is int and type(row.get('end_ns')) is int and
                 type(row.get('at_ns')) is int and row['start_ns'] <= row['at_ns'] <= row['end_ns'] and
                 _finite(row.get('dt_s')) and .035 <= row['dt_s'] <= .25 and
                 abs((row['end_ns'] - row['start_ns']) / 1e9 - row['dt_s']) < 1e-6 and
                 isinstance(vector, (list, tuple)) and len(vector) == 2 and all(_finite(v) for v in vector) and
                 _finite(inc.get('yaw_rad')) and all(_finite(row.get(k)) for k in ('speed_mps', 'yaw_rate_rps')) and
                 isinstance(quality, dict) and all(_finite(quality.get(k)) and quality[k] > 0 for k in ('translation_error_m', 'yaw_error_rad')))
        if isinstance(row,dict) and 'ego_increment_se3' in row and not rigid_valid(row['ego_increment_se3']):valid=False
        clock_valid = (isinstance(row, dict) and row.get('independent_of_comma') is True and
                       row.get('reference_kind') == 'lidar_scan_registration_empirical' and
                       type(row.get('start_ns')) is int and type(row.get('end_ns')) is int and
                       row['start_ns'] < row['end_ns'])
        if not clock_valid:
            if current: blocks.append(current)
            current = []; continue
        if current and (_key(row) != _key(current[-1][0]) or row['start_ns'] < current[-1][0]['end_ns'] or
                        row['start_ns'] - current[-1][0]['end_ns'] > 500_000_000):
            blocks.append(current); current = []
        # Discrete rate comparisons need trustworthy observed increments, not
        # every intervening increment. Keep the clock window when a scan match
        # fails, but NEVER synthesize or integrate the missing pose increment.
        current.append((row, valid))
    if current: blocks.append(current)
    windows = []
    for block_index, block in enumerate(blocks):
        start = 0
        while start < len(block):
            end = start
            while end < len(block) and block[end][0]['end_ns'] - block[start][0]['start_ns'] <= 8_000_000_000:
                end += 1
            subset = [row for row, valid in block[start:end] if valid]
            if len(subset) >= 35 and subset[-1]['end_ns'] - subset[0]['start_ns'] >= 4_000_000_000:
                windows.append({'rows': subset, 'key': _key(subset[0]), 'block': block_index,
                                'id': '|'.join(_key(subset[0])) + ':' + str(subset[0]['start_ns'])})
            start = max(end, start + 1)
    for index, window in enumerate(windows):
        window['role'] = 'holdout' if index % 3 == 2 else 'fit'
    return windows


def _comma_groups(rows):
    groups = {}; invalid = set(); repeated = 0
    for row in rows:
        if (not isinstance(row, dict) or row.get('motion_ready') is not True or
                not all(_finite(row.get(k)) for k in ('speed_mps', 'yaw_rate_rps', 'clock_bound_s')) or
                type(row.get('at_ns')) is not int or not 0 <= row['clock_bound_s'] <= .2):
            continue
        key = _key(row)
        if key in groups and groups[key]:
            prior = groups[key][-1]
            if row['at_ns'] == prior['at_ns'] and all(row.get(k) == prior.get(k) for k in
                    ('speed_mps', 'yaw_rate_rps', 'speed_source_ns', 'motion_source_ns')):
                # Re-reading an unchanged native sensor message is not a new
                # measurement and is not a clock reset. Retain its worst bound.
                prior['clock_bound_s'] = max(prior['clock_bound_s'], row['clock_bound_s'])
                repeated += 1; continue
            same_snapshot_with_native_channels = (row['at_ns'] == prior['at_ns'] and
                all(type(row.get(k)) is int and type(prior.get(k)) is int for k in ('speed_source_ns', 'motion_source_ns')))
            if row['at_ns'] < prior['at_ns'] or (row['at_ns'] == prior['at_ns'] and not same_snapshot_with_native_channels):
                invalid.add(key)
        groups.setdefault(key, []).append(dict(row))
    return {key: value for key, value in groups.items() if key not in invalid}, len(invalid), repeated


def _channel_series(comma, channel):
    """Separate native measurements from snapshots without inventing samples."""
    value_key, stamp_key, receipt_key = (
        ('speed_mps', 'speed_source_ns', 'speed_receipt_ns') if channel == 'speed' else
        ('yaw_rate_rps', 'motion_source_ns', 'motion_receipt_ns'))
    identity_key='speed_native_source_ns' if channel=='speed' else 'motion_native_source_ns'
    native_identity=any(identity_key in row for row in comma);fixed_offset=None
    if native_identity:
        intervals=[(row[stamp_key]-row[identity_key]-row['clock_bound_s']*1e9,
                    row[stamp_key]-row[identity_key]+row['clock_bound_s']*1e9) for row in comma
                   if type(row.get(identity_key)) is int and type(row.get(stamp_key)) is int]
        lower=max((v[0] for v in intervals),default=1.);upper=min((v[1] for v in intervals),default=0.)
        if lower>upper:return {'valid':False,'samples':[],'native_count':0,'repeats':0,'missing_availability':0,'invalid_stamps':len(comma),
            'reason':'Native clock mappings have no common interval in this source window.'}
        fixed_offset=int(round((lower+upper)/2))
    samples = []; native_count = repeats = missing_availability = invalid_stamps = 0
    for row in comma:
        native = stamp_key in row
        stamp = row.get(stamp_key) if native else row['at_ns']
        if native_identity:
            identity=row.get(identity_key)
            stamp=identity+fixed_offset if type(identity) is int and identity>0 else None
        if type(stamp) is not int or stamp <= 0:
            invalid_stamps += 1; continue
        # A receipt is strongest; an actual recorded query/selection time is a
        # conservative upper bound on receipt. Native timestamp alone does NOT
        # prove that the measurement was already available to a live prediction.
        availability = next((row.get(k) for k in (receipt_key, 'receipt_ns', 'laptop_receipt_ns', 'query_at_ns')
                             if type(row.get(k)) is int and row[k] > 0), None)
        basis = 'recorded_receipt_or_selection'
        if availability is None and not native:
            availability = stamp; basis = 'legacy_shared_at_contract'
        if availability is None:
            missing_availability += 1; basis = 'unknown'
        else:
            availability = max(availability, stamp)
        sample = {'at_ns': stamp, 'value': row[value_key], 'clock_bound_s': row['clock_bound_s'],
                  'available_ns': availability, 'availability_basis': basis}
        if samples:
            prior = samples[-1]
            if stamp < prior['at_ns'] or (stamp == prior['at_ns'] and sample['value'] != prior['value']):
                return {'valid': False, 'reason': channel + ' native clock regressed or reused with conflicting data',
                        'samples': [], 'native_count': native_count, 'repeats': repeats,
                        'missing_availability': missing_availability, 'invalid_stamps': invalid_stamps}
            if stamp == prior['at_ns']:
                prior['clock_bound_s'] = max(prior['clock_bound_s'], sample['clock_bound_s'])
                known = [v for v in (prior['available_ns'], availability) if v is not None]
                prior['available_ns'] = min(known) if known else None
                repeats += 1; continue
        samples.append(sample); native_count += native
    return {'valid': len(samples) >= 2, 'samples': samples, 'native_count': native_count,
            'repeats': repeats, 'missing_availability': missing_availability, 'invalid_stamps': invalid_stamps,
            'times': np.asarray([s['at_ns'] / 1e9 for s in samples]),
            'values': np.asarray([s['value'] for s in samples]),
            'native_identity_preserved':native_identity,'fixed_clock_offset_ns':fixed_offset,
            'timestamp_basis': stamp_key if native_count else 'legacy_shared_at_ns'}


def _interpolated(series, query):
    if not series.get('valid'): return None
    times = series['times']; pos = int(np.searchsorted(times, query))
    if pos == 0 or pos >= len(times) or times[pos] - times[pos - 1] > .3: return None
    return float(np.interp(query, times, series['values']))


def _causal(series, query_ns):
    if not series.get('valid'): return None
    pos = int(np.searchsorted(series['times'], query_ns / 1e9, side='right')) - 1
    while pos >= 0:
        sample = series['samples'][pos]
        if query_ns - sample['at_ns'] > 250_000_000: break
        if sample['at_ns'] <= query_ns and sample['available_ns'] is not None and sample['available_ns'] <= query_ns:
            return sample['value']
        pos -= 1
    return None


def _observed_trajectory(channels, start_ns, horizon, offset, bias, forward_lever=0.):
    """Retrospective diagnostic only: uses future measurements, never readiness.

    Integrate bounded 25ms midpoint steps without extrapolating clock gaps.
    This separates sensor agreement from constant-turn-rate forecasting error.
    """
    position=np.zeros(2); angle=0.; elapsed=0.
    while elapsed < horizon-1e-9:
        dt=min(.025,horizon-elapsed);query=start_ns/1e9+offset+elapsed+dt/2
        values=[_interpolated(channels[k],query) for k in ('speed','yaw')]
        if any(v is None for v in values):return None
        part,turn=_increment(values[0]-bias[0],values[1]-bias[1],dt)
        position+=_rot(angle)@part;angle+=turn;elapsed+=dt
    return position+forward_lever*np.array([math.cos(angle)-1,math.sin(angle)])


def _bounded_case_max(errors, reference_errors):
    """Add uncertainty to its own case before maximizing; never trim failures."""
    if not errors:return None
    return np.max(np.asarray(errors)+np.asarray(reference_errors)[:,None],axis=0)


def _timing_window(window, channels):
    rows = window['rows']; lt = np.array([r['at_ns'] / 1e9 for r in rows])
    shifts = np.arange(-.3, .3001, .005); estimates = {}; costs = []; unresolved = []
    for name, reference_key, minimum_variation in (('yaw', 'yaw_rate_rps', .008), ('speed', 'speed_mps', .4)):
        series = channels.get(name, {})
        if not series.get('valid') or len(series['samples']) < 30: continue
        reference = np.asarray([r[reference_key] for r in rows])
        if np.std(reference) < minimum_variation: continue
        ct = series['times']; observed = series['values']; scores = []
        for shift in shifts:
            query = lt + shift; pos = np.searchsorted(ct, query)
            valid = (pos > 0) & (pos < len(ct)); safe = np.clip(pos, 1, len(ct) - 1)
            valid &= ct[safe] - ct[safe - 1] <= .3
            if valid.sum() < max(30, len(rows) * .75):
                scores.append(float('inf')); continue
            a = reference[valid]; b = np.interp(query[valid], ct, observed)
            residual = b - a; residual -= np.median(residual)
            scores.append(float(np.mean(residual ** 2) / max(np.var(a), 1e-8)))
        scores = np.asarray(scores); best = int(np.argmin(scores))
        if not np.isfinite(scores[best]) or scores[best] > .5 or best in (0, len(shifts) - 1):
            unresolved.append(name + ': excited channel has inadequate overlap, disagreement, or an offset outside the search interval')
            continue
        near = shifts[scores <= scores[best] + max(.0005, scores[best] / max(8, len(rows) / 4))]
        width = float(max(abs(near - shifts[best]))) if len(near) else .3
        if width >= .15:
            unresolved.append(name + ': excited channel timing peak is too broad')
            continue
        relevant = [s for s in series['samples'] if lt[0] - .3 <= s['at_ns'] / 1e9 <= lt[-1] + .3]
        clock = max((s['clock_bound_s'] for s in relevant), default=.2)
        estimates[name] = {'offset_s': float(shifts[best]), 'peak_half_width_s': width,
                           'clock_bound_s': clock, 'score': float(scores[best]),
                           'timestamp_basis': series['timestamp_basis'], 'native_samples': len(relevant)}
        costs.append(scores)
    if unresolved:
        return {'available': False, 'id': window['id'], 'role': window['role'],
                'reasons': unresolved, 'per_channel': estimates}
    if not costs: return None
    scores = np.mean(np.asarray(costs), axis=0); best = int(np.argmin(scores))
    if not np.isfinite(scores[best]) or best in (0, len(shifts) - 1): return None
    offset = float(shifts[best])
    # One common mapping must contain each independently observed native-channel
    # peak. Do not average contradictory service delays into false precision.
    width = max(abs(value['offset_s'] - offset) + value['peak_half_width_s'] for value in estimates.values())
    return {'id': window['id'], 'role': window['role'], 'offset_s': offset,
            'peak_half_width_s': width, 'clock_bound_s': max(v['clock_bound_s'] for v in estimates.values()),
            'score': float(scores[best]), 'per_channel': estimates,
            'channel_offset_span_s': max(v['offset_s'] for v in estimates.values()) - min(v['offset_s'] for v in estimates.values()),
            'sample_count': len(rows), 'start_ns': rows[0]['start_ns'], 'end_ns': rows[-1]['end_ns']}


def fit_motion(rows, comma_rows, alignment=None):
    """Fit on two temporal blocks, validate on separate blocks, max2s horizon.

    Offset convention: comma query time = lidar reference time + offset_s.
    Relative trajectory error is kept separate from the shared static frame
    offset. We never accumulate a constant alignment offset once per scan.
    """
    rows = list(rows); comma_rows = list(comma_rows)
    result = {'schema_version': 1, 'version': VERSION, 'eligible': False, 'reasons': [],
              'evidence_kind': 'sensor_derived_empirical', 'surveyed_ground_truth': False,
              'timing': {'available': False, 'offset_s': None, 'error_bound_s': None, 'fit_windows': [], 'holdout_windows': []},
              'motion': {'available': False},
              'corridor': {'available': False, 'max_verified_horizon_s': None,
                           'holdout_lateral_error_m': None, 'holdout_longitudinal_error_m': None,
                           'horizons': [], 'range_qualified': False},
              'diagnostics': {'valid_increments': sum(r.get('valid') is True for r in rows if isinstance(r, dict))}}
    windows = _windows(rows); groups, clock_issues, repeats = _comma_groups(comma_rows)
    native_groups = {key: {channel: _channel_series(value, channel) for channel in ('speed', 'yaw')}
                     for key, value in groups.items()}
    result['diagnostics']['comma_clock_discontinuity_groups'] = clock_issues
    result['diagnostics']['unchanged_comma_snapshots_ignored'] = repeats
    result['diagnostics']['channel_sources'] = {channel: {
        'native_measurements': sum(group[channel]['native_count'] for group in native_groups.values()),
        'repeated_measurements_ignored': sum(group[channel]['repeats'] for group in native_groups.values()),
        'unknown_availability': sum(group[channel]['missing_availability'] for group in native_groups.values()),
        'invalid_source_stamps': sum(group[channel]['invalid_stamps'] for group in native_groups.values()),
        'invalid_groups': sum(not group[channel]['valid'] for group in native_groups.values())}
        for channel in ('speed', 'yaw')}
    result['diagnostics'].update(temporal_windows=len(windows), fit_windows=sum(w['role'] == 'fit' for w in windows),
                                 holdout_windows=sum(w['role'] == 'holdout' for w in windows))
    timing_windows = []; timing_rejections = []
    se3=any(r.get('ego_increment_se3') for r in rows if isinstance(r,dict))
    if se3 and any(r.get('valid') is True and not r.get('ego_increment_se3') for r in rows):
        result['reasons'].append('Mixed planar and full-3D motion cannot share one calibration fit.');return result
    speed_diagnostics=[]
    for window in windows:
        channels=native_groups.get(window['key'], {})
        if se3:
            speed_diagnostics.append(_timing_window(window,{'speed':channels.get('speed',{})}))
            candidate=_timing_window(window,{'yaw':channels.get('yaw',{})})
        else:candidate = _timing_window(window, channels)
        if candidate and candidate.get('available') is False: timing_rejections.append(candidate)
        elif candidate: timing_windows.append(candidate)
    result['timing']['window_rejections'] = timing_rejections
    if se3:
        result['timing'].update(reference_channel='yaw',speed_response_diagnostics=speed_diagnostics,
            speed_delay_applied=False,timing_basis='Independent lidar rotation vs native yaw; filtered speed response is evaluated separately in motion/trajectory holdouts.')
    fits = [w for w in timing_windows if w['role'] == 'fit']; checks = [w for w in timing_windows if w['role'] == 'holdout']
    result['timing'].update(fit_windows=fits, holdout_windows=checks)
    if len(fits) < 2 or not checks:
        reason = ('Independent yaw timing has '+str(len(fits))+' qualified fitting windows and '+str(len(checks))+
                  ' held-out windows; two fitting and one held-out window are required. See per-window overlap, disagreement and peak-width reasons.'
                  if se3 else 'Need several ordinary-drive windows with changing speed or gentle turns, including a separate held-out window; straight constant motion cannot identify timing.')
        result['reasons'].append(reason); result['timing']['reason'] = reason
        return result
    offset = float(np.median([w['offset_s'] for w in fits]))
    spread = max(abs(w['offset_s'] - offset) for w in timing_windows)
    error = max(abs(channel['offset_s'] - offset) + channel['peak_half_width_s'] + channel['clock_bound_s']
                for window in timing_windows for channel in window['per_channel'].values()) + .005
    channel_summary = {}
    for name in ('speed', 'yaw'):
        observed = [w['per_channel'][name] for w in fits if name in w['per_channel']]
        held = [w['per_channel'][name] for w in checks if name in w['per_channel']]
        if observed:
            own_offset = float(np.median([v['offset_s'] for v in observed]))
            channel_summary[name] = {'offset_s': own_offset, 'fitting_windows': len(observed),
                'held_out_windows': len(held), 'max_peak_half_width_s': max(v['peak_half_width_s'] for v in observed + held),
                'held_out_offset_difference_s': max((abs(v['offset_s'] - own_offset) for v in held), default=None),
                'difference_from_common_offset_s': abs(own_offset - offset)}
    result['timing'].update(per_channel=channel_summary,
        max_channel_offset_span_s=max(w['channel_offset_span_s'] for w in timing_windows),
        channel_timestamps_applied=True, common_offset_bound_includes_channel_disagreement=True)
    if spread > .1 or error > .2:
        result['reasons'].append('Timing shifts differ across independent windows; inspect clock segments or collect more varied static scenes.')
        result['timing'].update(offset_s=offset, error_bound_s=error, reason=result['reasons'][-1])
        return result
    result['timing'].update(available=True, offset_s=offset, error_bound_s=error,
                            uncertainty_scope='per-native-channel peak width + common-offset disagreement across channels/windows + measured clock bounds +5ms search allowance',
                            offset_convention='comma motion-reference time = lidar reference time + offset_s',
                            radar_native_delay_verified=False,
                            within_50ms=error <= .05)
    geometry = alignment.get('geometry', alignment) if isinstance(alignment, dict) else {}
    required = ('yaw_rad', 'sensor_origin_camera_flu_m', 'translation_error_bound_m', 'yaw_error_bound_rad')
    if (not all(key in geometry for key in required) or not _finite(geometry.get('yaw_rad')) or
            not _finite(geometry.get('yaw_error_bound_rad')) or geometry['yaw_error_bound_rad'] < 0 or
            not isinstance(geometry.get('sensor_origin_camera_flu_m'), (list, tuple)) or
            len(geometry['sensor_origin_camera_flu_m']) != 3 or
            not all(_finite(v) for v in geometry['sensor_origin_camera_flu_m'][:2]) or
            not isinstance(geometry.get('translation_error_bound_m'), (list, tuple)) or
            len(geometry['translation_error_bound_m']) != 3 or
            not all(_finite(v) and v >= 0 for v in geometry['translation_error_bound_m'][:2])):
        result['reasons'].append('Relative timing is available; camera/lidar planar alignment still needs enough matched observations.')
        return result
    if se3 and not all(_finite(v) for v in geometry['sensor_origin_camera_flu_m']):
        result['reasons'].append('Full 3D motion requires a finite three-dimensional reference lever arm.');return result
    result['frame_alignment_uncertainty'] = {key: geometry[key] for key in required}
    kinematic_rows=[]; fit_differences = []; held_differences = []; held_sources = []; held_lateral = []; speed_domain = []; yaw_domain = []
    for window in windows:
        channels = native_groups.get(window['key'], {})
        if not all(channels.get(k, {}).get('valid') for k in ('speed', 'yaw')): continue
        for row in window['rows']:
            query = row['at_ns'] / 1e9 + offset
            observed = [_interpolated(channels[k], query) for k in ('speed', 'yaw')]
            if any(value is None for value in observed): continue
            if se3:
                from selfcal_motion3d import twist
                rotation,translation=_body_transform(row,geometry);linear,angular=twist(rotation,translation)
                velocity=linear[:2]/row['dt_s'];angle=float(angular[2])
            else:
                p, angle = _body_increment(row, geometry); velocity = _velocity(p, angle, row['dt_s'])
            kinematic_rows.append({'role':window['role'],'window_id':window['id'],'at_ns':row['at_ns'],'velocity_xy':velocity.tolist(),'yaw':angle/row['dt_s'],'observed':observed})
            difference = [observed[0] - float(velocity[0]), observed[1] - angle / row['dt_s']]
            (fit_differences if window['role'] == 'fit' else held_differences).append(difference)
            if window['role'] == 'holdout':
                held_sources.append(row); held_lateral.append(abs(float(velocity[1])))
                speed_domain.append(observed[0]); yaw_domain.append(abs(observed[1]))
    if len(fit_differences) < 60 or len(held_differences) < 30:
        result['reasons'].append('Too few temporally matched fitting/held-out motion samples remain after clock-gap rejection.')
        return result
    bias = np.median(np.asarray(fit_differences), axis=0)
    # A camera ahead of the nonholonomic reference has v_y = yaw * lever.
    # Fit only training windows; retain held-out disagreement and geometry
    # uncertainty. This is an effective lever, not a measured axle location.
    from empirical_kinematics import MODEL,LEGACY
    turning=[r for r in kinematic_rows if r['role']=='fit' and abs(r['observed'][1]-bias[1])>.05]
    ratios={}
    for r in turning:ratios.setdefault(r['window_id'],[]).append(r['velocity_xy'][1]/(r['observed'][1]-bias[1]))
    blocks=[float(np.median(v)) for v in ratios.values() if len(v)>=5]
    estimate=float(np.median(blocks)) if blocks else float('nan')
    lever_fitted=bool(len(blocks)>=3 and len(turning)>=30 and 0<=estimate<=4)
    forward_lever=estimate if lever_fitted else 0.
    result['diagnostics']['kinematic_fit']={'fit_only':True,'fit_turning_samples':len(turning),'fit_windows':len(blocks),
        'effective_lever_m':estimate if math.isfinite(estimate) else None,'adopted_in_candidate':lever_fitted,
        'interpretation':'Effective camera-reference forward lever; not a surveyed axle location.'}
    held_lateral=[abs(r['velocity_xy'][1]-forward_lever*(r['observed'][1]-bias[1])) for r in kinematic_rows if r['role']=='holdout']
    held_residual = np.abs(np.asarray(held_differences) - bias)
    reference_speed = max(r['quality']['translation_error_m'] / r['dt_s'] for r in held_sources)
    reference_yaw = max(r['quality']['yaw_error_rad'] / r['dt_s'] for r in held_sources)
    from empirical_kinematics import origin_velocity_error
    geometry_errors=[]
    for row in held_sources:
        if se3:
            from selfcal_motion3d import twist
            linear,angular=twist(*_body_transform(row,geometry));velocity=linear/row['dt_s'];angular=angular/row['dt_s']
        else:
            part,angle=_body_increment(row,geometry);velocity=np.r_[_velocity(part,angle,row['dt_s']),0.];angular=np.array([0.,0.,angle/row['dt_s']])
        lever_error=origin_velocity_error(angular,geometry['translation_error_bound_m'])[:2]
        yaw_bound=geometry['yaw_error_bound_rad'];vx,vy=np.abs(velocity[:2])
        frame_error=np.array([vy*abs(math.sin(yaw_bound))+vx*(1-math.cos(yaw_bound)),
                              vx*abs(math.sin(yaw_bound))+vy*(1-math.cos(yaw_bound))])
        geometry_errors.append(lever_error+frame_error)
    lever_velocity_bounds=np.max(geometry_errors,axis=0)
    if se3:
        reference_speed+=max(r['quality'].get('rotation_error_rad',r['quality']['yaw_error_rad'])/r['dt_s'] for r in held_sources)*float(np.linalg.norm(geometry['sensor_origin_camera_flu_m']))
    bounds = np.max(held_residual, axis=0) + [reference_speed + lever_velocity_bounds[0], reference_yaw]
    motion_usable=bool(np.isfinite(bounds).all() and bounds[0]<=5 and bounds[1]<=.2 and abs(bias[0])<=5 and abs(bias[1])<=.2)
    result['diagnostics']['motion_error_components']={'heldout_max_residual':np.max(held_residual,axis=0).tolist(),'registration_speed_bound_mps':float(reference_speed),'registration_yaw_bound_rps':float(reference_yaw),'geometry_velocity_bound_xy_mps':lever_velocity_bounds.tolist(), 'geometry_bound_includes_frame_yaw':True}
    if not motion_usable:
        result['reasons'].append('Motion uncertainty exceeds supported bounds; inspect registration, frame geometry and native motion diagnostics before requesting more driving.')
    result['motion'] = {'available': motion_usable, 'speed_bias_mps': float(bias[0]), 'yaw_bias_rps': float(bias[1]),
        'speed_error_bound_mps': float(bounds[0]), 'yaw_error_bound_rps': float(bounds[1]),
        'lateral_velocity_model': MODEL if lever_fitted else LEGACY,
        'kinematic_forward_lever_m':forward_lever,
        'kinematic_forward_lever_error_m':float(max(.05,max((abs(v-forward_lever) for v in blocks),default=0.)+geometry['translation_error_bound_m'][0])),
        'lateral_velocity_error_bound_mps': float(max(held_lateral) + reference_speed + lever_velocity_bounds[1]),
        'max_verified_speed_mps': max(0., float(max(speed_domain) - bias[0])), 'min_observed_speed_mps': max(0., float(min(speed_domain) - bias[0])),
        'max_verified_yaw_rate_rps': max(0., float(max(yaw_domain) - abs(bias[1]))), 'fit_samples': len(fit_differences),
        'holdout_samples': len(held_differences), 'ego_origin': 'camera_ground_flu',
        'reference_scope': 'lidar-only registration transformed with empirical alignment uncertainty',
        'bias_fitted_on_holdout': False, 'extrapolation_supported': False}
    for horizon in HORIZONS:
        errors = []; independent_windows = set(); reference_errors = []; cases=[]; retrospective=[]; retrospective_bounds=[]
        for window in windows:
            if window['role'] != 'holdout': continue
            future = window['rows']; channels = native_groups.get(window['key'], {})
            if not all(channels.get(k, {}).get('valid') for k in ('speed', 'yaw')): continue
            for index in range(0, len(future), 5):
                start = future[index]['start_ns']; query_ns = start + int(round(offset * 1e9))
                observed = [_causal(channels[k], query_ns) for k in ('speed', 'yaw')]
                if any(value is None for value in observed): continue
                speed = observed[0] - bias[0]; yaw = observed[1] - bias[1]
                if speed < .5: continue
                lidar_translation = np.zeros(2); lidar_angle = 0.; covered = 0.; reference_error = 0.
                full_rotation=np.eye(3);full_translation=np.zeros(3)
                reference_angle_error=0.
                for row in future[index:]:
                    expected = start + int(round(covered * 1e9))
                    if row['start_ns'] != expected:
                        break  # Missing/rejected reference increment: never bridge it.
                    dt = min(row['dt_s'], horizon - covered)
                    fraction = dt / row['dt_s']
                    if se3:
                        from selfcal_motion3d import fraction as rigid_fraction
                        raw=row['ego_increment_se3'];rotation,part=rigid_fraction(np.asarray(raw['rotation']),np.asarray(raw['translation_m']),fraction)
                        full_translation+=full_rotation@part;full_rotation=full_rotation@rotation
                        angle_error=row['quality'].get('rotation_error_rad',row['quality']['yaw_error_rad'])*fraction
                        reference_error+=np.linalg.norm(part)*min(2.,reference_angle_error)+np.linalg.norm(geometry['sensor_origin_camera_flu_m'])*angle_error
                        reference_angle_error+=angle_error
                    else:
                        raw = row['ego_increment_se2']; angle = raw['yaw_rad']
                        velocity = _velocity(raw['translation_m'], angle, row['dt_s'])
                        if abs(angle) < 1e-7:part = np.asarray(raw['translation_m']) * fraction
                        else:
                            a = angle * fraction; aa, bb = math.sin(a) / a, (1 - math.cos(a)) / a
                            part = np.array([[aa, -bb], [bb, aa]]) @ velocity * dt
                        lidar_translation += _rot(lidar_angle) @ part; lidar_angle += angle * fraction
                    reference_error += row['quality']['translation_error_m'] * fraction
                    covered += dt
                    if covered >= horizon - 1e-8: break
                if covered < horizon - 1e-8: continue
                # Static frame alignment acts ONCE on the full relative transform.
                if se3:
                    _,translation=_body_transform({'ego_increment_se3':{'rotation':full_rotation,'translation_m':full_translation}},geometry)
                    reference=translation[:2]
                else:
                    origin = np.asarray(geometry['sensor_origin_camera_flu_m'][:2])
                    reference = _rot(geometry['yaw_rad']) @ lidar_translation + origin - _rot(lidar_angle) @ origin
                from empirical_kinematics import path as kinematic_path
                predicted=kinematic_path(speed,yaw,[horizon],forward_lever)[0]
                error=np.abs(predicted-reference)
                errors.append(error); reference_errors.append(reference_error)
                measured=_observed_trajectory(channels,start,horizon,offset,bias,forward_lever)
                observed_error=None if measured is None else np.abs(measured-reference)
                if observed_error is not None:
                    retrospective.append(observed_error);retrospective_bounds.append(reference_error)
                cases.append({'start_ns':start,'window_id':window['id'],'speed_mps':float(speed),'yaw_rate_rps':float(yaw),
                    'prediction_error_xy_m':error.tolist(),'reference_uncertainty_m':float(reference_error),
                    'retrospective_error_xy_m':None if observed_error is None else observed_error.tolist(),
                    'future_used_for_readiness':False})
                independent_windows.add(window['id'])
        if errors:
            maxima = _bounded_case_max(errors,reference_errors)
            longitudinal, lateral = float(maxima[0]), float(maxima[1])
        else: longitudinal = lateral = None
        supported = bool(len(errors) >= 6 and lateral <= 1. and longitudinal <= 4.)
        result['corridor']['horizons'].append({'horizon_s': horizon, 'samples': len(errors),
            'independent_holdout_windows': len(independent_windows), 'lateral_error_m': lateral,
            'longitudinal_error_m': longitudinal, 'supported': supported,
            'scope': 'maximum of each held-out error plus its own accumulated registration estimate; global alignment added separately', 'diagnostic_raw_max_error_xy_m': np.max(np.asarray(errors),axis=0).tolist() if errors else None, 'diagnostic_reference_error_max_m':max(reference_errors) if reference_errors else None,
            'worst_cases':sorted(cases,key=lambda c:max(c['prediction_error_xy_m'])+c['reference_uncertainty_m'],reverse=True)[:5],
            'retrospective_sensor_agreement':{'samples':len(retrospective),'bounded_error_xy_m':_bounded_case_max(retrospective,retrospective_bounds).tolist() if retrospective else None,
                'raw_max_error_xy_m':np.max(retrospective,axis=0).tolist() if retrospective else None,'uses_future_measurements':True,'qualifies_live_prediction':False}})
    supported = [row for row in result['corridor']['horizons'] if row['supported']]
    if supported and motion_usable:
        best = supported[-1]
        result['corridor'].update(available=True, max_verified_horizon_s=best['horizon_s'],
                                 holdout_lateral_error_m=best['lateral_error_m'],
                                 holdout_longitudinal_error_m=best['longitudinal_error_m'])
        result['eligible'] = True
    else:
        result['reasons'].append('Causal trajectory prediction or motion uncertainty does not support a 0.5s corridor; inspect separated forecast/reference diagnostics. More driving is not automatically the remedy.')
    return result
