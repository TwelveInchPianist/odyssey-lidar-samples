"""Small fitted OEM-radar measurement forecaster, not a collision classifier.

Future observations are training/evaluation targets only. Predictions stay in
front-bumper FLU radar-relative XY; no ego-motion/extrinsic/timing verification
is inferred. All caller FCW gates remain authoritative. NumPy is the only ML
dependency. Each active Session must freeze one ForecastModel instance.
"""
from collections import deque, defaultdict
from bisect import bisect_left
import hashlib
import json
import math
import re
import numpy as np

VERSION = 'radar-residual-ridge-v1'
OBJECTIVE = 'raw-oem-radar-relative-measurement-xy-v1'
FRAME = 'front_bumper_flu'
FRAME_TIME = 'future_ego_frame_at_each_horizon'
FEATURES = ('x', 'y', 'vx', 'vy_past', 'ax_past', 'ay_past',
            'x_innovation_scale', 'y_fit_scale', 'history_s', 'support')
HORIZONS = (0.5, 1.0, 2.0)
MAX_GAP_NS = 200_000_000
MAX_TRACKS = 64
MAX_HISTORY = 24
RIDGE = 2.0
SCALE_FLOORS = np.array([5., 1., 2., .5, .5, .5, .05, .05, .25, 4.])
ACCEPTANCE = {
    'minimum_training_trips': 2, 'minimum_evaluation_trips': 2,
    'minimum_training_examples_per_horizon': 100,
    'minimum_evaluation_examples_per_horizon': 100,
    'minimum_evaluation_tracks_per_horizon': 10,
    'minimum_examples_per_trip_horizon': 30,
    'minimum_prediction_coverage': .95,
    'minimum_rmse_fraction_improvement': .05,
    'minimum_rmse_absolute_improvement_m': .005,
    'maximum_p95_increase_m': .05,
    'maximum_worst_case_increase_m': .25,
    'maximum_trip_rmse_fraction_increase': .02,
    'maximum_trip_rmse_absolute_increase_m': .01,
    'maximum_track_p95_increase_m': .10,
    'minimum_examples_for_track_tail_check': 5,
    'maximum_active_rmse_increase_m': 1e-9,
    'maximum_active_p95_increase_m': .01,
    'maximum_active_worst_case_increase_m': .05,
    'minimum_active_rmse_fraction_improvement': .01,
    'minimum_active_rmse_absolute_improvement_m': .001,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def examples_digest(rows):
    """Streaming canonical corpus digest; avoids serializing a whole drive twice."""
    result = hashlib.sha256()
    for row in rows:
        result.update(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        result.update(b'\n')
    return result.hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _identity(value):
    return isinstance(value, str) and 0 < len(value) <= 256


def _hashes(value):
    return (isinstance(value, dict) and bool(value) and all(
        _identity(k) and isinstance(v, str) and len(v) == 64 and
        all(c in '0123456789abcdef' for c in v) for k, v in value.items()))


def source_profile_valid(value):
    return (isinstance(value, dict) and set(value) == {'openpilot_commit', 'car_fingerprint', 'radar_unavailable'} and
            isinstance(value.get('openpilot_commit'), str) and re.fullmatch(r'[0-9a-f]{40}', value['openpilot_commit']) is not None and
            _identity(value.get('car_fingerprint')) and value.get('radar_unavailable') is False)


def _numeric_features(value):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in 'fiu': return None
        a = raw.astype(float, copy=False)
        valid = (a.shape == (len(FEATURES),) and np.isfinite(a).all() and
                 0 < a[0] <= 200 and abs(a[1]) <= 40 and abs(a[2]) <= 80 and
                 abs(a[3]) <= 20 and abs(a[4]) <= 15 and abs(a[5]) <= 15 and
                 0 <= a[6] <= 100 and 0 <= a[7] <= 100 and .25 <= a[8] <= 1.01 and
                 6 <= a[9] <= MAX_HISTORY)
        return a if valid else None
    except (ValueError, TypeError, OverflowError):
        return None


def _point_ok(p):
    return (isinstance(p, dict) and type(p.get('trackId')) is int and
            p['trackId'] >= 0 and p.get('measured') is True and
            p.get('association_ambiguous') is not True and
            all(_finite(p.get(k)) for k in ('dRel', 'yRel', 'vRel')) and
            0 < p['dRel'] <= 200 and abs(p['yRel']) <= 40 and abs(p['vRel']) <= 80)


def _errors_present(value):
    if isinstance(value, dict):
        return any(bool(v) for v in value.values())
    return bool(value)


class FeatureBuilder:
    """Bounded causal histories. Invalid/gapped epochs never bridge track IDs.

    Source-only historical records can omit receipt_ns; the resulting time_basis
    explicitly says so. Callers needing live availability must first use the
    native RadarBuffer freshness/clock checks. No feature uses aRel/yvRel defaults.
    """
    def __init__(self):
        self.history = {}
        self.last_source_ns = None
        self.last_receipt_ns = None
        self.scope = None
        self.last_observations = []

    def clear(self):
        self.history.clear()
        self.last_source_ns = self.last_receipt_ns = None
        self.last_observations = []

    def update(self, record, physical_trip_id, source_route_id):
        self.last_observations = []
        if not _identity(physical_trip_id) or not _identity(source_route_id):
            raise ValueError('Physical trip and source route identities are required')
        if not isinstance(record, dict):
            self.clear(); return []
        t = record.get('source_ns'); receipt = record.get('receipt_ns')
        epoch = record.get('epoch')
        if type(epoch) not in (int, str) or isinstance(epoch, bool):
            self.clear(); return []
        scope = (physical_trip_id, source_route_id, str(epoch))
        valid = (record.get('valid') is True and record.get('available') is True and
                 record.get('kind') == 'raw_oem_tracks' and record.get('frame_origin') == FRAME and
                 not _errors_present(record.get('errors')) and not record.get('reasons') and
                 type(t) is int and t > 0 and isinstance(record.get('points'), list) and
                 len(record['points']) <= MAX_TRACKS and
                 (receipt is None or (type(receipt) is int and receipt > 0)))
        if not valid:
            self.clear(); self.scope = scope; return []
        if (scope != self.scope or (self.last_source_ns is not None and
                                   not 0 < t-self.last_source_ns <= MAX_GAP_NS) or
            (receipt is not None and self.last_receipt_ns is not None and receipt < self.last_receipt_ns)):
            self.clear()
        self.scope = scope; self.last_source_ns = t; self.last_receipt_ns = receipt
        ids = [p.get('trackId') for p in record['points'] if isinstance(p, dict)]
        duplicate = {i for i in ids if type(i) is int and ids.count(i) > 1}
        good = {p['trackId']: p for p in record['points'] if _point_ok(p) and p['trackId'] not in duplicate}
        # Native IDs alone do not prove identity through two overlapping targets.
        # Conservatively omit close, similarly moving competitors as teachers.
        ambiguous = set(); candidates = list(good.items())
        for i, (left_id, left) in enumerate(candidates):
            for right_id, right in candidates[i+1:]:
                if (math.hypot(left['dRel']-right['dRel'], left['yRel']-right['yRel']) < 1.5 and
                    abs(left['vRel']-right['vRel']) < 3.):
                    ambiguous.update((left_id, right_id))
        good = {i: p for i, p in good.items() if i not in ambiguous}
        # A missing/unmeasured observation breaks the continuity teacher.
        self.history = {i: h for i, h in self.history.items() if i in good}
        features = []
        for tid, p in good.items():
            h = self.history.setdefault(tid, deque(maxlen=MAX_HISTORY))
            if h:
                old = h[-1]; dt = (t-old['source_ns'])/1e9
                if (not 0 < dt <= MAX_GAP_NS/1e9 or
                    abs(p['dRel']-old['x']-old['vx']*dt) > 2.+6.*dt or
                    abs(p['yRel']-old['y']) > 1.+10.*dt or abs(p['vRel']-old['vx']) > 15.):
                    h.clear()
            segment = h[0]['segment_id'] if h else digest([*scope, tid, t])
            obs = {'source_ns': t, 'receipt_ns': receipt, 'x': float(p['dRel']),
                   'y': float(p['yRel']), 'vx': float(p['vRel']), 'segment_id': segment,
                   'track_id': tid}
            h.append(obs)
            while len(h) > 1 and t-h[0]['source_ns'] > 1_000_000_000:
                h.popleft()
            self.last_observations.append(obs)
            if len(h) < 6 or t-h[0]['source_ns'] < 250_000_000:
                continue
            times = np.array([(o['source_ns']-t)/1e9 for o in h])
            xs = np.array([o['x'] for o in h]); ys = np.array([o['y'] for o in h])
            vxs = np.array([o['vx'] for o in h]); span = float(-times[0])
            # Ordinary least-squares on past samples only; rank guarded by span.
            linear = np.column_stack((np.ones(len(h)), times))
            ycoef = np.linalg.lstsq(linear, ys, rcond=None)[0]
            ax = float(np.linalg.lstsq(linear, vxs, rcond=None)[0][1])
            quadratic = np.column_stack((np.ones(len(h)), times, times**2))
            qc = np.linalg.lstsq(quadratic, ys, rcond=None)[0]
            vy = float(qc[1]); ay = float(2*qc[2])
            if abs(vy) > 20 or abs(ax) > 15 or abs(ay) > 15:
                continue
            xscale = float(np.sqrt(np.mean((xs-(obs['x']+obs['vx']*times))**2)))
            yscale = float(np.sqrt(np.mean((ys-linear@ycoef)**2)))
            vector = [obs['x'], obs['y'], obs['vx'], vy, ax, ay, xscale, yscale, span, float(len(h))]
            features.append({'feature_version': VERSION, 'frame_origin': FRAME,
                'physical_trip_id': physical_trip_id, 'source_route_id': source_route_id,
                'segment_id': segment, 'track_id': tid, 'source_ns': t, 'receipt_ns': receipt,
                'time_basis': 'source_publication_only' if any(o['receipt_ns'] is None for o in h) else 'source_and_receipt',
                'max_feature_source_ns': t, 'max_feature_receipt_ns': receipt,
                'features': vector})
        return features


def build_examples(records, physical_trip_id, source_route_id, source_hashes,
                   sample_spacing_s=.25, max_examples=100_000):
    """Build causal features and delayed measured targets; no missing=false labels.

    physical_trip_id must identify the entire physical drive, NOT a minute/chunk.
    source_route_id is separately checked at qualification to reject split-route
    leakage. Returns research rows; quality support is not collision ground truth.
    """
    if not _hashes(source_hashes):
        raise ValueError('Nonempty source SHA256 manifest required')
    if not _finite(sample_spacing_s) or sample_spacing_s < .1 or type(max_examples) is not int or max_examples < 1:
        raise ValueError('Invalid bounded sampling settings')
    builder = FeatureBuilder(); anchors = []; observed = defaultdict(list); last = {}; observation_count = 0
    for record in records:
        candidates = builder.update(record, physical_trip_id, source_route_id)
        for obs in builder.last_observations:
            observed[obs['segment_id']].append(obs)
            observation_count += 1
            if observation_count > max_examples*24:
                raise ValueError('Observation memory budget exceeded')
        for row in candidates:
            key = row['segment_id']
            if row['source_ns']-last.get(key, -10**30) >= sample_spacing_s*1e9:
                anchors.append(row); last[key] = row['source_ns']
                if len(anchors) > max_examples:
                    raise ValueError('Example budget exceeded; split extraction at physical-trip-preserving boundaries')
    # Cache once per native-ID continuity segment; avoid rebuilding/converting
    # a long timestamp sequence for every anchor and horizon.
    timestamps = {key: [o['source_ns'] for o in observed[key]] for key in {r['segment_id'] for r in anchors}}
    if any(any(b <= a for a, b in zip(ts, ts[1:])) for ts in timestamps.values()):
        raise ValueError('Source epoch/segment timestamp collision')
    frozen_sources = dict(source_hashes)
    result = []
    for row in anchors:
        history = observed[row['segment_id']]; ts = timestamps[row['segment_id']]
        x = row['features']; targets = {}
        for horizon in HORIZONS:
            target_ns = row['source_ns']+int(horizon*1e9)
            right = bisect_left(ts, target_ns)
            if right >= len(ts): continue
            left = right if ts[right] == target_ns else right-1
            if left < 0 or ts[right]-ts[left] > MAX_GAP_NS: continue
            a, b = history[left], history[right]
            fraction = 0. if left == right else (target_ns-ts[left])/(ts[right]-ts[left])
            point = [a[k]+fraction*(b[k]-a[k]) for k in ('x', 'y')]
            base = [x[0]+x[2]*horizon, x[1]+x[3]*horizon]
            targets[str(horizon)] = {'xy': point, 'baseline_xy': base,
                'residual_xy': [point[i]-base[i] for i in (0, 1)],
                'support_source_ns': [ts[left], ts[right]], 'target_source_ns': target_ns}
        if targets:
            # Shared private snapshot reduces per-example memory. Treat example
            # rows and manifests as immutable after extraction.
            result.append({**row, 'targets': targets, 'source_hashes': frozen_sources})
    return result


def _valid_row(row):
    if not isinstance(row, dict): return False
    x = _numeric_features(row.get('features'))
    return (x is not None and row.get('feature_version') == VERSION and row.get('frame_origin') == FRAME and
            all(_identity(row.get(k)) for k in ('physical_trip_id', 'source_route_id', 'segment_id')) and
            type(row.get('track_id')) is int and row['track_id'] >= 0 and
            type(row.get('source_ns')) is int and row['source_ns'] > 0 and
            type(row.get('max_feature_source_ns')) is int and row['max_feature_source_ns'] <= row['source_ns'] and
            ((row.get('receipt_ns') is None and row.get('max_feature_receipt_ns') is None) or
             (type(row.get('receipt_ns')) is int and type(row.get('max_feature_receipt_ns')) is int and
              row['max_feature_receipt_ns'] <= row['receipt_ns'])) and _hashes(row.get('source_hashes')))


def _target(row, h):
    try:
        t = row['targets'][str(h)]; residual = np.asarray(t['residual_xy'], float)
        xy = np.asarray(t['xy'], float); baseline = np.asarray(t['baseline_xy'], float)
        support = t['support_source_ns']; anchor = row['source_ns']; target = anchor+int(h*1e9)
        if (residual.shape != (2,) or xy.shape != (2,) or baseline.shape != (2,) or
            not np.isfinite(np.r_[residual, xy, baseline]).all() or len(support) != 2 or
            any(type(v) is not int for v in support) or not anchor < support[0] <= target <= support[1] or
            support[1]-support[0] > MAX_GAP_NS or t['target_source_ns'] != target): return None
        expected = np.asarray(row['features'][:2])+h*np.asarray(row['features'][2:4])
        if not np.allclose(baseline, expected, atol=1e-8) or not np.allclose(residual, xy-baseline, atol=1e-8): return None
        return residual
    except (ValueError, TypeError, KeyError, OverflowError):
        return None


def _sources(rows):
    result = {}; route_owners = {}; anchors = set()
    for row in rows:
        route, trip = row['source_route_id'], row['physical_trip_id']
        if route in route_owners and route_owners[route] != trip:
            raise ValueError('One Comma route cannot be split into distinct physical trips')
        route_owners[route] = trip
        anchor = (trip, route, row['track_id'], row['source_ns'])
        if anchor in anchors: raise ValueError('Duplicate physical radar measurement anchor')
        anchors.add(anchor)
        for key, value in row['source_hashes'].items():
            if key in result and result[key] != value: raise ValueError('Conflicting source identity')
            result[key] = value
    return result


def _targets_ok(row):
    targets = row.get('targets')
    return (isinstance(targets, dict) and bool(targets) and
            all(k in {str(h) for h in HORIZONS} and _target(row, float(k)) is not None for k in targets))


def train_model(examples, provenance=None):
    """Fit actual ridge coefficients. One trip may fit a RESEARCH candidate only."""
    rows = list(examples)
    if not rows or len(rows) > 100_000 or any(not _valid_row(r) or not _targets_ok(r) for r in rows):
        raise ValueError('Training examples missing, malformed or causally invalid')
    sources = _sources(rows); models = {}
    for h in HORIZONS:
        selected = [(r, _target(r, h)) for r in rows]
        selected = [(r, y) for r, y in selected if y is not None]
        if len(selected) < 12: continue
        X = np.asarray([r['features'] for r, _ in selected]); Y = np.asarray([y for _, y in selected])
        center = np.median(X, axis=0); scale = np.maximum(np.percentile(X, 75, axis=0)-np.percentile(X, 25, axis=0), SCALE_FLOORS)
        Z = (X-center)/scale
        # Bounded training domain and targets; no truncation that conceals outliers.
        keep = (np.max(np.abs(Z), axis=1) <= 8) & (np.max(np.abs(Y), axis=1) <= 8*h*h+2)
        if int(keep.sum()) < 12: continue
        kept_rows = [r for (r, _), retained in zip(selected, keep) if retained]
        X, Y, Z = X[keep], Y[keep], Z[keep]
        design = np.column_stack((np.ones(len(Z)), Z))
        penalty = np.eye(design.shape[1])*RIDGE; penalty[0, 0] = RIDGE*.01
        coef = np.linalg.solve(design.T@design+penalty, design.T@Y)
        if not np.isfinite(coef).all() or np.max(np.abs(coef)) > 50:
            raise ValueError('Unbounded fitted coefficients')
        residual = np.linalg.norm(Y-design@coef, axis=1)
        models[str(h)] = {'center': center.tolist(), 'scale': scale.tolist(), 'coef': coef.tolist(),
            'minimum': X.min(axis=0).tolist(), 'maximum': X.max(axis=0).tolist(),
            'training_examples': len(X), 'training_rejected': len(selected)-len(X),
            'training_trip_ids': sorted({r['physical_trip_id'] for r in kept_rows}),
            'training_residual_p95_m': float(np.percentile(residual, 95))}
    if not models: raise ValueError('Insufficient supported examples to fit')
    model = {'schema_version': 1, 'version': VERSION, 'objective': OBJECTIVE, 'frame_origin': FRAME,
             'frame_time_basis': FRAME_TIME,
             'feature_names': list(FEATURES), 'ridge': RIDGE, 'models': models,
             'training_trip_ids': sorted({r['physical_trip_id'] for r in rows}),
             'training_route_ids': sorted({r['source_route_id'] for r in rows}),
             'training_data_sha256': examples_digest(rows), 'training_source_hashes': sources,
             'provenance': dict(provenance or {}),
             'scope': 'Each XY target is relative to the future moving ego frame at that horizon; not anchor-frame object motion, ground velocity, or collision truth',
             'qualification': 'research_candidate_until_independent_trip_evaluation'}
    model['model_id'] = digest(model)
    validate_model(model)
    return model


def validate_model(model):
    try:
        if not isinstance(model, dict) or model.get('model_id') != digest({k: v for k, v in model.items() if k != 'model_id'}): raise ValueError('Model hash mismatch')
        if (type(model['schema_version']) is not int or model['schema_version'] != 1 or model['version'] != VERSION or model['objective'] != OBJECTIVE or
            model['frame_origin'] != FRAME or model['feature_names'] != list(FEATURES) or model['ridge'] != RIDGE or
            model.get('frame_time_basis') != FRAME_TIME or
            not _hashes(model['training_source_hashes'])): raise ValueError('Incompatible model schema')
        if (not isinstance(model.get('provenance'), dict) or
            not _hashes({'training': model.get('training_data_sha256')})):
            raise ValueError('Invalid training provenance')
        for name in ('training_trip_ids', 'training_route_ids'):
            if not isinstance(model[name], list) or not model[name] or len(model[name]) != len(set(model[name])) or not all(_identity(v) for v in model[name]): raise ValueError('Invalid training identity')
        if not model['models'] or any(k not in {str(h) for h in HORIZONS} for k in model['models']): raise ValueError('Invalid model horizons')
        for p in model['models'].values():
            arrays = {k: np.asarray(p[k], float) for k in ('center', 'scale', 'coef', 'minimum', 'maximum')}
            if any(v.shape != ((len(FEATURES)+1, 2) if k == 'coef' else (len(FEATURES),)) or not np.isfinite(v).all() for k, v in arrays.items()): raise ValueError('Invalid model arrays')
            if (np.any(arrays['scale'] < SCALE_FLOORS) or np.any(arrays['minimum'] > arrays['maximum']) or
                np.max(np.abs(arrays['coef'])) > 50 or type(p['training_examples']) is not int or p['training_examples'] < 12 or
                not _finite(p['training_residual_p95_m']) or p['training_residual_p95_m'] < 0): raise ValueError('Invalid fitted bounds')
            if not set(p['training_trip_ids']) <= set(model['training_trip_ids']) or not p['training_trip_ids']:
                raise ValueError('Invalid per-horizon training identities')
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as exc:
        raise ValueError('Malformed model artifact') from exc
    return model


class ForecastModel:
    """Frozen per-session policy. Missing/OOD learned head explicitly uses CV.

    ForecastModel(None) is the same defined baseline used in qualification.
    Invalid causal features still yield no prediction at all. Per-horizon source
    fields distinguish learned output from fallback; fallback is not learned gain.
    """
    def __init__(self, artifact=None):
        self.artifact = json.loads(json.dumps(validate_model(artifact), allow_nan=False)) if artifact is not None else None
        self.parameters = {h: {k: np.asarray(v, float) if k in ('center', 'scale', 'coef', 'minimum', 'maximum') else v
                              for k, v in p.items()} for h, p in (self.artifact['models'].items() if self.artifact else [])}

    def predict(self, row):
        # Live feature rows have no training source manifest; all other feature
        # causality/identity checks still apply, with a local placeholder only.
        check = {**row, 'source_hashes': row.get('source_hashes', {'live': '0'*64})} if isinstance(row, dict) else row
        if not _valid_row(check): return {'available': False, 'learned_available': False, 'reason': 'invalid_or_noncausal_features', 'forecasts': []}
        x = np.asarray(row['features'], dtype=float); forecasts = []
        for h in HORIZONS:
            p = self.parameters.get(str(h)); learned = False
            baseline = x[:2]+h*x[2:4]; residual = np.zeros(2); point = baseline
            if p is not None:
                z = (x-p['center'])/p['scale']
                supported = (np.max(np.abs(z)) <= 8 and np.all(x >= p['minimum']-2*p['scale']) and
                             np.all(x <= p['maximum']+2*p['scale']))
                if supported:
                    candidate = np.r_[1., z]@p['coef']; projected = baseline+candidate
                    if (np.isfinite(candidate).all() and np.max(np.abs(candidate)) <= 8*h*h+2 and
                        0 < projected[0] <= 250 and abs(projected[1]) <= 60):
                        residual = candidate; point = projected; learned = True
            forecasts.append({'horizon_s': h, 'xy': point.tolist(), 'baseline_xy': baseline.tolist(),
                'residual_xy': residual.tolist(), 'source': 'learned' if learned else 'constant_relative_motion_fallback',
                'training_residual_p95_m': p['training_residual_p95_m'] if learned else None})
        learned_available = any(p['source'] == 'learned' for p in forecasts)
        return {'available': True, 'learned_available': learned_available,
                'reason': None if learned_available else 'baseline_only_missing_or_unsupported_model',
                'model_id': self.artifact['model_id'] if self.artifact else None, 'objective': OBJECTIVE, 'frame_origin': FRAME,
                'frame_time_basis': FRAME_TIME,
                'source_ns': row['source_ns'], 'time_basis': row.get('time_basis', 'source_publication_only'),
                'forecasts': forecasts, 'collision_probability': None, 'fcw_eligibility': False,
                'uncertainty_note': 'Training residual is not calibrated predictive certainty or a safety bound.'}


def _loss(values):
    a = np.asarray(values, float)
    return {'n': len(a), 'rmse_m': float(np.sqrt(np.mean(a*a))), 'median_m': float(np.median(a)),
            'p95_m': float(np.percentile(a, 95)), 'max_m': float(a.max())} if len(a) else None


def _metrics_valid(p):
    if type(p.get('supported_targets')) is not int or p['supported_targets'] < 0:
        return False
    n = p['supported_targets']
    for count_key, fraction_key in (('learned_predictions', 'coverage'), ('active_learned_predictions', 'active_coverage')):
        if (type(p.get(count_key)) is not int or not 0 <= p[count_key] <= n or
            not _finite(p.get(fraction_key)) or
            abs(p[fraction_key]-(p[count_key]/n if n else 0.)) > 1e-12): return False
    losses = [p.get(key) for key in ('baseline', 'candidate', 'active')]
    if not n: return all(v is None for v in losses)
    for loss in losses:
        if (not isinstance(loss, dict) or type(loss.get('n')) is not int or loss['n'] != n or
            any(not _finite(loss.get(k)) or loss[k] < 0 for k in ('rmse_m', 'median_m', 'p95_m', 'max_m')) or
            not loss['median_m'] <= loss['p95_m'] <= loss['max_m']): return False
    return True


def qualification_reasons(report):
    """Recompute fixed acceptance from metrics; never trust a supplied passed flag."""
    reasons = []
    try:
        if (report['version'] != VERSION or report['objective'] != OBJECTIVE or report['acceptance'] != ACCEPTANCE or
            report.get('fallback_policy') != 'constant_relative_motion_v1' or
            report.get('metrics_scope') != 'all_valid_targets_with_fallback' or report.get('frame_time_basis') != FRAME_TIME):
            return ['incompatible_acceptance_or_objective']
        admission = report.get('source_admission')
        if not isinstance(admission, dict) or admission.get('qualified') is not True or admission.get('reasons') != []:
            reasons.append('source_identity_or_admission_unqualified')
        if not source_profile_valid(report.get('source_profile')): reasons.append('source_profile_unverified')
        if (report.get('comparison_active_model_id') is not None and
            report.get('comparison_active_source_profile') != report.get('source_profile')):
            reasons.append('active_source_profile_mismatch')
        train, test = set(report['training_trip_ids']), set(report['evaluation_trip_ids'])
        if train & test: reasons.append('physical_trip_overlap')
        if set(report['training_route_ids']) & set(report['evaluation_route_ids']): reasons.append('source_route_overlap')
        if set(report['training_source_hashes'].values()) & set(report['evaluation_source_hashes'].values()): reasons.append('source_payload_overlap')
        if set(report['comparison_active_training_trip_ids']) & test: reasons.append('active_training_trip_overlap')
        if set(report['comparison_active_training_route_ids']) & set(report['evaluation_route_ids']): reasons.append('active_training_route_overlap')
        if set(report['comparison_active_training_source_hashes'].values()) & set(report['evaluation_source_hashes'].values()): reasons.append('active_training_payload_overlap')
        if len(train) < ACCEPTANCE['minimum_training_trips']: reasons.append('insufficient_training_trips')
        if len(test) < ACCEPTANCE['minimum_evaluation_trips']: reasons.append('insufficient_independent_evaluation_trips')
        if report['malformed_examples']: reasons.append('malformed_evaluation_examples')
        active_gain = False
        for h in HORIZONS:
            key = str(h); p = report['horizons'].get(key)
            if not p or p['training_examples'] < ACCEPTANCE['minimum_training_examples_per_horizon']:
                reasons.append('insufficient_training_h'+key); continue
            if len(set(p['training_trip_ids'])) < ACCEPTANCE['minimum_training_trips']:
                reasons.append('insufficient_training_trip_support_h'+key)
            if (not _metrics_valid(p) or p['track_groups'] != len(p['per_track']) or
                any(not _metrics_valid(v) for v in list(p['per_trip'].values())+list(p['per_track'].values()))):
                reasons.append('invalid_metrics_h'+key); continue
            if p['supported_targets'] < ACCEPTANCE['minimum_evaluation_examples_per_horizon'] or p['track_groups'] < ACCEPTANCE['minimum_evaluation_tracks_per_horizon']:
                reasons.append('insufficient_evaluation_h'+key)
            if p['coverage'] < ACCEPTANCE['minimum_prediction_coverage']: reasons.append('insufficient_prediction_coverage_h'+key)
            a, b, active = p['baseline'], p['candidate'], p['active']
            if (not a or not b or b['rmse_m'] > a['rmse_m']*(1-ACCEPTANCE['minimum_rmse_fraction_improvement']) or
                a['rmse_m']-b['rmse_m'] < ACCEPTANCE['minimum_rmse_absolute_improvement_m']): reasons.append('no_required_rmse_gain_h'+key)
            if a and b and b['p95_m'] > a['p95_m']+ACCEPTANCE['maximum_p95_increase_m']: reasons.append('tail_regression_h'+key)
            if a and b and b['max_m'] > a['max_m']+ACCEPTANCE['maximum_worst_case_increase_m']: reasons.append('worst_case_regression_h'+key)
            if not active or not b or b['rmse_m'] > active['rmse_m']+ACCEPTANCE['maximum_active_rmse_increase_m']:
                reasons.append('active_rmse_regression_h'+key)
            if active and b:
                if b['p95_m'] > active['p95_m']+ACCEPTANCE['maximum_active_p95_increase_m']: reasons.append('active_tail_regression_h'+key)
                if b['max_m'] > active['max_m']+ACCEPTANCE['maximum_active_worst_case_increase_m']: reasons.append('active_worst_case_regression_h'+key)
                active_gain |= (active['rmse_m']-b['rmse_m'] >= ACCEPTANCE['minimum_active_rmse_absolute_improvement_m'] and
                                b['rmse_m'] <= active['rmse_m']*(1-ACCEPTANCE['minimum_active_rmse_fraction_improvement']))
            if set(p['per_trip']) != test: reasons.append('missing_trip_h'+key)
            for trip, metrics in p['per_trip'].items():
                a, b = metrics['baseline'], metrics['candidate']
                if metrics['supported_targets'] < ACCEPTANCE['minimum_examples_per_trip_horizon'] or metrics['coverage'] < ACCEPTANCE['minimum_prediction_coverage']:
                    reasons.append('insufficient_trip_coverage_h'+key+':'+trip)
                if not a or not b or b['rmse_m'] > a['rmse_m']*(1+ACCEPTANCE['maximum_trip_rmse_fraction_increase'])+ACCEPTANCE['maximum_trip_rmse_absolute_increase_m']:
                    reasons.append('trip_regression_h'+key+':'+trip)
                active = metrics['active']
                if not active or not b or b['rmse_m'] > active['rmse_m']*(1+ACCEPTANCE['maximum_trip_rmse_fraction_increase'])+.001:
                    reasons.append('active_trip_regression_h'+key+':'+trip)
            for track, metrics in p['per_track'].items():
                a, b = metrics['baseline'], metrics['candidate']
                if a and b and a['n'] >= ACCEPTANCE['minimum_examples_for_track_tail_check'] and b['p95_m'] > a['p95_m']+ACCEPTANCE['maximum_track_p95_increase_m']:
                    reasons.append('track_tail_regression_h'+key+':'+track)
                active = metrics['active']
                if active and b and active['n'] >= ACCEPTANCE['minimum_examples_for_track_tail_check'] and b['p95_m'] > active['p95_m']+ACCEPTANCE['maximum_track_p95_increase_m']:
                    reasons.append('active_track_tail_regression_h'+key+':'+track)
        if not active_gain: reasons.append('no_measurable_gain_over_active_policy')
    except (KeyError, TypeError, ValueError, OverflowError):
        reasons.append('malformed_qualification_report')
    return sorted(set(reasons))


def evaluate_model(model, examples, evaluation_source_hashes=None, active_model=None):
    """Compare candidate, frozen active policy and CV on EVERY valid target.

    Each policy's missing/OOD forecast explicitly falls back to CV. Abstention
    cannot delete difficult cases from error/tail metrics; learned coverage is a
    separate qualification requirement. No active-model stacks or implicit mixing.
    """
    validate_model(model); rows = list(examples); predictor = ForecastModel(model)
    active_predictor = ForecastModel(active_model)
    valid = [r for r in rows if _valid_row(r) and _targets_ok(r)]; sources = _sources(valid) if valid else {}
    if evaluation_source_hashes is not None and sources != evaluation_source_hashes:
        raise ValueError('Evaluation source manifest mismatch')
    metrics = {}
    for h in HORIZONS:
        per_trip = defaultdict(list); per_track = defaultdict(list)
        for row in valid:
            y = _target(row, h)
            if y is None: continue
            pred = next((p for p in predictor.predict(row)['forecasts'] if p['horizon_s'] == h), None)
            active_pred = next((p for p in active_predictor.predict(row)['forecasts'] if p['horizon_s'] == h), None)
            pair = (float(np.linalg.norm(y)),
                    float(np.linalg.norm(y-(np.asarray(pred['residual_xy']) if pred else 0.))),
                    float(np.linalg.norm(y-(np.asarray(active_pred['residual_xy']) if active_pred else 0.))),
                    bool(pred and pred['source'] == 'learned'), bool(active_pred and active_pred['source'] == 'learned'))
            per_trip[row['physical_trip_id']].append(pair)
            # Gaps create target-continuity segments, not independent track groups.
            per_track[row['physical_trip_id']+'/'+row['source_route_id']+'/'+str(row['track_id'])].append(pair)
        def summarize(pairs):
            count = len(pairs); learned = sum(p[3] for p in pairs); active_learned = sum(p[4] for p in pairs)
            return {'supported_targets': count, 'learned_predictions': learned,
                    'active_learned_predictions': active_learned,
                    'coverage': learned/count if count else 0., 'active_coverage': active_learned/count if count else 0.,
                    'baseline': _loss([p[0] for p in pairs]), 'candidate': _loss([p[1] for p in pairs]),
                    'active': _loss([p[2] for p in pairs])}
        metrics[str(h)] = {**summarize([p for values in per_trip.values() for p in values]),
            'training_examples': model['models'].get(str(h), {}).get('training_examples', 0),
            'training_trip_ids': model['models'].get(str(h), {}).get('training_trip_ids', []),
            'track_groups': len(per_track), 'per_trip': {k: summarize(v) for k, v in per_trip.items()},
            'per_track': {k: summarize(v) for k, v in per_track.items()}}
    report = {'schema_version': 1, 'version': VERSION, 'objective': OBJECTIVE, 'model_id': model['model_id'],
        'source_admission': model.get('provenance', {}).get('source_admission', {'qualified': False, 'reasons': ['source_admission_missing']}),
        'source_profile': model.get('provenance', {}).get('source_profile'),
        'frame_time_basis': FRAME_TIME, 'fallback_policy': 'constant_relative_motion_v1',
        'metrics_scope': 'all_valid_targets_with_fallback',
        'comparison_active_model_id': active_model['model_id'] if active_model else None,
        'comparison_active_training_trip_ids': active_model['training_trip_ids'] if active_model else [],
        'comparison_active_training_route_ids': active_model['training_route_ids'] if active_model else [],
        'comparison_active_training_source_hashes': active_model['training_source_hashes'] if active_model else {},
        'comparison_active_source_profile': active_model.get('provenance', {}).get('source_profile') if active_model else None,
        'training_data_sha256': model['training_data_sha256'], 'evaluation_data_sha256': examples_digest(rows),
        'training_source_hashes': model['training_source_hashes'], 'evaluation_source_hashes': sources,
        'training_trip_ids': model['training_trip_ids'], 'training_route_ids': model['training_route_ids'],
        'evaluation_trip_ids': sorted({r['physical_trip_id'] for r in valid}),
        'evaluation_route_ids': sorted({r['source_route_id'] for r in valid}),
        'malformed_examples': len(rows)-len(valid), 'acceptance': dict(ACCEPTANCE), 'horizons': metrics,
        'scope': 'Delayed radar measurement prediction only; no collision ground truth or FCW readiness claim.'}
    report['reasons'] = qualification_reasons(report); report['qualified'] = not report['reasons']
    report['report_id'] = digest(report)
    return report


def validate_report(report, model, active_model=None):
    validate_model(model)
    if not isinstance(report, dict) or report.get('report_id') != digest({k: v for k, v in report.items() if k != 'report_id'}):
        raise ValueError('Evaluation hash mismatch')
    for key in ('model_id', 'training_data_sha256', 'training_source_hashes', 'training_trip_ids', 'training_route_ids'):
        if report.get(key) != model.get(key): raise ValueError('Evaluation/model provenance mismatch')
    expected_admission = model.get('provenance', {}).get('source_admission', {'qualified': False, 'reasons': ['source_admission_missing']})
    if (report.get('source_admission') != expected_admission or
        report.get('source_profile') != model.get('provenance', {}).get('source_profile')):
        raise ValueError('Source profile/admission provenance mismatch')
    if not _hashes(report.get('evaluation_source_hashes')) or report.get('schema_version') != 1:
        raise ValueError('Invalid evaluation provenance')
    if not _hashes({'evaluation': report.get('evaluation_data_sha256')}):
        raise ValueError('Invalid evaluated dataset identity')
    for key in ('evaluation_trip_ids', 'evaluation_route_ids'):
        value = report.get(key)
        if not isinstance(value, list) or not value or any(not _identity(v) for v in value) or len(value) != len(set(value)):
            raise ValueError('Invalid evaluation trip/route identities')
    active_id = report.get('comparison_active_model_id')
    if active_id is None:
        if any(report.get(k) for k in ('comparison_active_training_trip_ids', 'comparison_active_training_route_ids',
                                       'comparison_active_training_source_hashes')):
            raise ValueError('Unexpected active comparator provenance')
    elif not _hashes({'active': active_id}):
        raise ValueError('Invalid active comparator identity')
    if active_model is not None:
        validate_model(active_model)
        expected = {'comparison_active_model_id': active_model['model_id'],
                    'comparison_active_training_trip_ids': active_model['training_trip_ids'],
                    'comparison_active_training_route_ids': active_model['training_route_ids'],
                    'comparison_active_training_source_hashes': active_model['training_source_hashes'],
                    'comparison_active_source_profile': active_model.get('provenance', {}).get('source_profile')}
        if any(report.get(k) != v for k, v in expected.items()): raise ValueError('Active comparator provenance mismatch')
    reasons = qualification_reasons(report)
    if report.get('reasons') != reasons or report.get('qualified') is not (not reasons):
        raise ValueError('Evaluation eligibility mismatch')
    return report
