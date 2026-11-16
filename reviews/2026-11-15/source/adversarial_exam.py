"""Frozen synthetic ray-cast regression corpus; not sensor range certification.

Replay identical measured-in-simulation scans for baseline and candidate policies.
Fault schedules are explicit. No generation, network access or training at runtime.
"""
import json
import math
from pathlib import Path
import numpy as np
from fcw_engine import Engine, Mount, clusters, transform
from offline_jobs import check_work

CORPUS_SCHEMA = 'odyssey-fcw-regression-corpus-v1'
SEQUENCES = ('flat_shoulder_clutter', 'flat_stopped_car', 'uphill_poles',
             'uphill_stopped_car', 'banked_stopped_car', 'gentle_grade_clear',
             'gentle_grade_stopped_car', 'sag_with_gantry')
VARIANTS = ('normal', 'sustained', 'flicker')
TARGET_KINDS = ('pedestrian', 'child', 'deer_broadside', 'deer_headon')
SPARSE_IDS = tuple(f'{kind}-{distance}m-drop{drop}' for kind in TARGET_KINDS
                   for distance in (40, 70, 100) for drop in ('03', '40'))
QUALITY = {'completeness': 1., 'timestamp_valid': True, 'arrival_age_s': 0., 'timing_bound_s': .001}


def validate_corpus(corpus):
    if corpus.get('schema') != CORPUS_SCHEMA or corpus.get('provenance', {}).get('synthetic') is not True:
        raise ValueError('Unrecognized synthetic regression corpus')
    sequences, sparse = corpus['sequences'], corpus['sparse']
    if len(sequences) != len(SEQUENCES) or {c['id'] for c in sequences} != set(SEQUENCES):
        raise ValueError('Every fixed driving sequence is required')
    if len(sparse) != len(SPARSE_IDS) or {c['id'] for c in sparse} != set(SPARSE_IDS):
        raise ValueError('Every sparse target, distance and dropout combination is required')
    for case in sequences:
        if len(case['variants']) != 3 or {v['id'] for v in case['variants']} != set(VARIANTS):
            raise ValueError('All ground-loss variants are required')
        if case['frames'] < 20:
            raise ValueError('Driving sequence is truncated')
        target_names = {t['name'] for t in case['targets']}
        if ('stopped_car' in case['id']) != bool(target_names):
            raise ValueError('Fixed hazard inventory is missing or altered')
        for variant in case['variants']:
            expectation = variant['expectation']
            if expectation['off_target_max'] != 0 or set(expectation['required_targets']) != target_names:
                raise ValueError('Fixed scenario expectations may not be weakened')
            expected_failures = ([] if variant['id'] == 'normal' else
                                 [i for i in range(6, case['frames']) if variant['id'] == 'sustained' or i % 3 == 0])
            if variant['fail_indices'] != expected_failures:
                raise ValueError('Fixed ground-loss schedule may not be weakened')
    for case in sparse:
        if case['seeds'] != list(range(8)) or case['modes'] != ['fitted', 'held']:
            raise ValueError('Sparse sampling coverage is incomplete')
    return corpus


def load_scans(root, case):
    path = Path(root) / case['file']
    if path.resolve().parent != Path(root).resolve():
        raise ValueError('Corpus asset must be inside the exam directory')
    with np.load(path, allow_pickle=False) as z:
        data = {name: z[name] for name in z.files}
    p, offsets = data['points'], data['offsets']
    if p.ndim != 2 or p.shape[1] != 4 or not np.isfinite(p).all():
        raise ValueError('Invalid corpus point cloud')
    if offsets.ndim != 1 or offsets.dtype.kind not in 'iu' or offsets[0] != 0 or offsets[-1] != len(p) or np.any(np.diff(offsets) < 0):
        raise ValueError('Invalid corpus frame offsets')
    expected = case.get('frames', len(case.get('seeds', [])))
    if len(offsets) != expected + 1:
        raise ValueError('Incomplete corpus frames')
    return data


def sequence_outcome(data, case, variant, policy, cancel=None):
    engine = Engine(Mount(**case['mount']), policy)
    offsets, stamps, poses = data['offsets'], data['timestamps_ns'], data['ego_xy_yaw']
    count = case['frames']
    if stamps.shape != (count,) or poses.shape != (count, 3) or not np.isfinite(poses).all() or np.any(np.diff(stamps) <= 0):
        raise ValueError('Invalid sequence poses or timestamps')
    speed, yaw = float(case['speed_mps']), float(case['yaw_rate_rps'])
    s = np.linspace(0, 120, 61)
    k = yaw / max(speed, .1)
    path = np.column_stack((np.sin(k*s)/k, (1-np.cos(k*s))/k)) if abs(k) > 1e-6 else np.column_stack((s, s*0))
    failures = set(variant['fail_indices'])
    targets = {t['name']: {'first_detection_distance_m': None, 'first_warning_distance_m': None} for t in case['targets']}
    off_target, sources = 0, {}
    for i in range(count):
        check_work(cancel)
        ground = engine.ground
        def controlled_fit(points, _ground=ground, _fail=i in failures):
            fitted = _ground.__class__.fit(_ground, points)
            if _fail:
                _ground.stable = 0
                _ground.last_reason = 'forced regression-corpus ground loss'
                return None
            return fitted
        engine.ground.fit = controlled_fit
        context = {'at_ns': int(stamps[i]), 'speed_mps': speed, 'yaw_rate_rps': yaw,
                   'ready': True, 'motion_ready': True, 'path': path, 'reasons': []}
        result = engine.update(data['points'][offsets[i]:offsets[i+1]], context, QUALITY)
        if engine.ground is not ground:
            engine.ground.fit = engine.ground.__class__.fit.__get__(engine.ground)
        source = (result.get('ground') or {}).get('source', 'none')
        sources[source] = sources.get(source, 0) + 1
        x, y, angle = poses[i]
        R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        origin = np.array([x, y])
        for target in case['targets']:
            location = np.asarray(target['world_xy'])
            distance = float(np.linalg.norm(location-origin))
            found = any(t.get('misses', 0) == 0 and np.linalg.norm(origin+R@np.asarray(t['position'])-location) < target['match_radius_m'] for t in result['tracks'])
            if found and targets[target['name']]['first_detection_distance_m'] is None:
                targets[target['name']]['first_detection_distance_m'] = distance
        for alert in result['alerts']:
            world = origin + R @ np.asarray(alert['position_m'])
            matched = [t for t in case['targets'] if np.linalg.norm(world-np.asarray(t['world_xy'])) < t['match_radius_m']]
            if not matched:
                off_target += 1
            else:
                for target in matched:
                    if targets[target['name']]['first_warning_distance_m'] is None:
                        targets[target['name']]['first_warning_distance_m'] = float(np.linalg.norm(np.asarray(target['world_xy'])-origin))
    return {'frames': count, 'targets': targets, 'off_target_warnings': off_target, 'ground_sources': sources,
            'detection_misses': sum(t['first_detection_distance_m'] is None for t in targets.values()),
            'warning_misses': sum(t['first_warning_distance_m'] is None for t in targets.values())}


def sequence_regressions(baseline, candidate, required):
    reasons = []
    if candidate['off_target_warnings'] > baseline['off_target_warnings']:
        reasons.append('more off-target warnings')
    if candidate['off_target_warnings'] > 0:
        reasons.append('warning on known non-hazard geometry')
    for name in required:
        for metric in ('first_detection_distance_m', 'first_warning_distance_m'):
            a, b = baseline['targets'][name][metric], candidate['targets'][name][metric]
            if b is None:
                reasons.append(name + ': missing ' + metric)
            elif a is not None and b + .01 < a:
                reasons.append(name + ': shortened ' + metric)
    return reasons


def valid_sequence_outcome(value):
    if type(value['frames']) is not int or value['frames'] < 20: return False
    if type(value['off_target_warnings']) is not int or value['off_target_warnings'] < 0: return False
    for target in value['targets'].values():
        for key in ('first_detection_distance_m', 'first_warning_distance_m'):
            distance = target[key]
            if distance is not None and (isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(distance) or distance < 0): return False
    return (value['detection_misses'] == sum(t['first_detection_distance_m'] is None for t in value['targets'].values()) and
            value['warning_misses'] == sum(t['first_warning_distance_m'] is None for t in value['targets'].values()))


def sparse_outcome(data, case, policy, cancel=None):
    mount = Mount(xyz=[0, 0, 1.79], measured=True, alignment_verified=True)
    result = {}
    for mode in case['modes']:
        hits = []
        ground = {'abc': [0., 0., 0.]}
        if mode == 'held':
            ground.update(require_vertical_support=True, degraded=True, source='held')
        for index, seed in enumerate(case['seeds']):
            check_work(cancel)
            body = transform(data['points'][data['offsets'][index]:data['offsets'][index+1]], mount)
            found = clusters(body, ground, mount, policy)
            location = np.asarray(case['target_world_xy'])
            hits.append(any(np.linalg.norm(d['center']-location) < case['match_radius_m'] for d in found))
        result[mode] = {'seed_hits': hits, 'catches': sum(hits), 'misses': len(hits)-sum(hits)}
    return result


def run_corpus(root, corpus, baseline, candidate, progress=None, cancel=None):
    validate_corpus(corpus)
    sequence_rows, sparse_rows = [], []
    for case in corpus['sequences']:
        data = load_scans(root, case)
        for variant in case['variants']:
            if progress: progress('Road/clutter regression: ' + case['id'] + '/' + variant['id'])
            a = sequence_outcome(data, case, variant, baseline, cancel)
            b = a if baseline == candidate else sequence_outcome(data, case, variant, candidate, cancel)
            reasons = sequence_regressions(a, b, variant['expectation']['required_targets'])
            sequence_rows.append({'case': case['id'], 'variant': variant['id'], 'required_targets': variant['expectation']['required_targets'],
                                  'baseline': a, 'candidate': b, 'regression_reasons': reasons})
    for case in corpus['sparse']:
        if progress: progress('Ray-cast sparse target: ' + case['id'])
        data = load_scans(root, case)
        a = sparse_outcome(data, case, baseline, cancel)
        b = a if baseline == candidate else sparse_outcome(data, case, candidate, cancel)
        lost = sum(old and not new for mode in ('fitted', 'held') for old, new in zip(a[mode]['seed_hits'], b[mode]['seed_hits']))
        sparse_rows.append({'case': case['id'], 'baseline': a, 'candidate': b, 'lost_baseline_catches': lost})
    result = {'schema': CORPUS_SCHEMA, 'sequences': sequence_rows, 'sparse_targets': sparse_rows,
              'metric_note': 'Synthetic moving geometry and forced ground loss; sparse sampling does not model M1 MEMS timing, reflectivity, rain or multipath. Existing misses remain explicit.'}
    for side in ('baseline', 'candidate'):
        result[side+'_off_target_warnings'] = sum(r[side]['off_target_warnings'] for r in sequence_rows)
        result[side+'_sequence_warning_misses'] = sum(r[side]['warning_misses'] for r in sequence_rows)
        result[side+'_sparse_detection_misses'] = sum(r[side][m]['misses'] for r in sparse_rows for m in ('fitted', 'held'))
    result['passed'] = result_is_eligible(result)
    return result


def result_is_eligible(result):
    """Recheck coverage and measurements; a saved pass flag is insufficient."""
    try:
        if result['schema'] != CORPUS_SCHEMA: return False
        sequences, sparse = result['sequences'], result['sparse_targets']
        if len(sequences) != len(SEQUENCES)*3 or {(r['case'], r['variant']) for r in sequences} != {(c, v) for c in SEQUENCES for v in VARIANTS}: return False
        if len(sparse) != len(SPARSE_IDS) or {r['case'] for r in sparse} != set(SPARSE_IDS): return False
        for row in sequences:
            a, b, required = row['baseline'], row['candidate'], row['required_targets']
            if not valid_sequence_outcome(a) or not valid_sequence_outcome(b): return False
            if ('stopped_car' in row['case']) != bool(required): return False
            if a['frames'] < 20 or b['frames'] != a['frames'] or set(a['targets']) != set(required) or set(b['targets']) != set(required): return False
            if sequence_regressions(a, b, required): return False
        for row in sparse:
            for mode in ('fitted', 'held'):
                a, b = row['baseline'][mode], row['candidate'][mode]
                if len(a['seed_hits']) != 8 or len(b['seed_hits']) != 8: return False
                if any(type(v) is not bool for v in a['seed_hits']+b['seed_hits']): return False
                if any(old and not new for old, new in zip(a['seed_hits'], b['seed_hits'])): return False
                for side in (a, b):
                    if side['catches'] != sum(side['seed_hits']) or side['misses'] != 8-sum(side['seed_hits']): return False
        return True
    except (KeyError, TypeError, ValueError, IndexError):
        return False
