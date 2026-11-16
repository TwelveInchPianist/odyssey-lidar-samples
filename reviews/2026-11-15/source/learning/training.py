"""Bounded, resumable radar fitting/evaluation; activation belongs to Application.

train_available runs only in the capped optional worker. It never activates.
verify_promotion performs expensive source/cache hashes in that worker.
check_promotion_identity performs only small metadata/stat checks under app.lock.
Immutable-file assumptions and the short proof-to-activation interval are explicit;
the cheap check is not a second full-content scan or a protection against an
attacker who can rewrite every file and preserve its metadata.
"""
import hashlib
import heapq
from bisect import bisect_right
import json
import os
from pathlib import Path
import runtime_clock as time
import uuid

from offline_jobs import check_work
from .datasets import DatasetStore, iter_trip_records
from .radar_forecast import (build_examples, train_model, evaluate_model, examples_digest,
                             source_profile_valid, digest, OBJECTIVE, VERSION as FEATURE_VERSION)
from .model_registry import ModelRegistry
from .runtime_contract import load_contract

VERSION = 'radar-training-orchestration-v1'
MAX_TRIP_EXAMPLES = 50_000
MAX_ROLE_EXAMPLES = 50_000
MAX_LINE = 64_000
MAX_CACHE_BYTES = 128*1024*1024
PURGE_NS = 2_000_000_000
SAMPLE_SPACING_S = 1.0


def _root(data_root): return Path(data_root)/'learning/radar-training'


def _read(path, limit=2_000_000):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('Missing, linked or oversized training metadata')
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict): raise ValueError('Invalid training metadata')
    return value


def _write(path, value, store):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    store.check_space(len(data)); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    try:
        with temporary.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def _stat(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError('Training evidence missing or linked')
    value = path.stat()
    return {'bytes': value.st_size, 'mtime_ns': value.st_mtime_ns,
            'ctime_ns': value.st_ctime_ns, 'inode': value.st_ino}


def _hash(path, cancel=None):
    before = _stat(path); h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1024*1024): check_work(cancel); h.update(block)
    if _stat(path) != before: raise ValueError('Training input changed during verification')
    return h.hexdigest(), before


def _source_snapshot(source):
    return {key: source.get(key) for key in ('id', 'sha256', 'bytes', 'records', 'valid_records',
        'physical_trip_id', 'source_route_id', 'source_start_ns', 'source_end_ns',
        'independent_trip_eligible', 'qualification_eligible', 'provenance')}


def _inputs(data_root, assets, engine_version):
    store = DatasetStore(data_root); sources = store.sources(); contract = load_contract(assets)
    registry = ModelRegistry(Path(data_root)/'learning/radar-models')
    active_id = registry.active_id
    value = {'version': VERSION, 'engine_version': engine_version, 'objective': OBJECTIVE,
             'runtime_signature': contract['runtime_signature'], 'active_model_id': active_id,
             'sources': [_source_snapshot(s) for s in sorted(sources, key=lambda s: s['id'])]}
    return value, sources, store, registry


def input_signature(data_root, assets, engine_version):
    """Cheap JSON signature: immutable catalog identities/reviews, not raw hashes."""
    return _inputs(data_root, assets, engine_version)[0]


def _identity(source):
    review = source.get('provenance', {}).get('identity_review', {})
    profile = {k: review.get(k) for k in ('openpilot_commit', 'car_fingerprint', 'radar_unavailable')}
    return profile if (review.get('verified') is True and review.get('git_dirty') is False and
                       not review.get('issues') and source_profile_valid(profile)) else None


def _drive(source):
    review = source.get('provenance', {}).get('driving_review', {})
    group = review.get('physical_drive_group')
    if (review.get('verified') is True and review.get('is_driving') is True and
        isinstance(group, str) and 0 < len(group) <= 256 and not review.get('issues')):
        return group
    return None


def _moving_intervals(sources):
    """Source-time CAN moving-anchor evidence, not radar measurement latency."""
    intervals = []
    for source in sources:
        values = source.get('provenance', {}).get('driving_review', {}).get('movement', {}).get('moving_intervals_source_ns')
        if not isinstance(values, list) or not values or len(values) > 20_000: return None
        for value in values:
            if (not isinstance(value, list) or len(value) != 2 or any(type(t) is not int or t <= 0 for t in value) or
                value[1] < value[0]): return None
            intervals.append(value)
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]: merged[-1][1] = max(end, merged[-1][1])
        else: merged.append([start, end])
    return merged


def _routes(sources):
    routes = {}
    for source in sources:
        if source.get('independent_trip_eligible') is not True: continue
        routes.setdefault(source['source_route_id'], []).append(source)
    result = {}
    for route, members in routes.items():
        # Keep all evidence sources for a selected route, including fault-only or
        # parked partial duplicates; do not evade the conflict checker by dropping
        # a less convenient source. Entirely confirmed parked routes are omitted.
        reviews = [s.get('provenance', {}).get('driving_review', {}) for s in members]
        if all(r.get('verified') is True and r.get('is_driving') is False for r in reviews): continue
        groups = {_drive(s) for s in members}
        result[route] = {'sources': sorted(members, key=lambda s: s['id']),
                         'group': next(iter(groups)) if len(groups) == 1 and None not in groups else None}
    return result


def _ledger(data_root):
    path = _root(data_root)/'trip-roles.json'
    if not path.exists(): return {'schema_version': 1, 'objective': OBJECTIVE, 'roles': {}, 'trained_routes': []}
    value = _read(path)
    if value.get('digest') != digest({k: v for k, v in value.items() if k != 'digest'}):
        raise ValueError('Frozen split ledger integrity failed')
    if value.get('schema_version') != 1 or value.get('objective') != OBJECTIVE:
        raise ValueError('Incompatible frozen split ledger')
    if (not isinstance(value.get('roles'), dict) or any(v not in ('training', 'evaluation') for v in value['roles'].values()) or
        not isinstance(value.get('trained_routes'), list) or any(not isinstance(r, str) for r in value['trained_routes'])):
        raise ValueError('Invalid split role ledger')
    return {k: v for k, v in value.items() if k != 'digest'}


def _save_ledger(data_root, value, store):
    _write(_root(data_root)/'trip-roles.json', {**value, 'digest': digest(value)}, store)


def _split(data_root, routes, store):
    ledger = _ledger(data_root); groups = {}
    for route, data in routes.items():
        if data['group'] is not None: groups.setdefault(data['group'], []).append(route)
    if len(groups) < 4:
        # Expose only one physical route during research; do not contaminate every
        # future holdout merely because too few real drives exist today.
        candidates = sorted(routes, key=lambda r: (
            r in ledger['trained_routes'], routes[r]['group'] is not None,
            sum(s.get('valid_records', 0) for s in routes[r]['sources']), r), reverse=True)
        if not candidates: return None, ledger
        route = candidates[0]
        return {'mode': 'research_temporal_purged', 'routes': [route],
                'group_for_route': {route: routes[route]['group'] or 'unverified-physical-drive'},
                'training_groups': [], 'evaluation_groups': [], 'purge_ns': PURGE_NS}, ledger
    # Source routes ever used in research remain training-only after grouping
    # evidence is upgraded. A renamed physical group cannot create virgin data.
    for group, names in groups.items():
        if set(names) & set(ledger['trained_routes']):
            if ledger['roles'].get(group) == 'evaluation':
                raise ValueError('Reviewed physical group merges prior training with a held-out group')
            ledger['roles'][group] = 'training'
    ordered = sorted(groups, key=lambda g: digest(['fixed-drive-split-v1', g]))
    for group in ordered:
        if group in ledger['roles']: continue
        ntrain = sum(ledger['roles'].get(g) == 'training' for g in groups)
        neval = sum(ledger['roles'].get(g) == 'evaluation' for g in groups)
        ledger['roles'][group] = 'training' if ntrain < 2 or (neval >= 2 and ntrain <= neval) else 'evaluation'
    train = [g for g in ordered if ledger['roles'][g] == 'training']
    test = [g for g in ordered if ledger['roles'][g] == 'evaluation']
    # Bounded corpus: deterministic current group sample, including newest
    # registered group through source-end ordering. This is declared sampling,
    # not a claim every historical observation is in every fitted model.
    order_key = lambda g: (max(s.get('source_end_ns') or 0 for r in groups[g] for s in routes[r]['sources']), g)
    train = sorted(train, key=order_key)[-8:]
    test_order = sorted(test, key=order_key)
    test = list(dict.fromkeys(test_order[:2]+test_order[-2:]))
    selected = train+test
    names = sorted(r for g in selected for r in groups[g])
    _save_ledger(data_root, ledger, store)
    return {'mode': 'frozen_whole_physical_drives', 'routes': names,
            'group_for_route': {r: routes[r]['group'] for r in names},
            'training_groups': train, 'evaluation_groups': test, 'purge_ns': 0}, ledger


def _admission(sources, mode):
    reasons = []
    if mode != 'frozen_whole_physical_drives': reasons.append('research_only_insufficient_verified_physical_drives')
    if any(s.get('qualification_eligible') is not True for s in sources): reasons.append('source_qualification_missing')
    profiles = [_identity(s) for s in sources]
    profile = profiles[0] if profiles and profiles[0] is not None and all(p == profiles[0] for p in profiles) else None
    if any(p is None for p in profiles): reasons.append('source_identity_unverified')
    if len({digest(p) for p in profiles if p is not None}) > 1: reasons.append('mixed_vehicle_or_software_profiles')
    if any(_drive(s) is None for s in sources): reasons.append('physical_drive_group_unverified')
    if _moving_intervals(sources) is None: reasons.append('moving_anchor_intervals_unavailable')
    return {'qualified': not reasons, 'reasons': sorted(set(reasons)),
            'basis': 'all used immutable sources, reviewed vehicle identity and measured driving groups'}, profile


def _verify_source(source, cancel=None):
    actual, stat = _hash(Path(source['path']), cancel)
    if actual != source['sha256'] or stat['bytes'] != source['bytes']:
        raise ValueError('Registered dataset content failed independent source verification')
    return {'path': source['path'], 'sha256': actual, 'stat': stat}


def _cache(data_root, route, group, sources, signature, store, cancel, progress):
    key_data = {'version': VERSION, 'feature_version': FEATURE_VERSION,
                'runtime_signature': signature, 'route': route, 'physical_group': group,
                'sample_spacing_s': SAMPLE_SPACING_S, 'max_examples': MAX_TRIP_EXAMPLES,
                'sources': [_source_snapshot(s) for s in sources]}
    key = digest(key_data); root = _root(data_root)/'feature-cache'; root.mkdir(parents=True, exist_ok=True)
    meta = root/(key+'.json'); payload = root/(key+'.jsonl')
    for source in sources: _verify_source(source, cancel)
    if meta.exists():
        value = _read(meta)
        if (value.get('manifest_digest') != digest({k: v for k, v in value.items() if k != 'manifest_digest'}) or value.get('key_data') != key_data or
            value.get('key') != key or value.get('payload_file') != key+'.jsonl' or
            value.get('source_hashes') != {s['id']: s['sha256'] for s in sources} or
            type(value.get('count')) is not int or not 0 <= value['count'] <= MAX_TRIP_EXAMPLES):
            raise ValueError('Feature cache manifest integrity failed')
        sha, stat = _hash(payload, cancel)
        if sha != value['payload_sha256'] or stat['bytes'] != value['payload_bytes']:
            raise ValueError('Feature cache payload integrity failed')
        if progress: progress({'stage': 'features', 'route': route, 'cache': 'verified_reuse'})
        return value
    if progress: progress({'stage': 'features', 'route': route, 'cache': 'building'})
    rows = build_examples(iter_trip_records(sources, cancel), group, route,
                          {s['id']: s['sha256'] for s in sources}, sample_spacing_s=SAMPLE_SPACING_S,
                          max_examples=MAX_TRIP_EXAMPLES)
    intervals = _moving_intervals(sources)
    if intervals is None:
        rows = []  # Missing driving evidence is not a license to fit parked data.
    else:
        starts = [item[0] for item in intervals]
        def moving(row):
            i = bisect_right(starts, row['source_ns'])-1
            return i >= 0 and row['source_ns'] <= intervals[i][1]
        rows = [r for r in rows if moving(r)]
    check_work(cancel); temporary = payload.with_suffix('.jsonl.partial'); sha = hashlib.sha256(); total = 0
    try:
        with temporary.open('wb') as f:
            for i, row in enumerate(rows):
                if i % 250 == 0: check_work(cancel); store.check_space(1024*1024)
                line = (json.dumps({k: v for k, v in row.items() if k != 'source_hashes'}, sort_keys=True,
                                   separators=(',', ':'), allow_nan=False)+'\n').encode()
                total += len(line)
                if len(line) > MAX_LINE or total > MAX_CACHE_BYTES: raise ValueError('Feature cache size budget exceeded')
                f.write(line); sha.update(line)
            f.flush(); os.fsync(f.fileno())
        os.replace(temporary, payload)
    finally:
        if temporary.exists(): temporary.unlink()
    value = {'schema_version': 1, 'key': key, 'key_data': key_data, 'payload_file': payload.name,
             'payload_sha256': sha.hexdigest(), 'payload_bytes': total, 'count': len(rows),
             'anchor_filter': 'reviewed CAN moving intervals at source publication time; future stopping targets retained',
             'source_hashes': {s['id']: s['sha256'] for s in sources}}
    value['manifest_digest'] = digest(value); _write(meta, value, store)
    return value


def _load_rows(data_root, caches, limit, cancel=None):
    """Deterministic bounded reservoir by anchor identity; hash every payload."""
    selected = []; counts = {}; per_cache = max(1, limit//max(1, len(caches)))
    for meta in caches:
        if (not isinstance(meta.get('key'), str) or len(meta['key']) != 64 or
            any(c not in '0123456789abcdef' for c in meta['key']) or meta.get('payload_file') != meta['key']+'.jsonl'):
            raise ValueError('Invalid feature cache identity/path')
        path = _root(data_root)/'feature-cache'/meta['payload_file']
        sha = hashlib.sha256(); count = 0; heap = []
        with path.open('rb') as f:
            while line := f.readline(MAX_LINE+1):
                check_work(cancel)
                if len(line) > MAX_LINE or not line.endswith(b'\n'): raise ValueError('Malformed feature cache record')
                sha.update(line); row = json.loads(line); row['source_hashes'] = meta['source_hashes']; count += 1
                rank = int(digest([row['physical_trip_id'], row['source_route_id'], row['track_id'], row['source_ns']]), 16)
                item = (-rank, count, row)
                if len(heap) < per_cache: heapq.heappush(heap, item)
                elif rank < -heap[0][0]: heapq.heapreplace(heap, item)
        if sha.hexdigest() != meta['payload_sha256'] or count != meta['count']:
            raise ValueError('Feature cache changed or count mismatched')
        counts[meta['key']] = {'available': count, 'selected': len(heap)}
        selected.extend(item[2] for item in heap)
    selected.sort(key=lambda r: (r['physical_trip_id'], r['source_route_id'], r['source_ns'], r['track_id']))
    return selected, counts


def _corpora(data_root, split, caches, cancel=None):
    if split['mode'] == 'research_temporal_purged':
        rows, counts = _load_rows(data_root, caches, MAX_ROLE_EXAMPLES, cancel)
        if not rows: return [], [], counts, None
        times = sorted({r['source_ns'] for r in rows}); boundary = times[int((len(times)-1)*.6)]
        training = [r for r in rows if max(t['support_source_ns'][1] for t in r['targets'].values()) <= boundary]
        evaluation = [r for r in rows if r['source_ns']-round(r['features'][8]*1e9) >= boundary+PURGE_NS]
        return training, evaluation, counts, boundary
    training_cache = [m for m in caches if m['key_data']['physical_group'] in split['training_groups']]
    evaluation_cache = [m for m in caches if m['key_data']['physical_group'] in split['evaluation_groups']]
    training, a = _load_rows(data_root, training_cache, MAX_ROLE_EXAMPLES, cancel)
    evaluation, b = _load_rows(data_root, evaluation_cache, MAX_ROLE_EXAMPLES, cancel)
    return training, evaluation, {**a, **b}, None


def _result_digest(value): return digest({k: v for k, v in value.items() if k != 'result_digest'})


def train_available(data_root, assets, engine_version, cancel=None, progress=None):
    """Fit/evaluate/save only. Caller schedules this in its bounded worker."""
    check_work(cancel)
    signature, sources, store, registry = _inputs(data_root, assets, engine_version)
    key = digest(signature); result_path = _root(data_root)/'results'/(key+'.json')
    if result_path.exists():
        result = _read(result_path)
        if result.get('result_digest') != _result_digest(result) or result.get('input_signature') != signature:
            raise ValueError('Cached training result integrity failed')
        if result.get('model_id'): registry.load_candidate(result['model_id'], result['report_id'])
        for source in sources: _verify_source(source, cancel)
        if result.get('qualified'):
            verify_promotion(data_root, assets, engine_version, result, cancel)
        return result
    routes = _routes(sources); split, ledger = _split(data_root, routes, store)
    base = {'schema_version': 1, 'version': VERSION, 'input_signature': signature,
            'input_signature_sha256': key, 'activated': False, 'qualified': False,
            'model_id': None, 'report_id': None, 'created_at_unix_s': time.time()}
    if split is None:
        result = {**base, 'state': 'insufficient_data', 'reasons': ['no_attributed_moving_or_research_radar_source']}
        result['result_digest'] = _result_digest(result); _write(result_path, result, store); return result
    used_sources = [s for route in split['routes'] for s in routes[route]['sources']]
    admission, profile = _admission(used_sources, split['mode'])
    caches = [_cache(data_root, route, split['group_for_route'][route], routes[route]['sources'],
                     signature['runtime_signature'], store, cancel, progress) for route in split['routes']]
    training, evaluation, sampling, boundary = _corpora(data_root, split, caches, cancel)
    base.update(split=split, research_boundary_ns=boundary, source_admission=admission, source_profile=profile,
                used_source_ids=[s['id'] for s in used_sources], cache_keys=[m['key'] for m in caches],
                sampling=sampling, training_examples=len(training), evaluation_examples=len(evaluation),
                catalog_sources_excluded=len(sources)-len(used_sources))
    if len(training) < 12 or not evaluation:
        result = {**base, 'state': 'insufficient_data', 'reasons': ['insufficient_supported_targets_after_split_and_purge']}
    else:
        # Persist exposure before fitting; a crash must not later turn training
        # data into an allegedly unseen physical-drive holdout.
        ledger['trained_routes'] = sorted(set(ledger['trained_routes']) | {r['source_route_id'] for r in training})
        _save_ledger(data_root, ledger, store); check_work(cancel)
        active = registry.load_active(engine_version, expected_runtime_signature=signature['runtime_signature'])
        if (active['model_id'] if active else None) != signature['active_model_id']:
            raise ValueError('Active comparator changed before training')
        if progress: progress({'stage': 'fit', 'training_examples': len(training), 'evaluation_examples': len(evaluation)})
        model = train_model(training, {'engine_version': engine_version, 'runtime_signature': signature['runtime_signature'],
            'source_profile': profile, 'source_admission': admission, 'orchestrator_version': VERSION,
            'split_mode': split['mode'], 'input_signature_sha256': key})
        check_work(cancel)
        report = evaluate_model(model, evaluation, active_model=active)
        registry.save_candidate(model, report)
        result = {**base, 'state': 'qualified_candidate' if report['qualified'] else
                  'research_candidate' if split['mode'] == 'research_temporal_purged' else 'rejected_candidate',
                  'qualified': report['qualified'], 'model_id': model['model_id'], 'report_id': report['report_id'],
                  'reasons': sorted(set(report['reasons']+admission['reasons'])),
                  'evaluation_data_sha256': examples_digest(evaluation), 'training_data_sha256': examples_digest(training),
                  'comparison_active_model_id': signature['active_model_id'],
                  'metrics': {h: {k: v for k, v in m.items() if k not in ('per_trip', 'per_track')} for h, m in report['horizons'].items()},
                  'scope': 'Radar-relative measurement forecasting only; no collision truth or warning eligibility change.'}
    if input_signature(data_root, assets, engine_version) != signature:
        raise ValueError('Dataset/identity/runtime/active comparator changed during training')
    result['result_digest'] = _result_digest(result)
    if result['qualified']:
        verify_promotion(data_root, assets, engine_version, result, cancel)
    _write(result_path, result, store)
    return result


def _promotion_context(data_root, assets, engine_version, result):
    if result.get('result_digest') != _result_digest(result) or result.get('qualified') is not True:
        raise ValueError('Unqualified or modified training result')
    signature, sources, _, registry = _inputs(data_root, assets, engine_version)
    if signature != result['input_signature'] or digest(signature) != result['input_signature_sha256']:
        raise ValueError('Promotion corpus/identity/runtime/baseline changed')
    selected = {s['id']: s for s in sources}
    used = [selected[k] for k in result['used_source_ids']]
    admission, profile = _admission(used, result['split']['mode'])
    if not admission['qualified'] or profile != result['source_profile']:
        raise ValueError('Source or physical-drive identity no longer qualifies')
    model, report = registry.load_candidate(result['model_id'], result['report_id'])
    if not report['qualified'] or model['provenance'].get('source_profile') != profile:
        raise ValueError('Saved candidate no longer qualifies')
    return signature, used, model, report


def verify_promotion(data_root, assets, engine_version, result, cancel=None):
    """Worker-only full verification, returning the four registry keyword args.

    Also attaches promotion_proof and refreshes result_digest on the supplied
    result dict, so the foreground can later perform only metadata/stat checks.
    No active pointer is written here.
    """
    signature, used, model, report = _promotion_context(data_root, assets, engine_version, result)
    source_proofs = [_verify_source(s, cancel) for s in used]
    cache_root = _root(data_root)/'feature-cache'; caches = []; cache_proofs = []
    for key in result['cache_keys']:
        path = cache_root/(key+'.json'); meta = _read(path)
        if meta.get('manifest_digest') != digest({k: v for k, v in meta.items() if k != 'manifest_digest'}) or meta['key'] != key:
            raise ValueError('Changed feature cache manifest')
        payload = cache_root/meta['payload_file']
        if payload.name != key+'.jsonl': raise ValueError('Invalid feature cache path')
        sha, stat = _hash(payload, cancel)
        if sha != meta['payload_sha256'] or stat['bytes'] != meta['payload_bytes']: raise ValueError('Feature cache integrity failed')
        caches.append(meta); cache_proofs.extend([{'path': str(path), 'stat': _stat(path)}, {'path': str(payload), 'stat': stat}])
    training, evaluation, _, boundary = _corpora(data_root, result['split'], caches, cancel)
    train_hash = examples_digest(training); eval_hash = examples_digest(evaluation)
    if (train_hash != model['training_data_sha256'] or eval_hash != report['evaluation_data_sha256'] or
        train_hash != result['training_data_sha256'] or eval_hash != result['evaluation_data_sha256'] or boundary != result['research_boundary_ns']):
        raise ValueError('Recomputed training/evaluation corpus identity mismatch')
    training_sources = {k: v for row in training for k, v in row['source_hashes'].items()}
    evaluation_sources = {k: v for row in evaluation for k, v in row['source_hashes'].items()}
    if training_sources != model['training_source_hashes'] or evaluation_sources != report['evaluation_source_hashes']:
        raise ValueError('Recomputed model source manifests mismatch')
    roles = _root(data_root)/'trip-roles.json'
    model_path = Path(data_root)/'learning/radar-models/models'/(result['model_id']+'.json')
    report_path = Path(data_root)/'learning/radar-models/reports'/(result['report_id']+'.json')
    proof = {'schema_version': 1, 'source_proofs': source_proofs, 'cache_proofs': cache_proofs,
             'artifact_proofs': [{'path': str(p), 'stat': _stat(p)} for p in (model_path, report_path)],
             'roles_stat': _stat(roles), 'roles_digest': digest(_ledger(data_root)),
             'registry_arguments': {'expected_sources': {'training': training_sources, 'evaluation': evaluation_sources},
                  'expected_eval_digest': eval_hash, 'expected_active_id': signature['active_model_id'],
                  'expected_runtime_signature': signature['runtime_signature']},
             'scope': 'Full hashes checked in worker; foreground rechecks immutable catalog and file identities.'}
    proof['proof_digest'] = digest(proof)
    result['promotion_proof'] = proof
    result['result_digest'] = _result_digest(result)
    return proof['registry_arguments']


def check_promotion_identity(data_root, assets, engine_version, result):
    """Cheap final check. Caller holds app capture/start lock before activation."""
    if result.get('result_digest') != _result_digest(result) or result.get('qualified') is not True:
        raise ValueError('Unqualified or modified training result')
    signature, sources, _, _ = _inputs(data_root, assets, engine_version)
    if signature != result['input_signature'] or digest(signature) != result['input_signature_sha256']:
        raise ValueError('Promotion corpus/identity/runtime/baseline changed')
    by_id = {s['id']: s for s in sources}; used = [by_id[k] for k in result['used_source_ids']]
    admission, profile = _admission(used, result['split']['mode'])
    if not admission['qualified'] or profile != result['source_profile']:
        raise ValueError('Source/physical-drive identity no longer qualifies')
    proof = result.get('promotion_proof')
    if not isinstance(proof, dict) or proof.get('proof_digest') != digest({k: v for k, v in proof.items() if k != 'proof_digest'}):
        raise ValueError('Worker promotion proof missing or changed')
    if {p['path'] for p in proof['source_proofs']} != {s['path'] for s in used}:
        raise ValueError('Promotion source proof coverage mismatch')
    expected_artifacts = {str(Path(data_root)/'learning/radar-models/models'/(result['model_id']+'.json')),
                          str(Path(data_root)/'learning/radar-models/reports'/(result['report_id']+'.json'))}
    if {p['path'] for p in proof['artifact_proofs']} != expected_artifacts:
        raise ValueError('Model/report proof identity mismatch')
    for item in proof['source_proofs']+proof['cache_proofs']+proof['artifact_proofs']:
        if _stat(Path(item['path'])) != item['stat']: raise ValueError('Evidence changed since worker verification')
    if (_stat(_root(data_root)/'trip-roles.json') != proof['roles_stat'] or
        digest(_ledger(data_root)) != proof['roles_digest']): raise ValueError('Frozen physical-drive split changed')
    expected = proof['registry_arguments']
    if (expected.get('expected_eval_digest') != result['evaluation_data_sha256'] or
        expected.get('expected_active_id') != signature['active_model_id'] or
        expected.get('expected_runtime_signature') != signature['runtime_signature']):
        raise ValueError('Worker promotion arguments changed')
    # Source SHA values are recomputed from current catalog metadata, not copied
    # from report. Which sources influenced each corpus was proven in the worker.
    for manifest in expected['expected_sources'].values():
        if any(k not in by_id or by_id[k]['sha256'] != v for k, v in manifest.items()):
            raise ValueError('Worker source hash proof no longer matches catalog')
    return expected
