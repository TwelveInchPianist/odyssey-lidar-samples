"""Read-only radar evidence normalization; never a warning-readiness override.

liveTracks contains OEM radar targets before Comma vision association. radarState
contains selected/fused leads and can be vision-only. Coordinates retain the
native front-bumper origin; no installation offset or sensor delay is invented.
"""
from collections import deque
import copy
import math
import threading

SCHEMA_VERSION = 1
SERVICES = ('liveTracks', 'radarState')
ERROR_FIELDS = ('canError', 'radarFault', 'wrongConfig', 'radarUnavailableTemporary')
MAX_POINTS = 64  # bounded decoder contract; inspected Honda supplies <=16
MAX_AGE_NS = 250_000_000
MAX_CLOCK_BOUND_NS = 40_000_000


def _number(value):
    if type(value) not in (int, float): return None
    try:
        converted=float(value)
        return converted if math.isfinite(converted) else None
    except (OverflowError,ValueError): return None


def _timestamp(value):
    return value if type(value) is int and 0 < value < 2**63 else None


def _empty(service=None):
    return {'schema_version': SCHEMA_VERSION, 'service': service,
            'kind': 'raw_oem_tracks' if service == 'liveTracks' else 'fused_leads' if service == 'radarState' else 'unsupported',
            'valid': False, 'available': False, 'source_ns': None, 'receipt_ns': None,
            'native_valid': None,
            'frame_origin': 'front_bumper_flu', 'epoch': None, 'errors': {}, 'reasons': [], 'points': [],
            'diagnostic_only': True, 'fusion_eligible': False, 'timing_verified': False,
            'sensor_delay_s': None, 'extrinsics_verified': False,
            'measurement_time_basis': 'native_publication_only',
            'independent_of_comma_vision': service == 'liveTracks'}


def normalize_radar_record(item):
    """Normalize measurement validity without asserting clock/extrinsic validity.

    Native historical events may lack a laptop receipt timestamp. They remain
    usable source-time evidence; RadarBuffer will never call them live-available.
    Unsupported optional acceleration/lateral velocity remains None, not zero.
    """
    service = item.get('service') if isinstance(item, dict) else None
    result = _empty(service)
    if not isinstance(item, dict) or item.get('kind') != 'sample' or service not in SERVICES:
        result['reasons'] = ['unsupported radar record']; return result
    result['source_ns'] = _timestamp(item.get('logMonoTime', item.get('source_ns')))
    result['native_valid'] = item.get('valid') if type(item.get('valid')) is bool else None
    result['receipt_ns'] = _timestamp(item.get('laptop_receipt_ns', item.get('receipt_ns')))
    epoch = item.get('epoch', item.get('stream_epoch'))
    if type(epoch) in (str, int): result['epoch'] = epoch
    reasons = result['reasons']
    if item.get('valid') is not True: reasons.append('native radar event invalid')
    if result['source_ns'] is None: reasons.append('radar source timestamp invalid')
    data = item.get('data')
    if not isinstance(data, dict):
        reasons.append('radar payload missing or invalid'); return result
    errors = data.get('errors' if service == 'liveTracks' else 'radarErrors')
    if not isinstance(errors, dict) or any(type(errors.get(k)) is not bool for k in ERROR_FIELDS):
        reasons.append('radar error state missing or invalid')
    else:
        result['errors'] = {k: errors[k] for k in ERROR_FIELDS}
        for key, value in errors.items():
            if type(value) is not bool: reasons.append('radar error flag invalid: '+str(key))
            elif value: reasons.append('native radar fault: '+str(key))
    if service == 'liveTracks':
        values = data.get('points')
        if not isinstance(values, list) or len(values) > MAX_POINTS:
            reasons.append('radar point list missing or excessive'); return result
        used = set()
        for point in values:
            if not isinstance(point, dict): reasons.append('malformed radar point'); continue
            identity = point.get('trackId'); required = [_number(point.get(k)) for k in ('dRel', 'yRel', 'vRel')]
            if type(identity) is not int or not 0 <= identity < 2**64:
                reasons.append('radar track identity invalid'); continue
            if identity in used: reasons.append('duplicate radar track identity'); continue
            used.add(identity)
            if (None in required or not 0 <= required[0] <= 1000 or
                    abs(required[1]) > 500 or abs(required[2]) > 200):
                reasons.append('radar required measurement invalid'); continue
            if type(point.get('measured')) is not bool:
                reasons.append('radar measured flag missing or invalid'); continue
            result['points'].append({'trackId': identity, **dict(zip(('dRel', 'yRel', 'vRel'), required)),
                'aRel': _number(point.get('aRel')), 'yvRel': _number(point.get('yvRel')),
                'measured': point['measured'], 'independent_of_comma_vision': True})
    else:
        result['independent_of_comma_vision'] = False
        result['model_source_ns'] = _timestamp(data.get('mdMonoTime'))
        result['car_state_source_ns'] = _timestamp(data.get('carStateMonoTime'))
        for name in ('leadOne', 'leadTwo'):
            point = data.get(name)
            if not isinstance(point, dict) or type(point.get('status')) is not bool:
                reasons.append('radar lead status missing or invalid'); continue
            if not point['status']: continue
            required = [_number(point.get(k)) for k in ('dRel', 'yRel', 'vRel')]
            radar = point.get('radar'); identity = point.get('radarTrackId')
            if (None in required or not 0 <= required[0] <= 1000 or abs(required[1]) > 500 or
                    abs(required[2]) > 200 or type(radar) is not bool or type(identity) is not int or
                    (radar and identity < 0)):
                reasons.append('radar lead measurement or provenance invalid'); continue
            result['points'].append({'lead': name, 'trackId': identity, 'radarTrackId': identity,
                **dict(zip(('dRel', 'yRel', 'vRel'), required)), 'aRel': _number(point.get('aRel')),
                'yvRel': None, 'measured': None, 'radar': radar,
                'provenance': 'radar_backed_selected_lead' if radar else 'vision_only_lead',
                'independent_of_comma_vision': False, 'modelProb': _number(point.get('modelProb')),
                'vLead': _number(point.get('vLead')), 'vLeadK': _number(point.get('vLeadK')),
                'aLeadK': _number(point.get('aLeadK'))})
    result['reasons'] = list(dict.fromkeys(reasons))
    result['valid'] = result['available'] = not result['reasons']
    if not result['valid']: result['points'] = []
    return result


class RadarBuffer:
    """Small, thread-safe diagnostic history; latest fault never falls back good.

    A clock regression cannot establish a new source epoch by itself. Explicit
    hello/stream restart is required before accepting data under a new clock.
    """
    def __init__(self, max_records=64):
        if type(max_records) is not int or not 2 <= max_records <= 256: raise ValueError('Radar history bound must be 2..256')
        self.max_records = max_records; self.lock = threading.RLock(); self.epoch = 0
        self.route = None; self.source_epoch = None; self.blocked = False
        self.history = {s: deque(maxlen=max_records) for s in SERVICES}; self.last_source = {}; self.last_receipt = {}
        self.blocked_reason = None

    def reset(self, reason='source epoch reset'):
        with self.lock:
            self.epoch += 1; self.blocked = False; self.blocked_reason = None
            self.last_source.clear(); self.last_receipt.clear()
            for rows in self.history.values(): rows.clear()

    def accept(self, item, receipt_ns=None):
        if not isinstance(item, dict): return False
        with self.lock:
            kind = item.get('kind')
            if kind in ('hello', 'stream_restart'):
                self.reset(); self.route = None; self.source_epoch = item.get('epoch', item.get('stream_epoch')); return False
            if kind == 'route_reference':
                route = item.get('route')
                if isinstance(route, str) and route:
                    if self.route != route: self.reset('route changed')
                    self.route = route
                return False
            service = item.get('service')
            if service not in SERVICES: return False
            explicit_epoch = item.get('epoch', item.get('stream_epoch'))
            if explicit_epoch is not None and explicit_epoch != self.source_epoch:
                if self.source_epoch is not None: self.reset('source epoch changed')
                self.source_epoch = explicit_epoch
            if receipt_ns is not None: item = {**item, 'laptop_receipt_ns': receipt_ns}
            record = normalize_radar_record(item); stamp = record['source_ns']
            if stamp is not None and service in self.last_source and stamp <= self.last_source[service]:
                self.blocked = True; self.blocked_reason = 'radar source clock discontinuity'
            if stamp is not None: self.last_source[service] = max(stamp, self.last_source.get(service, stamp))
            received = record['receipt_ns']
            # Historical source-only rows have no receipt and never establish a
            # laptop time. Within an established live epoch, time cannot regress.
            if received is not None:
                if service in self.last_receipt and received < self.last_receipt[service]:
                    self.blocked = True; self.blocked_reason = 'radar receipt clock discontinuity'
                self.last_receipt[service] = max(received, self.last_receipt.get(service, received))
            record.update(epoch=self.epoch, route_id=self.route, source_epoch=self.source_epoch)
            self.history[service].append(record)
            return bool(record['valid'] and not self.blocked)

    def latest(self, at_ns, clock_estimate, service='liveTracks'):
        with self.lock:
            result = _empty(service); result.update(epoch=self.epoch, route_id=self.route, source_epoch=self.source_epoch)
            def unavailable(reason, evidence=None):
                if evidence: result.update({k: copy.deepcopy(evidence[k]) for k in ('source_ns','receipt_ns','errors')})
                result['reasons'] = [reason]; return result
            if service not in SERVICES: return unavailable('unsupported radar service')
            if self.blocked: return unavailable((self.blocked_reason or 'radar clock discontinuity')+'; await explicit epoch reset')
            if _timestamp(at_ns) is None: return unavailable('invalid radar query time')
            if not isinstance(clock_estimate, (tuple, list)) or len(clock_estimate) != 2:
                return unavailable('radar clock estimate unavailable')
            offset, bound = map(_number, clock_estimate)
            if offset is None or bound is None or not 0 <= bound <= MAX_CLOCK_BOUND_NS:
                return unavailable('radar clock bound invalid or above 40 ms')
            if self.history[service] and self.history[service][-1]['receipt_ns'] is None:
                return unavailable('latest radar receipt time unavailable')
            record = next((r for r in reversed(self.history[service]) if r['receipt_ns'] is not None and r['receipt_ns'] <= at_ns), None)
            if record is None: return unavailable('no radar record received by query time')
            if not record['valid']: return unavailable('; '.join(record['reasons']), record)
            source_time = record['source_ns'] + offset; source_age = at_ns-source_time
            receipt_age = at_ns-record['receipt_ns']
            if source_age < -bound or source_age+bound > MAX_AGE_NS or receipt_age > MAX_AGE_NS:
                return unavailable('radar evidence stale or source time in future', record)
            result = copy.deepcopy(record)
            result.update(available=True, publication_at_ns=round(source_time), source_age_s=source_age/1e9,
                          receipt_age_s=receipt_age/1e9, clock_bound_s=bound/1e9,
                          measurement_time_basis='bounded_publication_clock; sensor_latency_unverified')
            return result
