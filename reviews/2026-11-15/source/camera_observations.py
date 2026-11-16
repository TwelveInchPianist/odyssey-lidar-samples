"""Read-only current-lead evidence from comma modelV2, never camera clearance.

The installed LeadDataV3 schema defines x/y in device coordinates and v as an
absolute speed *norm*. We retain v for audit; it is never a signed velocity.
Only the probTime=0 hypothesis is a current observation. Future lead hypotheses
are not extra independent objects. Frame IDs and exposure timestamps survive.
"""
from collections import deque
import copy
import math
import threading

SCHEMA_VERSION = 1
MAX_AGE_NS = 250_000_000
MAX_CLOCK_BOUND_NS = 40_000_000


def _number(value):
    if type(value) not in (int, float):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, OverflowError):
        return None


def _stamp(value):
    return value if type(value) is int and 0 < value < 2**63 else None


def _empty():
    return {'schema_version': SCHEMA_VERSION, 'service': 'modelV2',
            'kind': 'camera_current_lead', 'frame_origin': 'device_frd',
            'valid': False, 'available': False, 'source_ns': None,
            'measurement_source_ns': None, 'receipt_ns': None,
            'frame_id': None, 'epoch': None, 'points': [], 'reasons': [],
            'diagnostic_only': True, 'fusion_eligible': False,
            'measurement_time_basis': 'exposure_end_unaligned',
            'speed_semantics': 'absolute_norm_not_longitudinal',
            'clearance_evidence': False, 'class_evidence': False}


def normalize_camera_record(item):
    result = _empty()
    reasons = result['reasons']
    if not isinstance(item, dict) or item.get('kind') != 'sample' or item.get('service') != 'modelV2':
        reasons.append('unsupported camera record')
        return result
    result['source_ns'] = _stamp(item.get('logMonoTime', item.get('source_ns')))
    result['receipt_ns'] = _stamp(item.get('laptop_receipt_ns', item.get('receipt_ns')))
    if item.get('valid') is not True:
        reasons.append('native camera model event invalid')
    if result['source_ns'] is None:
        reasons.append('camera source timestamp invalid')
    data = item.get('data')
    if not isinstance(data, dict):
        reasons.append('camera model payload invalid')
        return result
    frame = data.get('frameId')
    result['frame_id'] = frame if type(frame) is int and 0 <= frame < 2**32 else None
    result['measurement_source_ns'] = _stamp(data.get('timestampEof'))
    if result['frame_id'] is None:
        reasons.append('camera frame identity missing or invalid')
    if result['measurement_source_ns'] is None or (result['source_ns'] is not None and
                                                    result['measurement_source_ns'] > result['source_ns']):
        reasons.append('camera exposure timestamp missing or inconsistent')
    drop = _number(data.get('frameDropPerc'))
    if drop is None or not 0 <= drop <= 5:
        reasons.append('camera frame drop rate missing or above 5 percent')
    result['model_confidence'] = data.get('confidence') if isinstance(data.get('confidence'), (str, int)) else None
    leads = data.get('leadsV3')
    if not isinstance(leads, list) or len(leads) > 8:
        reasons.append('camera leads missing or excessive')
        return result
    current = []
    for index, lead in enumerate(leads):
        if not isinstance(lead, dict):
            reasons.append('malformed camera lead')
            continue
        probability_time = _number(lead.get('probTime'))
        probability = _number(lead.get('prob'))
        if probability_time is None or probability is None or not 0 <= probability <= 1:
            reasons.append('invalid camera lead probability/time')
            continue
        # probTime is when the lead-existence question is asked, not a new
        # observation time. Only zero represents the currently selected lead.
        if abs(probability_time) > 1e-6:
            continue
        values = {}
        for field in ('t', 'x', 'y', 'xStd', 'yStd'):
            raw = lead.get(field)
            values[field] = _number(raw[0]) if isinstance(raw, list) and 0 < len(raw) <= 64 else None
        if (any(v is None for v in values.values()) or abs(values['t']) > 1e-6 or
                not 0 < values['x'] <= 1000 or abs(values['y']) > 500 or
                not 0 < values['xStd'] <= 1000 or not 0 < values['yStd'] <= 500):
            reasons.append('invalid current camera lead position or uncertainty')
            continue
        speed = lead.get('v')
        norm = _number(speed[0]) if isinstance(speed, list) and speed else None
        current.append({'lead_index': index, 'probability': probability,
                        'position_device_xy_m': [values['x'], values['y']],
                        'reported_position_std_m': [values['xStd'], values['yStd']],
                        'absolute_speed_norm_mps': norm,
                        'speed_used_for_fusion': False, 'frame_id': result['frame_id'],
                        'independent_of_radar_state': False})
    if len(current) > 1:
        reasons.append('multiple current camera lead hypotheses are ambiguous')
    result['reasons'] = list(dict.fromkeys(reasons))
    result['valid'] = not result['reasons']
    if result['valid']:
        result['points'] = current
    return result


class CameraBuffer:
    """Bounded, causal camera evidence; a source reset requires a new epoch."""
    def __init__(self, max_records=64):
        if type(max_records) is not int or not 2 <= max_records <= 256:
            raise ValueError('Camera history bound must be 2..256')
        self.lock = threading.RLock()
        self.history = deque(maxlen=max_records)
        self.epoch = 0
        self.route = self.source_epoch = self.source_commit = None
        self.last_source = self.last_measurement = self.last_receipt = self.last_frame = None
        self.blocked = None

    def reset(self, reason='explicit source reset'):
        with self.lock:
            self.history.clear()
            self.epoch += 1
            self.last_source = self.last_measurement = self.last_receipt = self.last_frame = None
            self.blocked = None

    def accept(self, item, receipt_ns=None):
        if not isinstance(item, dict):
            return False
        with self.lock:
            kind = item.get('kind')
            if kind in ('hello', 'stream_restart'):
                self.reset()
                self.route = None
                self.source_epoch = item.get('epoch', item.get('stream_epoch'))
                self.source_commit = item.get('commit') if isinstance(item.get('commit'), str) else None
                return False
            if kind == 'route_reference':
                route = item.get('route')
                if isinstance(route, str) and route:
                    if self.route != route:
                        self.reset('route changed')
                    self.route = route
                return False
            if item.get('service') != 'modelV2':
                return False
            epoch = item.get('epoch', item.get('stream_epoch'))
            if epoch is not None and epoch != self.source_epoch:
                if self.source_epoch is not None:
                    self.reset('source epoch changed')
                self.source_epoch = epoch
            if receipt_ns is not None:
                item = {**item, 'laptop_receipt_ns': receipt_ns}
            record = normalize_camera_record(item)
            for key, attr in (('source_ns', 'last_source'), ('measurement_source_ns', 'last_measurement')):
                stamp = record[key]
                previous = getattr(self, attr)
                if stamp is not None:
                    if previous is not None and stamp <= previous:
                        self.blocked = 'camera source/exposure clock discontinuity'
                    setattr(self, attr, max(stamp, previous or stamp))
            received = record['receipt_ns']
            if received is not None:
                if self.last_receipt is not None and received < self.last_receipt:
                    self.blocked = 'camera receipt clock discontinuity'
                self.last_receipt = max(received, self.last_receipt or received)
            frame = record['frame_id']
            if frame is not None:
                if self.last_frame is not None and frame <= self.last_frame:
                    self.blocked = 'camera frame identity regression or duplicate'
                self.last_frame = max(frame, self.last_frame or frame)
            record.update(epoch=self.epoch, source_epoch=self.source_epoch, route_id=self.route,
                          source_commit=self.source_commit)
            self.history.append(record)
            return bool(record['valid'] and self.blocked is None)

    def latest(self, at_ns, clock_estimate):
        with self.lock:
            result = _empty()
            result.update(epoch=self.epoch, source_epoch=self.source_epoch, route_id=self.route,
                          source_commit=self.source_commit)
            def reject(reason):
                result['reasons'] = [reason]
                return result
            if self.blocked:
                return reject(self.blocked + '; await explicit epoch reset')
            if _stamp(at_ns) is None:
                return reject('invalid camera query timestamp')
            if not isinstance(clock_estimate, (tuple, list)) or len(clock_estimate) != 2:
                return reject('camera clock estimate unavailable')
            offset, bound = map(_number, clock_estimate)
            if offset is None or bound is None or not 0 <= bound <= MAX_CLOCK_BOUND_NS:
                return reject('camera clock bound invalid or above 40 ms')
            if self.history and self.history[-1]['receipt_ns'] is None:
                return reject('latest camera receipt timestamp unavailable')
            record = next((r for r in reversed(self.history) if r['receipt_ns'] is not None and
                           r['receipt_ns'] <= at_ns), None)
            if record is None:
                return reject('no camera record received by query timestamp')
            if not record['valid']:
                return reject('; '.join(record['reasons']))
            exposure = record['measurement_source_ns'] + offset
            age = at_ns - exposure
            receipt_age = at_ns - record['receipt_ns']
            if age < -bound or age + bound > MAX_AGE_NS or receipt_age > MAX_AGE_NS:
                return reject('camera exposure stale or in future')
            result = copy.deepcopy(record)
            result.update(available=True, publication_at_ns=round(record['source_ns'] + offset),
                          measurement_at_ns=round(exposure), source_age_s=age / 1e9,
                          receipt_age_s=receipt_age / 1e9, clock_bound_s=bound / 1e9,
                          measurement_time_basis='bounded_exposure_end; calibration_profile_required')
            return result
