"""Explicit retrospective timing calibration without editing recorded evidence."""
from collections import Counter
import json,math
from pathlib import Path
from recording import frames
from telemetry import Telemetry
from offline_jobs import check_work
from .trajectory import route_records

def checked_timing(cfg,override=None):
    if override is None:
        return {'adjustment_s':cfg.get('lidar_time_adjustment_s',0.),'bound_s':cfg.get('lidar_timing_bound_s'),
                'verified':cfg.get('timing_verified',False),'evidence_id':'recorded-session-calibration'}
    if not isinstance(override,dict) or set(override)!={'adjustment_s','bound_s','verified','evidence_id'}:raise ValueError('Timing revision requires adjustment, bound, verified flag and evidence identity')
    a,b=override['adjustment_s'],override['bound_s']
    if type(override['verified']) is not bool or not override['verified']:raise ValueError('Timing revision must already be explicitly verified')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (a,b)) or not -1<=a<=1 or not 0<b<=.05:raise ValueError('Timing revision is outside supported adjustment/uncertainty limits')
    if not isinstance(override['evidence_id'],str) or not override['evidence_id'].strip():raise ValueError('Timing evidence identity required')
    return dict(override)

class RetimedFrames:
    version='recorded-frame-retiming-v1'
    def __init__(self,timing,underlying=None):self.timing=timing;self.underlying=underlying
    def signature(self,folder):
        return {'timing':self.timing,'source':self.underlying.signature(folder) if self.underlying else 'original-recorded-M1'}
    def __call__(self,folder,cancel=None):
        folder=Path(folder);cfg=json.loads((folder/'session.json').read_text(encoding='utf-8-sig'))
        delta=round((self.timing['adjustment_s']-cfg.get('lidar_time_adjustment_s',0))*1e9)
        diagnostics=Counter();records=iter(route_records(folder,diagnostics,cancel));pending=next(records,None);tele=Telemetry()
        for points,stamps,original,q in (self.underlying or frames)(folder,cancel):
            check_work(cancel);at=original['at_ns']+delta
            # Offline recalibration may use data available by the revised scan
            # time. It never changes the original live decisions or their exam.
            cutoff=max(at,q.get('fusion_cutoff_ns',q.get('last_receipt_ns',at)))
            while pending is not None and pending[1].get('laptop_receipt_ns',0)<=cutoff:
                _,record=pending;receipt=record.get('laptop_receipt_ns')
                if type(receipt) is int:
                    if record.get('kind')=='pong' and 'laptop_send_ns' in record:
                        tele.clock.add(record['laptop_send_ns'],record['remote_receive_ns'],record['remote_send_ns'],receipt)
                    else:tele.accept(record,receipt)
                pending=next(records,None)
            if any(diagnostics.get(k) for k in ('invalid_records','truncated_records','missing_sequence_records','invalid_legacy_records')):
                raise ValueError('Incomplete/corrupt telemetry prevents checked timing recalibration')
            context=tele.context(at)  # No copied ready/motion-ready flags.
            yield points,stamps,context,{**q,'recorded_fusion_at_ns':original['at_ns'],'mapping_retiming':self.timing,
                                        'mapping_telemetry_cutoff_ns':cutoff,'timing_bound_s':self.timing['bound_s']}
