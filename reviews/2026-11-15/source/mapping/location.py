"""Optional location providers. Missing or invalid data never becomes (0, 0).

Raw GPS and fused ECEF are alternative observations, not independent evidence to
double-count. Consumers select one usable observation at a time. Clock validity
is separate from geographic fix validity. Receiver accuracy is not ground truth.
"""
import math
import numpy as np
from .geodesy import ecef_to_geodetic,enu_from_ecef_rotation,horizontal_distance

GPS_SERVICES=('gpsLocation','gpsLocationExternal')
ECEF_SERVICES=('gnssMeasurements','liveLocationKalman')
LOCATION_SERVICES=GPS_SERVICES+ECEF_SERVICES

def reason_text(reasons):
    messages={'service_invalid':'the source is not reporting valid data','no_confirmed_fix':'the receiver has no confirmed fix',
              'invalid_coordinate':'position coordinates are unavailable','invalid_or_poor_horizontal_accuracy':'reported accuracy is missing or too poor',
              'fix_timestamp_unverified_or_stale':'the position timestamp is stale or unverified','missing_receipt_clock':'receipt timing is unavailable',
              'invalid_ecef_position_or_uncertainty':'no usable position with an accuracy estimate is available','fused_location_not_gps_valid':'the device position is not GPS-valid',
              'non_increasing_fix_time':'the position time did not advance','implausible_location_jump':'the position jumped unexpectedly; waiting for consistent fixes'}
    return '; '.join(messages.get(r,'the position needs further validation') for r in reasons)

def finite(value):
    try:r=float(value)
    except (TypeError,ValueError):return None
    return r if math.isfinite(r) else None

def decode_fix(item,sync=None):
    service=item.get('service');d=item.get('data',{})
    if service not in LOCATION_SERVICES:return None
    out={'schema_version':1,'provider':service,'independence_group':'comma_gnss','position_frame':'WGS84',
         'antenna_frame':'comma_device_unsurveyed','source_ns':item.get('logMonoTime'),'receipt_ns':item.get('laptop_receipt_ns'),
         'utc_ms':None,'position':None,'horizontal_accuracy_m':None,'vertical_accuracy_m':None,
         'valid':False,'usable_for_fusion':False,'usable_for_trace':False,'usable_for_index':False,
         'clock_valid':False,'measurement_time_verified':False,'time_basis':'publication_or_receipt_hint','reasons':[]}
    source=out['source_ns'];receipt=out['receipt_ns']
    if type(receipt) is int:out['at_ns']=receipt
    else:out['at_ns']=None;out['reasons'].append('missing_receipt_clock')
    if sync is not None and type(source) is int and type(receipt) is int:
        offset,bound=sync;at=source+offset;age=(receipt-at)/1e9
        if math.isfinite(offset) and math.isfinite(bound) and 0<=bound<=100_000_000 and -bound/1e9-.01<=age<=2.:
            out.update(at_ns=int(at),clock_valid=True,clock_bound_s=bound/1e9)
    out['publication_at_ns']=out['at_ns']
    measurement=item.get('measurement_mono_ns');measurement_bound=finite(item.get('measurement_time_bound_s'))
    # Native GPS messages in the inspected comma fork do not establish a
    # measurement-to-monotonic time bound. Preserve them as geographic hints.
    # A future verified provider can supply this explicit clock contract.
    if out['clock_valid'] and type(measurement) is int and measurement_bound is not None and 0<=measurement_bound<=.05:
        if -.01e9<=source-measurement<=2e9 and out['clock_bound_s']+measurement_bound<=.1:
            out.update(at_ns=int(measurement+sync[0]),measurement_time_verified=True,time_basis='verified_source_measurement_clock',clock_bound_s=out['clock_bound_s']+measurement_bound)
    if not item.get('valid'):out['reasons'].append('service_invalid')
    if service in GPS_SERVICES:
        # Current schema explicitly defines hasFix. Module-specific legacy flags
        # are retained in the original event but never guessed to mean a fix.
        if d.get('hasFix') is not True:out['reasons'].append('no_confirmed_fix')
        lat,lon,alt=(finite(d.get(k)) for k in ('latitude','longitude','altitude'))
        h,v=finite(d.get('horizontalAccuracy')),finite(d.get('verticalAccuracy'))
        if lat is None or lon is None or not -90<=lat<=90 or not -180<=lon<=180:out['reasons'].append('invalid_coordinate')
        else:out['position']={'latitude_deg':lat,'longitude_deg':lon,'altitude_ellipsoid_m':alt}
        if h is None or not 0<h<=100:out['reasons'].append('invalid_or_poor_horizontal_accuracy')
        out['horizontal_accuracy_m']=h if h is not None and h>0 else None;out['vertical_accuracy_m']=v if v is not None and 0<v<=200 else None
        utc=d.get('unixTimestampMillis');wall=item.get('comma_wall_ns')
        if type(utc) is int and utc>0:out['utc_ms']=utc
        if type(utc) is not int or utc<=0 or type(wall) is not int or not -1<=wall/1e9-utc/1000<=5:
            out['reasons'].append('fix_timestamp_unverified_or_stale')
        out['receiver_source']=d.get('source')
        if d.get('source') in ('qcomdiag',8):
            out['vertical_accuracy_m']=None
            out['accuracy_note']='Inspected qcom publisher omits horizontal accuracy and assigns a VDOP-named field to verticalAccuracy; do not infer metre precision from it.'
        out['usable_for_trace']=bool(item.get('valid') and d.get('hasFix') is True and out['position'] is not None and type(utc) is int and utc>0 and type(wall) is int and abs(wall/1e9-utc/1000)<=60)
    else:
        measurement=d.get('positionECEF') or {}
        if service=='liveLocationKalman' and not (d.get('gpsOK') is True and d.get('inputsOK') is True and d.get('status') in ('valid',2)):
            out['reasons'].append('fused_location_not_gps_valid')
        try:
            xyz=np.asarray(measurement.get('value',[]),float);std=np.asarray(measurement.get('std',[]),float)
            if not measurement.get('valid'):raise ValueError()
            llh=ecef_to_geodetic(xyz)
            out['position']={'latitude_deg':float(llh[0]),'longitude_deg':float(llh[1]),'altitude_ellipsoid_m':float(llh[2])}
            if std.shape!=(3,) or not np.isfinite(std).all() or np.any(std<=0):raise ValueError()
            R=enu_from_ecef_rotation(*llh[:2]);cov=R@np.diag(std**2)@R.T
            h=float(np.sqrt(np.linalg.eigvalsh(cov[:2,:2]).max()));v=float(np.sqrt(cov[2,2]))
            out.update(horizontal_accuracy_m=h,vertical_accuracy_m=v)
            if h>100:raise ValueError()
            out.update(position={'latitude_deg':float(llh[0]),'longitude_deg':float(llh[1]),'altitude_ellipsoid_m':float(llh[2])},horizontal_accuracy_m=h,vertical_accuracy_m=v)
        except (TypeError,ValueError,np.linalg.LinAlgError):out['reasons'].append('invalid_ecef_position_or_uncertainty')
        out['receiver_source']='fused' if service=='liveLocationKalman' else 'gnss_solution'
        utc=d.get('unixTimestampMillis')
        if type(utc) is int and utc>0:out['utc_ms']=utc
        out['usable_for_trace']=bool(item.get('valid') and out['position'] is not None and 'fused_location_not_gps_valid' not in out['reasons'])
    out['valid']=not out['reasons'] and out['position'] is not None
    out['usable_for_fusion']=out['valid'] and out['clock_valid'] and out['measurement_time_verified']
    out['usable_for_index']=bool(out['usable_for_trace'] and (out['horizontal_accuracy_m'] is None or out['horizontal_accuracy_m']<=100))
    return out

class FixContinuity:
    """Detect implausible position jumps without drawing a connecting route line.

    A coherent new sequence after a jump starts a new geographic segment. This
    permits recovery from a bad first fix without silently bridging the jump.
    """
    def __init__(self):self.last=None;self.pending=None;self.pending_count=0;self.segment=0
    def reset(self):self.last=None;self.pending=None;self.pending_count=0;self.segment+=1
    def plausible(self,a,b):
        dt=(b['at_ns']-a['at_ns'])/1e9
        if dt<=0:return False
        pa,pb=a['position'],b['position']
        distance=horizontal_distance((pa['latitude_deg'],pa['longitude_deg']),(pb['latitude_deg'],pb['longitude_deg']))
        return distance<=70*dt+3*((a['horizontal_accuracy_m'] or 50)+(b['horizontal_accuracy_m'] or 50))+10
    def accept(self,fix):
        fix=dict(fix);fix['geographic_segment']=self.segment
        if not (fix.get('usable_for_fusion') or fix.get('usable_for_trace')):return fix
        if self.last and fix['at_ns']<=self.last['at_ns']:
            fix.update(usable_for_fusion=False,usable_for_trace=False,usable_for_index=False);fix['reasons']=[*fix['reasons'],'non_increasing_fix_time'];return fix
        if self.last and fix['at_ns']-self.last['at_ns']>5_000_000_000:
            self.reset();fix['geographic_segment']=self.segment;fix['segment_start_reason']='location_gap'
        if self.last and not self.plausible(self.last,fix):
            if self.pending and self.plausible(self.pending,fix):self.pending_count+=1
            else:self.pending_count=1
            self.pending=fix
            if self.pending_count<3:
                fix.update(usable_for_fusion=False,usable_for_trace=False,usable_for_index=False);fix['reasons']=[*fix['reasons'],'implausible_location_jump'];return fix
            self.reset();fix['geographic_segment']=self.segment;fix['segment_start_reason']='reacquired_after_jump'
        self.last=fix;self.pending=None;self.pending_count=0
        return fix

def best_fix(fixes,at_ns,max_age_s=2.,allow_hints=False,allow_trace=False):
    candidates=[f for f in fixes if (f.get('usable_for_fusion') or allow_hints and f.get('usable_for_index') or allow_trace and f.get('usable_for_trace')) and 0<=at_ns-f['at_ns']<=max_age_s*1e9]
    if not candidates:return None
    # Prefer a raw fix if comparably accurate; never fuse correlated providers.
    return min(candidates,key=lambda f:(not f.get('usable_for_fusion'),(f['horizontal_accuracy_m'] or 1000)+(0 if f['provider'] in GPS_SERVICES else 2)+(at_ns-f['at_ns'])/1e9))
