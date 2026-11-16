"""Bounded vehicle-motion evidence for notice continuity, independent of FCW.

Reuses the telemetry clock estimate and already received car-state history.
Missing context never prevents an initial presence advisory. This evidence is
not a furniture label, collision readiness result or independent motion survey.
"""
import bisect,math

WINDOW_NS=2_000_000_000
MAX_GAP_NS=200_000_000
MAX_AGE_NS=200_000_000
MAX_CLOCK_BOUND_NS=40_000_000
MAX_STATIONARY_SPEED_MPS=.1


def finite(value):
    if type(value) not in (int,float):return False
    try:return math.isfinite(value)
    except OverflowError:return False


def stamp(value):return type(value) is int and 0<value<2**63


def evidence_qualified(value,now_ns):
    if not isinstance(value,dict) or value.get('valid') is not True or not stamp(now_ns):return False
    t=value.get('source_ns');duration=value.get('duration_ns');gap=value.get('maximum_gap_ns')
    bound=value.get('clock_bound_ns');speed=value.get('maximum_abs_speed_mps')
    return (stamp(t) and 0<=now_ns-t<=MAX_AGE_NS
        and type(duration) is int and duration>=WINDOW_NS
        and type(gap) is int and 0<=gap<=MAX_GAP_NS
        and type(bound) is int and 0<=bound<=MAX_CLOCK_BOUND_NS
        and finite(speed) and 0<=speed<MAX_STATIONARY_SPEED_MPS)


def stationary_evidence(history,sync,now_ns):
    """Use only records received by now and worst-case source freshness.

    Offsets map native comma publication time to laptop monotonic time. Relative
    duration/gaps use the single native clock; the common offset is not applied
    separately to every historical record. The current upper bound is also
    constrained by actual receipt time. Clock bounds are transport assumptions,
    not physical sensor-time calibration.
    """
    if not stamp(now_ns) or not isinstance(sync,(list,tuple)) or len(sync)!=2:return None
    offset,bound=sync
    if not finite(offset) or not finite(bound) or not 0<=bound<=MAX_CLOCK_BOUND_NS:return None
    if not isinstance(history,(list,tuple)) or len(history)>512:return None
    arrived=[]
    for item in history:
        if not isinstance(item,dict):return None
        receipt=item.get('receipt_ns')
        if not stamp(receipt):return None
        if receipt>now_ns:continue
        source=item.get('logMonoTime')
        if not stamp(source):return None
        if arrived and source<=arrived[-1]['logMonoTime']:return None
        arrived.append(item)
    if not arrived:return None
    latest=arrived[-1];native=latest['logMonoTime'];receipt=latest['receipt_ns']
    earliest=math.floor(native+offset-bound);upper=min(math.ceil(native+offset+bound),receipt)
    if not stamp(earliest) or upper<earliest or now_ns-earliest>MAX_AGE_NS or now_ns-receipt>MAX_AGE_NS:return None
    times=[r['logMonoTime'] for r in arrived]
    start=bisect.bisect_right(times,native-WINDOW_NS)-1
    if start<0:return None
    selected=arrived[start:];speeds=[]
    for item in selected:
        data=item.get('data')
        if (item.get('valid') is not True or item.get('service')!='carState' or not isinstance(data,dict)
                or data.get('canValid') is not True or data.get('canTimeout') is not False):return None
        v=data.get('vEgo')
        if not finite(v) or abs(v)>=100:return None
        # A bound inconsistent with the historical receipt is not usable.
        if item['logMonoTime']+offset-bound>item['receipt_ns']:return None
        speeds.append(abs(v))
    gap=max((b-a for a,b in zip(times[start:],times[start+1:])),default=0)
    value={'valid':True,'source_ns':earliest,'source_interval_ns':[earliest,upper],
        'native_source_ns':native,'receipt_ns':receipt,'duration_ns':native-times[start],
        'maximum_gap_ns':gap,'clock_bound_ns':math.ceil(bound),'maximum_abs_speed_mps':max(speeds),
        'samples':len(selected),'basis':'received_valid_carState_history','furniture_evidence':False,
        'collision_readiness_required':False}
    return value if evidence_qualified(value,now_ns) else None
