"""Phase 1: evidence-only, post-session hindsight. Never changes live settings.

Local short-horizon odometry integrates comma wheel speed and calibrated pose
yaw. It is not a globally accurate trajectory, an impact detector, or ground
truth. Missing clocks, pose, response fields or footprint calibration abstain.
"""
from bisect import bisect_left,bisect_right
from collections import Counter
from dataclasses import dataclass
import hashlib,json,math
from pathlib import Path
import numpy as np
from telemetry import Telemetry,number
from offline_jobs import check_work

VERSION='1.2'
POLICY={'reaction_lookback_s':5.,'pass_horizon_s':6.,'pass_max_distance_m':40.,
        'max_sample_gap_s':.20,'hard_brake_mps2':-3.,'brake_duration_s':.20,
        'sharp_drop_mps':3.,'sharp_drop_window_s':1.,
        'steer_rate_deg_s':60.,'steer_duration_s':.15,'reaction_min_speed_mps':3.,
        'no_response_decel_mps2':-1.2,'no_response_steer_rate_deg_s':12.,
        'no_response_steer_change_deg':6.,'stationary_object_max_mps':.75}
DEFAULT_FOOTPRINT={'confirmed':False,'width_m':1.8,'front_m':2.,'rear_m':3.2,'center_y_m':0.}

def input_signature(folder):
    folder=Path(folder)
    return {name:[(folder/name).stat().st_size,(folder/name).stat().st_mtime_ns] for name in
            ('session.json','final-state.json','comma.jsonl','decisions.jsonl','frame-contexts.jsonl','raw-capture-index.json') if (folder/name).exists()}

def validate_footprint(p):
    if not isinstance(p,dict):raise ValueError('Hindsight footprint must be an object')
    if not isinstance(p.get('confirmed'),bool):raise ValueError('Footprint confirmation must be true or false')
    limits={'width_m':(1.,2.6),'front_m':(.1,4.),'rear_m':(.1,6.),'center_y_m':(-1.,1.)}
    for key,(lo,hi) in limits.items():
        if not math.isfinite(number(p.get(key))) or not lo<=float(p[key])<=hi:
            raise ValueError('Invalid hindsight footprint '+key)
    return p

def read_jsonl(path,diagnostics=None,cancel=None):
    if not path.exists():return
    with path.open(encoding='utf-8') as f:
        for line_no,line in enumerate(f,1):
            check_work(cancel)
            try:
                item=json.loads(line)
                if not isinstance(item,dict):raise ValueError('not an object')
                yield line_no,item
            except (ValueError,TypeError):
                if diagnostics is not None:diagnostics['malformed_'+path.name]+=1

def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(path)

def ident(kind,t,track=None):
    return kind+'-'+hashlib.sha256(f'{t}:{track}'.encode()).hexdigest()[:16]

@dataclass(slots=True)
class Sample:
    t:int
    speed:float
    accel:float
    angle:float
    rate:float
    brake:object
    valid:bool
    clock_ok:bool
    yaw:float
    motion_ok:bool
    epoch:int=0
    segment:int=-1
    x:float=0.
    y:float=0.
    heading:float=0.
    distance:float=0.
    clock_bound_s:float=1.

def load_samples(folder,diagnostics,cancel=None):
    tele=Telemetry();samples=[];last_record=0;last_car=0;epoch=0;last_clock_mode=None
    malformed_seen=diagnostics.get('malformed_comma.jsonl',0)
    def evidence_gap(service=None):
        nonlocal epoch
        epoch+=1
        # Keep valid clock exchanges, but never reuse a source whose newest
        # preserved record could not be interpreted. An unknown record clears
        # the motion messages rather than guessing which source it damaged.
        if isinstance(service,str):
            tele.messages.pop(service,None);tele.history.pop(service,None);tele.selected_times.pop(service,None)
        else:
            tele.messages.clear();tele.history.clear();tele.selected_times.clear()
    for _,r in read_jsonl(folder/'comma.jsonl',diagnostics,cancel):
        malformed_now=diagnostics.get('malformed_comma.jsonl',0)
        if malformed_now>malformed_seen:
            evidence_gap();diagnostics['malformed_motion_record_gaps']+=malformed_now-malformed_seen
            malformed_seen=malformed_now
        receipt=r.get('laptop_receipt_ns')
        if type(receipt) is not int:
            diagnostics['missing_receipt_time']+=1;evidence_gap(r.get('service'));continue
        if receipt<last_record:
            tele=Telemetry();last_car=0;last_clock_mode=None;epoch+=1;diagnostics['receipt_time_resets']+=1
        last_record=receipt
        if r.get('service') in ('carState','livePose','liveLocationKalman','liveCalibration') and r.get('kind')!='sample':
            diagnostics['invalid_motion_record_kind']+=1;evidence_gap(r.get('service'));continue
        if r.get('kind')=='hello':epoch+=1;last_car=0;last_clock_mode=None;tele.accept(r,receipt);continue
        if r.get('kind')=='pong' and 'laptop_send_ns' in r:
            try:
                if any(type(r.get(k)) is not int for k in ('laptop_send_ns','remote_receive_ns','remote_send_ns')):raise ValueError('invalid clock timestamps')
                tele.clock.add(r['laptop_send_ns'],r['remote_receive_ns'],r['remote_send_ns'],receipt)
            except (KeyError,TypeError,ValueError,OverflowError):diagnostics['invalid_clock_exchange']+=1
            continue
        if r.get('service') not in ('carState','livePose','liveLocationKalman','liveCalibration'):continue
        if not isinstance(r.get('data'),dict):
            # Raw records stay unchanged. Invalidate this source, and separate
            # evidence epochs so a bad record cannot become a driven-through label.
            diagnostics['malformed_telemetry_payloads']+=1;evidence_gap(r.get('service'))
            r={**r,'data':{},'valid':False}
        tele.accept(r,receipt)
        if r.get('service')!='carState':continue
        source=r.get('logMonoTime',0)
        if not isinstance(source,int) or source<=last_car:
            diagnostics['unordered_car_samples']+=1;epoch+=1;continue
        last_car=source
        try:
            sync=tele.clock.estimate(receipt);clock_ok=bool(sync and sync[1]<=.04e9)
            t=int(source+sync[0]) if clock_ok else receipt
        except (TypeError,ValueError,OverflowError):
            diagnostics['malformed_motion_timestamps']+=1;evidence_gap('carState');continue
        # Preserve even a brief timing-domain transition before 20 Hz thinning.
        if last_clock_mode is not None and clock_ok!=last_clock_mode:
            epoch+=1;diagnostics['clock_mode_changes_observed']+=1
        last_clock_mode=clock_ok
        d=r.get('data',{})
        try:
            v=number(d.get('vEgo'));a=number(d.get('aEgo'))
            angle=number(d.get('steeringAngleDeg'));rate=number(d.get('steeringRateDeg'))
        except OverflowError:
            diagnostics['malformed_motion_numbers']+=1;evidence_gap('carState');continue
        # Validate every preserved sample before deliberately sampling at 20 Hz;
        # a malformed intervening message is still an evidence gap.
        if samples and samples[-1].epoch==epoch and 0<=t-samples[-1].t<45_000_000:continue
        valid=bool(r.get('valid') and d.get('canValid') is True and d.get('canTimeout') is False and
                   math.isfinite(v) and 0<=v<70 and math.isfinite(a) and abs(a)<=12)
        try:ctx=tele.context(receipt) if clock_ok else {}
        except (AttributeError,KeyError,TypeError,ValueError,OverflowError):
            diagnostics['malformed_motion_contexts']+=1;epoch+=1;valid=False;ctx={}
        forward=d.get('gearShifter') in ('drive','sport','low','eco')
        motion=bool(valid and forward and ctx.get('motion_ready') and
                    ctx.get('motion_source_ns') is not None and abs(t-ctx['motion_source_ns'])<=150_000_000)
        samples.append(Sample(t,v if math.isfinite(v) else 0.,a if math.isfinite(a) else 0.,angle,rate,
                              d.get('brakePressed'),valid,clock_ok,float(ctx.get('yaw_rate_rps',0.)),motion,
                              epoch=epoch,clock_bound_s=sync[1]/1e9 if sync else 1.))
    return samples

def separate_time_epochs(samples,diagnostics):
    """Keep acquisition-order epochs; quarantine ambiguous timestamp overlaps.

    A receipt-time fallback switching to synchronized source time can step back
    slightly. Genuine resets can overlap a much larger interval. Neither is
    repaired by joining or retiming samples. All epochs sharing a timestamp
    range lose clock/motion eligibility in that range, including both sides.
    Sorting happens only after this quarantine, for timestamp-based lookup.
    """
    spans=[];epoch=-1;previous=None
    for s in samples:
        original=(s.epoch,s.clock_ok,s.t)
        source_change=previous is not None and original[0]!=previous[0]
        clock_change=previous is not None and original[1]!=previous[1]
        backwards=previous is not None and original[2]<=previous[2]
        if source_change:diagnostics['source_epoch_boundaries']+=1
        if clock_change:diagnostics['clock_mode_boundaries']+=1
        if backwards:diagnostics['timestamp_regression_boundaries']+=1
        if previous is None or source_change or clock_change or backwards:
            epoch+=1;spans.append([s.t,s.t])
        else:spans[-1][1]=s.t
        s.epoch=epoch;previous=original
    # Inclusive integer-nanosecond intervals, including duplicate endpoints.
    changes=Counter()
    for first,last in spans:changes[first]+=1;changes[last+1]-=1
    active=0;start=None;overlaps=[]
    for t,delta in sorted(changes.items()):
        old=active;active+=delta
        if old<2<=active:start=t
        elif old>=2>active:overlaps.append((start,t-1))
    starts=[a for a,b in overlaps]
    for s in samples:
        i=bisect_right(starts,s.t)-1
        if i>=0 and s.t<=overlaps[i][1]:
            s.clock_ok=False;s.motion_ok=False
            diagnostics['ambiguous_timestamp_samples']+=1
    diagnostics['time_epochs']=len(spans)
    diagnostics['ambiguous_timestamp_intervals']=len(overlaps)
    samples.sort(key=lambda s:(s.t,s.epoch))
    return samples

def clock_window_supported(samples,times,start,end,epoch):
    """A reaction lookback must stay inside one supported timing epoch."""
    first=bisect_left(times,start);last=bisect_right(times,end)
    window=samples[first:last]
    return bool(window and window[0].t-start<=250_000_000 and
                end-window[-1].t<=250_000_000 and
                all(s.clock_ok and s.epoch==epoch for s in window) and
                all(0<b.t-a.t<=200_000_000 for a,b in zip(window,window[1:])))

def build_odometry(samples):
    segment=-1;previous=None
    for s in samples:
        gap=previous is None or s.epoch!=previous.epoch or not 0<(s.t-previous.t)/1e9<=POLICY['max_sample_gap_s']
        if not s.motion_ok or not s.clock_ok:
            s.segment=-1;previous=None;continue
        if gap:
            segment+=1;s.segment=segment;s.x=s.y=s.heading=s.distance=0.
        else:
            dt=(s.t-previous.t)/1e9;speed=(s.speed+previous.speed)*.5;rate=(s.yaw+previous.yaw)*.5
            angle=previous.heading+rate*dt*.5
            s.segment=segment;s.heading=previous.heading+rate*dt
            s.x=previous.x+speed*dt*math.cos(angle);s.y=previous.y+speed*dt*math.sin(angle)
            s.distance=previous.distance+speed*dt
        previous=s

def reactions(samples):
    """Sustained vehicle responses; never infer their cause or driver intention."""
    events=[];starts={};active=None;last=None;recent=[]
    for s in samples:
        if not s.valid or (last is not None and (s.epoch!=last.epoch or not 0<(s.t-last.t)/1e9<=.2)):
            starts={};active=None;recent=[]
        if not s.valid:last=s;continue
        recent=[r for r in recent if s.t-r.t<=1_000_000_000];recent.append(s)
        drop=recent[0].speed-s.speed if s.t-recent[0].t>=700_000_000 else 0.
        rate=s.rate if math.isfinite(s.rate) else 0.
        if last and last.valid and s.epoch==last.epoch and 0<(s.t-last.t)/1e9<=.2 and math.isfinite(s.angle) and math.isfinite(last.angle):
            derived=(s.angle-last.angle)/((s.t-last.t)/1e9)
            if abs(derived)>abs(rate):rate=derived
        flags={'hard_braking':s.accel<=POLICY['hard_brake_mps2'] and s.speed>=1.,
               'sudden_steering':abs(rate)>=POLICY['steer_rate_deg_s'] and s.speed>=3.,
               'sharp_speed_drop':drop>=POLICY['sharp_drop_mps']}
        qualifies=[]
        for kind,flag in flags.items():
            if flag:starts.setdefault(kind,s.t)
            else:starts.pop(kind,None)
            duration=.20 if kind=='hard_braking' else .15 if kind=='sudden_steering' else 0.
            if flag and (s.t-starts[kind])/1e9>=duration-1e-6:qualifies.append(kind)
        if qualifies:
            onset=min(starts[k] for k in qualifies)
            if active is None or s.epoch!=active['epoch'] or s.t-active['last_response_ns']>2_000_000_000:
                active={'id':ident('reaction',onset),'at_ns':onset,'end_ns':s.t,'last_response_ns':s.t,
                        'epoch':s.epoch,'causes':[],'clock_ok':True,'peak_deceleration_mps2':0.,
                        'peak_steering_rate_deg_s':0.,'max_speed_drop_mps':0.}
                events.append(active)
            active['causes']=sorted(set(active['causes']+qualifies));active['end_ns']=s.t;active['last_response_ns']=s.t
            active['clock_ok'] &= s.clock_ok
            active['peak_deceleration_mps2']=min(active['peak_deceleration_mps2'],s.accel)
            active['peak_steering_rate_deg_s']=max(active['peak_steering_rate_deg_s'],abs(rate))
            active['max_speed_drop_mps']=max(active['max_speed_drop_mps'],drop)
        last=s
    return events

def pose_at(samples,times,t):
    i=bisect_left(times,t)
    if i<len(samples) and samples[i].t==t:
        s=samples[i];return s if s.segment>=0 else None
    if i==0 or i==len(samples):return None
    a,b=samples[i-1],samples[i]
    if a.segment<0 or a.segment!=b.segment or b.t-a.t>200_000_000:return None
    f=(t-a.t)/(b.t-a.t)
    return Sample(t,a.speed+(b.speed-a.speed)*f,0.,0.,0.,False,True,True,0.,True,
                  epoch=a.epoch,segment=a.segment,x=a.x+(b.x-a.x)*f,y=a.y+(b.y-a.y)*f,
                  heading=a.heading+(b.heading-a.heading)*f,distance=a.distance+(b.distance-a.distance)*f,
                  clock_bound_s=max(a.clock_bound_s,b.clock_bound_s))

def passed_through(candidate,samples,times,footprint):
    t=candidate['at_ns'];p=pose_at(samples,times,t)
    if p is None:return None,'no_reliable_odometry_at_decision'
    xy=candidate.get('position_m');velocity=candidate.get('ground_velocity_mps')
    if not isinstance(xy,list) or len(xy)!=2 or not np.isfinite(xy).all():return None,'unknown_obstacle_position'
    track=candidate.get('track',{})
    residual=number(track.get('residual_m'));hits=number(track.get('hits'))
    if (not isinstance(velocity,list) or len(velocity)!=2 or not np.isfinite(velocity).all() or
        np.linalg.norm(velocity)>POLICY['stationary_object_max_mps'] or track.get('association_ambiguous') or
        not math.isfinite(hits) or hits<5 or not math.isfinite(residual) or residual>.6):return None,'moving_or_uncertain_object'
    if xy[0]<=footprint['front_m']+.5:return None,'object_already_inside_or_near_footprint'
    c,s=math.cos(p.heading),math.sin(p.heading)
    wx=p.x+c*xy[0]-s*xy[1];wy=p.y+s*xy[0]+c*xy[1]
    start=bisect_left(times,t-500_000_000);end=bisect_right(times,t+int(POLICY['pass_horizon_s']*1e9))
    if start>=len(samples) or samples[start].t>t-400_000_000:return None,'insufficient_predecision_response_coverage'
    crossing=None;last=None;initial_angle=None
    for q in samples[start:end]:
        if q.segment!=p.segment or not q.valid or not q.clock_ok:return None,'motion_or_response_coverage_gap'
        if last and q.t-last.t>200_000_000:return None,'motion_or_response_coverage_gap'
        last=q
        if q.brake is not False or not math.isfinite(q.angle) or not math.isfinite(q.rate):return None,'response_fields_missing_or_brake_pressed'
        if initial_angle is None:initial_angle=q.angle
        if q.accel<POLICY['no_response_decel_mps2'] or abs(q.rate)>12 or abs(q.angle-initial_angle)>6:
            return None,'braking_or_steering_response_observed'
        if crossing and q.t>=crossing['crossed_at_ns']+500_000_000:
            crossing['response_window_ns']=[samples[start].t,q.t]
            return crossing,None
        if q.t<=t:continue
        travel=q.distance-p.distance;dt=(q.t-t)/1e9
        if travel>POLICY['pass_max_distance_m']:return None,'short_horizon_distance_exceeded'
        if q.speed<1.:continue
        # Conservative engineering allowance, NOT a calibrated probability bound.
        uncertainty=.15+.02*travel+.03*dt+max(p.speed,q.speed)*(p.clock_bound_s+q.clock_bound_s)
        inset=.2+uncertainty;half=footprint['width_m']/2-inset
        if half<=0:continue
        dx,dy=wx-q.x,wy-q.y;c,s=math.cos(q.heading),math.sin(q.heading)
        x=c*dx+s*dy;y=-s*dx+c*dy-footprint['center_y_m']
        if crossing is None and (-footprint['rear_m']+inset<x<footprint['front_m']-inset and abs(y)<half and travel>.5):
            crossing={'crossed_at_ns':q.t,'travel_m':travel,'elapsed_s':dt,'position_allowance_m':uncertainty,
                    'decision_position_local_m':[wx,wy],'trajectory_segment':p.segment,
                    'footprint_position_m':[x,y],'response_window_ns':[samples[start].t,q.t]}
    return None,'not_observed_driving_through_within_supported_horizon'

def analyze_session(folder,cancel=None):
    folder=Path(folder)
    if not (folder/'final-state.json').exists():raise ValueError('Finish recording before hindsight analysis')
    config=json.loads((folder/'session.json').read_text(encoding='utf-8-sig'))
    diagnostics=Counter();mode=config.get('capture_mode','unknown')
    footprint=config.get('hindsight_footprint',DEFAULT_FOOTPRINT)
    try:validate_footprint(footprint)
    except ValueError:footprint=DEFAULT_FOOTPRINT;diagnostics['invalid_footprint']=1
    check_work(cancel)
    samples=load_samples(folder,diagnostics,cancel) if mode in ('live','comma') else []
    separate_time_epochs(samples,diagnostics)
    build_odometry(samples);times=[s.t for s in samples];events=reactions(samples)
    frame_quality={}
    for line,r in read_jsonl(folder/'frame-contexts.jsonl',diagnostics,cancel):
        t=r.get('context',{}).get('at_ns')
        if isinstance(t,int):frame_quality[t]=(line,r.get('quality',{}))
    candidates={};decision_times=[];decision_status={};alerts=[];later_motion={}
    for line,r in read_jsonl(folder/'decisions.jsonl',diagnostics,cancel):
        t=r.get('at_ns')
        if not isinstance(t,int):continue
        decision_times.append(t)
        decision_status[t]=(r.get('status','unknown'),r.get('reasons',[]))
        for a in r.get('alerts',[]):alerts.append({'at_ns':t,'audio_delivery':r.get('audio_delivery','unknown'),'alert':a})
        tracks={v['id']:v for v in r.get('tracks',[]) if 'id' in v}
        warning_ids={a.get('track_id') for a in r.get('alerts',[])}
        for a in r.get('candidates',[])+r.get('alerts',[]):
            tid=a.get('track_id')
            if tid is None:continue
            is_warning=tid in warning_ids
            item={**a,'at_ns':t,'track':tracks.get(tid,{}),'decision_line':line,
                  'source_kind':'warning_decision' if is_warning else 'candidate',
                  'audio_delivery':r.get('audio_delivery','unknown')}
            previous=candidates.get(tid)
            if previous is None or (is_warning and previous['source_kind']!='warning_decision'):candidates[tid]=item
        for tid,track in tracks.items():
            if tid not in candidates:continue
            velocity=track.get('velocity')
            if (track.get('association_ambiguous') or (isinstance(velocity,list) and len(velocity)==2 and
                (not np.isfinite(velocity).all() or np.linalg.norm(velocity)>POLICY['stationary_object_max_mps']))):
                later_motion.setdefault(tid,[]).append(t)
    decision_times.sort();labels=[];event_specs=[]
    for event in events:
        check_work(cancel)
        t=event['at_ns'];start=t-5_000_000_000
        ix=bisect_left(decision_times,start);jx=bisect_right(decision_times,t)
        dt=decision_times[ix:jx]
        coverage=bool(dt and dt[0]-start<=250_000_000 and t-dt[-1]<=250_000_000 and
                      all(b-a<=250_000_000 for a,b in zip(dt,dt[1:])))
        preceding=[a for a in alerts if start<=a['at_ns']<=t]
        eligible=sum(decision_status[v][0]=='eligible_experimental' for v in dt)
        unavailable_reasons=Counter(reason for v in dt for reason in decision_status[v][1])
        audio=any(a['audio_delivery']=='submitted_to_windows' for a in preceding)
        unknown_audio=any(a['audio_delivery']=='unknown' for a in preceding)
        label='possible_hazard'
        reason='vehicle response; cause may be traffic, road geometry, automation or another factor'
        clock_coverage=clock_window_supported(samples,times,start,t,event['epoch'])
        if not preceding and coverage and event['clock_ok'] and clock_coverage and not diagnostics.get('malformed_decisions.jsonl'):
            label='candidate_miss';reason='no warning decision in five seconds before a vehicle response; not a confirmed hazard or miss'
        elif preceding and not audio and not unknown_audio:
            reason='warning decision present but audio was muted, failed or expired; possible delivery gap, not a detector miss'
        elif not coverage:reason+='; preceding lidar-decision coverage incomplete, so miss status is unknown'
        if not event['clock_ok']:reason+='; source clock unavailable, alignment to lidar is uncertain'
        elif not clock_coverage:reason+='; five-second lookback crosses a clock epoch or unsupported timing interval'
        row={**event,'label':label,'confidence':'low','confidence_is_probability':False,
             'reason':reason,'preceding_window_ns':[start,t],'lidar_decision_coverage_complete':coverage,
             'warning_decision_present':bool(preceding),'audio_submitted':audio,'audio_heard':'unknown',
             'warning_audio_status_unknown':unknown_audio,'requires_independent_confirmation':True,
             'preceding_window_clock_supported':clock_coverage,
             'eligible_warning_frames':eligible,'decision_frames_in_window':len(dt),
             'unavailable_reasons':dict(unavailable_reasons),
             'event_file':'auto-events/'+event['id']+'.json'}
        row.pop('last_response_ns',None);labels.append(row);event_specs.append(row)
    calibration_ok=bool(config.get('mount',{}).get('measured') and config.get('mount',{}).get('alignment_verified') and config.get('timing_verified'))
    for tid,candidate in candidates.items():
        check_work(cancel)
        if mode!='live' or not calibration_ok or not footprint.get('confirmed'):
            diagnostics['pass_through_requires_verified_mount_timing_and_footprint']+=1;continue
        if diagnostics.get('malformed_comma.jsonl') or diagnostics.get('malformed_decisions.jsonl'):
            diagnostics['pass_through_input_log_corrupt']+=1;continue
        _,q=frame_quality.get(candidate['at_ns'],(0,{}))
        if not (q.get('timestamp_valid') is True and q.get('completeness',0)>=.98 and number(q.get('timing_bound_s',1))<=.05):
            diagnostics['pass_through_scan_quality_unknown']+=1;continue
        result,reason=passed_through(candidate,samples,times,footprint)
        if result is None:diagnostics[reason]+=1;continue
        if any(candidate['at_ns']<=t<=result['crossed_at_ns'] for t in later_motion.get(tid,[])):
            diagnostics['object_moved_or_association_changed_before_passage']+=1;continue
        t=candidate['at_ns'];eid=ident('pass',t,tid)
        confidence='medium' if result['elapsed_s']<=2.5 and result['travel_m']<=10 and result['position_allowance_m']<=.35 and candidate['track'].get('hits',0)>=8 and candidate['track'].get('residual_m',99)<=.2 else 'low'
        row={'id':eid,'at_ns':t,'label':'probable_false_positive','confidence':confidence,
             'confidence_is_probability':False,'source_kind':candidate['source_kind'],'track_id':tid,
             'original_decision_line':candidate['decision_line'],'audio_delivery':candidate['audio_delivery'],
             'reason':'estimated van footprint later occupied this stationary candidate position without an observed braking or steering response',
             'caveats':['object could have moved or been mis-associated','no contact/impact ground truth','short-horizon odometry and footprint uncertainty'],
             'requires_independent_confirmation':True,'preceding_window_ns':[t-5_000_000_000,result['response_window_ns'][1]],
             'event_file':'auto-events/'+eid+'.json',**result}
        labels.append(row);event_specs.append(row)
    labels.sort(key=lambda r:(r['at_ns'],r['id']))
    for row in labels:
        row.update(schema_version=VERSION,evidence_only=True,source_session_version=config.get('version','unknown'))
    # Evidence bundles retain original decisions and exact frame references. The
    # original PCAP files remain the authoritative raw returns, without copies.
    capture_path=folder/'raw-capture-index.json'
    segments=json.loads(capture_path.read_text()).get('segments',[]) if capture_path.exists() else None
    bundles={}
    for e in event_specs:
        first,last=e['preceding_window_ns']
        relevant=[v for v in (segments or []) if v['first_receipt_ns']<=last+250_000_000 and v['last_receipt_ns']>=first-250_000_000]
        files=[v['file'] for v in relevant if (folder/v['file']).is_file()] if segments is not None else [p.name for p in sorted(folder.glob('lidar-*.pcap'))]
        status='recording_interval_partial'
        if not files:status='no_raw_files_for_window'
        elif segments is None:status='legacy_capture_intervals_unknown'
        else:
            covered=first
            for v in sorted(relevant,key=lambda v:v['first_receipt_ns']):
                if v['file'] not in files or v['first_receipt_ns']>covered+250_000_000:break
                covered=max(covered,v['last_receipt_ns'])
            if covered>=last-250_000_000:status='recording_interval_covered'
        bundles[e['id']]={'event':e,'decisions':[],'frame_references':[],
                         'raw_pcap_files':files,'raw_status':status,
                         'raw_note':'Original payloads remain in referenced session PCAPs. Interval coverage is not proof of lossless packets.',
                         'schema_version':VERSION}
    starts=sorted(event_specs,key=lambda e:e['preceding_window_ns'][0]);active=[];next_event=0
    for line,r in read_jsonl(folder/'decisions.jsonl',cancel=cancel):
        t=r.get('at_ns',0)
        while next_event<len(starts) and starts[next_event]['preceding_window_ns'][0]<=t:
            active.append(starts[next_event]);next_event+=1
        active=[e for e in active if e['preceding_window_ns'][1]>=t]
        for e in active:
            bundle=bundles[e['id']];bundle['decisions'].append({'line':line,**r})
            if t in frame_quality:
                context_line,q=frame_quality[t]
                bundle['frame_references'].append({'at_ns':t,'context_line':context_line,'first_device_ns':q.get('first_device_ns'),
                                                   'completeness':q.get('completeness'),'timestamp_valid':q.get('timestamp_valid')})
    events_dir=folder/'auto-events'
    if bundles:events_dir.mkdir(exist_ok=True)
    for eid,b in bundles.items():
        check_work(cancel);write_json(events_dir/(eid+'.json'),b)
    path=folder/'auto-labels.jsonl';temporary=path.with_suffix('.jsonl.tmp')
    with temporary.open('w',encoding='utf-8') as f:
        for row in labels:
            check_work(cancel);f.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
    temporary.replace(path)
    trajectory=folder/'hindsight-trajectory.jsonl';temporary=trajectory.with_suffix('.jsonl.tmp')
    with temporary.open('w',encoding='utf-8') as f:
        for s in samples:
            check_work(cancel)
            if s.segment>=0:f.write(json.dumps({'at_ns':s.t,'segment':s.segment,'x_m':s.x,'y_m':s.y,'heading_rad':s.heading,'distance_m':s.distance})+'\n')
    temporary.replace(trajectory)
    counts=Counter(r['label'] for r in labels)
    summary={'schema_version':VERSION,'status':'complete','phase':1,'mode':mode,
             'input_signature':input_signature(folder),
             'labels_file':'auto-labels.jsonl','manual_labels_file':'labels.jsonl','auto_label_counts':dict(counts),
             'probable_false_positive_warnings':sum(r['label']=='probable_false_positive' and r['source_kind']=='warning_decision' for r in labels),
             'candidate_misses':counts['candidate_miss'],'possible_hazards':counts['possible_hazard'],
             'reaction_events':len(events),'delivery_gaps':sum(r.get('warning_decision_present',False) and not r.get('audio_submitted') and not r.get('warning_audio_status_unknown') for r in labels),
             'candidate_misses_with_no_eligible_frames':sum(r['label']=='candidate_miss' and r.get('eligible_warning_frames',0)==0 for r in labels),
             'car_samples':len(samples),'odometry_samples':sum(s.segment>=0 for s in samples),
             'odometry_segments':len({s.segment for s in samples if s.segment>=0}),
             'odometry_source':'short-horizon wheel-speed + calibrated comma pose yaw dead reckoning; local segment coordinates',
             'footprint':footprint,'calibration_verified':calibration_ok,'diagnostics':dict(diagnostics),'policy':POLICY,
             'false_alarm_rate':'UNKNOWN; probable evidence is not confirmed ground truth',
             'miss_rate':'UNKNOWN; vehicle reactions are not a hazard inventory',
             'settings_changed':False,'labels':labels,
             'limitations':['No collision/contact sensor proof','Moving or uncertain candidates are excluded from pass-through labels',
                           'Confidence is qualitative (low or medium), never a probability or ground-truth claim','No map, parameter proposals, neural training or sync in Phase 1']}
    write_json(folder/'hindsight-review.json',summary)
    return summary

