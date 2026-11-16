"""Clock-aware, read-only comma telemetry normalization.

Internal FCW coordinates are forward/left/up (FLU). Comma's calibrated path is
forward/right/down (FRD). No raw IMU axis shortcuts or receipt-time-only fusion.
"""
from collections import deque
import math
import numpy as np

def number(value):
    try:return float(value)
    except (ValueError,TypeError):return float('nan')

def rotation(rpy):
    r,p,y = rpy
    cr,sr,cp,sp,cy,sy = math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                     [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])

class ClockSync:
    """NTP-style bounded offset using monotonic timestamps over SSH.

    Interval bounds assume nonnegative transit and identical clock rates over one
    exchange; drift allowance grows with sample age. Minimum RTT in a 30s window.
    Values are nanoseconds. A stream restart invalidates all estimates.
    """
    def __init__(self):
        self.samples = deque(maxlen=120)

    def add(self, local_send, remote_receive, remote_send, local_receive):
        if not (local_receive >= local_send and remote_send >= remote_receive):
            return False
        lower,upper = local_send-remote_receive, local_receive-remote_send
        if upper < lower or upper-lower > 2_000_000_000:
            return False
        self.samples.append((local_receive,(lower+upper)/2,(upper-lower)/2))
        return True

    def estimate(self, now_ns):
        recent=[s for s in self.samples if 0 <= now_ns-s[0] <= 30_000_000_000]
        if len(recent)<3:
            return None
        stamp,offset,bound=min(recent,key=lambda s:(s[2],-s[0]))
        # 100 ppm allowance, pending actual device drift measurement.
        return offset, bound+(now_ns-stamp)*0.0001

class Telemetry:
    def __init__(self):
        self.clock=ClockSync()
        self.messages={}
        self.history={}
        self.commit=None
        self.rejected=0
        self.selected_times={}
        self.selected_records={}

    def accept(self, item, receipt_ns):
        kind=item.get('kind')
        if kind=='hello':
            self.messages.clear();self.history.clear();self.clock=ClockSync();self.commit=item.get('commit')
            self.selected_records.clear()
        elif kind=='sample':
            service=item.get('service')
            stamp=item.get('logMonoTime',0)
            prior=self.messages.get(service)
            if not isinstance(stamp,int) or stamp<=0 or (prior and stamp<=prior['logMonoTime']):
                self.rejected+=1;return
            self.messages[service]={**item,'receipt_ns':receipt_ns}
            self.history.setdefault(service,deque(maxlen=200)).append(self.messages[service])

    def fresh(self, service, at_ns, max_age=.25):
        sync=self.clock.estimate(at_ns)
        if sync is None:
            return None
        item=next((m for m in reversed(self.history.get(service,[])) if m['logMonoTime']+sync[0]<=at_ns+sync[1]+10_000_000),None)
        if not item or not item.get('valid'):return None
        stamp=item['logMonoTime']+sync[0]
        age=(at_ns-stamp)/1e9
        bound=sync[1]/1e9
        if bound>.04 or age < -bound-.01 or age+bound>max_age or at_ns-item['receipt_ns']>max_age*1e9:
            return None
        self.selected_times[service]=int(stamp)
        self.selected_records[service]=item
        return item['data']

    def context(self, at_ns):
        reasons=[]
        sync=self.clock.estimate(at_ns)
        if sync is None or sync[1]>.04e9:reasons.append('comma clock not bounded within 40 ms')
        car=self.fresh('carState',at_ns)
        model=self.fresh('modelV2',at_ns)
        calib=self.fresh('liveCalibration',at_ns,2.0)
        pose=self.fresh('livePose',at_ns)
        if not car:reasons.append('speed/vehicle state missing or stale')
        if not model:reasons.append('predicted path missing or stale')
        if not calib or calib.get('calStatus') not in ('calibrated',1):
            reasons.append('comma calibration not valid')
        speed=number((car or {}).get('vEgo',0))
        speed_valid=math.isfinite(speed) and 0<=speed<70
        if not speed_valid:reasons.append('invalid vehicle speed');speed=0
        acceleration=number((car or {}).get('aEgo',0))
        if not math.isfinite(acceleration) or abs(acceleration)>12:
            reasons.append('invalid vehicle acceleration');acceleration=0
        if car and (not car.get('canValid') or car.get('canTimeout')):reasons.append('vehicle CAN inputs invalid')
        if car and car.get('gearShifter') not in ('drive','sport','low','eco'):
            reasons.append('vehicle not in a forward driving gear')
        yaw_rate=None
        motion_source_ns=None
        motion_service=None
        if pose and calib and pose.get('inputsOK') and pose.get('sensorsOK') and pose.get('posenetOK'):
            av=pose.get('angularVelocityDevice',{})
            if av.get('valid') and number(av.get('zStd',99))<.15:
                v=np.array([av.get(k,float('nan')) for k in ('x','y','z')],dtype=float)
                rpy=np.asarray(calib.get('rpyCalib',[]),dtype=float)
                if len(rpy)==3 and np.isfinite(v).all() and np.isfinite(rpy).all():
                    # Source: locationd's device_from_calib = rot_from_euler(rpy).
                    # Rotation vector is a proper axial vector; FRD -> FLU is Rx(pi).
                    yaw_rate=-float((rotation(rpy).T@v)[2])
                    motion_source_ns=self.selected_times.get('livePose')
                    motion_service='livePose'
        if yaw_rate is None:
            ll=self.fresh('liveLocationKalman',at_ns)
            av=(ll or {}).get('angularVelocityCalibrated',{})
            if (ll and ll.get('status') in ('valid',2) and ll.get('inputsOK') and ll.get('sensorsOK') and
                av.get('valid') and len(av.get('value',[]))==3 and np.isfinite(np.asarray(av['value'],dtype=float)).all()):
                yaw_rate=-float(av['value'][2])
                motion_source_ns=self.selected_times.get('liveLocationKalman')
                motion_service='liveLocationKalman'
        if yaw_rate is None or not math.isfinite(yaw_rate) or abs(yaw_rate)>1.5:
            reasons.append('calibrated yaw rate missing or implausible');yaw_rate=0
        motion_ready=not any(r not in ('predicted path missing or stale','vehicle not in a forward driving gear') for r in reasons)
        stationary=bool(car and car.get('canValid') and not car.get('canTimeout') and speed_valid and speed<.2)
        if car and car.get('gearShifter') not in ('drive','sport','low','eco') and not stationary:motion_ready=False
        path=[];lanes=[];vision_leads=[];geo=None
        llg=self.fresh('liveLocationKalman',at_ns)   # Claude fork v12: position + heading for the map prior
        try:
            pg=(llg or {}).get('positionGeodetic',{});on=(llg or {}).get('calibratedOrientationNED',{})
            if llg and llg.get('status') in ('valid',2) and pg.get('valid') and on.get('valid') and len(pg.get('value',[]))>=2 and len(on.get('value',[]))==3:
                geo={'lat':float(pg['value'][0]),'lon':float(pg['value'][1]),'heading_deg':float(np.degrees(on['value'][2]))%360.,'source':'liveLocationKalman'}
                if not all(np.isfinite(v) for v in (geo['lat'],geo['lon'],geo['heading_deg'])):geo=None
        except Exception:geo=None
        if model:   # Claude fork v11: comma vision leads (leadsV3) for association; prob/x/y/v at t=0
            for ld in (model.get('leadsV3') or [])[:3]:
                try:
                    pr=float(ld.get('prob',0.));xs=ld.get('x') or [];ys=ld.get('y') or [];vs=ld.get('v') or []
                    if xs and ys and vs:vision_leads.append({'prob':pr,'x':float(xs[0]),'y':-float(ys[0]),'v':float(vs[0])})
                except Exception:pass
        path_uncertainty='unavailable';path_horizon=0.
        if model:
            position=model.get('position',{})
            x,y=position.get('x',[]),position.get('y',[])
            if len(x)>=6 and len(x)==len(y):
                path=np.column_stack((x,-np.asarray(y,dtype=float))).astype(float)
                if not np.isfinite(path).all() or np.any(np.diff(path[:,0])<=0) or path[-1,0]<10:
                    reasons.append('invalid path geometry');path=[]
                else:
                    std=np.asarray(position.get('yStd',[]),dtype=float)
                    if len(std)!=len(path) or not np.isfinite(std).all() or np.any(std<0):
                        reasons.append('path lateral uncertainty missing or invalid')
                    else:
                        uncertain=np.flatnonzero(std>.5)
                        if len(uncertain):path=path[:int(uncertain[0])]
                        path_uncertainty='limited to predicted lateral std <= 0.5 m'
                        if len(path)<6 or path[-1,0]<10:
                            reasons.append('insufficient confident path horizon');path=[]
                        else:path_horizon=float(path[-1,0])
            else:reasons.append('insufficient path geometry')
            confidence=model.get('confidence')
            if ((isinstance(confidence,str) and confidence in ('low','red')) or
                (type(confidence) is int and confidence==0)):
                reasons.append('low model confidence')
            elif not ((isinstance(confidence,str) and confidence in ('yellow','green')) or
                      (type(confidence) is int and confidence in (1,2))):
                # Missing or unknown schema values must not silently pass the
                # native red/yellow/green confidence gate (enum ordinals 0/1/2).
                reasons.append('model confidence missing or invalid')
            # timestampEof catches stale camera frames even when inference messages are fresh.
            eof=model.get('timestampEof',0)
            if not eof or sync is None or not -sync[1]/1e9-.01 <= (at_ns-(eof+sync[0]))/1e9 <= .35-sync[1]/1e9:
                reasons.append('camera frame missing or stale')
            drop=number(model.get('frameDropPerc',100))
            if not math.isfinite(drop) or drop>30:reasons.append('excessive or unknown camera frame drops')
            for lane,prob in zip(model.get('laneLines',[]),model.get('laneLineProbs',[])):
                lx,ly=lane.get('x',[]),lane.get('y',[])
                if number(prob)>.6 and len(lx)==len(ly) and len(lx)>2 and np.isfinite(np.asarray(lx+ly,dtype=float)).all():
                    lanes.append({'x':lx,'y':[-v for v in ly],'probability':prob})
        # Additional measured evidence for the independently commissioned corridor.
        # Never promote the Honda interface's unpopulated steering-rate zero.
        steering_angle=None;steering_rate=None;steering_source=None
        if sync is not None:
            recent=[]
            for item in reversed(self.history.get('carState',[])):
                stamp=int(item['logMonoTime']+sync[0]);data=item.get('data',{})
                if stamp>at_ns or at_ns-stamp>250_000_000 or item.get('receipt_ns',at_ns+1)>at_ns:continue
                if not isinstance(data,dict):break
                try:angle=number(data.get('steeringAngleDeg'))
                except OverflowError:angle=float('nan')
                if not (item.get('valid') and data.get('canValid') is True and data.get('canTimeout') is False and math.isfinite(angle) and abs(angle)<=1080):break
                recent.append((stamp,angle))
                if len(recent)==2:break
            if len(recent)==2:
                delta=(recent[0][0]-recent[1][0])/1e9
                if .005<=delta<=.25:
                    rate=(recent[0][1]-recent[1][1])/delta
                    if math.isfinite(rate) and abs(rate)<=1500:
                        steering_source,steering_angle=recent[0];steering_rate=rate
        return {'at_ns':at_ns,'speed_mps':speed,'acceleration_mps2':acceleration,'yaw_rate_rps':yaw_rate,
                'brake_pressed':(bool(car.get('brakePressed')) if car and 'brakePressed' in car else None),
                'speed_source_ns':self.selected_times.get('carState') if car else None,
                'speed_native_source_ns':self.selected_records.get('carState',{}).get('logMonoTime') if car else None,
                'speed_receipt_ns':self.selected_records.get('carState',{}).get('receipt_ns') if car else None,
                'motion_native_source_ns':self.selected_records.get(motion_service,{}).get('logMonoTime') if motion_service else None,
                'motion_receipt_ns':self.selected_records.get(motion_service,{}).get('receipt_ns') if motion_service else None,
                'speed_measurement_kind':'carState.vEgo; vehicle-interface speed estimate, not assumed axle speed',
                'steering_source_ns':steering_source,'steering_angle_deg':steering_angle,
                'steering_rate_deg_s':steering_rate,'steering_valid':steering_rate is not None,
                'stationary_confirmed':stationary,'motion_ready':motion_ready or stationary,
                'motion_source_ns':motion_source_ns,
                'path':np.asarray(path).reshape(-1,2), 'lanes':lanes,'vision_leads':vision_leads,'geo':geo,
                'path_uncertainty':path_uncertainty,'path_horizon_m':path_horizon,
                'clock_bound_s':sync[1]/1e9 if sync else None,
                'ready':not reasons,'reasons':reasons}
