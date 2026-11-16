"""Parked known-target workflow; does not ask the driver to interpret a cloud."""
from collections import deque
from dataclasses import asdict
import math
import numpy as np
from calibration import planar_alignment,euler
from telemetry import rotation

class TargetCalibration:
    def __init__(self):
        self.history={};self.baseline=None;self.pairs=[];self.proposal=None;self.verified=None
        self.stationary=False;self.last_ns=0

    def observe(self,result,context):
        self.stationary=context.get('stationary_confirmed',False);self.last_ns=context['at_ns']
        if not self.stationary:self.history.clear();return
        live=set()
        for t in result.get('tracks',[]):
            if t['misses'] or t['points']<12:continue
            live.add(t['id']);self.history.setdefault(t['id'],deque(maxlen=20)).append(t['position'])
        self.history={k:v for k,v in self.history.items() if k in live}

    def centers(self):
        stable=[]
        for history in self.history.values():
            if len(history)<15:continue
            a=np.asarray(history);center=np.median(a,axis=0)
            if np.max(np.linalg.norm(a-center,axis=1))<.12:stable.append(center)
        return stable

    def set_baseline(self,now_ns):
        self.check_stationary(now_ns)
        self.baseline=self.centers();self.pairs=[];self.proposal=None;self.verified=None
        return {'message':'Empty-scene baseline saved. Place one upright box at a measured target position.'}

    def check_stationary(self,now_ns):
        if not self.stationary or now_ns-self.last_ns>500_000_000:
            raise ValueError('Need fresh stationary vehicle data. Park on level pavement; ignition on.')

    def capture(self,reference,now_ns,check=False):
        self.check_stationary(now_ns)
        if self.baseline is None:raise ValueError('Capture the empty scene before placing the target')
        ref=np.asarray(reference,float)
        if ref.shape!=(2,) or not np.isfinite(ref).all() or not 6<=ref[0]<=35 or abs(ref[1])>5:
            raise ValueError('Target must be 6–35 m forward and within 5 m laterally of the comma ground origin')
        choices=[]
        for p in self.centers():
            if any(np.linalg.norm(p-b)<.8 for b in self.baseline):continue
            if np.linalg.norm(p-ref)<4:choices.append(p)
        if len(choices)!=1:raise ValueError('Need exactly one stable new target near that position; wait, remove clutter, or use a larger box')
        point=choices[0]
        if check:
            if not self.proposal or not self.proposal['accepted']:raise ValueError('Capture three well-spread targets first')
            if any(np.linalg.norm(ref-pair['reference'])<1 for pair in self.pairs):raise ValueError('Use a fourth position at least 1 m from the fitting targets')
            theta=math.radians(self.proposal['rotation_deg']);R=rotation([0,0,theta])[:2,:2]
            predicted=R@point+self.proposal['translation_xy_m'];error=float(np.linalg.norm(predicted-ref))
            self.verified={'accepted':error<.2,'independent_error_m':error,'reference':ref.tolist(),'measured':point.tolist()}
            return self.verified
        if any(np.linalg.norm(ref-pair['reference'])<1 for pair in self.pairs):raise ValueError('Move the target at least 1 m from an earlier position')
        self.pairs.append({'measured':point.tolist(),'reference':ref.tolist()})
        if len(self.pairs)>=3:self.proposal=planar_alignment([p['measured'] for p in self.pairs],[p['reference'] for p in self.pairs])
        return {'targets':len(self.pairs),'proposal':self.proposal}

    def resulting_mount(self,mount):
        if not self.verified or not self.verified['accepted']:raise ValueError('Independent fourth-target check has not passed')
        p=self.proposal;R=rotation([0,0,math.radians(p['rotation_deg'])])
        out=asdict(mount);t=R@mount.xyz;t[:2]+=p['translation_xy_m']
        out['xyz']=t.tolist();out['rpy_deg']=np.degrees(euler(R@rotation(np.radians(mount.rpy_deg)))).tolist()
        out['measured']=True;out['alignment_verified']=True
        return out

    def snapshot(self):
        return {'baseline_captured':self.baseline is not None,'targets':len(self.pairs),'proposal':self.proposal,
                'independent_check':self.verified,'stationary':self.stationary}
