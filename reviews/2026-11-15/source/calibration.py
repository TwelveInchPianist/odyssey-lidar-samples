"""Calibration proposals with explicit observability/acceptance diagnostics."""
from dataclasses import asdict
import math
import numpy as np
from telemetry import rotation

def level_rotation(normal):
    n=np.array(normal,dtype=float);n/=np.linalg.norm(n);z=np.array([0.,0.,1.])
    v=np.cross(n,z);c=float(n@z)
    K=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+K+K@K/(1+c)

def euler(R):
    return np.array([math.atan2(R[2,1],R[2,2]),math.asin(float(np.clip(-R[2,0],-1,1))),math.atan2(R[1,0],R[0,0])])

def ground_proposal(mount,observations):
    valid=[o for o in observations if o and not o.get('degraded') and o.get('rms_m',float('inf'))<.08 and o.get('support',0)>=50]
    if len(valid)<30:return {'accepted':False,'reason':'collect at least 30 well-supported parked ground scans'}
    planes=np.array([o['abc'] for o in valid]);abc=np.median(planes,axis=0)
    spread=np.max(np.abs(planes-abc),axis=0)
    if spread[0]>.015 or spread[1]>.015 or spread[2]>.08:
        return {'accepted':False,'reason':'ground estimates are not stable enough','spread':spread.tolist()}
    a,b,c=abc;n=np.array([-a,-b,1.]);height=float((n@np.asarray(mount.xyz)-c)/np.linalg.norm(n))
    R=level_rotation(n)@rotation(np.radians(mount.rpy_deg))
    proposal=asdict(mount);proposal['xyz'][2]=height;proposal['rpy_deg']=np.degrees(euler(R)).tolist()
    proposal['measured']=mount.measured;proposal['alignment_verified']=False
    return {'accepted':True,'mount_proposal':proposal,'height_m':height,'plane_spread':spread.tolist(),
            'note':'Height/pitch/roll proposal only. Flat parked surface required. Yaw, XY and timing remain unverified.'}

def planar_alignment(measured_xy,reference_xy):
    """Known target correspondences -> 2D rigid calibration.

    measured_xy in current leveled lidar frame; reference in camera-ground FLU.
    Caller must establish target correspondence independently, never nearest-fit it.
    """
    a,b=np.asarray(measured_xy,float),np.asarray(reference_xy,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=2 or len(a)<3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {'accepted':False,'reason':'need at least three finite matching target coordinates'}
    ac,bc=a-a.mean(0),b-b.mean(0)
    if np.linalg.svd(bc,compute_uv=False)[-1]<.4:
        return {'accepted':False,'reason':'targets must span longitudinal and lateral directions'}
    U,_,Vt=np.linalg.svd(ac.T@bc);R=Vt.T@U.T
    if np.linalg.det(R)<0:Vt[-1]*=-1;R=Vt.T@U.T
    translation=b.mean(0)-a.mean(0)@R.T
    residual=np.linalg.norm(a@R.T+translation-b,axis=1)
    return {'accepted':bool(residual.max()<.15),'rotation_deg':math.degrees(math.atan2(R[1,0],R[0,0])),
            'translation_xy_m':translation.tolist(),'residual_m':residual.tolist(),
            'reason':'fit residual exceeds 15 cm' if residual.max()>=.15 else 'independent check target still required'}

def timing_alignment(lidar_times,lidar_yaw_rates,comma_times,comma_yaw_rates):
    """Time shift from independently measured scan-registration yaw rate.

    Positive offset: comma_time = lidar_time + offset. Lidar yaw must not be
    derived from comma: that would make this circular validation. Proposal only.
    """
    lt,ly,ct,cy=map(lambda v:np.asarray(v,float),(lidar_times,lidar_yaw_rates,comma_times,comma_yaw_rates))
    if min(len(lt),len(ct))<100 or len(lt)!=len(ly) or len(ct)!=len(cy) or not all(np.isfinite(a).all() for a in (lt,ly,ct,cy)):
        return {'accepted':False,'reason':'need finite independent yaw series (at least 100 samples each)'}
    if np.any(np.diff(lt)<=0) or np.any(np.diff(ct)<=0) or lt[-1]-lt[0]<15:
        return {'accepted':False,'reason':'need ordered samples covering at least 15 seconds'}
    if np.std(ly)<.02 or np.std(cy)<.02:return {'accepted':False,'reason':'not enough safe turning variation to identify timing'}
    shifts=np.arange(-.5,.501,.005);scores=[]
    for shift in shifts:
        valid=(lt+shift>=ct[0])&(lt+shift<=ct[-1])
        if valid.sum()<100:scores.append(-1);continue
        y=np.interp(lt[valid]+shift,ct,cy)
        scores.append(float(np.corrcoef(ly[valid],y)[0,1]))
    scores=np.array(scores);best=int(np.argmax(scores));shift=float(shifts[best])
    far=np.abs(shifts-shift)>.08
    separated=scores[best]-max(scores[far])>.015
    accepted=bool(scores[best]>.9 and separated and best not in (0,len(shifts)-1))
    near=shifts[scores>=scores[best]-.002]
    peak_half_width=float(max(abs(near-shift))) if len(near) else .5
    accepted=bool(accepted and peak_half_width<=.04)
    return {'accepted':accepted,'offset_s':shift,'correlation':float(scores[best]),
            'reason':'candidate only; validate on independent maneuver' if accepted else 'ambiguous or weak timing correlation',
            'empirical_peak_half_width_s':peak_half_width,
            'search_step_s':.005}

def combine_timing_reviews(reviews):
    """Require two independent, compatible sessions and a narrow consistent peak.

    The returned error is an empirical estimate, not a hardware timestamp bound.
    Keep both evidence files; this does not verify FCW performance.
    """
    usable=[]
    for r in reviews:
        p=r.get('timing_proposal',{})
        if p.get('accepted') and p.get('empirical_peak_half_width_s',1)<=.04:
            usable.append(r)
    for i,new in enumerate(usable):
        for old in usable[i+1:]:
            if new.get('session')==old.get('session') or new.get('mount')!=old.get('mount'):continue
            if new.get('comma_commit')!=old.get('comma_commit') or not new.get('comma_commit'):continue
            a=new['total_time_adjustment_s'];b=old['total_time_adjustment_s']
            discrepancy=abs(a-b)
            uncertainty=max(new['timing_proposal']['empirical_peak_half_width_s'],old['timing_proposal']['empirical_peak_half_width_s'])+discrepancy/2+.01
            if discrepancy<=.02 and uncertainty<=.05:
                return {'accepted':True,'adjustment_s':(a+b)/2,'estimated_error_s':uncertainty,
                        'sessions':[new['session'],old['session']], 'mount':new['mount'],
                        'note':'Repeatable empirical timing alignment only; systematic error and FCW performance remain to be validated.'}
    return {'accepted':False,'reason':'Need two independent sessions with the same mount/firmware and consistent, narrow timing estimates'}
