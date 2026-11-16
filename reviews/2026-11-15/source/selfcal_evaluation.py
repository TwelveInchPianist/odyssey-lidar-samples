"""Frozen-parameter cross-session diagnostics. Never fits or approves a profile."""
import math
import numpy as np
from selfcal_motion import _windows,_comma_groups,_channel_series,_causal,_body_transform,_bounded_case_max
from selfcal_motion3d import fraction
from empirical_kinematics import path,lever

def evaluate_frozen_motion(capture,geometry,motion):
    result={'available':False,'parameters_refitted':False,'eligible':False,'diagnostic_only':True,'horizons':[]}
    if not geometry or motion.get('timing',{}).get('available') is not True or motion.get('motion',{}).get('available') is not True:
        return dict(result,reason='Development geometry/timing/motion parameters unavailable; no evaluation parameters invented.')
    m=motion['motion'];offset=motion['timing']['offset_s'];L=lever(m)
    groups,_,_=_comma_groups(capture['comma_rows']);native={k:{ch:_channel_series(v,ch) for ch in ('speed','yaw')} for k,v in groups.items()}
    windows=_windows(capture['motion_rows']);result['available']=True
    for horizon in (.5,1.,1.5,2.):
        errors=[];bounds=[];cases=[]
        for window in windows:
            rows=window['rows'];channels=native.get(window['key'],{})
            if not all(channels.get(k,{}).get('valid') for k in ('speed','yaw')):continue
            for i in range(0,len(rows),5):
                start=rows[i]['start_ns'];query=start+int(round(offset*1e9))
                obs=[_causal(channels[k],query) for k in ('speed','yaw')]
                if any(v is None for v in obs):continue
                speed,yaw=obs[0]-m['speed_bias_mps'],obs[1]-m['yaw_bias_rps']
                if speed<.5:continue
                R=np.eye(3);t=np.zeros(3);covered=0.;error=0.;angular_error=0.
                for row in rows[i:]:
                    if row['start_ns']!=start+int(round(covered*1e9)) or not row.get('ego_increment_se3'):break
                    dt=min(row['dt_s'],horizon-covered);f=dt/row['dt_s'];raw=row['ego_increment_se3']
                    r,p=fraction(np.array(raw['rotation']),np.array(raw['translation_m']),f)
                    t+=R@p;R=R@r
                    q=row['quality'];ae=q.get('rotation_error_rad',q['yaw_error_rad'])*f
                    error+=q['translation_error_m']*f+np.linalg.norm(p)*min(2.,angular_error)+np.linalg.norm(geometry['sensor_origin_camera_flu_m'])*ae
                    angular_error+=ae;covered+=dt
                    if covered>=horizon-1e-8:break
                if covered<horizon-1e-8:continue
                _,reference=_body_transform({'ego_increment_se3':{'rotation':R,'translation_m':t}},geometry)
                residual=np.abs(path(speed,yaw,[horizon],L)[0]-reference[:2]);errors.append(residual);bounds.append(error)
                cases.append({'start_ns':start,'raw_error_xy_m':residual.tolist(),'reference_error_m':float(error)})
        maximum=_bounded_case_max(errors,bounds)
        result['horizons'].append({'horizon_s':horizon,'samples':len(errors),'bounded_error_xy_m':None if maximum is None else maximum.tolist(),
            'raw_max_error_xy_m':None if not errors else np.max(errors,axis=0).tolist(),
            'worst_cases':sorted(cases,key=lambda c:max(c['raw_error_xy_m'])+c['reference_error_m'],reverse=True)[:5]})
    result['scope']='All evaluable windows; frozen development geometry, clock offset, biases and kinematic lever. No later-session fitting or activation.'
    return result
