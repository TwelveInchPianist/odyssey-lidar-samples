"""Bounded comparison to a settled road; never a substitute for a fresh fit."""
import math
import numpy as np

MAX_AGE_S=.75
MAX_TRAVEL_M=5.
MAX_TURN_RAD=math.radians(5)
MAX_HEIGHT_DIFFERENCE_M=.08
MIN_NEAR_SUPPORT=.65
PROBES=np.array([[x,y,1.] for x in (4.,8.,12.) for y in (-2.,0.,2.)])

def healthy(context,quality):
    values=(context.get('speed_mps'),context.get('yaw_rate_rps'))
    return (context.get('ready') is True and context.get('motion_ready',False) is True
            and all(isinstance(x,(int,float)) and math.isfinite(x) for x in values)
            and values[0]>=0 and quality.get('completeness',0)>=.98
            and quality.get('timestamp_valid') is True and quality.get('arrival_age_s',1)<=.2
            and not quality.get('processing_overrun',False))

def advance_reference(reference,context,quality,dt,now,discontinuity=False):
    if reference is None:return None
    if discontinuity or not .025<=dt<=.25 or not healthy(context,quality):return None
    from fcw_engine import ego_delta_origin_heading,rotate2
    movement,angle=ego_delta_origin_heading(context['speed_mps'],context['yaw_rate_rps'],dt)
    a,b,c=reference['abc'];slopes=rotate2(np.array([a,b]),-angle)
    result={**reference,'abc':[float(slopes[0]),float(slopes[1]),float(c+np.dot([a,b],movement))],
            'travel_m':reference['travel_m']+float(np.linalg.norm(movement)),
            'turn_rad':reference['turn_rad']+abs(angle)}
    if (not 0<=now-reference['at']<=MAX_AGE_S or result['travel_m']>MAX_TRAVEL_M
        or result['turn_rad']>MAX_TURN_RAD):return None
    return result

def recover_fresh(reference,ground,points):
    if reference is None or ground.get('stable_frames',0)>=5:return None
    abc=np.asarray(ground['abc']);previous=np.asarray(reference['abc'])
    difference=float(np.max(np.abs(PROBES@(abc-previous))))
    if not np.isfinite(difference) or difference>MAX_HEIGHT_DIFFERENCE_M:return None
    from fcw_engine import _near_road_share
    fresh=_near_road_share(points,abc);old=_near_road_share(points,previous)
    if fresh is None or old is None or min(fresh,old)<MIN_NEAR_SUPPORT:return None
    return {'basis':'fresh_fit_agrees_with_recent_settled_road','max_probe_difference_m':difference,
            'fresh_near_support':fresh,'reference_near_support':old,'uses_held_geometry':False}

def remember_fresh(reference,ground,context,quality,now):
    if not healthy(context,quality):return None
    if ground is not None and ground.get('stable_frames',0)>=5:
        return {'abc':list(ground['abc']),'at':now,'travel_m':0.,'turn_rad':0.}
    return reference
