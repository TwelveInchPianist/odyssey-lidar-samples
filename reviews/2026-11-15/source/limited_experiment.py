"""Explicit limits for an ordinary-drive calibrated experimental FCW mode.

This module grants no calibration authority. A valid, source-checked empirical
profile is required before an audible candidate can be eligible. Provisional
mount geometry may improve perception while the existing approval gates stay on.
"""
import math

VERSION='limited-experimental-fcw-v1'
LIMITS={'min_speed_mps':6.7056,'max_speed_mps':13.4112,'max_abs_yaw_rps':.04,
        'max_abs_acceleration_mps2':1.5,'max_abs_steering_rate_deg_s':15.,
        'max_range_m':40.,'max_horizon_s':2.,'min_horizon_s':1.}

def settings(value):
    if value is None:return None
    if not isinstance(value,dict):raise ValueError('Malformed limited experimental settings')
    if value.get('enabled') is False:return None
    if value.get('enabled') is not True:raise ValueError('Explicit experimental enable choice required')
    if value.get('version')!=VERSION:raise ValueError('Unknown limited experimental mode')
    # Each release has one documented scope. Configuration edits cannot widen it.
    if value.get('limits')!=LIMITS:raise ValueError('Limited experimental operating limits changed')
    if value.get('audible_requested') not in (True,False):raise ValueError('Explicit audio choice required')
    return value

def prepare_context(context, value, *, empirical_trusted=False):
    out=dict(context);reasons=[]
    try:cfg=settings(value)
    except ValueError as e:cfg=value if isinstance(value,dict) else {};reasons.append(str(e))
    if cfg is None:return out
    if not empirical_trusted or context.get('empirical_profile_validated') is not True:
        reasons.append('Experimental calibration has not passed its evidence checks')
    for key,lo,hi in [('speed_mps',LIMITS['min_speed_mps'],LIMITS['max_speed_mps']),
                      ('yaw_rate_rps',-LIMITS['max_abs_yaw_rps'],LIMITS['max_abs_yaw_rps']),
                      ('steering_rate_deg_s',-LIMITS['max_abs_steering_rate_deg_s'],LIMITS['max_abs_steering_rate_deg_s'])]:
        v=context.get(key)
        if type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi:
            reasons.append('Outside limited experimental '+key+' coverage')
    acc=context.get('empirical_observed_acceleration_mps2',context.get('acceleration_mps2'))
    if type(acc) not in (int,float) or not math.isfinite(acc) or abs(acc)>LIMITS['max_abs_acceleration_mps2']:
        reasons.append('Outside limited experimental acceleration coverage')
    horizon=context.get('warning_horizon_s')
    if type(horizon) not in (int,float) or not math.isfinite(horizon) or not LIMITS['min_horizon_s']<=horizon<=LIMITS['max_horizon_s']:
        reasons.append('No supported one-to-two-second experimental warning horizon')
    out['limited_experiment']={'version':VERSION,'enabled':True,'eligible':not reasons,
        'limits':dict(LIMITS),'reasons':reasons,'audible_requested':cfg.get('audible_requested') is True,
        'calibration_authority_granted':False,'physical_collision_recall_verified':False,
        'message':'Experimental 15–30 mph, shallow-turn coverage; at most 40 m and two seconds. No highway or animal-detection claim.'}
    if reasons:out['ready']=False;out['reasons']=list(dict.fromkeys(list(out.get('reasons',[]))+reasons))
    return out

def constrain_ground(ground,context):
    """Apply scope before clustering, not by discarding already-issued alerts."""
    mode=context.get('limited_experiment')
    if not isinstance(mode,dict) or mode.get('enabled') is not True:return ground,None
    if not isinstance(ground,dict) or ground.get('source')!='fit' or ground.get('degraded'):
        return None,'Experimental mode requires a fresh observed road plane'
    out=dict(ground);out['max_range_m']=min(float(out.get('max_range_m',LIMITS['max_range_m'])),LIMITS['max_range_m'])
    out['experimental_range_cap_m']=LIMITS['max_range_m']
    return out,None

def allow_audio(config,integrity):
    value=config.get('limited_experiment')
    try:mode=settings(value)
    except ValueError:return False
    if mode is None:return bool(config.get('audio_enabled'))
    return bool(config.get('audio_enabled') and mode['audible_requested'] and integrity.get('trusted') is True)
