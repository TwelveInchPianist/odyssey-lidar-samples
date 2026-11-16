"""Short-lived ego-compensated echo suppression for a previously warned surface.

No scene map or semantic class veto. A different/deeper/moving hazard keeps
normal alert authority. A new track ID alone is not a new physical obstacle.
"""
import math
import numpy as np

def shallow(c):
    gap=(c.get('corridor') or {}).get('min_body_gap_m')
    return isinstance(gap,(int,float)) and math.isfinite(gap) and -.1<gap<=0

def remember_surface(c,track,now):
    motion=c.get('motion_evidence') or {}
    if (track is None or not shallow(c) or not motion.get('trusted') or c.get('oncoming_candidate')
        or np.linalg.norm(c.get('ground_velocity_mps',[math.inf,math.inf]))>1.):return None
    return {'track_id':c['track_id'],'direction':c['direction'],'at':now,
            'position':track.position.tolist(),'lo':track.lo[:2].tolist(),'hi':track.hi[:2].tolist()}

def advance_surface(state,context,quality,dt,now):
    if state is None:return None
    from ground_recovery import healthy
    if not healthy(context,quality) or not .025<=dt<=.25 or not 0<=now-state['at']<=3.:return None
    from fcw_engine import ego_delta,rotate2
    move,angle=ego_delta(context['speed_mps'],context['yaw_rate_rps'],dt)
    lo=state['lo'];hi=state['hi'];corners=np.array([[lo[0],lo[1]],[lo[0],hi[1]],[hi[0],lo[1]],[hi[0],hi[1]]])
    corners=rotate2(corners-move,-angle)
    return {**state,'position':rotate2(np.asarray(state['position'])-move,-angle).tolist(),
            'lo':corners.min(axis=0).tolist(),'hi':corners.max(axis=0).tolist()}

def repeat_surface(state,c,track,now,previous_track=None):
    # Concurrent fresh observations are distinct candidates, not an ID handoff.
    if previous_track is not None and previous_track.misses==0:return False
    if (state is None or c['track_id']==state['track_id'] or not shallow(c)
        or c.get('direction')!=state['direction'] or c.get('oncoming_candidate')
        or (c.get('motion_evidence') or {}).get('trusted') or not 0<=now-state['at']<=3.):return False
    if np.linalg.norm(track.position-np.asarray(state['position']))>1.:return False
    lo=np.maximum(track.lo[:2],state['lo']);hi=np.minimum(track.hi[:2],state['hi'])
    intersection=float(np.prod(np.maximum(hi-lo,0)))
    minimum_area=min(float(np.prod(np.maximum(track.hi[:2]-track.lo[:2],0))),float(np.prod(np.maximum(np.asarray(state['hi'])-state['lo'],0))))
    if minimum_area<=0 or intersection/minimum_area<.5:return False
    c['repeat_surface']={'previous_track_id':state['track_id'],'age_s':now-state['at'],
                         'basis':'ego_compensated_overlap_with_previously_warned_stationary_surface'}
    return True
