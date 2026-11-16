"""Explicit user-requested dogfooding. Never represents calibration approval."""
import copy,hashlib,json,math
import numpy as np
from corridor import MODEL_PATH_REASONS
from empirical_kinematics import path

VERSION='forced-dogfood-v1'
YAW_DEADBAND_RPS=.02   # Claude fork: below this the provisional path is straight
YAW_DECAY_S=1e9        # v15 candidate (2026-10-20): yaw held CONSTANT over the forced horizon (v14 and earlier: exp(-t/1.5 s) decay).
# MEASURED on 271.7 h of Odyssey qlogs (results\misc\path-error-model\dr-constant-yaw.md): inside the horizon the constant-yaw path is better
# on every yaw/speed band (60 m p50 2.41 -> 0.70 m in the 0.05-0.15 rad/s band, straight band p90 1.01 -> 0.94 m) and removes the
# out-of-turn bias; the residual moves INTO the turn (+1.2 m p90 at 30 m), where kerbs and parked cars stand: hence the full replay set.

def decaying_path(speed,yaw,times):
    """Constant speed, yaw rate decaying as exp(-t/YAW_DECAY_S); matches empirical_kinematics.path when yaw is 0."""
    if abs(yaw)<1e-9:return path(speed,0.,times)
    dt=np.diff(times,prepend=0.);heading=np.cumsum(yaw*np.exp(-times/YAW_DECAY_S)*dt)
    x=np.cumsum(speed*np.cos(heading)*dt);y=np.cumsum(speed*np.sin(heading)*dt)
    return np.column_stack((x,y))
def finite(x):return type(x) in (int,float) and math.isfinite(x)

# Claude fork v8: the comma model path and lane lines (already in every context, whatever the camera "confidence" enum
# says) describe where the road and the driver go far better than dead reckoning at 25-60 m. Quality is gated on the
# data itself: a model path reaching >= MODEL_PATH_MIN_M with increasing x, lane lines with probability >= LANE_MIN_PROB
# on both sides and a plausible lane width. Dead reckoning (decaying yaw) stays the fallback.
LANE_MIN_PROB=.8
LANE_WIDTH_M=(2.4,5.0)
MODEL_PATH_MIN_M=15.

def _curve(ln):
    try:xs=np.asarray(ln['x'],float);ys=np.asarray(ln['y'],float);pr=float(ln.get('probability',0.))
    except Exception:return None
    if len(xs)<4 or len(xs)!=len(ys) or not (np.isfinite(xs).all() and np.isfinite(ys).all()) or np.any(np.diff(xs)<=0):return None
    return xs,ys,pr

def lane_midline(lanes,reach_m):
    """Midline of the two lane lines bracketing the van, anchored at the van's current lateral position; None if unusable."""
    left=right=None
    for ln in lanes or []:
        c=_curve(ln) if isinstance(ln,dict) else None
        if c is None or c[2]<LANE_MIN_PROB or c[0][0]>.5:continue
        y0=float(np.interp(0.,c[0],c[1]))
        if y0>0 and (left is None or y0<left[0]):left=(y0,)+c
        if y0<0 and (right is None or y0>right[0]):right=(y0,)+c
    if left is None or right is None or not LANE_WIDTH_M[0]<=left[0]-right[0]<=LANE_WIDTH_M[1]:return None
    end=min(float(reach_m),float(left[1][-1]),float(right[1][-1]))
    if end<MODEL_PATH_MIN_M:return None
    xs=np.arange(0.,end+1e-6,1.);y=(np.interp(xs,left[1],left[2])+np.interp(xs,right[1],right[2]))/2
    return np.column_stack((xs,y-y[0]))

MAP_PRIOR_FILE=None   # set by runtime from config['map_prior_file']; None = no map
MAP_CONTINUE_MIN_MPS=6.0   # v12c: below ~13 mph (residential turns) the map does not extend the dead-reckoned route
MAP_CONTINUE_MAX_YAW_RPS=0.15   # v13c: no map continuation while turning hard. Drive B 10/18 19:47 line 3392: a 0.49 rad/s corner at 14 mph continued the
                                # 4.7 m dead-reckoned stub 38 m along the map and put the inner-kerb parked cars in the corridor (passed by 5.8 m). The
                                # map path is rotated by the Kalman heading, whose lag error scales with yaw rate; the forced mode keeps its own path short in turns for the same reason.

def map_path(context,reach_m):
    """Live map prior path in the vehicle frame (anchored at the van) or (None, reason)."""
    geo=context.get('geo')
    if not MAP_PRIOR_FILE or not isinstance(geo,dict):return None,'no map file' if not MAP_PRIOR_FILE else 'no position'
    from map_live import live_map
    m=live_map(MAP_PRIOR_FILE)
    if m is None:return None,'map file unreadable'
    q=m.query(geo.get('lat'),geo.get('lon'),geo.get('heading_deg'),reach_m)
    # scalars only: JSON-safe for the context log, nothing stale to carry into a replay
    # Route-shape continuation is unchanged. Its lateral origin is not the
    # registered edge origin; never compare those two coordinate systems.
    edge_compatible=bool(q.get('edges_usable') and q.get('edges_coordinate_frame')=='forced_path_lateral')
    context['map_query']={'usable':bool(q['usable']),'stop_reason':q['stop_reason'],'reach_m':float(q['reach_m']),'n_passes':int(q['n_passes']),
                          'cell':q['cell'],'a_rel':float(q['a_rel']) if edge_compatible and finite(q.get('a_rel')) else None,
                          'b_rel':float(q['b_rel']) if edge_compatible and finite(q.get('b_rel')) else None,
                          'edges_usable':edge_compatible,'edges_basis':q.get('edges_basis'),
                          'edges_reason':q.get('edges_reason') or ('edge coordinates differ from anchored forced path' if not edge_compatible else None),
                          'path_basis':q.get('path_basis','route_shape_anchored_at_vehicle')}
    return (q['path'],None) if q['usable'] else (None,q['stop_reason'])

def camera_path(context,reach_m):
    """Comma model path where it reaches, continued along the lane midline beyond its end. (path, source) or (None, None)."""
    model=None
    try:
        p=np.asarray(context.get('path'),float).reshape(-1,2)
        if len(p)>=6 and np.isfinite(p).all() and np.all(np.diff(p[:,0])>0) and p[-1,0]>=MODEL_PATH_MIN_M and abs(p[0,1])<1.:model=p
    except Exception:model=None
    lane=lane_midline(context.get('lanes'),reach_m)
    map_path(context,reach_m)   # v12/v13b: records context['map_query']; the tail is appended to the chosen base route in prepare()
    if model is None and lane is None:return None,None
    if model is None:return lane,'comma lane-line midline (probability >= %.2f both sides)'%LANE_MIN_PROB
    if lane is None or lane[-1,0]<=model[-1,0]+1.:return model,'comma model path'
    x_end=float(model[-1,0]);shift=float(model[-1,1])-float(np.interp(x_end,lane[:,0],lane[:,1]))
    tail=lane[lane[:,0]>x_end+.5];tail=np.column_stack((tail[:,0],tail[:,1]+shift))
    return np.vstack((model,tail)),'comma model path, continued along the lane-line midline'

class ForcedExperiment:
    def __init__(self,value,epoch):
        if not isinstance(value,dict) or value.get('enabled') is not True or value.get('version')!=VERSION:
            raise ValueError('Invalid forced experimental mode')
        if value.get('acknowledgement')!='unvalidated_warning_dogfood' or not epoch or value.get('mount_epoch')!=epoch:
            raise ValueError('Forced experiment must explicitly acknowledge the current mounting period')
        from fcw_engine import Mount
        self.mount=copy.deepcopy(value['provisional_mount']);Mount(**self.mount).validate()
        if self.mount.get('measured') is not False or self.mount.get('alignment_verified') is not False:
            raise ValueError('Provisional forced geometry must remain unverified')
        for key,lo,hi in [('max_range_m',10.,40.),('horizon_s',.5,3.5),('speed_scale',.95,1.05)]:
            if not finite(value.get(key)) or not lo<=value[key]<=hi:raise ValueError('Invalid forced '+key)
        self.config=copy.deepcopy(value);self.digest=hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

    def prepare(self,context):
        out=dict(context);original=list(context.get('reasons',[]))
        bypass=[r for r in original if r in MODEL_PATH_REASONS]
        reasons=[r for r in original if r not in MODEL_PATH_REASONS]
        if context.get('motion_ready') is not True:reasons.append('Forced experiment still requires healthy vehicle motion')
        if context.get('ready') is not True and not original:reasons.append('Unexplained invalid source context')
        for key,lo,hi in [('speed_mps',0.,55.),('yaw_rate_rps',-1.5,1.5),('acceleration_mps2',-12.,12.),('clock_bound_s',0.,.04)]:
            v=context.get(key)
            if not finite(v) or not lo<=v<=hi:reasons.append('Forced experiment invalid '+key)
        source_speed=context.get('speed_mps',0.);yaw=context.get('yaw_rate_rps',0.)
        speed=source_speed*self.config['speed_scale'] if finite(source_speed) else 0.
        # Claude fork: yaw rates below the dead-band are lane-keeping noise, not a turn; the path stays straight.
        if finite(yaw) and abs(yaw)<YAW_DEADBAND_RPS:yaw=0.
        horizon=min(self.config['horizon_s'],.35/max(abs(yaw),1e-6)) if finite(yaw) else 0.
        times=np.arange(0.,horizon+1e-8,.05)
        route=decaying_path(speed,yaw,times) if finite(yaw) else np.empty((0,2))
        out.pop('map_query',None);context=dict(context);context.pop('map_query',None)   # v12b: never inherit a logged query
        chosen,path_source=camera_path(context,self.reach_m(speed))
        if 'map_query' in context:out['map_query']=context['map_query']   # v12: scalars the engine tiers candidates against
        dr_end_m=None
        if chosen is not None and len(route):route=chosen
        else:path_source='provisional constant-speed/decaying-yaw dead reckoning';dr_end_m=float(route[-1,0]) if len(route) else None
        # v13b: the map prior CONTINUES whatever base route was chosen (model path, lane midline or dead reckoning) beyond its end
        mq=context.get('map_query')
        if len(route) and mq and mq.get('usable') and (dr_end_m is None or speed>=MAP_CONTINUE_MIN_MPS) and finite(yaw) and abs(yaw)<=MAP_CONTINUE_MAX_YAW_RPS:   # v13c: not in a hard turn
            from map_live import live_map
            m=live_map(MAP_PRIOR_FILE);geo=context.get('geo') or {}
            q=m.query(geo.get('lat'),geo.get('lon'),geo.get('heading_deg'),self.reach_m(speed)) if m is not None else None
            if q and q['usable'] and q['path'][-1,0]>route[-1,0]+1.:
                x_end=float(route[-1,0]);shift=float(route[-1,1])-float(np.interp(x_end,q['path'][:,0],q['path'][:,1]))
                tail=q['path'][q['path'][:,0]>x_end+.5];route=np.vstack((route,np.column_stack((tail[:,0],tail[:,1]+shift))))
                path_source=path_source+', continued along the map prior (%d passes)'%q['n_passes']
        if speed<1 or len(route)<6 or not np.isfinite(route).all() or np.any(np.diff(route[:,0])<=0):
            reasons.append('Forced experiment motion too slow or turn too sharp for its provisional path')
        out.update(speed_mps=speed,acceleration_mps2=0.,path=route,path_horizon_m=float(route[-1,0]) if len(route) else 0.,
                   path_uncertainty='UNVERIFIED forced path: '+path_source,ready=not reasons,reasons=list(dict.fromkeys(reasons)),
                   calibration_verified=False,motion_verified=False,empirical_profile_validated=False,
                   independent_corridor={'active':False,'eligible':False,'reason':'Forced provisional experiment, not approved calibration'},
                   forced_experiment={'version':VERSION,'model_digest':self.digest,'enabled':True,'active':not reasons,
                     'label':'FORCED EXPERIMENT — CALIBRATION UNVERIFIED','max_range_m':self.config['max_range_m'],
                     'path_horizon_s':horizon,'bypassed_checks':bypass,'original_reasons':original,
                     'original_speed_mps':source_speed,'original_acceleration_mps2':context.get('acceleration_mps2'),
                     'path_source':path_source,'dr_end_m':dr_end_m,
                     'source_latency_verified':False,'physical_performance_verified':False})
        return out

    def matches(self,context):
        v=context.get('forced_experiment')
        return isinstance(v,dict) and v.get('version')==VERSION and v.get('model_digest')==self.digest

    def reach_m(self,speed_mps):
        """Detection reach: the warning horizon plus a margin, never below the configured cap, never beyond the decoder's 120 m."""
        v=float(speed_mps) if isinstance(speed_mps,(int,float)) and math.isfinite(speed_mps) else 0.
        return float(min(120.,max(self.config['max_range_m'],self.config['horizon_s']*v+15.)))

    def ground(self,value,speed_mps=None):
        if not isinstance(value,dict) or value.get('source')!='fit' or value.get('degraded'):
            return None,'Forced experiment requires a fresh settled road observation'
        return {**value,'max_range_m':min(value.get('max_range_m',120.),self.reach_m(speed_mps))},None

def configured(config):
    value=config.get('forced_experiment')
    if value is None or isinstance(value,dict) and value.get('enabled') is False:return None
    return ForcedExperiment(value,config.get('mount_epoch'))
