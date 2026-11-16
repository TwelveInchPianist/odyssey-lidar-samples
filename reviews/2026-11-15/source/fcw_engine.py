"""Experimental geometric FCW engine. Not validated road-safety equipment.

Processes full scans, estimates local ground, tracks clusters with ego motion,
and evaluates future overlap with a curved swept path. All decisions are logged.
No semantic deer/pedestrian classifier and no control interface are present.
"""
from dataclasses import dataclass, field
from functools import lru_cache
import math
import runtime_clock as time
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label as label_cells, maximum_filter
from telemetry import rotation
from policy import canonical_policy,DEFAULT_POLICY
from road_surface import local_heights
from object_fusion import FusionEngine

@dataclass
class Mount:
    # Origin: on ground directly below comma road-facing camera; FLU axes.
    xyz: list = field(default_factory=lambda:[0.,0.,1.79034])
    rpy_deg: list = field(default_factory=lambda:[0.,0.,0.])
    width_m: float = 2.345
    vehicle_height_m: float = 1.813
    bumper_x_m: float = 2.0
    measured: bool = False
    alignment_verified: bool = False

    def validate(self):
        v=np.asarray(self.xyz+self.rpy_deg+[self.width_m,self.vehicle_height_m,self.bumper_x_m])
        if not np.isfinite(v).all():raise ValueError('Mount values must be finite')
        if not (1<self.width_m<3 and 1<self.vehicle_height_m<4 and .5<self.xyz[2]<4 and 0<self.bumper_x_m<5):
            raise ValueError('Mount/vehicle measurements out of range')
        if max(abs(v) for v in self.rpy_deg)>30:raise ValueError('Expected a roughly forward-facing M1 mount')

def transform(points, mount):
    return points[:,:3]@rotation(np.radians(mount.rpy_deg)).T+mount.xyz

# Claude fork v23: the engine origin (under the comma camera) sits EGO_LEVER_ARM_M ahead of the rear axle, and the van turns about
# the rear axle, so in a turn the origin also moves sideways at yaw x lever arm. Without this term every stationary object showed a
# fake lateral velocity of about -2.5 m x yaw rate (MEASURED 2026-10-21 on 219 k track-frames of nine drives: -0.91 m/s median at
# 0.3-0.5 rad/s, symmetric left/right; range split: ~1.85 m offset + ~4 % yaw term) -> parked cars "crossing" in turns (drive F).
EGO_LEVER_ARM_M=1.85

def ego_delta(speed,yaw_rate,dt):
    angle=yaw_rate*dt
    if abs(yaw_rate)<1e-6:return np.array([speed*dt,0.]),angle
    L=EGO_LEVER_ARM_M   # rear-axle arc + rotation of the lever arm (exact rigid-body motion of the origin)
    return np.array([speed/yaw_rate*math.sin(angle)+L*(math.cos(angle)-1.),speed/yaw_rate*(1-math.cos(angle))+L*math.sin(angle)]),angle

def ego_delta_origin_heading(speed,yaw_rate,dt):
    """The original planar step (origin moving along its own heading), kept for the held ground plane only."""
    angle=yaw_rate*dt
    if abs(yaw_rate)<1e-6:return np.array([speed*dt,0.]),angle
    return np.array([speed/yaw_rate*math.sin(angle),speed/yaw_rate*(1-math.cos(angle))]),angle

def rotate2(points,angle):
    c,s=math.cos(angle),math.sin(angle)
    return np.asarray(points)@np.array([[c,s],[-s,c]])

def deskew(points, relative_times, speed, yaw_rate):
    """Transform each FLU point to scan-end ego frame; t <= 0 relative to end.

    Constant planar speed/yaw is an approximation, explicitly audited at runtime.
    Roll/pitch scan motion and independently wobbling mounts are not corrected.
    """
    out=points.copy()
    dt=-np.asarray(relative_times)
    ang=yaw_rate*dt
    if abs(yaw_rate)<1e-6:
        dx=speed*dt;dy=np.zeros_like(dt)
    else:
        dx=speed/yaw_rate*np.sin(ang)+EGO_LEVER_ARM_M*(np.cos(ang)-1.);dy=speed/yaw_rate*(1-np.cos(ang))+EGO_LEVER_ARM_M*np.sin(ang)   # v23 lever arm
    x,y=out[:,0]-dx,out[:,1]-dy
    out[:,0]=np.cos(ang)*x+np.sin(ang)*y
    out[:,1]=-np.sin(ang)*x+np.cos(ang)*y
    return out

GROUND_HOLD_S=2.0
GROUND_HOLD_TRAVEL_M=10.0
GROUND_HOLD_TURN_RAD=math.radians(10)
PRIOR_MIN_HEIGHT_M=.5
PRIOR_MAX_RANGE_M=25.


def upright_columns(points, minimum_span=.5, boundary_safe=False):
    """Point-local upright evidence; no component may borrow a distant pole."""
    if not len(points):return np.zeros(0,bool)
    result=np.zeros(len(points),bool)
    # Offset grids protect a thin object's vertical evidence when its returns
    # straddle an arbitrary cell edge. This protection is optional in normal
    # ground; degraded support keeps the independently reviewed strict rule.
    offsets=((0.,0.),(.125,0.),(0.,.125),(.125,.125)) if boundary_safe else ((0.,0.),)
    for offset in offsets:
        # Separate contiguous axes avoid allocating/reducing an Nx2 temporary
        # for each offset. Cell membership and all extrema remain identical.
        x=np.floor((points[:,0]+np.float64(offset[0]))/.25).astype(np.int64)
        y=np.floor((points[:,1]+np.float64(offset[1]))/.25).astype(np.int64)
        x-=x.min();y-=y.min()
        shape=(int(x.max())+1,int(y.max())+1)
        lo=np.full(shape,np.inf);hi=np.full(shape,-np.inf)
        indices=(x,y)
        # The same cells and extrema, using a scalar address for each return.
        # This avoids repeated two-axis indexing without dropping any points.
        flat=x*shape[1]+y
        np.minimum.at(lo.ravel(),flat,points[:,2]);np.maximum.at(hi.ravel(),flat,points[:,2])
        # Dense bounded ROI grids avoid repeated sorting and Python neighbour
        # walks for the four boundary-safe grids.
        near=maximum_filter(hi-lo>=minimum_span,size=3,mode='constant',cval=False)
        result|=near[indices]
    return result


def vertical_support(group,minimum_span=.5):
    """Fallback evidence: two local XY columns must contain vertical returns.

    A long, rising road patch can be tall in aggregate without being an upright
    object. This deliberately withholds poorly sampled fallback candidates.
    It is not a semantic object classification or a guarantee on irregular roads.
    """
    keys=np.floor(group[:,:2]/.25).astype(np.int64)
    _,membership=np.unique(keys,axis=0,return_inverse=True)
    lo=np.full(membership.max()+1,np.inf);hi=np.full(len(lo),-np.inf)
    np.minimum.at(lo,membership,group[:,2]);np.maximum.at(hi,membership,group[:,2])
    return np.count_nonzero(hi-lo>=minimum_span)>=2


GROUND_MISS_TOLERANCE=3
THREE_SENSOR_MIN_MPS=3.0;THREE_SENSOR_TOL_MPS=1.5;THREE_SENSOR_MIN_PROB=.9   # v21

NEAR_ROAD_MARGIN=.15   # v25
SIZE_VETO_FREE_M=.8   # v26

def _near_road_share(points,abc):
    """Share of near-road cells (4-15 m ahead, |y| < 3 m, lowest return per 0.5 m cell) within 0.08 m of the plane; None if < 50 returns."""
    p=points[(points[:,0]>4)&(points[:,0]<15)&(np.abs(points[:,1])<3)&(points[:,2]>-1.2)&(points[:,2]<1.2)]
    if len(p)<50:return None
    p=p[np.argsort(p[:,2])];key=np.floor(p[:,:2]/.5).astype(np.int64)
    _,idx=np.unique(key[:,0]*1000+key[:,1],return_index=True);q=p[idx]
    return float(np.mean(np.abs(q[:,2]-(q[:,0]*abc[0]+q[:,1]*abc[1]+abc[2]))<.08))

@lru_cache(maxsize=128)
def _ground_samples(count):
    # Ground.fit always used a fresh seed17 generator. These immutable index
    # draws depend only on point count, never on prior sensor observations.
    rng=np.random.default_rng(17)
    indices=np.array([rng.choice(count,3,replace=False) for _ in range(60)])
    indices.flags.writeable=False
    return indices


class Ground:
    def __init__(self):
        self.plane=None;self.stable=0;self.last_reason='no ground observations'
        self.miss_state={'miss':0,'tol':0}

    def _fail(self,reason):
        # Claude fork v20: forced mode tolerates up to GROUND_MISS_TOLERANCE consecutive failed fits without restarting the 5-frame
        # settling (one bad frame used to cost ~0.6 s of blank-out: 367 episodes / 4.4 min on nine drives).
        # v30e: counters belong to the ground estimator so clones/checkpoints preserve them.
        st=self.miss_state;st['miss']+=1
        if st['miss']>st['tol']:self.stable=0
        self.last_reason=reason;return None

    def fit(self,points):
        # Fit only a physically plausible road-height slab after rough mount transform.
        p=points[(points[:,0]>4)&(points[:,0]<40)&(np.abs(points[:,1])<9)&
                 (points[:,2]>-1.2)&(points[:,2]<1.2)]
        if len(p)<100:
            return self._fail('too few ground candidates')
        # Voxel low percentile prevents dense nearby returns dominating the fit.
        p=p[np.argsort(p[:,2])]
        key=np.floor(p[:,:2]/.75).astype(int)
        # The ROI above bounds Y cells to [-12, 11]. Packing the pair keeps
        # lexicographic cell order and the identical first (lowest) point.
        _,idx=np.unique(key[:,0]*24+key[:,1]+12,return_index=True)
        q=p[idx]
        if len(q)<40:
            return self._fail('insufficient ground coverage')
        if len(q)>2500:q=q[::math.ceil(len(q)/2500)]
        best=None;best_count=0
        A=np.column_stack((q[:,0],q[:,1],np.ones(len(q))))
        indices=_ground_samples(len(q))
        indices=indices[np.abs(np.linalg.det(A[indices]))>=.1]
        solutions=(np.linalg.solve(A[indices],q[indices,2,None])[:,:,0]
                   if len(indices) else np.empty((0,3)))
        for abc in solutions:
            if np.linalg.norm(abc[:2])>.25 or abs(abc[2])>.65:continue
            mask=np.abs(A@abc-q[:,2])<.12
            if np.count_nonzero(mask)>best_count:best,best_count=mask,np.count_nonzero(mask)
        if best is None or best_count<35 or best_count/len(q)<.4:
            return self._fail('no reliable road plane')
        inlier=q[best]
        if np.ptp(inlier[:,0])<8 or np.ptp(inlier[:,1])<3:
            return self._fail('ground support too narrow')
        abc=np.linalg.lstsq(A[best],q[best,2],rcond=None)[0]
        rms=float(np.sqrt(np.mean((A[best]@abc-q[best,2])**2)))
        if rms>.1:
            return self._fail('ground residual too large')
        changed=self.plane is not None and (np.linalg.norm(abc[:2]-self.plane[:2])>.04 or abs(abc[2]-self.plane[2])>.25)
        arbitration=None
        if changed and self.miss_state['tol']>0:
            # v25 (forced mode only): a plane jump is arbitrated on the near road (4-15 m ahead, |y| < 3 m, lowest return per 0.5 m cell)
            # instead of always restarting the 5-frame settling. MEASURED on drive H: at 90 jumps the near-road inlier share decided
            # clearly (> 0.15 apart) in 58 (new better 34, old better 24); undecided jumps keep the original settling.
            so=_near_road_share(points,self.plane);sn=_near_road_share(points,abc)
            if so is not None and sn is not None:
                if so-sn>NEAR_ROAD_MARGIN:
                    # keep the previous plane: this fit latched onto another surface; the frame counts as a held observation
                    self.stable+=1;self.miss_state['miss']=0;self.last_reason=''
                    return {'abc':list(map(float,self.plane)),'rms_m':rms,'support':int(best_count),'support_fraction':best_count/len(q),
                            'stable_frames':self.stable,'arbitration':'kept_previous'}
                if sn-so>NEAR_ROAD_MARGIN:changed=False;arbitration='accepted_new'
        self.stable=0 if changed else self.stable+1
        self.miss_state['miss']=0
        self.plane=abc
        self.last_reason='ground plane changed abruptly' if changed else ''
        out={'abc':abc.tolist(),'rms_m':rms,'support':int(best_count),
                'support_fraction':best_count/len(q),'stable_frames':self.stable}
        if arbitration:out['arbitration']=arbitration
        return out

def point_components(p,cell_m):
    """The detector's unchanged eight-connected XY components."""
    x=np.floor(p[:,0]/cell_m).astype(np.int32)
    y=np.floor(p[:,1]/cell_m).astype(np.int32)
    # The existing ROI bounds this occupancy image to a few tens of thousands
    # of cells. Label the same 8-connected components in compiled code instead
    # of walking a Python dictionary once for every return and neighboring cell.
    x-=x.min();y-=y.min()
    occupied=np.zeros((int(x.max())+1,int(y.max())+1),dtype=bool)
    occupied[x,y]=True
    labels,_=label_cells(occupied,structure=np.ones((3,3),dtype=np.uint8))
    membership=labels[x,y]
    order=np.argsort(membership,kind='stable');sorted_labels=membership[order]
    starts=np.r_[0,np.flatnonzero(np.diff(sorted_labels))+1];ends=np.r_[starts[1:],len(order)]
    # Preserve first-seen component order so track ID creation remains identical
    # to the former dictionary implementation, including tied assignments.
    components=sorted(zip(starts,ends),key=lambda se:order[se[0]])
    return [order[begin:end] for begin,end in components]


def compact_candidates(points,mask,policy):
    """Keep original compact objects when filtering a broad local road surface.

    A road patch can supply enough points/height for a nearby low object's final
    cluster. Filtering those points must not erase an already observable compact
    candidate. Broad terrain components are still eligible for surface removal.
    """
    protected=np.zeros(len(points),bool);observed=protected.copy();ids=np.flatnonzero(mask)
    if len(ids)<policy['min_cluster_points']:return protected,observed
    selected=points[ids]
    for component in point_components(selected,policy['cluster_cell_m']):
        if len(component)<policy['min_cluster_points']:continue
        span=np.ptp(selected[component],axis=0)
        if span[2]>=.22:observed[ids[component]]=True
        if span[2]>=.22 and max(span[:2])<=6. and min(span[:2])<=3.:
            protected[ids[component]]=True
    return protected,observed


def degraded_compact_support(points,mask,policy):
    """Preserve a bounded object's own two-column upright evidence."""
    result=np.zeros(len(points),bool);ids=np.flatnonzero(mask)
    if len(ids)<policy['min_cluster_points']:return result
    selected=points[ids]
    for component in point_components(selected,policy['cluster_cell_m']):
        if len(component)<policy['min_cluster_points']:continue
        group=selected[component];span=np.ptp(group,axis=0)
        if max(span[:2])<=6. and min(span[:2])<=3. and vertical_support(group):
            result[ids[component]]=True
    return result


def clusters(points,ground,mount,policy=None,return_membership=False):
    policy=DEFAULT_POLICY if policy is None else policy
    a,b,c=ground['abc']
    heights=points[:,2]-(a*points[:,0]+b*points[:,1]+c)
    # Keep low and tall unknown obstacles; remove roadway and clear overhead space.
    min_h=ground.get('min_height_m',.20);max_x=ground.get('max_range_m',120)
    height_mask=(heights>min_h)&(heights<mount.vehicle_height_m+.25)
    roi=(points[:,0]>mount.bumper_x_m+.3)&(points[:,0]<max_x)&(np.abs(points[:,1])<35)&np.isfinite(points).all(axis=1)
    original_mask=height_mask&roi
    observed=np.zeros(len(points),bool)
    if not ground.get('require_vertical_support'):
        surface_evidence={}
        local,supported=local_heights(points,ground['abc'],evidence=surface_evidence)
        ground['local_surface_points']=supported
        corrected=np.abs(local-heights)>.15
        # A better road estimate may reveal targets on rising terrain. It does
        # not withdraw existing high returns simply by moving the overhead cutoff.
        height_mask|=corrected&(local>min_h)&(local<mount.vehicle_height_m+.25)
        surface=corrected&(np.abs(local)<.10)&roi
        if surface.any():
            upright=np.zeros(len(points),bool)
            upright[roi]=upright_columns(points[roi],minimum_span=.22,boundary_safe=True)
            protected,observed=compact_candidates(points,original_mask,policy)
            # Remove only the directly observed road surface. Low/high parts of
            # upright obstacles keep their original evidence; isolated sparse
            # objects are never required to pass this support test in normal mode.
            height_mask&=~(surface&~upright&~protected)
    mask=height_mask&roi
    if ground.get('require_vertical_support') and ground.get('source')=='held':
        # Provenance only: these are candidate-mask returns, not identified
        # objects or confirmed hazards. The restoration/filter stays unchanged.
        ground['held_restoration']={'schema_version':1,'extra_candidate_points':0,'strict_candidate_points':0,
                                    'stage':'before_clustering','object_identity':'unknown'}
    if ground.get('require_vertical_support') and mask.any():
        idx=np.flatnonzero(mask)
        keep=np.zeros(len(points),bool)
        keep[idx[upright_columns(points[idx])]]=True
        if ground.get('source')=='held':
            restored=degraded_compact_support(points,mask,policy)
            ground['held_restoration'].update(extra_candidate_points=int(np.count_nonzero(restored&~keep)),
                                               strict_candidate_points=int(np.count_nonzero(keep)))
            mask=keep|restored
        else:mask=keep
    p=points[mask]
    if len(p)<policy['min_cluster_points']:return []
    components=point_components(p,policy['cluster_cell_m'])
    detections=[]
    source_indices=np.flatnonzero(mask)
    for component in components:
        if len(component)<policy['min_cluster_points']:continue
        group=p[component];lo=group.min(axis=0);hi=group.max(axis=0)
        # Flat isolated speckle cannot establish an obstacle. Do not demand vehicle size.
        if hi[2]-lo[2]<.22:
            # Removing road returns must not erase a previously observed low
            # object. Four existing-candidate returns above verified local road
            # establish height directly, without requiring artificial raw Z span.
            if ground.get('require_vertical_support'):continue
            member=source_indices[component]
            span=hi[:2]-lo[:2]
            height_evidence=observed[member]&surface_evidence['supported'][member]&(surface_evidence['heights'][member]>=.22)
            if max(span)>6. or min(span)>3. or np.count_nonzero(height_evidence)<policy['min_cluster_points']:continue
        if ground.get('require_vertical_support') and not vertical_support(group):continue
        detections.append({'center':np.median(group[:,:2],axis=0),'lo':lo,'hi':hi,'points':len(group)})
        if return_membership:detections[-1]['indices']=source_indices[component]
    return detections

@dataclass
class Track:
    id:int
    position:np.ndarray
    velocity:np.ndarray
    lo:np.ndarray
    hi:np.ndarray
    hits:int=1
    misses:int=0
    near_edge_state:object=None
    age_s:float=0
    residual_m:float=0
    points:int=0
    danger_frames:int=0
    association_ambiguous:bool=False
    observed_position:object=None
    measured_velocity:object=None
    motion_samples:int=0
    strong_motion_s:float=0.
    stable_motion_s:float=0.
    motion_edge_error_m:float=0.
    detection_index:object=None

class Tracker:
    def __init__(self):self.tracks={};self.next_id=1

    def reset(self):
        # Do not recycle identities: an old alert cooldown must not mask a new object.
        self.tracks.clear()

    def update(self,detections,dt,speed,yaw_rate):
        movement,angle=ego_delta(speed,yaw_rate,dt)
        predicted={}
        for key,t in self.tracks.items():
            # Object's ground velocity expressed in previous ego axes.
            predicted[key]=rotate2(t.position+t.velocity*dt-movement,-angle)
            t.velocity=rotate2(t.velocity,-angle)
            t.age_s+=dt
        keys=list(predicted);n,m=len(keys),len(detections)
        cost=np.full((n,m+n),1e6);allowed=np.zeros((n,m),bool);distances=np.zeros((n,m))
        gates=np.array([1.2+(45 if self.tracks[k].hits<3 else 12)*dt for k in keys])
        if n:cost[np.arange(n),m+np.arange(n)]=gates+.5
        if n and m:
            positions=np.array([predicted[k] for k in keys]);centers=np.array([d['center'] for d in detections])
            distances=np.linalg.norm(positions[:,None,:]-centers[None,:,:],axis=2)
            old_size=np.maximum(.2,np.array([self.tracks[k].hi-self.tracks[k].lo for k in keys]))
            new_size=np.maximum(.2,np.array([d['hi']-d['lo'] for d in detections]))
            difference=np.linalg.norm(old_size[:,None,:]-new_size[None,:,:],axis=2)
            ratio=np.max(np.maximum(old_size[:,None,:]/new_size[None,:,:],new_size[None,:,:]/old_size[:,None,:]),axis=2)
            # v26: the size-change veto no longer applies when the detection's centre is within SIZE_VETO_FREE_M of the prediction
            # (SYNTHETIC NHTSA braking lead 10/21: a braking car's track was re-born every 3-5 frames at the same centre, resetting
            # danger_frames and delaying the warning by 1-1.5 s)
            allowed=(distances<gates[:,None])&~((ratio>3)&(difference>1)&(distances>SIZE_VETO_FREE_M))
            cost[:,:m]=np.where(allowed,distances+.15*difference,1e6)
        used_tracks=set();used_detections=set()
        rows,cols=linear_sum_assignment(cost) if n else ([],[])
        for row,i in zip(rows,cols):
            if i>=m or not allowed[row,i]:continue
            key=keys[row];residual=float(distances[row,i])
            t=self.tracks[key];d=detections[i]
            row_alternatives=np.delete(cost[row,:m],i)
            col_alternatives=np.delete(cost[:,i],row)
            t.association_ambiguous=bool((len(row_alternatives) and np.min(row_alternatives)-cost[row,i]<.2) or
                                         (len(col_alternatives) and np.min(col_alternatives)-cost[row,i]<.2))
            from motion_trust import observe
            __import__('near_edge_motion').update(t,d,movement,angle,dt,rotate2)   # v22: before observe() overwrites the track's observed state
            observe(t,d,movement,angle,dt)
            innovation=d['center']-predicted[key]
            # alpha-beta tracker; beta increases during velocity acquisition.
            beta=.65 if t.hits<5 else .25
            t.velocity=t.velocity+beta*innovation/max(dt,.025)
            t.position=predicted[key]+.85*innovation
            t.residual_m=.7*t.residual_m+.3*residual
            t.lo,t.hi,t.points=d['lo'],d['hi'],d['points'];t.detection_index=i
            t.hits+=1;t.misses=0
            used_tracks.add(key);used_detections.add(i)
        for key in list(self.tracks):
            if key not in used_tracks:
                t=self.tracks[key];t.misses+=1;t.position=predicted[key];t.danger_frames=0;t.detection_index=None
                from motion_trust import reset
                reset(t)
                __import__('near_edge_motion').reset(t)
                if t.misses>2:del self.tracks[key]
        for i,d in enumerate(detections):
            if i not in used_detections:
                key=self.next_id;self.next_id+=1
                self.tracks[key]=Track(key,d['center'].copy(),np.zeros(2),d['lo'],d['hi'],points=d['points'],observed_position=d['center'].copy(),detection_index=i)
        return list(self.tracks.values())

def independent_corridor_envelope(context):
    """Validate only the commissioned path extension; legacy paths stay exact."""
    diagnostic=context.get('independent_corridor')
    if not isinstance(diagnostic,dict) or diagnostic.get('active') is not True:return None
    try:
        empirical=(diagnostic.get('empirical') is True and context.get('empirical_profile_validated') is True and
                   diagnostic.get('method')=='sensor_derived_empirical')
        lateral_limit=1. if empirical else .5
        longitudinal_limit=4.5 if empirical else 2.
        horizon_limit=2. if empirical else 5.
        path=np.asarray(context['path'],float);times=np.asarray(context['path_times_s'],float)
        lateral=np.asarray(context['path_lateral_error_m'],float)
        longitudinal=np.asarray(context['path_longitudinal_error_m'],float)
        horizon=context['warning_horizon_s']
        valid=(diagnostic.get('eligible') is True and context.get('motion_verified') is True and
               context.get('coordinate_frame')=='camera_ground_flu' and
               isinstance(context.get('calibration_epoch'),(str,int)) and not isinstance(context.get('calibration_epoch'),bool) and
               path.ndim==2 and path.shape[1]==2 and 6<=len(path)<=256 and
               times.shape==lateral.shape==longitudinal.shape==(len(path),) and
               all(np.isfinite(v).all() for v in (path,times,lateral,longitudinal)) and
               abs(times[0])<1e-9 and np.all(np.diff(times)>0) and 0<times[-1]<=horizon_limit and
               np.linalg.norm(path[0])<.5 and np.all(np.diff(path[:,0])>0) and
               np.all((lateral>=0)&(lateral<=lateral_limit)) and np.all((longitudinal>=0)&(longitudinal<=longitudinal_limit)) and
               type(horizon) in (int,float) and math.isfinite(horizon) and 0<horizon<=min(4.5,times[-1]+1e-9))
        if not valid:return {'valid':False,'reason':'independent corridor envelope malformed or unverified'}
        return {'valid':True,'path':path,'times':times,'lateral':lateral,'longitudinal':longitudinal,
                'warning_horizon_s':float(horizon)}
    except (KeyError,TypeError,ValueError,OverflowError):
        return {'valid':False,'reason':'independent corridor envelope malformed or unverified'}


def _threat_geometry(context):
    """Immutable ego-path geometry shared only within one decision frame."""
    envelope=independent_corridor_envelope(context)
    if envelope is not None and not envelope['valid']:return None
    path=context['path'];speed=context['speed_mps']
    if len(path)<6 or speed<1:return None
    times=np.arange(0.,5.01,.1) if envelope is None else envelope['times']
    # Path is spatial. Use measured current ego speed to advance arc length.
    segments=np.linalg.norm(np.diff(path,axis=0),axis=1)
    arc=np.r_[0,np.cumsum(segments)]
    arc0=float(np.interp(0,path[:,0],arc))
    acceleration=min(2.,max(-6.,context.get('acceleration_mps2',0)))
    if envelope is None:
        motion_times=np.minimum(times,-speed/acceleration) if acceleration<-.1 else times
        distances=arc0+speed*motion_times+.5*acceleration*motion_times**2
        if distances[-1]>arc[-1]:
            stop=int(np.searchsorted(distances,arc[-1]))
            if stop==0:return None
            end_time=float(np.interp(arc[-1],distances[stop-1:stop+1],times[stop-1:stop+1]))
            times=np.r_[times[:stop],end_time];distances=np.r_[distances[:stop],arc[-1]]
        ego=np.column_stack([np.interp(distances,arc,path[:,0]),np.interp(distances,arc,path[:,1])])
    else:
        # The independent corridor already binds its geometry to physical time.
        # Re-projecting by speed would lose that binding during acceleration.
        ego=envelope['path'];distances=arc
    mid=(distances[:-1]+distances[1:])/2
    j=np.clip(np.searchsorted(arc,mid)-1,0,len(path)-2)
    tangent=path[j+1]-path[j];tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-9)
    normal=np.column_stack([-tangent[:,1],tangent[:,0]])
    return envelope,path,speed,times,ego,acceleration,tangent,normal


def threat(track, context, mount, *, hypothesis_velocity=None, geometry=None):
    geometry=_threat_geometry(context) if geometry is None else geometry
    if geometry is None:return None
    envelope,path,speed,times,ego,acceleration,tangent,normal=geometry
    if track.misses or track.hits<3 or track.age_s<.2 or track.association_ambiguous:return None
    velocity=track.velocity if hypothesis_velocity is None else hypothesis_velocity
    half_width=mount.width_m/2+.25
    object_half=np.maximum(.15,(track.hi[:2]-track.lo[:2])/2)
    relative=track.position+times[:,None]*velocity-ego
    start=np.column_stack([np.sum(relative[:-1]*tangent,axis=1),np.sum(relative[:-1]*normal,axis=1)])
    end=np.column_stack([np.sum(relative[1:]*tangent,axis=1),np.sum(relative[1:]*normal,axis=1)])
    ext_along=np.abs(tangent)@object_half;ext_cross=np.abs(normal)@object_half
    # Continuous swept slab intersections within each interval. A high-closing
    # object cannot disappear merely because impact lies between 100 ms samples.
    along_error=cross_error=0.
    if envelope is not None:
        # Largest endpoint envelope conservatively covers the complete interval.
        along_error=np.maximum(envelope['longitudinal'][:-1],envelope['longitudinal'][1:])
        cross_error=np.maximum(envelope['lateral'][:-1],envelope['lateral'][1:])
    low=np.column_stack([-1-ext_along-along_error,-half_width-ext_cross-cross_error])
    high=np.column_stack([mount.bumper_x_m+ext_along+along_error,half_width+ext_cross+cross_error])
    delta=end-start;moving=np.abs(delta)>1e-10
    safe_delta=np.where(moving,delta,1.)
    u0=(low-start)/safe_delta;u1=(high-start)/safe_delta
    enter=np.where(moving,np.minimum(u0,u1),-np.inf)
    leave=np.where(moving,np.maximum(u0,u1),np.inf)
    outside=(~moving)&((start<low)|(start>high))
    lower=np.maximum(0,np.max(enter,axis=1));upper=np.minimum(1,np.min(leave,axis=1))
    hits=np.flatnonzero((lower<=upper)&(~np.any(outside,axis=1)))
    if not len(hits):return None
    i=int(hits[0]);t=float(times[i]+lower[i]*(times[i+1]-times[i]))
    impact_speed=max(0,speed+acceleration*t)
    closing=impact_speed-float(velocity@tangent[i])
    if closing<1. and abs(velocity@normal[i])<.8:return None
    signed_offset=track.position[1]-float(np.interp(track.position[0],path[:,0],path[:,1]))
    direction='LEFT' if signed_offset>.45 else 'RIGHT' if signed_offset<-.45 else 'AHEAD'
    oncoming=velocity[0]<-3 and closing>impact_speed+3
    result={'track_id':track.id,'phrase':('ONCOMING OBSTACLE ' if oncoming else 'OBSTACLE ')+direction,
            'direction':direction,'oncoming_candidate':oncoming,'ttc_s':round(t,3),
            'range_m':round(float(np.linalg.norm(track.position)),2),'closing_mps':round(closing,2),
            'ground_velocity_mps':velocity.round(3).tolist(),'position_m':track.position.round(3).tolist(),
            'residual_m':round(track.residual_m,3)}
    if envelope is not None:
        result['corridor_evidence']={'source':'independently_commissioned_motion',
            'supported_horizon_s':float(times[-1]),'warning_horizon_s':envelope['warning_horizon_s'],
            'lateral_error_bound_m':float(cross_error[i]),'longitudinal_error_bound_m':float(along_error[i]),
            'uncertainty_applied':True,'coverage_qualified':False}
    return result

MIN_FORCED_SPEED_MPS=3.0   # usual FCW activation floor (~11 km/h); below it the forced mode logs but does not warn
STATIC_TTC_THRESHOLD_S=2.7 # stationary in-corridor objects: typical FCW timing for stationary targets (2.0-2.7 s)
STATIC_TTC_MIN_S,STATIC_TTC_MAX_S=2.0,3.0   # v8: static timing = reaction 1.2 s + v/(2 x 5.9 m/s^2), clamped to this band
DRIVER_BRAKING_TTC_S=1.2                    # v8 (superseded by v9 a_req rule; kept for the record)
DRIVER_BRAKING_REQ_DECEL_MPS2=4.0           # v9: while braking, warn only if stopping short needs >= this (0.4 g) and it is not being delivered
CROSSING_FAR_M=30.0;CROSSING_FAR_SUSTAIN_S=0.6   # v13: crossing predictions beyond 30 m need 0.6 s of consistent motion
NEAR_EDGE_BAND_M=0.5                        # v13: static returns within this band outside the half-width+margin are logged as near_edge
TURN_CONSENSUS_RPS=0.08                     # v9: dead-reckoned path in a turn: the object must be in the way under the straight path too
DRIVER_BRAKING_DECEL_MPS2=-1.0              # v8: measured deceleration that counts as 'already braking' when no pedal flag
from lead_gate import LEAD_GATE_AREQ_MPS2   # v16 moving-lead gate, see lead_gate.py
def static_ttc_threshold(speed):return float(min(STATIC_TTC_MAX_S,max(STATIC_TTC_MIN_S,1.2+float(speed)/11.8)))

class Engine:
    def __init__(self,mount=None,policy=None,forced_experiment=None):
        self.forced_experiment=forced_experiment
        self.mount=mount or Mount();self.mount.validate()
        self.policy=canonical_policy(policy)
        self.ground=Ground();self.tracker=Tracker();self.last_ns=None;self.held=None
        self._settled_road_reference=None
        self._warning_surface=None
        self.last_alert={};self.last_any_alert=-1e9;self.last_warning=None
        self.last_alert_urgent={}
        # Warning-relevant state is owned explicitly, never in identity-keyed side registries.
        self._far_accumulator=None;self._lead_history={}
        self._radar_streaks={};self._radar_publication_times={};self._radar_last_update_ns=None
        from longitudinal_evidence import LongitudinalEvidence
        self._longitudinal_evidence=LongitudinalEvidence()

    def _ensure_fusion_state(self):
        # Legacy replay checkpoints retain their exact original Engine fields.
        if not hasattr(self,'fusion_engine'):
            self.fusion_engine=FusionEngine();self.fusion_danger={}
            self.fusion_last_alert={};self.fusion_last_alert_urgent={};self.fusion_last_any_alert=-1e9

    def reject_delivery(self,track_id,at_ns):
        """Release only this decision's cooldown if audio was never submitted."""
        if self.last_alert.get(track_id)==at_ns/1e9:
            self.last_alert.pop(track_id,None);self.last_alert_urgent.pop(track_id,None)
            self.last_any_alert=max(self.last_alert.values(),default=-1e9)
            if self.last_warning and self.last_warning['track_id']==track_id:self.last_warning=None
            if self._warning_surface and self._warning_surface['track_id']==track_id:self._warning_surface=None
        if getattr(self,'fusion_last_alert',{}).get(track_id)==at_ns/1e9:
            self.fusion_last_alert.pop(track_id,None);self.fusion_last_alert_urgent.pop(track_id,None)
            self.fusion_last_any_alert=max(self.fusion_last_alert.values(),default=-1e9)

    def update(self,points,context,quality=None,relative_times=None,processing_deadline_ns=None,fusion_inputs=None):
        if fusion_inputs is not None:self._ensure_fusion_state()
        now=context['at_ns']/1e9
        dt=0 if self.last_ns is None else (context['at_ns']-self.last_ns)/1e9
        self.last_ns=context['at_ns']
        reasons=list(context.get('reasons',[]));quality=quality or {}
        from warning_surface import advance_surface, repeat_surface, remember_surface
        self._warning_surface=advance_surface(getattr(self,'_warning_surface',None),context,quality,dt,now)
        # Empirical mode carries its measured timing error through the corridor
        # uncertainty; it never reports that80ms is a verified50ms alignment.
        timing_limit=.1 if context.get('empirical_profile_validated') is True else .05
        corridor_envelope=independent_corridor_envelope(context)
        if corridor_envelope is not None and not corridor_envelope['valid']:
            reasons.append(corridor_envelope['reason'])
        if not context.get('ready') and not reasons:reasons.append('fusion inputs invalid')
        forced=bool(self.forced_experiment is not None and self.forced_experiment.matches(context))
        if self.forced_experiment is not None and not forced:reasons.append('Forced experiment context identity missing or changed')
        bypass=[]
        if not self.mount.measured:
            (bypass if forced else reasons).append('mount measurements required')
        if not self.mount.alignment_verified:
            (bypass if forced else reasons).append('lidar/path alignment not verified')
        if quality.get('completeness',0)<.98:reasons.append('incomplete lidar frame')
        if quality.get('timestamp_valid') is not True:reasons.append('lidar time invalid')
        if quality.get('arrival_age_s',1)>.2:reasons.append('lidar frame stale')
        if quality.get('timing_bound_s',1)>timing_limit:
            (bypass if forced else reasons).append('lidar/comma time alignment unverified')
        ground_continuity=None
        if not .025<=dt<=.25:
            # A short processing gap invalidates object history, but is not an
            # observation that the road changed. Retain only the old road-fit
            # comparison state, and require a fresh successful fit below.
            # Changed/absent ground still settles or blocks normally. No held
            # plane, track, fused association or far return survives this reset.
            preserve_ground=(.25<dt<=.5 and self.ground.plane is not None and self.ground.stable>=5 and
                bool(context.get('ready')) and bool(context.get('motion_ready',context.get('ready'))) and
                quality.get('completeness',0)>=.98 and quality.get('timestamp_valid') is True and
                quality.get('arrival_age_s',1)<=.2 and
                math.isfinite(float(context.get('speed_mps') or 0.)) and
                math.isfinite(float(context.get('yaw_rate_rps') or 0.)))
            self.tracker.reset()
            __import__('far_objects').accumulator_for(self).reset()
            self.held=None
            if preserve_ground:
                ground_continuity={'gap_s':dt,'fresh_fit_required':True,'object_history_reset':True}
            else:self.ground=Ground()
            if hasattr(self,'fusion_engine'):self.fusion_engine.reset();self.fusion_danger.clear()
            if dt:reasons.append('scan timing discontinuity')
            dt=.1
        p=transform(points,self.mount)
        if relative_times is not None and context.get('ready'):
            p=deskew(p,relative_times,context['speed_mps'],context['yaw_rate_rps'])
        self.ground.miss_state['tol']=GROUND_MISS_TOLERANCE if forced else 0
        # A reference is comparison evidence only: never output or cluster against
        # it. Fresh valid points must independently support the fresh fit again.
        from ground_recovery import advance_reference, recover_fresh, remember_fresh
        self._settled_road_reference=advance_reference(
            getattr(self,'_settled_road_reference',None),context,quality,dt,now,
            discontinuity=('scan timing discontinuity' in reasons))
        ground=self.ground.fit(p)
        if forced and ground is not None:
            recovery=recover_fresh(self._settled_road_reference,ground,p)
            if recovery:
                self.ground.stable=max(5,self.ground.stable)
                ground['stable_frames']=self.ground.stable
                ground['fresh_reference_recovery']=recovery
        self._settled_road_reference=remember_fresh(
            self._settled_road_reference,ground,context,quality,now)
        if ground is not None and ground_continuity is not None:
            ground['processing_gap_recheck']={**ground_continuity,'settled':ground['stable_frames']>=5}
        speed=float(context.get('speed_mps') or 0.)
        yaw=float(context.get('yaw_rate_rps') or 0.)
        reliable=(context.get('ready') and context.get('motion_ready',context.get('ready')) and
                  quality.get('completeness',0)>=.98 and quality.get('timestamp_valid') is True and
                  quality.get('arrival_age_s',1)<=.2 and quality.get('timing_bound_s',1)<=timing_limit and
                  math.isfinite(speed) and math.isfinite(yaw) and speed>=0)
        if not reliable:self.held=None
        elif self.held is not None:
            # Planar ego transform: old XY = R(yaw)*new XY + translation.
            # Vertical/pitch motion is not measured here: constrain travel/turn
            # and require upright support rather than extrapolate indefinitely.
            movement,angle=ego_delta_origin_heading(speed,yaw,dt)   # v23: the held plane keeps the original (lever-free) motion; Astra's test pins it and forced mode never holds
            a,b,c=self.held['abc'];slopes=rotate2(np.array([a,b]),-angle)
            self.held['abc']=[*slopes,float(c+np.dot([a,b],movement))]
            self.held['travel_m']+=float(np.linalg.norm(movement));self.held['turn_rad']+=abs(angle)
            if (now-self.held['at']>GROUND_HOLD_S or self.held['travel_m']>GROUND_HOLD_TRAVEL_M or
                self.held['turn_rad']>GROUND_HOLD_TURN_RAD):self.held=None
        if (ground is not None and ground['stable_frames']<5 and self.held is not None and
            np.linalg.norm(np.asarray(ground['abc'][:2])-np.asarray(self.held['abc'][:2]))<=.04 and
            abs(ground['abc'][2]-self.held['abc'][2])<=.25):
            # Same plane as the recently verified one after a brief dropout:
            # not a new grade, so do not restart the 5-frame settling blackout.
            # (A genuinely different fresh plane still settles, as before.)
            self.ground.stable=5;ground['stable_frames']=5;ground['resumed_after_dropout']=True
        if ground is not None:
            # A fresh fit wins even when settling. Never hide a newly observed
            # grade change behind an older, apparently stable plane.
            ground['source']='fit' if ground['stable_frames']>=5 else 'fit_settling'
            if ground['stable_frames']<5:
                self.held=None;reasons.append('ground estimate settling')
            elif reliable and ((self.mount.measured and self.mount.alignment_verified) or forced):   # v19: forced mode keeps the last settled plane too, so a brief fit
                self.held={'abc':list(ground['abc']),'at':now,'travel_m':0.,'turn_rad':0.}             # dropout that resumes on the SAME plane skips the 0.5 s re-settling blackout
        elif reliable and self.mount.measured and self.mount.alignment_verified:
            if self.held is not None:
                ground={'abc':list(self.held['abc']),'source':'held','held_age_s':round(now-self.held['at'],3),
                        'stable_frames':0,'require_vertical_support':True,'degraded':True}
            else:
                ground={'abc':[0.,0.,0.],'source':'mount_prior','stable_frames':0,'degraded':True,
                        'min_height_m':PRIOR_MIN_HEIGHT_M,'max_range_m':PRIOR_MAX_RANGE_M,'require_vertical_support':True}
        if ground is None:reasons.append(self.ground.last_reason)
        if context.get('empirical_requires_observed_ground') is True and ground and ground.get('source')=='mount_prior':
            reasons.append('Road-relative calibration needs an observed ground plane; mount-height-only fallback unavailable')
        if forced:
            ground,forced_reason=self.forced_experiment.ground(ground,context.get('speed_mps'))
            if forced_reason:reasons.append(forced_reason)
        from limited_experiment import constrain_ground
        ground,experimental_reason=constrain_ground(ground,context)
        if experimental_reason:reasons.append(experimental_reason)
        detections=clusters(p,ground,self.mount,self.policy,return_membership=forced) if ground else []
        if forced and ground:
            # v14: the far stage looks to FAR_STAGE_REACH_M at every speed over an ego-compensated union of the last frames' far returns;
            # candidates / warnings keep the forced envelope (threat() sweeps 5 s, the corridor reach stays the forced reach).
            from far_objects import far_clusters,accumulator_for,FAR_STAGE_REACH_M,ACCUM_MIN_FRAMES,MIN_POINTS as FAR_MIN_POINTS
            far_acc=accumulator_for(self)
            far_acc.frame+=1;mv,ang=ego_delta(speed,yaw,dt)
            from far_objects import curvature as _far_curv
            p_aug,tags=far_acc.step(p,mv,ang,far_acc.frame,abc=ground['abc'],k=_far_curv(p,tuple(ground['abc'])))
            _FO=__import__('far_objects')   # v30: established low, near-static far tracks predicted into this frame (hysteresis anchors)
            held=[tuple(rotate2(tr.position-mv,-ang)) for tr in self.tracker.tracks.values()
                  if tr.hits>=_FO.HOLD_MIN_HITS and float(tr.position[0])>=_FO.HOLD_MIN_X_M and float(tr.hi[2]-tr.lo[2])<=1.0 and float(np.hypot(*tr.velocity))<2.]
            far=far_clusters(p_aug,ground,self.mount,self.policy,max(float(ground.get('max_range_m',40.)),FAR_STAGE_REACH_M),point_components,detections,return_membership=True,held=held)
            kept=[]
            for d in far:
                idx=np.asarray(d['indices']);cur=idx[idx<len(p)];nfr=len(set(tags[idx].tolist()))
                if len(cur)==0 or (len(cur)<FAR_MIN_POINTS and nfr<ACCUM_MIN_FRAMES):continue   # sparse now and not seen before: wait
                if d.get('low_far') and not d.get('vface') and not d.get('held_far') and nfr<__import__('far_objects').LOW_MIN_FRAMES:continue    # v17: a low far object must persist (>= 3 of the last 4 frames); v18: a dense vertical face relies on the tracker's 3 hits instead (plane jitter breaks the accumulator's height filter)
                # v30e: a moving track needs its fresh measurement; only historical returns are excluded.
                d['_ownership_indices']=idx.copy()
                d['indices']=cur;d['frames_seen']=nfr;d['points_now']=int(len(cur));kept.append(d)
            detections=detections+kept
        motion_ready=context.get('motion_ready',context.get('ready')) or context.get('stationary_confirmed',False)
        observation_reliable=(quality.get('completeness',0)>=.98 and quality.get('timestamp_valid') is True and quality.get('arrival_age_s',1)<=.2)
        motion_resolutions={}
        if motion_ready and observation_reliable:
            tracks=self.tracker.update(detections,dt,context['speed_mps'],context['yaw_rate_rps'])
            if forced:
                from motion_trust import inherited_motion_contradiction,resolve_motion_evidence
                for track in tracks:
                    raw=inherited_motion_contradiction(track)
                    if raw['contradicted']:motion_resolutions[track.id]=resolve_motion_evidence(track,context,raw)
            if forced and ground:
                far_acc.assign_owners(detections,tracks,{key:value['authority'] for key,value in motion_resolutions.items()})
        else:
            self.tracker.reset();tracks=[]
            __import__('far_objects').accumulator_for(self).reset()
        longitudinal=self._longitudinal_evidence.update(tracks,motion_resolutions,context) if forced else {}
        candidates=[];alerts=[];deferred_crossings=[];radar_matched=set()
        radar_pts=(context.get('radar_points') or []) if forced else []
        geometry=_threat_geometry(context) if context.get('ready') and tracks else None
        for track in tracks:
            candidate=threat(track,context,self.mount,geometry=geometry) if geometry is not None else None
            if forced and context.get('ready'):
                resolution=motion_resolutions.get(track.id)
                authority=resolution['authority'] if resolution is not None else {'contradicted':False}
                if authority['contradicted']:
                    if candidate is None:
                        candidate=threat(track,context,self.mount,hypothesis_velocity=np.zeros(2),geometry=geometry) if geometry is not None else None
                        if candidate:
                            candidate['initial_threat_hypothesis']={'velocity_mps':[0.,0.],
                                'basis':'stationary_possibility_after_inherited_motion_contradiction',
                                'measured_stationary_motion':False}
                    if candidate:candidate['inherited_motion_contradiction']=authority
                elif candidate and authority.get('lidar_contradiction'):
                    candidate['inherited_motion_resolution']=authority
            if candidate and forced:
                from motion_trust import resolve_motion_evidence
                from corridor_occupancy import assess
                resolved=motion_resolutions.get(track.id)
                if resolved is None:resolved=resolve_motion_evidence(track,context,{'contradicted':False})
                motion_evidence=resolved['evidence'];cap=self.forced_experiment.reach_m(context.get('speed_mps'))
                from radar_fusion import contradiction
                assoc=resolved['radar']
                if assoc:radar_matched.add(assoc['point']['id']);candidate['radar']={k:v for k,v in assoc.items() if k!='point'}|{'radar_id':assoc['point']['id']}
                else:candidate['radar']=None
                candidate['vision_lead']=resolved['vision_lead']
                member=detections[track.detection_index]['indices'] if track.detection_index is not None and track.detection_index<len(detections) else None
                fe=context.get('forced_experiment') or {};src=str(fe.get('path_source',''));camera_route=src.startswith('comma') or 'map prior' in src   # v13b: a map-continued route follows the road beyond the dead-reckoned part
                dr_end=fe.get('dr_end_m')   # v13b: where the dead-reckoned part ends (None for a camera base route)
                # A camera-derived route already follows the road: extend it to the reach even while turning.
                occ=assess(p[member][:,:2] if member is not None else None,context['path'],self.mount,cap,0. if camera_route else context.get('yaw_rate_rps',0.))
                yaw_now=float(context.get('yaw_rate_rps') or 0.)
                if occ['in_path'] and (not src.startswith('comma')) and abs(yaw_now)>TURN_CONSENSUS_RPS and member is not None and (dr_end is None or (occ.get('nearest_along_m') or 0.)<=dr_end+1.):   # v13b: consensus only inside the dead-reckoned part
                    straight=assess(p[member][:,:2],np.array([[0.,0.],[cap,0.]]),self.mount,cap,0.)
                    if not straight['in_path']:occ={**occ,'in_path':False,'turn_consensus':False}
                box=track.hi[:2]-track.lo[:2];compact_box=bool(max(box)<=8.)
                # Crossing predictions need sustained, not momentary, motion agreement (0.9.8 trusted after 2 frames).
                v_obj=np.asarray(track.velocity,float);moving_obj=bool(np.hypot(*v_obj)>=1.)
                # A crossing prediction needs a compact object that is actually moving, with a real component across the path.
                # v13: lateral velocity noise grows with range; beyond CROSSING_FAR_M a crossing prediction needs CROSSING_FAR_SUSTAIN_S of
                # consistent motion (night drive 10/19: two roadside signs at 38-43 m became crossing candidates from 0.3 s of noise).
                need_s=CROSSING_FAR_SUSTAIN_S if float(track.position[0])>=CROSSING_FAR_M else .3
                crossing_ok=bool(motion_evidence['trusted'] and compact_box and moving_obj and abs(float(v_obj[1]))>=.8 and
                                 (motion_evidence['strong_consistency_s']>=need_s or (motion_evidence['basis']=='sustained_observation_agreement' and float(track.position[0])<CROSSING_FAR_M)))
                # v10: the radar's range-rate is a measurement; a lidar 'moving' verdict it contradicts is clutter velocity noise.
                radar_contra=contradiction(assoc,bool(motion_evidence['trusted']),float(np.hypot(*v_obj)))
                if radar_contra:crossing_ok=False
                # v22: crossing motion needs the near corner to move with the centroid (a revealed parked car's centroid slides, its near corner does not)
                if crossing_ok and not __import__('near_edge_motion').agrees(track):crossing_ok=False;candidate['near_edge_disagrees']=True
                candidate.update(__import__('near_edge_motion').info(track))
                candidate['motion_evidence']={**motion_evidence,'observed_footprint_overlaps_path':occ['in_path'],'crossing_ok':crossing_ok};candidate['corridor']=occ
                speed=float(context['speed_mps'])
                # Measured braking (never acceleration) shortens the reach: a driver already stopping is not warned.
                braking=fe.get('original_acceleration_mps2')
                braking=min(0.,float(braking)) if isinstance(braking,(int,float)) and np.isfinite(braking) else 0.
                # Pedal onset is driver response, not proof of adequate deceleration.
                # Suppression below must be supported by measured braking.
                driver_braking=bool(context.get('brake_pressed') is True or braking<=DRIVER_BRAKING_DECEL_MPS2)
                def reach_time(distance,closing):
                    if speed<MIN_FORCED_SPEED_MPS or closing<1.:return None
                    if braking<-.3:
                        disc=closing*closing+2*braking*distance
                        if disc<=0:return None          # stops short of it at the current braking
                        return (-closing+disc**.5)/braking
                    return distance/closing
                if occ['parallel_structure'] and not crossing_ok:
                    deferred_crossings.append({**candidate,'reason':'Roadside structure alongside the path, not an obstacle in it','label':'parallel_structure_not_a_false_alarm_label'})
                    candidate=None
                elif occ['in_path']:
                    mq=context.get('map_query');candidate['map_tier']=(__import__('map_live').LiveMap.tier_rel(mq.get('a_rel'),mq.get('b_rel'),float(occ['mean_cross_m'])) if mq and mq.get('usable') else 'no_map')
                    candidate['map_conflict']=candidate['map_tier']=='outside'
                    candidate['map_warning_override']=False
                    # A trusted-moving object in the corridor (a slower car ahead) closes at the relative speed, not the ego speed.
                    rel=float(v_obj[0]) if (motion_evidence['trusted'] and moving_obj) else 0.   # +: pulling away, -: oncoming
                    if assoc and assoc['stationary']:rel=0.;candidate['ttc_basis_velocity']='radar_stationary'                 # v10b: radar says not moving -> static timing
                    elif assoc and motion_evidence['trusted'] and moving_obj:rel=float(assoc['ground_speed_mps']);candidate['ttc_basis_velocity']='radar_range_rate'   # v10: measured closing speed, lidar agrees it moves
                    elif track.id in longitudinal and __import__('longitudinal_evidence').usable_for_timing(self,track.id,longitudinal[track.id],context['at_ns']/1e9):
                        rel=longitudinal[track.id]['ground_speed_mps'];candidate['ttc_basis_velocity']='persistent_unique_radar_lidar_longitudinal_agreement';candidate['longitudinal_evidence']=longitudinal[track.id]
                    elif assoc:candidate['ttc_basis_velocity']='radar_motion_unconfirmed_by_lidar'                              # v10b: radar alone may not add motion
                    lead=max(0.,rel);closing=speed-rel
                    distance=max(0.,occ['nearest_along_m']-self.mount.bumper_x_m);t=reach_time(distance,closing)
                    if lead>0 and motion_evidence['trusted']:   # v19: a braking lead closes faster than constant-velocity / ego-braking-only TTC says
                        tb=__import__('lead_gate').braking_lead_ttc(self,track.id,context['at_ns']/1e9,distance,speed,lead,braking if braking<-.3 else 0.)
                        if tb is not None and (t is None or tb<t):t=tb;candidate['ttc_basis_velocity']='braking_lead_decel'
                    if t is None:
                        deferred_crossings.append({**candidate,'reason':'In-corridor object but van too slow or already stopping short','label':'braking_or_slow_not_a_false_alarm_label'});candidate=None
                    elif driver_braking and (lambda ar:__import__('warning_arbitration').braking_adequate(ar,-braking))(__import__('lead_gate').required_decel(self,track.id,context['at_ns']/1e9,distance,speed,lead,closing) if candidate.get('ttc_basis_velocity')=='braking_lead_decel' else closing*closing/(2*max(distance,.5))):
                        a_req=__import__('lead_gate').required_decel(self,track.id,context['at_ns']/1e9,distance,speed,lead,closing) if candidate.get('ttc_basis_velocity')=='braking_lead_decel' else closing*closing/(2*max(distance,.5))
                        deferred_crossings.append({**candidate,'ttc_s':round(t,3),'required_decel_mps2':round(a_req,2),'measured_decel_mps2':round(-braking,2),'reason':'Driver braking and stopping short needs only %.1f m/s2'%a_req,'label':'driver_braking_adequate_not_a_false_alarm_label'});candidate=None
                    else:
                        mc=occ['mean_cross_m'];direction='LEFT' if mc>.45 else 'RIGHT' if mc<-.45 else 'AHEAD'
                        candidate.update(ttc_s=round(t,3),range_m=round(float(np.hypot(*track.position)),2),closing_mps=round(closing,2),
                                         direction=direction,phrase='OBSTACLE '+direction,oncoming_candidate=bool(rel<-1.),ttc_basis='in_corridor_nearest_return',
                                         ttc_threshold_s=static_ttc_threshold(closing) if not lead else None)
                elif crossing_ok and speed>=MIN_FORCED_SPEED_MPS:
                    # moving compact object predicted to cross: keep threat()'s own estimate unless the driver is already braking
                    a_req=speed*speed/(2*max(float(candidate.get('ttc_s',0.))*speed-self.mount.bumper_x_m,.5))
                    if driver_braking and __import__('warning_arbitration').braking_adequate(a_req,-braking):
                        deferred_crossings.append({**candidate,'required_decel_mps2':round(a_req,2),'measured_decel_mps2':round(-braking,2),'reason':'Driver braking and stopping short needs only %.1f m/s2'%a_req,'label':'driver_braking_adequate_not_a_false_alarm_label'});candidate=None
                else:
                    reason=('Turning: not in the way under a straight path too' if occ.get('turn_consensus') is False else 'Edge-grazing returns only' if occ['edge_only'] else 'Radar sees it stationary: lidar motion rejected' if radar_contra else 'Crossing prediction requires sustained object motion' if not crossing_ok else 'Van below forced-mode speed floor')
                    nc=occ.get('nearest_cross_any_m');na=occ.get('nearest_along_any_m')
                    near_edge=bool(nc is not None and na is not None and not moving_obj and compact_box and self.mount.width_m/2+0.25<nc<=self.mount.width_m/2+0.25+NEAR_EDGE_BAND_M and speed>=MIN_FORCED_SPEED_MPS and (na-self.mount.bumper_x_m)/max(speed,.1)<=2.5)   # v13b: outside the padded corridor, compact cluster
                    deferred_crossings.append({**candidate,'reason':reason,'label':'unconfirmed_not_a_false_alarm_label','near_edge':near_edge,'near_edge_ttc_s':(round((na-self.mount.bumper_x_m)/max(speed,.1),2) if near_edge else None)})
                    candidate=None
            if candidate:
                candidates.append(candidate);track.danger_frames+=1
            else:track.danger_frames=0
        radar_alerts=[]
        if forced:
            from radar_fusion import shadow_candidates,radar_only_candidates
            # Count distinct publications, including an explicit empty update to reset gaps.
            stamp=int(context['at_ns'])
            if self._radar_last_update_ns is not None and not 0<=stamp-self._radar_last_update_ns<=250_000_000:
                self._radar_streaks.clear();self._radar_publication_times.clear()
            self._radar_last_update_ns=stamp
            ego=float(context.get('speed_mps') or 0.);fe_=context.get('forced_experiment') or {};brk=fe_.get('original_acceleration_mps2')
            brk=min(0.,float(brk)) if isinstance(brk,(int,float)) and np.isfinite(brk) else 0.
            drv_braking=bool(context.get('brake_pressed') is True or brk<=DRIVER_BRAKING_DECEL_MPS2)
            lidar_x=[float(c.get('range_m',0.)) for c in candidates]
            for rc in radar_only_candidates(radar_pts,ego,radar_matched,self._radar_streaks,
                    publication_ns=context.get('radar_publication_ns'),now_ns=stamp,
                    publication_times=self._radar_publication_times,publication_age_s=context.get('radar_publication_age_s'),
                    publication_clock_bound_s=context.get('radar_clock_bound_s',0.)):
                if any(abs(rc['x_m']-x)<6. for x in lidar_x):continue        # a lidar candidate already covers it
                a_req=rc['closing_mps']**2/(2*max(rc['x_m']-self.mount.bumper_x_m,.5))
                cand={'track_id':-int(rc['radar_id'])-1,'phrase':'OBSTACLE AHEAD','direction':'AHEAD','oncoming_candidate':False,
                      'ttc_s':rc['ttc_s'],'range_m':rc['x_m'],'closing_mps':rc['closing_mps'],'ground_velocity_mps':[rc['ground_speed_mps'],0.],
                      'position_m':[rc['x_m'],rc['y_m']],'residual_m':0.,'motion_evidence':{'trusted':True,'basis':'radar_range_rate','samples':rc['streak']},
                      'corridor':{'in_path':True,'basis':'radar_lateral'},'ttc_basis':'radar_only_same_direction','ttc_threshold_s':3.5,'radar':rc,'vision_lead':None,
                      'severity':'urgent' if rc['ttc_s']<=1.5 else 'warning'}
                if drv_braking and __import__('warning_arbitration').braking_adequate(a_req,-brk):
                    deferred_crossings.append({**cand,'reason':'Driver braking and stopping short needs only %.1f m/s2'%a_req,'label':'driver_braking_adequate_not_a_false_alarm_label'});continue
                if speed<MIN_FORCED_SPEED_MPS if (speed:=ego) is not None else False:continue
                elapsed=now-self.last_alert.get(cand['track_id'],-1e9)
                if __import__('lead_gate').blocks(self,cand,self.mount,now):deferred_crossings.append({**cand,'reason':'Radar-only lead: stopping short needs < %.1f m/s2 (v16 lead gate)'%LEAD_GATE_AREQ_MPS2,'label':'lead_gate_not_a_false_alarm_label'});continue
                if not reasons and rc['ttc_s']<=3.5 and elapsed>5 and __import__('warning_arbitration').global_slot(self,cand,now):radar_alerts.append(cand)
                else:deferred_crossings.append({**cand,'reason':'Radar-only same-direction target: gate/cooldown not met','label':'radar_only_gated_not_a_false_alarm_label'})
            for s_ in shadow_candidates(radar_pts,float(context.get('speed_mps') or 0.),radar_matched):
                deferred_crossings.append({**s_,'reason':'Radar-only moving target on a collision course (shadow, no warning)'})
        if quality.get('processing_overrun') or (processing_deadline_ns is not None and time.monotonic_ns()>processing_deadline_ns):
            reasons.append('inference exceeded 150 ms')
        __import__('lead_gate').update(self,self.tracker.tracks.values(),radar_pts,context.get('speed_mps'),now)
        if reasons:self._warning_surface=None
        for candidate in candidates:
            track=self.tracker.tracks[candidate['track_id']]
            prior_surface_track=(self.tracker.tracks.get(self._warning_surface['track_id']) if self._warning_surface else None)
            if forced and repeat_surface(self._warning_surface,candidate,track,now,prior_surface_track):
                deferred_crossings.append({**candidate,'reason':'Previously announced shallow surface reacquired under a new track ID',
                                           'label':'same_surface_repeat_not_a_semantic_label'})
                continue
            threshold=(corridor_envelope['warning_horizon_s'] if corridor_envelope is not None and corridor_envelope['valid']
                       else 4.5 if candidate['oncoming_candidate'] else 3.5)
            if candidate.get('ttc_threshold_s'):threshold=min(threshold,candidate['ttc_threshold_s'])
            urgent=candidate['ttc_s']<=1.5
            candidate['severity']='urgent' if urgent else 'warning'
            elapsed=now-self.last_alert.get(track.id,-1e9)
            escalating=urgent and not self.last_alert_urgent.get(track.id,False) and elapsed>1.5
            if __import__('lead_gate').blocks(self,candidate,self.mount,now):
                deferred_crossings.append({**candidate,'reason':'Moving lead: stopping short needs < %.1f m/s2 (v16 lead gate)'%LEAD_GATE_AREQ_MPS2,'label':'lead_gate_not_a_false_alarm_label'});continue
            if (not reasons and candidate['ttc_s']<=threshold and track.danger_frames>=2 and
                track.residual_m<2 and (elapsed>5 or escalating) and __import__('warning_arbitration').global_slot(self,candidate,now)):
                alerts.append(candidate)
        # Shared final authority covers radar as well as lidar, and includes time spent in all policy gates.
        if processing_deadline_ns is not None and time.monotonic_ns()>processing_deadline_ns and 'inference exceeded 150 ms' not in reasons:
            reasons.append('inference exceeded 150 ms')
        alerts=[] if reasons else alerts+radar_alerts
        if alerts:
            alerts=[min(alerts,key=lambda a:a['ttc_s'])]
            self._warning_surface=remember_surface(alerts[0],self.tracker.tracks.get(alerts[0]['track_id']),now)
            self.last_alert[alerts[0]['track_id']]=now;self.last_any_alert=now;self.last_warning={k:alerts[0][k] for k in ('track_id','direction')}
            self.last_alert_urgent[alerts[0]['track_id']]=alerts[0]['severity']=='urgent'
        # Bound cooldown memory even on long drives.
        self.last_alert={k:v for k,v in self.last_alert.items() if now-v<30}
        self.last_alert_urgent={k:v for k,v in self.last_alert_urgent.items() if k in self.last_alert}
        result={'status':'eligible_experimental' if not reasons else 'not_ready',
                'reasons':list(dict.fromkeys(reasons)),'ground':ground,'detections':len(detections),
                'tracks':[{'id':t.id,'position':t.position.round(3).tolist(),'velocity':t.velocity.round(3).tolist(),
                           'hits':t.hits,'misses':t.misses,'points':t.points,'residual_m':round(t.residual_m,3),
                           'bounds_m':{'lo':t.lo.round(3).tolist(),'hi':t.hi.round(3).tolist()},
                           'association_ambiguous':t.association_ambiguous} for t in tracks],
                'candidates':candidates,'alerts':alerts,'at_ns':context['at_ns']}
        if forced:
            result['deferred_crossings']=deferred_crossings
            result['forced_experiment']={**context['forced_experiment'],'bypassed_checks':list(dict.fromkeys(context['forced_experiment']['bypassed_checks']+bypass))}
            result['calibration_verified']=False
        if 'independent_corridor' in context:
            result['independent_corridor']=dict(context['independent_corridor']) if isinstance(context['independent_corridor'],dict) else {'active':False}
            if corridor_envelope is not None and not corridor_envelope['valid']:
                result['independent_corridor'].update(active=False,reasons=[corridor_envelope['reason']])
        if fusion_inputs is not None:
            self._augment(result,tracks,context,quality,fusion_inputs,processing_deadline_ns,corridor_envelope)
        elif hasattr(self,'fusion_danger'):self.fusion_danger.clear()
        return result

    def _augment(self,result,tracks,context,quality,inputs,deadline_ns,envelope):
        """Optional processing owns no baseline gates, persistence or cooldowns."""
        audit={'schema_version':1,'status':'inactive','baseline_preserved':True,'reasons':[]}
        map_audit={'status':'unknown','reason':'no_snapshot','warning_override':False,'stage':'predecision',
                   'ranked_tracks':[],'ranking_used':False}
        result['fusion']=audit;result['map_prior']=map_audit
        def stop(reason):
            audit['reasons'].append(reason);self.fusion_danger.clear()
        if not isinstance(inputs,dict):return stop('optional fusion inputs malformed')
        if result['alerts']:return stop('baseline alert has priority')
        if result['reasons'] or not context.get('ready'):return stop('baseline readiness unavailable')
        if deadline_ns is not None and deadline_ns-time.monotonic_ns()<25_000_000:
            return stop('less than 25 ms remains; optional work skipped')
        optional_deadline=min(time.monotonic_ns()+20_000_000,deadline_ns) if deadline_ns is not None else time.monotonic_ns()+20_000_000
        ranked=list(tracks)
        snapshot=inputs.get('map_snapshot')
        if snapshot is not None:
            try:
                incoming=snapshot.audit()
                map_audit.update(incoming)
                map_audit['warning_override']=False
                epoch=context.get('calibration_epoch',context.get('mount_epoch'))
                lineage=(incoming.get('mount_epoch')==epoch and incoming.get('calibration_id')==context.get('calibration_id') and
                         epoch is not None and context.get('calibration_id') is not None)
                if (incoming.get('status')=='available' and incoming.get('warning_override') is False and
                        incoming.get('at_ns')==context['at_ns'] and lineage and context.get('calibration_verified') is True):
                    scored=[]
                    for track in tracks[:256]:
                        if time.monotonic_ns()>=optional_deadline:break
                        inside=self._inside_or_unknown(track,context,envelope)
                        row={'id':track.id,'position':track.position.tolist(),
                             'bounds_m':{'lo':track.lo.tolist(),'hi':track.hi.tolist()}}
                        evidence=snapshot.query(row,in_corridor_or_unknown=inside)
                        delta=evidence.get('evidence_delta',0.) if isinstance(evidence,dict) else 0.
                        if type(delta) not in (int,float) or not math.isfinite(delta):delta=0.
                        delta=0. if inside else min(.1,max(-.1,delta))
                        scored.append((0 if inside else 1,-delta,float(np.linalg.norm(track.position)),track))
                        map_audit['ranked_tracks'].append({'track_id':track.id,'in_corridor_or_unknown':inside,
                                                         'evidence_delta':delta,'evidence':evidence})
                    if len(scored)==min(256,len(tracks)):
                        # Only outside-corridor priorities can change; every
                        # baseline object still reaches the baseline detector.
                        ranked=[row[-1] for row in sorted(scored,key=lambda x:x[:3])]+list(tracks[256:])
                        map_audit['ranking_used']=any(row[1] for row in scored)
                    map_audit['ranked_tracks']=map_audit['ranked_tracks'][:64]
                elif incoming.get('status')=='available':
                    map_audit.update(status='unknown',reason='map_lineage_or_calibration_mismatch')
            except Exception as error:
                map_audit.update(status='unknown',reason='map_query_failed:'+type(error).__name__)
        try:
            fused=self.fusion_engine.update(ranked,context,inputs.get('radar_snapshot'),inputs.get('camera_snapshot'),
                                            inputs.get('calibration'),phase=inputs.get('phase',3),deadline_ns=optional_deadline)
            audit.update(fused['audit']);audit['status']='diagnostic'
        except Exception as error:
            self.fusion_engine.reset()
            return stop('optional fusion failed:'+type(error).__name__)
        if audit.get('budget_exhausted') or time.monotonic_ns()>=optional_deadline:
            return stop('optional fusion budget exhausted')
        optional_candidates=[];optional_alerts=[];now=context['at_ns']/1e9;active=set()
        for track in fused['fused_track_candidates']:
            if time.monotonic_ns()>=optional_deadline:return stop('optional threat budget exhausted')
            candidate=threat(track,context,self.mount)
            if candidate is None:continue
            active.add(track.id);self.fusion_danger[track.id]=self.fusion_danger.get(track.id,0)+1
            candidate['fusion_evidence']=track.fusion;candidate['evidence_source']='optional_sensor_fusion'
            threshold=(envelope['warning_horizon_s'] if envelope is not None and envelope['valid'] else
                       4.5 if candidate['oncoming_candidate'] else 3.5)
            urgent=candidate['ttc_s']<=1.5;candidate['severity']='urgent' if urgent else 'warning'
            elapsed=now-self.fusion_last_alert.get(track.id,-1e9)
            escalating=urgent and not self.fusion_last_alert_urgent.get(track.id,False) and elapsed>1.5
            optional_candidates.append(candidate)
            if (candidate['ttc_s']<=threshold and self.fusion_danger[track.id]>=2 and track.residual_m<2 and
                    (elapsed>5 or escalating) and now-self.fusion_last_any_alert>1.5 and now-self.last_any_alert>1.5):
                optional_alerts.append(candidate)
        self.fusion_danger={k:v for k,v in self.fusion_danger.items() if k in active}
        if time.monotonic_ns()>=optional_deadline:return stop('optional threat budget exhausted')
        if optional_alerts:
            alert=min(optional_alerts,key=lambda a:a['ttc_s']);result['alerts']=[alert]
            self.fusion_last_alert[alert['track_id']]=now;self.fusion_last_any_alert=now
            self.fusion_last_alert_urgent[alert['track_id']]=alert['severity']=='urgent';audit['status']='additional_warning'
        elif optional_candidates:audit['status']='supported_tracks'
        result['fusion_candidates']=optional_candidates
        self.fusion_last_alert={k:v for k,v in self.fusion_last_alert.items() if now-v<30}
        self.fusion_last_alert_urgent={k:v for k,v in self.fusion_last_alert_urgent.items() if k in self.fusion_last_alert}

    def _inside_or_unknown(self,track,context,envelope):
        """Only confidently static roadside structure may be deprioritized."""
        try:
            if track.misses or track.hits<5 or track.association_ambiguous or np.linalg.norm(track.velocity)>.5:return True
            path=np.asarray(context.get('path'),float)
            if path.ndim!=2 or path.shape[1]!=2 or len(path)<2 or not np.isfinite(path).all():return True
            if track.position[0]<path[0,0] or track.position[0]>path[-1,0]:return True
            delta=path[1:]-path[:-1];length=np.sum(delta**2,axis=1)
            fraction=np.clip(np.sum((track.position-path[:-1])*delta,axis=1)/np.maximum(length,1e-9),0,1)
            distance=float(np.min(np.linalg.norm(track.position-(path[:-1]+fraction[:,None]*delta),axis=1)))
            radius=max(.5,float(np.linalg.norm(track.hi[:2]-track.lo[:2])/2))
            uncertainty=0. if envelope is None else float(np.max(envelope['lateral'])+np.max(envelope['longitudinal']))
            return distance<=self.mount.width_m/2+.25+radius+uncertainty
        except (KeyError,TypeError,ValueError,OverflowError):return True
